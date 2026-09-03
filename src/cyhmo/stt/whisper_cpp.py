"""Transcrição via whisper.cpp, o backend padrão do mod.

O executável roda como servidor local e mantém o modelo carregado entre enunciados:
subir um processo por comando jogaria fora o carregamento, que é a parte cara.
"""

from __future__ import annotations

import io
import logging
import subprocess
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from cyhmo.domain.contracts import DEFAULT_SAMPLE_RATE, AudioSegment, Transcript
from cyhmo.domain.errors import CyhmoError
from cyhmo.stt.degeneration import is_degenerate
from cyhmo.stt.hotwords import SceneHotwords
from cyhmo.stt.normalization import normalize_text
from cyhmo.stt.resampling import resample_all

log = logging.getLogger("cyhmo.stt.whisper_cpp")

WARM_UP_MS = 1000
READY_TIMEOUT_S = 90.0
READY_POLL_S = 0.25
SHUTDOWN_TIMEOUT_S = 5.0


class WhisperCppError(CyhmoError):
    """O servidor whisper.cpp não subiu, morreu ou recusou a transcrição."""


@dataclass(frozen=True)
class ServerSpec:
    binary: Path
    model: Path
    host: str
    port: int
    language: str
    beam_size: int
    threads: int
    use_gpu: bool
    flash_attn: bool
    audio_ctx: int
    temperature_fallback: bool = False

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def temperature_inc(self) -> float:
        """O ``-nf`` do whisper-server é parseado e nunca aplicado; quem liga e desliga
        o fallback de temperatura de fato é este campo do request."""
        return 0.2 if self.temperature_fallback else 0.0

    def command(self) -> list[str]:
        argv = [
            str(self.binary),
            "-m", str(self.model),
            "--host", self.host,
            "--port", str(self.port),
            "-l", self.language,
            "-t", str(self.threads),
            "-bs", str(self.beam_size),
            "-nt",
        ]
        if self.audio_ctx > 0:
            argv += ["-ac", str(self.audio_ctx)]
        if not self.flash_attn:
            argv.append("-nfa")
        if not self.use_gpu:
            argv.append("-ng")
        return argv


class WhisperCppTranscriber:
    def __init__(
        self,
        spec: ServerSpec,
        hotwords: Sequence[str] = (),
        timeout_s: float = 15.0,
        auto_start: bool = True,
        scene: SceneHotwords | None = None,
    ) -> None:
        self._spec = spec
        self._prompt = ", ".join(hotwords)
        self._scene = scene
        self._timeout_s = timeout_s
        self._auto_start = auto_start
        self._process: subprocess.Popen[bytes] | None = None
        self._lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return self._spec.model.stem.replace("ggml-", "")

    @property
    def device(self) -> str:
        return "gpu" if self._spec.use_gpu else "cpu"

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def hotwords(self) -> str:
        return self._prompt

    def set_grammar(self, grammar: Sequence[str]) -> None:
        """A âncora acompanha a cena: sem isso o vocabulário da sala anterior fica
        puxando a fala da cena atual."""
        if self._scene is None:
            return
        self._prompt = ", ".join(self._scene.for_grammar(grammar))
        log.info("âncora do STT ajustada à cena: %d palavras", len(self._prompt.split(", ")) if self._prompt else 0)

    def ensure_ready(self) -> None:
        """Sobe o servidor se preciso e levanta ``WhisperCppError`` se não der. Barato com
        ele de pé — é só conferir o processo —, e é o gancho que deixa o mod trocar de
        backend antes de perder um enunciado."""
        self._ensure_server()

    def warm_up(self) -> None:
        self._ensure_server()
        started = time.perf_counter()
        self._post(AudioSegment.silence(WARM_UP_MS))
        log.info(
            "whisper.cpp %s (%s) aquecido em %.0f ms",
            self.model_name,
            self.device,
            (time.perf_counter() - started) * 1000.0,
        )

    def transcribe(self, audio: AudioSegment) -> Transcript:
        t_speech_end = audio.t_end if audio.t_end is not None else 0.0
        if audio.is_empty:
            log.warning("áudio vazio recebido para transcrição")
            return Transcript.empty(t_speech_end, self._spec.language)
        try:
            self._ensure_server()
            raw_text = self._post(audio)
        except Exception as exc:
            log.warning("transcrição falhou (%s: %s); devolvendo Transcript vazio", type(exc).__name__, exc)
            return Transcript.empty(t_speech_end, self._spec.language)
        text = normalize_text(raw_text)
        if is_degenerate(text):
            log.warning("saída degenerada do decoder descartada: %r", raw_text[:80])
            text = ""
        return Transcript(
            text=text,
            lang=self._spec.language,
            confidence=1.0 if text.strip() else 0.0,
            t_speech_end=t_speech_end,
            raw_text=raw_text,
        )

    def stop(self) -> None:
        with self._lock:
            process, self._process = self._process, None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=SHUTDOWN_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            process.kill()
        log.info("servidor whisper.cpp encerrado")

    def _ensure_server(self) -> None:
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                return
            if not self._auto_start:
                if not self._alive():
                    raise WhisperCppError(
                        f"nenhum servidor whisper.cpp respondendo em {self._spec.base_url}. "
                        "Suba-o à mão ou ligue stt.whisper_cpp.auto_start no config.toml."
                    )
                return
            self._process = self._spawn()

    def _spawn(self) -> subprocess.Popen[bytes]:
        if not self._spec.binary.is_file():
            raise WhisperCppError(
                f"executável do whisper.cpp não encontrado em {self._spec.binary}. "
                "Rode `CYHMO.cmd setup` para baixá-lo, ou corrija stt.whisper_cpp.binary no config.toml."
            )
        if not self._spec.model.is_file():
            raise WhisperCppError(
                f"modelo ggml não encontrado em {self._spec.model}. "
                "Rode `CYHMO.cmd setup` para baixá-lo, ou corrija stt.whisper_cpp.model no config.toml."
            )
        log.info("subindo whisper.cpp: %s", " ".join(self._spec.command()))
        process = subprocess.Popen(
            self._spec.command(),
            cwd=str(self._spec.binary.parent),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        self._wait_ready(process)
        return process

    def _wait_ready(self, process: subprocess.Popen[bytes]) -> None:
        deadline = time.perf_counter() + READY_TIMEOUT_S
        while time.perf_counter() < deadline:
            if process.poll() is not None:
                raise WhisperCppError(
                    f"o servidor whisper.cpp saiu com código {process.returncode} antes de responder. "
                    f"Rode à mão para ver o erro: {' '.join(self._spec.command())}"
                )
            if self._alive():
                return
            time.sleep(READY_POLL_S)
        process.kill()
        raise WhisperCppError(
            f"o servidor whisper.cpp não respondeu em {READY_TIMEOUT_S:.0f} s "
            f"({self._spec.base_url}). Modelo grande demais para a VRAM?"
        )

    def _alive(self) -> bool:
        import httpx

        try:
            httpx.get(self._spec.base_url, timeout=1.0)
        except Exception:
            return False
        return True

    def _post(self, audio: AudioSegment) -> str:
        import httpx

        response = httpx.post(
            f"{self._spec.base_url}/inference",
            files={"file": ("utterance.wav", encode_wav(audio), "audio/wav")},
            data={
                "temperature": "0.0",
                "temperature_inc": str(self._spec.temperature_inc),
                "response_format": "json",
                "prompt": self._prompt,
            },
            timeout=self._timeout_s,
        )
        response.raise_for_status()
        return _extract_text(response.json())


def encode_wav(audio: AudioSegment) -> bytes:
    samples = audio.samples
    if audio.sample_rate != DEFAULT_SAMPLE_RATE:
        samples = resample_all(samples, audio.sample_rate, DEFAULT_SAMPLE_RATE)
    pcm = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(DEFAULT_SAMPLE_RATE)
        handle.writeframes((pcm * 32767.0).astype("<i2").tobytes())
    return buffer.getvalue()


def _extract_text(payload: Any) -> str:
    if isinstance(payload, dict):
        if isinstance(text := payload.get("text"), str):
            return text.strip()
        segments = payload.get("segments")
        if isinstance(segments, list):
            return " ".join(str(item.get("text", "")).strip() for item in segments if isinstance(item, dict)).strip()
    return ""
