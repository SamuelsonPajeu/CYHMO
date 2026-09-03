"""Transcrição local com faster-whisper e o transcritor roteirizado de ``stt.engine = "fake"``."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from cyhmo.config.schema import SttConfig
from cyhmo.domain.language_pack import LanguagePack
from cyhmo.domain.contracts import DEFAULT_SAMPLE_RATE, AudioSegment, Transcript
from cyhmo.domain.errors import CyhmoError
from cyhmo.domain.ports import Transcriber
from cyhmo.stt.confidence import confidence_from_segments
from cyhmo.stt.degeneration import is_degenerate
from cyhmo.stt.hotwords import SceneHotwords, pack_hotwords
from cyhmo.stt.normalization import normalize_text
from cyhmo.stt.resampling import resample_all
from cyhmo.stt.speech_gate import SpeechGate

log = logging.getLogger("cyhmo.stt.transcriber")

AUTO = "auto"
PACK = "pack"
WARM_UP_MS = 1000
STT_MODELS_SUBDIR = "stt"
FALLBACK_FAKE_LANGUAGE = "en"


class SttModelError(CyhmoError):
    """Modelo de STT não pôde ser carregado (arquivos ausentes, dispositivo indisponível)."""


@dataclass(frozen=True)
class WhisperLoadSpec:
    model: str
    device: str
    compute_type: str
    download_root: Path
    cpu_threads: int = 0


ModelFactory = Callable[[WhisperLoadSpec], Any]
Script = str | dict[str, str] | Callable[[AudioSegment], str]


def resolve_device_kind(device: str) -> str:
    if device != AUTO:
        return device
    return "cuda" if _cuda_available() else "cpu"


def resolve_compute_type(compute_type: str, device: str) -> str:
    if compute_type != AUTO:
        return compute_type
    return "float16" if device == "cuda" else "int8"


class FasterWhisperTranscriber:
    def __init__(
        self,
        model: str,
        compute_type: str,
        device: str,
        language: str,
        beam_size: int,
        models_dir: Path,
        model_factory: ModelFactory | None = None,
        cpu_threads: int = 0,
        temperature_fallback: bool = False,
        hotwords: Sequence[str] = (),
        scene: SceneHotwords | None = None,
    ) -> None:
        device_kind = resolve_device_kind(device)
        self._spec = WhisperLoadSpec(
            model=model,
            device=device_kind,
            compute_type=resolve_compute_type(compute_type, device_kind),
            download_root=Path(models_dir) / STT_MODELS_SUBDIR,
            cpu_threads=cpu_threads,
        )
        self._language = language.strip().lower()
        self._beam_size = beam_size
        self._temperature_fallback = temperature_fallback
        self._hotwords = ", ".join(hotwords)
        self._scene = scene
        self._model_factory = model_factory or _load_whisper_model
        self._model: Any | None = None
        self._lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return self._spec.model

    @property
    def device(self) -> str:
        return self._spec.device

    @property
    def compute_type(self) -> str:
        return self._spec.compute_type

    @property
    def auto_language(self) -> bool:
        return self._language == AUTO

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def hotwords(self) -> str:
        return self._hotwords

    def set_grammar(self, grammar: Sequence[str]) -> None:
        """A âncora acompanha a cena: sem isso o vocabulário da sala anterior fica
        puxando a fala da cena atual."""
        if self._scene is None:
            return
        self._hotwords = ", ".join(self._scene.for_grammar(grammar))

    def warm_up(self) -> None:
        model = self._ensure_model()
        started = time.perf_counter()
        self._run(model, AudioSegment.silence(WARM_UP_MS))
        log.info(
            "modelo %s (%s/%s) aquecido em %.0f ms",
            self._spec.model,
            self._spec.device,
            self._spec.compute_type,
            (time.perf_counter() - started) * 1000.0,
        )

    def transcribe(self, audio: AudioSegment) -> Transcript:
        t_speech_end = _speech_end(audio)
        if audio.is_empty:
            log.warning("áudio vazio recebido para transcrição")
            return Transcript.empty(t_speech_end, self._fixed_language)
        try:
            return self._run(self._ensure_model(), audio)
        except Exception as exc:
            log.warning("transcrição falhou (%s: %s); devolvendo Transcript vazio", type(exc).__name__, exc)
            return Transcript.empty(t_speech_end, self._fixed_language)

    @property
    def _fixed_language(self) -> str:
        return "" if self.auto_language else self._language

    @property
    def _decode_options(self) -> dict[str, Any]:
        """Comando isolado dá log-prob baixo, e o fallback de temperatura do Whisper responde
        redecodificando até 6 vezes — é o que transforma 2 s de latência em 6 s sem melhorar nada."""
        options: dict[str, Any] = {}
        if not self._temperature_fallback:
            options.update(temperature=[0.0], compression_ratio_threshold=None, log_prob_threshold=None)
        if self._hotwords:
            options["hotwords"] = self._hotwords
        return options

    def _ensure_model(self) -> Any:
        with self._lock:
            if self._model is None:
                self._model = self._load()
            return self._model

    def _load(self) -> Any:
        started = time.perf_counter()
        try:
            model = self._model_factory(self._spec)
        except Exception as exc:
            raise SttModelError(
                f"não foi possível carregar o modelo de STT {self._spec.model!r} "
                f"({self._spec.device}/{self._spec.compute_type}) em {self._spec.download_root}: {exc}. "
                "Confira stt.model, stt.device e stt.compute_type no config.toml e se o modelo está em models/stt."
            ) from exc
        log.info("modelo %s carregado em %.0f ms", self._spec.model, (time.perf_counter() - started) * 1000.0)
        return model

    def _run(self, model: Any, audio: AudioSegment) -> Transcript:
        samples = audio.samples
        if audio.sample_rate != DEFAULT_SAMPLE_RATE:
            samples = resample_all(samples, audio.sample_rate, DEFAULT_SAMPLE_RATE)
        segments, info = model.transcribe(
            samples,
            language=None if self.auto_language else self._language,
            beam_size=self._beam_size,
            vad_filter=False,
            condition_on_previous_text=False,
            without_timestamps=True,
            **self._decode_options,
        )
        segments = list(segments)
        raw_text = " ".join(str(segment.text).strip() for segment in segments if getattr(segment, "text", "")).strip()
        lang = str(getattr(info, "language", "") or "") if self.auto_language else self._language
        text = normalize_text(raw_text)
        if is_degenerate(text):
            log.warning("saída degenerada do decoder descartada: %r", raw_text[:80])
            text = ""
        return Transcript(
            text=text,
            lang=lang,
            confidence=confidence_from_segments(segments) if text else 0.0,
            t_speech_end=_speech_end(audio),
            raw_text=raw_text,
        )


class FakeTranscriber:
    def __init__(
        self,
        script: Script = "",
        lang: str = FALLBACK_FAKE_LANGUAGE,
        confidence: float = 0.9,
        latency_s: float = 0.0,
        fail: bool = False,
        model_name: str = "fake",
    ) -> None:
        self._script = script
        self._lang = lang
        self._confidence = confidence
        self._latency_s = latency_s
        self._fail = fail
        self._model_name = model_name
        self.call_count = 0
        self.warmed_up = False

    @property
    def model_name(self) -> str:
        return self._model_name

    def warm_up(self) -> None:
        self.warmed_up = True

    def transcribe(self, audio: AudioSegment) -> Transcript:
        self.call_count += 1
        if self._latency_s > 0.0:
            time.sleep(self._latency_s)
        t_speech_end = _speech_end(audio)
        if self._fail or audio.is_empty:
            return Transcript.empty(t_speech_end, self._lang)
        raw_text = self._resolve(audio)
        return Transcript(
            text=normalize_text(raw_text),
            lang=self._lang,
            confidence=self._confidence if raw_text else 0.0,
            t_speech_end=t_speech_end,
            raw_text=raw_text,
        )

    def _resolve(self, audio: AudioSegment) -> str:
        if callable(self._script):
            return self._script(audio)
        if isinstance(self._script, dict):
            name = getattr(audio, "name", None)
            return self._script.get(audio.sha256()) or self._script.get(str(name), "")
        return self._script


class FallbackTranscriber:
    """Mantém o mod ouvindo quando o backend padrão não sobe.

    O whisper.cpp é o padrão por ser ~3x mais rápido, mas depende de um executável e de um
    peso baixados fora do pip. Faltando qualquer um dos dois — ou com a porta ocupada, ou
    com o servidor morrendo no boot — o mod troca para o faster-whisper, que se resolve
    sozinho pelo pip, em vez de ficar sem reconhecimento nenhum."""

    def __init__(self, primary: Transcriber, build_backup: Callable[[], Transcriber]) -> None:
        self._primary = primary
        self._build_backup = build_backup
        self._backup: Transcriber | None = None
        self._grammar: tuple[str, ...] = ()

    @property
    def active(self) -> Transcriber:
        return self._backup if self._backup is not None else self._primary

    @property
    def degraded(self) -> bool:
        return self._backup is not None

    def set_grammar(self, grammar: Sequence[str]) -> None:
        self._grammar = tuple(grammar)
        _apply_grammar(self.active, self._grammar)

    def warm_up(self) -> None:
        if self._ready():
            self.active.warm_up()

    def transcribe(self, audio: AudioSegment) -> Transcript:
        self._ready()
        return self.active.transcribe(audio)

    def stop(self) -> None:
        stop = getattr(self._primary, "stop", None)
        if callable(stop):
            stop()

    def _ready(self) -> bool:
        """``ensure_ready`` é barato com o servidor de pé: só confere o processo."""
        if self._backup is not None:
            return True
        ensure = getattr(self._primary, "ensure_ready", None)
        if not callable(ensure):
            return True
        try:
            ensure()
        except CyhmoError as exc:
            return self._switch(exc)
        return True

    def _switch(self, reason: Exception) -> bool:
        log.warning("backend de STT indisponível (%s); usando o faster-whisper", reason)
        try:
            self._backup = self._build_backup()
        except Exception as exc:
            log.error("o faster-whisper também não subiu (%s); o mod fica sem transcrição", exc)
            return False
        _apply_grammar(self._backup, self._grammar)
        return True

    def __getattr__(self, name: str) -> Any:
        return getattr(self.active, name)


def _apply_grammar(transcriber: Transcriber, grammar: Sequence[str]) -> None:
    setter = getattr(transcriber, "set_grammar", None)
    if callable(setter) and grammar:
        setter(grammar)


class GatedTranscriber:
    """Envolve o transcritor com o portão de fala: enunciado fraco demais nem chega ao
    decoder, o que evita a injeção falsa e ainda devolve os ~700 ms do STT."""

    def __init__(self, inner: Transcriber, gate: SpeechGate) -> None:
        self._inner = inner
        self._gate = gate

    @property
    def inner(self) -> Transcriber:
        return self._inner

    def transcribe(self, audio: AudioSegment) -> Transcript:
        if not self._gate.accepts(audio):
            log.info("enunciado abaixo do nível de fala do jogador; descartado sem transcrever")
            return Transcript.empty(_speech_end(audio), getattr(self._inner, "language", ""))
        return self._inner.transcribe(audio)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def resolve_stt_language(config: SttConfig, primary_pack_language: str) -> str:
    if config.language == PACK:
        return primary_pack_language.strip().lower()
    return config.language




def build_transcriber(
    config: SttConfig,
    stt_language: str,
    models_dir: Path,
    pack: LanguagePack | None = None,
    base_dir: Path | None = None,
) -> Transcriber:
    limit = config.max_hotwords if config.hotwords else 0
    hotwords = pack_hotwords(pack, limit)
    scene = SceneHotwords(pack, limit, hotwords) if config.hotwords else None
    if config.engine == "fake":
        return FakeTranscriber(lang=FALLBACK_FAKE_LANGUAGE if stt_language == AUTO else stt_language)

    def faster_whisper() -> Transcriber:
        return FasterWhisperTranscriber(
            model=config.model,
            compute_type=config.compute_type,
            device=config.device,
            language=stt_language,
            beam_size=config.beam_size,
            models_dir=Path(models_dir),
            cpu_threads=config.cpu_threads,
            temperature_fallback=config.temperature_fallback,
            hotwords=hotwords,
            scene=scene,
        )

    if config.engine == "whisper-cpp":
        primary = _build_whisper_cpp(config, stt_language, base_dir or Path.cwd(), hotwords, scene)
        inner: Transcriber = FallbackTranscriber(primary, faster_whisper)
    else:
        inner = faster_whisper()
    if config.silence_gate_ratio <= 0.0:
        return inner
    return GatedTranscriber(inner, SpeechGate(ratio=config.silence_gate_ratio))


def _build_whisper_cpp(
    config: SttConfig,
    stt_language: str,
    base_dir: Path,
    hotwords: Sequence[str],
    scene: SceneHotwords | None,
) -> Transcriber:
    from cyhmo.stt.whisper_cpp import ServerSpec, WhisperCppTranscriber

    settings = config.whisper_cpp
    spec = ServerSpec(
        binary=(base_dir / settings.binary).resolve(),
        model=(base_dir / settings.model).resolve(),
        host=settings.host,
        port=settings.port,
        language=stt_language,
        beam_size=config.beam_size,
        threads=settings.threads,
        use_gpu=settings.use_gpu,
        flash_attn=settings.flash_attn,
        audio_ctx=settings.audio_ctx,
        temperature_fallback=config.temperature_fallback,
    )
    return WhisperCppTranscriber(
        spec,
        hotwords=hotwords,
        timeout_s=settings.timeout_ms / 1000.0,
        auto_start=settings.auto_start,
        scene=scene,
    )


def _speech_end(audio: AudioSegment) -> float:
    return audio.t_end if audio.t_end is not None else 0.0


def _cuda_available() -> bool:
    try:
        import ctranslate2

        return int(ctranslate2.get_cuda_device_count()) > 0
    except Exception:
        return False


def _load_whisper_model(spec: WhisperLoadSpec) -> Any:
    from faster_whisper import WhisperModel

    return WhisperModel(
        spec.model,
        device=spec.device,
        compute_type=spec.compute_type,
        download_root=str(spec.download_root),
        local_files_only=False,
        cpu_threads=spec.cpu_threads,
    )
