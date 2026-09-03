"""Serviço de captura: dispositivo -> captura -> ativação -> ``Utterance``, com telemetria no barramento."""

from __future__ import annotations

import logging
import math
import re
import threading
import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from cyhmo.config.schema import ActivationConfig, AppConfig, AudioConfig, ProjectPaths
from cyhmo.domain.contracts import Utterance
from cyhmo.domain.errors import AudioDeviceError, CyhmoError
from cyhmo.domain.events import AudioLevel, ComponentChanged, UtteranceCaptured
from cyhmo.domain.ports import Activation, AudioCapture, EventSink, UtteranceHandler
from cyhmo.stt.activation import IdFactory, build_activation
from cyhmo.stt.capture import SoundDeviceCapture
from cyhmo.stt.devices import CaptureDevice, list_capture_devices, open_candidates, resolve_device
from cyhmo.stt.wav import write_wav

log = logging.getLogger("cyhmo.stt.service")

CaptureFactory = Callable[[CaptureDevice, AudioConfig], AudioCapture]
ActivationFactory = Callable[[ActivationConfig, EventSink, IdFactory], Activation]
DevicesProvider = Callable[[], Sequence[CaptureDevice]]

LEVEL_PUBLISH_INTERVAL_S = 0.1
_UNSAFE_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*]')


class CaptureService:
    def __init__(
        self,
        config: AppConfig,
        bus: EventSink,
        on_utterance: UtteranceHandler,
        session_id: str,
        capture_factory: CaptureFactory | None = None,
        activation_factory: ActivationFactory | None = None,
        devices_provider: DevicesProvider | None = None,
        raw_devices_provider: DevicesProvider | None = None,
        paths: ProjectPaths | None = None,
    ) -> None:
        self._config = config
        self._bus = bus
        self._on_utterance = on_utterance
        self._session_id = session_id
        self._capture_factory = capture_factory or _default_capture_factory
        self._activation_factory = activation_factory or build_activation
        self._devices_provider = devices_provider or list_capture_devices
        self._raw_devices_provider = raw_devices_provider or (lambda: list_capture_devices(dedupe=False))
        self._paths = paths or config.paths(Path.cwd())
        self._enabled = True
        self._counter = 0
        self._counter_lock = threading.Lock()
        self._capture: AudioCapture | None = None
        self._activation: Activation | None = None
        self._device: CaptureDevice | None = None
        self._meter = _LevelMeter(LEVEL_PUBLISH_INTERVAL_S)
        self._tap: _SignalStats | None = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def running(self) -> bool:
        return self._capture is not None

    @property
    def device(self) -> CaptureDevice | None:
        return self._device

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
        log.info("escuta %s", "ativada" if enabled else "desativada")

    def start(self) -> None:
        if self._capture is not None:
            return
        device = self._resolve_device()
        activation = self._start_activation()
        device, capture = self._open_capture(device, activation)
        self._device, self._activation, self._capture = device, activation, capture

    def stop(self) -> None:
        activation, self._activation = self._activation, None
        capture, self._capture = self._capture, None
        for component, name in ((activation, "activation"), (capture, "capture")):
            if component is None:
                continue
            try:
                component.stop()
            except Exception:
                log.exception("erro ao parar %s", name)
            self._bus.publish(ComponentChanged(component=name, status="off"))

    def mic_test(self, seconds: float) -> dict[str, float | str]:
        capture = self._capture
        temporary: AudioCapture | None = None
        if capture is None:
            temporary = self._capture_factory(self._resolve_device(), self._config.audio)
            temporary.start(self._on_block)
            capture = temporary
        stats = _SignalStats()
        self._tap = stats
        try:
            time.sleep(max(0.0, seconds))
        finally:
            self._tap = None
            if temporary is not None:
                temporary.stop()
        return {"peak": stats.peak, "rms": stats.rms, "device": capture.device_name}

    def _next_id(self) -> str:
        with self._counter_lock:
            self._counter += 1
            return f"{self._session_id}#{self._counter:05d}"

    def _resolve_device(self) -> CaptureDevice:
        try:
            return resolve_device(self._config.audio.device, self._devices_provider())
        except CyhmoError as exc:
            self._bus.publish(ComponentChanged(component="capture", status="error", detail=str(exc)))
            raise
        except Exception as exc:
            message = f"falha ao enumerar dispositivos de captura: {exc}"
            self._bus.publish(ComponentChanged(component="capture", status="error", detail=message))
            raise AudioDeviceError(message) from exc

    def _start_activation(self) -> Activation:
        activation = self._activation_factory(self._config.activation, self._bus, self._next_id)
        try:
            activation.start(self._handle_utterance)
        except CyhmoError as exc:
            self._bus.publish(ComponentChanged(component="activation", status="error", detail=str(exc)))
            raise
        detail = _activation_detail(self._config.activation)
        self._bus.publish(ComponentChanged(component="activation", status="ready", detail=detail))
        return activation

    def _open_capture(self, device: CaptureDevice, activation: Activation) -> tuple[CaptureDevice, AudioCapture]:
        """O endpoint padrão do Windows pode apontar para uma host API que não abre (WDM-KS);
        o mesmo microfone em outra host API salva a sessão em vez de derrubar o mod."""
        failures: list[str] = []
        for candidate in open_candidates(device, self._raw_devices_provider()):
            capture = self._capture_factory(candidate, self._config.audio)
            try:
                capture.start(self._on_block)
            except CyhmoError as exc:
                failures.append(str(exc))
                continue
            if candidate.index != device.index:
                log.warning("%s não abriu; usando %s", device.label, candidate.label)
            self._bus.publish(ComponentChanged(component="capture", status="ready", detail=candidate.label))
            return candidate, capture
        detail = "; ".join(failures)
        self._bus.publish(ComponentChanged(component="capture", status="error", detail=detail))
        activation.stop()
        raise AudioDeviceError(detail)

    def _on_block(self, block: np.ndarray, t_block_end: float) -> None:
        try:
            level = self._meter.update(block, t_block_end)
            if level is not None:
                self._bus.publish(level)
            tap = self._tap
            if tap is not None:
                tap.update(block)
            if self._enabled and self._activation is not None:
                self._activation.feed(block, t_block_end)
        except Exception:
            log.exception("erro ao processar bloco de áudio")

    def _handle_utterance(self, utterance: Utterance) -> None:
        try:
            if not self._enabled:
                log.debug("%s: escuta desativada; enunciado ignorado", utterance.utt_id)
                return
            wav_path = self._save_audio(utterance) if self._config.debug.save_audio else None
            self._bus.publish(
                UtteranceCaptured(
                    utt_id=utterance.utt_id,
                    duration_ms=utterance.audio.duration_ms,
                    source=utterance.source,
                    wav_path=None if wav_path is None else str(wav_path),
                )
            )
            self._on_utterance(utterance)
        except Exception:
            log.exception("falha ao tratar o enunciado %s", utterance.utt_id)

    def _save_audio(self, utterance: Utterance) -> Path | None:
        target = self._paths.audio_dir / f"{_safe_filename(utterance.utt_id)}.wav"
        try:
            write_wav(target, utterance.audio)
        except Exception as exc:
            log.warning("não foi possível salvar %s: %s", target, exc)
            return None
        log.info("%s: áudio salvo em %s (%.0f ms)", utterance.utt_id, target, utterance.audio.duration_ms)
        return target


class _SignalStats:
    def __init__(self) -> None:
        self._sum_squares = 0.0
        self._count = 0
        self.peak = 0.0

    def update(self, block: np.ndarray) -> None:
        if block.size == 0:
            return
        self._sum_squares += float(np.dot(block, block))
        self._count += int(block.size)
        self.peak = max(self.peak, float(np.max(np.abs(block))))

    @property
    def rms(self) -> float:
        return math.sqrt(self._sum_squares / self._count) if self._count else 0.0


class _LevelMeter:
    def __init__(self, interval_s: float) -> None:
        self._interval_s = interval_s
        self._stats = _SignalStats()
        self._last_publish: float | None = None

    def update(self, block: np.ndarray, now: float) -> AudioLevel | None:
        self._stats.update(block)
        if self._last_publish is None:
            self._last_publish = now
        if now - self._last_publish < self._interval_s:
            return None
        level = AudioLevel(rms=self._stats.rms, peak=self._stats.peak)
        self._stats = _SignalStats()
        self._last_publish = now
        return level


def _default_capture_factory(device: CaptureDevice, audio: AudioConfig) -> AudioCapture:
    return SoundDeviceCapture(device, target_rate=audio.sample_rate, block_ms=audio.block_ms)


def _activation_detail(config: ActivationConfig) -> str:
    if config.mode == "ptt":
        return f"ptt: {config.ptt_hotkey}"
    return f"vad: limiar {config.vad.threshold:.2f}, tail {config.vad.tail_ms} ms"


def _safe_filename(utt_id: str) -> str:
    return _UNSAFE_FILENAME_CHARS.sub("_", utt_id)
