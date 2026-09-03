"""Captura do microfone via sounddevice em WASAPI compartilhado."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

import numpy as np

from cyhmo.domain.contracts import DEFAULT_SAMPLE_RATE
from cyhmo.domain.errors import AudioDeviceError
from cyhmo.domain.ports import AudioBlockHandler
from cyhmo.stt.devices import CaptureDevice
from cyhmo.stt.resampling import Resampler, to_mono

log = logging.getLogger("cyhmo.stt.capture")

WASAPI_HOST_API = "Windows WASAPI"
DEFAULT_BLOCK_MS = 20


class SoundDeviceCapture:
    def __init__(
        self,
        device: CaptureDevice,
        target_rate: int = DEFAULT_SAMPLE_RATE,
        block_ms: int = DEFAULT_BLOCK_MS,
    ) -> None:
        self._device = device
        self._target_rate = target_rate
        self._block_ms = block_ms
        self._stream: Any | None = None

    @property
    def device_name(self) -> str:
        return self._device.name

    @property
    def sample_rate(self) -> int:
        return self._target_rate

    @property
    def running(self) -> bool:
        return self._stream is not None

    def start(self, on_block: AudioBlockHandler) -> None:
        if self._stream is not None:
            return
        import sounddevice as sd

        capture_rate = self._device.sample_rate
        callback = _make_callback(on_block, Resampler(capture_rate, self._target_rate))
        stream = self._open_stream(sd, capture_rate, max(1, capture_rate * self._block_ms // 1000), callback)
        try:
            stream.start()
        except Exception as exc:
            stream.close()
            raise AudioDeviceError(f"não foi possível iniciar a captura em {self._device.label}: {exc}") from exc
        self._stream = stream
        log.info(
            "captura iniciada: %s a %d Hz -> %d Hz, blocos de %d ms",
            self._device.label,
            capture_rate,
            self._target_rate,
            self._block_ms,
        )

    def stop(self) -> None:
        stream, self._stream = self._stream, None
        if stream is None:
            return
        try:
            stream.stop()
            stream.close()
        except Exception:
            log.exception("erro ao encerrar a captura de %s", self._device.label)

    def _open_stream(self, sd: Any, capture_rate: int, blocksize: int, callback: Callable[..., None]) -> Any:
        extra_settings = sd.WasapiSettings(exclusive=False) if self._device.host_api == WASAPI_HOST_API else None
        failures: list[str] = []
        for channels in _channel_candidates(self._device.channels):
            try:
                return sd.InputStream(
                    device=self._device.index,
                    samplerate=capture_rate,
                    blocksize=blocksize,
                    channels=channels,
                    dtype="float32",
                    extra_settings=extra_settings,
                    callback=callback,
                )
            except Exception as exc:
                failures.append(f"{channels} canal(is): {exc}")
        raise AudioDeviceError(
            f"não foi possível abrir {self._device.label} ({'; '.join(failures)}). "
            "Verifique se o microfone está conectado e se nenhum programa o usa em modo exclusivo."
        )


def _make_callback(on_block: AudioBlockHandler, resampler: Resampler) -> Callable[..., None]:
    def callback(indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        try:
            if status:
                log.debug("status da captura: %s", status)
            block = resampler.process(to_mono(indata))
            if block.size:
                on_block(block, time.perf_counter())
        except Exception:
            log.exception("erro no callback de captura")

    return callback


def _channel_candidates(max_channels: int) -> tuple[int, ...]:
    candidates = (1, min(max_channels, 2), max_channels)
    return tuple(dict.fromkeys(channels for channels in candidates if channels >= 1))
