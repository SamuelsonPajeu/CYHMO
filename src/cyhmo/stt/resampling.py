"""Reamostragem em streaming (soxr) e redução a mono da cadeia de captura."""

from __future__ import annotations

from typing import Any

import numpy as np

DEFAULT_QUALITY = "HQ"


class Resampler:
    def __init__(
        self,
        src_rate: int,
        dst_rate: int,
        channels: int = 1,
        quality: str = DEFAULT_QUALITY,
    ) -> None:
        if src_rate <= 0 or dst_rate <= 0:
            raise ValueError(f"taxas de amostragem devem ser positivas (recebi {src_rate} -> {dst_rate})")
        self.src_rate = src_rate
        self.dst_rate = dst_rate
        self._stream: Any | None = None
        if src_rate != dst_rate:
            self._stream = _open_stream(src_rate, dst_rate, channels, quality)

    @property
    def passthrough(self) -> bool:
        return self._stream is None

    def process(self, block: np.ndarray) -> np.ndarray:
        block = np.ascontiguousarray(block, dtype=np.float32)
        if self._stream is None:
            return block
        return np.asarray(self._stream.resample_chunk(block), dtype=np.float32)

    def flush(self) -> np.ndarray:
        if self._stream is None:
            return np.zeros(0, dtype=np.float32)
        tail = self._stream.resample_chunk(np.zeros(0, dtype=np.float32), last=True)
        return np.asarray(tail, dtype=np.float32)


def resample_all(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    resampler = Resampler(src_rate, dst_rate)
    return np.concatenate([resampler.process(samples), resampler.flush()])


def to_mono(block: np.ndarray) -> np.ndarray:
    block = np.asarray(block, dtype=np.float32)
    if block.ndim == 1:
        return block
    if block.shape[1] == 1:
        return np.ascontiguousarray(block[:, 0])
    return block.mean(axis=1, dtype=np.float32)


def _open_stream(src_rate: int, dst_rate: int, channels: int, quality: str) -> Any:
    import soxr

    return soxr.ResampleStream(src_rate, dst_rate, channels, dtype="float32", quality=quality)
