"""Detector de fala (silero-vad em ONNX, sem rede) e o Protocol que a ativação consome."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from cyhmo.domain.contracts import DEFAULT_SAMPLE_RATE

FRAME_SAMPLES = 512


@runtime_checkable
class SpeechDetector(Protocol):
    def probability(self, frame: np.ndarray) -> float: ...

    def reset(self) -> None: ...


class SileroVad:
    def __init__(self, threshold: float, sample_rate: int = DEFAULT_SAMPLE_RATE) -> None:
        if sample_rate != DEFAULT_SAMPLE_RATE:
            raise ValueError(f"o silero-vad deste pipeline opera a {DEFAULT_SAMPLE_RATE} Hz (recebi {sample_rate})")
        self.threshold = threshold
        self._sample_rate = sample_rate
        self._model, self._as_tensor = _load_onnx_model()

    def probability(self, frame: np.ndarray) -> float:
        frame = np.ascontiguousarray(frame, dtype=np.float32)
        if frame.shape != (FRAME_SAMPLES,):
            raise ValueError(f"o VAD espera quadros de {FRAME_SAMPLES} amostras (recebi {frame.shape})")
        return float(self._model(self._as_tensor(frame), self._sample_rate).item())

    def is_speech(self, frame: np.ndarray) -> bool:
        return self.probability(frame) >= self.threshold

    def reset(self) -> None:
        self._model.reset_states()


def _load_onnx_model() -> tuple[Any, Any]:
    import torch
    from silero_vad import load_silero_vad

    return load_silero_vad(onnx=True), torch.from_numpy
