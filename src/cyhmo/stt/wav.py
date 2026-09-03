"""Escrita de WAV PCM16 mono com a stdlib."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from cyhmo.domain.contracts import AudioSegment

_PCM16_MAX = 32767.0


def write_wav(path: Path, audio: AudioSegment) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(audio.samples, -1.0, 1.0) * _PCM16_MAX).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(audio.sample_rate)
        handle.writeframes(pcm.tobytes())
    return path
