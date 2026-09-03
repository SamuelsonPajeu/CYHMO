"""Gravação do áudio de cada enunciado em .wav PCM16 mono via stdlib."""

from __future__ import annotations

import re
import wave
from pathlib import Path

import numpy as np

from cyhmo.domain.contracts import AudioSegment, Utterance

PCM16_SCALE = 32767.0
PCM16_BYTES = 2
UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")


def safe_file_stem(utt_id: str) -> str:
    return UNSAFE_CHARS.sub("_", utt_id)


def write_wav(path: Path, audio: AudioSegment) -> None:
    pcm = (np.clip(audio.samples, -1.0, 1.0) * PCM16_SCALE).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(PCM16_BYTES)
        handle.setframerate(audio.sample_rate)
        handle.writeframes(pcm.tobytes())


class UtteranceRecorder:
    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)

    def save(self, utterance: Utterance) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{safe_file_stem(utterance.utt_id)}.wav"
        write_wav(path, utterance.audio)
        return path
