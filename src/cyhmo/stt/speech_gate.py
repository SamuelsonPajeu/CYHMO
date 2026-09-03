"""Portão de presença de fala, calibrado sozinho pela fala do próprio jogador.

Um enunciado muito mais fraco que a fala habitual do usuário não é comando: é um toque
acidental do push-to-talk, uma respiração ou um ruído da sala. Mandá-lo ao decoder gasta
~700 ms e, pior, volta como palavra — na batalha de 2026-08-26 três ruídos viraram
``corre`` e teriam injetado ``run``.

O limiar é **relativo** à mediana dos enunciados já aceitos, nunca um nível absoluto:
ganho de microfone varia por usuário e um piso fixo emudeceria quem grava baixo.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from cyhmo.domain.contracts import AudioSegment

FRAME_MS = 20
LOUD_PERCENTILE = 90


def speech_level(audio: AudioSegment) -> float:
    """Energia dos quadros mais fortes: imune a silêncio antes e depois da fala."""
    samples = np.asarray(audio.samples, dtype=np.float32)
    size = max(1, audio.sample_rate * FRAME_MS // 1000)
    if samples.size < size:
        return float(np.sqrt(np.mean(samples**2))) if samples.size else 0.0
    frames = samples[: samples.size // size * size].reshape(-1, size)
    return float(np.percentile(np.sqrt((frames**2).mean(axis=1)), LOUD_PERCENTILE))


@dataclass
class SpeechGate:
    ratio: float = 0.125
    warmup: int = 5
    window: int = 40
    _levels: deque[float] = field(default_factory=lambda: deque(maxlen=40), init=False)

    def accepts(self, audio: AudioSegment) -> bool:
        if self.ratio <= 0.0:
            return True
        level = speech_level(audio)
        if len(self._levels) < self.warmup:
            self._remember(level)
            return True
        if level < float(np.median(self._levels)) * self.ratio:
            return False
        self._remember(level)
        return True

    def _remember(self, level: float) -> None:
        if level > 0.0:
            self._levels.append(level)
