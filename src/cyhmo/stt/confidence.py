"""Confiança heurística do STT em [0, 1] a partir dos escores por segmento.

Mapeamento: média, ponderada pela duração de cada segmento, de
``exp(avg_logprob) * (1 - no_speech_prob)``, com clamp em [0, 1]; sem segmentos
o resultado é 0.0. Não é probabilidade calibrada — a camada 2 a trata como
sinal ordinal.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

MIN_SEGMENT_WEIGHT_S = 0.01


def confidence_from_segments(segments: Iterable[Any]) -> float:
    weighted_sum = 0.0
    total_weight = 0.0
    for segment in segments:
        weight = _segment_weight(segment)
        weighted_sum += _segment_score(segment) * weight
        total_weight += weight
    if total_weight == 0.0:
        return 0.0
    return _clamp(weighted_sum / total_weight)


def _segment_score(segment: Any) -> float:
    avg_logprob = _attribute(segment, "avg_logprob")
    no_speech_prob = _clamp(_attribute(segment, "no_speech_prob"))
    return math.exp(min(avg_logprob, 0.0)) * (1.0 - no_speech_prob)


def _segment_weight(segment: Any) -> float:
    duration = _attribute(segment, "end") - _attribute(segment, "start")
    return max(duration, MIN_SEGMENT_WEIGHT_S)


def _attribute(segment: Any, name: str) -> float:
    value = getattr(segment, name, None)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 0.0
    return float(value)


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, value))
