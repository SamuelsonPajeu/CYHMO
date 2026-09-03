"""Orçamento de latência por etapa e os percentis que o comparam com ele."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

BUDGET_MS: dict[str, float] = {
    "vad_tail": 250,
    "stt": 400,
    "interpret": 150,
    "state_inject": 50,
    "total": 1000,
}


def percentiles(values: Iterable[float]) -> tuple[float, float, float]:
    """Percentis por posto mais próximo (nearest rank): cada valor devolvido é uma amostra real."""
    ordered = sorted(values)
    if not ordered:
        return (0.0, 0.0, 0.0)
    return (_nearest_rank(ordered, 50), _nearest_rank(ordered, 95), ordered[-1])


def _nearest_rank(ordered: Sequence[float], percentile: float) -> float:
    rank = max(1, math.ceil(percentile / 100.0 * len(ordered)))
    return ordered[rank - 1]
