"""Inferência do modo de jogo a partir da gramática ativa: o rótulo da gramática
serve de ``mode`` sem caça na memória."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from cyhmo.state.grammar import normalize_entry

MODE_BATTLE = "battle"
MODE_NORMAL = "normal"
MODE_UNKNOWN = "unknown"

BATTLE_WITNESSES = frozenset({"reload", "dodge", "flee", "auto fire", "shoot the enemy"})
NORMAL_WITNESSES = frozenset({"walk", "run", "stop", "go back"})
DIALOGUE_WITNESSES = frozenset({"yes", "nope", "roger", "gotcha"})


@dataclass(frozen=True)
class ModeInference:
    mode: str
    witnesses: tuple[str, ...]

    @property
    def is_known(self) -> bool:
        return self.mode != MODE_UNKNOWN


def infer_mode(entries: Sequence[str]) -> ModeInference:
    present = {normalize_entry(entry) for entry in entries}
    battle = sorted(present & BATTLE_WITNESSES)
    if battle:
        return ModeInference(MODE_BATTLE, tuple(battle))
    normal = sorted(present & (NORMAL_WITNESSES | DIALOGUE_WITNESSES))
    if normal:
        return ModeInference(MODE_NORMAL, tuple(normal))
    return ModeInference(MODE_UNKNOWN, ())
