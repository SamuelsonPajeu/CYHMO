"""Detecção de saída degenerada do decoder — o loop de repetição do Whisper.

O servidor roda com ``-nt``: sem token de timestamp o decoder não tem como encerrar
o segmento, e uma âncora repetitiva pode fazê-lo repetir até o teto de tokens. O loop
aparece nas duas formas, ambas observadas em áudio real: palavra solta
("ouça ouça ouça…") e frase inteira ("Atira no olho direito" três vezes seguidas).
Texto degenerado nunca é comando: barrar aqui evita que a camada 2 gaste orçamento.
"""

from __future__ import annotations

from collections import Counter

from cyhmo.domain.writing import is_continuous, is_mostly_continuous

MIN_WORDS = 5
MIN_CONTINUOUS_UNITS = 12
DOMINANCE = 0.5
MAX_CYCLE_UNITS = 6
MIN_CYCLE_REPEATS = 3


def is_degenerate(text: str) -> bool:
    units, minimum = _units(text)
    if len(units) < minimum:
        return False
    return _unit_dominates(units) or _cycle_repeats(units)


def _units(text: str) -> tuple[list[str], int]:
    """Escrita contínua não tem palavra a contar, então a unidade vira o caractere — e o piso
    sobe junto: um comando repetido de propósito ("Shoot shoot shoot", entrada legítima da
    gramática) cabe em seis caracteres, enquanto o loop do decoder rende dezenas."""
    if is_mostly_continuous(text):
        return [char for char in text if is_continuous(char)], MIN_CONTINUOUS_UNITS
    return text.split(), MIN_WORDS


def _unit_dominates(units: list[str]) -> bool:
    return max(Counter(units).values()) / len(units) >= DOMINANCE


def _cycle_repeats(units: list[str]) -> bool:
    """Uma frase curta emendada em si mesma três vezes ou mais é loop, não fala."""
    for size in range(1, min(MAX_CYCLE_UNITS, len(units) // MIN_CYCLE_REPEATS) + 1):
        cycle = units[:size]
        repeats = len(units) // size
        if repeats >= MIN_CYCLE_REPEATS and units[: size * repeats] == cycle * repeats:
            return True
    return False
