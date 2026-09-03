"""Resolução de argumentos: alvo numerado, parte do corpo, direção e dígito do cofre."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cyhmo.domain.contracts import GameState
from cyhmo.intent.language_packs import LanguagePackSet
from cyhmo.intent.normalization import normalize_text
from cyhmo.intent.numerals import NumeralParser
from cyhmo.intent.translation import LexicalTranslator

ENGLISH_DIGITS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")


@dataclass(frozen=True)
class ResolvedSegment:
    query_text: str
    english_candidates: list[str] = field(default_factory=list)
    args: dict[str, Any] = field(default_factory=dict)
    reject_reason: str | None = None


class ArgumentResolver:
    def __init__(self, packs: LanguagePackSet) -> None:
        self._numerals = NumeralParser(packs)
        self._translator = LexicalTranslator(packs)
        self._tokenizer = packs.tokenizer
        self._target_words = _by_length(packs.target_words)
        self._body_parts = {source: target.upper() for source, target in packs.body_parts.items()}
        self._directions = {source: target.upper() for source, target in packs.directions.items()}
        self._english_parts = {target.lower(): target for target in self._body_parts.values()}
        self._english_directions = {target.lower(): target for target in self._directions.values()}

    def resolve(self, segment: str, state: GameState) -> ResolvedSegment:
        segment = normalize_text(segment)
        if not segment:
            return ResolvedSegment("", reject_reason="empty")
        tokens = self._tokenizer.split(self._numerals.replace_numerals(segment))
        numbers = self._numerals.find_numbers(tokens)
        variants = self._translator.translated_variants(segment)
        english = self._tokenizer.split(variants[0])
        direction = self._find(tokens, self._directions) or self._find(english, self._english_directions)
        part = self._find(tokens, self._body_parts) or self._find(english, self._english_parts)
        if direction and numbers and not part:
            return self._dial(segment, direction, numbers[0][1])
        if numbers and self._is_target(segment, tokens, direction, part):
            return self._target(segment, numbers[0][1], state)
        if part:
            return ResolvedSegment(segment, variants + [part.lower()], {"part": part})
        if direction:
            return ResolvedSegment(segment, variants + [direction.lower()], {"direction": direction})
        return ResolvedSegment(segment, variants)

    def _is_target(self, segment: str, tokens: list[str], direction: str | None, part: str | None) -> bool:
        if direction or part:
            return False
        if any(self._numerals.is_ordinal(token) for token in self._tokenizer.split(segment)):
            return True
        if self._find(tokens, self._target_words):
            return True
        return len(tokens) == 1

    def _target(self, segment: str, number: int, state: GameState) -> ResolvedSegment:
        args = {"target": number}
        candidates = [f"number {number}", f"number {_english_digit(number)}"]
        enemy_count = state.enemy_count
        if enemy_count is not None and number > enemy_count:
            return ResolvedSegment(segment, candidates, args, f"target_out_of_range:{number}>{enemy_count}")
        if number < 1:
            return ResolvedSegment(segment, candidates, args, f"target_out_of_range:{number}<1")
        return ResolvedSegment(segment, candidates, args)

    @staticmethod
    def _dial(segment: str, direction: str, digit: int) -> ResolvedSegment:
        word = direction.lower()
        candidates = [f"{word} {digit}", f"{word} {_english_digit(digit)}"]
        return ResolvedSegment(segment, candidates, {"direction": direction, "digit": digit})

    def _find(self, tokens: list[str], table: dict[str, str]) -> str | None:
        longest = max((len(self._tokenizer.split(source)) for source in table), default=0)
        for span in range(longest, 0, -1):
            for start in range(0, len(tokens) - span + 1):
                phrase = self._tokenizer.join(tokens[start : start + span])
                if phrase in table:
                    return table[phrase]
        return None


def _by_length(words: tuple[str, ...]) -> dict[str, str]:
    return {word: word for word in words}


def _english_digit(number: int) -> str:
    return ENGLISH_DIGITS[number] if 0 <= number < len(ENGLISH_DIGITS) else str(number)
