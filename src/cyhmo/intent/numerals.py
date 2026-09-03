"""Numerais por extenso → dígitos, nos idiomas habilitados."""

from __future__ import annotations

from cyhmo.intent.language_packs import LanguagePackSet
from cyhmo.intent.normalization import normalize_text


class NumeralParser:
    def __init__(self, packs: LanguagePackSet) -> None:
        self._packs = packs

    def value_of(self, word: str) -> int | None:
        word = normalize_text(word)
        if word.isdigit():
            return int(word)
        return self._packs.numeral(word)

    def is_ordinal(self, word: str) -> bool:
        return self._packs.ordinal(word) is not None

    def replace_numerals(self, text: str) -> str:
        replaced = []
        for token in self._packs.tokenizer.split(normalize_text(text)):
            value = self.value_of(token)
            replaced.append(token if value is None else str(value))
        return self._packs.tokenizer.join(replaced)

    def find_numbers(self, tokens: list[str]) -> list[tuple[int, int]]:
        found: list[tuple[int, int]] = []
        for index, token in enumerate(tokens):
            value = self.value_of(token)
            if value is not None:
                found.append((index, value))
        return found
