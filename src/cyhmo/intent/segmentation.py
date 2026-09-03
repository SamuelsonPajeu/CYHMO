"""Segmentação do enunciado em até N comandos: quantificador, repetição, conectivos."""

from __future__ import annotations

from dataclasses import dataclass, field

from cyhmo.intent.language_packs import LanguagePackSet
from cyhmo.intent.numerals import NumeralParser
from cyhmo.intent.translation import LexicalTranslator
from cyhmo.intent.vocabulary import ActiveGrammar


@dataclass(frozen=True)
class SegmentationResult:
    segments: list[str] = field(default_factory=list)
    truncated: bool = False
    repeated_from: str | None = None


@dataclass(frozen=True)
class _Expansion:
    segments: list[str]
    repeated_from: str | None = None


class UtteranceSegmenter:
    def __init__(self, packs: LanguagePackSet) -> None:
        self._numerals = NumeralParser(packs)
        self._translator = LexicalTranslator(packs)
        self._tokenizer = packs.tokenizer
        self._quantifiers = frozenset(packs.quantifiers)
        self._connectives = sorted(packs.connectives, key=lambda phrase: -len(self._tokenizer.split(phrase)))

    def segment(self, normalized_text: str, grammar: ActiveGrammar, max_commands: int) -> SegmentationResult:
        max_commands = max(1, max_commands)
        if self._is_grammar_entry(normalized_text, grammar):
            return SegmentationResult([normalized_text])
        pieces = self._split_by_connectives(normalized_text)
        pieces = self._reassemble(pieces, grammar)
        expanded: list[str] = []
        repeated_from: str | None = None
        for piece in pieces:
            expansion = self._expand(piece, grammar)
            expanded.extend(expansion.segments)
            repeated_from = repeated_from or expansion.repeated_from
        truncated = len(expanded) > max_commands
        return SegmentationResult(expanded[:max_commands], truncated, repeated_from)

    def _is_grammar_entry(self, text: str, grammar: ActiveGrammar) -> bool:
        if grammar.contains(text):
            return True
        return any(grammar.contains(variant) for variant in self._translator.translated_variants(text))

    def _expand(self, piece: str, grammar: ActiveGrammar) -> _Expansion:
        if self._is_grammar_entry(piece, grammar):
            return _Expansion([piece])
        quantified = self._quantified(piece)
        if quantified is not None:
            base, count = quantified
            return _Expansion([base] * count, base)
        repeated = self._repeated(piece)
        if repeated is not None:
            base, count = repeated
            return _Expansion([base] * count, base)
        return _Expansion([piece])

    def _quantified(self, piece: str) -> tuple[str, int] | None:
        tokens = self._tokenizer.split(piece)
        if len(tokens) < 3 or tokens[-1] not in self._quantifiers:
            return None
        count = self._numerals.value_of(tokens[-2])
        if count is None or count < 1:
            return None
        return self._tokenizer.join(tokens[:-2]), count

    def _repeated(self, piece: str) -> tuple[str, int] | None:
        tokens = self._tokenizer.split(piece)
        for size in range(1, len(tokens) // 2 + 1):
            if len(tokens) % size:
                continue
            groups = [tokens[start : start + size] for start in range(0, len(tokens), size)]
            if all(group == groups[0] for group in groups):
                return self._tokenizer.join(groups[0]), len(groups)
        return None

    def _split_by_connectives(self, text: str) -> list[str]:
        tokens = self._tokenizer.split(text)
        pieces: list[str] = []
        current: list[str] = []
        position = 0
        while position < len(tokens):
            span = self._connective_span(tokens, position)
            if span:
                if current:
                    pieces.append(self._tokenizer.join(current))
                    current = []
                position += span
                continue
            current.append(tokens[position])
            position += 1
        if current:
            pieces.append(self._tokenizer.join(current))
        return pieces or [text]

    def _connective_span(self, tokens: list[str], position: int) -> int:
        for connective in self._connectives:
            words = self._tokenizer.split(connective)
            if tokens[position : position + len(words)] == words and position + len(words) < len(tokens):
                return len(words)
        return 0

    def _reassemble(self, pieces: list[str], grammar: ActiveGrammar) -> list[str]:
        """Junção gulosa de pedaços adjacentes que juntos formam uma entrada da gramática."""
        merged: list[str] = []
        position = 0
        while position < len(pieces):
            span = 1
            for end in range(len(pieces), position + 1, -1):
                if self._is_grammar_entry(self._tokenizer.join(pieces[position:end]), grammar):
                    span = end - position
                    break
            merged.append(self._tokenizer.join(pieces[position : position + span]))
            position += span
        return merged
