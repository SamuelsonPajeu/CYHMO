"""Tradução lexical pt/es → inglês para casar literalmente com a gramática ("atira na boca" → "shoot mouth")."""

from __future__ import annotations

from cyhmo.intent.language_packs import LanguagePackSet
from cyhmo.intent.normalization import normalize_text
from cyhmo.intent.numerals import NumeralParser

_FUNCTION_WORDS: dict[str, frozenset[str]] = {
    "pt": frozenset({"no", "na", "o", "a", "os", "as", "de", "do", "da", "em", "pra", "para", "ao", "à"}),
    "es": frozenset({"el", "la", "los", "las", "de", "del", "al", "en", "a"}),
    "en": frozenset({"the", "a", "an", "to", "at", "on", "in"}),
}


class LexicalTranslator:
    def __init__(self, packs: LanguagePackSet) -> None:
        self._numerals = NumeralParser(packs)
        self._tokenizer = packs.tokenizer
        self._table = _build_table(packs)
        self._longest_phrase = max((len(self._tokenizer.split(source)) for source in self._table), default=1)
        self._function_words = frozenset().union(
            *(_FUNCTION_WORDS.get(language, frozenset()) for language in packs.stt_languages)
        )

    def translate(self, normalized_text: str) -> str:
        tokens = self._tokenizer.split(self._numerals.replace_numerals(normalized_text))
        translated: list[str] = []
        position = 0
        while position < len(tokens):
            span, target = self._longest_match(tokens, position)
            translated.extend(self._tokenizer.split(target) if span else [tokens[position]])
            position += span or 1
        return self._tokenizer.join(translated)

    def translated_variants(self, normalized_text: str) -> list[str]:
        translated = self.translate(normalized_text)
        variants = [
            translated,
            self._strip_function_words(translated),
            normalize_text(normalized_text),
            self._strip_function_words(normalize_text(normalized_text)),
        ]
        return [variant for variant in dict.fromkeys(variants) if variant]

    def _longest_match(self, tokens: list[str], position: int) -> tuple[int, str]:
        for span in range(min(self._longest_phrase, len(tokens) - position), 0, -1):
            phrase = self._tokenizer.join(tokens[position : position + span])
            target = self._table.get(phrase)
            if target is not None:
                return span, target
        return 0, ""

    def _strip_function_words(self, text: str) -> str:
        kept = [token for token in self._tokenizer.split(text) if token not in self._function_words]
        return self._tokenizer.join(kept)


def _build_table(packs: LanguagePackSet) -> dict[str, str]:
    table: dict[str, str] = {}
    for source, target in packs.lexicon.items():
        table.setdefault(source, normalize_text(target))
    for mapping in (packs.body_parts, packs.directions):
        for source, target in mapping.items():
            table.setdefault(source, normalize_text(target))
    return {source: target for source, target in table.items() if source and target and source != target}
