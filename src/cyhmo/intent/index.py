"""Índice de candidatos da gramática ativa: exemplos embutidos + busca exaustiva."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from cyhmo.domain.contracts import Candidate
from cyhmo.domain.ports import TextEmbedder
from cyhmo.intent.annex import Annex
from cyhmo.intent.embedding_cache import EmbeddingCache
from cyhmo.intent.language_packs import LanguagePackSet
from cyhmo.intent.normalization import normalized_key
from cyhmo.intent.vocabulary import ActiveGrammar

ExampleProvider = Callable[[str], list[tuple[str, str]]]
LITERAL_LANGUAGE = "en"
DEFAULT_BATCH_SIZE = 64


def build_example_provider(packs: LanguagePackSet, annex: Annex) -> ExampleProvider:
    """A própria string literal + anexo nos idiomas habilitados + exemplos dos pacotes."""

    enabled = set(packs.codes)

    def provide(literal: str) -> list[tuple[str, str]]:
        examples: dict[tuple[str, str], None] = {(literal, LITERAL_LANGUAGE): None}
        for lang, phrases in annex.examples_for(literal).items():
            if lang in enabled:
                for phrase in phrases:
                    examples.setdefault((phrase, lang), None)
        for code, phrases in packs.examples_for(literal).items():
            for phrase in phrases:
                examples.setdefault((phrase, code), None)
        return list(examples)

    return provide


@dataclass(frozen=True)
class _Example:
    text: str
    lang: str
    key_index: int


@dataclass(frozen=True)
class ExactMatch:
    key: str
    example: str
    lang: str


class CandidateIndex:
    def __init__(
        self,
        grammar: ActiveGrammar,
        examples: Sequence[_Example],
        matrix: np.ndarray,
        primary_language: str,
    ) -> None:
        self._grammar = grammar
        self._examples = tuple(examples)
        self._matrix = matrix
        self._example_keys = np.array([example.key_index for example in examples], dtype=np.int64)
        self._has_primary = self._primary_coverage(primary_language)
        self._by_example = self._example_lookup()

    @classmethod
    def build(
        cls,
        grammar: ActiveGrammar,
        provider: ExampleProvider,
        embedder: TextEmbedder,
        cache: EmbeddingCache,
        primary_language: str,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> "CandidateIndex":
        examples = _collect_examples(grammar, provider)
        texts = list(dict.fromkeys(example.text for example in examples))
        vectors = _embed_with_cache(texts, embedder, cache, batch_size)
        matrix = (
            np.stack([vectors[example.text] for example in examples])
            if examples
            else np.zeros((0, embedder.dimension), dtype=np.float32)
        )
        return cls(grammar, examples, _normalize_rows(matrix), primary_language)

    @property
    def grammar(self) -> ActiveGrammar:
        return self._grammar

    @property
    def keys(self) -> tuple[str, ...]:
        return self._grammar.entries

    @property
    def size(self) -> int:
        return len(self._grammar.entries)

    @property
    def example_count(self) -> int:
        return len(self._examples)

    def has_primary_language_examples(self, key: str) -> bool:
        literal = self._grammar.literal_for(key)
        return bool(literal) and self._has_primary.get(literal, False)

    def exact_example(self, text: str) -> ExactMatch | None:
        """Frase curadas: casar exato tem precedência sobre score e margem."""
        example = self._by_example.get(normalized_key(text))
        if example is None:
            return None
        return ExactMatch(self._grammar.entries[example.key_index], example.text, example.lang)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[Candidate]:
        return self.search_variants([query_vector], top_k)

    def search_variants(self, query_vectors: Sequence[np.ndarray], top_k: int) -> list[Candidate]:
        """Cada variante da consulta (original e traduzida) pontua contra o índice e vale o
        melhor score: "armario" sozinho não alcança "Lockers", mas "locker" alcança."""
        if self.example_count == 0 or top_k <= 0 or len(query_vectors) == 0:
            return []
        stacked = np.stack([np.asarray(vector, dtype=np.float32).reshape(-1) for vector in query_vectors])
        scores = (self._matrix @ stacked.T).max(axis=1)
        best_per_key: dict[int, int] = {}
        for row in np.argsort(-scores, kind="stable"):
            key_index = int(self._example_keys[row])
            if key_index not in best_per_key:
                best_per_key[key_index] = int(row)
                if len(best_per_key) == top_k:
                    break
        return [self._candidate(row, float(scores[row])) for row in best_per_key.values()]

    def _candidate(self, row: int, score: float) -> Candidate:
        example = self._examples[row]
        key = self._grammar.entries[example.key_index]
        return Candidate(
            key=key,
            score=score,
            matched_example=example.text,
            example_lang=example.lang,
            has_primary_language_examples=self._has_primary.get(key, False),
        )

    def _example_lookup(self) -> dict[str, _Example]:
        lookup: dict[str, _Example] = {}
        for example in self._examples:
            lookup.setdefault(normalized_key(example.text), example)
        return lookup

    def _primary_coverage(self, primary_language: str) -> dict[str, bool]:
        coverage = {key: False for key in self._grammar.entries}
        for example in self._examples:
            if example.lang == primary_language:
                coverage[self._grammar.entries[example.key_index]] = True
        return coverage


def _collect_examples(grammar: ActiveGrammar, provider: ExampleProvider) -> list[_Example]:
    examples: list[_Example] = []
    for key_index, literal in enumerate(grammar.entries):
        seen: set[str] = set()
        for text, lang in provider(literal):
            if text and text not in seen:
                seen.add(text)
                examples.append(_Example(text=text, lang=lang, key_index=key_index))
    return examples


def _embed_with_cache(
    texts: list[str], embedder: TextEmbedder, cache: EmbeddingCache, batch_size: int
) -> dict[str, np.ndarray]:
    found, missing = cache.get_many(texts, "passage")
    for start in range(0, len(missing), batch_size):
        batch = missing[start : start + batch_size]
        vectors = np.asarray(embedder.embed_passages(batch), dtype=np.float32)
        cache.put_many(batch, "passage", vectors)
        found.update(zip(batch, vectors))
    return found


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return np.ascontiguousarray(matrix / norms, dtype=np.float32)
