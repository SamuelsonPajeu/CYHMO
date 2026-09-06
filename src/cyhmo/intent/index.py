"""Índice de candidatos da gramática ativa: exemplos embutidos + busca exaustiva."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from cyhmo.domain.contracts import Candidate
from cyhmo.domain.ports import TextEmbedder
from cyhmo.intent.annex import Annex
from cyhmo.intent.embedding_cache import EmbeddingCache
from cyhmo.intent.gloss import GrammarGlosser, auto_language, is_auto
from cyhmo.intent.language_packs import LanguagePackSet
from cyhmo.intent.lexical import SpatialIntent, spatial_words
from cyhmo.intent.normalization import normalized_key
from cyhmo.intent.vocabulary import ActiveGrammar

ExampleProvider = Callable[[str], list[tuple[str, str]]]
LITERAL_LANGUAGE = "en"
DEFAULT_BATCH_SIZE = 64


def build_example_provider(
    packs: LanguagePackSet, annex: Annex, glosser: GrammarGlosser | None = None
) -> ExampleProvider:
    """A própria string literal + anexo nos idiomas habilitados + exemplos dos pacotes + a
    tradução automática da cena, quando houver.

    A tradução vem POR ÚLTIMO e com o idioma marcado (``pt-BR~auto``): frase escrita por gente
    tem precedência, e a marca deixa a telemetria mostrar de onde veio o casamento."""

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
        if glosser is not None:
            gloss = glosser.cached([literal]).get(literal)
            if gloss:
                examples.setdefault((gloss, auto_language(glosser.language)), None)
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
        # Por CHAVE, não por exemplo: a chave é o literal que será injetado e é sempre inglesa.
        self._key_spatial = tuple(spatial_words(literal) for literal in grammar.entries)

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

    def search_variants(
        self,
        query_vectors: Sequence[np.ndarray],
        top_k: int,
        intent: SpatialIntent | None = None,
        bonus: float = 0.0,
        band: float = 0.0,
    ) -> list[Candidate]:
        """Cada variante da consulta (original e traduzida) pontua contra o índice e vale o
        melhor score: "armario" sozinho não alcança "Lockers", mas "locker" alcança.

        Os três últimos parâmetros têm default inerte de propósito: ``search`` e quem chamava
        com dois argumentos continuam valendo, e ``bonus = 0`` deixa o caminho idêntico ao de
        antes do desempate lexical."""
        if self.example_count == 0 or top_k <= 0 or len(query_vectors) == 0:
            return []
        stacked = np.stack([np.asarray(vector, dtype=np.float32).reshape(-1) for vector in query_vectors])
        scores = (self._matrix @ stacked.T).max(axis=1)
        key_bonus = self._lexical_bonus(scores, intent, bonus, band)
        if key_bonus is not None:
            scores = scores + key_bonus[self._example_keys]
        best_per_key: dict[int, int] = {}
        for row in np.argsort(-scores, kind="stable"):
            key_index = int(self._example_keys[row])
            if key_index not in best_per_key:
                best_per_key[key_index] = int(row)
                if len(best_per_key) == top_k:
                    break
        return [self._candidate(row, float(scores[row]), key_bonus) for row in best_per_key.values()]

    def _lexical_bonus(
        self, scores: np.ndarray, intent: SpatialIntent | None, bonus: float, band: float
    ) -> np.ndarray | None:
        """Desempate por relação espacial, com quatro invariantes deliberadas.

        1. Inerte quando o jogador não nomeou relação nenhuma — a maioria dos enunciados.
        2. Roda, mas não muda nada, a menos que exista candidato DENTRO DA FAIXA que concorde:
           sem concordância a função devolve ``None``, para que o termo nunca troque uma
           injeção errada por silêncio.
        3. Preserva a ordem do cosseno dentro de cada classe (concorda / neutro / discorda).
        4. A oscilação é limitada a ``2 * bonus`` e confinada à faixa, então um comando que o
           cosseno ganhou com folga não é alcançável.

        Viés conhecido e aceito: entrada com dois membros de famílias diferentes ("Look on the
        back of the invitation" tem "on" e "back") só consegue concordar, nunca discordar — um
        pendor sistemático a favor de entradas longas, que soma ao pendor a favor das curtas
        que o ``.max(axis=1)`` já tem."""
        if bonus <= 0.0 or intent is None or intent.is_empty or self.size == 0:
            return None
        # O melhor score de uma chave só é conhecido depois de reduzir as linhas: cada chave
        # tem uma linha por exemplo, e é o melhor exemplo dela que a representa.
        best = np.full(self.size, -np.inf, dtype=np.float32)
        np.maximum.at(best, self._example_keys, scores)
        ceiling = float(scores.max())
        key_bonus = np.zeros(self.size, dtype=np.float32)
        agreed = False
        for key_index, relations in enumerate(self._key_spatial):
            if not relations or ceiling - float(best[key_index]) > band:
                continue
            if relations & intent.wanted:
                key_bonus[key_index] = bonus
                agreed = True
            else:
                key_bonus[key_index] = -bonus
        return key_bonus if agreed else None

    def _candidate(self, row: int, score: float, key_bonus: np.ndarray | None = None) -> Candidate:
        example = self._examples[row]
        key = self._grammar.entries[example.key_index]
        return Candidate(
            key=key,
            score=score,
            matched_example=example.text,
            example_lang=example.lang,
            has_primary_language_examples=self._has_primary.get(key, False),
            lexical_bonus=0.0 if key_bonus is None else float(key_bonus[example.key_index]),
        )

    def _example_lookup(self) -> dict[str, _Example]:
        """Só frase escrita por GENTE entra no casamento exato.

        Casar exato vale 1,0 e passa por cima de score e margem — é o instrumento mais afiado
        do interpretador, e por isso ele fica com quem responde pelo texto. Tradução de máquina
        entra pelo embedding, onde erra empurrando o score em vez de injetar com certeza."""
        lookup: dict[str, _Example] = {}
        for example in self._examples:
            if is_auto(example.lang):
                continue
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
