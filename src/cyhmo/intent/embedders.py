"""Backends de embedding que implementam ``ports.TextEmbedder``."""

from __future__ import annotations

import hashlib
import logging
import unicodedata
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from cyhmo.config.schema import IntentConfig
from cyhmo.domain.errors import CyhmoError
from cyhmo.domain.ports import TextEmbedder

log = logging.getLogger("cyhmo.intent.embedders")

E5_QUERY_PREFIX = "query: "
E5_PASSAGE_PREFIX = "passage: "


class EmbeddingModelError(CyhmoError):
    """Modelo de embeddings indisponível ou incompatível."""


class HashingEmbedder:
    """Vetor determinístico por n-gramas de caracteres — sem modelo; para testes e modo degradado."""

    def __init__(self, dimension: int = 256) -> None:
        if dimension < 8:
            raise ValueError("dimension deve ser >= 8")
        self._dimension = dimension

    @property
    def identity(self) -> str:
        return "hashing:v1"

    @property
    def dimension(self) -> int:
        return self._dimension

    def warm_up(self) -> None:
        return None

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._embed(texts)

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        return self._embed(texts)

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self._dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for ngram in _character_ngrams(text):
                index, sign = _hash_slot(ngram, self._dimension)
                matrix[row, index] += sign
        return _l2_normalize(matrix)


class SentenceTransformersEmbedder:
    def __init__(self, model_name: str, models_dir: Path, device: str = "cpu") -> None:
        self._model_name = model_name
        self._models_dir = Path(models_dir)
        self._device = device
        self._uses_e5_prefixes = "e5" in model_name.lower()
        self._model: Any = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def prefix_scheme(self) -> str:
        return "e5" if self._uses_e5_prefixes else "none"

    @property
    def identity(self) -> str:
        return f"{self._model_name}|{self.prefix_scheme}"

    @property
    def dimension(self) -> int:
        """sentence-transformers 6 renomeou o método; aceitar os dois nomes cobre as duas versões."""
        model = self._load()
        reader = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
        return int(reader())

    def warm_up(self) -> None:
        self.embed_queries(["warm up"])

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        prefix = E5_QUERY_PREFIX if self._uses_e5_prefixes else ""
        return self._encode([prefix + text for text in texts])

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        prefix = E5_PASSAGE_PREFIX if self._uses_e5_prefixes else ""
        return self._encode([prefix + text for text in texts])

    def _encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vectors = self._load().encode(
            texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vectors, dtype=np.float32)

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbeddingModelError(
                    "sentence-transformers não instalado; instale as dependências ou use embedding_backend = \"hashing\""
                ) from exc
            cache_folder = self._models_dir / "embeddings"
            try:
                self._model = SentenceTransformer(
                    self._model_name, cache_folder=str(cache_folder), device=self._device
                )
            except Exception as exc:
                raise EmbeddingModelError(
                    f"não foi possível carregar o modelo {self._model_name!r} em {cache_folder}: {exc}"
                ) from exc
            log.info("modelo de embeddings %s carregado (%s)", self._model_name, self._device)
        return self._model


def build_embedder(config: IntentConfig, models_dir: Path) -> TextEmbedder:
    if config.embedding_backend == "hashing":
        return HashingEmbedder()
    return SentenceTransformersEmbedder(config.embedding_model, models_dir)


def _character_ngrams(text: str, smallest: int = 2, largest: int = 4) -> list[str]:
    padded = f" {unicodedata.normalize('NFKC', text).casefold().strip()} "
    return [
        padded[start : start + size]
        for size in range(smallest, largest + 1)
        for start in range(0, max(0, len(padded) - size + 1))
    ]


def _hash_slot(ngram: str, dimension: int) -> tuple[int, float]:
    digest = hashlib.blake2b(ngram.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "little")
    return value % dimension, 1.0 if (value >> 60) & 1 else -1.0


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (matrix / norms).astype(np.float32)
