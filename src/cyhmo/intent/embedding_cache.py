"""Cache persistido de embeddings por string: uma string é embutida uma vez na vida."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
from io import BytesIO
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np

log = logging.getLogger("cyhmo.intent.cache")

EmbeddingKind = Literal["query", "passage"]
KEY_SEPARATOR = "\x1f"
VECTORS_FILE = "vectors.npy"
KEYS_FILE = "keys.json"


class EmbeddingCache:
    def __init__(self, directory: Path, identity: str, dimension: int) -> None:
        self._directory = Path(directory) / _slug(identity)
        self._identity = identity
        self._dimension = int(dimension)
        self._lock = threading.RLock()
        self._rows: dict[str, int] = {}
        self._vectors: list[np.ndarray] = []
        self._persisted_count = 0
        self._hits = 0
        self._misses = 0
        self._load()

    @property
    def directory(self) -> Path:
        return self._directory

    @property
    def identity(self) -> str:
        return self._identity

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._rows)

    def contains(self, text: str, kind: EmbeddingKind) -> bool:
        with self._lock:
            return _cache_key(kind, text) in self._rows

    def get_many(self, texts: Iterable[str], kind: EmbeddingKind) -> tuple[dict[str, np.ndarray], list[str]]:
        found: dict[str, np.ndarray] = {}
        missing: list[str] = []
        with self._lock:
            for text in dict.fromkeys(texts):
                row = self._rows.get(_cache_key(kind, text))
                if row is None:
                    missing.append(text)
                    self._misses += 1
                else:
                    found[text] = self._vectors[row]
                    self._hits += 1
        return found, missing

    def put_many(self, texts: Sequence[str], kind: EmbeddingKind, vectors: np.ndarray) -> int:
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(texts) or matrix.shape[1] != self._dimension:
            raise ValueError(
                f"vetores {matrix.shape} não casam com {len(texts)} textos de dimensão {self._dimension}"
            )
        added = 0
        with self._lock:
            for text, vector in zip(texts, matrix):
                key = _cache_key(kind, text)
                if key in self._rows:
                    continue
                self._rows[key] = len(self._vectors)
                self._vectors.append(np.ascontiguousarray(vector))
                added += 1
        return added

    def flush(self) -> bool:
        with self._lock:
            if len(self._vectors) == self._persisted_count:
                return False
            self._directory.mkdir(parents=True, exist_ok=True)
            matrix = np.stack(self._vectors) if self._vectors else np.zeros((0, self._dimension), np.float32)
            keys = sorted(self._rows, key=self._rows.__getitem__)
            _atomic_write_bytes(self._directory / VECTORS_FILE, _npy_bytes(matrix))
            _atomic_write_text(self._directory / KEYS_FILE, json.dumps(keys, ensure_ascii=False))
            self._persisted_count = len(self._vectors)
            return True

    def stats(self) -> dict[str, int | str]:
        with self._lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "size": len(self._rows),
                "identity": self._identity,
                "directory": str(self._directory),
            }

    def _load(self) -> None:
        vectors_path = self._directory / VECTORS_FILE
        keys_path = self._directory / KEYS_FILE
        if not vectors_path.exists() or not keys_path.exists():
            return
        try:
            matrix = np.load(vectors_path)
            keys = json.loads(keys_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.warning("cache de embeddings ilegível em %s (%s); recomeçando vazio", self._directory, exc)
            return
        if matrix.ndim != 2 or matrix.shape[1] != self._dimension or matrix.shape[0] != len(keys):
            log.warning("cache de embeddings em %s incompatível com dimensão %d; ignorado", self._directory, self._dimension)
            return
        self._vectors = [np.ascontiguousarray(row, dtype=np.float32) for row in matrix]
        self._rows = {key: index for index, key in enumerate(keys)}
        self._persisted_count = len(self._vectors)


def _cache_key(kind: EmbeddingKind, text: str) -> str:
    return f"{kind}{KEY_SEPARATOR}{text}"


def _slug(identity: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9._-]+", "_", identity).strip("_")[:48]
    digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:8]
    return f"{readable or 'model'}-{digest}"


def _npy_bytes(matrix: np.ndarray) -> bytes:
    buffer = BytesIO()
    np.save(buffer, matrix, allow_pickle=False)
    return buffer.getvalue()


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _atomic_write_text(path: Path, payload: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)
