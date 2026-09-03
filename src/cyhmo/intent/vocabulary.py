"""Níveis de vocabulário: gramática ativa (nível 2) e observado (nível 1)."""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import cached_property
from pathlib import Path
from typing import Iterable, Mapping

import yaml

from cyhmo.intent.normalization import normalized_key

log = logging.getLogger("cyhmo.intent.vocabulary")


@dataclass(frozen=True)
class ActiveGrammar:
    """Nível 2: o único pool de candidatos. ``entries`` preserva a grafia literal do jogo."""

    entries: tuple[str, ...]
    stale: bool = False
    pointer: int | None = None
    version: int = 0

    @classmethod
    def from_entries(
        cls,
        entries: Iterable[str],
        stale: bool = False,
        pointer: int | None = None,
        version: int = 0,
    ) -> "ActiveGrammar":
        cleaned = dict.fromkeys(entry.strip() for entry in entries if entry and entry.strip())
        return cls(tuple(cleaned), stale=stale, pointer=pointer, version=version)

    @cached_property
    def by_key(self) -> Mapping[str, str]:
        mapping: dict[str, str] = {}
        for literal in self.entries:
            mapping.setdefault(normalized_key(literal), literal)
        return mapping

    @cached_property
    def fingerprint(self) -> str:
        material = "\n".join(self.entries).encode("utf-8")
        return hashlib.sha256(material).hexdigest()[:16]

    @property
    def size(self) -> int:
        return len(self.entries)

    @property
    def is_empty(self) -> bool:
        return not self.entries

    def contains(self, text: str) -> bool:
        return normalized_key(text) in self.by_key

    def literal_for(self, text: str) -> str | None:
        return self.by_key.get(normalized_key(text))

    def with_flags(self, stale: bool, pointer: int | None = None) -> "ActiveGrammar":
        return ActiveGrammar(self.entries, stale=stale, pointer=pointer, version=self.version)


EMPTY = ActiveGrammar(())


class ObservedVocabulary:
    """Nível 1: tudo que já apareceu em alguma gramática desta campanha.

    Serve para aquecer o cache de embeddings e para debug. NUNCA é fonte de
    candidatos: uma string daqui pode não existir na cena atual e a
    injeção falharia em silêncio.
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._entries: dict[str, str] = {}
        self._lock = threading.RLock()
        self._dirty = False

    @property
    def path(self) -> Path:
        return self._path

    @property
    def entries(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._entries)

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._entries)

    def load(self) -> int:
        with self._lock:
            self._entries = _read_entries(self._path)
            self._dirty = False
            return len(self._entries)

    def record(self, entries: Iterable[str]) -> int:
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._lock:
            added = 0
            for entry in entries:
                literal = entry.strip()
                if literal and literal not in self._entries:
                    self._entries[literal] = stamp
                    added += 1
            self._dirty = self._dirty or added > 0
            return added

    def save(self) -> bool:
        with self._lock:
            if not self._dirty:
                return False
            payload = {"entries": dict(self._entries)}
            self._path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self._path.with_suffix(self._path.suffix + ".tmp")
            temporary.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=True), encoding="utf-8")
            os.replace(temporary, self._path)
            self._dirty = False
            return True


def _read_entries(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        log.warning("vocabulário observado ilegível em %s (%s); começando vazio", path, exc)
        return {}
    entries = raw.get("entries") if isinstance(raw, dict) else None
    if not isinstance(entries, dict):
        return {}
    return {str(literal): str(seen) for literal, seen in entries.items() if str(literal).strip()}
