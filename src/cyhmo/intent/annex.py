"""Anexo semântico ``catalog/commands.yaml``: categorias e exemplos por idioma."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from cyhmo.domain.errors import CyhmoError
from cyhmo.intent.normalization import normalized_key

log = logging.getLogger("cyhmo.intent.annex")


class AnnexError(CyhmoError):
    """Anexo ilegível ou fora do formato esperado."""


@dataclass(frozen=True)
class AnnexEntry:
    key: str
    category: str | None
    examples: dict[str, tuple[str, ...]] = field(default_factory=dict)


class Annex:
    def __init__(self, entries: dict[str, AnnexEntry] | None = None, source: Path | None = None) -> None:
        self._entries = dict(entries or {})
        self._source = source

    @classmethod
    def load(cls, path: Path | None) -> "Annex":
        if path is None or not Path(path).exists():
            log.warning("anexo de comandos ausente (%s): matching seguirá só com as strings do jogo", path)
            return cls()
        raw = _read_yaml(Path(path))
        commands = raw.get("commands") if isinstance(raw, dict) else None
        if not isinstance(commands, list):
            raise AnnexError(f"{path}: esperado um mapeamento com a lista 'commands'")
        entries: dict[str, AnnexEntry] = {}
        for item in commands:
            entry = _parse_entry(item, Path(path))
            entries.setdefault(normalized_key(entry.key), entry)
        return cls(entries, Path(path))

    @property
    def source(self) -> Path | None:
        return self._source

    @property
    def size(self) -> int:
        return len(self._entries)

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(entry.key for entry in self._entries.values())

    def contains(self, key: str) -> bool:
        return normalized_key(key) in self._entries

    def category(self, key: str) -> str | None:
        entry = self._entries.get(normalized_key(key))
        return None if entry is None else entry.category

    def examples_for(self, key: str) -> dict[str, list[str]]:
        entry = self._entries.get(normalized_key(key))
        if entry is None:
            return {}
        return {lang: list(phrases) for lang, phrases in entry.examples.items()}

    def all_examples(self, languages: tuple[str, ...] | None = None) -> list[str]:
        phrases: dict[str, None] = {}
        for entry in self._entries.values():
            for lang, examples in entry.examples.items():
                if languages is None or lang in languages:
                    for phrase in examples:
                        phrases.setdefault(phrase, None)
        return list(phrases)


def _read_yaml(path: Path) -> Any:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise AnnexError(f"{path}: não foi possível ler o anexo — {exc}") from exc


def _parse_entry(item: Any, path: Path) -> AnnexEntry:
    if not isinstance(item, dict) or not str(item.get("key", "")).strip():
        raise AnnexError(f"{path}: entrada sem 'key': {item!r}")
    raw_examples = item.get("examples") or {}
    if not isinstance(raw_examples, dict):
        raise AnnexError(f"{path}: 'examples' de {item['key']!r} deve mapear idioma → lista de frases")
    examples = {
        str(lang): tuple(str(phrase).strip() for phrase in phrases if str(phrase).strip())
        for lang, phrases in raw_examples.items()
        if isinstance(phrases, list)
    }
    category = item.get("category")
    return AnnexEntry(
        key=str(item["key"]).strip(),
        category=None if category is None else str(category),
        examples={lang: phrases for lang, phrases in examples.items() if phrases},
    )
