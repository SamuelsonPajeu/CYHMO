"""Conjunto de pacotes de idioma habilitados, com o primário definindo precedência."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml
from pydantic import ValidationError

from cyhmo.domain.errors import LanguagePackError
from cyhmo.domain.language_pack import LanguagePack
from cyhmo.intent.normalization import normalize_text, normalized_key
from cyhmo.intent.tokenization import WordTokenizer

PACK_SUFFIX = ".yaml"


class LanguagePackSet:
    def __init__(self, packs: Sequence[LanguagePack], primary: str) -> None:
        if not packs:
            raise LanguagePackError("nenhum pacote de idioma habilitado")
        by_code = {pack.code: pack for pack in packs}
        if primary not in by_code:
            raise LanguagePackError(
                f"idioma primário {primary!r} não está entre os habilitados {sorted(by_code)}"
            )
        ordered = [by_code[primary]] + [pack for pack in packs if pack.code != primary]
        self._packs = tuple(ordered)
        self._primary = by_code[primary]
        self.connectives = _merge_words(pack.connectives for pack in self._packs)
        self.quantifiers = _merge_words(pack.quantifiers for pack in self._packs)
        self.affirmations = _merge_words(pack.affirmations for pack in self._packs)
        self.negations = _merge_words(pack.negations for pack in self._packs)
        self.target_words = _merge_words(pack.target_words for pack in self._packs)
        self.body_parts = _merge_mapping(pack.body_parts for pack in self._packs)
        self.directions = _merge_mapping(pack.directions for pack in self._packs)
        self.lexicon = _merge_mapping(pack.lexicon for pack in self._packs)
        self._cardinals = _merge_numerals(pack.numerals.cardinal for pack in self._packs)
        self._ordinals = _merge_numerals(pack.numerals.ordinal for pack in self._packs)
        self._examples = {
            pack.code: {normalized_key(key): phrases for key, phrases in pack.command_examples.items()}
            for pack in self._packs
        }
        self.tokenizer = self._build_tokenizer()

    @classmethod
    def load(cls, packs_dir: Path, enabled: Sequence[str], primary: str) -> "LanguagePackSet":
        packs: list[LanguagePack] = []
        for code in enabled:
            packs.append(_load_pack(packs_dir, code))
        if primary not in enabled:
            raise LanguagePackError(
                f"languages.primary {primary!r} precisa estar em languages.enabled {list(enabled)}"
            )
        return cls(packs, primary)

    @staticmethod
    def available(packs_dir: Path) -> tuple[list[LanguagePack], list[str]]:
        valid: list[LanguagePack] = []
        problems: list[str] = []
        for path in sorted(Path(packs_dir).glob(f"*{PACK_SUFFIX}")):
            try:
                valid.append(_parse_pack(path))
            except LanguagePackError as exc:
                problems.append(str(exc))
        return valid, problems

    def _build_tokenizer(self) -> WordTokenizer:
        """Basta um pacote de escrita contínua para o corte por espaço deixar de servir: o
        tokenizador contínuo trata trecho latino por espaço, então habilitar zh junto com en
        não prejudica o inglês."""
        vocabulary = [
            *self.connectives,
            *self.quantifiers,
            *self.affirmations,
            *self.negations,
            *self.target_words,
            *self.body_parts,
            *self.directions,
            *self.lexicon,
            *self._cardinals,
            *self._ordinals,
        ]
        spaced = all(pack.word_separator == "space" for pack in self._packs)
        return WordTokenizer(vocabulary, spaced)

    @property
    def primary(self) -> LanguagePack:
        return self._primary

    @property
    def packs(self) -> tuple[LanguagePack, ...]:
        return self._packs

    @property
    def codes(self) -> tuple[str, ...]:
        return tuple(pack.code for pack in self._packs)

    @property
    def stt_language(self) -> str:
        return self._primary.stt_language

    @property
    def stt_languages(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(pack.stt_language for pack in self._packs))

    def numeral(self, word: str) -> int | None:
        word = normalize_text(word)
        value = self._cardinals.get(word)
        return value if value is not None else self._ordinals.get(word)

    def ordinal(self, word: str) -> int | None:
        return self._ordinals.get(normalize_text(word))

    def examples_for(self, key: str) -> dict[str, list[str]]:
        wanted = normalized_key(key)
        found: dict[str, list[str]] = {}
        for code, examples in self._examples.items():
            phrases = examples.get(wanted)
            if phrases:
                found[code] = list(phrases)
        return found

    def summary(self) -> list[dict[str, Any]]:
        return [dict(pack.to_summary(), primary=pack.code == self._primary.code) for pack in self._packs]


def _load_pack(packs_dir: Path, code: str) -> LanguagePack:
    path = Path(packs_dir) / f"{code}{PACK_SUFFIX}"
    if not path.exists():
        available = sorted(p.stem for p in Path(packs_dir).glob(f"*{PACK_SUFFIX}"))
        raise LanguagePackError(
            f"pacote de idioma {code!r} não encontrado em {packs_dir} (disponíveis: {available})"
        )
    pack = _parse_pack(path)
    if pack.code != code:
        raise LanguagePackError(f"{path}: o campo code={pack.code!r} difere do nome do arquivo ({code!r})")
    return pack


def _parse_pack(path: Path) -> LanguagePack:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise LanguagePackError(f"{path}: não foi possível ler o pacote — {exc}") from exc
    if not isinstance(raw, dict):
        raise LanguagePackError(f"{path}: o pacote deve ser um mapeamento YAML")
    try:
        return LanguagePack.model_validate(raw)
    except ValidationError as exc:
        issues = "; ".join(
            ".".join(str(part) for part in issue["loc"]) + ": " + issue["msg"] for issue in exc.errors()
        )
        raise LanguagePackError(f"{path}: pacote inválido — {issues}") from exc


def _merge_words(groups: Iterable[Sequence[str]]) -> tuple[str, ...]:
    merged: dict[str, None] = {}
    for group in groups:
        for word in group:
            merged.setdefault(normalize_text(word), None)
    merged.pop("", None)
    return tuple(merged)


def _merge_mapping(groups: Iterable[dict[str, str]]) -> dict[str, str]:
    merged: dict[str, str] = {}
    for group in groups:
        for source, target in group.items():
            merged.setdefault(normalize_text(source), target)
    return merged


def _merge_numerals(groups: Iterable[dict[str, int]]) -> dict[str, int]:
    merged: dict[str, int] = {}
    for group in groups:
        for word, value in group.items():
            merged.setdefault(normalize_text(word), value)
    return merged
