"""Pacote de idioma: dados (não código) que adaptam a interpretação a uma língua.

Um pacote vive em ``languages/<code>.yaml``. O usuário pode criar o seu; o mod
só exige o esquema abaixo. Chaves de texto são comparadas já normalizadas
(minúsculas, sem pontuação), então grafe-as assim no arquivo.

``word_separator`` diz como a escrita separa palavras: ``space`` para as que usam
espaço e ``none`` para as que escrevem sem ele (chinês, japonês). Escolher errado
não dá erro — deixa a interpretação cega para tudo que dependa de palavra.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CODE_PATTERN = re.compile(r"^[a-z]{2,3}(-[A-Za-z]{2,4})?$")

WordSeparator = Literal["space", "none"]


class NumeralTable(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cardinal: dict[str, int] = Field(default_factory=dict)
    ordinal: dict[str, int] = Field(default_factory=dict)

    @field_validator("cardinal", "ordinal")
    @classmethod
    def _lowercase_keys(cls, table: dict[str, int]) -> dict[str, int]:
        return {word.strip().lower(): value for word, value in table.items()}

    def lookup(self, word: str) -> int | None:
        return self.cardinal.get(word, self.ordinal.get(word))


class LanguagePack(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    name: str
    stt_language: str
    word_separator: WordSeparator = "space"
    numerals: NumeralTable = Field(default_factory=NumeralTable)
    quantifiers: list[str] = Field(default_factory=list)
    connectives: list[str] = Field(default_factory=list)
    affirmations: list[str] = Field(default_factory=list)
    negations: list[str] = Field(default_factory=list)
    target_words: list[str] = Field(default_factory=list)
    body_parts: dict[str, str] = Field(default_factory=dict)
    directions: dict[str, str] = Field(default_factory=dict)
    lexicon: dict[str, str] = Field(default_factory=dict)
    command_examples: dict[str, list[str]] = Field(default_factory=dict)

    @field_validator("code")
    @classmethod
    def _valid_code(cls, code: str) -> str:
        if not CODE_PATTERN.match(code):
            raise ValueError(f"código de idioma inválido: {code!r} (esperado ex.: 'pt-BR', 'en')")
        return code

    @field_validator("stt_language")
    @classmethod
    def _valid_stt_language(cls, language: str) -> str:
        language = language.strip().lower()
        if not re.fullmatch(r"[a-z]{2,3}", language):
            raise ValueError(f"stt_language deve ser um código ISO curto (recebi {language!r})")
        return language

    @field_validator("quantifiers", "connectives", "affirmations", "negations", "target_words")
    @classmethod
    def _clean_words(cls, words: list[str]) -> list[str]:
        cleaned = [word.strip().lower() for word in words if word and word.strip()]
        return list(dict.fromkeys(cleaned))

    @field_validator("body_parts", "directions", "lexicon")
    @classmethod
    def _clean_mapping(cls, mapping: dict[str, str]) -> dict[str, str]:
        return {source.strip().lower(): target.strip() for source, target in mapping.items()}

    @field_validator("command_examples")
    @classmethod
    def _clean_examples(cls, examples: dict[str, list[str]]) -> dict[str, list[str]]:
        cleaned: dict[str, list[str]] = {}
        for key, phrases in examples.items():
            phrases = [phrase.strip() for phrase in phrases if phrase and phrase.strip()]
            if phrases:
                cleaned[key.strip()] = list(dict.fromkeys(phrases))
        return cleaned

    @model_validator(mode="after")
    def _no_empty_targets(self) -> "LanguagePack":
        for table_name in ("body_parts", "directions", "lexicon"):
            table: dict[str, str] = getattr(self, table_name)
            empty = [source for source, target in table.items() if not target]
            if empty:
                raise ValueError(f"{table_name}: alvo vazio para {empty}")
        return self

    @property
    def is_english(self) -> bool:
        return self.stt_language == "en"

    def examples_for(self, key: str) -> list[str]:
        return self.command_examples.get(key, [])

    def to_summary(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "name": self.name,
            "stt_language": self.stt_language,
            "command_examples": sum(len(phrases) for phrases in self.command_examples.values()),
            "lexicon": len(self.lexicon),
        }
