"""Receita de injeção: descrição declarativa dos endereços e valores
que a camada 4 escreve. Vive em ``config/write_recipe.yaml``; o código não tem endereço.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from cyhmo.domain.errors import InjectionError

SLOT_ALIGNMENT = 8
MAX_PHYSICAL_SLOTS = 8


def _parse_int(value: Any) -> Any:
    if isinstance(value, bool):
        raise ValueError("esperava inteiro, recebi booleano")
    if isinstance(value, str):
        return int(value.strip(), 0)
    return value


def _crc_text(value: Any) -> Any:
    if isinstance(value, int) and not isinstance(value, bool):
        return f"{value:08X}"
    if isinstance(value, str):
        return value.strip().upper()
    return value


Address = Annotated[int, BeforeValidator(_parse_int), Field(ge=0, lt=0x100000000)]
Word = Annotated[int, BeforeValidator(_parse_int), Field(ge=0, lt=0x100000000)]
Crc = Annotated[str, BeforeValidator(_crc_text)]


class RecipeModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AddressedValue(RecipeModel):
    addr: Address
    value: Word


class CommitField(RecipeModel):
    addr: Address
    width: Literal[8, 16, 32, 64] = 32
    value: Word


class CodePatch(RecipeModel):
    addr: Address
    original: Word
    patched: Word
    verify_original: bool = True
    on_orphan: Literal["adopt", "abort"] = "adopt"
    restore: Literal["always"] = "always"


class ConfidenceField(RecipeModel):
    """Onde o jogo guarda o `conf` do motor de voz e contra que limiar ele o compara.

    ``maximum`` é do formato, não preferência: o jogo só contabiliza a pontuação quando ela
    cabe abaixo de 10000 (`if (conf < 10000)` no registrador do resultado)."""

    addr: Address
    width: Literal[8, 16, 32, 64] = 32
    threshold_addr: Address
    maximum: int = Field(default=9999, ge=1)


class OracleAddresses(RecipeModel):
    candidate_ids: Address
    accepted_ids: Address
    matched_text: Address
    action_marks: tuple[Address, ...] = ()


class AsciiWordsRecipe(RecipeModel):
    words_addr: Address
    slot_size: int = Field(ge=SLOT_ALIGNMENT)
    max_slots: int = Field(ge=1, le=MAX_PHYSICAL_SLOTS)
    ptrs_addr: Address
    count_addr: Address
    encoding: Literal["ascii"] = "ascii"
    terminator: str = "\x00"
    zero_slot_before_write: bool = True
    id_addr: Address
    id_sentinel: Word
    id_default: Word
    asr_state: AddressedValue
    listening: AddressedValue
    confidence: ConfidenceField
    commit: CommitField
    code_patch: CodePatch
    oracle: OracleAddresses

    @field_validator("slot_size")
    @classmethod
    def _aligned_slot(cls, size: int) -> int:
        if size % SLOT_ALIGNMENT:
            raise ValueError(f"slot_size deve ser múltiplo de {SLOT_ALIGNMENT} (matcher de 8 bytes); recebi {size}")
        return size

    @field_validator("zero_slot_before_write")
    @classmethod
    def _zeroing_is_mandatory(cls, enabled: bool) -> bool:
        if not enabled:
            raise ValueError(
                "zero_slot_before_write precisa ser true: o matcher compara em blocos de 8 bytes "
                "e resíduo no slot faz o comando falhar em silêncio"
            )
        return enabled

    @field_validator("terminator")
    @classmethod
    def _nul_terminator(cls, terminator: str) -> str:
        if terminator != "\x00":
            raise ValueError("terminator deve ser o byte nulo (\\x00)")
        return terminator

    @model_validator(mode="after")
    def _default_id_is_not_sentinel(self) -> "AsciiWordsRecipe":
        if self.id_default == self.id_sentinel:
            raise ValueError("id_default não pode ser igual a id_sentinel (o jogo trataria como falha)")
        return self

    @property
    def max_text_length(self) -> int:
        return self.slot_size - 1

    def slot_address(self, index: int) -> int:
        return self.words_addr + index * self.slot_size

    def pointer_address(self, index: int) -> int:
        return self.ptrs_addr + index * 4


class GrammarRecipe(RecipeModel):
    context_pointer: Address
    magic: str = "VGP 2.00"
    ee_size: Address = 0x02000000
    span: Word = 0x20000
    gap: Word = 0x400

    @field_validator("magic")
    @classmethod
    def _ascii_magic(cls, magic: str) -> str:
        if not magic or not magic.isascii():
            raise ValueError("magic deve ser uma assinatura ASCII não vazia")
        return magic

    @property
    def magic_bytes(self) -> bytes:
        return self.magic.encode("ascii")


class WriteRecipe(RecipeModel):
    version: Literal[2]
    serial: str
    crc: Crc
    active: Literal["ascii_words"]
    ascii_words: AsciiWordsRecipe
    grammar: GrammarRecipe

    @field_validator("serial")
    @classmethod
    def _serial_text(cls, serial: str) -> str:
        serial = serial.strip().upper()
        if not serial:
            raise ValueError("serial vazio")
        return serial

    @classmethod
    def load(cls, path: Path | str) -> "WriteRecipe":
        path = Path(path)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise InjectionError(f"receita de injeção não encontrada: {path} ({exc})") from exc
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise InjectionError(f"{path}: YAML inválido — {exc}") from exc
        return cls.from_mapping(data, source=str(path))

    @classmethod
    def from_mapping(cls, data: Any, source: str = "receita") -> "WriteRecipe":
        if not isinstance(data, dict):
            raise InjectionError(f"{source}: a receita deve ser um mapeamento YAML com version/serial/ascii_words")
        try:
            return cls.model_validate(data)
        except ValidationError as exc:
            raise InjectionError(_describe(exc, source)) from exc

    def validate_serial(self, serial: str) -> None:
        """A receita só vale para a versão do jogo em que foi derivada."""
        if serial.strip().upper() != self.serial:
            raise InjectionError(
                f"receita é de outra versão do jogo: receita para {self.serial}, jogo rodando {serial.strip() or '?'}"
            )


def _describe(error: ValidationError, source: str) -> str:
    lines = [f"{source}: receita de injeção inválida"]
    for issue in error.errors():
        location = ".".join(str(part) for part in issue["loc"]) or "(raiz)"
        detail = "chave desconhecida" if issue["type"] == "extra_forbidden" else issue["msg"]
        lines.append(f"  - {location}: {detail} (recebido: {issue.get('input')!r})")
    return "\n".join(lines)
