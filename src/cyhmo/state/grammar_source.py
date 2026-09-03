"""Fontes da gramática ativa: memória do jogo via PINE ou arquivo/lista
para desenvolver sem o jogo aberto."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence, runtime_checkable

import yaml

from cyhmo.domain.errors import GrammarUnavailableError
from cyhmo.inject.pine import BATCH_MAX_BYTES, PineClient, PineError
from cyhmo.inject.recipe import GrammarRecipe
from cyhmo.state.grammar import HEADER_SIZE, extract_vocabulary, parse_records, physical, read_layout

FILE_SOURCE_POINTER = 1

log = logging.getLogger("cyhmo.state.grammar_source")


@dataclass(frozen=True)
class GrammarSnapshot:
    entries: tuple[str, ...]
    pointer: int | None
    blob_address: int | None
    rejected: tuple[str, ...] = ()
    source: str = "records"

    @property
    def size(self) -> int:
        return len(self.entries)


@runtime_checkable
class GrammarSource(Protocol):
    def read_pointer(self) -> int | None: ...

    def read(self, pointer: int | None) -> GrammarSnapshot | None: ...


class PineGrammarSource:
    """Lê o ponteiro de contexto e o blob ``VGP 2.00`` da RAM do EE; erros de PINE viram ``None``."""

    def __init__(self, client: PineClient, recipe_grammar: GrammarRecipe) -> None:
        self._client = client
        self._grammar = recipe_grammar

    def read_pointer(self) -> int | None:
        try:
            return self._client.read32(self._grammar.context_pointer)
        except PineError as exc:
            log.debug("ponteiro de gramática ilegível: %s", exc)
            return None

    def read(self, pointer: int | None) -> GrammarSnapshot | None:
        """A rota estruturada lê só o pool declarado no cabeçalho — alguns KB em vez dos
        128 KB da varredura — e devolve os limites exatos das strings."""
        try:
            blob = self.locate_blob(pointer)
            if blob is None:
                return None
            structured = self._read_records(blob)
            if structured is not None:
                return GrammarSnapshot(structured.entries, pointer, blob, structured.rejected, "records")
            data = self._client.read_range(blob, self._span(blob), block=BATCH_MAX_BYTES)
        except PineError as exc:
            log.warning("leitura da gramática falhou: %s", exc)
            return None
        extraction = extract_vocabulary(data, blob, self._grammar.gap)
        if not extraction.entries:
            return None
        log.info("blob 0x%08X sem tabela de strings utilizável; caindo na varredura por bytes", blob)
        return GrammarSnapshot(extraction.entries, pointer, blob, extraction.rejected, "scan")

    def _read_records(self, blob: int):
        layout = read_layout(self._client.read_range(blob, HEADER_SIZE), self._grammar.magic_bytes)
        if layout is None or layout.required_bytes > self._span(blob):
            return None
        data = self._client.read_range(blob, layout.required_bytes, block=BATCH_MAX_BYTES)
        extraction = parse_records(data, blob, self._grammar.magic_bytes)
        return extraction if extraction is not None and extraction.entries else None

    def _span(self, blob: int) -> int:
        return min(self._grammar.span, self._grammar.ee_size - blob)

    def locate_blob(self, pointer: int | None) -> int | None:
        """O ponteiro de contexto é a única fonte: sem assinatura no alvo, a cena não tem
        gramática (menu, tela de título, carregamento) e a resposta honesta é ``None``.

        Aqui existia uma varredura pelos 31 MB da RAM que adotava o blob de endereço mais alto.
        Ela custava minutos por passada, prendia a thread de polling — o ponteiro deixava de ser
        relido e entrar na cena não era notado — e, quando achava algo, adotava como fresco um
        entre os ~14 blobs residentes, escolhido por endereço. Palpite marcado como certeza é
        exatamente o que faz o mod mandar o comando de outra cena.
        """
        if pointer is None or not self._has_magic(physical(pointer)):
            return None
        return physical(pointer)

    def _has_magic(self, address: int) -> bool:
        magic = self._grammar.magic_bytes
        if not 0 < address <= self._grammar.ee_size - len(magic):
            return False
        return self._client.read_range(address, len(magic)) == magic


class FileGrammarSource:
    """Gramática vinda de um arquivo YAML (lista, ``{entries: [...]}`` ou ``{gramaticas: [...]}``) ou de uma lista."""

    def __init__(self, source: Path | str | Sequence[str], context: str | None = None) -> None:
        self._entries = tuple(load_grammar_entries(source, context))

    @property
    def entries(self) -> tuple[str, ...]:
        return self._entries

    def read_pointer(self) -> int | None:
        return FILE_SOURCE_POINTER if self._entries else None

    def read(self, pointer: int | None) -> GrammarSnapshot | None:
        if not self._entries:
            return None
        return GrammarSnapshot(self._entries, pointer, None, source="file")


class SeededGrammarSource:
    """Semente de arquivo enquanto a memória do jogo nunca respondeu, e só até lá.

    O mod agora nasce com o cliente PINE mesmo sem o PCSX2 aberto, para conectar sozinho
    quando o emulador aparecer. Sem esta fonte, quem desenvolve sem o jogo perderia a
    gramática de arquivo que a ``state.grammar_seed`` promete. A semente sai de cena na
    primeira leitura viva e não volta: uma cena real nunca deve ser interpretada contra o
    catálogo inteiro.
    """

    def __init__(self, live: GrammarSource, seed: GrammarSource) -> None:
        self._live = live
        self._seed = seed
        self._live_answered = False
        self._serving_seed = False

    @property
    def serving_seed(self) -> bool:
        return self._serving_seed

    def read_pointer(self) -> int | None:
        pointer = self._live.read_pointer()
        if pointer is not None:
            self._live_answered = True
        self._serving_seed = pointer is None and not self._live_answered
        return self._seed.read_pointer() if self._serving_seed else pointer

    def read(self, pointer: int | None) -> GrammarSnapshot | None:
        return self._seed.read(pointer) if self._serving_seed else self._live.read(pointer)


class EmptyGrammarSource:
    """Sem PCSX2 e sem semente: nunca há gramática."""

    def read_pointer(self) -> int | None:
        return None

    def read(self, pointer: int | None) -> GrammarSnapshot | None:
        return None


def load_grammar_entries(source: Path | str | Sequence[str], context: str | None = None) -> list[str]:
    if isinstance(source, (str, Path)):
        return _entries_from_file(Path(source), context)
    return [str(entry) for entry in source if str(entry).strip()]


def _entries_from_file(path: Path, context: str | None) -> list[str]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise GrammarUnavailableError(f"arquivo de gramática não encontrado: {path} ({exc})") from exc
    except yaml.YAMLError as exc:
        raise GrammarUnavailableError(f"{path}: YAML inválido — {exc}") from exc
    if isinstance(data, list):
        return [str(entry) for entry in data]
    if isinstance(data, dict):
        if isinstance(data.get("entries"), list):
            return [str(entry) for entry in data["entries"]]
        if isinstance(data.get("gramaticas"), list):
            return _entries_from_catalog(data["gramaticas"], context, path)
    raise GrammarUnavailableError(f"{path}: formato não reconhecido (esperava lista, 'entries' ou 'gramaticas')")


def _entries_from_catalog(grammars: list[dict[str, Any]], context: str | None, path: Path) -> list[str]:
    if not grammars:
        raise GrammarUnavailableError(f"{path}: catálogo sem gramáticas")
    if context is None:
        return [str(entry) for entry in grammars[0].get("comandos", [])]
    for grammar in grammars:
        if grammar.get("contexto") == context:
            return [str(entry) for entry in grammar.get("comandos", [])]
    available = sorted({str(grammar.get("contexto")) for grammar in grammars})
    raise GrammarUnavailableError(f"{path}: contexto {context!r} não existe (disponíveis: {available})")
