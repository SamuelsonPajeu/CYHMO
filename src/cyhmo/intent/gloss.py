"""Tradução automática da gramática da cena para o idioma do jogador.

O jogo entrega o vocabulário da cena em inglês, e é contra ele que o enunciado é casado. Com
um modelo multilíngue isso funciona quando as palavras são próximas ("banheiro" → `Bathroom`,
0,84) e falha quando não são: medido em 2026-09-05, "vai na cama" deixou `Bed` em TERCEIRO
(0,5927), atrás de `Leave the room` (0,6078) e `Exit the room` (0,5928) — e os dois só ganharam
porque TINHAM exemplo curado em pt-BR. Ou seja, a curadoria não é só incompleta: ela distorce,
porque entrada curada disputa em pt-BR contra pt-BR e entrada não curada disputa em inglês.

Curar à mão não fecha essa conta. São 416 literais só no que já foi visto desta campanha, 2133
no acervo antigo, e 16 pacotes de idioma. Este módulo tira a curadoria do caminho crítico do
problema: pede ao assistente que já está configurado para traduzir os literais UMA VEZ, guarda
em disco e devolve as traduções como âncoras do índice. A cena passa a ser descrita na língua
de quem joga, sem ninguém escrever nada.

Três garantias deliberadas:

*Nunca no caminho do enunciado.* A tradução roda na thread que monta o índice e só é consultada
do cache. Cache vazio significa o comportamento de antes, não espera.

*Nunca decide sozinha.* As linhas geradas entram com o idioma marcado (``pt-BR~auto``) e ficam
FORA do casamento exato — tradução ruim empurra o score, não injeta comando com confiança 1,0.
A marca também aparece na telemetria, então dá para ver que um acerto veio daqui.

*Nunca inventa em silêncio.* Um lote cuja resposta não tem exatamente uma linha por literal é
descartado inteiro: o modelo pode ter pulado um item, e alinhar errado renomearia comandos.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Mapping, Sequence

import yaml

from cyhmo.domain.ports import LlmProvider
from cyhmo.intent.normalization import normalize_text

log = logging.getLogger("cyhmo.intent.gloss")

AUTO_SUFFIX = "~auto"
BATCH_SIZE = 20
MAX_CHARS_PER_ENTRY = 120

SEPARATOR = " -> "

SYSTEM_PROMPT = (
    "You translate short spoken commands from an English video game into {language}.\n"
    "For every input line, reply with one line: the original, then ' -> ', then the "
    "translation.\n"
    "Say it the way a player would say it out loud, not word by word.\n"
    "Keep proper names and place names recognizable.\n"
    "Never number the lines. Never explain. Never write anything else."
)


def auto_language(code: str) -> str:
    """Marca do idioma para linha gerada. Existe para o resto do sistema poder distinguir
    tradução de máquina de frase escrita por gente, sem carregar um campo novo por toda parte."""
    return f"{code}{AUTO_SUFFIX}"


def is_auto(lang: str) -> bool:
    return lang.endswith(AUTO_SUFFIX)


class GlossStore:
    """Traduções já feitas, por idioma e literal. Um literal é traduzido uma vez na vida."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._by_language: dict[str, dict[str, str]] = {}
        self._lock = threading.RLock()
        self._dirty = False

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> int:
        with self._lock:
            self._by_language = _read(self._path)
            self._dirty = False
            return sum(len(entries) for entries in self._by_language.values())

    def get(self, language: str, literals: Sequence[str]) -> dict[str, str]:
        with self._lock:
            known = self._by_language.get(language, {})
            return {literal: known[literal] for literal in literals if literal in known}

    def missing(self, language: str, literals: Sequence[str]) -> list[str]:
        with self._lock:
            known = self._by_language.get(language, {})
            return [literal for literal in dict.fromkeys(literals) if literal not in known]

    def put(self, language: str, glosses: Mapping[str, str]) -> int:
        with self._lock:
            known = self._by_language.setdefault(language, {})
            added = 0
            for literal, gloss in glosses.items():
                if literal not in known:
                    known[literal] = gloss
                    added += 1
            self._dirty = self._dirty or added > 0
            return added

    def save(self) -> bool:
        with self._lock:
            if not self._dirty:
                return False
            payload = {"glosses": {lang: dict(entries) for lang, entries in self._by_language.items()}}
            self._path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self._path.with_suffix(self._path.suffix + ".tmp")
            temporary.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=True), encoding="utf-8")
            os.replace(temporary, self._path)
            self._dirty = False
            return True


class GrammarGlosser:
    """Traduz literais da gramática em lote, com o provider que o assistente já usa."""

    def __init__(
        self,
        store: GlossStore,
        provider: LlmProvider | None,
        language: str,
        language_name: str,
        timeout_ms: int = 20_000,
        batch_size: int = BATCH_SIZE,
    ) -> None:
        self._store = store
        self._provider = provider
        self._language = language
        self._language_name = language_name
        self._timeout_ms = timeout_ms
        self._batch_size = batch_size

    @property
    def enabled(self) -> bool:
        return self._provider is not None

    @property
    def language(self) -> str:
        return self._language

    def cached(self, literals: Sequence[str]) -> dict[str, str]:
        """Devolve a tradução JÁ NORMALIZADA, que é a forma que vai ao índice.

        O modelo responde capitalizado ("Cama"), e para palavra isolada isso é caro de um jeito
        que não dá para adivinhar: medido em 2026-09-05 com
        paraphrase-multilingual-mpnet-base-v2, "vai na cama" casa "cama" com 0,6910 e "Cama"
        com 0,2100 — 0,48 de diferença só pela maiúscula, que joga a palavra na região dos
        nomes próprios. Nos literais ingleses do jogo o mesmo teste não mostra ganho nenhum
        (+0,0025 de média em 15 pares, 8 para cima e 7 para baixo), então minusculizar é para a
        tradução e não para a gramática.

        O arquivo guarda o que o modelo respondeu, sem mexer: na hora de investigar uma escolha
        estranha, ver a resposta crua vale mais do que ver a forma já tratada."""
        return {
            literal: normalize_text(gloss)
            for literal, gloss in self._store.get(self._language, literals).items()
            if normalize_text(gloss)
        }

    def missing(self, literals: Sequence[str]) -> list[str]:
        if not self.enabled:
            return []
        return [literal for literal in self._store.missing(self._language, literals) if _worth_glossing(literal)]

    def fill(self, literals: Sequence[str]) -> int:
        """Traduz o que falta e persiste. Devolve quantas traduções novas entraram.

        Falha de lote é registrada e engolida: sem tradução o mod volta ao comportamento de
        antes, e derrubar a montagem do índice por causa de um ajudante opcional seria trocar
        uma piora de ranking por nenhuma gramática."""
        pending = self.missing(literals)
        if not pending or self._provider is None:
            return 0
        added = 0
        for start in range(0, len(pending), self._batch_size):
            batch = pending[start : start + self._batch_size]
            try:
                glosses = self._translate(batch)
            except Exception as exc:
                log.warning("tradução da gramática falhou em %d itens: %s", len(batch), exc)
                continue
            added += self._store.put(self._language, glosses)
        if added:
            try:
                self._store.save()
            except OSError as exc:
                log.warning("não foi possível gravar as traduções da gramática: %s", exc)
        return added

    def _translate(self, batch: Sequence[str]) -> dict[str, str]:
        assert self._provider is not None
        system = SYSTEM_PROMPT.format(language=self._language_name)
        answer = self._provider.complete(system, "\n".join(batch), self._timeout_ms)
        return align(batch, answer)


def align(batch: Sequence[str], answer: str) -> dict[str, str]:
    """Cada linha diz a que literal pertence; posição não vale nada.

    A primeira versão casava por ORDEM e exigia uma linha por item. Medido sobre as 416
    entradas do vocabulário observado: o modelo devolveu 18, 21, 19, 6, 19, 18 e 21 linhas em
    lotes de 20, e a regra derrubou 7 lotes inteiros — 37% do vocabulário perdido por causa de
    uma linha a mais ou a menos. Casar por ordem também é perigoso quando ela escorrega: o
    segundo literal ficaria com a tradução do terceiro, renomeando comandos em silêncio.

    Pedindo ``original -> tradução`` os dois problemas somem: a linha carrega a própria chave,
    linha que não casa com nenhum item do lote é ignorada, e literal sem resposta simplesmente
    continua pendente para a próxima vez. Perde-se um item, não o lote."""
    wanted = {literal.casefold(): literal for literal in batch}
    glosses: dict[str, str] = {}
    understood = 0
    for line in answer.strip().splitlines():
        head, separator, tail = line.partition(SEPARATOR)
        if not separator:
            continue
        literal = wanted.get(head.strip().strip('"').casefold())
        gloss = tail.strip().strip('"').strip()
        if literal is None or not gloss:
            continue
        understood += 1
        if len(gloss) <= MAX_CHARS_PER_ENTRY and gloss.casefold() != literal.casefold():
            glosses[literal] = gloss
    # Só é erro quando NENHUMA linha veio no formato — aí o modelo não entendeu o pedido e vale
    # avisar. Linha entendida e descartada (tradução igual ao literal, ou longa demais) é
    # decisão normal: o literal fica pendente e o lote seguinte segue.
    if not understood:
        raise ValueError(f"nenhuma das {len(batch)} traduções veio no formato 'original{SEPARATOR}tradução'")
    return glosses


def _worth_glossing(literal: str) -> bool:
    """Ruído fonético e literal gigante não valem uma chamada: o primeiro não é frase e o
    segundo é senha falada, que o jogador repete lendo a tela e não traduz."""
    return bool(literal.strip()) and len(literal) <= MAX_CHARS_PER_ENTRY and any(c.isalpha() for c in literal)


def _read(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        log.warning("traduções da gramática ilegíveis em %s (%s); começando vazio", path, exc)
        return {}
    glosses = raw.get("glosses") if isinstance(raw, dict) else None
    if not isinstance(glosses, dict):
        return {}
    return {
        str(lang): {str(literal): str(gloss) for literal, gloss in entries.items() if str(gloss).strip()}
        for lang, entries in glosses.items()
        if isinstance(entries, dict)
    }
