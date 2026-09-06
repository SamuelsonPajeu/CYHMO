"""Leitura da resposta do assistente: o literal da lista, com ``NONE`` para recusa.

Resposta que não é nenhum dos dois é ILEGÍVEL, nunca "corrigida" para o item mais parecido:
uma key inventada viraria silêncio no jogo, e o chamador tem saída melhor — tratar como falha
do assistente e resgatar o palpite do matcher.

Não há caminho por número: a lista enviada não é numerada desde 2026-08-31, então um dígito
solto não tem a que se referir. Interpretá-lo como posição seria adivinhar em cima de uma
âncora que o prompt não ofereceu — e é exatamente o erro que motivou a troca de protocolo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from cyhmo.domain.contracts import CommandRef
from cyhmo.intent.llm.prompt import REFUSAL_WORD


@dataclass(frozen=True)
class ParsedResponse:
    commands: tuple[CommandRef, ...]

    @property
    def is_empty(self) -> bool:
        return not self.commands


def parse_response(text: str, allowed_keys: Sequence[str]) -> ParsedResponse | None:
    """``None`` é resposta ilegível (o assistente falhou); ``ParsedResponse`` vazio é recusa
    declarada, que é decisão dele e não erro.

    A resposta inteira é tentada primeiro e a PRIMEIRA LINHA depois. Modelo pequeno às vezes
    responde certo e continua falando: em 2026-09-05 o log traz
    ``'Look behind\\nmatcher_guess: NONE'``, em que a primeira linha é uma key válida e o resto
    é o modelo ecoando o rótulo do prompt — como ``canonical_key`` junta tudo numa linha só, a
    resposta virava ilegível e o comando se perdia. Cortar a continuação NÃO é o mesmo que
    corrigir para o item mais parecido, que segue proibido: cada linha ainda precisa casar
    EXATO com um literal da lista.

    Mas a primeira linha só vale quando ela é a ÚNICA resposta ali dentro. Se linhas diferentes
    casam com keys diferentes, o assistente não respondeu uma coisa: respondeu duas, e escolher
    a primeira seria desempatar por posição — a mesma adivinhação que a resposta por índice
    trouxe e que fez o protocolo mudar em 2026-08-31. Aí a resposta é ilegível, e o chamador
    resgata o palpite do matcher, que é uma âncora melhor que a ordem em que o modelo falou."""
    if not allowed_keys:
        return None
    whole = _match(canonical_key(text), allowed_keys)
    if whole is not None:
        return whole
    return _single_line_answer(text, allowed_keys)


def _single_line_answer(text: str, allowed_keys: Sequence[str]) -> ParsedResponse | None:
    """Só a PRIMEIRA linha responde, e só se nenhuma outra disser coisa diferente.

    Aceitar uma key achada em qualquer linha seria pescar: "blue box\\nWalk" viraria ``Walk``
    por causa de uma palavra solta. E duas linhas com keys diferentes não são uma resposta —
    ficar com a primeira desempataria por posição."""
    lines = text.splitlines()
    if not lines:
        return None
    first = _match(canonical_key(lines[0]), allowed_keys)
    if first is None:
        return None
    answer = _answer_of(first)
    for line in lines[1:]:
        other = _match(canonical_key(line), allowed_keys)
        if other is not None and _answer_of(other) != answer:
            return None
    return first


def _answer_of(parsed: ParsedResponse) -> str:
    return parsed.commands[0].key if parsed.commands else REFUSAL_WORD


def _match(wanted: str, allowed_keys: Sequence[str]) -> ParsedResponse | None:
    for key in allowed_keys:
        if canonical_key(key) == wanted:
            return ParsedResponse(commands=(CommandRef(key, {}),))
    if wanted == canonical_key(REFUSAL_WORD):
        return ParsedResponse(commands=())
    return None


def canonical_key(key: str) -> str:
    return " ".join(key.split()).casefold().strip(".,;:!?\"'")
