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
    declarada, que é decisão dele e não erro."""
    if not allowed_keys:
        return None
    wanted = canonical_key(text)
    if not wanted:
        return None
    for key in allowed_keys:
        if canonical_key(key) == wanted:
            return ParsedResponse(commands=(CommandRef(key, {}),))
    if wanted == canonical_key(REFUSAL_WORD):
        return ParsedResponse(commands=())
    return None


def canonical_key(key: str) -> str:
    return " ".join(key.split()).casefold().strip(".,;:!?\"'")
