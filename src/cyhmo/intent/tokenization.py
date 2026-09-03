"""Fronteira de palavra conforme a escrita do idioma.

Chinês e japonês escrevem sem espaço: ``射击嘴巴`` é uma string só, e ``split()`` devolve
um token que nada no pacote casa. Sem fronteira, tudo que trabalha por palavra —
tradução lexical, conectivos, quantificadores, numerais, partes do corpo — fica cego.

A fronteira sai do próprio vocabulário do pacote, por casamento mais longo; o que ele
não cobre vira um token por caractere, que é a menor unidade com significado nessas
escritas. Trecho latino no meio (``shoot 嘴巴`` depois de traduzir em parte, ``3号``) é
cortado por espaço como sempre.

``join`` desfaz o corte: ela é a razão de o tokenizador existir como par e não como
função solta. Espaço entre caracteres Han **degrada o embedding** — medido contra a
gramática de exploração, o acerto em 1º lugar cai de 91,7% para 70,8% e a prosa fora
do jogo *sobe* de 0,429 para 0,490, encostando no limiar de aceite. Por isso o texto
que chega ao modelo nunca carrega espaço que o idioma não escreve.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from cyhmo.domain.writing import is_continuous


class WordTokenizer:
    """``spaced=True`` é o comportamento histórico: corte e junção por espaço."""

    def __init__(self, vocabulary: Iterable[str] = (), spaced: bool = True) -> None:
        self._spaced = spaced
        known = {term for term in vocabulary if term and not term.isspace()}
        self._vocabulary = frozenset(known)
        self._longest = max((len(term) for term in known), default=1)

    @property
    def spaced(self) -> bool:
        return self._spaced

    def split(self, text: str) -> list[str]:
        if self._spaced:
            return text.split()
        tokens: list[str] = []
        position = 0
        while position < len(text):
            if text[position].isspace():
                position += 1
                continue
            span = self._span_at(text, position)
            tokens.append(text[position : position + span])
            position += span
        return tokens

    def join(self, tokens: Sequence[str]) -> str:
        present = [token for token in tokens if token]
        if self._spaced:
            return " ".join(present)
        joined = ""
        for token in present:
            if joined and not (is_continuous(joined[-1]) and is_continuous(token[0])):
                joined += " "
            joined += token
        return joined

    def _span_at(self, text: str, position: int) -> int:
        if not is_continuous(text[position]):
            return self._latin_span(text, position)
        limit = min(self._longest, len(text) - position)
        for span in range(limit, 1, -1):
            if text[position : position + span] in self._vocabulary:
                return span
        return 1

    @staticmethod
    def _latin_span(text: str, position: int) -> int:
        span = 0
        while position + span < len(text):
            char = text[position + span]
            if char.isspace() or is_continuous(char):
                break
            span += 1
        return span
