"""Evidência lexical de relação espacial, para desempatar comandos quase idênticos.

O modelo de embeddings multilíngue comprime a faixa e não distingue preposição de lugar:
medido em 2026-09-05, "olha atrás do convite" pontuou ``look in the invitation`` 0,9155
contra ``Look on the backside of the invitation`` 0,8994 e ``Look on the back of the
invitation`` 0,8993 — três frases a 0,017 uma da outra, e a mais curta e genérica ganhou.
O limiar fez o papel dele (margem 0,0165 < 0,05 mandou ao assistente); quem errou foi a
RECUPERAÇÃO, então corrigir limiar não resolveria.

As famílias em inglês vivem aqui, em CÓDIGO, e não no pacote de idioma, porque descrevem a
língua do JOGO: a gramática que o Lifeline aceita é sempre inglesa. Ao pacote cabe só o lado
do jogador ("atrás" → ``back``), que é o que mantém os outros 15 pacotes uma edição de dados.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping, Sequence

_WORDS = re.compile(r"[\w']+", re.UNICODE)

# Cada família é um conjunto de sinônimos que o jogo usa para a MESMA relação. Concordar com
# qualquer membro conta como concordar com a família.
SPATIAL_FAMILIES: tuple[tuple[str, ...], ...] = (
    ("back", "backside", "behind", "rear", "reverse"),
    ("inside", "in", "into", "within"),
    ("under", "underneath", "beneath", "below", "bottom"),
    ("on", "top", "above"),
    ("side", "beside", "alongside"),
    ("between",),
    ("around",),
    ("front", "forward", "ahead"),
    ("outside", "outer"),
)

# Fora de propósito, porque são palavras comuns da gramática e não relação de lugar: "close"
# ("Close the shutter"), "next" ("The next room"), "away" ("get away"), "over" ("Look over
# your shoulder" — a injeção errada que o interpretador cita como caso conhecido) e toda a
# família perto/longe. Uma delas aqui faria uma entrada legítima levar -bonus por discordar.

_FAMILY_OF: dict[str, frozenset[str]] = {
    word: frozenset(family) for family in SPATIAL_FAMILIES for word in family
}
SPATIAL_EN: frozenset[str] = frozenset(_FAMILY_OF)


@dataclass(frozen=True)
class SpatialIntent:
    """A relação que o jogador nomeou, já expandida para os sinônimos do jogo.

    Vazio é o caso normal: a maioria dos enunciados não fala de lugar nenhum, e aí o termo
    lexical não roda."""

    wanted: frozenset[str] = frozenset()

    @property
    def is_empty(self) -> bool:
        return not self.wanted


EMPTY = SpatialIntent()


def words(text: str) -> list[str]:
    return _WORDS.findall(text.lower())


def spatial_words(text: str) -> frozenset[str]:
    """Relações que aparecem numa entrada da gramática, comparando por PALAVRA.

    Comparar por substring seria armadilha silenciosa: ``"in" in "look in the invitation"`` é
    verdadeiro, mas também é verdadeiro para ``Invitation`` e ``Read the invitation``, e as
    três passariam a concordar com "dentro"."""
    return frozenset(word for word in words(text) if word in SPATIAL_EN)


def spatial_intent(texts: Sequence[str], table: Mapping[str, str]) -> SpatialIntent:
    """Lê a relação de lugar do enunciado do jogador.

    ``table`` é o ``spatial`` do pacote de idioma (relação falada → relação canônica). Casa a
    expressão mais longa primeiro, porque "do outro lado" e "em cima" só significam o que
    significam inteiras. Também recolhe relação já dita em inglês, que é o caso de quem joga
    em inglês e o de STT que devolve a palavra original."""
    found: set[str] = set()
    longest = max((len(key.split()) for key in table), default=0)
    for text in texts:
        tokens = words(text)
        for span in range(longest, 0, -1):
            for start in range(0, len(tokens) - span + 1):
                canonical = table.get(" ".join(tokens[start : start + span]))
                if canonical is not None:
                    found |= _FAMILY_OF.get(canonical, frozenset({canonical}))
        found |= {token for token in tokens if token in SPATIAL_EN}
    return SpatialIntent(frozenset(found)) if found else EMPTY
