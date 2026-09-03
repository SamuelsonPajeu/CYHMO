"""Âncora de decodificação do Whisper.

O ``initial_prompt`` é contexto de **continuação**, não lista de vocabulário: o modelo
escreve o que continua a âncora. Daí duas regras, medidas em áudio real:

* um termo por conceito, para o orçamento comprar cobertura e não sinônimos;
* ordem da palavra mais curta para a mais longa, porque a cauda da âncora contamina a
  primeira palavra transcrita e palavra curta no fim atrai qualquer fragmento acústico.

E daí ``SceneHotwords``: a âncora segue a gramática viva. Uma lista fixa leva
substantivos da cena errada para dentro da cena atual — foi o que fez "olho direito"
virar "olhe direito" no meio de uma batalha.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Sequence

from cyhmo.domain.language_pack import LanguagePack

_SEPARATORS = re.compile(r"[^0-9a-zà-ÿ]+")


def pack_hotwords(pack: LanguagePack | None, limit: int) -> tuple[str, ...]:
    """Âncora de partida, usada enquanto nenhuma gramática foi lida."""
    if pack is None or limit <= 0:
        return ()
    words = _one_per_concept(pack.lexicon, pack.body_parts, pack.directions)
    words += pack.affirmations + pack.negations + pack.target_words
    words += [phrase for phrases in pack.command_examples.values() for phrase in phrases]
    single = [word.strip() for word in words if word and " " not in word.strip()]
    return anchor_order(single, limit)


@dataclass(frozen=True)
class SceneHotwords:
    """Para cada conceito presente na gramática viva, as palavras que o pacote do
    jogador já mapeia para ele. Cena nunca mapeada rende menos palavras — nunca
    palavras erradas."""

    pack: LanguagePack | None
    limit: int
    fallback: tuple[str, ...] = ()

    def for_grammar(self, grammar: Sequence[str]) -> tuple[str, ...]:
        if self.pack is None or self.limit <= 0:
            return ()
        scene = tuple(concept_words(entry) for entry in grammar)

        def in_scene(concept: str) -> bool:
            return _concept_in_scene(concept_words(concept), scene)

        phrases = _one_per_concept(self.pack.lexicon, self.pack.body_parts, self.pack.directions, keep=in_scene)
        words = [word for phrase in phrases for word in phrase.split()]
        return anchor_order(words, self.limit) or self.fallback


def concept_words(text: str) -> tuple[str, ...]:
    return tuple(part for part in _SEPARATORS.split(text.strip().lower()) if part)


def anchor_order(words: Sequence[str], limit: int) -> tuple[str, ...]:
    unique = dict.fromkeys(word for word in words if word)
    return tuple(sorted(list(unique)[:limit], key=lambda word: (len(word), word)))


def _concept_in_scene(concept: tuple[str, ...], scene: tuple[tuple[str, ...], ...]) -> bool:
    """Casa por palavra inteira: `shoot` alcança `shoot the enemy`, `eye` não alcança `eyeglass`."""
    if not concept:
        return False
    size = len(concept)
    return any(
        any(entry[start : start + size] == concept for start in range(len(entry) - size + 1))
        for entry in scene
    )


def _one_per_concept(*tables: dict[str, str], keep: Callable[[str], bool] | None = None) -> list[str]:
    """A primeira grafia cadastrada para o conceito ganha a vaga — é o que mantém a âncora chinesa numa escrita só em vez
    de gastar metade do orçamento repetindo cada palavra em simplificado e tradicional."""
    chosen: dict[str, str] = {}
    for table in tables:
        for word, concept in table.items():
            if keep is None or keep(concept):
                chosen.setdefault(concept.strip().lower(), word)
    return list(chosen.values())
