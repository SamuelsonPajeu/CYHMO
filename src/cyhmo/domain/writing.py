"""Escrita contínua: a que não separa palavras por espaço (chinês, japonês).

Mora no domínio porque as camadas 1 e 2 precisam da mesma resposta e não podem
depender uma da outra — o STT para decidir a unidade de contagem da guarda de
degeneração, a interpretação para achar fronteira de palavra.
"""

from __future__ import annotations

import re

_CONTINUOUS = re.compile(
    "[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff66-\uff9d]"
)


def is_continuous(char: str) -> bool:
    return bool(_CONTINUOUS.match(char))


def is_mostly_continuous(text: str) -> bool:
    written = [char for char in text if not char.isspace()]
    if not written:
        return False
    return sum(is_continuous(char) for char in written) * 2 >= len(written)
