"""Normalização do texto do STT: NFKC, minúsculas, sem pontuação, espaços colapsados.

Decisão de projeto: números por extenso NÃO são convertidos em dígitos aqui — a
conversão depende do idioma e pertence ao pacote de idioma da camada 2. Dígitos
já emitidos pelo modelo são preservados; o texto cru fica em ``Transcript.raw_text``.
"""

from __future__ import annotations

import unicodedata

_APOSTROPHES = frozenset({"'", "’", "‘", "ʼ"})
_HYPHENS = frozenset({"-", "‐", "‑"})


def normalize_text(raw: str) -> str:
    text = unicodedata.normalize("NFKC", raw).lower()
    characters = [_keep_or_space(text, position) for position in range(len(text))]
    return " ".join("".join(characters).split())


def _keep_or_space(text: str, position: int) -> str:
    """Hífen e apóstrofo sobrevivem apenas entre alfanuméricos ("number-one", "don't")."""
    char = text[position]
    if char in _APOSTROPHES:
        return "'" if _between_alphanumerics(text, position) else " "
    if char in _HYPHENS:
        return "-" if _between_alphanumerics(text, position) else " "
    return char if char.isalnum() else " "


def _between_alphanumerics(text: str, position: int) -> bool:
    has_neighbors = 0 < position < len(text) - 1
    return has_neighbors and text[position - 1].isalnum() and text[position + 1].isalnum()
