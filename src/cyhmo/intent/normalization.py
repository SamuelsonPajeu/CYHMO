"""Normalização determinística do enunciado; nunca se aplica à key emitida."""

from __future__ import annotations

import re
import unicodedata

_QUOTE_LIKE = str.maketrans({"’": "'", "‘": "'", "`": "'", "´": "'", "–": "-", "—": "-", "_": " "})
_NOT_WORD = re.compile(r"[^\w\s'\-]", re.UNICODE)
_EDGE_MARKS = re.compile(r"(?<!\w)['\-]+|['\-]+(?!\w)", re.UNICODE)
_SPACES = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """NFKC + minúsculas + sem pontuação, preservando hífen e apóstrofo internos ("what's", "e-mail")."""
    text = unicodedata.normalize("NFKC", text).translate(_QUOTE_LIKE).lower()
    text = _NOT_WORD.sub(" ", text)
    text = _EDGE_MARKS.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def normalized_key(text: str) -> str:
    """Forma usada para comparar com a gramática/anexo: hífen vira espaço ("AUTO-FIRE" casa
    "Auto fire") e acento é dobrado ("checa a camera" casa "checa a câmera") — o STT nem sempre
    acentua e o usuário nem sempre digita com acento. Nunca se aplica à key emitida."""
    return _SPACES.sub(" ", _fold_diacritics(normalize_text(text)).replace("-", " ")).strip()


def _fold_diacritics(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    return unicodedata.normalize("NFC", "".join(char for char in decomposed if not unicodedata.combining(char)))
