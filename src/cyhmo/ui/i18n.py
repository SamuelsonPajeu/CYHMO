"""Textos da interface: dados em ``locales/*.json``, nunca embutidos na View.

O idioma efetivo sai de ``ui.language``. Em ``auto`` ele segue o pacote de idioma
primário, para que quem fala português não precise configurar nada duas vezes; quando não
existe locale para esse pacote, o idioma do dispositivo decide; e quando nem isso casa,
o inglês. Um jogador que fala chinês com o jogo num Windows em português lê a interface
em português, em vez de cair no inglês por não haver locale zh-CN.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

from cyhmo.config.locale import system_language
from cyhmo.domain.errors import UiServerError

LOCALES_DIR = Path(__file__).resolve().parent / "locales"
LOCALE_SUFFIX = ".json"
FALLBACK_LANGUAGE = "en"


def available_languages() -> tuple[str, ...]:
    return tuple(sorted(path.stem for path in LOCALES_DIR.glob(f"*{LOCALE_SUFFIX}")))


def resolve_language(ui_language: str, primary_pack: str) -> str:
    known = available_languages()
    if ui_language != "auto":
        return ui_language if ui_language in known else FALLBACK_LANGUAGE
    for candidate in (primary_pack, system_language()):
        matched = _closest(candidate, known)
        if matched is not None:
            return matched
    return FALLBACK_LANGUAGE


def _closest(wanted: str, known: Sequence[str]) -> str | None:
    """Casa ``pt-BR`` exato e depois a base ``pt``; ``None`` quando nada serve."""
    wanted = (wanted or "").strip().replace("_", "-")
    if not wanted:
        return None
    if wanted in known:
        return wanted
    prefix = wanted.split("-", 1)[0].casefold()
    for code in known:
        if code.split("-", 1)[0].casefold() == prefix:
            return code
    return None


@lru_cache(maxsize=8)
def load_locale(code: str) -> dict[str, Any]:
    path = LOCALES_DIR / f"{code}{LOCALE_SUFFIX}"
    if not path.is_file():
        raise UiServerError(f"idioma de interface {code!r} não encontrado em {LOCALES_DIR}")
    try:
        strings = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise UiServerError(f"{path}: arquivo de textos inválido — {exc}") from exc
    if not isinstance(strings, dict):
        raise UiServerError(f"{path}: os textos devem ser um objeto JSON")
    return strings


def language_catalog() -> list[dict[str, str]]:
    """Código e nome de cada idioma de interface, para o seletor da View."""
    catalog: list[dict[str, str]] = []
    for code in available_languages():
        meta = load_locale(code).get("meta", {})
        catalog.append({"code": code, "name": str(meta.get("name", code))})
    return catalog


def bundle(ui_language: str, primary_pack: str) -> dict[str, Any]:
    language = resolve_language(ui_language, primary_pack)
    return {
        "language": language,
        "requested": ui_language,
        "available": language_catalog(),
        "strings": load_locale(language),
    }


def flatten(strings: dict[str, Any], prefix: str = "") -> dict[str, str]:
    """Chaves pontilhadas de um bundle — a View resolve por ``t('tabs.panel')``."""
    flat: dict[str, str] = {}
    for key, value in strings.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, f"{dotted}."))
        else:
            flat[dotted] = str(value)
    return flat
