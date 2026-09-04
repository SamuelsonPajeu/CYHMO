"""Idioma do dispositivo, usado só quando ainda não há escolha do usuário.

O ``config.toml`` não vem no pacote: ele nasce na primeira execução. É esse o único
momento em que ninguém escolheu nada, e é onde perguntar ao sistema faz sentido — depois
disso o arquivo é a escolha, e o mod não passa por cima dela.

O casamento é em dois passos e sempre cai em pé: ``pt-BR`` exato primeiro, ``pt`` como
base depois, inglês quando nada casa. Um Windows em finlandês abre o mod em inglês em vez
de abrir num idioma que ninguém pediu.

Só biblioteca padrão: isto roda antes de qualquer dependência estar garantida.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Sequence

log = logging.getLogger("cyhmo.config.locale")

FALLBACK_LANGUAGE = "en"
PACK_SUFFIX = ".yaml"
LOCALE_NAME_MAX_LENGTH = 85


def system_language() -> str:
    """Idioma do sistema no formato ``pt-BR``, ou ``""`` quando não dá para saber."""
    if sys.platform == "win32":
        windows = _windows_locale_name()
        if windows:
            return windows
    return _environment_locale_name()


def available_packs(packs_dir: Path | str) -> tuple[str, ...]:
    """Códigos dos pacotes de idioma presentes, pelo nome do arquivo."""
    try:
        return tuple(sorted(path.stem for path in Path(packs_dir).glob(f"*{PACK_SUFFIX}")))
    except OSError as exc:
        log.debug("não consegui listar os pacotes em %s: %s", packs_dir, exc)
        return ()


def match_language(wanted: str, available: Sequence[str], fallback: str = FALLBACK_LANGUAGE) -> str:
    """Melhor código de ``available`` para ``wanted``, com o fallback ao fim da fila."""
    if not available:
        return fallback
    wanted = wanted.strip().replace("_", "-")
    if wanted:
        exact = {code.casefold(): code for code in available}.get(wanted.casefold())
        if exact is not None:
            return exact
        base = wanted.split("-", 1)[0].casefold()
        for code in available:
            if code.split("-", 1)[0].casefold() == base:
                return code
    return fallback if fallback in available else available[0]


def initial_languages(packs_dir: Path | str) -> tuple[list[str], str] | None:
    """``(enabled, primary)`` para o primeiro arranque, ou ``None`` sem pacote nenhum.

    O inglês entra junto mesmo sem ser o primário porque os comandos do jogo são em
    inglês: o casamento de intenção compara contra eles, e tirá-los da lista custaria
    acerto sem economizar nada."""
    packs = available_packs(packs_dir)
    if not packs:
        return None
    detected = system_language()
    primary = match_language(detected, packs)
    enabled = [primary]
    if primary != FALLBACK_LANGUAGE and FALLBACK_LANGUAGE in packs:
        enabled.append(FALLBACK_LANGUAGE)
    log.info(
        "primeiro arranque: idioma do sistema %r -> pacote primário %r",
        detected or "(desconhecido)",
        primary,
    )
    return enabled, primary


def _windows_locale_name() -> str:
    """``GetUserDefaultLocaleName`` devolve o nome BCP 47 direto (``pt-BR``), sem a
    tradução de LCID que a API antiga exigia."""
    import ctypes

    buffer = ctypes.create_unicode_buffer(LOCALE_NAME_MAX_LENGTH)
    try:
        written = ctypes.windll.kernel32.GetUserDefaultLocaleName(buffer, len(buffer))
    except (AttributeError, OSError) as exc:
        log.debug("GetUserDefaultLocaleName indisponível: %s", exc)
        return ""
    return buffer.value.strip() if written else ""


def _environment_locale_name() -> str:
    for variable in ("LC_ALL", "LC_MESSAGES", "LANG", "LANGUAGE"):
        value = os.environ.get(variable, "").strip()
        if not value or value.upper() in {"C", "POSIX"}:
            continue
        # pt_BR.UTF-8 e pt_BR:en_US chegam assim; só a primeira etiqueta interessa.
        return value.split(":")[0].split(".")[0].replace("_", "-")
    return ""
