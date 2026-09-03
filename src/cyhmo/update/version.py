"""Versão instalada e comparação entre releases.

Só biblioteca padrão: isto é lido no boot, antes de qualquer dependência pesada.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

VERSION_FILE = "VERSION"
UNKNOWN = "0.0.0"
_SEMVER = re.compile(r"v?(\d+)\.(\d+)\.(\d+)")


@dataclass(frozen=True, order=True)
class Version:
    major: int = 0
    minor: int = 0
    patch: int = 0

    @classmethod
    def parse(cls, text: str) -> "Version | None":
        match = _SEMVER.match(text.strip())
        return None if match is None else cls(*(int(part) for part in match.groups()))

    @property
    def released(self) -> bool:
        """0.0.0 é o que um clone de desenvolvimento reporta: não veio de release nenhuma."""
        return self != Version()

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


def current_version(base_dir: Path) -> str:
    """O pacote portátil traz o ``VERSION`` carimbado pela release; a instalação por pip
    responde pelos metadados. Sem os dois é clone de desenvolvimento, e 0.0.0 é a resposta
    honesta — o mod não vai se oferecer para trocar o código de quem o está escrevendo."""
    stamped = _read_stamp(base_dir / VERSION_FILE)
    if stamped is not None:
        return str(stamped)
    installed = Version.parse(package_version())
    return str(installed) if installed is not None else UNKNOWN


def package_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("cyhmo")
    except PackageNotFoundError:
        return UNKNOWN


def is_newer(candidate: str, installed: str) -> bool:
    left, right = Version.parse(candidate), Version.parse(installed)
    return left is not None and right is not None and left > right


def _read_stamp(path: Path) -> Version | None:
    try:
        return Version.parse(path.read_text(encoding="utf-8-sig"))
    except OSError:
        return None
