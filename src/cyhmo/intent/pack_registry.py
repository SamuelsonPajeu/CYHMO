"""Índice remoto de pacotes de idioma: listar e instalar sem sair da interface.

O arquivo baixado é validado contra ``LanguagePack`` antes de tocar o disco — um
pacote quebrado no repositório não pode derrubar a inicialização do mod.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit

import yaml
from pydantic import ValidationError

from cyhmo.domain.errors import LanguagePackError
from cyhmo.domain.language_pack import LanguagePack
from cyhmo.intent.language_packs import PACK_SUFFIX

DEFAULT_TIMEOUT_S = 8.0
MAX_PACK_BYTES = 512 * 1024


@dataclass(frozen=True)
class RegistryEntry:
    code: str
    name: str
    file: str
    description: str = ""

    def to_dict(self, installed: bool) -> dict[str, Any]:
        return {
            "code": self.code,
            "name": self.name,
            "description": self.description,
            "installed": installed,
        }


def fetch_registry(url: str, timeout_s: float = DEFAULT_TIMEOUT_S, transport: Any = None) -> tuple[RegistryEntry, ...]:
    payload = _get_json(url, timeout_s, transport)
    items = payload.get("languages") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        raise LanguagePackError(f"{url}: o índice de idiomas deve ser uma lista de pacotes")
    return tuple(_parse_entry(item, url) for item in items)


def install_pack(
    registry_url: str,
    entry: RegistryEntry,
    packs_dir: Path,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    transport: Any = None,
) -> Path:
    text = _get_text(_pack_url(registry_url, entry), timeout_s, transport)
    pack = _validate(text, entry.code)
    destination = Path(packs_dir) / f"{pack.code}{PACK_SUFFIX}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")
    return destination


def _pack_url(registry_url: str, entry: RegistryEntry) -> str:
    """O índice só pode apontar para arquivos do próprio host: um `file` com host
    arbitrário transformaria o índice num vetor de download de qualquer coisa."""
    target = urljoin(registry_url, entry.file)
    if urlsplit(target).netloc != urlsplit(registry_url).netloc:
        raise LanguagePackError(
            f"o índice tentou apontar {entry.code!r} para outro servidor ({target}); download recusado"
        )
    return target


def _validate(text: str, expected_code: str) -> LanguagePack:
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise LanguagePackError(f"pacote {expected_code!r} não é um YAML válido — {exc}") from exc
    if not isinstance(raw, dict):
        raise LanguagePackError(f"pacote {expected_code!r} deveria ser um mapeamento YAML")
    try:
        pack = LanguagePack.model_validate(raw)
    except ValidationError as exc:
        issues = "; ".join(".".join(str(part) for part in issue["loc"]) + ": " + issue["msg"] for issue in exc.errors())
        raise LanguagePackError(f"pacote {expected_code!r} inválido — {issues}") from exc
    if pack.code != expected_code:
        raise LanguagePackError(f"o pacote baixado diz ser {pack.code!r}, mas o índice o anunciou como {expected_code!r}")
    return pack


def _parse_entry(item: Any, url: str) -> RegistryEntry:
    if not isinstance(item, dict):
        raise LanguagePackError(f"{url}: entrada do índice não é um objeto: {item!r}")
    code = str(item.get("code", "")).strip()
    if not code:
        raise LanguagePackError(f"{url}: entrada do índice sem 'code': {item!r}")
    return RegistryEntry(
        code=code,
        name=str(item.get("name", code)),
        file=str(item.get("file", f"{code}.yaml")),
        description=str(item.get("description", "")),
    )


def _get_json(url: str, timeout_s: float, transport: Any) -> Any:
    import json

    try:
        return json.loads(_get_text(url, timeout_s, transport))
    except json.JSONDecodeError as exc:
        raise LanguagePackError(f"{url}: o índice de idiomas não é um JSON válido — {exc}") from exc


def _get_text(url: str, timeout_s: float, transport: Any) -> str:
    import httpx

    try:
        with httpx.Client(timeout=timeout_s, follow_redirects=True, transport=transport) as client:
            response = client.get(url)
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise LanguagePackError(
            f"{url} respondeu {exc.response.status_code}; confira languages.registry_url"
        ) from exc
    except httpx.HTTPError as exc:
        raise LanguagePackError(f"erro ao baixar {url}: {exc}") from exc
    if len(response.content) > MAX_PACK_BYTES:
        raise LanguagePackError(f"{url}: resposta muito grande")
    return response.text
