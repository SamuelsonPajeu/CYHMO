"""Última release publicada no GitHub e download do pacote portátil.

Só leitura da API pública: da máquina do usuário saem apenas o endereço de origem, o
cabeçalho de agente do httpx e o nome do repositório configurado. O httpx entra em
import tardio porque este módulo é alcançado pelo boot.

O pacote nunca é aceito sem conferência: a release publica o ``.sha256`` ao lado do zip
e o download só vira arquivo definitivo quando o hash calculado aqui bate com ele.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from cyhmo.domain.errors import UpdateError
from cyhmo.update.version import Version

ASSET_NAME = "CYHMO_portable.zip"
DIGEST_SUFFIX = ".sha256"
ASSET_DIGEST_NAME = ASSET_NAME + DIGEST_SUFFIX
LATEST_URL = "https://api.github.com/repos/{repository}/releases/latest"
API_HEADERS = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
CHECK_TIMEOUT_S = 10.0
DOWNLOAD_TIMEOUT_S = 600.0
MAX_ASSET_BYTES = 256 * 1024 * 1024
MAX_DIGEST_BYTES = 4096
CHUNK_BYTES = 1 << 16
NOTES_LIMIT = 1200

_SHA256_HEX = re.compile(r"\b([0-9a-fA-F]{64})\b")

Progress = Callable[[int, int], None]


@dataclass(frozen=True)
class Release:
    version: str
    tag: str
    page_url: str
    notes: str
    asset_url: str
    asset_bytes: int
    digest_url: str = ""


def fetch_latest(repository: str, transport: Any = None) -> Release:
    payload = _get(LATEST_URL.format(repository=repository.strip().strip("/")), transport)
    tag = str(payload.get("tag_name", "")).strip()
    version = Version.parse(tag)
    if version is None:
        raise UpdateError(
            f"a última release de {repository} está marcada como {tag!r}, fora do padrão vX.Y.Z; "
            "sem isso não dá para comparar versões"
        )
    page_url = str(payload.get("html_url", ""))
    assets = payload.get("assets")
    asset = _find_asset(assets, ASSET_NAME)
    if asset is None:
        raise UpdateError(
            f"a release {tag} não publicou {ASSET_NAME}; baixe o pacote à mão em {page_url or repository}"
        )
    digest = _find_asset(assets, ASSET_DIGEST_NAME)
    return Release(
        version=str(version),
        tag=tag,
        page_url=page_url,
        notes=_trim(str(payload.get("body") or "")),
        asset_url=str(asset["browser_download_url"]),
        asset_bytes=int(asset.get("size") or 0),
        digest_url="" if digest is None else str(digest["browser_download_url"]),
    )


def download(release: Release, target: Path, progress: Progress | None = None, transport: Any = None) -> Path:
    """Grava num ``.part`` e só então renomeia: interromper no meio deixaria um zip
    truncado que passaria por completo na execução seguinte.

    O hash esperado é buscado ANTES de qualquer byte do pacote: sem ele publicado não há
    o que conferir, e um pacote não conferido não entra na instalação de ninguém."""
    import httpx

    expected = _expected_digest(release, transport)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    partial.unlink(missing_ok=True)
    digest = hashlib.sha256()
    try:
        with httpx.Client(timeout=DOWNLOAD_TIMEOUT_S, follow_redirects=True, transport=transport) as client:
            with client.stream("GET", release.asset_url) as response:
                response.raise_for_status()
                total = _expected_bytes(response.headers.get("Content-Length"), release.asset_bytes)
                with partial.open("wb") as sink:
                    _pump(response, sink, total, progress, digest)
    except httpx.HTTPError as exc:
        partial.unlink(missing_ok=True)
        raise UpdateError(f"falhou ao baixar {ASSET_NAME} da versão {release.version}: {exc}") from exc
    except OSError as exc:
        partial.unlink(missing_ok=True)
        raise UpdateError(f"falhou ao gravar {partial}: {exc}") from exc
    actual = digest.hexdigest()
    if actual != expected:
        partial.unlink(missing_ok=True)
        raise UpdateError(
            f"o pacote da versão {release.version} não confere com o hash publicado na release "
            f"(esperado {expected[:16]}…, baixado {actual[:16]}…); a troca foi cancelada e nada "
            "foi instalado. Tente de novo; se repetir, baixe o pacote à mão em "
            f"{release.page_url or ASSET_NAME} e confira o sha256 antes de usar"
        )
    partial.replace(target)
    return target


def _expected_digest(release: Release, transport: Any) -> str:
    """O hash sai do arquivo publicado ao lado do pacote, não das notas: notas são texto
    livre que qualquer edição posterior reescreve sem deixar rastro."""
    if not release.digest_url:
        raise UpdateError(
            f"a release {release.tag} não publicou {ASSET_DIGEST_NAME} e o pacote não é instalado "
            "sem conferência de integridade. Atualize à mão pela página da release, conferindo o "
            f"sha256 do {ASSET_NAME} antes de extrair"
        )
    text = _get_text(release.digest_url, transport)
    match = _SHA256_HEX.search(text)
    if match is None:
        raise UpdateError(
            f"{ASSET_DIGEST_NAME} da versão {release.version} não traz um sha256 legível; "
            "a troca foi cancelada"
        )
    return match.group(1).lower()


def _pump(response: Any, sink: Any, total: int, progress: Progress | None, digest: Any) -> None:
    written = 0
    for chunk in response.iter_bytes(CHUNK_BYTES):
        written += len(chunk)
        if written > MAX_ASSET_BYTES:
            raise UpdateError(f"o download passou de {MAX_ASSET_BYTES // (1024 * 1024)} MB; pacote inesperado")
        digest.update(chunk)
        sink.write(chunk)
        if progress is not None:
            progress(written, total)


def _expected_bytes(header: str | None, announced: int) -> int:
    declared = int(header) if header and header.isdigit() else announced
    if declared > MAX_ASSET_BYTES:
        raise UpdateError(f"o pacote anunciado tem {declared // (1024 * 1024)} MB; grande demais para ser o portátil")
    return declared


def _find_asset(assets: Any, name: str) -> dict[str, Any] | None:
    if not isinstance(assets, list):
        return None
    for asset in assets:
        if isinstance(asset, dict) and asset.get("name") == name and asset.get("browser_download_url"):
            return asset
    return None


def _trim(notes: str) -> str:
    clean = notes.replace("\r\n", "\n").strip()
    return clean if len(clean) <= NOTES_LIMIT else clean[:NOTES_LIMIT].rstrip() + "\n..."


def _get(url: str, transport: Any) -> dict[str, Any]:
    import httpx

    try:
        with httpx.Client(timeout=CHECK_TIMEOUT_S, follow_redirects=True, transport=transport) as client:
            response = client.get(url, headers=API_HEADERS)
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPStatusError as exc:
        raise UpdateError(_status_hint(url, exc.response.status_code)) from exc
    except httpx.HTTPError as exc:
        raise UpdateError(f"não consegui consultar {url}: {exc}") from exc
    if not isinstance(payload, dict):
        raise UpdateError(f"{url} respondeu algo que não é uma release")
    return payload


def _get_text(url: str, transport: Any) -> str:
    """Teto de bytes porque o corpo vem de fora: o arquivo de hash tem 100 bytes, e o que
    passar disso não é o arquivo de hash."""
    import httpx

    try:
        with httpx.Client(timeout=CHECK_TIMEOUT_S, follow_redirects=True, transport=transport) as client:
            with client.stream("GET", url) as response:
                response.raise_for_status()
                body = b""
                for chunk in response.iter_bytes(MAX_DIGEST_BYTES):
                    body += chunk
                    if len(body) > MAX_DIGEST_BYTES:
                        raise UpdateError(f"{url} respondeu um corpo grande demais para um sha256")
    except httpx.HTTPError as exc:
        raise UpdateError(f"não consegui baixar {url}: {exc}") from exc
    return body.decode("utf-8", errors="replace")


def _status_hint(url: str, status: int) -> str:
    if status == 404:
        return (
            f"{url} respondeu 404: o repositório não tem release pública — ele pode estar privado, "
            "ainda sem release, ou update.repository está errado no config.toml"
        )
    if status == 403:
        return f"{url} respondeu 403: o limite de consultas do GitHub foi atingido; tente de novo mais tarde"
    return f"{url} respondeu {status}; confira update.repository no config.toml"
