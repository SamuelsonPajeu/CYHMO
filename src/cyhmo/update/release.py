"""Última release publicada no GitHub e download do pacote portátil.

Só leitura da API pública: nada da máquina do usuário é enviado. O httpx entra em
import tardio porque este módulo é alcançado pelo boot.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from cyhmo.domain.errors import UpdateError
from cyhmo.update.version import Version

ASSET_NAME = "CYHMO_portable.zip"
LATEST_URL = "https://api.github.com/repos/{repository}/releases/latest"
API_HEADERS = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
CHECK_TIMEOUT_S = 10.0
DOWNLOAD_TIMEOUT_S = 600.0
MAX_ASSET_BYTES = 256 * 1024 * 1024
CHUNK_BYTES = 1 << 16
NOTES_LIMIT = 1200

Progress = Callable[[int, int], None]


@dataclass(frozen=True)
class Release:
    version: str
    tag: str
    page_url: str
    notes: str
    asset_url: str
    asset_bytes: int


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
    asset = _find_asset(payload.get("assets"))
    if asset is None:
        raise UpdateError(
            f"a release {tag} não publicou {ASSET_NAME}; baixe o pacote à mão em {page_url or repository}"
        )
    return Release(
        version=str(version),
        tag=tag,
        page_url=page_url,
        notes=_trim(str(payload.get("body") or "")),
        asset_url=str(asset["browser_download_url"]),
        asset_bytes=int(asset.get("size") or 0),
    )


def download(release: Release, target: Path, progress: Progress | None = None, transport: Any = None) -> Path:
    """Grava num ``.part`` e só então renomeia: interromper no meio deixaria um zip
    truncado que passaria por completo na execução seguinte."""
    import httpx

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    partial.unlink(missing_ok=True)
    try:
        with httpx.Client(timeout=DOWNLOAD_TIMEOUT_S, follow_redirects=True, transport=transport) as client:
            with client.stream("GET", release.asset_url) as response:
                response.raise_for_status()
                total = _expected_bytes(response.headers.get("Content-Length"), release.asset_bytes)
                with partial.open("wb") as sink:
                    _pump(response, sink, total, progress)
    except httpx.HTTPError as exc:
        partial.unlink(missing_ok=True)
        raise UpdateError(f"falhou ao baixar {ASSET_NAME} da versão {release.version}: {exc}") from exc
    except OSError as exc:
        partial.unlink(missing_ok=True)
        raise UpdateError(f"falhou ao gravar {partial}: {exc}") from exc
    partial.replace(target)
    return target


def _pump(response: Any, sink: Any, total: int, progress: Progress | None) -> None:
    written = 0
    for chunk in response.iter_bytes(CHUNK_BYTES):
        written += len(chunk)
        if written > MAX_ASSET_BYTES:
            raise UpdateError(f"o download passou de {MAX_ASSET_BYTES // (1024 * 1024)} MB; pacote inesperado")
        sink.write(chunk)
        if progress is not None:
            progress(written, total)


def _expected_bytes(header: str | None, announced: int) -> int:
    declared = int(header) if header and header.isdigit() else announced
    if declared > MAX_ASSET_BYTES:
        raise UpdateError(f"o pacote anunciado tem {declared // (1024 * 1024)} MB; grande demais para ser o portátil")
    return declared


def _find_asset(assets: Any) -> dict[str, Any] | None:
    if not isinstance(assets, list):
        return None
    for asset in assets:
        if isinstance(asset, dict) and asset.get("name") == ASSET_NAME and asset.get("browser_download_url"):
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


def _status_hint(url: str, status: int) -> str:
    if status == 404:
        return (
            f"{url} respondeu 404: o repositório não tem release pública — ele pode estar privado, "
            "ainda sem release, ou update.repository está errado no config.toml"
        )
    if status == 403:
        return f"{url} respondeu 403: o limite de consultas do GitHub foi atingido; tente de novo mais tarde"
    return f"{url} respondeu {status}; confira update.repository no config.toml"
