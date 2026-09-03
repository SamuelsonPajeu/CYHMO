"""Preparação e troca dos arquivos da nova versão.

A troca acontece com o mod já encerrado — o processo seguinte é que carrega o código
novo. O estado do usuário (config.toml, dados, modelos, venv) nunca é tocado.
"""

from __future__ import annotations

import json
import logging
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

from cyhmo.domain.errors import UpdateError
from cyhmo.update.version import VERSION_FILE

log = logging.getLogger("cyhmo.update")

UPDATE_DIRNAME = "update"
PENDING_NAME = "pending.json"
BACKUP_DIRNAME = "backup"
STAGING_PREFIX = "staging-"
ARCHIVE_NAME = "CYHMO_portable.zip"
REINSTALL_FLAG = ".cyhmo-reinstall"
REQUIREMENTS = "requirements.txt"
REQUIRED_ENTRIES = ("src", "pyproject.toml", "CYHMO.cmd")
USER_STATE = frozenset({"config.toml", "data", "models", "models.lock", "whisper_cpp", "cheats", ".venv"})


@dataclass(frozen=True)
class PendingUpdate:
    version: str
    staging: Path


def update_dir(data_dir: Path) -> Path:
    return data_dir / UPDATE_DIRNAME


def archive_path(data_dir: Path) -> Path:
    return update_dir(data_dir) / ARCHIVE_NAME


def stage(archive: Path, version: str, data_dir: Path) -> PendingUpdate:
    """Extrai o pacote e deixa a troca pronta para o encerramento aplicar."""
    staging = update_dir(data_dir) / f"{STAGING_PREFIX}{version}"
    _reset(staging)
    _extract(archive, staging)
    missing = [entry for entry in REQUIRED_ENTRIES if not (staging / entry).exists()]
    if missing:
        _reset(staging)
        raise UpdateError(
            f"o pacote da versão {version} não traz {', '.join(missing)}; "
            "o formato da release mudou e a troca foi cancelada"
        )
    (staging / VERSION_FILE).write_text(f"{version}\n", encoding="utf-8")
    archive.unlink(missing_ok=True)
    _write_pending(data_dir, version, staging.name)
    log.info("versão %s preparada em %s", version, staging)
    return PendingUpdate(version=version, staging=staging)


def read_pending(data_dir: Path) -> PendingUpdate | None:
    path = update_dir(data_dir) / PENDING_NAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    version, name = str(payload.get("version", "")), str(payload.get("staging", ""))
    staging = update_dir(data_dir) / name
    if not version or not name or not staging.is_dir():
        return None
    return PendingUpdate(version=version, staging=staging)


def apply_pending(base_dir: Path, data_dir: Path) -> str:
    """Troca os arquivos preparados pelos atuais e devolve a versão que passou a valer.

    Cada item vai para o backup antes de o novo entrar: uma falha no meio restaura o que
    já havia sido trocado, em vez de deixar meia instalação."""
    pending = read_pending(data_dir)
    if pending is None:
        raise UpdateError("nenhuma atualização preparada para aplicar")
    backup = update_dir(data_dir) / BACKUP_DIRNAME
    _reset(backup)
    swapped: list[str] = []
    try:
        for source in sorted(pending.staging.iterdir()):
            swapped.append(source.name)
            _swap(source, base_dir / source.name, backup / source.name)
    except OSError as exc:
        _rollback(swapped, base_dir, backup)
        raise UpdateError(
            f"a troca dos arquivos da versão {pending.version} falhou ({exc}); "
            "a instalação anterior foi restaurada e nada mudou"
        ) from exc
    _mark_reinstall(base_dir, backup)
    _discard(data_dir)
    log.info("versão %s aplicada em %s", pending.version, base_dir)
    return pending.version


def discard_pending(data_dir: Path) -> None:
    _discard(data_dir)


def _swap(source: Path, target: Path, saved: Path) -> None:
    if target.exists():
        shutil.move(str(target), str(saved))
    shutil.move(str(source), str(target))


def _rollback(swapped: list[str], base_dir: Path, backup: Path) -> None:
    for name in reversed(swapped):
        target, saved = base_dir / name, backup / name
        _remove(target)
        if saved.exists():
            shutil.move(str(saved), str(target))


def _mark_reinstall(base_dir: Path, backup: Path) -> None:
    """Dependência nova só entra no venv rodando o instalador outra vez; o lançador lê
    esta marca antes de reabrir o mod."""
    flag = base_dir / REINSTALL_FLAG
    if _same_bytes(backup / REQUIREMENTS, base_dir / REQUIREMENTS):
        flag.unlink(missing_ok=True)
        return
    flag.write_text("1\n", encoding="utf-8")


def _same_bytes(left: Path, right: Path) -> bool:
    try:
        return left.read_bytes() == right.read_bytes()
    except OSError:
        return False


def _extract(archive: Path, staging: Path) -> None:
    try:
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(staging, [name for name in bundle.namelist() if _accepts(name)])
    except zipfile.BadZipFile as exc:
        raise UpdateError(f"o pacote baixado não é um zip válido ({exc}); tente atualizar de novo") from exc
    except OSError as exc:
        raise UpdateError(f"falhou ao extrair o pacote em {staging}: {exc}") from exc


def _accepts(name: str) -> bool:
    """Um zip pode carregar caminho absoluto ou ``..`` e escrever fora da pasta preparada,
    e release nenhuma tem por que trazer o estado do usuário junto."""
    normalized = name.replace("\\", "/")
    parts = [part for part in normalized.split("/") if part]
    if not parts or normalized.startswith("/") or ":" in normalized or ".." in parts:
        return False
    return parts[0] not in USER_STATE


def _write_pending(data_dir: Path, version: str, staging: str) -> None:
    path = update_dir(data_dir) / PENDING_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": version, "staging": staging}), encoding="utf-8")


def _discard(data_dir: Path) -> None:
    root = update_dir(data_dir)
    (root / PENDING_NAME).unlink(missing_ok=True)
    for leftover in (*root.glob(f"{STAGING_PREFIX}*"), *root.glob(f"{ARCHIVE_NAME}*")):
        _remove(leftover)
    _remove(root / BACKUP_DIRNAME)


def _reset(directory: Path) -> None:
    _remove(directory)
    directory.mkdir(parents=True, exist_ok=True)


def _remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)
