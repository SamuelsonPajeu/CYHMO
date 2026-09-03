"""Manifesto de integridade dos modelos baixados.

Cobre os arquivos que o mod baixa por conta própria — os pesos ``ggml`` do
backend whisper.cpp. Os modelos vindos do Hugging Face ficam de fora de
propósito: o hub já valida cada blob no download e guarda tudo sob nomes
derivados do próprio hash, então repetir a conferência aqui só criaria um
manifesto que quebra a cada atualização de snapshot.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml

from cyhmo.domain.errors import CyhmoError

LOCK_VERSION = 1
LOCK_NAME = "models.lock"
QUARANTINE_SUFFIX = ".corrupted"
READ_CHUNK = 1 << 20


class ModelsLockError(CyhmoError):
    """Manifesto ausente, malformado ou incompatível."""


@dataclass(frozen=True)
class ModelFile:
    path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class Verdict:
    path: str
    status: str
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(READ_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ModelsLock:
    version: int
    files: tuple[ModelFile, ...]

    @classmethod
    def load(cls, path: Path | str) -> "ModelsLock":
        path = Path(path)
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ModelsLockError(f"manifesto de modelos não encontrado: {path} ({exc})") from exc
        except yaml.YAMLError as exc:
            raise ModelsLockError(f"{path}: YAML inválido — {exc}") from exc
        if not isinstance(data, dict) or "files" not in data:
            raise ModelsLockError(f"{path}: esperado um mapeamento com 'version' e 'files'")
        version = int(data.get("version", 0))
        if version != LOCK_VERSION:
            raise ModelsLockError(f"{path}: versão {version} não suportada (esperada {LOCK_VERSION})")
        return cls(version=version, files=tuple(_parse_file(item, path) for item in data["files"] or ()))

    @classmethod
    def from_directory(cls, models_dir: Path | str, patterns: tuple[str, ...] = ("**/ggml/*.bin",)) -> "ModelsLock":
        models_dir = Path(models_dir)
        found = sorted({match for pattern in patterns for match in models_dir.glob(pattern) if match.is_file()})
        return cls(
            version=LOCK_VERSION,
            files=tuple(
                ModelFile(
                    path=match.relative_to(models_dir).as_posix(),
                    sha256=sha256_of(match),
                    size_bytes=match.stat().st_size,
                )
                for match in found
            ),
        )

    def to_yaml(self) -> str:
        body = {
            "version": self.version,
            "files": [
                {"path": item.path, "sha256": item.sha256, "size_bytes": item.size_bytes} for item in self.files
            ],
        }
        header = (
            f"# {LOCK_NAME} — integridade dos modelos baixados.\n"
            "# Gerado por: cyhmo models --lock\n"
            "# Caminhos são relativos a models_dir. Modelos do Hugging Face não entram aqui:\n"
            "# o hub já verifica cada blob no download.\n"
        )
        return header + yaml.safe_dump(body, sort_keys=False, allow_unicode=True)

    def write(self, path: Path | str) -> Path:
        path = Path(path)
        path.write_text(self.to_yaml(), encoding="utf-8")
        return path

    def verify(self, models_dir: Path | str, quarantine: bool = False) -> tuple[Verdict, ...]:
        models_dir = Path(models_dir)
        return tuple(self._verify_one(models_dir / item.path, item, quarantine) for item in self.files)

    def _verify_one(self, target: Path, expected: ModelFile, quarantine: bool) -> Verdict:
        if not target.is_file():
            return Verdict(expected.path, "ausente", "baixe o modelo de novo (veja o README)")
        size = target.stat().st_size
        if size != expected.size_bytes:
            return self._reject(target, expected, f"tamanho {size} != {expected.size_bytes}", quarantine)
        actual = sha256_of(target)
        if actual != expected.sha256:
            return self._reject(target, expected, f"sha256 {actual[:16]}… != {expected.sha256[:16]}…", quarantine)
        return Verdict(expected.path, "ok")

    def _reject(self, target: Path, expected: ModelFile, detail: str, quarantine: bool) -> Verdict:
        if not quarantine:
            return Verdict(expected.path, "corrompido", detail)
        moved = target.with_suffix(target.suffix + QUARANTINE_SUFFIX)
        try:
            target.replace(moved)
        except OSError as exc:
            return Verdict(expected.path, "corrompido", f"{detail}; não foi possível isolar: {exc}")
        return Verdict(expected.path, "corrompido", f"{detail}; isolado em {moved.name}, baixe de novo")


def _parse_file(item: object, source: Path) -> ModelFile:
    if not isinstance(item, dict):
        raise ModelsLockError(f"{source}: cada item de 'files' deve ser um mapeamento")
    missing = [key for key in ("path", "sha256", "size_bytes") if key not in item]
    if missing:
        raise ModelsLockError(f"{source}: item sem {', '.join(missing)}")
    return ModelFile(str(item["path"]), str(item["sha256"]).lower(), int(item["size_bytes"]))
