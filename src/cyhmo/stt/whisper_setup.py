"""Provisiona o backend padrão de transcrição: o binário oficial do whisper.cpp e o peso ggml.

Nada é compilado. O projeto whisper.cpp publica um build de Windows x64 pronto, e é dele
que saem o ``whisper-server.exe`` e as DLLs do ggml. Só a CPU: o build com GPU exige CUDA,
e a medição do projeto mostrou a CPU ganhando do Vulkan nesta classe de máquina.

Usa apenas a biblioteca padrão de propósito — este módulo roda no instalador, antes de as
dependências do mod estarem garantidas.
"""

from __future__ import annotations

import shutil
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from cyhmo.config.schema import AppConfig, ProjectPaths
from cyhmo.domain.errors import CyhmoError

WHISPER_RELEASE = "b4938"
WHISPER_ASSET = "whisper-bin-x64.zip"
WHISPER_URL = f"https://github.com/ggml-org/whisper.cpp/releases/download/{WHISPER_RELEASE}/{WHISPER_ASSET}"
MODEL_BASE_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"
ARCHIVE_PREFIX = "Release/"
CHUNK_BYTES = 1 << 20
MODEL_MINIMUM_BYTES = 20 * 1024 * 1024
DOWNLOAD_TIMEOUT_S = 60.0

Reporter = Callable[[str], None]


class WhisperSetupError(CyhmoError):
    """Não foi possível baixar ou instalar o backend whisper.cpp."""


@dataclass(frozen=True)
class SetupReport:
    binary: Path
    model: Path
    actions: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return self.binary.is_file() and self.model.is_file()


def wanted_from_archive(name: str) -> bool:
    """Só o servidor e as bibliotecas que ele carrega: o zip oficial traz dezenas de
    exemplos e ferramentas que triplicariam a pasta sem servir para nada aqui."""
    if not name.startswith(ARCHIVE_PREFIX) or name.endswith("/"):
        return False
    leaf = name[len(ARCHIVE_PREFIX) :]
    if "/" in leaf:
        return False
    return leaf == "whisper-server.exe" or leaf == "whisper.dll" or leaf.startswith("ggml")


def ensure_backend(
    config: AppConfig,
    paths: ProjectPaths,
    skip_model: bool = False,
    force: bool = False,
    report: Reporter | None = None,
) -> SetupReport:
    settings = config.stt.whisper_cpp
    binary = (paths.base_dir / settings.binary).resolve()
    model = (paths.base_dir / settings.model).resolve()
    say = report or (lambda _message: None)
    actions: list[str] = []

    if force or not binary.is_file():
        _install_binary(binary, say)
        actions.append(f"whisper.cpp {WHISPER_RELEASE} instalado em {binary.parent}")
    else:
        say(f"whisper.cpp já está em {binary}")

    if skip_model:
        say("modelo ignorado (--skip-model)")
    elif force or not model.is_file():
        _download_model(model, say)
        actions.append(f"modelo {model.name} baixado")
    else:
        say(f"modelo já está em {model}")

    return SetupReport(binary=binary, model=model, actions=tuple(actions))


def _install_binary(binary: Path, say: Reporter) -> None:
    destination = binary.parent
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / WHISPER_ASSET
    say(f"baixando {WHISPER_ASSET} ({WHISPER_RELEASE})...")
    _download(WHISPER_URL, archive, say)
    try:
        _extract(archive, destination, say)
    finally:
        archive.unlink(missing_ok=True)
    if not binary.is_file():
        raise WhisperSetupError(
            f"{binary.name} não apareceu em {destination} depois de extrair {WHISPER_ASSET}. "
            "O conteúdo do release do whisper.cpp mudou; baixe-o à mão e ajuste "
            "stt.whisper_cpp.binary no config.toml."
        )


def _extract(archive: Path, destination: Path, say: Reporter) -> None:
    try:
        with zipfile.ZipFile(archive) as bundle:
            wanted = [name for name in bundle.namelist() if wanted_from_archive(name)]
            if not wanted:
                raise WhisperSetupError(
                    f"{WHISPER_ASSET} não traz {ARCHIVE_PREFIX}whisper-server.exe; "
                    "o formato do release mudou."
                )
            for name in wanted:
                target = destination / Path(name).name
                with bundle.open(name) as source, target.open("wb") as sink:
                    shutil.copyfileobj(source, sink)
        say(f"{len(wanted)} arquivos extraídos para {destination}")
    except zipfile.BadZipFile as exc:
        raise WhisperSetupError(f"{archive} não é um zip válido ({exc}); apague-o e rode de novo") from exc


def _download_model(model: Path, say: Reporter) -> None:
    if not model.name.startswith("ggml-") or model.suffix != ".bin":
        raise WhisperSetupError(
            f"não sei de onde baixar {model.name}: stt.whisper_cpp.model deve apontar para um "
            "arquivo ggml-*.bin. Baixe-o à mão de huggingface.co/ggerganov/whisper.cpp."
        )
    model.parent.mkdir(parents=True, exist_ok=True)
    say(f"baixando {model.name} (algumas centenas de MB, uma vez só)...")
    _download(f"{MODEL_BASE_URL}/{model.name}", model, say)
    if model.stat().st_size < MODEL_MINIMUM_BYTES:
        model.unlink(missing_ok=True)
        raise WhisperSetupError(
            f"o download de {model.name} veio pequeno demais para ser um modelo; "
            "confira a conexão e rode de novo"
        )


def _download(url: str, target: Path, say: Reporter) -> None:
    """Grava num arquivo temporário e só então renomeia: interromper no meio deixaria um
    arquivo truncado que passaria por completo em toda execução seguinte."""
    partial = target.with_suffix(target.suffix + ".part")
    partial.unlink(missing_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_S) as response:
            total = int(response.headers.get("Content-Length") or 0)
            with partial.open("wb") as sink:
                for written in _pump(response, sink):
                    _announce(say, written, total)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        partial.unlink(missing_ok=True)
        raise WhisperSetupError(f"falhou ao baixar {url}: {exc}") from exc
    partial.replace(target)


def _pump(response: object, sink: object) -> Iterable[int]:
    written = 0
    step = 0
    while True:
        chunk = response.read(CHUNK_BYTES)  # type: ignore[attr-defined]
        if not chunk:
            break
        sink.write(chunk)  # type: ignore[attr-defined]
        written += len(chunk)
        step += 1
        if step % 25 == 0:
            yield written
    yield written


def _announce(say: Reporter, written: int, total: int) -> None:
    megabytes = written / (1024 * 1024)
    if total:
        say(f"  {megabytes:6.0f} MB de {total / (1024 * 1024):.0f} MB ({written * 100 // total}%)")
    else:
        say(f"  {megabytes:6.0f} MB")
