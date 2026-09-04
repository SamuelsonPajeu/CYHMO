"""Suporte a GPU do whisper.cpp: reconhecer a placa e instalar o build que a usa.

O build que ``cyhmo setup`` instala é só CPU. O zip oficial ``whisper-bin-x64`` não traz
backend de GPU nenhum — só ``ggml-cpu-*.dll`` —, então ligar ``use_gpu`` sobre ele apenas
tira o ``-ng`` da linha de comando e a conta continua inteira no processador, sem uma
palavra na tela dizendo isso. É por causa dessa armadilha que a interface só oferece a
opção GPU depois de saber que existe placa capaz e build instalado.

O único build com GPU que o whisper.cpp publica para Windows é o cuBLAS, ou seja, NVIDIA;
não há build Vulkan oficial, e placa AMD ou Intel não tem para onde ir. Entre os dois
cuBLAS publicados na b4938 só o de CUDA 12.4 se sustenta sozinho: o pacote de 11.8 vem sem
``cublas64_11.dll`` e ``cublasLt64_11.dll``, então o ``ggml-cuda.dll`` dele só carrega em
quem já tem o toolkit CUDA instalado à parte — conferido lendo o índice central dos dois
zips. Por isso o catálogo aqui tem um item só.

Só biblioteca padrão: este módulo é alcançado pelo boot.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import sys
import threading
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from cyhmo.domain.errors import CyhmoError
from cyhmo.stt.whisper_setup import ARCHIVE_PREFIX, WHISPER_RELEASE

log = logging.getLogger("cyhmo.stt.gpu")

DEFAULT_GPU_BINARY = "whisper_cpp/cuda/whisper-server.exe"
SERVER_EXE = "whisper-server.exe"
BACKEND_DLL = "ggml-cuda.dll"

BUILD_NAME = "cuBLAS (CUDA 12.4)"
BUILD_ASSET = "whisper-cublas-12.4.0-bin-x64.zip"
BUILD_URL = f"https://github.com/ggml-org/whisper.cpp/releases/download/{WHISPER_RELEASE}/{BUILD_ASSET}"
BUILD_BYTES = 671_045_732
BUILD_SHA256 = "c1b17166e1e31a91cc8e9c1f910d3785e3ce757bb2958bf9dce13fdb4880005f"
# Soma do que é extraído (servidor + DLLs), medida no índice do zip.
BUILD_DISK_BYTES = 1_177_862_656

# ``cuDriverGetVersion`` em centenas: 12000 é CUDA 12.0. O build é de 12.4, e a
# compatibilidade de versão menor do CUDA 12 faz ele rodar em qualquer driver da série 12.
MIN_DRIVER = 12_000

CHUNK_BYTES = 1 << 20
DOWNLOAD_TIMEOUT_S = 60.0
PART_SUFFIX = ".part"

CANCELLED = "cancelado"

PHASE_DOWNLOAD = "download"
PHASE_EXTRACT = "extract"

REASON_NO_NVIDIA = "no-nvidia"
REASON_OLD_DRIVER = "old-driver"
REASON_NOT_WINDOWS = "not-windows"


class WhisperGpuError(CyhmoError):
    """Não foi possível instalar, conferir ou remover o build com GPU do whisper.cpp."""


class _Cancelled(Exception):
    """Interrupção pedida pela interface; não é erro para mostrar como falha."""


@dataclass(frozen=True)
class GpuAdapter:
    name: str
    driver: int

    @property
    def driver_label(self) -> str:
        return f"{self.driver // 1000}.{self.driver % 1000 // 10}"


def driver_label(version: int) -> str:
    return f"{version // 1000}.{version % 1000 // 10}"


@lru_cache(maxsize=1)
def nvidia_adapter() -> GpuAdapter | None:
    """Primeira placa NVIDIA visível e a versão de CUDA que o driver dela aceita.

    Pergunta ao ``nvcuda.dll`` do próprio driver em vez de chamar ``nvidia-smi``: não cria
    processo, responde em milissegundos e existe em toda máquina com driver NVIDIA — o
    ``nvidia-smi`` falta em parte das instalações de notebook. Em máquina sem NVIDIA o
    carregamento falha e a resposta é ``None``, que é exatamente o caso a distinguir.

    O resultado é memorizado: nem a placa nem o driver mudam durante a sessão, e a
    interface pergunta isso a cada atualização do painel."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes

        cuda = ctypes.WinDLL("nvcuda.dll")
        if cuda.cuInit(0) != 0:
            return None
        version = ctypes.c_int(0)
        count = ctypes.c_int(0)
        device = ctypes.c_int(0)
        if cuda.cuDriverGetVersion(ctypes.byref(version)) != 0:
            return None
        if cuda.cuDeviceGetCount(ctypes.byref(count)) != 0 or count.value < 1:
            return None
        if cuda.cuDeviceGet(ctypes.byref(device), 0) != 0:
            return None
        name = ctypes.create_string_buffer(128)
        if cuda.cuDeviceGetName(name, len(name), device) != 0:
            return None
    except Exception as exc:  # driver ausente, quebrado ou de uma geração que não conheço
        log.debug("nenhuma placa NVIDIA utilizável: %s", exc)
        return None
    return GpuAdapter(name=name.value.decode("utf-8", "replace").strip(), driver=int(version.value))


def effective_binary(base_dir: Path, binary: str, gpu_binary: str, use_gpu: bool) -> tuple[Path, bool]:
    """O executável que vai subir de fato, e se ele realmente vai usar a GPU.

    Cair de volta na CPU quando o build com GPU não está instalado é deliberado: a
    alternativa é o servidor não subir e o mod trocar para o faster-whisper, três vezes mais
    lento, por causa de uma opção marcada no painel. Função pura de propósito — quem avisa
    no log é quem monta o transcritor, uma vez por sessão, e não o diagnóstico."""
    cpu = (base_dir / binary).resolve()
    if not use_gpu:
        return cpu, False
    gpu = (base_dir / (gpu_binary or DEFAULT_GPU_BINARY)).resolve()
    if gpu.is_file():
        return gpu, True
    return cpu, False


def wanted_from_archive(name: str) -> bool:
    """Do zip oficial só interessam o servidor e as DLLs.

    Todas as DLLs, não uma lista escolhida a dedo: o backend CUDA carrega cudart, cuBLAS e
    cuBLASLt em tempo de execução, e um nome que faltasse apareceria como o servidor morrendo
    no arranque sem dizer qual. O que fica de fora são as dezenas de exemplos e testes do
    zip, que não somam 6 MB de DLL a mais mas somam ruído na pasta."""
    if not name.startswith(ARCHIVE_PREFIX) or name.endswith("/"):
        return False
    leaf = name[len(ARCHIVE_PREFIX) :]
    if "/" in leaf:
        return False
    return leaf == SERVER_EXE or leaf.lower().endswith(".dll")


@dataclass
class InstallProgress:
    phase: str = ""
    active: bool = False
    percent: float = 0.0
    error: str = ""
    done: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "active": self.active,
            "percent": round(self.percent, 1),
            "error": self.error,
            "done": self.done,
        }


class WhisperGpuAdmin:
    """Instala, confere e remove o build com GPU numa pasta só dele.

    Pasta separada da do build de CPU de propósito: sobrescrever o que já funciona para
    ganhar GPU deixaria quem cancelasse no meio sem reconhecimento nenhum, e voltar para a
    CPU passa a ser trocar uma opção em vez de baixar 640 MB de novo."""

    def __init__(self, base_dir: Path, opener: Callable[..., Any] = urllib.request.urlopen) -> None:
        self._base_dir = Path(base_dir).resolve()
        self._opener = opener
        self._lock = threading.Lock()
        self._progress = InstallProgress()
        self._cancel = threading.Event()

    def server_path(self, gpu_binary: str) -> Path:
        return (self._base_dir / (gpu_binary.strip() or DEFAULT_GPU_BINARY)).resolve()

    def directory_for(self, gpu_binary: str) -> Path:
        return self.server_path(gpu_binary).parent

    def installed(self, gpu_binary: str) -> bool:
        """Servidor E backend: o zip do build de CPU também traz um ``whisper-server.exe``,
        e uma pasta com o servidor sem o ``ggml-cuda.dll`` é uma instalação pela metade."""
        server = self.server_path(gpu_binary)
        return server.is_file() and (server.parent / BACKEND_DLL).is_file()

    def support(self) -> tuple[bool, str, GpuAdapter | None]:
        if sys.platform != "win32":
            return False, REASON_NOT_WINDOWS, None
        adapter = nvidia_adapter()
        if adapter is None:
            return False, REASON_NO_NVIDIA, None
        if adapter.driver < MIN_DRIVER:
            return False, REASON_OLD_DRIVER, adapter
        return True, "", adapter

    def status(self, gpu_binary: str) -> dict[str, Any]:
        supported, reason, adapter = self.support()
        with self._lock:
            progress = self._progress.to_dict()
        return {
            "supported": supported,
            "reason": reason,
            "adapter": adapter.name if adapter else "",
            "driver": adapter.driver_label if adapter else "",
            "required_driver": driver_label(MIN_DRIVER),
            "build": BUILD_NAME,
            "installed": self.installed(gpu_binary),
            "directory": str(self.directory_for(gpu_binary)),
            "download_bytes": BUILD_BYTES,
            "disk_bytes": BUILD_DISK_BYTES,
            "install": progress,
        }

    def start_install(self, gpu_binary: str) -> dict[str, Any]:
        supported, reason, adapter = self.support()
        if not supported:
            raise WhisperGpuError(self._unsupported_message(reason, adapter))
        server = self.server_path(gpu_binary)
        self._check_space(server.parent)
        with self._lock:
            if self._progress.active:
                return self._progress.to_dict()
            self._cancel.clear()
            self._progress = InstallProgress(phase=PHASE_DOWNLOAD, active=True)
            snapshot = self._progress.to_dict()
        threading.Thread(
            target=self._install,
            args=(server,),
            name="cyhmo-whisper-gpu-install",
            daemon=True,
        ).start()
        return snapshot

    def cancel_install(self) -> dict[str, Any]:
        self._cancel.set()
        with self._lock:
            return self._progress.to_dict()

    def progress(self) -> dict[str, Any]:
        with self._lock:
            return self._progress.to_dict()

    def remove(self, binary: str, gpu_binary: str) -> dict[str, Any]:
        """Apaga a pasta do build com GPU, e só ela.

        A guarda contra a pasta do build de CPU não é hipotética: basta ``gpu_binary`` ficar
        igual a ``binary`` no config.toml para o pedido de liberar espaço apagar o
        reconhecimento inteiro."""
        with self._lock:
            if self._progress.active:
                raise WhisperGpuError("a instalação do build com GPU ainda está em andamento; cancele-a antes")
        directory = self.directory_for(gpu_binary)
        if directory == (self._base_dir / binary).resolve().parent:
            raise WhisperGpuError(
                f"{directory} é a pasta do build de CPU; ajuste stt.whisper_cpp.gpu_binary "
                "para uma pasta própria antes de remover"
            )
        if not self.installed(gpu_binary):
            raise WhisperGpuError(f"não há build com GPU instalado em {directory}")
        try:
            shutil.rmtree(directory)
        except OSError as exc:
            raise WhisperGpuError(
                f"não consegui remover {directory}: {exc}. Se o reconhecimento acabou de rodar "
                "na GPU, reinicie o mod para soltar os arquivos e tente de novo."
            ) from exc
        log.info("build com GPU removido de %s", directory)
        return {"removed": True, "directory": str(directory)}

    def _unsupported_message(self, reason: str, adapter: GpuAdapter | None) -> str:
        if reason == REASON_OLD_DRIVER and adapter is not None:
            return (
                f"o driver da {adapter.name} entrega CUDA {adapter.driver_label} e o build "
                f"precisa de {driver_label(MIN_DRIVER)} ou mais; atualize o driver da NVIDIA"
            )
        if reason == REASON_NOT_WINDOWS:
            return "o build com GPU do whisper.cpp só é publicado para Windows"
        return (
            "nenhuma placa NVIDIA encontrada. O único build com GPU publicado pelo whisper.cpp "
            "é o cuBLAS, que exige CUDA; para placa AMD ou Intel o reconhecimento fica na CPU."
        )

    def _check_space(self, directory: Path) -> None:
        """O zip e o extraído convivem no disco até o fim da instalação."""
        anchor = directory
        while not anchor.exists() and anchor != anchor.parent:
            anchor = anchor.parent
        needed = BUILD_BYTES + BUILD_DISK_BYTES
        try:
            free = shutil.disk_usage(anchor).free
        except OSError:
            return
        if free < needed:
            raise WhisperGpuError(
                f"faltam {(needed - free) / (1024 ** 3):.1f} GB em {anchor} para instalar o build "
                f"com GPU: ele baixa {BUILD_BYTES / (1024 ** 3):.1f} GB e ocupa "
                f"{BUILD_DISK_BYTES / (1024 ** 3):.1f} GB depois de extraído"
            )

    def _install(self, server: Path) -> None:
        directory = server.parent
        archive = directory / (BUILD_ASSET + PART_SUFFIX)
        try:
            directory.mkdir(parents=True, exist_ok=True)
            archive.unlink(missing_ok=True)
            self._download(archive)
            self._extract(archive, server)
        except _Cancelled:
            self._finish(error=CANCELLED)
            return
        except (WhisperGpuError, urllib.error.URLError, zipfile.BadZipFile, OSError, TimeoutError) as exc:
            log.warning("instalação do build com GPU falhou: %s", exc)
            self._finish(error=str(exc))
            return
        finally:
            archive.unlink(missing_ok=True)
        log.info("build com GPU instalado em %s", directory)
        self._finish()

    def _download(self, archive: Path) -> None:
        digest = hashlib.sha256()
        written = 0
        with self._opener(BUILD_URL, timeout=DOWNLOAD_TIMEOUT_S) as response:
            with archive.open("wb") as sink:
                while True:
                    if self._cancel.is_set():
                        raise _Cancelled()
                    chunk = response.read(CHUNK_BYTES)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > BUILD_BYTES:
                        raise WhisperGpuError(
                            f"o download passou dos {BUILD_BYTES} bytes publicados para o {BUILD_ASSET}"
                        )
                    digest.update(chunk)
                    sink.write(chunk)
                    self._advance(PHASE_DOWNLOAD, written / BUILD_BYTES)
        if written != BUILD_BYTES:
            raise WhisperGpuError(
                f"o download veio com {written} bytes e o {BUILD_ASSET} publicado tem {BUILD_BYTES}"
            )
        if digest.hexdigest() != BUILD_SHA256:
            raise WhisperGpuError(
                f"sha256 {digest.hexdigest()[:12]}… não bate com o {BUILD_SHA256[:12]}… publicado "
                f"para o {BUILD_ASSET}"
            )

    def _extract(self, archive: Path, server: Path) -> None:
        """O executável sai com o nome que a configuração pede: quem apontou
        ``gpu_binary`` para outro arquivo continuaria com a pasta certa e o servidor
        invisível para o resto do mod."""
        with zipfile.ZipFile(archive) as bundle:
            wanted = [item for item in bundle.infolist() if wanted_from_archive(item.filename)]
            if not any(Path(item.filename).name == BACKEND_DLL for item in wanted):
                raise WhisperGpuError(
                    f"{BUILD_ASSET} não traz {BACKEND_DLL}; o conteúdo do release do whisper.cpp mudou"
                )
            total = sum(item.file_size for item in wanted) or 1
            done = 0
            for item in wanted:
                if self._cancel.is_set():
                    raise _Cancelled()
                leaf = Path(item.filename).name
                target = server if leaf == SERVER_EXE else server.parent / leaf
                with bundle.open(item) as source, target.open("wb") as sink:
                    shutil.copyfileobj(source, sink, CHUNK_BYTES)
                done += item.file_size
                self._advance(PHASE_EXTRACT, done / total)
        if not server.is_file():
            raise WhisperGpuError(f"{server.name} não apareceu em {server.parent} depois de extrair {BUILD_ASSET}")

    def _advance(self, phase: str, ratio: float) -> None:
        with self._lock:
            if not self._progress.active:
                return
            self._progress.phase = phase
            self._progress.percent = max(0.0, min(1.0, ratio)) * 100.0

    def _finish(self, error: str = "") -> None:
        with self._lock:
            self._progress.active = False
            self._progress.done = not error
            self._progress.error = error
            if not error:
                self._progress.phase = ""
                self._progress.percent = 100.0
