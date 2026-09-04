"""Catálogo dos pesos ggml do whisper.cpp e a gerência que a interface usa.

Cada peso aqui tem SHA-1 publicado pelo próprio projeto whisper.cpp, em
``models/README.md``, e é contra ele que o download é conferido antes de virar arquivo
definitivo. O ETag do Hugging Face não serve para isso: o repositório passou a usar
armazenamento Xet e o ETag deixou de ser o hash do conteúdo — conferi baixando o
``ggml-tiny.bin`` e comparando, e os dois valores não batem.

SHA-1 não resiste a colisão deliberada. O que ele pega é download truncado, espelho
trocado e arquivo corrompido, que são as falhas que acontecem de verdade aqui; contra
adulteração quem responde é o TLS até huggingface.co.

O catálogo é curto de propósito. A escada vai do mais rápido ao mais preciso e cada
degrau precisa se pagar: ``large-v3-turbo-q5_0`` entrega precisão de modelo grande
ocupando quase o mesmo disco do ``small``, e por isso o ``medium`` ficou de fora — é mais
lento e menos preciso que ele. Quem quiser outro peso continua podendo apontar
``stt.whisper_cpp.model`` para o arquivo que quiser, no modo avançado.

Só biblioteca padrão: este módulo é alcançado pelo boot.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from cyhmo.domain.errors import CyhmoError

log = logging.getLogger("cyhmo.stt.models")

BASE_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"
DEFAULT_SUBDIR = "stt/ggml"
CHUNK_BYTES = 1 << 20
DOWNLOAD_TIMEOUT_S = 60.0
PART_SUFFIX = ".part"

CANCELLED = "cancelado"


class WhisperModelError(CyhmoError):
    """Não foi possível baixar, conferir ou remover um peso do whisper.cpp."""


class _Cancelled(Exception):
    """Interrupção pedida pela interface; não é erro para mostrar como falha."""


@dataclass(frozen=True)
class WhisperModel:
    """``audio_ctx`` é a janela do encoder que este peso aguenta, e 0 quer dizer que só a
    janela cheia serve.

    O corte em 512 quadros que acelera o ``small`` foi medido com ele; nos pesos grandes
    ele tira o encoder da distribuição em que foi treinado e o decodificador entra em
    laço. Medido em 2026-09-03, mesma fala, mesma máquina: ``large-v3-turbo-q5_0`` com
    ``-ac 512`` devolveu "Atira no olho direito." repetido cinco vezes, e com a janela
    cheia devolveu a frase uma vez só."""

    name: str
    file: str
    size_bytes: int
    sha1: str
    audio_ctx: int
    recommended: bool = False


CATALOG: tuple[WhisperModel, ...] = (
    WhisperModel("tiny", "ggml-tiny.bin", 77_691_713, "bd577a113a864445d4c299885e0cb97d4ba92b5f", 512),
    WhisperModel("base", "ggml-base.bin", 147_951_465, "465707469ff3a37a2b9b8d8f89f2f99de7299dac", 512),
    WhisperModel(
        "small",
        "ggml-small.bin",
        487_601_967,
        "55356645c2b361a969dfd0ef2c5a50d530afd8d5",
        512,
        recommended=True,
    ),
    WhisperModel(
        "large-v3-turbo-q5_0",
        "ggml-large-v3-turbo-q5_0.bin",
        574_041_195,
        "e050f7970618a659205450ad97eb95a18d69c9ee",
        0,
    ),
    WhisperModel(
        "large-v3-turbo",
        "ggml-large-v3-turbo.bin",
        1_624_555_275,
        "4af2b29d7ec73d781377bfd1758ca957a807e941",
        0,
    ),
)

BY_NAME = {model.name: model for model in CATALOG}
BY_FILE = {model.file: model for model in CATALOG}
RECOMMENDED = next(model for model in CATALOG if model.recommended)


def sha1_of(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def mismatch(size: int, sha1: str, model: WhisperModel) -> str | None:
    """Por que o arquivo não serve, ou ``None`` quando confere."""
    if size != model.size_bytes:
        return f"o arquivo tem {size} bytes e o {model.name} publicado tem {model.size_bytes}"
    if sha1 != model.sha1:
        return f"sha1 {sha1[:12]}… não bate com o {model.sha1[:12]}… publicado para o {model.name}"
    return None


def resolve_audio_ctx(model_file: str, wanted: int) -> tuple[int, str]:
    """A janela do encoder que vai ser usada de fato, e o aviso quando ela não é a pedida.

    Corrigir aqui em vez de só avisar é deliberado: um ``audio_ctx`` pequeno demais não
    deixa o mod mais lento, deixa o reconhecimento devolvendo a mesma palavra em laço — e
    quem trocou de peso pelo painel não tem como adivinhar que o número calibrado para o
    anterior é a causa. Um peso fora do catálogo passa intocado; ali não há o que saber."""
    known = BY_FILE.get(model_file)
    if known is None:
        return wanted, ""
    if known.audio_ctx == 0:
        if wanted == 0:
            return 0, ""
        return 0, (
            f"stt.whisper_cpp.audio_ctx = {wanted} faz o {known.name} repetir a mesma palavra "
            "em laço; usando a janela cheia nesta sessão. Grave audio_ctx = 0 no config.toml "
            "para não ver este aviso de novo."
        )
    if wanted != 0 and wanted < known.audio_ctx:
        return known.audio_ctx, (
            f"stt.whisper_cpp.audio_ctx = {wanted} é curto demais para o {known.name}; "
            f"usando {known.audio_ctx} nesta sessão."
        )
    return wanted, ""


def lookup(name: str) -> WhisperModel:
    model = BY_NAME.get(name.strip())
    if model is None:
        raise WhisperModelError(
            f"modelo {name!r} não está no catálogo (conhecidos: {', '.join(BY_NAME)})"
        )
    return model


@dataclass
class DownloadProgress:
    model: str = ""
    active: bool = False
    percent: float = 0.0
    downloaded_bytes: int = 0
    total_bytes: int = 0
    error: str = ""
    done: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "active": self.active,
            "percent": round(self.percent, 1),
            "downloaded_bytes": self.downloaded_bytes,
            "total_bytes": self.total_bytes,
            "error": self.error,
            "done": self.done,
        }


class WhisperModelAdmin:
    """Lista, baixa e remove os pesos do whisper.cpp.

    A pasta sai da própria configuração, não de uma constante: quem apontou
    ``stt.whisper_cpp.model`` para outro lugar continua gerenciando a pasta dele."""

    def __init__(
        self,
        base_dir: Path,
        models_dir: Path,
        opener: Callable[..., Any] = urllib.request.urlopen,
    ) -> None:
        # Resolvido dos dois lados: no Windows a pasta pode chegar na forma curta
        # (``SAMUEL~1``) e o alvo, já resolvido, deixaria de ser relativo a ela — o valor
        # gravado na config viraria um caminho absoluto sem necessidade.
        self._base_dir = Path(base_dir).resolve()
        self._models_dir = Path(models_dir).resolve()
        self._opener = opener
        self._lock = threading.Lock()
        self._progress = DownloadProgress()
        self._cancel = threading.Event()

    def directory_for(self, configured: str) -> Path:
        if configured.strip():
            return (self._base_dir / configured).resolve().parent
        return (self._models_dir / DEFAULT_SUBDIR).resolve()

    def config_value(self, configured: str, model: WhisperModel) -> str:
        """O que gravar em ``stt.whisper_cpp.model``: relativo à pasta do mod quando dá."""
        target = self.directory_for(configured) / model.file
        try:
            return target.relative_to(self._base_dir).as_posix()
        except ValueError:
            return target.as_posix()

    def status(self, configured: str) -> dict[str, Any]:
        directory = self.directory_for(configured)
        current_file = Path(configured).name if configured.strip() else ""
        models = []
        for model in CATALOG:
            path = directory / model.file
            models.append(
                {
                    "name": model.name,
                    "file": model.file,
                    "size_bytes": model.size_bytes,
                    "audio_ctx": model.audio_ctx,
                    "recommended": model.recommended,
                    "installed": path.is_file(),
                    "current": model.file == current_file,
                    "config_value": self.config_value(configured, model),
                }
            )
        with self._lock:
            progress = self._progress.to_dict()
        return {
            "directory": str(directory),
            "current_file": current_file,
            "current_known": current_file in BY_FILE,
            "recommended": RECOMMENDED.name,
            "models": models,
            "download": progress,
        }

    def start_download(self, configured: str, name: str) -> dict[str, Any]:
        model = lookup(name)
        directory = self.directory_for(configured)
        with self._lock:
            if self._progress.active:
                return self._progress.to_dict()
            self._cancel.clear()
            self._progress = DownloadProgress(model=model.name, active=True, total_bytes=model.size_bytes)
            snapshot = self._progress.to_dict()
        threading.Thread(
            target=self._download,
            args=(model, directory),
            name="cyhmo-whisper-download",
            daemon=True,
        ).start()
        return snapshot

    def cancel_download(self) -> dict[str, Any]:
        self._cancel.set()
        with self._lock:
            return self._progress.to_dict()

    def progress(self) -> dict[str, Any]:
        with self._lock:
            return self._progress.to_dict()

    def delete(self, configured: str, name: str) -> dict[str, Any]:
        """Remover o peso em uso deixaria o reconhecimento quebrado no próximo arranque,
        e não há escolha óbvia para pôr no lugar; quem decide é o usuário, antes."""
        model = lookup(name)
        if Path(configured).name == model.file:
            raise WhisperModelError(
                f"{model.name} é o modelo em uso; escolha outro antes de removê-lo"
            )
        target = self.directory_for(configured) / model.file
        if not target.is_file():
            raise WhisperModelError(f"{model.name} não está instalado em {target.parent}")
        try:
            target.unlink()
        except OSError as exc:
            raise WhisperModelError(f"não consegui remover {target}: {exc}") from exc
        log.info("peso %s removido de %s", model.name, target.parent)
        return {"model": model.name, "removed": True}

    def _download(self, model: WhisperModel, directory: Path) -> None:
        target = directory / model.file
        partial = target.with_suffix(target.suffix + PART_SUFFIX)
        try:
            directory.mkdir(parents=True, exist_ok=True)
            partial.unlink(missing_ok=True)
            written, sha1 = self._stream(model, partial)
            problem = mismatch(written, sha1, model)
            if problem is not None:
                raise WhisperModelError(f"o download não confere: {problem}")
            partial.replace(target)
        except _Cancelled:
            partial.unlink(missing_ok=True)
            self._finish(model, error=CANCELLED)
            return
        except (WhisperModelError, urllib.error.URLError, OSError, TimeoutError) as exc:
            partial.unlink(missing_ok=True)
            log.warning("download do peso %s falhou: %s", model.name, exc)
            self._finish(model, error=str(exc))
            return
        log.info("peso %s instalado em %s", model.name, target.parent)
        self._finish(model)

    def _stream(self, model: WhisperModel, partial: Path) -> tuple[int, str]:
        digest = hashlib.sha1()
        written = 0
        with self._opener(f"{BASE_URL}/{model.file}", timeout=DOWNLOAD_TIMEOUT_S) as response:
            with partial.open("wb") as sink:
                while True:
                    if self._cancel.is_set():
                        raise _Cancelled()
                    chunk = response.read(CHUNK_BYTES)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > model.size_bytes:
                        raise WhisperModelError(
                            f"o download passou dos {model.size_bytes} bytes publicados para o {model.name}"
                        )
                    digest.update(chunk)
                    sink.write(chunk)
                    self._advance(model, written)
        return written, digest.hexdigest()

    def _advance(self, model: WhisperModel, written: int) -> None:
        with self._lock:
            if self._progress.model != model.name:
                return
            self._progress.downloaded_bytes = written
            self._progress.percent = written / model.size_bytes * 100.0 if model.size_bytes else 0.0

    def _finish(self, model: WhisperModel, error: str = "") -> None:
        with self._lock:
            if self._progress.model != model.name:
                return
            self._progress.active = False
            self._progress.done = not error
            self._progress.error = error
            if not error:
                self._progress.percent = 100.0
                self._progress.downloaded_bytes = model.size_bytes
