"""Serviço de atualização: consulta a release mais nova, baixa e prepara a troca.

Checagem e download vivem em thread própria — nem o boot nem o pipeline esperam a rede.
Aplicar a troca é do encerramento do processo, não daqui.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Callable, Literal

from cyhmo.config.loader import ConfigStore
from cyhmo.config.schema import ProjectPaths
from cyhmo.domain.errors import UpdateError
from cyhmo.domain.events import LogLine
from cyhmo.pipeline.bus import EventBus
from cyhmo.update.installer import archive_path, discard_pending, stage
from cyhmo.update.release import Release, download, fetch_latest
from cyhmo.update.version import Version, current_version, is_newer

log = logging.getLogger("cyhmo.update")

LOG_SOURCE = "update"
PHASE_IDLE = "idle"
PHASE_CHECKING = "checking"
PHASE_DOWNLOADING = "downloading"
PHASE_STAGING = "staging"
PHASE_READY = "ready"
PHASE_ERROR = "error"
BUSY_PHASES = frozenset({PHASE_CHECKING, PHASE_DOWNLOADING, PHASE_STAGING, PHASE_READY})

DEV_INSTALL = (
    "esta é uma instalação de desenvolvimento (sem arquivo VERSION): a atualização automática "
    "fica de fora para não sobrescrever o clone; use git para atualizar"
)


@dataclass
class UpdateState:
    current: str
    phase: str = PHASE_IDLE
    latest: str = ""
    notes: str = ""
    url: str = ""
    percent: float = 0.0
    error: str = ""
    checked: bool = False
    dismissed: bool = False

    def to_dict(self, skipped: str, supported: bool) -> dict[str, Any]:
        available = bool(self.latest) and is_newer(self.latest, self.current)
        return {
            "current": self.current,
            "latest": self.latest,
            "available": available,
            "prompt": available and not self.dismissed and self.latest != skipped,
            "phase": self.phase,
            "percent": round(self.percent, 1),
            "error": self.error,
            "notes": self.notes,
            "url": self.url,
            "checked": self.checked,
            "skipped": skipped,
            "supported": supported,
            "busy": self.phase in BUSY_PHASES,
        }


class UpdateService:
    """Estado consultável pela interface e as três ações que ela dispara: procurar,
    instalar e adiar (por esta sessão ou de vez)."""

    def __init__(
        self,
        config_store: ConfigStore,
        paths: ProjectPaths,
        bus: EventBus,
        transport: Any = None,
    ) -> None:
        self._config_store = config_store
        self._paths = paths
        self._bus = bus
        self._transport = transport
        self._lock = threading.RLock()
        self._state = UpdateState(current=current_version(paths.base_dir))
        self._release: Release | None = None
        self._closed = False
        self.on_ready: Callable[[], None] | None = None

    @property
    def supported(self) -> bool:
        version = Version.parse(self._state.current)
        return version is not None and version.released

    def status(self) -> dict[str, Any]:
        with self._lock:
            return self._state.to_dict(self._skipped(), self.supported)

    def start(self) -> None:
        """Limpa a sobra de uma troca interrompida e, se a config permitir, procura release
        nova sem segurar o boot."""
        discard_pending(self._paths.data_dir)
        if self._config_store.config.update.check_on_start and self.supported:
            self.check()

    def check(self) -> dict[str, Any]:
        if self._begin(PHASE_CHECKING):
            self._spawn("cyhmo-update-check", self._run_check)
        return self.status()

    def install(self) -> dict[str, Any]:
        """A resposta sai antes do download: quem pediu precisa saber que foi aceito."""
        if not self.supported:
            raise UpdateError(DEV_INSTALL)
        if self._begin(PHASE_DOWNLOADING):
            self._spawn("cyhmo-update-install", self._run_install)
        return self.status()

    def skip(self) -> dict[str, Any]:
        with self._lock:
            version = self._state.latest
        if not version:
            raise UpdateError("nenhuma versão nova para pular")
        self._config_store.update({"update": {"skipped_version": version}})
        self._log(f"versão {version} ignorada a pedido; o aviso volta na próxima release")
        return self.status()

    def postpone(self) -> dict[str, Any]:
        with self._lock:
            self._state.dismissed = True
        return self.status()

    def close(self) -> None:
        self._closed = True

    def check_now(self) -> dict[str, Any]:
        """Versão síncrona da checagem, para a linha de comando."""
        self._remember(fetch_latest(self._repository(), self._transport), PHASE_IDLE)
        return self.status()

    def download_now(self) -> str:
        """Baixa e prepara a troca; devolve a versão preparada."""
        if not self.supported:
            raise UpdateError(DEV_INSTALL)
        release = self._release or fetch_latest(self._repository(), self._transport)
        if not is_newer(release.version, self._state.current):
            raise UpdateError(f"a versão instalada ({self._state.current}) já é a mais recente")
        self._remember(release, PHASE_DOWNLOADING)
        archive = download(release, archive_path(self._paths.data_dir), self._progress, self._transport)
        self._phase(PHASE_STAGING)
        stage(archive, release.version, self._paths.data_dir)
        self._phase(PHASE_READY, percent=100.0)
        return release.version

    def _run_check(self) -> None:
        try:
            self._remember(fetch_latest(self._repository(), self._transport), PHASE_IDLE)
        except Exception as exc:
            self._fail(f"não consegui procurar atualizações: {exc}")

    def _run_install(self) -> None:
        try:
            version = self.download_now()
        except Exception as exc:
            self._fail(f"a atualização não foi instalada: {exc}")
            return
        self._log(f"versão {version} pronta; reiniciando o mod para concluir a instalação")
        callback = self.on_ready
        if callback is not None and not self._closed:
            callback()

    def _remember(self, release: Release, phase: str) -> None:
        """A release é relembrada a cada passo do fluxo; o aviso sai uma vez por versão."""
        with self._lock:
            fresh = self._state.latest != release.version
            self._release = release
            self._state.latest = release.version
            self._state.notes = release.notes
            self._state.url = release.page_url
            self._state.checked = True
            self._state.error = ""
            self._state.phase = phase
            newer = is_newer(release.version, self._state.current)
            current = self._state.current
        if fresh and newer:
            self._log(f"versão {release.version} disponível (a instalada é a {current})")

    def _begin(self, phase: str) -> bool:
        with self._lock:
            if self._state.phase in BUSY_PHASES:
                return False
            self._state.phase = phase
            self._state.percent = 0.0
            self._state.error = ""
            return True

    def _phase(self, phase: str, percent: float | None = None) -> None:
        with self._lock:
            self._state.phase = phase
            if percent is not None:
                self._state.percent = percent

    def _progress(self, written: int, total: int) -> None:
        with self._lock:
            self._state.percent = (written / total * 100.0) if total > 0 else 0.0

    def _fail(self, message: str) -> None:
        with self._lock:
            self._state.phase = PHASE_ERROR
            self._state.error = message
        log.warning("%s", message)
        self._log(message, level="warning")

    def _repository(self) -> str:
        return self._config_store.config.update.repository

    def _skipped(self) -> str:
        return self._config_store.config.update.skipped_version

    def _spawn(self, name: str, target: Callable[[], None]) -> None:
        threading.Thread(target=target, name=name, daemon=True).start()

    def _log(self, message: str, level: Literal["debug", "info", "warning", "error"] = "info") -> None:
        self._bus.publish(LogLine(level=level, message=message, source=LOG_SOURCE))
