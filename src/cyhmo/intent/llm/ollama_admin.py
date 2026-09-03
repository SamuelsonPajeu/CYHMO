"""Administração do Ollama pela interface: detectar, listar e baixar modelos.

Instalar o Ollama continua sendo do usuário — o mod não instala software de
terceiros. O que ele faz é dizer com clareza o que falta e baixar o modelo.
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from cyhmo.domain.errors import CyhmoError

log = logging.getLogger("cyhmo.intent.llm.ollama")


class LlmAdminError(CyhmoError):
    """Falha ao administrar os modelos locais do Ollama."""

BINARY_NAME = "ollama"
INSTALL_URL = "https://ollama.com/download"
SUGGESTED_MODELS = ("qwen2.5:3b", "llama3.2:3b", "gemma2:2b", "qwen2.5:7b")
SUGGESTED_MODEL = SUGGESTED_MODELS[0]
PROBE_TIMEOUT_S = 2.0
PULL_TIMEOUT_S = 3600.0
DELETE_TIMEOUT_S = 30.0

KNOWN_LOCATIONS = (
    ("LOCALAPPDATA", "Programs/Ollama/ollama.exe"),
    ("PROGRAMFILES", "Ollama/ollama.exe"),
    ("USERPROFILE", "AppData/Local/Programs/Ollama/ollama.exe"),
)


def find_binary(lookup: Callable[[str], str | None] = shutil.which) -> str | None:
    """O instalador do Ollama só acrescenta o PATH para processos NOVOS: um mod já
    aberto nunca enxerga a instalação recém-feita. Por isso o PATH é a primeira
    tentativa, não a única."""
    found = lookup(BINARY_NAME)
    if found:
        return found
    for variable, relative in KNOWN_LOCATIONS:
        base = os.environ.get(variable)
        if not base:
            continue
        candidate = Path(base) / relative
        if candidate.is_file():
            return str(candidate)
    return None


@dataclass
class PullProgress:
    model: str = ""
    active: bool = False
    percent: float = 0.0
    detail: str = ""
    error: str = ""
    done: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "active": self.active,
            "percent": round(self.percent, 1),
            "detail": self.detail,
            "error": self.error,
            "done": self.done,
        }


@dataclass
class OllamaAdmin:
    endpoint: str
    transport: Any = None
    binary_lookup: Callable[[str], str | None] = shutil.which
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _progress: PullProgress = field(default_factory=PullProgress, repr=False)
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    def status(self) -> dict[str, Any]:
        """Responder no endereço já prova que o Ollama existe: exigir o binário no PATH
        para dizer "instalado" reprova uma instalação que está atendendo agora."""
        binary = find_binary(self.binary_lookup)
        models, error = self._tags()
        online = error == ""
        with self._lock:
            progress = self._progress.to_dict()
        return {
            "installed": binary is not None or online,
            "binary": binary,
            "install_url": INSTALL_URL,
            "endpoint": self.endpoint,
            "online": online,
            "error": error,
            "models": list(models),
            "suggested_model": SUGGESTED_MODEL,
            "suggested_models": list(SUGGESTED_MODELS),
            "pull": progress,
        }

    def delete_model(self, model: str) -> dict[str, Any]:
        """Apagar é imediato e irreversível; quem confirma é a interface."""
        import httpx

        model = model.strip()
        if not model:
            raise LlmAdminError("informe o modelo a remover")
        try:
            with httpx.Client(timeout=DELETE_TIMEOUT_S, transport=self.transport) as client:
                response = client.request("DELETE", self._url("delete"), json={"model": model})
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LlmAdminError(f"não consegui remover {model}: {exc}") from exc
        log.info("modelo %s removido do Ollama", model)
        return {"model": model, "removed": True}

    def start_pull(self, model: str) -> dict[str, Any]:
        model = model.strip()
        if not model:
            return self._fail("informe o nome do modelo, por exemplo " + SUGGESTED_MODEL)
        with self._lock:
            if self._progress.active:
                return self._progress.to_dict()
            self._cancel.clear()
            self._progress = PullProgress(model=model, active=True, detail="iniciando")
            snapshot = self._progress.to_dict()
        threading.Thread(target=self._pull, args=(model,), name="cyhmo-ollama-pull", daemon=True).start()
        return snapshot

    def cancel_pull(self) -> dict[str, Any]:
        self._cancel.set()
        with self._lock:
            return self._progress.to_dict()

    def progress(self) -> dict[str, Any]:
        with self._lock:
            return self._progress.to_dict()

    def _pull(self, model: str) -> None:
        try:
            for update in self._stream_pull(model):
                if self._cancel.is_set():
                    self._finish(model, error="download cancelado")
                    return
                self._publish(model, update)
        except Exception as exc:
            log.warning("download do modelo %s falhou: %s", model, exc)
            self._finish(model, error=str(exc))
            return
        self._finish(model)

    def _stream_pull(self, model: str) -> Iterator[dict[str, Any]]:
        import json

        import httpx

        with httpx.Client(timeout=PULL_TIMEOUT_S, transport=self.transport) as client:
            with client.stream("POST", self._url("pull"), json={"model": model, "stream": True}) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.strip():
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(payload, dict):
                        yield payload

    def _url(self, route: str) -> str:
        return f"{self.endpoint.rstrip('/')}/api/{route}"

    def _publish(self, model: str, update: dict[str, Any]) -> None:
        total = _number(update.get("total"))
        completed = _number(update.get("completed"))
        percent = (completed / total * 100.0) if total > 0 else 0.0
        with self._lock:
            if self._progress.model != model:
                return
            self._progress.detail = str(update.get("status", ""))
            if total > 0:
                self._progress.percent = percent
            if isinstance(update.get("error"), str):
                self._progress.error = update["error"]

    def _finish(self, model: str, error: str = "") -> None:
        with self._lock:
            if self._progress.model != model:
                return
            self._progress.active = False
            self._progress.done = not error
            self._progress.error = error or self._progress.error
            if self._progress.done:
                self._progress.percent = 100.0
                self._progress.detail = "concluído"

    def _fail(self, message: str) -> dict[str, Any]:
        with self._lock:
            self._progress = PullProgress(error=message)
            return self._progress.to_dict()

    def _tags(self) -> tuple[tuple[dict[str, Any], ...], str]:
        import httpx

        try:
            with httpx.Client(timeout=PROBE_TIMEOUT_S, transport=self.transport) as client:
                response = client.get(self._url("tags"))
                response.raise_for_status()
                payload = response.json()
        except Exception as exc:
            return (), str(exc)
        models = payload.get("models") if isinstance(payload, dict) else None
        if not isinstance(models, list):
            return (), ""
        entries = tuple(_model_entry(item) for item in models if isinstance(item, dict) and item.get("name"))
        return tuple(sorted(entries, key=lambda entry: entry["name"])), ""


def _model_entry(item: dict[str, Any]) -> dict[str, Any]:
    details = item.get("details") if isinstance(item.get("details"), dict) else {}
    return {
        "name": str(item["name"]),
        "size": int(_number(item.get("size"))),
        "modified_at": str(item.get("modified_at", "")),
        "parameter_size": str(details.get("parameter_size", "")),
    }


def _number(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0
