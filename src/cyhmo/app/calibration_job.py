"""Calibração de limiares rodando em segundo plano.

A varredura leva minutos e nasceu como subcomando de terminal; aqui ela vira uma
tarefa observável, para que a interface mostre progresso em vez de congelar.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from cyhmo.config.schema import AppConfig, ProjectPaths
from cyhmo.domain.errors import ConfigError, CyhmoError

log = logging.getLogger("cyhmo.app.calibration")

DEFAULT_DATASET = "datasets/intent_pt-BR.yaml"
DEFAULT_SPONTANEOUS = "datasets/intent_spontaneous_pt-BR.yaml"
DEFAULT_GRID = "0.74,0.50 0.78,0.55 0.82,0.60 0.86,0.66"


def parse_grid(text: str) -> tuple[tuple[float, float], ...]:
    pairs: list[tuple[float, float]] = []
    for token in text.split():
        accept, _, reject = token.partition(",")
        try:
            values = (float(accept), float(reject))
        except ValueError as exc:
            raise ConfigError(f"par de limiares inválido: {token!r} (esperado 'aceite,rejeição')") from exc
        if values[1] >= values[0]:
            raise ConfigError(f"em {token!r} a rejeição deve ser menor que o aceite")
        pairs.append(values)
    if not pairs:
        raise ConfigError("a grade de limiares está vazia")
    return tuple(pairs)


@dataclass
class CalibrationRunner:
    """Uma calibração por vez: duas em paralelo disputariam o mesmo modelo de embeddings."""

    config: AppConfig
    paths: ProjectPaths
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)
    _state: dict[str, Any] = field(default_factory=lambda: _idle_state(), repr=False)

    def defaults(self) -> dict[str, Any]:
        return {
            "dataset": DEFAULT_DATASET,
            "spontaneous": DEFAULT_SPONTANEOUS,
            "grid": DEFAULT_GRID,
            "dataset_exists": (self.paths.base_dir / DEFAULT_DATASET).is_file(),
            "spontaneous_exists": (self.paths.base_dir / DEFAULT_SPONTANEOUS).is_file(),
        }

    def status(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._state, **self.defaults())

    def cancel(self) -> dict[str, Any]:
        self._cancel.set()
        return self.status()

    def start(self, dataset: str, spontaneous: str, grid: str) -> dict[str, Any]:
        pairs = parse_grid(grid)
        dataset_path = self._resolve(dataset or DEFAULT_DATASET)
        spontaneous_path = self._resolve(spontaneous or DEFAULT_SPONTANEOUS)
        with self._lock:
            if self._state["running"]:
                return dict(self._state, **self.defaults())
            self._cancel.clear()
            self._state = _idle_state(running=True, total=len(pairs))
            snapshot = dict(self._state, **self.defaults())
        thread = threading.Thread(
            target=self._run,
            args=(dataset_path, spontaneous_path, pairs),
            name="cyhmo-calibration",
            daemon=True,
        )
        thread.start()
        return snapshot

    def _resolve(self, relative: str) -> Path:
        path = Path(relative)
        return path if path.is_absolute() else (self.paths.base_dir / path)

    def _run(self, dataset_path: Path, spontaneous_path: Path, grid: Sequence[tuple[float, float]]) -> None:
        try:
            report = self._sweep(dataset_path, spontaneous_path, grid)
        except CyhmoError as exc:
            self._fail(str(exc))
            return
        except Exception as exc:
            log.exception("calibração falhou")
            self._fail(f"{type(exc).__name__}: {exc}")
            return
        with self._lock:
            self._state.update(running=False, report=report, cancelled=self._cancel.is_set())

    def _sweep(self, dataset_path: Path, spontaneous_path: Path, grid: Sequence[tuple[float, float]]) -> dict[str, Any]:
        from cyhmo.intent.calibration import load_dataset, load_spontaneous, run_calibration, to_payload
        from cyhmo.intent.embedders import build_embedder
        from cyhmo.intent.interpreter import build_interpreter
        from cyhmo.intent.language_packs import LanguagePackSet

        packs = LanguagePackSet.load(self.paths.packs_dir, self.config.languages.enabled, self.config.languages.primary)
        embedder = build_embedder(self.config.intent, self.paths.models_dir)

        def make(accept: float, reject: float) -> Any:
            tuned = self.config.replace(
                intent=self.config.intent.model_copy(
                    update={"accept_threshold": accept, "reject_threshold": reject}
                )
            )
            return build_interpreter(tuned, self.paths, packs=packs, embedder=embedder)

        report = run_calibration(
            make,
            load_dataset(dataset_path),
            load_spontaneous(spontaneous_path),
            grid,
            on_progress=self._progress,
        )
        return to_payload(report)

    def _progress(self, done: int, total: int) -> bool:
        with self._lock:
            self._state.update(done=done, total=total)
        return not self._cancel.is_set()

    def _fail(self, message: str) -> None:
        with self._lock:
            self._state.update(running=False, error=message)


def _idle_state(running: bool = False, total: int = 0) -> dict[str, Any]:
    return {"running": running, "done": 0, "total": total, "report": None, "error": "", "cancelled": False}
