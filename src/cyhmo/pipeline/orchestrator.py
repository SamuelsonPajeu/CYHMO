"""Orquestração do pipeline de voz: STT → estado → intenção → injeção.

O loop asyncio só coordena, cronometra e publica eventos; toda chamada às
camadas roda no executor, e um enunciado novo cancela o que estiver em curso.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from cyhmo.config.schema import AppConfig
from cyhmo.domain.contracts import (
    CommandRef,
    GameState,
    InjectResult,
    Interpretation,
    Transcript,
    Utterance,
    UtteranceStatus,
)
from cyhmo.domain.errors import CyhmoError
from cyhmo.domain.events import (
    Event,
    GrammarChanged,
    InjectionDone,
    InterpretationReady,
    ListeningPhase,
    LogLine,
    PhaseChanged,
    TranscriptReady,
    UtteranceCaptured,
    UtteranceFinished,
)
from cyhmo.domain.ports import CommandInjector, GameStateReader, Interpreter, Transcriber
from cyhmo.pipeline.budget import percentiles
from cyhmo.pipeline.bus import EventBus, Unsubscribe
from cyhmo.pipeline.recorder import UtteranceRecorder
from cyhmo.pipeline.session import Session
from cyhmo.pipeline.telemetry import STAGES, TelemetryWriter, build_record

log = logging.getLogger("cyhmo.pipeline")

GrammarListener = Callable[[GrammarChanged], None]
LOG_SOURCE = "pipeline"
STATS_WINDOW = 200
EXECUTOR_WORKERS = 4

_STOP = object()


class PipelineError(CyhmoError):
    """Uso indevido do pipeline (submit antes de run, run duplicado)."""


@dataclass
class _UtteranceRun:
    utterance: Utterance
    wall_time: datetime
    transcript: Transcript | None = None
    state: GameState | None = None
    interpretation: Interpretation | None = None
    injection: InjectResult | None = None
    status: UtteranceStatus = "error"
    error: str | None = None
    durations_ms: dict[str, float] = field(default_factory=dict)
    last_stage_end: float | None = None

    @property
    def utt_id(self) -> str:
        return self.utterance.utt_id

    @property
    def commands(self) -> tuple[CommandRef, ...]:
        return () if self.interpretation is None else self.interpretation.commands

    def record_stage(self, stage: str, t_start: float, t_end: float) -> None:
        self.durations_ms[stage] = self.durations_ms.get(stage, 0.0) + (t_end - t_start) * 1000.0
        self.last_stage_end = t_end

    def latencies(self) -> dict[str, float | None]:
        utterance = self.utterance
        latencies: dict[str, float | None] = {stage: self.durations_ms.get(stage) for stage in STAGES}
        latencies["vad_tail"] = _vad_tail_ms(utterance)
        if self.last_stage_end is not None:
            latencies["total"] = (self.last_stage_end - utterance.t_release) * 1000.0
        return latencies


def _vad_tail_ms(utterance: Utterance) -> float:
    if utterance.t_vad_end is None:
        return 0.0
    return max(0.0, (utterance.t_release - utterance.t_vad_end) * 1000.0)


class _LatencyStats:
    def __init__(self, window: int) -> None:
        self._window: deque[dict[str, float | None]] = deque(maxlen=window)
        self._by_status: Counter[str] = Counter()
        self._lock = threading.Lock()

    def add(self, status: str, latencies: dict[str, float | None]) -> None:
        with self._lock:
            self._window.append(latencies)
            self._by_status[status] += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            samples = list(self._window)
            by_status = dict(self._by_status)
        return {
            "window": len(samples),
            "count": sum(by_status.values()),
            "by_status": by_status,
            "stages": {stage: _stage_stats(samples, stage) for stage in STAGES},
        }


def _stage_stats(samples: list[dict[str, float | None]], stage: str) -> dict[str, float | int]:
    values = [sample[stage] for sample in samples if sample.get(stage) is not None]
    p50, p95, maximum = percentiles(values)
    return {"p50_ms": round(p50, 2), "p95_ms": round(p95, 2), "max_ms": round(maximum, 2), "n": len(values)}


class VoicePipeline:
    def __init__(
        self,
        config: AppConfig,
        bus: EventBus,
        transcriber: Transcriber,
        interpreter: Interpreter,
        state_reader: GameStateReader,
        injector: CommandInjector,
        session: Session,
        telemetry: TelemetryWriter | None = None,
        recorder: UtteranceRecorder | None = None,
        executor: ThreadPoolExecutor | None = None,
        record: bool = False,
    ) -> None:
        self._config = config
        self._bus = bus
        self._transcriber = transcriber
        self._interpreter = interpreter
        self.state_reader = state_reader
        self._injector = injector
        self._session = session
        self._telemetry = telemetry
        self._recorder = recorder
        self._save_audio = record or config.debug.save_audio
        self._owns_executor = executor is None
        self._executor = executor or ThreadPoolExecutor(
            max_workers=EXECUTOR_WORKERS, thread_name_prefix="cyhmo-pipeline"
        )
        self._loop = _running_loop_or_none()
        self._queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=1)
        self._current: asyncio.Task[None] | None = None
        self._running = False
        self._stopping = False
        self._wav_paths: dict[str, str] = {}
        self._grammar_listeners: list[GrammarListener] = []
        self._grammar_pending: GrammarChanged | None = None
        self._grammar_task: asyncio.Task[None] | None = None
        self._stats = _LatencyStats(STATS_WINDOW)
        self._lock = threading.Lock()

    @property
    def bus(self) -> EventBus:
        return self._bus

    @property
    def session(self) -> Session:
        return self._session

    @property
    def config(self) -> AppConfig:
        return self._config

    @property
    def is_running(self) -> bool:
        return self._running

    def submit(self, utterance: Utterance) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            raise PipelineError("pipeline não está em execução: chame run() antes de submit()")
        if self._stopping:
            log.warning("pipeline encerrando; enunciado %s descartado", utterance.utt_id)
            return
        loop.call_soon_threadsafe(self._enqueue, utterance)

    def set_wav_path(self, utt_id: str, path: str) -> None:
        with self._lock:
            self._wav_paths[utt_id] = path

    def add_grammar_listener(self, listener: GrammarListener) -> Unsubscribe:
        with self._lock:
            self._grammar_listeners.append(listener)

        def unsubscribe() -> None:
            with self._lock:
                if listener in self._grammar_listeners:
                    self._grammar_listeners.remove(listener)

        return unsubscribe

    def stats(self) -> dict[str, Any]:
        snapshot = self._stats.snapshot()
        snapshot["by_source"] = self._session.counts_by_source()
        return snapshot

    async def warm_up(self) -> None:
        loop = asyncio.get_running_loop()
        t_start = time.perf_counter()
        await loop.run_in_executor(self._executor, self._transcriber.warm_up)
        for hook_name in ("warm_up", "wait_ready"):
            hook = getattr(self._interpreter, hook_name, None)
            if callable(hook):
                await loop.run_in_executor(self._executor, hook)
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self._log("info", f"aquecimento concluído em {elapsed_ms:.0f} ms")

    async def run(self) -> None:
        if self._running:
            raise PipelineError("run() já está em execução neste pipeline")
        self._running = True
        self._loop = asyncio.get_running_loop()
        unsubscribe = self._bus.subscribe(self._on_bus_event)
        try:
            if self._stopping:
                return
            await self._consume_queue()
        finally:
            unsubscribe()
            self._stopping = True
            await self._settle_grammar_task()
            self._release_resources()
            self._running = False

    def stop(self) -> None:
        self._stopping = True
        loop = self._loop
        if loop is None or loop.is_closed() or not self._running:
            return
        try:
            loop.call_soon_threadsafe(self._begin_shutdown)
        except RuntimeError:
            log.debug("loop já encerrado ao pedir stop()")

    async def _consume_queue(self) -> None:
        while True:
            item = await self._queue.get()
            if item is _STOP:
                return
            await self._process_current(item)

    def _enqueue(self, utterance: Utterance) -> None:
        if self._stopping:
            log.warning("pipeline encerrando; enunciado %s descartado", utterance.utt_id)
            return
        if self._queue.full():
            self._finish_unprocessed(self._queue.get_nowait())
        self._cancel_current()
        self._queue.put_nowait(utterance)

    def _begin_shutdown(self) -> None:
        while not self._queue.empty():
            item = self._queue.get_nowait()
            if item is not _STOP:
                self._finish_unprocessed(item)
        self._cancel_current()
        self._queue.put_nowait(_STOP)

    def _cancel_current(self) -> None:
        if self._current is not None and not self._current.done():
            self._current.cancel()

    async def _process_current(self, utterance: Utterance) -> None:
        """Garante exatamente um registro por enunciado, mesmo cancelado antes do primeiro passo."""
        run = _UtteranceRun(utterance, wall_time=datetime.now().astimezone())
        self._current = self._loop.create_task(self._process(run))
        try:
            await self._current
        except asyncio.CancelledError:
            run.status = "cancelled"
            self._finish(run)
            if asyncio.current_task().cancelling():
                raise
        finally:
            self._current = None

    async def _process(self, run: _UtteranceRun) -> None:
        try:
            await self._run_stages(run)
        except Exception as exc:
            run.status = "error"
            run.error = f"{type(exc).__name__}: {exc}"
            log.exception("falha ao processar o enunciado %s", run.utt_id)
            self._log("error", f"enunciado {run.utt_id} falhou: {run.error}")
        self._finish(run)

    async def _run_stages(self, run: _UtteranceRun) -> None:
        utterance = run.utterance
        self._phase("transcribing", run.utt_id)
        run.transcript = await self._timed(run, "stt", self._transcriber.transcribe, utterance.audio)
        self._publish(TranscriptReady(utt_id=run.utt_id, transcript=run.transcript, latency_ms=run.durations_ms["stt"]))
        run.state = await self._timed(run, "state", self.state_reader.read_state)
        self._phase("interpreting", run.utt_id)
        run.interpretation = await self._timed(run, "interpret", self._interpreter.interpret, run.transcript, run.state)
        self._publish(InterpretationReady(utt_id=run.utt_id, interpretation=run.interpretation, state=run.state))
        run.status = await self._decide_and_inject(run)

    async def _decide_and_inject(self, run: _UtteranceRun) -> UtteranceStatus:
        if run.interpretation is None or run.interpretation.is_empty:
            return "no_command"
        if not self._config.inject.enabled:
            return "dry_run"
        if self._config.inject.require_can_talk:
            run.state = await self._timed(run, "state", self.state_reader.read_state)
            if run.state.can_talk is False:
                return "blocked_cannot_talk"
        self._phase("injecting", run.utt_id)
        commands = run.interpretation.commands
        run.injection = await self._timed(run, "inject", self._injector.inject, commands)
        self._publish(InjectionDone(utt_id=run.utt_id, commands=commands, result=run.injection))
        if run.injection.ok:
            return "injected"
        run.error = run.injection.error or "injeção falhou sem detalhe"
        self._log("error", f"injeção de {run.utt_id} falhou: {run.error}")
        return "error"

    async def _timed(self, run: _UtteranceRun, stage: str, call: Callable[..., Any], *args: Any) -> Any:
        t_start = time.perf_counter()
        result = await self._loop.run_in_executor(self._executor, call, *args)
        run.record_stage(stage, t_start, time.perf_counter())
        return result

    def _finish(self, run: _UtteranceRun) -> None:
        self._complete(run)
        self._phase("idle", run.utt_id)

    def _finish_unprocessed(self, utterance: Utterance) -> None:
        run = _UtteranceRun(utterance, wall_time=datetime.now().astimezone(), status="cancelled")
        self._complete(run)

    def _complete(self, run: _UtteranceRun) -> None:
        latencies = run.latencies()
        wav_path = self._resolve_wav_path(run.utterance)
        self._write_record(run, latencies, wav_path)
        self._stats.add(run.status, latencies)
        self._publish(
            UtteranceFinished(
                utt_id=run.utt_id,
                status=run.status,
                latency_ms={stage: value for stage, value in latencies.items() if value is not None},
                error=run.error,
            )
        )

    def _resolve_wav_path(self, utterance: Utterance) -> str | None:
        with self._lock:
            known = self._wav_paths.pop(utterance.utt_id, None)
        if known is not None or self._recorder is None or not self._save_audio:
            return known
        try:
            return str(self._recorder.save(utterance))
        except Exception as exc:
            self._log("warning", f"não gravei o áudio de {utterance.utt_id}: {exc}")
            return None

    def _write_record(self, run: _UtteranceRun, latencies: dict[str, float | None], wav_path: str | None) -> None:
        if self._telemetry is None:
            return
        record = build_record(
            self._session,
            run.utterance,
            run.status,
            run.transcript,
            run.state,
            run.interpretation,
            run.injection,
            latencies,
            wav_path,
            run.error,
            run.wall_time,
        )
        try:
            self._telemetry.write(record)
        except Exception as exc:
            self._log("warning", f"telemetria de {run.utt_id} não gravada: {exc}")

    def _on_bus_event(self, event: Event) -> None:
        if isinstance(event, UtteranceCaptured) and event.wav_path:
            self.set_wav_path(event.utt_id, event.wav_path)
        if isinstance(event, GrammarChanged) and self._running and not self._stopping:
            self._loop.call_soon_threadsafe(self._schedule_grammar_update, event)

    def _schedule_grammar_update(self, event: GrammarChanged) -> None:
        """Coalesce: só a gramática mais recente é aplicada quando trocas chegam em rajada."""
        self._grammar_pending = event
        if self._grammar_task is None or self._grammar_task.done():
            self._grammar_task = self._loop.create_task(self._drain_grammar_updates())

    async def _drain_grammar_updates(self) -> None:
        while self._grammar_pending is not None and not self._stopping:
            event, self._grammar_pending = self._grammar_pending, None
            await self._loop.run_in_executor(self._executor, self._apply_grammar, event)

    def _apply_grammar(self, event: GrammarChanged) -> None:
        update = getattr(self._interpreter, "update_grammar", None)
        if callable(update):
            try:
                update(event.grammar, stale=event.stale, pointer=event.pointer_new)
            except Exception as exc:
                self._log("error", f"interpretador rejeitou a gramática nova: {exc}")
        retune = getattr(self._transcriber, "set_grammar", None)
        if callable(retune) and event.grammar:
            try:
                retune(event.grammar)
            except Exception as exc:
                self._log("error", f"não consegui ajustar a âncora do STT à cena: {exc}")
        for listener in self._grammar_listeners_snapshot():
            try:
                listener(event)
            except Exception as exc:
                self._log("error", f"ouvinte de gramática falhou: {exc}")

    def _grammar_listeners_snapshot(self) -> tuple[GrammarListener, ...]:
        with self._lock:
            return tuple(self._grammar_listeners)

    async def _settle_grammar_task(self) -> None:
        task = self._grammar_task
        if task is None or task.done():
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            if asyncio.current_task().cancelling():
                raise

    def _release_resources(self) -> None:
        if self._telemetry is not None:
            self._telemetry.close()
        if self._owns_executor:
            self._executor.shutdown(wait=False, cancel_futures=True)

    def _phase(self, phase: ListeningPhase, utt_id: str) -> None:
        self._publish(PhaseChanged(phase=phase, utt_id=utt_id))

    def _log(self, level: str, message: str) -> None:
        getattr(log, level)(message)
        self._publish(LogLine(level=level, message=message, source=LOG_SOURCE))

    def _publish(self, event: Event) -> None:
        self._bus.publish(event)


def _running_loop_or_none() -> asyncio.AbstractEventLoop | None:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None
