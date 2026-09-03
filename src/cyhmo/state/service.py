"""Camada 3 — serviço de estado de jogo.

Polling barato do ponteiro de gramática; releitura do blob na própria thread
de polling, fora do caminho crítico; ``read_state()`` só devolve o cache.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any, Callable

from cyhmo.config.schema import AppConfig, ProjectPaths, StateConfig
from cyhmo.domain.contracts import GameState
from cyhmo.domain.errors import GrammarUnavailableError
from cyhmo.domain.events import ComponentChanged, Event, GrammarChanged, LogLine, StateChanged
from cyhmo.domain.ports import EventSink
from cyhmo.inject.pine import PineClient
from cyhmo.inject.recipe import WriteRecipe
from cyhmo.pipeline.bus import NullSink
from cyhmo.state.grammar import normalize_entry
from cyhmo.state.grammar_source import (
    EmptyGrammarSource,
    FileGrammarSource,
    GrammarSnapshot,
    GrammarSource,
    PineGrammarSource,
    SeededGrammarSource,
)
from cyhmo.state.mode import MODE_UNKNOWN, ModeInference, infer_mode

Clock = Callable[[], float]

LOG_SOURCE = "state"
STALE_AFTER_MS = 200.0
TICK_BUDGET_MS = 5.0
TICK_SAMPLES = 256
RELOAD_RETRY_S = 2.0
STOP_TIMEOUT_S = 1.0
REJECTED_PREVIEW = 8
ROUTE_LABELS = {
    "records": "tabela do blob",
    "scan": "varredura por bytes (sem tabela legível)",
    "file": "arquivo",
}

_UNSET = object()

log = logging.getLogger("cyhmo.state.service")


class GameStateService:
    """Implementa ``ports.GameStateReader``; ``accepts(key)`` serve de gate de gramática."""

    def __init__(
        self,
        config: StateConfig,
        source: GrammarSource,
        bus: EventSink | None = None,
        clock: Clock = time.perf_counter,
    ) -> None:
        self._config = config
        self._source = source
        self._bus: EventSink = bus if bus is not None else NullSink()
        self._clock = clock
        self._period_s = 1.0 / config.polling_hz
        self._cache_lock = threading.Lock()
        self._tick_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._pointer: Any = _UNSET
        self._grammar: GrammarSnapshot | None = None
        self._normalized: frozenset[str] | None = None
        self._grammar_stale = False
        self._mode = ModeInference(MODE_UNKNOWN, ())
        self._version = 0
        self._discarded = 0
        self._seen: set[str] = set()
        self._last_tick_ts: float | None = None
        self._tick_costs: deque[float] = deque(maxlen=TICK_SAMPLES)
        self._tick_p95_ms = 0.0
        self._budget_warned = False
        self._retry_at: float | None = None
        self._reported_unreadable: int | None = None

    def start(self) -> None:
        """Thread que sobreviveu a um ``stop()`` é readotada limpando o evento de parada.
        Só voltar aqui a deixaria sair no tique seguinte e o serviço ficaria sem polling —
        o estado congelado que esta camada existe para não ter."""
        if self._thread is not None and self._thread.is_alive():
            self._stop_event.clear()
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="cyhmo-state", daemon=True)
        self._thread.start()
        self._publish(ComponentChanged(component="state", status="ready", detail=f"polling a {self._config.polling_hz} Hz"))

    def stop(self) -> None:
        """Thread que não morreu continua sendo a thread desta instância: esquecê-la deixaria
        ``start()`` abrir uma segunda, e as duas disputariam o mesmo socket PINE."""
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=STOP_TIMEOUT_S)
            if thread.is_alive():
                log.warning("a thread de estado não encerrou em %.0f s", STOP_TIMEOUT_S)
            else:
                self._thread = None
        self._publish(ComponentChanged(component="state", status="off"))

    def refresh_now(self) -> None:
        self._safe_tick()

    def read_state(self) -> GameState:
        with self._cache_lock:
            grammar = self._grammar
            stale = self._grammar_stale
            mode = self._mode
            version = self._version
            pointer = None if self._pointer is _UNSET else self._pointer
            last_tick = self._last_tick_ts
            discarded = self._discarded
            tick_p95 = self._tick_p95_ms
        age_ms = None if last_tick is None else (self._clock() - last_tick) * 1000.0
        raw = {
            "ts": last_tick,
            "age_ms": age_ms,
            "stale": age_ms is None or age_ms > STALE_AFTER_MS,
            "pointer": pointer,
            "blob_address": None if grammar is None else grammar.blob_address,
            "mode_source": "grammar" if mode.witnesses else "none",
            "mode_witnesses": list(mode.witnesses),
            "grammar_version": version,
            "grammar_stale": stale,
            "discarded": discarded,
            "tick_p95_ms": round(tick_p95, 3),
            "can_talk_source": "unmapped",
        }
        return GameState(
            mode=mode.mode,
            can_talk=None,
            grammar=None if grammar is None else grammar.entries,
            grammar_stale=stale,
            raw=raw,
        )

    def accepts(self, key: str) -> bool:
        """Sem gramática conhecida deixa passar (fail-open): bloquear deixaria o mod mudo."""
        with self._cache_lock:
            normalized = self._normalized
        return normalized is None or normalize_entry(key) in normalized

    def _run(self) -> None:
        self._safe_tick()
        while not self._stop_event.wait(self._period_s):
            self._safe_tick()

    def _safe_tick(self) -> None:
        """Nenhum erro do tique derruba a thread de polling nem chega a ``read_state()``."""
        with self._tick_lock:
            try:
                self._tick()
            except Exception as exc:
                log.exception("tique do estado falhou")
                self._publish(LogLine(level="error", message=f"tique do estado falhou: {exc}", source=LOG_SOURCE))
                self._publish(ComponentChanged(component="state", status="error", detail=str(exc)))

    def _tick(self) -> None:
        started = self._clock()
        pointer = self._source.read_pointer()
        changed = pointer != self._pointer
        with self._cache_lock:
            self._last_tick_ts = started
        self._record_tick_cost((self._clock() - started) * 1000.0)
        if changed:
            self._on_pointer_changed(pointer)
        elif pointer is not None and self._grammar_stale and self._retry_due(started):
            self._reload(pointer, pointer)

    def _on_pointer_changed(self, pointer: int | None) -> None:
        previous = None if self._pointer is _UNSET else self._pointer
        with self._cache_lock:
            self._pointer = pointer
            self._grammar_stale = True
        if pointer is None:
            self._discarded += 1
            self._publish(
                LogLine(
                    level="warning",
                    message="ponteiro da gramática ilegível (PCSX2 fechado ou PINE fora do ar); servindo a última gramática conhecida",
                    source=LOG_SOURCE,
                )
            )
            self._publish(ComponentChanged(component="state", status="error", detail="sem leitura do ponteiro"))
            self._publish(StateChanged(state=self.read_state()))
            return
        self._publish(ComponentChanged(component="state", status="busy", detail="relendo gramática"))
        self._reload(previous, pointer)

    def _reload(self, previous: int | None, pointer: int) -> None:
        started = self._clock()
        try:
            snapshot = self._source.read(pointer)
        except Exception as exc:
            log.exception("fonte de gramática falhou")
            self._publish(LogLine(level="error", message=f"fonte de gramática falhou: {exc}", source=LOG_SOURCE))
            snapshot = None
        elapsed_ms = (self._clock() - started) * 1000.0
        if snapshot is None or not snapshot.entries:
            self._keep_previous(pointer, elapsed_ms)
            return
        self._adopt(previous, pointer, snapshot, elapsed_ms)

    def _keep_previous(self, pointer: int, elapsed_ms: float) -> None:
        """Menu, tela de título e carregamento não têm gramática nenhuma, e o ponteiro fica
        parado ali por minutos. Sem o corte por repetição, a releitura de 2 s viraria um aviso
        de 'perdi a cena' a cada 2 s para uma cena que nunca existiu."""
        never_had_one = self._grammar is None
        repeated = self._reported_unreadable == pointer
        with self._cache_lock:
            self._discarded += 1
            self._grammar_stale = True
            self._retry_at = self._clock() + RELOAD_RETRY_S
        self._reported_unreadable = pointer
        if repeated:
            return
        message = (
            f"cena sem gramática no ponteiro 0x{pointer:08X} ({elapsed_ms:.0f} ms); "
            "o jogo ainda não expôs a lista de comandos"
            if never_had_one
            else f"gramática ilegível no ponteiro 0x{pointer:08X} ({elapsed_ms:.0f} ms); mantendo a anterior como stale"
        )
        self._publish(LogLine(level="warning", message=message, source=LOG_SOURCE))
        self._publish(StateChanged(state=self.read_state()))
        detail = "cena sem gramática" if never_had_one else "gramática stale"
        self._publish(ComponentChanged(component="state", status="ready", detail=detail))

    def _adopt(self, previous: int | None, pointer: int, snapshot: GrammarSnapshot, elapsed_ms: float) -> None:
        entries = snapshot.entries
        new_in_session = sum(1 for entry in entries if entry not in self._seen)
        self._seen.update(entries)
        mode = infer_mode(entries) if self._config.infer_mode_from_grammar else ModeInference(MODE_UNKNOWN, ())
        with self._cache_lock:
            self._grammar = snapshot
            self._normalized = frozenset(normalize_entry(entry) for entry in entries)
            self._grammar_stale = False
            self._mode = mode
            self._version += 1
            self._retry_at = None
        self._reported_unreadable = None
        self._publish(
            GrammarChanged(
                pointer_old=previous,
                pointer_new=pointer,
                blob_address=snapshot.blob_address,
                size=len(entries),
                new_in_session=new_in_session,
                elapsed_ms=elapsed_ms,
                stale=False,
                grammar=entries,
            )
        )
        self._publish(LogLine(level="info", message=self._describe_change(previous, pointer, snapshot, new_in_session, elapsed_ms), source=LOG_SOURCE))
        if snapshot.rejected:
            preview = ", ".join(repr(text) for text in snapshot.rejected[:REJECTED_PREVIEW])
            self._publish(LogLine(level="debug", message=f"filtro anti-ruído rejeitou {len(snapshot.rejected)} token(s): {preview}", source=LOG_SOURCE))
        self._publish(StateChanged(state=self.read_state()))
        self._publish(ComponentChanged(component="state", status="ready", detail=f"{len(entries)} comandos, modo {mode.mode}"))

    @staticmethod
    def _describe_change(previous: int | None, pointer: int, snapshot: GrammarSnapshot, new_in_session: int, elapsed_ms: float) -> str:
        old = "—" if previous is None else f"0x{previous:08X}"
        blob = "arquivo" if snapshot.blob_address is None else f"0x{snapshot.blob_address:08X}"
        return (
            f"gramática trocada: ponteiro {old} → 0x{pointer:08X}, blob {blob}, "
            f"{len(snapshot.entries)} comandos ({new_in_session} inéditos na sessão, "
            f"{len(snapshot.rejected)} rejeitados) em {elapsed_ms:.0f} ms "
            f"[{ROUTE_LABELS.get(snapshot.source, snapshot.source)}]"
        )

    def _retry_due(self, now: float) -> bool:
        return self._retry_at is not None and now >= self._retry_at

    def _record_tick_cost(self, cost_ms: float) -> None:
        self._tick_costs.append(cost_ms)
        ordered = sorted(self._tick_costs)
        p95 = ordered[int(0.95 * (len(ordered) - 1))]
        with self._cache_lock:
            self._tick_p95_ms = p95
        if p95 > TICK_BUDGET_MS and not self._budget_warned:
            self._budget_warned = True
            self._publish(LogLine(level="warning", message=f"tique do estado acima do orçamento: p95 = {p95:.1f} ms (> {TICK_BUDGET_MS:.0f} ms)", source=LOG_SOURCE))
        elif p95 <= TICK_BUDGET_MS:
            self._budget_warned = False

    def _publish(self, event: Event) -> None:
        self._bus.publish(event)


def build_state_service(
    config: AppConfig,
    paths: ProjectPaths,
    client: PineClient | None,
    recipe: WriteRecipe | None,
    bus: EventSink | None,
) -> GameStateService:
    seed = _load_seed(paths)
    if client is None or recipe is None:
        return GameStateService(config.state, seed or EmptyGrammarSource(), bus)
    live = PineGrammarSource(client, recipe.grammar)
    source: GrammarSource = SeededGrammarSource(live, seed) if seed is not None else live
    return GameStateService(config.state, source, bus)


def _load_seed(paths: ProjectPaths) -> FileGrammarSource | None:
    """Semente ausente não impede o mod de subir: a gramática de arquivo é cache local e
    não é distribuída, então a configuração padrão aponta para um arquivo que a maioria das
    cópias não tem."""
    if paths.grammar_seed is None:
        return None
    try:
        return FileGrammarSource(paths.grammar_seed)
    except GrammarUnavailableError as exc:
        log.warning("gramática semente ignorada: %s", exc)
        return None
