"""ViewModel da interface: projeta os eventos do barramento e a
config num ``UiState`` serializável e expõe os comandos que a View dispara.
Não conhece HTML nem HTTP."""

from __future__ import annotations

import copy
import logging
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Protocol

from cyhmo.config.loader import ConfigStore
from cyhmo.config.schema import AppConfig
from cyhmo.config.template import CONFIG_TEMPLATE
from cyhmo.domain.contracts import InjectResult, Interpretation
from cyhmo.domain.errors import ConfigError, CyhmoError
from cyhmo.domain.events import (
    AudioLevel,
    ComponentChanged,
    ConfigChanged,
    Event,
    GrammarChanged,
    InjectionDone,
    InterpretationReady,
    LogLine,
    PhaseChanged,
    StateChanged,
    TranscriptReady,
    UtteranceCaptured,
    UtteranceFinished,
)
from cyhmo.pipeline.bus import EventBus

log = logging.getLogger("cyhmo.ui")

RESTART_SECTIONS = frozenset({"audio", "activation", "stt", "languages", "pine", "state"})
RESTART_INTENT_FIELDS = frozenset({"embedding_backend", "embedding_model", "embedding_cache", "annex"})
DEFAULT_GRAMMAR_LIMIT = 500

_TEMPLATE_LINE = re.compile(r"^\s*\w+\s*=\s*\{([a-z_.]+)\}\s*(?:#\s*(.*?))?\s*$")


class UiServices(Protocol):
    """Serviços que o integrador entrega à ViewModel (ações sobre as camadas)."""

    def list_devices(self) -> list[dict[str, Any]]: ...

    def available_language_packs(self) -> list[dict[str, Any]]: ...

    def language_registry(self) -> list[dict[str, Any]]: ...

    def install_language_pack(self, code: str) -> dict[str, Any]: ...

    def llm_status(self) -> dict[str, Any]: ...

    def llm_pull(self, model: str) -> dict[str, Any]: ...

    def llm_pull_cancel(self) -> dict[str, Any]: ...

    def llm_delete_model(self, model: str) -> dict[str, Any]: ...

    def calibration_status(self) -> dict[str, Any]: ...

    def calibration_start(self, dataset: str, spontaneous: str, grid: str) -> dict[str, Any]: ...

    def calibration_cancel(self) -> dict[str, Any]: ...

    def start_listening(self) -> None: ...

    def stop_listening(self) -> None: ...

    def is_listening(self) -> bool: ...

    def request_restart(self) -> None: ...

    def request_exit(self) -> None: ...

    def update_status(self) -> dict[str, Any]: ...

    def update_check(self) -> dict[str, Any]: ...

    def update_install(self) -> dict[str, Any]: ...

    def update_skip(self) -> dict[str, Any]: ...

    def update_postpone(self) -> dict[str, Any]: ...

    def interpret_text(self, text: str) -> Interpretation: ...

    def inject_text(self, text: str) -> InjectResult: ...

    def mic_test(self, seconds: float) -> dict[str, Any]: ...

    def capture_hotkey(self, timeout_s: float) -> str | None: ...

    def grammar_entries(self) -> list[str]: ...

    def injector_status(self) -> dict[str, Any]: ...


@dataclass
class UtteranceRecord:
    """Agregado de um enunciado em curso, montado a partir dos eventos com o mesmo ``utt_id``."""

    utt_id: str
    t_start: float
    source: str = "mic"
    duration_ms: float = 0.0
    wav_path: str | None = None
    transcript: dict[str, Any] | None = None
    interpretation: dict[str, Any] | None = None
    state: dict[str, Any] | None = None
    injection: dict[str, Any] | None = None
    latency_ms: dict[str, float] = field(default_factory=dict)
    status: str | None = None
    error: str | None = None
    finished: bool = False

    @property
    def text(self) -> str:
        return "" if self.transcript is None else str(self.transcript.get("text", ""))

    @property
    def keys(self) -> list[str]:
        if self.interpretation is None:
            return []
        return [command["key"] for command in self.interpretation.get("commands", [])]

    def total_ms(self) -> float:
        if "total" in self.latency_ms:
            return self.latency_ms["total"]
        return sum(self.latency_ms.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "utt_id": self.utt_id,
            "t_start": self.t_start,
            "source": self.source,
            "duration_ms": self.duration_ms,
            "wav_path": self.wav_path,
            "transcript": self.transcript,
            "interpretation": self.interpretation,
            "state": self.state,
            "injection": self.injection,
            "latency_ms": dict(self.latency_ms),
            "total_ms": round(self.total_ms(), 2),
            "status": self.status,
            "error": self.error,
            "finished": self.finished,
        }

    def to_history_row(self, budget_ms: dict[str, int]) -> dict[str, Any]:
        interpretation = self.interpretation or {}
        return {
            "utt_id": self.utt_id,
            "t": self.t_start,
            "status": self.status,
            "text": self.text,
            "keys": self.keys,
            "method": interpretation.get("method", "none"),
            "confidence": interpretation.get("confidence", 0.0),
            "total_ms": round(self.total_ms(), 2),
            "over_budget": is_over_budget(self.latency_ms, budget_ms),
            "detail": self.to_dict(),
        }


@dataclass
class UiState:
    session_id: str
    started_at: float
    budget_ms: dict[str, int]
    phase: str = "idle"
    utt_id: str | None = None
    components: dict[str, dict[str, str]] = field(default_factory=dict)
    level: dict[str, float] = field(default_factory=lambda: {"rms": 0.0, "peak": 0.0})
    last_utterance: UtteranceRecord | None = None
    history: deque[dict[str, Any]] = field(default_factory=deque)
    grammar: dict[str, Any] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)
    console: deque[dict[str, Any]] = field(default_factory=deque)
    config: dict[str, Any] = field(default_factory=dict)
    config_schema: dict[str, Any] = field(default_factory=dict)
    languages_available: list[dict[str, Any]] = field(default_factory=list)
    devices: list[dict[str, Any]] = field(default_factory=list)
    update: dict[str, Any] = field(default_factory=dict)
    listening: bool = False
    restart_pending: bool = False

    def languages_view(self) -> dict[str, Any]:
        languages = self.config.get("languages", {})
        return {
            "available": copy.deepcopy(self.languages_available),
            "enabled": list(languages.get("enabled", [])),
            "primary": languages.get("primary"),
        }

    def to_dict(self, dropped_events: int) -> dict[str, Any]:
        return {
            "mode": self.config.get("ui", {}).get("mode", "basic"),
            "restart_pending": self.restart_pending,
            "phase": self.phase,
            "utt_id": self.utt_id,
            "components": copy.deepcopy(self.components),
            "level": dict(self.level),
            "last_utterance": None if self.last_utterance is None else self.last_utterance.to_dict(),
            "history": copy.deepcopy(list(self.history)),
            "grammar": dict(self.grammar),
            "state": copy.deepcopy(self.state),
            "console": list(self.console),
            "config": copy.deepcopy(self.config),
            "config_schema": self.config_schema,
            "languages": self.languages_view(),
            "devices": copy.deepcopy(self.devices),
            "update": dict(self.update),
            "listening": self.listening,
            "budget_ms": dict(self.budget_ms),
            "session_id": self.session_id,
            "started_at": self.started_at,
            "dropped_events": dropped_events,
        }


def empty_grammar_view() -> dict[str, Any]:
    return {
        "size": 0,
        "stale": True,
        "pointer": None,
        "blob_address": None,
        "changed_at": None,
        "version": 0,
        "new_in_session": 0,
    }


def is_over_budget(latency_ms: dict[str, float], budget_ms: dict[str, int]) -> bool:
    total_budget = budget_ms.get("total")
    if total_budget is not None:
        total = latency_ms.get("total", sum(latency_ms.values()))
        return total > total_budget
    return any(latency_ms.get(stage, 0.0) > limit for stage, limit in budget_ms.items())


def restart_required_for(patch: dict[str, Any]) -> bool:
    if any(section in RESTART_SECTIONS for section in patch):
        return True
    intent_patch = patch.get("intent")
    return isinstance(intent_patch, dict) and any(key in RESTART_INTENT_FIELDS for key in intent_patch)


def template_descriptions() -> dict[str, str]:
    """As descrições dos campos vivem só nos comentários do modelo do ``config.toml``."""
    descriptions: dict[str, str] = {}
    for line in CONFIG_TEMPLATE.splitlines():
        match = _TEMPLATE_LINE.match(line)
        if match:
            descriptions[match.group(1)] = match.group(2) or ""
    return descriptions


def reduce_config_schema(schema: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """Achata o JSON Schema do pydantic em seções pontilhadas com só o que a View precisa."""
    schema = schema or AppConfig.model_json_schema()
    definitions = schema.get("$defs", {})
    descriptions = template_descriptions()
    sections: dict[str, dict[str, Any]] = {}

    def walk(properties: dict[str, Any], prefix: str) -> None:
        for name, prop in properties.items():
            dotted = f"{prefix}{name}"
            reference = prop.get("$ref")
            if reference:
                walk(definitions[reference.rsplit("/", 1)[-1]].get("properties", {}), f"{dotted}.")
                continue
            section, _, key = dotted.rpartition(".")
            sections.setdefault(section, {})[key] = _reduce_field(prop, descriptions.get(dotted, ""))

    walk(schema.get("properties", {}), "")
    sections.pop("", None)
    return sections


def _reduce_field(prop: dict[str, Any], description: str) -> dict[str, Any]:
    field_type = prop.get("type")
    if field_type is None and "anyOf" in prop:
        options = [option.get("type") for option in prop["anyOf"] if option.get("type") != "null"]
        field_type = options[0] if options else "string"
    reduced: dict[str, Any] = {
        "type": field_type or "string",
        "default": prop.get("default"),
        "description": description,
    }
    if "enum" in prop:
        reduced["enum"] = list(prop["enum"])
    minimum = prop.get("minimum", prop.get("exclusiveMinimum"))
    maximum = prop.get("maximum", prop.get("exclusiveMaximum"))
    if minimum is not None:
        reduced["min"] = minimum
    if maximum is not None:
        reduced["max"] = maximum
    if field_type == "array":
        reduced["items"] = prop.get("items", {}).get("type", "string")
    return reduced


class UiViewModel:
    def __init__(
        self,
        bus: EventBus,
        config_store: ConfigStore,
        services: UiServices,
        session_id: str,
        budget_ms: dict[str, int],
        history_size: int = 50,
        console_size: int = 200,
    ) -> None:
        self._bus = bus
        self._config_store = config_store
        self._services = services
        self._lock = threading.RLock()
        self._hotkey_capture = threading.Lock()
        self._grammar_entries: tuple[str, ...] = ()
        self._state = UiState(
            session_id=session_id,
            started_at=time.time(),
            budget_ms=dict(budget_ms),
            history=deque(maxlen=history_size),
            console=deque(maxlen=console_size),
            grammar=empty_grammar_view(),
            config=config_store.config.model_dump(mode="json"),
            config_schema=reduce_config_schema(),
        )
        self._handlers: dict[str, Callable[[Any], None]] = {
            PhaseChanged.kind: self._on_phase,
            UtteranceCaptured.kind: self._on_utterance,
            TranscriptReady.kind: self._on_transcript,
            InterpretationReady.kind: self._on_interpretation,
            InjectionDone.kind: self._on_injection,
            UtteranceFinished.kind: self._on_finished,
            GrammarChanged.kind: self._on_grammar,
            StateChanged.kind: self._on_state,
            ComponentChanged.kind: self._on_component,
            AudioLevel.kind: self._on_level,
        }
        self._unsubscribe = bus.subscribe(self.handle_event)
        config_store.subscribe(self._on_config_replaced)
        self._load_language_packs()

    @property
    def bus(self) -> EventBus:
        return self._bus

    def close(self) -> None:
        self._unsubscribe()

    def snapshot(self, dropped_events: int = 0) -> dict[str, Any]:
        listening = self._safe_call("is_listening", self._services.is_listening, default=None)
        update = self._safe_call("update_status", self._services.update_status, default=None)
        with self._lock:
            if listening is not None:
                self._state.listening = bool(listening)
            if update is not None:
                self._state.update = dict(update)
            return self._state.to_dict(dropped_events)

    def handle_event(self, event: Event) -> None:
        handler = self._handlers.get(event.kind)
        try:
            with self._lock:
                if handler is not None:
                    handler(event)
                if not isinstance(event, AudioLevel):
                    self._append_console(event)
        except Exception:
            log.exception("ViewModel falhou ao projetar o evento %s", event.kind)

    def apply_config_patch(self, patch: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(patch, dict) or not patch:
            raise ConfigError("o patch de configuração deve ser um objeto com ao menos uma seção")
        new_config = self._config_store.update(patch)
        sections = list(patch.keys())
        restart_required = restart_required_for(patch)
        with self._lock:
            self._state.config = new_config.model_dump(mode="json")
            self._state.restart_pending = self._state.restart_pending or restart_required
        self._bus.publish(ConfigChanged(sections=tuple(sections), restart_required=restart_required))
        return {
            "config": new_config.model_dump(mode="json"),
            "restart_required": restart_required,
            "sections": sections,
        }

    def refresh_devices(self) -> list[dict[str, Any]]:
        devices = self._safe_call("list_devices", self._services.list_devices, [])
        with self._lock:
            self._state.devices = list(devices)
        return list(devices)

    def devices(self) -> list[dict[str, Any]]:
        """Carga preguiçosa: quem abre a interface encontra a lista pronta, sem pedir atualização."""
        with self._lock:
            cached = list(self._state.devices)
        return cached or self.refresh_devices()

    def refresh_languages(self) -> dict[str, Any]:
        self._load_language_packs(raise_errors=True)
        with self._lock:
            return self._state.languages_view()

    def i18n_bundle(self) -> dict[str, Any]:
        from cyhmo.ui.i18n import bundle

        config = self._config_store.config
        return bundle(config.ui.language, config.languages.primary)

    def language_registry(self) -> dict[str, Any]:
        """Sem rede, a lista vem vazia com o motivo — a interface nunca fica em branco sem explicação."""
        installed = {pack["code"] for pack in self._services.available_language_packs() if pack.get("valid", True)}
        try:
            entries = list(self._services.language_registry())
        except CyhmoError as exc:
            return {"entries": [], "error": str(exc)}
        for entry in entries:
            entry["installed"] = entry.get("code") in installed
        return {"entries": entries, "error": ""}

    def install_language_pack(self, code: str) -> dict[str, Any]:
        code = code.strip()
        if not code:
            raise CyhmoError("informe o código do idioma a instalar")
        result = self._services.install_language_pack(code)
        self._load_language_packs()
        self._log(f"pacote de idioma instalado: {code}")
        with self._lock:
            return {**result, "languages": self._state.languages_view()}

    def llm_status(self) -> dict[str, Any]:
        return dict(self._services.llm_status())

    def llm_pull(self, model: str) -> dict[str, Any]:
        model = model.strip()
        if not model:
            raise CyhmoError("informe o nome do modelo a baixar")
        self._log(f"baixando modelo do assistente: {model}")
        return dict(self._services.llm_pull(model))

    def llm_pull_cancel(self) -> dict[str, Any]:
        return dict(self._services.llm_pull_cancel())

    def llm_delete_model(self, model: str) -> dict[str, Any]:
        model = model.strip()
        if not model:
            raise CyhmoError("informe o modelo a remover")
        result = dict(self._services.llm_delete_model(model))
        self._log(f"modelo do assistente removido: {model}")
        return result

    def calibration_status(self) -> dict[str, Any]:
        return dict(self._services.calibration_status())

    def calibration_start(self, dataset: str, spontaneous: str, grid: str) -> dict[str, Any]:
        self._log("calibração iniciada pela interface")
        return dict(self._services.calibration_start(dataset, spontaneous, grid))

    def calibration_cancel(self) -> dict[str, Any]:
        return dict(self._services.calibration_cancel())

    def start_listening(self) -> bool:
        self._services.start_listening()
        return self._sync_listening(True)

    def stop_listening(self) -> bool:
        self._services.stop_listening()
        return self._sync_listening(False)

    def request_restart(self) -> dict[str, Any]:
        self._services.request_restart()
        return {"action": "restart"}

    def request_exit(self) -> dict[str, Any]:
        self._services.request_exit()
        return {"action": "exit"}

    def update_status(self) -> dict[str, Any]:
        return dict(self._services.update_status())

    def update_check(self) -> dict[str, Any]:
        self._log("procurando atualizações")
        return dict(self._services.update_check())

    def update_install(self) -> dict[str, Any]:
        """A troca dos arquivos é do encerramento: aqui só começa o download."""
        return dict(self._services.update_install())

    def update_skip(self) -> dict[str, Any]:
        return dict(self._services.update_skip())

    def update_postpone(self) -> dict[str, Any]:
        return dict(self._services.update_postpone())

    def interpret_text(self, text: str) -> dict[str, Any]:
        return self._services.interpret_text(text).to_dict()

    def inject_text(self, text: str) -> dict[str, Any]:
        return self._services.inject_text(text).to_dict()

    def inject_commands(self, commands: list[dict[str, Any]]) -> dict[str, Any]:
        """Sem ``inject_commands`` nos serviços, a pilha é injetada um comando por vez."""
        keys = [str(command.get("key", "")).strip() for command in commands]
        keys = [key for key in keys if key]
        if not keys:
            raise CyhmoError("nenhum comando com 'key' preenchida para injetar")
        stack_injector = getattr(self._services, "inject_commands", None)
        if callable(stack_injector):
            return stack_injector(commands).to_dict()
        results = [self._services.inject_text(key).to_dict() for key in keys]
        failures = [result["error"] for result in results if not result["ok"]]
        return {
            "ok": not failures,
            "latency_ms": round(sum(result["latency_ms"] for result in results), 2),
            "error": failures[0] if failures else None,
            "matched": all(result.get("matched") is True for result in results),
            "payload_echo": {"keys": keys, "results": results},
        }

    def mic_test(self, seconds: float) -> dict[str, Any]:
        return dict(self._services.mic_test(seconds))

    def capture_hotkey(self, timeout_s: float) -> dict[str, Any]:
        """Só descobre a tecla — quem grava é ``apply_config_patch``. Uma captura por vez:
        duas em paralelo instalariam hooks concorrentes sobre a mesma tecla."""
        if not self._hotkey_capture.acquire(blocking=False):
            raise CyhmoError("já existe uma gravação de tecla em andamento; espere ela terminar")
        try:
            key = self._services.capture_hotkey(timeout_s)
        finally:
            self._hotkey_capture.release()
        if key is None:
            self._log(f"nenhuma tecla pressionada em {timeout_s:.0f} s; gravação cancelada")
            return {"key": "", "accepted": False, "reason": "timeout"}
        self._log(f"tecla capturada para o push-to-talk: {key}")
        return {"key": key, "accepted": True}

    def grammar_entries(self, filter: str = "", limit: int = DEFAULT_GRAMMAR_LIMIT) -> list[str]:
        entries = self._safe_call("grammar_entries", self._services.grammar_entries, default=None)
        if not entries:
            with self._lock:
                entries = list(self._grammar_entries)
        needle = filter.strip().casefold()
        matches = [entry for entry in entries if needle in entry.casefold()] if needle else list(entries)
        return matches[: max(0, limit)]

    def injector_status(self) -> dict[str, Any]:
        return dict(self._services.injector_status())

    def _sync_listening(self, fallback: bool) -> bool:
        listening = self._safe_call("is_listening", self._services.is_listening, default=fallback)
        with self._lock:
            self._state.listening = bool(listening)
        return bool(listening)

    def _load_language_packs(self, raise_errors: bool = False) -> None:
        try:
            packs = list(self._services.available_language_packs())
        except Exception as exc:
            if raise_errors:
                raise
            log.warning("não foi possível listar os pacotes de idioma: %s", exc)
            return
        with self._lock:
            self._state.languages_available = packs

    def _log(self, message: str, level: Literal["debug", "info", "warning", "error"] = "info") -> None:
        self._bus.publish(LogLine(level=level, message=message, source="ui"))

    def _safe_call(self, name: str, call: Callable[[], Any], default: Any) -> Any:
        try:
            return call()
        except Exception as exc:
            log.warning("serviço %s falhou: %s", name, exc)
            return default

    def _on_config_replaced(self, _old: AppConfig, new: AppConfig) -> None:
        with self._lock:
            self._state.config = new.model_dump(mode="json")

    def _utterance_for(self, utt_id: str, t: float) -> UtteranceRecord:
        current = self._state.last_utterance
        if current is None or current.utt_id != utt_id or current.finished:
            current = UtteranceRecord(utt_id=utt_id, t_start=t)
            self._state.last_utterance = current
            self._state.utt_id = utt_id
        return current

    def _on_phase(self, event: PhaseChanged) -> None:
        """A volta a idle carrega o utt_id recém-encerrado: criar registro aqui
        apagaria o resultado do Painel no instante em que ele aparece."""
        self._state.phase = event.phase
        if event.utt_id and event.phase != "idle":
            self._utterance_for(event.utt_id, event.t)

    def _on_utterance(self, event: UtteranceCaptured) -> None:
        record = self._utterance_for(event.utt_id, event.t)
        record.duration_ms = event.duration_ms
        record.source = event.source
        record.wav_path = event.wav_path

    def _on_transcript(self, event: TranscriptReady) -> None:
        record = self._utterance_for(event.utt_id, event.t)
        record.transcript = None if event.transcript is None else event.transcript.to_dict()
        record.latency_ms["stt"] = event.latency_ms

    def _on_interpretation(self, event: InterpretationReady) -> None:
        record = self._utterance_for(event.utt_id, event.t)
        if event.interpretation is not None:
            record.interpretation = event.interpretation.to_dict()
            record.latency_ms["intent"] = event.interpretation.latency_ms
        if event.state is not None:
            record.state = event.state.to_dict()

    def _on_injection(self, event: InjectionDone) -> None:
        record = self._utterance_for(event.utt_id, event.t)
        record.injection = {
            "commands": [command.to_dict() for command in event.commands],
            "result": None if event.result is None else event.result.to_dict(),
        }
        if event.result is not None:
            record.latency_ms["inject"] = event.result.latency_ms

    def _on_finished(self, event: UtteranceFinished) -> None:
        record = self._utterance_for(event.utt_id, event.t)
        record.latency_ms.update(event.latency_ms)
        record.status = event.status
        record.error = event.error
        record.finished = True
        self._state.history.appendleft(record.to_history_row(self._state.budget_ms))
        self._state.utt_id = None

    def _on_grammar(self, event: GrammarChanged) -> None:
        grammar = self._state.grammar
        grammar.update(
            {
                "size": event.size,
                "stale": event.stale,
                "pointer": event.pointer_new,
                "blob_address": event.blob_address,
                "changed_at": event.t,
                "version": grammar.get("version", 0) + 1,
                "new_in_session": event.new_in_session,
            }
        )
        if event.grammar:
            self._grammar_entries = tuple(event.grammar)

    def _on_state(self, event: StateChanged) -> None:
        if event.state is not None:
            self._state.state = event.state.to_dict()

    def _on_component(self, event: ComponentChanged) -> None:
        self._state.components[event.component] = {"status": event.status, "detail": event.detail}

    def _on_level(self, event: AudioLevel) -> None:
        self._state.level = {"rms": event.rms, "peak": event.peak}

    def _append_console(self, event: Event) -> None:
        if isinstance(event, LogLine):
            line = {"t": event.t, "level": event.level, "source": event.source, "message": event.message}
        else:
            line = {"t": event.t, "level": "debug", "source": event.kind, "message": summarize_event(event)}
        self._state.console.append(line)


def summarize_event(event: Event) -> str:
    summarizer = _SUMMARIZERS.get(event.kind)
    return summarizer(event) if summarizer else event.kind


def _summarize_phase(event: PhaseChanged) -> str:
    return f"fase: {event.phase}" + (f" (enunciado {event.utt_id})" if event.utt_id else "")


def _summarize_utterance(event: UtteranceCaptured) -> str:
    return f"enunciado {event.utt_id} capturado: {event.duration_ms:.0f} ms via {event.source}"


def _summarize_transcript(event: TranscriptReady) -> str:
    if event.transcript is None:
        return f"transcrição {event.utt_id}: vazia ({event.latency_ms:.0f} ms)"
    transcript = event.transcript
    return (
        f"transcrição {event.utt_id}: {transcript.text!r} "
        f"[{transcript.lang}, conf. {transcript.confidence:.2f}] em {event.latency_ms:.0f} ms"
    )


def _summarize_interpretation(event: InterpretationReady) -> str:
    if event.interpretation is None:
        return f"interpretação {event.utt_id}: nenhuma"
    interpretation = event.interpretation
    keys = ", ".join(interpretation.keys) or "(nenhum comando)"
    detail = f"{interpretation.method}, conf. {interpretation.confidence:.2f}"
    if interpretation.reason:
        detail += f", motivo: {interpretation.reason}"
    return f"interpretação {event.utt_id}: {keys} ({detail})"


def _summarize_injection(event: InjectionDone) -> str:
    keys = ", ".join(command.key for command in event.commands) or "(vazio)"
    if event.result is None:
        return f"injeção {event.utt_id}: {keys} (sem resultado)"
    verdict = "ok" if event.result.ok else f"falhou: {event.result.error}"
    return f"injeção {event.utt_id}: {keys} → {verdict} em {event.result.latency_ms:.0f} ms"


def _summarize_finished(event: UtteranceFinished) -> str:
    total = event.latency_ms.get("total", sum(event.latency_ms.values()))
    message = f"enunciado {event.utt_id} encerrado: {event.status} (total {total:.0f} ms)"
    return message + (f" — {event.error}" if event.error else "")


def _summarize_grammar(event: GrammarChanged) -> str:
    pointer = "?" if event.pointer_new is None else f"0x{event.pointer_new:08X}"
    stale = ", stale" if event.stale else ""
    return f"gramática trocada: {event.size} entradas (ponteiro {pointer}, {event.new_in_session} inéditas{stale})"


def _summarize_state(event: StateChanged) -> str:
    if event.state is None:
        return "estado do jogo: desconhecido"
    return f"estado do jogo: modo={event.state.mode}, can_talk={event.state.can_talk}"


def _summarize_component(event: ComponentChanged) -> str:
    return f"componente {event.component}: {event.status}" + (f" — {event.detail}" if event.detail else "")


def _summarize_config(event: ConfigChanged) -> str:
    sections = ", ".join(event.sections) or "(nenhuma seção)"
    suffix = " (requer reinício)" if event.restart_required else ""
    return f"configuração alterada: {sections}{suffix}"


_SUMMARIZERS: dict[str, Callable[[Any], str]] = {
    PhaseChanged.kind: _summarize_phase,
    UtteranceCaptured.kind: _summarize_utterance,
    TranscriptReady.kind: _summarize_transcript,
    InterpretationReady.kind: _summarize_interpretation,
    InjectionDone.kind: _summarize_injection,
    UtteranceFinished.kind: _summarize_finished,
    GrammarChanged.kind: _summarize_grammar,
    StateChanged.kind: _summarize_state,
    ComponentChanged.kind: _summarize_component,
    ConfigChanged.kind: _summarize_config,
}
