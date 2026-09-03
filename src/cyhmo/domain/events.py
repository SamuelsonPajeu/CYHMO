"""Eventos publicados no barramento interno, consumidos pela ViewModel.

Cada evento é imutável e vira JSON por ``to_dict()``; ``kind`` identifica o tipo.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

from cyhmo.domain.contracts import (
    CommandRef,
    GameState,
    InjectResult,
    Interpretation,
    Transcript,
    UtteranceSource,
    UtteranceStatus,
)

ComponentName = Literal["capture", "activation", "stt", "intent", "llm", "state", "pine", "inject", "ui"]
ComponentStatus = Literal["off", "loading", "ready", "busy", "error"]
ListeningPhase = Literal["idle", "listening", "transcribing", "interpreting", "injecting"]


@dataclass(frozen=True)
class Event:
    kind: ClassVar[str] = "event"
    t: float = field(default_factory=time.perf_counter, kw_only=True)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "t": self.t, **self._payload()}

    def _payload(self) -> dict[str, Any]:
        return {}


@dataclass(frozen=True)
class PhaseChanged(Event):
    kind: ClassVar[str] = "phase"
    phase: ListeningPhase = "idle"
    utt_id: str | None = None

    def _payload(self) -> dict[str, Any]:
        return {"phase": self.phase, "utt_id": self.utt_id}


@dataclass(frozen=True)
class UtteranceCaptured(Event):
    kind: ClassVar[str] = "utterance"
    utt_id: str = ""
    duration_ms: float = 0.0
    source: UtteranceSource = "mic"
    wav_path: str | None = None

    def _payload(self) -> dict[str, Any]:
        return {
            "utt_id": self.utt_id,
            "duration_ms": round(self.duration_ms, 1),
            "source": self.source,
            "wav_path": self.wav_path,
        }


@dataclass(frozen=True)
class TranscriptReady(Event):
    kind: ClassVar[str] = "transcript"
    utt_id: str = ""
    transcript: Transcript | None = None
    latency_ms: float = 0.0

    def _payload(self) -> dict[str, Any]:
        return {
            "utt_id": self.utt_id,
            "transcript": None if self.transcript is None else self.transcript.to_dict(),
            "latency_ms": round(self.latency_ms, 2),
        }


@dataclass(frozen=True)
class InterpretationReady(Event):
    kind: ClassVar[str] = "interpretation"
    utt_id: str = ""
    interpretation: Interpretation | None = None
    state: GameState | None = None

    def _payload(self) -> dict[str, Any]:
        return {
            "utt_id": self.utt_id,
            "interpretation": None if self.interpretation is None else self.interpretation.to_dict(),
            "state": None if self.state is None else self.state.to_dict(),
        }


@dataclass(frozen=True)
class InjectionDone(Event):
    kind: ClassVar[str] = "injection"
    utt_id: str = ""
    commands: tuple[CommandRef, ...] = ()
    result: InjectResult | None = None

    def _payload(self) -> dict[str, Any]:
        return {
            "utt_id": self.utt_id,
            "commands": [command.to_dict() for command in self.commands],
            "result": None if self.result is None else self.result.to_dict(),
        }


@dataclass(frozen=True)
class UtteranceFinished(Event):
    kind: ClassVar[str] = "finished"
    utt_id: str = ""
    status: UtteranceStatus = "no_command"
    latency_ms: dict[str, float] = field(default_factory=dict)
    error: str | None = None

    def _payload(self) -> dict[str, Any]:
        return {
            "utt_id": self.utt_id,
            "status": self.status,
            "latency_ms": {stage: round(value, 2) for stage, value in self.latency_ms.items()},
            "error": self.error,
        }


@dataclass(frozen=True)
class GrammarChanged(Event):
    kind: ClassVar[str] = "grammar"
    pointer_old: int | None = None
    pointer_new: int | None = None
    blob_address: int | None = None
    size: int = 0
    new_in_session: int = 0
    elapsed_ms: float = 0.0
    stale: bool = False
    grammar: tuple[str, ...] = ()

    def _payload(self) -> dict[str, Any]:
        return {
            "pointer_old": self.pointer_old,
            "pointer_new": self.pointer_new,
            "blob_address": self.blob_address,
            "size": self.size,
            "new_in_session": self.new_in_session,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "stale": self.stale,
            "grammar": list(self.grammar),
        }


@dataclass(frozen=True)
class StateChanged(Event):
    kind: ClassVar[str] = "state"
    state: GameState | None = None

    def _payload(self) -> dict[str, Any]:
        return {"state": None if self.state is None else self.state.to_dict()}


@dataclass(frozen=True)
class ComponentChanged(Event):
    kind: ClassVar[str] = "component"
    component: ComponentName = "ui"
    status: ComponentStatus = "off"
    detail: str = ""

    def _payload(self) -> dict[str, Any]:
        return {"component": self.component, "status": self.status, "detail": self.detail}


@dataclass(frozen=True)
class AudioLevel(Event):
    kind: ClassVar[str] = "level"
    rms: float = 0.0
    peak: float = 0.0

    def _payload(self) -> dict[str, Any]:
        return {"rms": round(self.rms, 4), "peak": round(self.peak, 4)}


@dataclass(frozen=True)
class LogLine(Event):
    kind: ClassVar[str] = "log"
    level: Literal["debug", "info", "warning", "error"] = "info"
    message: str = ""
    source: str = ""

    def _payload(self) -> dict[str, Any]:
        return {"level": self.level, "message": self.message, "source": self.source}


@dataclass(frozen=True)
class ConfigChanged(Event):
    kind: ClassVar[str] = "config"
    sections: tuple[str, ...] = ()
    restart_required: bool = False

    def _payload(self) -> dict[str, Any]:
        return {"sections": list(self.sections), "restart_required": self.restart_required}
