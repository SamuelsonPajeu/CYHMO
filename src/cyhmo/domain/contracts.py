"""Este arquivo é o contrato canônico entre as camadas.

Os tipos são imutáveis e serializáveis via ``to_dict()``; a serialização é o
formato que a telemetria e a ViewModel consomem.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

Method = Literal["embeddings", "llm", "none"]
UtteranceSource = Literal["mic"]
UtteranceStatus = Literal[
    "injected", "no_command", "blocked_cannot_talk", "cancelled", "error", "dry_run"
]

DEFAULT_SAMPLE_RATE = 16_000
MAX_STACKED_COMMANDS = 3


@dataclass(frozen=True)
class AudioSegment:
    """PCM mono float32 já recortado pelo modo de ativação.

    ``t_end`` é o instante monotônico (``time.perf_counter``) do fim da fala —
    a âncora da latência fim a fim.
    """

    samples: np.ndarray
    sample_rate: int = DEFAULT_SAMPLE_RATE
    t_end: float | None = None

    def __post_init__(self) -> None:
        samples = np.asarray(self.samples, dtype=np.float32).reshape(-1)
        object.__setattr__(self, "samples", samples)
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate deve ser positivo (recebi {self.sample_rate})")

    @property
    def duration_ms(self) -> float:
        return len(self.samples) * 1000.0 / self.sample_rate

    @property
    def is_empty(self) -> bool:
        return len(self.samples) == 0

    def sha256(self) -> str:
        return hashlib.sha256(np.ascontiguousarray(self.samples).tobytes()).hexdigest()

    def with_end(self, t_end: float) -> "AudioSegment":
        return AudioSegment(self.samples, self.sample_rate, t_end)

    @classmethod
    def silence(cls, duration_ms: float, sample_rate: int = DEFAULT_SAMPLE_RATE) -> "AudioSegment":
        return cls(np.zeros(int(sample_rate * duration_ms / 1000.0), dtype=np.float32), sample_rate)


@dataclass(frozen=True)
class Transcript:
    """Saída da camada 1."""

    text: str
    lang: str
    confidence: float
    t_speech_end: float
    raw_text: str = ""

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()

    @classmethod
    def empty(cls, t_speech_end: float, lang: str = "") -> "Transcript":
        return cls(text="", lang=lang, confidence=0.0, t_speech_end=t_speech_end, raw_text="")

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "lang": self.lang,
            "confidence": round(self.confidence, 4),
            "t_speech_end": self.t_speech_end,
            "raw_text": self.raw_text,
        }


@dataclass(frozen=True)
class GameState:
    """Saída da camada 3."""

    mode: str = "unknown"
    can_talk: bool | None = None
    room: int | None = None
    hp: float | None = None
    enemies: tuple[dict[str, Any], ...] | None = None
    inventory: tuple[dict[str, Any], ...] | None = None
    grammar: tuple[str, ...] | None = None
    grammar_stale: bool = False
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def has_grammar(self) -> bool:
        return bool(self.grammar)

    @property
    def in_battle(self) -> bool:
        return self.mode == "battle"

    @property
    def enemy_count(self) -> int | None:
        return None if self.enemies is None else len(self.enemies)

    @classmethod
    def unknown(cls) -> "GameState":
        return cls()

    def with_grammar(self, grammar: tuple[str, ...] | None, stale: bool = False) -> "GameState":
        return GameState(
            mode=self.mode,
            can_talk=self.can_talk,
            room=self.room,
            hp=self.hp,
            enemies=self.enemies,
            inventory=self.inventory,
            grammar=grammar,
            grammar_stale=stale,
            raw=self.raw,
        )

    def to_dict(self, include_grammar: bool = False) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "mode": self.mode,
            "can_talk": self.can_talk,
            "room": self.room,
            "hp": self.hp,
            "enemies": None if self.enemies is None else len(self.enemies),
            "inventory": None if self.inventory is None else list(self.inventory),
            "grammar_size": None if self.grammar is None else len(self.grammar),
            "grammar_stale": self.grammar_stale,
            "raw_digest": _digest(self.raw),
        }
        if include_grammar:
            summary["grammar"] = None if self.grammar is None else list(self.grammar)
        return summary


@dataclass(frozen=True)
class CommandRef:
    """A ``key`` é a string literal da gramática — exatamente o que será escrito."""

    key: str
    args: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key, "args": dict(self.args)}


@dataclass(frozen=True)
class Candidate:
    """Um candidato do matching, para log/debug e prompt do LLM."""

    key: str
    score: float
    matched_example: str = ""
    example_lang: str = "en"
    has_primary_language_examples: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "score": round(self.score, 4),
            "matched_example": self.matched_example,
            "example_lang": self.example_lang,
            "has_primary_language_examples": self.has_primary_language_examples,
        }


@dataclass(frozen=True)
class Interpretation:
    """Saída da camada 2. ``commands`` tem no máximo 3 itens."""

    commands: tuple[CommandRef, ...]
    confidence: float
    method: Method
    reason: str = ""
    candidates: tuple[Candidate, ...] = ()
    normalized_text: str = ""
    latency_ms: float = 0.0

    def __post_init__(self) -> None:
        if len(self.commands) > MAX_STACKED_COMMANDS:
            raise ValueError(f"máximo {MAX_STACKED_COMMANDS} comandos (recebi {len(self.commands)})")

    @property
    def is_empty(self) -> bool:
        return not self.commands

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(command.key for command in self.commands)

    @classmethod
    def none(
        cls,
        reason: str,
        candidates: tuple[Candidate, ...] = (),
        normalized_text: str = "",
        latency_ms: float = 0.0,
    ) -> "Interpretation":
        return cls(
            commands=(),
            confidence=0.0,
            method="none",
            reason=reason,
            candidates=candidates,
            normalized_text=normalized_text,
            latency_ms=latency_ms,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "commands": [command.to_dict() for command in self.commands],
            "confidence": round(self.confidence, 4),
            "method": self.method,
            "reason": self.reason,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "normalized_text": self.normalized_text,
            "latency_ms": round(self.latency_ms, 2),
        }


@dataclass(frozen=True)
class InjectResult:
    """Saída da camada 4, com o eco do oráculo."""

    ok: bool
    latency_ms: float
    error: str | None = None
    matched: bool | None = None
    payload_echo: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "latency_ms": round(self.latency_ms, 2),
            "error": self.error,
            "matched": self.matched,
            "payload_echo": dict(self.payload_echo),
        }


@dataclass(frozen=True)
class Utterance:
    """Evento produzido pela captura ao fim de um enunciado."""

    utt_id: str
    audio: AudioSegment
    t_release: float
    t_press: float | None = None
    t_vad_end: float | None = None
    source: UtteranceSource = "mic"

    @property
    def t_speech_end(self) -> float:
        return self.t_vad_end if self.t_vad_end is not None else self.t_release


def _digest(raw: dict[str, Any]) -> str:
    if not raw:
        return ""
    material = repr(sorted(raw.items(), key=lambda item: item[0])).encode("utf-8", "replace")
    return hashlib.sha256(material).hexdigest()[:16]
