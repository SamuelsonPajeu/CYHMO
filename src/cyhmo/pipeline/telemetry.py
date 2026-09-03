"""Registro JSONL por enunciado: construção e escrita."""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from cyhmo.domain.contracts import GameState, InjectResult, Interpretation, Transcript, Utterance
from cyhmo.pipeline.session import Session

log = logging.getLogger("cyhmo.pipeline.telemetry")

SCHEMA_VERSION = 1
TELEMETRY_CANDIDATES = 3
STAGES: tuple[str, ...] = ("vad_tail", "stt", "state", "interpret", "inject", "total")


class TelemetryWriter:
    def __init__(self, directory: Path, session: Session) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"session-{session.label}.jsonl"
        self._file = self.path.open("a", encoding="utf-8")
        self._lock = threading.Lock()
        self._written = 0

    def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"), default=str)
        with self._lock:
            if self._file.closed:
                log.warning("telemetria já fechada; registro %s descartado", record.get("utt_id"))
                return
            self._file.write(line + "\n")
            self._file.flush()
            self._written += 1

    def close(self) -> None:
        with self._lock:
            if not self._file.closed:
                self._file.close()

    @property
    def closed(self) -> bool:
        return self._file.closed

    @property
    def records_written(self) -> int:
        with self._lock:
            return self._written


def build_record(
    session: Session,
    utterance: Utterance,
    status: str,
    transcript: Transcript | None,
    state: GameState | None,
    interpretation: Interpretation | None,
    injection: InjectResult | None,
    latency_ms: dict[str, float | None],
    wav_path: str | None,
    error: str | None,
    wall_time: datetime,
) -> dict[str, Any]:
    if wall_time.tzinfo is None:
        wall_time = wall_time.astimezone()
    return {
        "schema": SCHEMA_VERSION,
        "utt_id": utterance.utt_id,
        "wall_time": wall_time.isoformat(timespec="milliseconds"),
        "source": utterance.source,
        "status": status,
        "audio": {
            "sha256": utterance.audio.sha256(),
            "duration_ms": round(utterance.audio.duration_ms, 1),
            "wav_path": wav_path,
        },
        "transcript": None if transcript is None else _transcript_summary(transcript),
        "state": None if state is None else state.to_dict(),
        "interpretation": None if interpretation is None else _interpretation_summary(interpretation),
        "injection": None if injection is None else _injection_summary(injection),
        "latency_ms": {stage: _round_or_none(latency_ms.get(stage)) for stage in STAGES},
        "config_id": session.config_id,
        "error": error,
    }


def _transcript_summary(transcript: Transcript) -> dict[str, Any]:
    return {
        "text": transcript.text,
        "lang": transcript.lang,
        "confidence": round(transcript.confidence, 4),
    }


def _interpretation_summary(interpretation: Interpretation) -> dict[str, Any]:
    """``reason`` e ``candidates`` são o que separa "não casou" de "casou e foi barrado":
    sem eles a telemetria de um enunciado sem comando não diz nada sobre a causa."""
    return {
        "commands": [command.to_dict() for command in interpretation.commands],
        "confidence": round(interpretation.confidence, 4),
        "method": interpretation.method,
        "reason": interpretation.reason,
        "normalized_text": interpretation.normalized_text,
        "candidates": [candidate.to_dict() for candidate in interpretation.candidates[:TELEMETRY_CANDIDATES]],
    }


def _injection_summary(injection: InjectResult) -> dict[str, Any]:
    return {
        "ok": injection.ok,
        "error": injection.error,
        "payload_echo": dict(injection.payload_echo),
    }


def _round_or_none(value: float | None) -> float | None:
    return None if value is None else round(value, 2)
