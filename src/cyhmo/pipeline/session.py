"""Identidade da sessão de jogo e gerador de ids de enunciado."""

from __future__ import annotations

import threading
from collections import Counter
from datetime import datetime

DEFAULT_CONFIG_ID = "uncalibrated"
LABEL_FORMAT = "%Y%m%d-%H%M%S"


class Session:
    def __init__(self, started_at: datetime | None = None, config_id: str = DEFAULT_CONFIG_ID) -> None:
        started_at = started_at or datetime.now()
        self.started_at = started_at if started_at.tzinfo else started_at.astimezone()
        self.label = self.started_at.strftime(LABEL_FORMAT)
        self.id = f"s{self.label}"
        self.config_id = config_id
        self._counter = 0
        self._by_source: Counter[str] = Counter()
        self._lock = threading.Lock()

    def next_utterance_id(self, source: str = "mic") -> str:
        with self._lock:
            self._counter += 1
            self._by_source[source] += 1
            return f"{self.id}#{self._counter:05d}"

    @property
    def utterance_count(self) -> int:
        with self._lock:
            return self._counter

    def counts_by_source(self) -> dict[str, int]:
        with self._lock:
            return dict(self._by_source)
