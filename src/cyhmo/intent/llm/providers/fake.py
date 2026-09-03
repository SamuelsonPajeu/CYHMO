"""Provider falso para testes e para ``provider = "fake"`` na config: responde sem rede."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Sequence

Responder = Callable[[str, str], str]


@dataclass(frozen=True)
class FakeCall:
    system: str
    user: str
    timeout_ms: int


class FakeProvider:
    def __init__(
        self,
        responses: str | Sequence[str] | Responder,
        delay_s: float = 0.0,
        name: str = "fake",
    ) -> None:
        self._respond = _as_responder(responses)
        self._delay_s = delay_s
        self._name = name
        self.calls: list[FakeCall] = []

    @property
    def name(self) -> str:
        return self._name

    def complete(self, system: str, user: str, timeout_ms: int) -> str:
        self.calls.append(FakeCall(system, user, timeout_ms))
        if self._delay_s > 0:
            time.sleep(self._delay_s)
        return self._respond(system, user)


def _as_responder(responses: str | Sequence[str] | Responder) -> Responder:
    if callable(responses):
        return responses
    if isinstance(responses, str):
        return lambda system, user: responses
    queue = list(responses)
    if not queue:
        raise ValueError("FakeProvider precisa de ao menos uma resposta")

    def next_response(system: str, user: str) -> str:
        return queue.pop(0) if len(queue) > 1 else queue[0]

    return next_response
