"""Fallback LLM: chama o provider com timeout duro e separa falha de recusa.

A chamada roda numa thread do pool; estourado o prazo, o chamador recebe
``LlmUnavailableError`` na hora e a thread pendente termina sozinha (os providers impõem o
mesmo timeout ao transporte).

``None`` passou a significar uma coisa só: o assistente respondeu e recusou. Prazo estourado,
provider fora do ar, resposta ilegível e número fora da lista viram ``LlmUnavailableError``,
para o interpretador poder cair no palpite do matcher em vez de perder o enunciado.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Literal, Sequence

from cyhmo.domain.contracts import Candidate, GameState, Interpretation
from cyhmo.domain.errors import LlmUnavailableError
from cyhmo.domain.events import LogLine
from cyhmo.domain.ports import EventSink, LlmProvider
from cyhmo.intent.llm.parsing import parse_response
from cyhmo.intent.llm.prompt import SYSTEM_PROMPT, build_prompt, select_candidates

LOG_SOURCE = "llm"
LOGGED_CANDIDATES = 3
WORKER_THREADS = 4
WARM_UP_USER = "Commands:\nWalk\n\nPlayer said: walk\nCommand:"
WARM_UP_TIMEOUT_MS = 20_000

LogLevel = Literal["debug", "info", "warning", "error"]


class LlmFallback:
    def __init__(
        self,
        provider: LlmProvider,
        timeout_ms: int,
        prompt_top_k: int,
        bus: EventSink | None = None,
        allow_in_battle: bool = False,
    ) -> None:
        self._provider = provider
        self._timeout_ms = timeout_ms
        self._prompt_top_k = prompt_top_k
        self._bus = bus
        self._allow_in_battle = allow_in_battle
        self._executor = ThreadPoolExecutor(max_workers=WORKER_THREADS, thread_name_prefix="llm-fallback")
        self.last_prompt: tuple[str, str] | None = None

    @property
    def provider_name(self) -> str:
        return self._provider.name

    @property
    def timeout_ms(self) -> int:
        return self._timeout_ms

    @property
    def prompt_top_k(self) -> int:
        return self._prompt_top_k

    def resolve(
        self,
        normalized_text: str,
        candidates: Sequence[Candidate],
        state: GameState,
        primary_language: str,
        best_guess: str | None = None,
    ) -> Interpretation | None:
        if state.in_battle and not self._allow_in_battle:
            self._log("debug", "fallback LLM suprimido em combate")
            return None
        try:
            return self._resolve(normalized_text, candidates, state, primary_language, best_guess)
        except LlmUnavailableError:
            raise
        except Exception as error:
            message = f"fallback LLM ({self.provider_name}) falhou: {error}"
            self._log("warning", message)
            raise LlmUnavailableError(message) from error

    def warm_up(self) -> None:
        """Carrega o modelo fora do caminho crítico. A primeira chamada a um Ollama frio custou
        6,5 s medidos, quase tudo em carregamento — o primeiro comando do jogador pagaria isso
        e estouraria o timeout.

        O prazo é generoso perto do de uma consulta, mas não infinito: a thread do pool não é
        daemon, então um aquecimento pendurado adia o encerramento do processo pelo mesmo tempo."""
        self._executor.submit(self._warm_up)

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _warm_up(self) -> None:
        started = time.perf_counter()
        try:
            self._provider.complete(SYSTEM_PROMPT, WARM_UP_USER, WARM_UP_TIMEOUT_MS)
        except Exception as error:
            self._log("debug", f"aquecimento do assistente ({self.provider_name}) não completou: {error}")
            return
        self._log("info", f"assistente ({self.provider_name}) aquecido em {(time.perf_counter() - started) * 1000:.0f} ms")

    def _resolve(
        self,
        normalized_text: str,
        candidates: Sequence[Candidate],
        state: GameState,
        primary_language: str,
        best_guess: str | None = None,
    ) -> Interpretation | None:
        selected = select_candidates(candidates, self._prompt_top_k)
        system, user = build_prompt(
            normalized_text, selected, state, primary_language, self._prompt_top_k, best_guess
        )
        self.last_prompt = (system, user)
        started = time.perf_counter()
        text = self._complete_within_deadline(system, user)
        latency_ms = (time.perf_counter() - started) * 1000
        parsed = parse_response(text, [candidate.key for candidate in selected])
        if parsed is None:
            message = (
                f"fallback LLM ({self.provider_name}) devolveu resposta ilegível em "
                f"{latency_ms:.0f} ms: {text.strip()[:40]!r}"
            )
            self._log("warning", message)
            raise LlmUnavailableError(message)
        self._log(
            "debug",
            f"fallback LLM ({self.provider_name}): {latency_ms:.0f} ms, "
            f"keys={[command.key for command in parsed.commands]}",
        )
        if parsed.is_empty:
            return None
        return Interpretation(
            commands=parsed.commands,
            confidence=_score_of(selected, parsed.commands[0].key),
            method="llm",
            reason="llm",
            candidates=tuple(candidates[:LOGGED_CANDIDATES]),
            normalized_text=normalized_text,
            latency_ms=latency_ms,
        )

    def _complete_within_deadline(self, system: str, user: str) -> str:
        future = self._executor.submit(self._provider.complete, system, user, self._timeout_ms)
        try:
            return future.result(timeout=self._timeout_ms / 1000)
        except TimeoutError as error:
            message = f"fallback LLM ({self.provider_name}) estourou o timeout de {self._timeout_ms} ms"
            self._log("warning", message)
            raise LlmUnavailableError(message) from error

    def _log(self, level: LogLevel, message: str) -> None:
        if self._bus is not None:
            self._bus.publish(LogLine(level=level, message=message, source=LOG_SOURCE))


def _score_of(selected: Sequence[Candidate], key: str) -> float:
    """O protocolo por índice não devolve confiança, e a autodeclarada por modelo pequeno era
    ruído. O score do matcher para a key escolhida é a medida honesta que já existe."""
    for candidate in selected:
        if candidate.key == key:
            return candidate.score
    return 0.0
