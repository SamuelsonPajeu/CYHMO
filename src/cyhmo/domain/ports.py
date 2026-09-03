"""Ports (interfaces) que as camadas implementam e o pipeline consome.

A comunicação entre camadas passa exclusivamente por estes contratos.
"""

from __future__ import annotations

from typing import Callable, Protocol, Sequence, runtime_checkable

import numpy as np

from cyhmo.domain.contracts import (
    AudioSegment,
    CommandRef,
    GameState,
    InjectResult,
    Interpretation,
    Transcript,
    Utterance,
)
from cyhmo.domain.events import Event

AudioBlockHandler = Callable[[np.ndarray, float], None]
UtteranceHandler = Callable[[Utterance], None]


@runtime_checkable
class EventSink(Protocol):
    def publish(self, event: Event) -> None: ...


@runtime_checkable
class AudioCapture(Protocol):
    """Entrega blocos PCM float32 mono a 16 kHz com o timestamp monotônico do fim do bloco."""

    @property
    def device_name(self) -> str: ...

    @property
    def sample_rate(self) -> int: ...

    def start(self, on_block: AudioBlockHandler) -> None: ...

    def stop(self) -> None: ...


@runtime_checkable
class Activation(Protocol):
    """Decide quando um enunciado começa e termina (PTT ou VAD) e emite ``Utterance``."""

    @property
    def mode(self) -> str: ...

    def start(self, on_utterance: UtteranceHandler) -> None: ...

    def stop(self) -> None: ...

    def feed(self, block: np.ndarray, t_block_end: float) -> None: ...


@runtime_checkable
class Transcriber(Protocol):
    @property
    def model_name(self) -> str: ...

    def warm_up(self) -> None: ...

    def transcribe(self, audio: AudioSegment) -> Transcript: ...


@runtime_checkable
class TextEmbedder(Protocol):
    """``identity`` entra na chave do cache: trocar modelo ou prefixo invalida o cache."""

    @property
    def identity(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def warm_up(self) -> None: ...

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray: ...


@runtime_checkable
class LlmProvider(Protocol):
    @property
    def name(self) -> str: ...

    def complete(self, system: str, user: str, timeout_ms: int) -> str: ...


@runtime_checkable
class Interpreter(Protocol):
    def interpret(self, transcript: Transcript, state: GameState) -> Interpretation: ...


@runtime_checkable
class GameStateReader(Protocol):
    def start(self) -> None: ...

    def stop(self) -> None: ...

    def read_state(self) -> GameState: ...


@runtime_checkable
class CommandInjector(Protocol):
    def inject(self, commands: Sequence[CommandRef]) -> InjectResult: ...
