"""Barramento interno de eventos: produtores publicam, assinantes observam.

``publish`` nunca lança e nunca bloqueia quem publica: exceções de assinantes
vão para o log, e a ponte para ``asyncio`` descarta o evento mais antigo quando
a fila enche.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Callable

from cyhmo.domain.events import Event

EventHandler = Callable[[Event], None]
Unsubscribe = Callable[[], None]

log = logging.getLogger("cyhmo.bus")


class EventBus:
    def __init__(self) -> None:
        self._handlers: list[EventHandler] = []
        self._lock = threading.RLock()

    def subscribe(self, handler: EventHandler) -> Unsubscribe:
        with self._lock:
            self._handlers.append(handler)

        def unsubscribe() -> None:
            with self._lock:
                if handler in self._handlers:
                    self._handlers.remove(handler)

        return unsubscribe

    def publish(self, event: Event) -> None:
        with self._lock:
            handlers = tuple(self._handlers)
        for handler in handlers:
            try:
                handler(event)
            except Exception:
                log.exception("assinante falhou ao tratar %s", event.kind)

    def attach_queue(self, loop: asyncio.AbstractEventLoop, maxsize: int = 256) -> "AsyncEventQueue":
        return AsyncEventQueue(self, loop, maxsize)

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._handlers)


class AsyncEventQueue:
    """Ponte thread → loop asyncio com política descarta-o-antigo."""

    def __init__(self, bus: EventBus, loop: asyncio.AbstractEventLoop, maxsize: int) -> None:
        self._loop = loop
        self._queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=maxsize)
        self._dropped = 0
        self._unsubscribe = bus.subscribe(self._on_event)

    def _on_event(self, event: Event) -> None:
        self._loop.call_soon_threadsafe(self._enqueue, event)

    def _enqueue(self, event: Event) -> None:
        if self._queue.full():
            self._queue.get_nowait()
            self._dropped += 1
        self._queue.put_nowait(event)

    async def get(self) -> Event:
        return await self._queue.get()

    @property
    def dropped(self) -> int:
        return self._dropped

    def close(self) -> None:
        self._unsubscribe()


class NullSink:
    def publish(self, event: Event) -> None:
        return None
