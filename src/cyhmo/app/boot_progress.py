"""Sinal de vida no terminal enquanto as camadas sobem.

Sem isto o usuário encara uma linha parada por um minuto no primeiro start — o tempo que
o modelo de comparação leva para carregar — conclui que o mod travou e fecha a janela,
que é justamente o que mata o processo.
"""

from __future__ import annotations

import contextlib
import sys
import threading
import time
from typing import Iterator, TextIO

BOOT_STEPS = 5
HEARTBEAT_S = 12.0
ELAPSED_FLOOR_S = 1.0

class BootProgress:
    """Escreve linhas inteiras no mesmo fluxo do log: linha parcial some no meio da
    barra de progresso das bibliotecas de modelo."""

    def __init__(self, total: int = BOOT_STEPS, stream: TextIO | None = None) -> None:
        self._total = total
        self._stream = stream
        self._done = 0

    @classmethod
    def silent(cls) -> "BootProgress":
        """Todo comando que não é o ``run`` monta a aplicação sem plateia."""
        return cls()

    @classmethod
    def to_console(cls, total: int = BOOT_STEPS) -> "BootProgress":
        return cls(total, sys.stderr)

    def opening(self) -> None:
        self._write("")
        self._write("  CYHMO — preparando o mod. NÃO FECHE esta janela.")

    @contextlib.contextmanager
    def step(self, label: str, hint: str = "") -> Iterator[None]:
        self._done += 1
        self._write(f"  [{self._done}/{self._total}] {label}...")
        if hint:
            self._write(f"        {hint}")
        started = time.perf_counter()
        stop = self._start_heartbeat(label, started)
        try:
            yield
        finally:
            stop.set()
        self._write(f"        pronto{self._elapsed(started)}")

    def ready(self, url: str = "") -> None:
        self._write("")
        self._write(f"  CYHMO pronto — interface em {url}" if url else "  CYHMO pronto (sem interface).")
        self._write("  Deixe esta janela aberta: fechá-la encerra o mod.")
        self._write("  Para encerrar, use SAIR na interface ou Ctrl+C aqui.")
        self._write("")

    def _start_heartbeat(self, label: str, started: float) -> threading.Event:
        stop = threading.Event()
        if self._stream is None:
            return stop
        thread = threading.Thread(
            target=self._beat, args=(label, started, stop), name="cyhmo-boot-progress", daemon=True
        )
        thread.start()
        return stop

    def _beat(self, label: str, started: float, stop: threading.Event) -> None:
        while not stop.wait(HEARTBEAT_S):
            self._write(f"        ... {label} ({time.perf_counter() - started:.0f} s)")

    def _elapsed(self, started: float) -> str:
        seconds = time.perf_counter() - started
        return f" em {seconds:.0f} s" if seconds >= ELAPSED_FLOOR_S else ""

    def _write(self, text: str) -> None:
        if self._stream is None:
            return
        print(text, file=self._stream, flush=True)
