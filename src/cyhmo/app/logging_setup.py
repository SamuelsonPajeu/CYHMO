"""Logging do processo: console + arquivo rotativo, com as bibliotecas ruidosas caladas."""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
DATE_FORMAT = "%H:%M:%S"
FILE_MAX_BYTES = 1024 * 1024
FILE_BACKUP_COUNT = 5
NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "uvicorn.access",
    "sentence_transformers",
    "transformers",
    "urllib3",
    "filelock",
    "asyncio",
)

_OWNER_ATTRIBUTE = "_cyhmo_handler"
_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}


def setup_logging(level: str, log_file: Path | None, debug: bool = False) -> None:
    """Idempotente: chamadas repetidas substituem os handlers do mod em vez de empilhá-los."""
    resolved = logging.DEBUG if debug else _LEVELS.get(str(level).strip().lower(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(resolved)
    _replace(root, "console", _console_handler(resolved))
    _replace(root, "file", _file_handler(log_file, resolved))
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def _console_handler(level: int) -> logging.Handler:
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    return handler


def _file_handler(log_file: Path | None, level: int) -> logging.Handler | None:
    if log_file is None:
        return None
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            log_file, maxBytes=FILE_MAX_BYTES, backupCount=FILE_BACKUP_COUNT, encoding="utf-8"
        )
    except OSError as exc:
        logging.getLogger("cyhmo.app").warning("log em arquivo desabilitado (%s): %s", log_file, exc)
        return None
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    return handler


def _replace(root: logging.Logger, owner: str, handler: logging.Handler | None) -> None:
    for existing in list(root.handlers):
        if getattr(existing, _OWNER_ATTRIBUTE, None) == owner:
            root.removeHandler(existing)
            existing.close()
    if handler is None:
        return
    setattr(handler, _OWNER_ATTRIBUTE, owner)
    root.addHandler(handler)
