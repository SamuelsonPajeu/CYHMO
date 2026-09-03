"""Camada de aplicação: composition root, CLI, logging e diagnóstico."""

from cyhmo.app.bootstrap import AppError, Application, config_id
from cyhmo.app.doctor import CheckResult, run_doctor
from cyhmo.app.logging_setup import setup_logging
from cyhmo.app.services import AppServices

__all__ = [
    "AppError",
    "AppServices",
    "Application",
    "CheckResult",
    "config_id",
    "run_doctor",
    "setup_logging",
]
