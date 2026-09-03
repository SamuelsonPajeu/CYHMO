"""Carga, validação e gravação do ``config.toml``."""

from __future__ import annotations

import re
import threading
import tomllib
from pathlib import Path
from typing import Any, Callable

from pydantic import ValidationError

from cyhmo.config.schema import AppConfig, ProjectPaths
from cyhmo.config.template import CONFIG_TEMPLATE
from cyhmo.domain.errors import ConfigError

DEFAULT_CONFIG_NAME = "config.toml"
_PLACEHOLDER = re.compile(r"\{([a-z_.]+)\}")


def load_config(path: Path | None, create_if_missing: bool = True) -> AppConfig:
    path = path or Path.cwd() / DEFAULT_CONFIG_NAME
    if not path.exists():
        if not create_if_missing:
            raise ConfigError(f"config não encontrada: {path}")
        write_default_config(path)
        return AppConfig()
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8-sig"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: TOML inválido — {exc}") from exc
    return parse_config(raw, source=str(path))


def parse_config(raw: dict[str, Any], source: str = "config") -> AppConfig:
    try:
        return AppConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(_describe(exc, source)) from exc


def write_default_config(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_config(AppConfig()), encoding="utf-8")
    return path


def render_config(config: AppConfig) -> str:
    values = _flatten(config.model_dump(mode="json"))

    def substitute(match: re.Match[str]) -> str:
        return _toml_value(values[match.group(1)])

    return _PLACEHOLDER.sub(substitute, CONFIG_TEMPLATE)


def _flatten(tree: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in tree.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{dotted}."))
        else:
            flat[dotted] = value
    return flat


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if value is None:
        return '""'
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _describe(error: ValidationError, source: str) -> str:
    lines = [f"{source}: configuração inválida"]
    for issue in error.errors():
        location = ".".join(str(part) for part in issue["loc"]) or "(raiz)"
        received = issue.get("input")
        detail = issue["msg"]
        if issue["type"] == "extra_forbidden":
            detail = "chave desconhecida"
        lines.append(f"  - {location}: {detail} (recebido: {received!r})")
    return "\n".join(lines)


class ConfigStore:
    """Config viva do processo: leitura thread-safe e gravação preservando comentários."""

    def __init__(self, config: AppConfig, path: Path) -> None:
        self._config = config
        self._path = path.resolve()
        self._lock = threading.RLock()
        self._listeners: list[Callable[[AppConfig, AppConfig], None]] = []

    @classmethod
    def open(cls, path: Path | None) -> "ConfigStore":
        path = (path or Path.cwd() / DEFAULT_CONFIG_NAME).resolve()
        return cls(load_config(path), path)

    @property
    def path(self) -> Path:
        return self._path

    @property
    def config(self) -> AppConfig:
        with self._lock:
            return self._config

    @property
    def paths(self) -> ProjectPaths:
        return self.config.paths(self._path.parent)

    def subscribe(self, listener: Callable[[AppConfig, AppConfig], None]) -> None:
        self._listeners.append(listener)

    def update(self, patch: dict[str, Any]) -> AppConfig:
        with self._lock:
            merged = _deep_merge(self._config.model_dump(mode="json"), patch)
            new_config = parse_config(merged, source=str(self._path))
            old_config = self._config
            self._config = new_config
            self._path.write_text(render_config(new_config), encoding="utf-8")
        for listener in self._listeners:
            listener(old_config, new_config)
        return new_config


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
