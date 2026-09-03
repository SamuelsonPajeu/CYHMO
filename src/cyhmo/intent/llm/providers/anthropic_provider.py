"""Provider Anthropic via SDK oficial (``anthropic``, importado tardiamente).

A chave sai de ``os.environ[api_key_env]``: a config guarda só o NOME da variável.
"""

from __future__ import annotations

import os
from typing import Any, Callable

from cyhmo.intent.llm.providers.base import LlmProviderError

DEFAULT_MODEL = "claude-opus-5"

ClientFactory = Callable[[str], Any]


class AnthropicProvider:
    def __init__(
        self,
        model: str,
        api_key_env: str,
        client_factory: ClientFactory | None = None,
        max_output_tokens: int = 8,
    ) -> None:
        self._model = model or DEFAULT_MODEL
        self._max_output_tokens = max_output_tokens
        self._client = (client_factory or _sdk_client)(_api_key_from_env(api_key_env))

    @property
    def name(self) -> str:
        return "anthropic"

    def complete(self, system: str, user: str, timeout_ms: int) -> str:
        messages = self._client.with_options(timeout=timeout_ms / 1000, max_retries=0).messages
        try:
            response = messages.create(
                model=self._model,
                max_tokens=self._max_output_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except _sdk_errors() as error:
            raise LlmProviderError(f"API Anthropic ({self._model}) falhou: {error}") from error
        if response.stop_reason == "refusal":
            raise LlmProviderError(f"API Anthropic ({self._model}) recusou a requisição")
        return "".join(block.text for block in response.content if block.type == "text")


def _api_key_from_env(api_key_env: str) -> str:
    api_key = os.environ.get(api_key_env, "").strip()
    if not api_key:
        raise LlmProviderError(
            f"variável de ambiente {api_key_env} não definida: exporte nela a chave da API Anthropic "
            "(intent.llm.api_key_env guarda só o NOME da variável, nunca a chave)"
        )
    return api_key


def _sdk_client(api_key: str) -> Any:
    try:
        import anthropic
    except ImportError as error:
        raise LlmProviderError("pacote 'anthropic' não instalado: rode pip install anthropic") from error
    return anthropic.Anthropic(api_key=api_key)


def _sdk_errors() -> tuple[type[Exception], ...]:
    try:
        import anthropic
    except ImportError:
        return ()
    return (anthropic.APIError,)
