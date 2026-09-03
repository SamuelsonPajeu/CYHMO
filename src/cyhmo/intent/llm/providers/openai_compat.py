"""Provider para APIs no formato OpenAI (``/v1/chat/completions``): LM Studio, vLLM, OpenAI…

A chave, quando existe, sai de ``os.environ[api_key_env]``; servidores locais funcionam sem ela.
"""

from __future__ import annotations

import os
from typing import Any

from cyhmo.intent.llm.providers.base import LlmProviderError, post_json


class OpenAiCompatProvider:
    def __init__(
        self,
        endpoint: str,
        model: str,
        api_key_env: str,
        transport: Any | None = None,
        max_output_tokens: int = 8,
    ) -> None:
        self._url = chat_completions_url(endpoint)
        self._model = model
        self._api_key_env = api_key_env
        self._transport = transport
        self._max_output_tokens = max_output_tokens

    @property
    def name(self) -> str:
        return "openai_compat"

    def complete(self, system: str, user: str, timeout_ms: int) -> str:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "max_tokens": self._max_output_tokens,
        }
        body = post_json(self._url, payload, self._headers(), timeout_ms, self._transport)
        return self._content_of(body)

    def _headers(self) -> dict[str, str]:
        api_key = os.environ.get(self._api_key_env, "").strip()
        return {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def _content_of(self, body: dict[str, Any]) -> str:
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            content = None
        if not isinstance(content, str):
            raise LlmProviderError(
                f"API OpenAI-compat ({self._model}) devolveu resposta sem choices[0].message.content"
            )
        return content


def chat_completions_url(endpoint: str) -> str:
    base = endpoint.rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return f"{base}/chat/completions"
