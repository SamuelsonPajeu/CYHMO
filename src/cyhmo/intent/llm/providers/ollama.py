"""Provider Ollama: ``POST /api/chat`` com temperatura zero, teto de saída e modelo residente."""

from __future__ import annotations

from typing import Any

from cyhmo.intent.llm.providers.base import LlmProviderError, post_json


class OllamaProvider:
    """``keep_alive`` é o que separa 600 ms de 6,5 s: sem ele o Ollama descarrega o modelo
    entre comandos esparsos e cada chamada volta a pagar o carregamento do disco."""

    def __init__(
        self,
        endpoint: str,
        model: str,
        keep_alive: str = "30m",
        max_output_tokens: int = 8,
        transport: Any | None = None,
    ) -> None:
        self._url = f"{endpoint.rstrip('/')}/api/chat"
        self._model = model
        self._keep_alive = keep_alive
        self._max_output_tokens = max_output_tokens
        self._transport = transport

    @property
    def name(self) -> str:
        return "ollama"

    def complete(self, system: str, user: str, timeout_ms: int) -> str:
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "keep_alive": self._keep_alive,
            "options": {"temperature": 0, "num_predict": self._max_output_tokens},
        }
        body = post_json(self._url, payload, {}, timeout_ms, self._transport)
        message = body.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise LlmProviderError(f"Ollama ({self._model}) devolveu resposta sem message.content")
        return content
