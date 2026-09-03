"""Erro comum dos providers e o transporte HTTP compartilhado por Ollama e OpenAI-compat.

A interface dos providers é ``cyhmo.domain.ports.LlmProvider``. ``httpx`` só é
importado dentro de ``post_json`` para o pacote carregar sem a dependência.
"""

from __future__ import annotations

from typing import Any, Mapping

from cyhmo.domain.errors import CyhmoError

ERROR_BODY_PREVIEW = 200


class LlmProviderError(CyhmoError):
    """Provider indisponível, resposta malformada ou chamada recusada."""


def post_json(
    url: str,
    payload: Mapping[str, Any],
    headers: Mapping[str, str],
    timeout_ms: int,
    transport: Any | None = None,
) -> dict[str, Any]:
    import httpx

    try:
        with httpx.Client(transport=transport, timeout=timeout_ms / 1000) as client:
            response = client.post(url, json=dict(payload), headers=dict(headers))
            response.raise_for_status()
            body = response.json()
    except httpx.TimeoutException as error:
        raise LlmProviderError(f"LLM em {url} não respondeu em {timeout_ms} ms") from error
    except httpx.HTTPStatusError as error:
        preview = error.response.text[:ERROR_BODY_PREVIEW]
        raise LlmProviderError(
            f"LLM em {url} respondeu HTTP {error.response.status_code}: {preview}"
        ) from error
    except httpx.HTTPError as error:
        raise LlmProviderError(f"falha ao falar com o LLM em {url}: {error}") from error
    except ValueError as error:
        raise LlmProviderError(f"LLM em {url} devolveu um corpo que não é JSON") from error
    if not isinstance(body, dict):
        raise LlmProviderError(f"LLM em {url} devolveu JSON inesperado (esperava um objeto)")
    return body
