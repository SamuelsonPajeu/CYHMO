"""Constrói provider e fallback a partir de ``LlmConfig``."""

from __future__ import annotations

from cyhmo.config.schema import LlmConfig
from cyhmo.domain.errors import ConfigError
from cyhmo.domain.ports import EventSink, LlmProvider
from cyhmo.intent.llm.fallback import LlmFallback
from cyhmo.intent.llm.providers.anthropic_provider import AnthropicProvider
from cyhmo.intent.llm.providers.fake import FakeProvider
from cyhmo.intent.llm.providers.ollama import OllamaProvider
from cyhmo.intent.llm.prompt import REFUSAL_WORD
from cyhmo.intent.llm.providers.openai_compat import OpenAiCompatProvider

FAKE_RESPONSE = REFUSAL_WORD


def build_llm_provider(config: LlmConfig) -> LlmProvider:
    if config.provider == "fake":
        return FakeProvider(FAKE_RESPONSE)
    if config.provider == "anthropic":
        return AnthropicProvider(config.model, config.api_key_env, max_output_tokens=config.max_output_tokens)
    model = _required_model(config)
    if config.provider == "ollama":
        return OllamaProvider(config.endpoint, model, config.keep_alive, config.max_output_tokens)
    if config.provider == "openai_compat":
        return OpenAiCompatProvider(
            config.endpoint, model, config.api_key_env, max_output_tokens=config.max_output_tokens
        )
    raise ConfigError(
        f"intent.llm.provider desconhecido: {config.provider!r} (use ollama, openai_compat, anthropic ou fake)"
    )


def build_llm_fallback(config: LlmConfig, bus: EventSink | None = None) -> LlmFallback | None:
    if not config.enabled:
        return None
    fallback = LlmFallback(
        build_llm_provider(config),
        config.timeout_ms,
        config.prompt_top_k,
        bus,
        allow_in_battle=config.in_battle,
    )
    if config.warm_up:
        fallback.warm_up()
    return fallback


def _required_model(config: LlmConfig) -> str:
    model = config.model.strip()
    if not model:
        raise ConfigError(
            f"intent.llm.model é obrigatório para provider {config.provider!r} (ex.: 'qwen2.5:3b' no Ollama)"
        )
    return model
