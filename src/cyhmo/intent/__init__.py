"""Camada 2 — interpretação de intenção: texto do jogador → strings da gramática ativa."""

from cyhmo.intent.interpreter import IntentInterpreter, LlmFallbackPort, build_interpreter

__all__ = ["IntentInterpreter", "LlmFallbackPort", "build_interpreter"]
