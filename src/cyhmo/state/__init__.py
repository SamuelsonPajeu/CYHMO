"""Camada 3 — estado de jogo: gramática ativa lida em runtime."""

from cyhmo.state.grammar import extract_vocabulary, looks_like_command, looks_like_phrase, physical
from cyhmo.state.grammar_source import (
    EmptyGrammarSource,
    FileGrammarSource,
    GrammarSnapshot,
    GrammarSource,
    PineGrammarSource,
)
from cyhmo.state.mode import ModeInference, infer_mode
from cyhmo.state.service import GameStateService, build_state_service

__all__ = [
    "EmptyGrammarSource",
    "FileGrammarSource",
    "GameStateService",
    "GrammarSnapshot",
    "GrammarSource",
    "ModeInference",
    "PineGrammarSource",
    "build_state_service",
    "extract_vocabulary",
    "infer_mode",
    "looks_like_command",
    "looks_like_phrase",
    "physical",
]
