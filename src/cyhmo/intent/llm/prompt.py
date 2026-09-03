"""Prompt do assistente: lista de comandos e resposta com o comando literal.

Medido em 2026-08-26 contra o formato JSON, mesmo conjunto e mesmo modelo: acerto igual,
recusa de ruído 10/10 contra 8/10, mediana 53 ms contra 333 ms e 2,4 tokens de saída contra
26. O JSON segue descartado.

Medido em 2026-08-31 contra a resposta por ÍNDICE, 18 falas pt-BR na mesma lista de 20 e com
o jogo aberto: `qwen2.5:3b` acerta 13/18 contra 11/18, em 34,8 ms contra 40,2 ms. Quem decidiu
não foi o acerto e sim a FORMA do erro — erro de índice cai sempre num item válido e vira
comando errado injetado (7 dos 18: `abre o armário` → `Zoom In`, `anda` → `Look behind you`),
enquanto erro de literal vira composição fora da lista ou `NONE`, que o mod trata como falha e
resgata com o palpite do matcher (1 dos 18).
"""

from __future__ import annotations

from typing import Sequence

from cyhmo.domain.contracts import Candidate, GameState

REFUSAL_WORD = "NONE"

SYSTEM_PROMPT = (
    "You match what a player said to the voice commands a PS2 game accepts right now.\n"
    "The player speaks another language; the commands are English.\n"
    "Reply with the command text, copied exactly from the list.\n"
    f"Reply {REFUSAL_WORD} when the player said something that is not any of the commands.\n"
    "Never explain. Never write anything else."
)

PAIR_SYSTEM_PROMPT = SYSTEM_PROMPT + (
    "\nA matcher already guessed one of the commands; it is shown as the guess.\n"
    "Keep the guess unless the player clearly meant a different command on the list."
)


def build_prompt(
    normalized_text: str,
    candidates: Sequence[Candidate],
    state: GameState,
    primary_language: str,
    top_k: int,
    best_guess: str | None = None,
) -> tuple[str, str]:
    """A opção de recusa vive só no texto de sistema. Listá-la como item da lista derruba o
    acerto de 8/11 para 5/11 (medido): o modelo passa a tratá-la como escolha plausível."""
    selected = select_candidates(candidates, top_k)
    lines = ["Commands:"]
    lines.extend(candidate.key for candidate in selected)
    situation = summarize_state(state)
    if situation:
        lines.append(f"\nSituation: {situation}")
    lines.append(f"\nPlayer said ({primary_language}): {normalized_text}")
    guess = guess_literal(selected, best_guess)
    if guess is not None:
        lines.append(f"Matcher guess: {guess}")
    lines.append("Command:")
    return (PAIR_SYSTEM_PROMPT if guess is not None else SYSTEM_PROMPT), "\n".join(lines)


def select_candidates(candidates: Sequence[Candidate], top_k: int) -> tuple[Candidate, ...]:
    """Primeiros ``top_k`` candidatos com key única, na ordem do matching."""
    selected: dict[str, Candidate] = {}
    for candidate in candidates:
        if len(selected) >= top_k:
            break
        selected.setdefault(candidate.key, candidate)
    return tuple(selected.values())


def guess_literal(selected: Sequence[Candidate], key: str | None) -> str | None:
    """Palpite que não está na lista enviada não é anunciado: citá-lo convidaria o assistente a
    responder um comando que o prompt não ofereceu."""
    if key is None:
        return None
    for candidate in selected:
        if candidate.key == key:
            return candidate.key
    return None


def summarize_state(state: GameState) -> str:
    """Campo desconhecido não entra: ``hp=unknown`` gasta token e não desambigua nada."""
    fields = {"mode": _known_mode(state.mode), "enemies": state.enemy_count, "hp": state.hp}
    known = {name: value for name, value in fields.items() if value is not None}
    return "; ".join(f"{name}={_format_value(value)}" for name, value in known.items())


def _known_mode(mode: str) -> str | None:
    return None if mode == "unknown" else mode


def _format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)
