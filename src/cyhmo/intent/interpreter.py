"""Interpretador de intenção: implementa ``ports.Interpreter`` sobre a gramática ativa."""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, replace
from typing import Any, Iterable, Protocol, Sequence

import numpy as np

from cyhmo.config.schema import AppConfig, IntentConfig, ProjectPaths
from cyhmo.domain.contracts import MAX_STACKED_COMMANDS, Candidate, CommandRef, GameState, Interpretation, Transcript
from cyhmo.domain.errors import LlmUnavailableError
from cyhmo.domain.events import ComponentChanged, LogLine
from cyhmo.domain.ports import EventSink, TextEmbedder
from cyhmo.intent.annex import Annex
from cyhmo.intent.arguments import ArgumentResolver, ResolvedSegment
from cyhmo.intent.embedders import build_embedder
from cyhmo.intent.embedding_cache import EmbeddingCache
from cyhmo.intent.index import CandidateIndex, build_example_provider
from cyhmo.intent.language_packs import LanguagePackSet
from cyhmo.intent.normalization import normalize_text
from cyhmo.intent.segmentation import UtteranceSegmenter
from cyhmo.intent.vocabulary import ActiveGrammar, ObservedVocabulary

log = logging.getLogger("cyhmo.intent")

LOG_SOURCE = "intent"
FIRST_GRAMMAR_WAIT_S = 5.0
LOGGED_CANDIDATES = 3
LLM_UNAVAILABLE_REASONS = frozenset({"llm_unavailable", "llm_error"})


class LlmFallbackPort(Protocol):
    def resolve(
        self,
        normalized_text: str,
        candidates: Sequence[Candidate],
        state: GameState,
        primary_language: str,
        best_guess: str | None = None,
    ) -> Interpretation | None: ...


@dataclass(frozen=True)
class _IndexState:
    index: CandidateIndex
    grammar: ActiveGrammar


@dataclass(frozen=True)
class _SegmentOutcome:
    commands: tuple[CommandRef, ...] = ()
    confidence: float = 0.0
    method: str = "none"
    candidates: tuple[Candidate, ...] = ()
    reason: str = ""
    has_primary_examples: bool = False
    detail: str = ""

    @property
    def rejected(self) -> bool:
        return not self.commands


class IntentInterpreter:
    """Modo ``unknown`` permite pilha de 3: a batalha jogada antes de o modo estar mapeado
    não pode ficar sem pilha; fora de batalha com modo conhecido o limite é 1."""

    def __init__(
        self,
        config: IntentConfig,
        packs: LanguagePackSet,
        annex: Annex,
        embedder: TextEmbedder,
        cache: EmbeddingCache,
        llm_fallback: LlmFallbackPort | None = None,
        bus: EventSink | None = None,
        observed_vocabulary: ObservedVocabulary | None = None,
    ) -> None:
        self._config = config
        self._packs = packs
        self._annex = annex
        self._embedder = embedder
        self._cache = cache
        self._llm = llm_fallback
        self._bus = bus
        self._observed = observed_vocabulary
        self._provider = build_example_provider(packs, annex)
        self._segmenter = UtteranceSegmenter(packs)
        self._arguments = ArgumentResolver(packs)
        self._lock = threading.RLock()
        self._ready = threading.Event()
        self._ready.set()
        self._state: _IndexState | None = None
        self._pending: ActiveGrammar | None = None
        self._generation = 0

    @property
    def current_grammar(self) -> ActiveGrammar | None:
        with self._lock:
            return None if self._state is None else self._state.grammar

    @property
    def index_ready(self) -> bool:
        with self._lock:
            return self._state is not None and self._ready.is_set()

    @property
    def primary_language(self) -> str:
        return self._packs.primary.code

    @property
    def observed_vocabulary(self) -> ObservedVocabulary | None:
        return self._observed

    def apply_intent(self, config: IntentConfig, llm_fallback: "LlmFallbackPort | None") -> None:
        """Adota a seção ``intent`` recém-salva sem reconstruir o interpretador.

        Antes, o que a interface salvava aqui só valia no reinício seguinte — e nada dizia
        isso, então ligar o assistente parecia funcionar e não funcionava. Os campos que
        exigem reconstrução de verdade (backend e modelo de embeddings, cache e anexo) são
        lidos só na montagem e continuam anunciados como "requer reinício" pela View.

        O índice não é tocado: limiar e top-k são lidos a cada enunciado."""
        with self._lock:
            self._config = config
            self._llm = llm_fallback

    def wait_ready(self, timeout: float | None = None) -> bool:
        return self._ready.wait(timeout) and self.current_grammar is not None

    def has_primary_language_examples(self, key: str) -> bool:
        with self._lock:
            state = self._state
        return state is not None and state.index.has_primary_language_examples(key)

    def update_grammar(self, entries: Sequence[str], stale: bool = False, pointer: int | None = None) -> None:
        grammar = ActiveGrammar.from_entries(entries, stale=stale, pointer=pointer)
        with self._lock:
            current = self._state
            if current is not None and current.grammar.entries == grammar.entries:
                self._state = replace(current, grammar=current.grammar.with_flags(stale, pointer))
                return
            if self._pending is not None and self._pending.entries == grammar.entries:
                return
            self._generation += 1
            grammar = replace(grammar, version=self._generation)
            self._pending = grammar
            if current is not None:
                self._state = replace(current, grammar=current.grammar.with_flags(True, current.grammar.pointer))
            self._ready.clear()
        self._publish(ComponentChanged(component="intent", status="busy", detail=f"indexando {grammar.size} entradas"))
        threading.Thread(target=self._build_index, args=(grammar,), name="intent-index", daemon=True).start()

    def warm_up(self, texts: Iterable[str] | None = None) -> int:
        """Aquece o cache com passagens. Sem ``texts``, usa vocabulário observado + anexo."""
        if texts is None:
            texts = self._warm_up_material()
        _, missing = self._cache.get_many(list(dict.fromkeys(texts)), "passage")
        if missing:
            threading.Thread(target=self._embed_passages, args=(missing,), name="intent-warmup", daemon=True).start()
        return len(missing)

    def interpret(self, transcript: Transcript, state: GameState) -> Interpretation:
        started = time.perf_counter()
        try:
            return self._interpret(transcript, state, started)
        except Exception as exc:
            log.exception("falha inesperada ao interpretar %r", transcript.text)
            self._publish(LogLine(level="error", message=f"interpretação falhou: {exc}", source=LOG_SOURCE))
            return Interpretation.none("error", latency_ms=_elapsed_ms(started))

    def _interpret(self, transcript: Transcript, state: GameState, started: float) -> Interpretation:
        if state.can_talk is False:
            return self._finish(transcript, state, Interpretation.none("cannot_talk"), started)
        normalized = normalize_text(transcript.text)
        if not normalized:
            return self._finish(transcript, state, Interpretation.none("empty"), started)
        snapshot = self._synchronized_snapshot(state)
        if snapshot is None:
            return self._finish(transcript, state, Interpretation.none("no_grammar", normalized_text=normalized), started)
        if state.grammar_stale:
            return self._finish(transcript, state, self._stale_refusal(normalized, started), started)
        max_commands = MAX_STACKED_COMMANDS if state.in_battle or state.mode == "unknown" else 1
        segmentation = self._segmenter.segment(normalized, snapshot.grammar, max_commands)
        accept = self._accept_threshold(state, snapshot.grammar)
        outcomes: list[_SegmentOutcome] = []
        for segment in segmentation.segments:
            outcome = self._resolve_segment(segment, state, snapshot, accept)
            if outcome.rejected:
                rejection = Interpretation.none(outcome.reason, outcome.candidates, normalized, _elapsed_ms(started))
                return self._finish(transcript, state, rejection, started, outcomes + [outcome])
            outcomes.append(outcome)
        commands = tuple(command for outcome in outcomes for command in outcome.commands)[:max_commands]
        interpretation = Interpretation(
            commands=commands,
            confidence=min(outcome.confidence for outcome in outcomes),
            method="llm" if any(outcome.method == "llm" for outcome in outcomes) else "embeddings",
            reason="truncated" if segmentation.truncated else "ok",
            candidates=outcomes[-1].candidates[:LOGGED_CANDIDATES],
            normalized_text=normalized,
            latency_ms=_elapsed_ms(started),
        )
        return self._finish(transcript, state, interpretation, started, outcomes)

    def _synchronized_snapshot(self, state: GameState) -> _IndexState | None:
        if state.grammar is not None:
            with self._lock:
                current = self._state
            known = current is not None and current.grammar.entries == ActiveGrammar.from_entries(state.grammar).entries
            if not known:
                self.update_grammar(state.grammar, stale=state.grammar_stale)
                if current is None:
                    self.wait_ready(FIRST_GRAMMAR_WAIT_S)
            elif current is not None and current.grammar.stale != state.grammar_stale:
                self.update_grammar(state.grammar, stale=state.grammar_stale)
        with self._lock:
            return self._state

    def _stale_refusal(self, normalized: str, started: float) -> Interpretation:
        """Gramática velha é gramática de outra cena: casar contra ela injeta comando errado
        com confiança (batalha de 2026-08-26, 'atira no olho do meio' virou 'Look over your
        shoulder'). Sem saber o que a cena aceita, o mod prefere não agir.

        Mira em ``state.grammar_stale`` — a camada 3 dizendo que não conseguiu reler a cena —
        e não no ``stale`` do índice, que também marca o intervalo de reindexação de uma
        gramática nova e legítima."""
        self._publish(
            LogLine(
                level="warning",
                message="gramática desatualizada: nada injetado até o mod reler a cena",
                source=LOG_SOURCE,
            )
        )
        return Interpretation.none("grammar_stale", normalized_text=normalized, latency_ms=_elapsed_ms(started))

    def _is_accepted(self, score: float, margin: float, accept: float) -> bool:
        """Aceite em dois níveis. Score alto basta por si; score médio precisa
        também de distância para o segundo colocado.

        Uma margem única não resolve: medido na batalha de 2026-08-26, exigir 0.05 de
        todo mundo derrubaria 'anda para a direita' → `move right` (0.988, margem 0.034)
        junto com o lixo, levando o acerto de 18 para 9. Com o nível de confiança,
        o acerto fica em 16 e o erro cai de 7 para 2."""
        confident = max(self._config.confident_threshold, accept)
        return score >= confident or (score >= accept and margin >= self._config.accept_margin)

    def _accept_threshold(self, state: GameState, grammar: ActiveGrammar) -> float:
        without_context = state.mode == "unknown" and state.can_talk is None and state.grammar is None
        threshold = self._config.accept_threshold_no_context if without_context else self._config.accept_threshold
        return threshold + (self._config.stale_grammar_penalty if grammar.stale else 0.0)

    def _resolve_segment(self, segment: str, state: GameState, snapshot: _IndexState, accept: float) -> _SegmentOutcome:
        resolved = self._arguments.resolve(segment, state)
        if resolved.reject_reason:
            return _SegmentOutcome(reason=resolved.reject_reason)
        literal = self._literal_match(segment, resolved, snapshot)
        if literal is not None:
            return literal
        candidates = tuple(self._search(resolved, snapshot))
        if not candidates:
            return _SegmentOutcome(reason="no_candidates")
        best = candidates[0]
        accepted = self._is_accepted(best.score, _margin(candidates), accept)
        llm = self._consulted_llm(state)
        if llm is not None and self._config.llm.mode == "pair":
            return self._arbitrated(llm, segment, state, snapshot, candidates, resolved, accepted, accept)
        if accepted:
            return self._accepted(best, resolved.args, candidates, "embeddings")
        if best.score < self._config.reject_threshold:
            return _SegmentOutcome(candidates=candidates, reason="below_reject")
        if llm is None:
            return _SegmentOutcome(candidates=candidates, reason="ambiguous")
        outcome = self._via_llm(llm, segment, state, snapshot, candidates)
        if outcome.reason in LLM_UNAVAILABLE_REASONS and best.score >= accept:
            return self._matcher_rescue(best, resolved, candidates)
        return outcome

    def _matcher_rescue(
        self, best: Candidate, resolved: ResolvedSegment, candidates: tuple[Candidate, ...]
    ) -> _SegmentOutcome:
        """Assistente que não responde não pode custar o comando.

        Só vale onde faltou apenas a margem sobre o segundo colocado: o palpite já passava
        do limiar de aceite, e injetá-lo é melhor que o silêncio. Score abaixo do limiar
        continua recusando — resgatar ali seria trocar o silêncio por comando errado."""
        self._publish(
            LogLine(
                level="warning",
                message=f"assistente indisponível: injetado o palpite do matcher {best.key!r}",
                source=LOG_SOURCE,
            )
        )
        return replace(self._accepted(best, resolved.args, candidates, "embeddings"), detail="llm_unavailable_kept")

    def _consulted_llm(self, state: GameState) -> LlmFallbackPort | None:
        """Combate é onde a latência mais dói, então o assistente fica de fora enquanto
        ``llm.in_battle`` for falso — vale para o fallback e para o par."""
        if self._llm is None or (state.in_battle and not self._config.llm.in_battle):
            return None
        return self._llm

    def _arbitrated(
        self,
        llm: LlmFallbackPort,
        segment: str,
        state: GameState,
        snapshot: _IndexState,
        candidates: tuple[Candidate, ...],
        resolved: ResolvedSegment,
        accepted: bool,
        accept: float,
    ) -> _SegmentOutcome:
        """Modo par: o assistente opina em todo segmento, inclusive nos que o matcher
        recusaria sozinho — é o caso do papel em cima da mesa que a cena só aceita como
        ``Invitation``.

        Invariante: resposta inútil devolve o que o modo fallback devolveria — o palpite do
        matcher onde ele aceitou, a recusa dele onde não. Ligar o par nunca perde comando
        que o fallback acertava."""
        best = candidates[0]
        verdict = self._via_llm(llm, segment, state, snapshot, candidates, best.key)
        if verdict.rejected:
            return self._matcher_prevails(best, resolved, candidates, accepted, verdict, accept)
        chosen = verdict.commands[0].key
        if not accepted:
            return replace(verdict, detail="pair_rescue")
        if chosen == best.key:
            return replace(self._accepted(best, resolved.args, candidates, "embeddings"), detail="pair_agree")
        self._publish(
            LogLine(
                level="info",
                message=f"assistente trocou o palpite {best.key!r} por {chosen!r}",
                source=LOG_SOURCE,
            )
        )
        return replace(verdict, detail="pair_override")

    def _matcher_prevails(
        self,
        best: Candidate,
        resolved: ResolvedSegment,
        candidates: tuple[Candidate, ...],
        accepted: bool,
        verdict: _SegmentOutcome,
        accept: float,
    ) -> _SegmentOutcome:
        """Rótulos distintos porque os desfechos são opostos: com o matcher aceito o comando
        dele é injetado; sem ele, nada é injetado. Um rótulo só obrigaria a telemetria a ser
        cruzada com a decisão para saber qual dos dois aconteceu.

        O resgate do palpite do matcher quando o assistente não responde também roda aqui:
        sem ele, ligar o par renderia menos que o modo ``fallback`` no mesmo enunciado, e o
        par nunca pode render menos que o modo que estende."""
        if accepted:
            return replace(self._accepted(best, resolved.args, candidates, "embeddings"), detail="pair_kept")
        if verdict.reason in LLM_UNAVAILABLE_REASONS and best.score >= accept:
            return self._matcher_rescue(best, resolved, candidates)
        if best.score < self._config.reject_threshold:
            return _SegmentOutcome(candidates=candidates, reason="below_reject", detail="pair_refused")
        return replace(verdict, detail="pair_refused")

    def _literal_match(self, segment: str, resolved: ResolvedSegment, snapshot: _IndexState) -> _SegmentOutcome | None:
        """Literal da gramática primeiro, exemplo curado depois: os dois são casamento exato,
        e o literal é a grafia que o jogo espera."""
        texts = [segment, *resolved.english_candidates]
        for text in texts:
            key = snapshot.grammar.literal_for(text)
            if key is not None:
                return self._exact(key, key, "en", resolved, snapshot)
        for text in texts:
            match = snapshot.index.exact_example(text)
            if match is not None:
                return self._exact(match.key, match.example, match.lang, resolved, snapshot)
        return None

    def _exact(
        self, key: str, example: str, lang: str, resolved: ResolvedSegment, snapshot: _IndexState
    ) -> _SegmentOutcome:
        candidate = Candidate(
            key=key,
            score=1.0,
            matched_example=example,
            example_lang=lang,
            has_primary_language_examples=snapshot.index.has_primary_language_examples(key),
        )
        return self._accepted(candidate, resolved.args, (candidate,), "embeddings")

    @staticmethod
    def _accepted(
        candidate: Candidate, args: dict[str, Any], candidates: tuple[Candidate, ...], method: str
    ) -> _SegmentOutcome:
        return _SegmentOutcome(
            commands=(CommandRef(candidate.key, dict(args)),),
            confidence=candidate.score,
            method=method,
            candidates=candidates,
            has_primary_examples=candidate.has_primary_language_examples,
        )

    def _search(self, resolved: ResolvedSegment, snapshot: _IndexState) -> list[Candidate]:
        top_k = max(self._config.top_k, self._config.llm.prompt_top_k if self._llm is not None else 0)
        return snapshot.index.search_variants(self._query_vectors(resolved), top_k)

    def _query_vectors(self, resolved: ResolvedSegment) -> list[np.ndarray]:
        """As variantes vão ao modelo num lote só: três chamadas separadas custam quase o
        triplo de uma, e isso está no caminho crítico do enunciado."""
        texts = [text for text in dict.fromkeys([resolved.query_text, *resolved.english_candidates]) if text]
        texts = texts[: self._config.max_query_variants]
        found, missing = self._cache.get_many(texts, "query")
        if missing:
            vectors = np.asarray(self._embedder.embed_queries(missing), dtype=np.float32)
            self._cache.put_many(missing, "query", vectors)
            found.update(zip(missing, vectors))
        return [found[text] for text in texts]

    def _via_llm(
        self,
        llm: LlmFallbackPort,
        segment: str,
        state: GameState,
        snapshot: _IndexState,
        candidates: tuple[Candidate, ...],
        best_guess: str | None = None,
    ) -> _SegmentOutcome:
        prompt_candidates = candidates[: self._config.llm.prompt_top_k]
        try:
            answer = llm.resolve(segment, prompt_candidates, state, self.primary_language, best_guess)
        except LlmUnavailableError as exc:
            log.warning("assistente indisponível: %s", exc)
            return _SegmentOutcome(candidates=candidates, reason="llm_unavailable")
        except Exception as exc:
            log.warning("fallback LLM falhou: %s", exc)
            return _SegmentOutcome(candidates=candidates, reason="llm_error")
        if answer is None or answer.is_empty:
            return _SegmentOutcome(candidates=candidates, reason="llm_none")
        commands = tuple(self._only_in_grammar(answer.commands, snapshot.grammar))
        if not commands:
            return _SegmentOutcome(candidates=candidates, reason="llm_outside_grammar")
        return _SegmentOutcome(
            commands=commands,
            confidence=answer.confidence,
            method="llm",
            candidates=candidates,
            has_primary_examples=snapshot.index.has_primary_language_examples(commands[0].key),
        )

    @staticmethod
    def _only_in_grammar(commands: Sequence[CommandRef], grammar: ActiveGrammar) -> list[CommandRef]:
        kept: list[CommandRef] = []
        for command in commands:
            literal = grammar.literal_for(command.key)
            if literal is None:
                log.warning("LLM devolveu key fora da gramática, descartada: %r", command.key)
            else:
                kept.append(CommandRef(literal, dict(command.args)))
        return kept

    def _build_index(self, grammar: ActiveGrammar) -> None:
        started = time.perf_counter()
        try:
            index = CandidateIndex.build(grammar, self._provider, self._embedder, self._cache, self.primary_language)
        except Exception as exc:
            log.exception("falha ao indexar a gramática (%d entradas)", grammar.size)
            self._install_index(grammar, None)
            self._publish(ComponentChanged(component="intent", status="error", detail=f"indexação falhou: {exc}"))
            self._signal_ready(grammar)
            return
        installed = self._install_index(grammar, index)
        self._record_observed(grammar)
        detail = (
            f"gramática v{grammar.version}: {index.size} entradas, {index.example_count} exemplos, "
            f"{_elapsed_ms(started):.0f} ms" + ("" if installed else " (descartada: gramática mais nova)")
        )
        self._publish(LogLine(level="info", message=detail, source=LOG_SOURCE))
        self._publish(ComponentChanged(component="intent", status="ready", detail=detail))
        self._persist_to_disk()
        self._signal_ready(grammar)

    def _install_index(self, grammar: ActiveGrammar, index: CandidateIndex | None) -> bool:
        with self._lock:
            newest = grammar.version == self._generation
            if newest:
                if index is not None:
                    self._state = _IndexState(index, grammar)
                self._pending = None
            return newest and index is not None

    def _signal_ready(self, grammar: ActiveGrammar) -> None:
        """Sinalizar por último garante que quem esperou já enxerga o índice, o evento e o observado."""
        with self._lock:
            if grammar.version == self._generation:
                self._ready.set()

    def _record_observed(self, grammar: ActiveGrammar) -> None:
        if self._observed is not None:
            self._observed.record(grammar.entries)

    def _persist_to_disk(self) -> None:
        try:
            self._cache.flush()
            if self._observed is not None:
                self._observed.save()
        except OSError as exc:
            log.warning("não foi possível persistir cache/vocabulário observado: %s", exc)

    def _warm_up_material(self) -> list[str]:
        material: list[str] = []
        if self._observed is not None:
            material.extend(self._observed.entries)
        material.extend(self._annex.all_examples(self._packs.codes))
        for pack in self._packs.packs:
            for phrases in pack.command_examples.values():
                material.extend(phrases)
        return material

    def _embed_passages(self, texts: list[str]) -> None:
        try:
            for start in range(0, len(texts), 64):
                batch = texts[start : start + 64]
                self._cache.put_many(batch, "passage", np.asarray(self._embedder.embed_passages(batch), dtype=np.float32))
            self._cache.flush()
            self._publish(LogLine(level="info", message=f"cache aquecido com {len(texts)} passagens", source=LOG_SOURCE))
        except Exception as exc:
            log.warning("aquecimento do cache falhou: %s", exc)

    def _finish(
        self,
        transcript: Transcript,
        state: GameState,
        interpretation: Interpretation,
        started: float,
        outcomes: Sequence[_SegmentOutcome] = (),
    ) -> Interpretation:
        if interpretation.latency_ms == 0.0:
            interpretation = replace(interpretation, latency_ms=_elapsed_ms(started))
        self._log_decision(transcript, state, interpretation, outcomes)
        return interpretation

    def _log_decision(
        self, transcript: Transcript, state: GameState, interpretation: Interpretation, outcomes: Sequence[_SegmentOutcome]
    ) -> None:
        grammar = self.current_grammar
        record: dict[str, Any] = {
            "raw": transcript.text,
            "normalized": interpretation.normalized_text,
            "mode": state.mode,
            "grammar": None if grammar is None else {"size": grammar.size, "stale": grammar.stale, "fp": grammar.fingerprint},
            "top": [candidate.to_dict() for candidate in interpretation.candidates[:LOGGED_CANDIDATES]],
            "decision": interpretation.reason,
            "method": interpretation.method,
            "commands": [command.to_dict() for command in interpretation.commands],
            "primary_examples": [outcome.has_primary_examples for outcome in outcomes if not outcome.rejected],
            "latency_ms": round(interpretation.latency_ms, 2),
        }
        arbitration = [outcome.detail for outcome in outcomes if outcome.detail]
        if arbitration:
            record["arbitration"] = arbitration
        message = "intent " + json.dumps(record, ensure_ascii=False)
        log.debug(message)
        self._publish(LogLine(level="debug", message=message, source=LOG_SOURCE))

    def _publish(self, event: LogLine | ComponentChanged) -> None:
        if self._bus is None:
            return
        try:
            self._bus.publish(event)
        except Exception:
            log.exception("barramento falhou ao receber %s", event.kind)


def build_interpreter(
    config: AppConfig,
    paths: ProjectPaths,
    packs: LanguagePackSet | None = None,
    embedder: TextEmbedder | None = None,
    llm_fallback: LlmFallbackPort | None = None,
    bus: EventSink | None = None,
) -> IntentInterpreter:
    packs = packs or LanguagePackSet.load(paths.packs_dir, config.languages.enabled, config.languages.primary)
    annex = Annex.load(paths.annex)
    embedder = embedder or build_embedder(config.intent, paths.models_dir)
    cache = EmbeddingCache(paths.embedding_cache, embedder.identity, embedder.dimension)
    observed = ObservedVocabulary(paths.observed_vocab)
    observed.load()
    return IntentInterpreter(config.intent, packs, annex, embedder, cache, llm_fallback, bus, observed)


def _margin(candidates: Sequence[Candidate]) -> float:
    """Distância do primeiro para o segundo. Score absoluto sozinho não distingue acerto de
    empate: modelo multilíngue comprime a faixa, e empate é justamente o caso do fallback."""
    return candidates[0].score if len(candidates) < 2 else candidates[0].score - candidates[1].score


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0
