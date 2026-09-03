"""Camada 4 — injeção de comandos.

Escreve o TEXTO do comando onde o matcher nativo lê, liga a flag de commit e o
jogo faz o resto. O patch de 1 instrução é aplicado antes e restaurado sempre.
"""

from __future__ import annotations

import atexit
import struct
import threading
import time
from typing import Any, Callable, Sequence

from cyhmo.config.schema import AppConfig, InjectConfig, PineConfig, ProjectPaths
from cyhmo.domain.contracts import MAX_STACKED_COMMANDS, CommandRef, InjectResult
from cyhmo.domain.errors import InjectionError
from cyhmo.domain.events import LogLine
from cyhmo.domain.ports import CommandInjector, EventSink
from cyhmo.inject.pine import STATUS_NAMES, STATUS_RUNNING, MemoryWrite, PineClient, PineError
from cyhmo.inject.recipe import AsciiWordsRecipe, CodePatch, WriteRecipe
from cyhmo.pipeline.bus import NullSink
from cyhmo.state.grammar import normalize_entry

GrammarGate = Callable[[str], bool]
CanTalkReader = Callable[[], bool | None]
Sleep = Callable[[float], None]
Clock = Callable[[], float]

LOG_SOURCE = "inject"
EMPTY_ORACLE_ID = 0xFFFF
ORACLE_ID_SLOTS = 8
ORACLE_TEXT_LIMIT = 64
DRY_RUN_MAX_TEXT_LENGTH = 63
ERROR_CANNOT_TALK = "cannot_talk"
ERROR_CAN_TALK_UNKNOWN = "can_talk desconhecido (endereço não mapeado); desligue inject.require_can_talk ou mapeie o campo"


def validate_commands(commands: Sequence[CommandRef], max_text_length: int) -> str | None:
    """1..3 comandos, ASCII imprimível, cabendo no slot. Devolve o motivo da recusa."""
    if not 1 <= len(commands) <= MAX_STACKED_COMMANDS:
        return f"esperava de 1 a {MAX_STACKED_COMMANDS} comandos, recebi {len(commands)}"
    for command in commands:
        text = command.key
        if not text.strip():
            return "comando vazio"
        if any(not 0x20 <= ord(char) <= 0x7E for char in text):
            return f"comando com caractere fora do ASCII imprimível: {text!r}"
        if len(text) > max_text_length:
            return f"comando longo demais ({len(text)} > {max_text_length} caracteres): {text!r}"
    return None


def slot_bytes(text: str, slot_size: int) -> bytes:
    """Slot inteiro: texto + terminador + zeros até ``slot_size`` (armadilha dos 8 bytes)."""
    data = text.encode("ascii", errors="replace") + b"\x00"
    return data.ljust(slot_size, b"\x00")


def build_payload(recipe: AsciiWordsRecipe, words: Sequence[str], confidence: int) -> list[MemoryWrite]:
    """Escritas da receita na ordem normativa, SEM o commit (que vai por último)."""
    writes: list[MemoryWrite] = []
    for slot in range(recipe.max_slots):
        used = slot < len(words)
        data = slot_bytes(words[slot] if used else "", recipe.slot_size)
        for offset, (word,) in enumerate(struct.iter_unpack("<I", data)):
            writes.append((recipe.slot_address(slot) + offset * 4, 32, word))
        writes.append((recipe.pointer_address(slot), 32, recipe.slot_address(slot) if used else 0))
    writes.append((recipe.count_addr, 32, len(words)))
    writes.append((recipe.id_addr, 32, recipe.id_default))
    writes.append((recipe.asr_state.addr, 32, recipe.asr_state.value))
    writes.append((recipe.listening.addr, 32, recipe.listening.value))
    writes.append((recipe.confidence.addr, recipe.confidence.width, min(confidence, recipe.confidence.maximum)))
    return writes


def commit_write(recipe: AsciiWordsRecipe) -> MemoryWrite:
    return (recipe.commit.addr, recipe.commit.width, recipe.commit.value)


def describe_words(words: Sequence[str]) -> str:
    if len(words) == 1:
        return repr(words[0])
    return " + ".join(repr(word) for word in words) + f" ({len(words)} slots)"


class Injector:
    """Implementa ``ports.CommandInjector`` com a receita ``ascii_words``."""

    def __init__(
        self,
        client: PineClient,
        recipe: WriteRecipe,
        pine_config: PineConfig,
        inject_config: InjectConfig,
        grammar_gate: GrammarGate | None = None,
        can_talk_reader: CanTalkReader | None = None,
        bus: EventSink | None = None,
        sleep: Sleep = time.sleep,
        clock: Clock = time.perf_counter,
    ) -> None:
        self._client = client
        self._recipe = recipe
        self._words = recipe.ascii_words
        self._pine_config = pine_config
        self._inject_config = inject_config
        self._grammar_gate = grammar_gate
        self._can_talk_reader = can_talk_reader
        self._bus: EventSink = bus if bus is not None else NullSink()
        self._sleep = sleep
        self._clock = clock
        self._lock = threading.Lock()
        self._serial_verified_on_connection: int | None = None
        self._confidence_checked_on_connection: int | None = None
        self._patch_pending_restore = False
        atexit.register(self._restore_at_exit)

    @property
    def patch_pending_restore(self) -> bool:
        return self._patch_pending_restore

    def inject(self, commands: Sequence[CommandRef]) -> InjectResult:
        started = self._clock()
        words = [command.key for command in commands]
        error = validate_commands(commands, self._words.max_text_length)
        if error is not None:
            return self._blocked(started, words, error)
        with self._lock:
            try:
                self.restore_patch_if_pending()
                error = self._check_gates(words)
                if error is not None:
                    return self._blocked(started, words, error)
                return self._write_and_commit(started, words)
            except (PineError, InjectionError) as exc:
                self._recover_connection()
                return self._failed(started, words, str(exc))

    def read_status(self) -> dict[str, Any]:
        recipe = self._words
        patch = recipe.code_patch
        asr_state, count, word_id, listening, pending, gate, confidence, threshold = self._client.read_many(
            [
                (recipe.asr_state.addr, 32),
                (recipe.count_addr, 32),
                (recipe.id_addr, 32),
                (recipe.listening.addr, 32),
                (recipe.commit.addr, recipe.commit.width),
                (patch.addr, 32),
                (recipe.confidence.addr, recipe.confidence.width),
                (recipe.confidence.threshold_addr, 32),
            ]
        )
        slots = range(count) if count <= recipe.max_slots else range(0)
        words = [self._client.read_cstring(recipe.slot_address(slot), recipe.slot_size) for slot in slots]
        return {
            "asr_state": asr_state,
            "word_count": count,
            "word_id": word_id,
            "words": words,
            "listening": listening,
            "confidence": confidence,
            "confidence_threshold": threshold,
            "pending": pending,
            "gate_patched": gate != patch.original,
            "gate_orphan": self._is_orphan_gate(gate, patch),
            "oracle": self._read_oracle(words),
        }

    def restore_patch_if_pending(self) -> None:
        if self._patch_pending_restore:
            self._restore_patch()

    def close(self) -> None:
        """Solta o handler de ``atexit``: um injetor velho com reconexão automática restauraria
        o patch por cima do injetor atual depois de um reinício da interface. Toma o lock porque
        também escreve na PINE — sem ele, restauraria o patch no meio de uma injeção em curso."""
        try:
            with self._lock:
                self.restore_patch_if_pending()
        except (PineError, InjectionError) as exc:
            self._log("error", f"patch pendente não restaurado no encerramento: {exc}")
        finally:
            atexit.unregister(self._restore_at_exit)

    def _is_orphan_gate(self, gate: int, patch: CodePatch) -> bool:
        if self._pine_config.patch_mode != "runtime" or self._patch_pending_restore:
            return False
        return gate == patch.patched

    def _restore_at_exit(self) -> None:
        """Última chance no fim do processo, deliberadamente sem o lock: esperar por uma injeção
        travada penduraria o encerramento do interpretador."""
        try:
            self.restore_patch_if_pending()
        except Exception as exc:
            self._log("error", f"patch pendente não restaurado no encerramento: {exc}")

    def _check_gates(self, words: Sequence[str]) -> str | None:
        return self._serial_gate() or self._status_gate() or self._can_talk_gate() or self._grammar_check(words)

    def _serial_gate(self) -> str | None:
        if self._client.is_connected and self._serial_verified_on_connection == self._client.connection_count:
            return None
        serial = self._client.serial()
        try:
            self._recipe.validate_serial(serial)
        except InjectionError as exc:
            return str(exc)
        self._serial_verified_on_connection = self._client.connection_count
        return None

    def _status_gate(self) -> str | None:
        status = self._client.status()
        if status == STATUS_RUNNING:
            return None
        return f"VM não está rodando: {STATUS_NAMES.get(status, f'desconhecido ({status})')}"

    def _can_talk_gate(self) -> str | None:
        if not self._inject_config.require_can_talk:
            return None
        can_talk = self._can_talk_reader() if self._can_talk_reader is not None else None
        if can_talk is None:
            return ERROR_CAN_TALK_UNKNOWN
        if not can_talk:
            return ERROR_CANNOT_TALK
        return None

    def _grammar_check(self, words: Sequence[str]) -> str | None:
        if self._grammar_gate is None:
            return None
        outside = [word for word in words if not self._grammar_gate(word)]
        if not outside:
            return None
        return "fora da gramática ativa: " + ", ".join(outside)

    def _write_and_commit(self, started: float, words: Sequence[str]) -> InjectResult:
        confidence = self._confidence()
        try:
            adopted = self._apply_patch()
            self._client.write_many(build_payload(self._words, words, confidence) + [commit_write(self._words)])
            commit_readback = self._verify_readback(words)
            committed = self._clock()
            if self._patch_pending_restore:
                self._sleep(self._pine_config.patch_hold_ms / 1000.0)
        finally:
            if self._patch_pending_restore:
                self._restore_patch()
        oracle = self._read_oracle(words) if self._pine_config.verify_oracle else None
        latency_ms = (committed - started) * 1000.0
        verdict = "" if oracle is None else (" — oráculo: casou" if oracle["matched"] else " — oráculo: NÃO casou")
        self._log("info", f"injetado {describe_words(words)} em {latency_ms:.1f} ms{verdict}")
        return InjectResult(
            ok=True,
            latency_ms=latency_ms,
            matched=None if oracle is None else oracle["matched"],
            payload_echo={
                "words": list(words),
                "slots": len(words),
                "commit_addr": f"0x{self._words.commit.addr:08X}",
                "commit_readback": commit_readback,
                "patch_mode": self._pine_config.patch_mode,
                "patch_adopted": adopted,
                "confidence": confidence,
                "oracle": oracle,
            },
        )

    def _confidence(self) -> int:
        """Confere a nota contra o limiar do próprio jogo, uma vez por conexão.

        Abaixo do limiar o jogo devolve "Failed in recognition" **e mais nada** — sem o aviso,
        uma nota mal escolhida na config vira um tutorial que nunca passa, sem pista de por quê.
        """
        confidence = min(self._inject_config.confidence, self._words.confidence.maximum)
        if self._confidence_checked_on_connection == self._client.connection_count:
            return confidence
        try:
            threshold = self._client.read32(self._words.confidence.threshold_addr)
        except PineError as exc:
            self._log("debug", f"limiar de confiança ilegível: {exc}")
            return confidence
        self._confidence_checked_on_connection = self._client.connection_count
        if confidence < threshold:
            self._log(
                "warning",
                f"inject.confidence = {confidence} está abaixo do limiar do jogo ({threshold}): "
                "as cenas que pontuam o reconhecimento vão responder 'Failed in recognition'",
            )
        return confidence

    def _apply_patch(self) -> bool:
        """Devolve True quando adotou um patch órfão. A pendência de restauração é ligada
        ANTES da escrita: falha de framing no meio dela deixaria o código do jogo alterado."""
        if self._pine_config.patch_mode != "runtime":
            return False
        patch = self._words.code_patch
        adopted = self._inspect_gate(patch) if patch.verify_original else False
        self._patch_pending_restore = True
        self._client.write32(patch.addr, patch.patched)
        self._log("debug", f"code_patch aplicado em 0x{patch.addr:08X}")
        return adopted

    def _inspect_gate(self, patch: CodePatch) -> bool:
        current = self._client.read32(patch.addr)
        if current == patch.original:
            return False
        if current == patch.patched and patch.on_orphan == "adopt":
            self._log(
                "warning",
                f"patch órfão em 0x{patch.addr:08X}: execução anterior não restaurou o portão do pad; "
                "adotado e restaurado ao fim desta injeção",
            )
            return True
        self._log("warning", f"patch_verify_failed: 0x{patch.addr:08X} = 0x{current:08X}")
        raise InjectionError(
            f"valor original inesperado em 0x{patch.addr:08X}: lido 0x{current:08X}, "
            f"esperado 0x{patch.original:08X} (endereço errado, outra versão do jogo ou patch já aplicado)"
        )

    def _restore_patch(self) -> None:
        """A confirmação por leitura fica no mesmo try: chamado do ``finally``, um erro solto
        aqui mascararia a exceção original da injeção."""
        patch = self._words.code_patch
        try:
            self._client.write32(patch.addr, patch.original)
            landed = self._client.read32(patch.addr)
        except PineError as exc:
            self._log("error", f"patch NÃO restaurado em 0x{patch.addr:08X}: {exc} — nova tentativa na próxima injeção")
            return
        if landed != patch.original:
            self._log(
                "error",
                f"patch NÃO restaurado em 0x{patch.addr:08X}: lido 0x{landed:08X} depois da escrita "
                "— nova tentativa na próxima injeção",
            )
            return
        self._patch_pending_restore = False
        self._log("debug", f"code_restore em 0x{patch.addr:08X}")

    def _verify_readback(self, words: Sequence[str]) -> int:
        """O commit lido como 0 não é divergência: o jogo consome a flag (1 → 0) no frame seguinte."""
        commit = self._words.commit
        count, commit_value = self._client.read_many([(self._words.count_addr, 32), (commit.addr, commit.width)])
        if count != len(words):
            raise InjectionError(f"read-back divergente: contagem escrita {len(words)}, lida {count}")
        if commit_value not in (commit.value, 0):
            raise InjectionError(f"read-back divergente: commit escrito {commit.value}, lido {commit_value}")
        return commit_value

    def _read_oracle(self, words: Sequence[str]) -> dict[str, Any] | None:
        oracle = self._words.oracle
        try:
            raw_ids = self._client.read_range(oracle.accepted_ids, ORACLE_ID_SLOTS * 2)
            text = self._client.read_cstring(oracle.matched_text, ORACLE_TEXT_LIMIT)
        except PineError as exc:
            self._log("warning", f"oráculo ilegível: {exc}")
            return None
        accepted = [value for (value,) in struct.iter_unpack("<H", raw_ids) if value != EMPTY_ORACLE_ID]
        matched = bool(text) and normalize_entry(text) in {normalize_entry(word) for word in words}
        return {"accepted_ids": accepted, "matched_text": text, "matched": matched}

    def _recover_connection(self) -> None:
        try:
            self._client.ensure_connected()
        except PineError as exc:
            self._log("warning", f"PINE fora do ar: {exc}")

    def _blocked(self, started: float, words: Sequence[str], reason: str) -> InjectResult:
        self._log("warning", f"injeção bloqueada ({reason}): {describe_words(words) if words else '—'}")
        return self._result_without_write(started, words, reason)

    def _failed(self, started: float, words: Sequence[str], reason: str) -> InjectResult:
        self._log("error", f"injeção falhou ({reason}): {describe_words(words)}")
        return self._result_without_write(started, words, reason)

    def _result_without_write(self, started: float, words: Sequence[str], reason: str) -> InjectResult:
        return InjectResult(
            ok=False,
            latency_ms=(self._clock() - started) * 1000.0,
            error=reason,
            payload_echo={"words": list(words), "slots": len(words)},
        )

    def _log(self, level: str, message: str) -> None:
        self._bus.publish(LogLine(level=level, message=message, source=LOG_SOURCE))  # type: ignore[arg-type]


class DryRunInjector:
    """``CommandInjector`` que não escreve: usado sem PCSX2 ou com ``inject.enabled = false``."""

    def __init__(self, bus: EventSink | None = None, clock: Clock = time.perf_counter) -> None:
        self._bus: EventSink = bus if bus is not None else NullSink()
        self._clock = clock

    def inject(self, commands: Sequence[CommandRef]) -> InjectResult:
        started = self._clock()
        words = [command.key for command in commands]
        error = validate_commands(commands, DRY_RUN_MAX_TEXT_LENGTH)
        echo = {"dry_run": True, "words": words, "slots": len(words)}
        if error is not None:
            return InjectResult(ok=False, latency_ms=(self._clock() - started) * 1000.0, error=error, payload_echo=echo)
        self._bus.publish(
            LogLine(level="info", message=f"dry-run: {describe_words(words)} não foi injetado", source=LOG_SOURCE)
        )
        return InjectResult(ok=True, latency_ms=(self._clock() - started) * 1000.0, matched=None, payload_echo=echo)


def build_injector(
    config: AppConfig,
    paths: ProjectPaths,
    client: PineClient | None,
    grammar_gate: GrammarGate | None = None,
    can_talk_reader: CanTalkReader | None = None,
    bus: EventSink | None = None,
) -> CommandInjector:
    """Sem cliente ou com ``inject.enabled = false`` → dry-run. A receita é validada contra
    ``pine.expected_serial`` aqui; o serial do jogo em execução, na primeira injeção."""
    if client is None or not config.inject.enabled:
        return DryRunInjector(bus)
    recipe = WriteRecipe.load(paths.recipe)
    recipe.validate_serial(config.pine.expected_serial)
    return Injector(client, recipe, config.pine, config.inject, grammar_gate, can_talk_reader, bus)
