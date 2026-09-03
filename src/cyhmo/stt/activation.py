"""Ativação: decide quando um enunciado começa e termina — push-to-talk ou VAD."""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np

from cyhmo.config.schema import ActivationConfig, VadConfig
from cyhmo.domain.contracts import DEFAULT_SAMPLE_RATE, AudioSegment, Utterance
from cyhmo.domain.errors import CyhmoError
from cyhmo.domain.events import PhaseChanged
from cyhmo.domain.ports import Activation, EventSink, UtteranceHandler
from cyhmo.stt.vad import FRAME_SAMPLES, SileroVad, SpeechDetector

log = logging.getLogger("cyhmo.stt.activation")

IdFactory = Callable[[], str]
DetectorFactory = Callable[[float], SpeechDetector]
DEFAULT_PRE_ROLL_MS = 200


class HotkeyError(CyhmoError):
    """Hotkey do push-to-talk inválida ou hook global de teclado indisponível."""


@runtime_checkable
class KeyboardHook(Protocol):
    def add_press(self, key: str, callback: Callable[[], None]) -> None: ...

    def add_release(self, key: str, callback: Callable[[], None]) -> None: ...

    def remove_all(self) -> None: ...

    def read_key(self, timeout_s: float) -> str | None: ...


class KeyboardLibHook:
    """Hook global via a lib ``keyboard``; ``suppress=False`` deixa a tecla chegar ao PCSX2."""

    def __init__(self) -> None:
        self._handles: list[Callable[..., None]] = []

    def add_press(self, key: str, callback: Callable[[], None]) -> None:
        self._hook_key(key, "down", callback)

    def add_release(self, key: str, callback: Callable[[], None]) -> None:
        self._hook_key(key, "up", callback)

    def _hook_key(self, key: str, event_type: str, callback: Callable[[], None]) -> None:
        """``on_press_key``/``on_release_key`` indexam a remoção pelo NOME da tecla, então registrar
        press e release na mesma tecla faz o segundo unhook estourar antes de desligar o callback —
        e ele fica escutando para sempre. ``hook`` indexa pelo callback e não tem essa colisão."""
        keyboard = self._keyboard()
        scan_codes = frozenset(keyboard.key_to_scan_codes(key))

        def on_event(event: Any) -> None:
            if getattr(event, "event_type", None) == event_type and getattr(event, "scan_code", None) in scan_codes:
                callback()

        self._handles.append(keyboard.hook(on_event, suppress=False))

    def remove_all(self) -> None:
        handles, self._handles = self._handles, []
        if not handles:
            return
        keyboard = self._keyboard()
        for handle in handles:
            try:
                keyboard.unhook(handle)
            except (KeyError, ValueError):
                log.debug("hook de teclado já removido")

    def read_key(self, timeout_s: float) -> str | None:
        """Escuta o teclado inteiro até a primeira tecla pressionada; ``read_event`` bloquearia sem prazo."""
        keyboard = self._keyboard()
        captured: list[str] = []
        pressed = threading.Event()

        def on_event(event: Any) -> None:
            if pressed.is_set() or getattr(event, "event_type", None) != "down":
                return
            name = str(getattr(event, "name", "") or "").strip().lower()
            if not name:
                return
            captured.append(name)
            pressed.set()

        handle = keyboard.hook(on_event, suppress=False)
        try:
            pressed.wait(max(0.0, timeout_s))
        finally:
            try:
                keyboard.unhook(handle)
            except (KeyError, ValueError):
                log.debug("hook de captura já removido")
        return captured[0] if captured else None

    @staticmethod
    def _keyboard():
        import keyboard

        return keyboard


@dataclass
class _PressSession:
    utt_id: str
    t_press: float
    chunks: list[np.ndarray] = field(default_factory=list)


class PushToTalkActivation:
    """O hook de teclado e o reflexo humano de já falar ao apertar custam algumas dezenas de
    milissegundos: sem pré-roll o enunciado chega decapitado e o STT erra a primeira sílaba."""

    def __init__(
        self,
        hotkey: str,
        bus: EventSink,
        id_factory: IdFactory,
        hook: KeyboardHook | None = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        pre_roll_ms: int = DEFAULT_PRE_ROLL_MS,
    ) -> None:
        self._hotkey = hotkey
        self._bus = bus
        self._id_factory = id_factory
        self._hook: KeyboardHook = hook if hook is not None else KeyboardLibHook()
        self._sample_rate = sample_rate
        self._pre_roll_samples = int(max(0, pre_roll_ms) * sample_rate / 1000.0)
        self._pre_roll: deque[np.ndarray] = deque()
        self._pre_roll_size = 0
        self._lock = threading.Lock()
        self._on_utterance: UtteranceHandler | None = None
        self._session: _PressSession | None = None

    @property
    def mode(self) -> str:
        return "ptt"

    @property
    def hotkey(self) -> str:
        return self._hotkey

    @property
    def pressed(self) -> bool:
        with self._lock:
            return self._session is not None

    def start(self, on_utterance: UtteranceHandler) -> None:
        self._on_utterance = on_utterance
        try:
            self._hook.add_press(self._hotkey, self._on_press)
            self._hook.add_release(self._hotkey, self._on_release)
        except Exception as exc:
            self._hook.remove_all()
            raise HotkeyError(
                f"não foi possível registrar a hotkey {self._hotkey!r}: {exc}. "
                "Grave outra tecla no painel (Configurações → Ativação) ou edite "
                "activation.ptt_hotkey no config.toml (ex.: 'right ctrl', 'f9')."
            ) from exc
        log.info("push-to-talk pronto: segure %r para falar", self._hotkey)

    def stop(self) -> None:
        self._hook.remove_all()
        with self._lock:
            self._session = None
            self._pre_roll.clear()
            self._pre_roll_size = 0
        self._on_utterance = None

    def feed(self, block: np.ndarray, t_block_end: float) -> None:
        chunk = np.array(block, dtype=np.float32, copy=True)
        with self._lock:
            if self._session is not None:
                self._session.chunks.append(chunk)
                return
            if self._pre_roll_samples == 0:
                return
            self._pre_roll.append(chunk)
            self._pre_roll_size += chunk.size
            while self._pre_roll_size - self._pre_roll[0].size >= self._pre_roll_samples:
                self._pre_roll_size -= self._pre_roll.popleft().size

    def _on_press(self) -> None:
        t_press = time.perf_counter()
        with self._lock:
            if self._session is not None:
                return
            pre_roll = self._take_pre_roll()
            captured = sum(chunk.size for chunk in pre_roll)
            self._session = _PressSession(
                self._id_factory(), t_press - captured / self._sample_rate, pre_roll
            )
            utt_id = self._session.utt_id
        self._bus.publish(PhaseChanged(phase="listening", utt_id=utt_id))

    def _take_pre_roll(self) -> list[np.ndarray]:
        chunks = list(self._pre_roll)
        self._pre_roll.clear()
        self._pre_roll_size = 0
        if not chunks:
            return []
        head = _concatenate(chunks)
        return [head[-self._pre_roll_samples :] if head.size > self._pre_roll_samples else head]

    def _on_release(self) -> None:
        t_release = time.perf_counter()
        with self._lock:
            session, self._session = self._session, None
        if session is None:
            return
        self._bus.publish(PhaseChanged(phase="idle", utt_id=session.utt_id))
        samples = _concatenate(session.chunks)
        if samples.size == 0:
            log.debug("%s: hotkey solta sem áudio capturado; ignorado", session.utt_id)
            return
        utterance = Utterance(
            utt_id=session.utt_id,
            audio=AudioSegment(samples, self._sample_rate, t_end=t_release),
            t_release=t_release,
            t_press=session.t_press,
            t_vad_end=None,
            source="mic",
        )
        _deliver(self._on_utterance, utterance)


@dataclass
class _OpenUtterance:
    utt_id: str
    t_start: float
    frames: list[np.ndarray]
    pre_roll_frames: int
    t_last_speech: float
    speech_span_frames: int = 1
    silence_s: float = 0.0

    @property
    def total_samples(self) -> int:
        return sum(len(frame) for frame in self.frames)

    def mark_speech(self, t_frame_end: float) -> None:
        self.t_last_speech = t_frame_end
        self.silence_s = 0.0
        self.speech_span_frames = len(self.frames) - self.pre_roll_frames


class VadActivation:
    def __init__(
        self,
        detector: SpeechDetector,
        config: VadConfig,
        bus: EventSink,
        id_factory: IdFactory,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        pre_roll_ms: int = DEFAULT_PRE_ROLL_MS,
    ) -> None:
        self._detector = detector
        self._config = config
        self._bus = bus
        self._id_factory = id_factory
        self._sample_rate = sample_rate
        self._frame_seconds = FRAME_SAMPLES / sample_rate
        self._max_samples = int(config.max_utterance_s * sample_rate)
        self._pre_roll: deque[np.ndarray] = deque(maxlen=max(1, math.ceil(pre_roll_ms / (self._frame_seconds * 1000.0))))
        self._pending = np.zeros(0, dtype=np.float32)
        self._current: _OpenUtterance | None = None
        self._lock = threading.Lock()
        self._on_utterance: UtteranceHandler | None = None

    @property
    def mode(self) -> str:
        return "vad"

    @property
    def listening(self) -> bool:
        with self._lock:
            return self._current is not None

    def start(self, on_utterance: UtteranceHandler) -> None:
        self._on_utterance = on_utterance
        self._reset()
        log.info(
            "VAD pronto: limiar %.2f, tail %d ms, mínimo %d ms, máximo %.1f s",
            self._config.threshold,
            self._config.tail_ms,
            self._config.min_utterance_ms,
            self._config.max_utterance_s,
        )

    def stop(self) -> None:
        self._reset()
        self._on_utterance = None

    def feed(self, block: np.ndarray, t_block_end: float) -> None:
        with self._lock:
            utterances = [
                utterance
                for frame, t_frame_end in self._cut_frames(block, t_block_end)
                if (utterance := self._advance(frame, t_frame_end)) is not None
            ]
        for utterance in utterances:
            _deliver(self._on_utterance, utterance)

    def _reset(self) -> None:
        with self._lock:
            self._detector.reset()
            self._pre_roll.clear()
            self._pending = np.zeros(0, dtype=np.float32)
            self._current = None

    def _cut_frames(self, block: np.ndarray, t_block_end: float) -> list[tuple[np.ndarray, float]]:
        data = np.concatenate([self._pending, np.asarray(block, dtype=np.float32).reshape(-1)])
        frame_count = len(data) // FRAME_SAMPLES
        consumed = frame_count * FRAME_SAMPLES
        self._pending = data[consumed:].copy()
        t_consumed_end = t_block_end - len(self._pending) / self._sample_rate
        return [
            (
                data[index * FRAME_SAMPLES : (index + 1) * FRAME_SAMPLES].copy(),
                t_consumed_end - (frame_count - 1 - index) * self._frame_seconds,
            )
            for index in range(frame_count)
        ]

    def _advance(self, frame: np.ndarray, t_frame_end: float) -> Utterance | None:
        is_speech = self._detector.probability(frame) >= self._config.threshold
        if self._current is None:
            if is_speech:
                self._open(frame, t_frame_end)
            else:
                self._pre_roll.append(frame)
            return None
        current = self._current
        current.frames.append(frame)
        if is_speech:
            current.mark_speech(t_frame_end)
        else:
            current.silence_s += self._frame_seconds
        tail_s = self._config.tail_ms / 1000.0
        if current.silence_s >= tail_s:
            return self._close(current, current.t_last_speech + tail_s)
        if current.total_samples >= self._max_samples:
            return self._close(current, t_frame_end, trim_to=self._max_samples)
        return None

    def _open(self, frame: np.ndarray, t_frame_end: float) -> None:
        frames = [*self._pre_roll, frame]
        self._pre_roll.clear()
        self._current = _OpenUtterance(
            utt_id=self._id_factory(),
            t_start=t_frame_end - len(frames) * self._frame_seconds,
            frames=frames,
            pre_roll_frames=len(frames) - 1,
            t_last_speech=t_frame_end,
        )
        self._bus.publish(PhaseChanged(phase="listening", utt_id=self._current.utt_id))

    def _close(self, current: _OpenUtterance, t_vad_end: float, trim_to: int | None = None) -> Utterance | None:
        self._current = None
        self._bus.publish(PhaseChanged(phase="idle", utt_id=current.utt_id))
        speech_ms = current.speech_span_frames * self._frame_seconds * 1000.0
        if speech_ms < self._config.min_utterance_ms:
            log.debug("%s: fala de %.0f ms abaixo do mínimo; descartada", current.utt_id, speech_ms)
            return None
        samples = _concatenate(current.frames)
        if trim_to is not None and len(samples) > trim_to:
            t_vad_end -= (len(samples) - trim_to) / self._sample_rate
            samples = samples[:trim_to]
        return Utterance(
            utt_id=current.utt_id,
            audio=AudioSegment(samples, self._sample_rate, t_end=t_vad_end),
            t_release=t_vad_end,
            t_press=current.t_start,
            t_vad_end=t_vad_end,
            source="mic",
        )


def capture_hotkey(timeout_s: float = 10.0, hook: KeyboardHook | None = None) -> str | None:
    """Devolve a próxima tecla pressionada, já validada contra o mesmo hook que o push-to-talk usará;
    ``None`` quando ninguém apertou nada dentro do prazo."""
    active = hook if hook is not None else KeyboardLibHook()
    try:
        key = active.read_key(timeout_s)
    except HotkeyError:
        raise
    except Exception as exc:
        raise HotkeyError(
            "o hook global de teclado não está disponível para gravar a tecla "
            f"({exc.__class__.__name__}: {exc}). Rode o painel com permissão para ler o teclado, "
            "ou edite activation.ptt_hotkey no config.toml à mão."
        ) from exc
    if key is None:
        return None
    _ensure_usable(active, key)
    return key


def _ensure_usable(hook: KeyboardHook, key: str) -> None:
    try:
        hook.add_press(key, _noop)
        hook.add_release(key, _noop)
    except Exception as exc:
        raise HotkeyError(
            f"a tecla {key!r} foi capturada, mas o hook global não consegue usá-la: {exc}. "
            "Grave outra tecla (ex.: 'right ctrl', 'f9')."
        ) from exc
    finally:
        hook.remove_all()


def _noop() -> None:
    return None


def build_activation(
    config: ActivationConfig,
    bus: EventSink,
    id_factory: IdFactory,
    detector_factory: DetectorFactory | None = None,
    hook: KeyboardHook | None = None,
) -> Activation:
    if config.mode == "ptt":
        return PushToTalkActivation(
            config.ptt_hotkey, bus, id_factory, hook=hook, pre_roll_ms=config.pre_roll_ms
        )
    detector = (detector_factory or _default_detector)(config.vad.threshold)
    return VadActivation(detector, config.vad, bus, id_factory, pre_roll_ms=config.pre_roll_ms)


def _default_detector(threshold: float) -> SpeechDetector:
    return SileroVad(threshold)


def _concatenate(chunks: list[np.ndarray]) -> np.ndarray:
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32, copy=False)


def _deliver(handler: UtteranceHandler | None, utterance: Utterance) -> None:
    if handler is None:
        return
    try:
        handler(utterance)
    except Exception:
        log.exception("consumidor do enunciado %s falhou", utterance.utt_id)
