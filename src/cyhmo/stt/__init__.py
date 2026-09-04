"""Camada 1 — captura de áudio, ativação (PTT/VAD) e transcrição local.

Nenhuma dependência pesada é importada aqui: sounddevice, soxr, silero-vad,
faster-whisper e keyboard entram apenas dentro dos adapters que os usam.
"""

from cyhmo.stt.activation import (
    HotkeyError,
    KeyboardHook,
    PushToTalkActivation,
    VadActivation,
    build_activation,
    capture_hotkey,
)
from cyhmo.stt.capture import SoundDeviceCapture
from cyhmo.stt.confidence import confidence_from_segments
from cyhmo.stt.devices import (
    CaptureDevice,
    format_devices,
    list_capture_devices,
    open_candidates,
    resolve_device,
)
from cyhmo.stt.normalization import normalize_text
from cyhmo.stt.resampling import Resampler, to_mono
from cyhmo.stt.service import CaptureService
from cyhmo.stt.transcriber import (
    FakeTranscriber,
    FasterWhisperTranscriber,
    SttModelError,
    build_transcriber,
    pack_hotwords,
    resolve_stt_language,
)
from cyhmo.stt.vad import FRAME_SAMPLES, SileroVad, SpeechDetector
from cyhmo.stt.wav import write_wav
from cyhmo.stt.whisper_cpp import ServerSpec, WhisperCppError, WhisperCppTranscriber
from cyhmo.stt.whisper_gpu import (
    WhisperGpuAdmin,
    WhisperGpuError,
    effective_binary,
    nvidia_adapter,
)
from cyhmo.stt.whisper_models import (
    CATALOG,
    WhisperModel,
    WhisperModelAdmin,
    WhisperModelError,
)

__all__ = [
    "CATALOG",
    "CaptureDevice",
    "CaptureService",
    "FRAME_SAMPLES",
    "FakeTranscriber",
    "FasterWhisperTranscriber",
    "HotkeyError",
    "KeyboardHook",
    "PushToTalkActivation",
    "Resampler",
    "ServerSpec",
    "SileroVad",
    "SoundDeviceCapture",
    "SpeechDetector",
    "SttModelError",
    "VadActivation",
    "WhisperCppError",
    "WhisperCppTranscriber",
    "WhisperGpuAdmin",
    "WhisperGpuError",
    "WhisperModel",
    "WhisperModelAdmin",
    "WhisperModelError",
    "build_activation",
    "build_transcriber",
    "capture_hotkey",
    "confidence_from_segments",
    "effective_binary",
    "format_devices",
    "list_capture_devices",
    "normalize_text",
    "nvidia_adapter",
    "open_candidates",
    "pack_hotwords",
    "resolve_device",
    "resolve_stt_language",
    "to_mono",
    "write_wav",
]
