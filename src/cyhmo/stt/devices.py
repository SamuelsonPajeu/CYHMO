"""Enumeração, seleção e listagem de dispositivos de captura."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any, Sequence

from cyhmo.domain.errors import AudioDeviceError

DEFAULT_SPEC = "default"
PREFERRED_HOST_APIS = ("Windows WASAPI", "Windows DirectSound", "MME", "Windows WDM-KS")
MIN_SHARED_PREFIX = 8
_NAME_WRAPPER = re.compile(r"^[^(]*\(")
_ENUMERATION_PREFIX = re.compile(r"^\d+-\s*")


@dataclass(frozen=True)
class CaptureDevice:
    index: int
    name: str
    default: bool
    sample_rate: int
    host_api: str
    channels: int

    @property
    def label(self) -> str:
        return f"[{self.index}] {self.name} ({self.host_api})"


def list_capture_devices(dedupe: bool = True) -> list[CaptureDevice]:
    import sounddevice as sd

    host_apis = [str(api["name"]) for api in sd.query_hostapis()]
    default_index = _default_input_index(sd)
    devices = [
        _to_capture_device(entry, host_apis, default_index)
        for entry in sd.query_devices()
        if int(entry["max_input_channels"]) > 0
    ]
    return _one_entry_per_microphone(devices) if dedupe else sorted(devices, key=lambda device: device.index)


def open_candidates(chosen: CaptureDevice, devices: Sequence[CaptureDevice] | None = None) -> list[CaptureDevice]:
    """O mesmo microfone existe em várias host APIs e nem todas abrem — o WDM-KS falha com frequência."""
    pool = list(devices) if devices is not None else list_capture_devices(dedupe=False)
    siblings = [
        device
        for device in pool
        if device.index != chosen.index and _same_microphone(device.name, chosen.name)
    ]
    return [chosen, *sorted(siblings, key=_host_api_rank)]


def resolve_device(spec: str | int | None, devices: Sequence[CaptureDevice]) -> CaptureDevice:
    if not devices:
        raise AudioDeviceError(
            "nenhum dispositivo de captura encontrado. Conecte o microfone e confira com --list-devices."
        )
    if _is_default_spec(spec):
        return _default_device(devices)
    if _is_index_spec(spec):
        return _by_index(int(spec), devices)
    return _by_name(str(spec), devices)


def format_devices(devices: Sequence[CaptureDevice]) -> str:
    if not devices:
        return "(nenhum dispositivo de captura)"
    header = ("idx", "padrão", "nome", "host API", "taxa nativa", "canais")
    rows = [
        (
            str(device.index),
            "*" if device.default else "",
            device.name,
            device.host_api,
            f"{device.sample_rate} Hz",
            str(device.channels),
        )
        for device in devices
    ]
    widths = [max(len(line[column]) for line in (header, *rows)) for column in range(len(header))]

    def render(line: tuple[str, ...]) -> str:
        return "  ".join(cell.ljust(widths[column]) for column, cell in enumerate(line)).rstrip()

    return "\n".join([render(header), *(render(row) for row in rows)])


def _default_input_index(sd: Any) -> int | None:
    try:
        index = sd.default.device[0]
    except Exception:
        return None
    return int(index) if isinstance(index, int) and index >= 0 else None


def _to_capture_device(entry: Any, host_apis: list[str], default_index: int | None) -> CaptureDevice:
    host_api_index = int(entry["hostapi"])
    host_api = host_apis[host_api_index] if 0 <= host_api_index < len(host_apis) else "?"
    return CaptureDevice(
        index=int(entry["index"]),
        name=str(entry["name"]).strip(),
        default=int(entry["index"]) == default_index,
        sample_rate=int(round(float(entry["default_samplerate"]))),
        host_api=host_api,
        channels=int(entry["max_input_channels"]),
    )


def _one_entry_per_microphone(devices: list[CaptureDevice]) -> list[CaptureDevice]:
    """O mesmo microfone aparece uma vez por host API; o MME trunca o nome em 31 caracteres e o
    Windows ainda prefixa o endpoint com um número ("2- Microfone"), então agrupamos pelo nome-núcleo."""
    groups: list[list[CaptureDevice]] = []
    for device in devices:
        for group in groups:
            if _same_microphone(device.name, group[0].name):
                group.append(device)
                break
        else:
            groups.append([device])
    chosen = []
    for members in groups:
        best = min(members, key=_host_api_rank)
        is_default = any(member.default for member in members)
        chosen.append(replace(best, default=is_default))
    return sorted(chosen, key=lambda device: device.index)


def _same_microphone(left: str, right: str) -> bool:
    """O MME corta o nome em 31 caracteres, então comparar por prefixo é o que reúne o mesmo
    microfone visto por host APIs diferentes."""
    first, second = _core_name(left), _core_name(right)
    if first == second:
        return True
    shorter, longer = sorted((first, second), key=len)
    return len(shorter) >= MIN_SHARED_PREFIX and longer.startswith(shorter)


def _core_name(name: str) -> str:
    unwrapped = _NAME_WRAPPER.sub("", name.strip()).rstrip(")")
    return _ENUMERATION_PREFIX.sub("", unwrapped).casefold().strip()


def _host_api_rank(device: CaptureDevice) -> int:
    if device.host_api in PREFERRED_HOST_APIS:
        return PREFERRED_HOST_APIS.index(device.host_api)
    return len(PREFERRED_HOST_APIS)


def _is_default_spec(spec: str | int | None) -> bool:
    return spec is None or (isinstance(spec, str) and spec.strip().lower() in ("", DEFAULT_SPEC))


def _is_index_spec(spec: str | int | None) -> bool:
    return isinstance(spec, int) or (isinstance(spec, str) and spec.strip().isdigit())


def _default_device(devices: Sequence[CaptureDevice]) -> CaptureDevice:
    for device in devices:
        if device.default:
            return device
    return devices[0]


def _by_index(index: int, devices: Sequence[CaptureDevice]) -> CaptureDevice:
    for device in devices:
        if device.index == index:
            return device
    raise AudioDeviceError(_not_found_message(str(index), devices))


def _by_name(needle: str, devices: Sequence[CaptureDevice]) -> CaptureDevice:
    """Um nome gravado na config pode ser o de outra host API do mesmo microfone: casa por
    trecho e, se não achar, pelo nome-núcleo — senão trocar de host API perderia a escolha."""
    wanted = needle.strip().casefold()
    matches = [device for device in devices if wanted in device.name.casefold()]
    if not matches:
        matches = [device for device in devices if _same_microphone(needle, device.name)]
    if not matches:
        raise AudioDeviceError(_not_found_message(needle, devices))
    return _default_device(matches)


def _not_found_message(spec: str, devices: Sequence[CaptureDevice]) -> str:
    return (
        f"dispositivo de captura {spec!r} não encontrado (audio.device no config.toml). "
        f"Dispositivos disponíveis:\n{format_devices(devices)}"
    )
