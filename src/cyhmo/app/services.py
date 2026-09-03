"""Adaptador entre a ``Application`` e o protocolo ``UiServices`` da ViewModel."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Sequence

from cyhmo.app.calibration_job import CalibrationRunner
from cyhmo.domain.contracts import CommandRef, InjectResult, Interpretation, Transcript
from cyhmo.domain.errors import AudioDeviceError, LanguagePackError, CyhmoError
from cyhmo.intent.language_packs import LanguagePackSet
from cyhmo.intent.llm.ollama_admin import OllamaAdmin

if TYPE_CHECKING:
    from cyhmo.app.bootstrap import Application

DRY_RUN_STATUS = {"dry_run": True, "detail": "modo dry-run: nada é escrito na memória do jogo"}
OFFLINE_DETAIL = "sem conexão com o PCSX2: nada foi lido da memória do jogo"
MANAGED_PROVIDER = "ollama"


class AppServices:
    """Cada método é uma ação da interface; erros previstos viram ``CyhmoError`` legível."""

    def __init__(self, application: "Application") -> None:
        self._app = application
        self._ollama = OllamaAdmin(endpoint=application.config.intent.llm.endpoint)
        self._calibration = CalibrationRunner(config=application.config, paths=application.paths)

    def list_devices(self) -> list[dict[str, Any]]:
        from cyhmo.stt import list_capture_devices

        try:
            devices = list_capture_devices()
        except CyhmoError:
            raise
        except Exception as exc:
            raise AudioDeviceError(
                f"não consegui listar os dispositivos de captura: {exc}. "
                "Instale as dependências de áudio (sounddevice) e confira o microfone."
            ) from exc
        return [
            {
                "index": device.index,
                "name": device.name,
                "default": device.default,
                "sample_rate": device.sample_rate,
                "host_api": device.host_api,
            }
            for device in devices
        ]

    def available_language_packs(self) -> list[dict[str, Any]]:
        packs, problems = LanguagePackSet.available(self._app.paths.packs_dir)
        entries = [
            {
                "code": pack.code,
                "name": pack.name,
                "stt_language": pack.stt_language,
                "summary": pack.to_summary(),
                "valid": True,
            }
            for pack in packs
        ]
        entries.extend(
            {"code": "", "name": "(pacote inválido)", "stt_language": "", "summary": problem, "valid": False}
            for problem in problems
        )
        return entries

    def language_registry(self) -> list[dict[str, Any]]:
        from cyhmo.intent.pack_registry import fetch_registry

        url = self._live_config().languages.registry_url
        if not url.strip():
            raise LanguagePackError("nenhum índice de idiomas configurado em languages.registry_url")
        return [entry.to_dict(installed=False) for entry in fetch_registry(url)]

    def install_language_pack(self, code: str) -> dict[str, Any]:
        from cyhmo.intent.pack_registry import fetch_registry, install_pack

        url = self._live_config().languages.registry_url
        entry = next((item for item in fetch_registry(url) if item.code == code), None)
        if entry is None:
            raise LanguagePackError(f"o índice de idiomas não tem o pacote {code!r}")
        path = install_pack(url, entry, self._app.paths.packs_dir)
        return {"code": entry.code, "name": entry.name, "path": str(path)}

    def llm_status(self) -> dict[str, Any]:
        llm = self._live_config().intent.llm
        if llm.provider != MANAGED_PROVIDER:
            return {
                "managed": False,
                "provider": llm.provider,
                "model": llm.model,
                "endpoint": llm.endpoint,
                "enabled": llm.enabled,
            }
        self._ollama.endpoint = llm.endpoint
        return {"managed": True, "provider": llm.provider, "model": llm.model, "enabled": llm.enabled, **self._ollama.status()}

    def llm_pull(self, model: str) -> dict[str, Any]:
        llm = self._live_config().intent.llm
        if llm.provider != MANAGED_PROVIDER:
            raise CyhmoError(f"só sei baixar modelos do Ollama; o provedor atual é {llm.provider!r}")
        self._ollama.endpoint = llm.endpoint
        return self._ollama.start_pull(model)

    def llm_pull_cancel(self) -> dict[str, Any]:
        return self._ollama.cancel_pull()

    def llm_delete_model(self, model: str) -> dict[str, Any]:
        llm = self._live_config().intent.llm
        if llm.provider != MANAGED_PROVIDER:
            raise CyhmoError(f"só sei remover modelos do Ollama; o provedor atual é {llm.provider!r}")
        self._ollama.endpoint = llm.endpoint
        return self._ollama.delete_model(model)

    def calibration_status(self) -> dict[str, Any]:
        return self._calibration.status()

    def calibration_start(self, dataset: str, spontaneous: str, grid: str) -> dict[str, Any]:
        self._calibration.config = self._live_config()
        return self._calibration.start(dataset, spontaneous, grid)

    def calibration_cancel(self) -> dict[str, Any]:
        return self._calibration.cancel()

    def start_listening(self) -> None:
        self._app.capture.set_enabled(True)

    def stop_listening(self) -> None:
        self._app.capture.set_enabled(False)

    def is_listening(self) -> bool:
        return self._app.capture.enabled and self._app.capture.running

    def request_restart(self) -> None:
        self._app.request_restart()

    def request_exit(self) -> None:
        self._app.request_exit()

    def update_status(self) -> dict[str, Any]:
        return self._app.updates.status()

    def update_check(self) -> dict[str, Any]:
        return self._app.updates.check()

    def update_install(self) -> dict[str, Any]:
        return self._app.updates.install()

    def update_skip(self) -> dict[str, Any]:
        return self._app.updates.skip()

    def update_postpone(self) -> dict[str, Any]:
        return self._app.updates.postpone()

    def interpret_text(self, text: str) -> Interpretation:
        transcript = Transcript(
            text=text,
            lang=self._app.packs.stt_language,
            confidence=1.0,
            t_speech_end=time.perf_counter(),
        )
        return self._app.interpreter.interpret(transcript, self._app.state_service.read_state())

    def inject_text(self, text: str) -> InjectResult:
        return self.inject_commands([{"key": text}])

    def inject_commands(self, commands: Sequence[dict[str, Any]]) -> InjectResult:
        """Injeta o texto literal — sem interpretar — pelo mesmo portão de gramática do injetor."""
        refs = [CommandRef(key=str(command.get("key", "")), args=dict(command.get("args", {}))) for command in commands]
        return self._app.injector.inject(refs)

    def mic_test(self, seconds: float) -> dict[str, Any]:
        return dict(self._app.capture.mic_test(seconds))

    def capture_hotkey(self, timeout_s: float) -> str | None:
        from cyhmo.stt.activation import capture_hotkey

        return capture_hotkey(timeout_s)

    def grammar_entries(self) -> list[str]:
        grammar = self._app.state_service.read_state().grammar
        return list(grammar or ())

    def injector_status(self) -> dict[str, Any]:
        """O injetor real existe desde o boot, mesmo antes de o PCSX2 abrir: sem esta guarda
        a aba Ferramentas responderia com erro em vez do estado do elo."""
        from cyhmo.inject.pine import PineError

        reader = getattr(self._app.injector, "read_status", None)
        if not callable(reader):
            return dict(DRY_RUN_STATUS)
        try:
            return dict(reader())
        except PineError as exc:
            return {"connected": False, "detail": f"{OFFLINE_DETAIL} ({exc})"}

    def _live_config(self) -> Any:
        """A config salva pela interface só existe no ``ConfigStore``; a da ``Application``
        é a do boot e ficaria para trás em tudo que a interface acabou de mudar."""
        return self._app.config_store.config
