"""Composition root do mod: monta as quatro camadas, opera o ciclo de vida e desliga limpo."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Any, Callable

from cyhmo.app.boot_progress import BootProgress
from cyhmo.app.logging_setup import setup_logging
from cyhmo.config.loader import ConfigStore
from cyhmo.config.schema import AppConfig, ProjectPaths
from cyhmo.domain.errors import AudioDeviceError, InjectionError, LanguagePackError, CyhmoError
from cyhmo.domain.events import ComponentChanged, GrammarChanged, LogLine
from cyhmo.domain.ports import CommandInjector, Transcriber
from cyhmo.inject import PineClient, PineConnectError, WriteRecipe, build_injector
from cyhmo.intent import build_interpreter
from cyhmo.intent.interpreter import IntentInterpreter
from cyhmo.intent.language_packs import LanguagePackSet
from cyhmo.intent.llm.factory import build_llm_fallback
from cyhmo.pipeline.budget import BUDGET_MS
from cyhmo.pipeline.bus import EventBus
from cyhmo.pipeline.orchestrator import VoicePipeline
from cyhmo.pipeline.recorder import UtteranceRecorder
from cyhmo.pipeline.session import Session
from cyhmo.pipeline.telemetry import TelemetryWriter
from cyhmo.state import GameStateService, build_state_service
from cyhmo.stt import CaptureService, build_transcriber, resolve_stt_language
from cyhmo.update.service import UpdateService

log = logging.getLogger("cyhmo.app")

LOG_SOURCE = "app"
EXIT_MODE_STOP = "stop"
EXIT_MODE_RESTART = "restart"
EXIT_MODE_UPDATE = "update"
PIPELINE_START_TIMEOUT_S = 5.0
TASK_STOP_TIMEOUT_S = 3.0
POLL_INTERVAL_S = 0.01
CONFIG_ID_LENGTH = 8
CTRL_CLOSE_EVENT = 2
CTRL_LOGOFF_EVENT = 5
CTRL_SHUTDOWN_EVENT = 6
CONSOLE_CLOSE_EVENTS = frozenset({CTRL_CLOSE_EVENT, CTRL_LOGOFF_EVENT, CTRL_SHUTDOWN_EVENT})


class AppError(CyhmoError):
    """Falha ao montar ou operar a aplicação."""


def config_id(config: AppConfig) -> str:
    """Identidade estável da configuração efetiva, gravada na telemetria de cada enunciado."""
    material = json.dumps(config.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
    return "cfg-" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:CONFIG_ID_LENGTH]


class Application:
    """Objeto montado por ``build()``: expõe as camadas prontas e o ciclo de vida ``run()``/``shutdown()``."""

    def __init__(
        self,
        config_store: ConfigStore,
        config: AppConfig,
        paths: ProjectPaths,
        bus: EventBus,
        session: Session,
        packs: LanguagePackSet,
        interpreter: IntentInterpreter,
        state_service: GameStateService,
        injector: CommandInjector,
        transcriber: Transcriber,
        telemetry: TelemetryWriter,
        pipeline: VoicePipeline,
        capture: CaptureService,
        pine_client: PineClient | None,
        recipe: WriteRecipe | None,
        updates: UpdateService,
        llm: Any = None,
        progress: BootProgress | None = None,
    ) -> None:
        self._config_store = config_store
        self._config = config
        self._paths = paths
        self._bus = bus
        self._session = session
        self._packs = packs
        self._interpreter = interpreter
        self._state_service = state_service
        self._injector = injector
        self._transcriber = transcriber
        self._telemetry = telemetry
        self._pipeline = pipeline
        self._capture = capture
        self._pine_client = pine_client
        self._recipe = recipe
        self._updates = updates
        self._llm = llm
        self._progress = progress or BootProgress.silent()
        updates.on_ready = self.request_update
        self._viewmodel: Any = None
        self._fastapi: Any = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_event: asyncio.Event | None = None
        self._stop_requested = False
        self._exit_mode = EXIT_MODE_STOP
        self._shutdown_done = False
        self._pipeline_task: asyncio.Task[None] | None = None
        self._server_task: asyncio.Task[None] | None = None
        self._previous_signals: list[tuple[int, Any]] = []
        self._console_handler: Any = None
        self._unsubscribe_grammar: Callable[[], None] | None = None

    @classmethod
    def build(
        cls,
        config_path: Path | None = None,
        headless: bool = False,
        record: bool = False,
        debug: bool = False,
        inject_enabled: bool | None = None,
        open_browser: bool | None = None,
        connect_pine: bool = True,
        progress: BootProgress | None = None,
    ) -> "Application":
        """``inject_enabled``, ``open_browser`` e ``connect_pine`` sobrepõem a config apenas
        em memória: o ``config.toml`` do usuário nunca é reescrito por uma opção de linha de
        comando."""
        progress = progress or BootProgress.silent()
        store = ConfigStore.open(config_path)
        config = _with_overrides(store.config, inject_enabled, open_browser)
        paths = store.paths
        paths.ensure_directories()
        setup_logging(config.ui.log_level, paths.log_file, debug)
        log.info("configuração carregada de %s", store.path)

        bus = EventBus()
        boot_events = _BootLog(bus)
        session = Session(config_id=config_id(config))
        with progress.step("pacotes de idioma"):
            packs = _load_packs(config, paths)
        with progress.step("modelo de comparação de comandos"):
            llm = _build_llm(config, bus)
            interpreter = build_interpreter(config, paths, packs=packs, llm_fallback=llm, bus=bus)
        bus.publish(ComponentChanged(component="intent", status="ready", detail=_intent_detail(config, packs)))

        with progress.step("conexão com o PCSX2"):
            client = _connect_pine(config, bus) if connect_pine else None
            recipe = _load_recipe(paths, config, bus) if client is not None else None
            if client is not None and recipe is None:
                client.close()
                client = None

            state_service = build_state_service(config, paths, client, recipe, bus)
            injector = build_injector(
                config,
                paths,
                client,
                grammar_gate=state_service.accepts,
                can_talk_reader=lambda: state_service.read_state().can_talk,
                bus=bus,
            )
        bus.publish(ComponentChanged(component="inject", status="ready", detail=_inject_detail(config, client)))
        bus.publish(ComponentChanged(component="state", status="loading", detail=_state_detail(config, client)))

        with progress.step("reconhecimento de fala"):
            transcriber = build_transcriber(
                config.stt,
                resolve_stt_language(config.stt, packs.stt_language),
                paths.models_dir,
                packs.primary,
                paths.base_dir,
            )
        bus.publish(ComponentChanged(component="stt", status="ready", detail=_stt_detail(config, transcriber)))

        telemetry = TelemetryWriter(paths.telemetry_dir, session)
        pipeline = VoicePipeline(
            config,
            bus,
            transcriber,
            interpreter,
            state_service,
            injector,
            session,
            telemetry=telemetry,
            recorder=UtteranceRecorder(paths.audio_dir),
            record=record,
        )
        capture = CaptureService(config, bus, pipeline.submit, session.id, paths=paths)

        application = cls(
            store,
            config,
            paths,
            bus,
            session,
            packs,
            interpreter,
            state_service,
            injector,
            transcriber,
            telemetry,
            pipeline,
            capture,
            client,
            recipe,
            UpdateService(store, paths, bus),
            llm,
            progress,
        )
        application._unsubscribe_grammar = pipeline.add_grammar_listener(application._on_grammar_change)
        if not headless:
            application._attach_ui(boot_events.close())
        else:
            boot_events.close()
        return application

    @property
    def config_store(self) -> ConfigStore:
        return self._config_store

    @property
    def config(self) -> AppConfig:
        return self._config

    @property
    def paths(self) -> ProjectPaths:
        return self._paths

    @property
    def bus(self) -> EventBus:
        return self._bus

    @property
    def session(self) -> Session:
        return self._session

    @property
    def packs(self) -> LanguagePackSet:
        return self._packs

    @property
    def interpreter(self) -> IntentInterpreter:
        return self._interpreter

    @property
    def state_service(self) -> GameStateService:
        return self._state_service

    @property
    def injector(self) -> CommandInjector:
        return self._injector

    @property
    def transcriber(self) -> Transcriber:
        return self._transcriber

    @property
    def pipeline(self) -> VoicePipeline:
        return self._pipeline

    @property
    def capture(self) -> CaptureService:
        return self._capture

    @property
    def pine_client(self) -> PineClient | None:
        return self._pine_client

    @property
    def recipe(self) -> WriteRecipe | None:
        return self._recipe

    @property
    def updates(self) -> UpdateService:
        return self._updates

    @property
    def viewmodel(self) -> Any:
        return self._viewmodel

    @property
    def app(self) -> Any:
        return self._fastapi

    async def run(self) -> None:
        """Aquece, sobe as camadas e fica no ar até ``shutdown()`` ou Ctrl+C."""
        loop = asyncio.get_running_loop()
        self._loop = loop
        self._stop_event = asyncio.Event()
        if self._stop_requested:
            self._stop_event.set()
        self._install_signal_handlers(loop)
        try:
            with self._progress.step("aquecendo o reconhecimento"):
                await self.start_pipeline()
            self._state_service.start()
            self._start_capture()
            self._server_task = self._start_server(loop)
            self._updates.start()
            self._progress.ready(self._ui_url() if self._fastapi is not None else "")
            await self._stop_event.wait()
        except (KeyboardInterrupt, asyncio.CancelledError):
            log.info("interrupção recebida; encerrando")
        finally:
            self.shutdown()
            await self._finish_tasks()
            self._restore_signal_handlers()

    async def start_pipeline(self) -> None:
        """Aquece e põe o pipeline no ar, antes de ligar captura e interface."""
        if self._pipeline_task is not None:
            return
        self._loop = asyncio.get_running_loop()
        await self._pipeline.warm_up()
        self._pipeline_task = self._loop.create_task(self._pipeline.run())
        await self._wait_pipeline_running()

    def shutdown(self) -> None:
        """Encerra tudo na ordem segura; idempotente e chamável fora do loop asyncio."""
        self._request_stop()
        if self._shutdown_done:
            return
        self._shutdown_done = True
        for label, action in self._teardown_steps():
            try:
                action()
            except Exception:
                log.exception("falha ao encerrar %s", label)

    def _teardown_steps(self) -> tuple[tuple[str, Callable[[], Any]], ...]:
        return (
            ("captura", self._capture.stop),
            ("atualizações", self._updates.close),
            ("estado", self._state_service.stop),
            ("ouvinte de gramática", self._drop_grammar_listener),
            ("pipeline", self._pipeline.stop),
            ("servidor de STT", self._stop_transcriber),
            ("assistente", self._close_llm),
            ("injetor", self._close_injector),
            ("telemetria", self._telemetry.close),
            ("interface", self._close_viewmodel),
            ("PINE", self._close_pine),
        )

    def _drop_grammar_listener(self) -> None:
        if self._unsubscribe_grammar is not None:
            self._unsubscribe_grammar()
            self._unsubscribe_grammar = None

    def _stop_transcriber(self) -> None:
        """Só o backend whisper.cpp mantém processo próprio; sem isso o servidor
        sobreviveria ao mod e seguraria a GPU."""
        stop = getattr(self._transcriber, "stop", None)
        if callable(stop):
            stop()

    def _close_llm(self) -> None:
        """As threads do pool do assistente não são daemon: enquanto uma chamada estiver em voo,
        o processo não termina. Sem este passo, cada Reiniciar pela interface também deixava um
        pool de 4 threads vivo para sempre."""
        close = getattr(self._llm, "close", None)
        if callable(close):
            close()

    def _close_injector(self) -> None:
        """Restaura o patch e solta o handler de ``atexit`` do injetor; sem isso cada reinício
        pela interface deixa vivo mais um injetor velho, capaz de reabrir a PINE ao fim do processo."""
        close = getattr(self._injector, "close", None)
        if callable(close):
            close()

    def _close_viewmodel(self) -> None:
        if self._viewmodel is not None:
            self._viewmodel.close()

    def _close_pine(self) -> None:
        if self._pine_client is not None:
            self._pine_client.close()

    def _attach_ui(self, boot_events: tuple[ComponentChanged, ...]) -> None:
        """A ViewModel nasce depois do boot: os eventos de componente já publicados são reapresentados."""
        from cyhmo.app.services import AppServices
        from cyhmo.ui.server import create_app
        from cyhmo.ui.viewmodel import UiViewModel

        budget = {stage: int(value) for stage, value in BUDGET_MS.items()}
        self._viewmodel = UiViewModel(self._bus, self._config_store, AppServices(self), self._session.id, budget)
        for event in boot_events:
            self._viewmodel.handle_event(event)
        self._fastapi = create_app(self._viewmodel)
        self._bus.publish(ComponentChanged(component="ui", status="ready", detail=self._ui_url()))

    def _ui_url(self) -> str:
        return f"http://{self._config.ui.host}:{self._config.ui.port}/"

    def _on_grammar_change(self, event: GrammarChanged) -> None:
        log.debug(
            "gramática reindexada: %d entradas, %d inéditas na sessão (%.0f ms)",
            event.size,
            event.new_in_session,
            event.elapsed_ms,
        )

    def _start_capture(self) -> None:
        try:
            self._capture.start()
        except AudioDeviceError as exc:
            self._bus.publish(LogLine(level="error", message=f"microfone indisponível: {exc}", source=LOG_SOURCE))
            raise

    def _start_server(self, loop: asyncio.AbstractEventLoop) -> asyncio.Task[None] | None:
        if self._fastapi is None:
            return None
        from cyhmo.ui.server import serve

        ui = self._config.ui
        log.info("interface em %s", self._ui_url())
        return loop.create_task(
            serve(self._fastapi, ui.host, ui.port, ui.open_browser, self._stop_event)
        )

    async def _wait_pipeline_running(self) -> None:
        task = self._pipeline_task
        deadline = time.perf_counter() + PIPELINE_START_TIMEOUT_S
        while not self._pipeline.is_running:
            if task is not None and task.done():
                await task
                return
            if time.perf_counter() > deadline:
                raise AppError(f"o pipeline não iniciou em {PIPELINE_START_TIMEOUT_S:.0f} s")
            await asyncio.sleep(POLL_INTERVAL_S)

    async def _finish_tasks(self) -> None:
        for task in (self._server_task, self._pipeline_task):
            if task is None:
                continue
            task.cancel()
            try:
                await asyncio.wait_for(task, TASK_STOP_TIMEOUT_S)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                continue
            except Exception:
                log.exception("tarefa encerrou com erro")
        self._server_task = None
        self._pipeline_task = None

    @property
    def exit_mode(self) -> str:
        """``restart`` faz o CLI reconstruir a aplicação em vez de sair — é como uma
        mudança de configuração passa a valer sem o usuário procurar o terminal."""
        return self._exit_mode

    def request_restart(self) -> None:
        log.info("reinício solicitado pela interface")
        self._exit_mode = EXIT_MODE_RESTART
        self._request_stop()

    def request_update(self) -> None:
        """Chamada pelo serviço de atualização quando o pacote novo está preparado: a troca
        dos arquivos é do encerramento, com o mod já parado."""
        log.info("atualização preparada; encerrando para aplicar")
        self._exit_mode = EXIT_MODE_UPDATE
        self._request_stop()

    def request_exit(self) -> None:
        log.info("encerramento solicitado pela interface")
        self._exit_mode = EXIT_MODE_STOP
        self._request_stop()

    def _request_stop(self) -> None:
        self._stop_requested = True
        loop, event = self._loop, self._stop_event
        if loop is None or event is None or loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(event.set)
        except RuntimeError:
            log.debug("loop já encerrado ao pedir parada")

    def _install_signal_handlers(self, loop: asyncio.AbstractEventLoop) -> None:
        for name in ("SIGINT", "SIGTERM"):
            number = getattr(signal, name, None)
            if number is None:
                continue
            try:
                loop.add_signal_handler(number, self._request_stop)
            except (NotImplementedError, RuntimeError, ValueError, AttributeError, OSError):
                self._install_fallback_handler(number)
        self._install_console_close_handler()

    def _install_console_close_handler(self) -> None:
        """Fechar a janela do terminal no Windows não vira sinal nenhum para o Python: o
        processo é morto inteiro e deixa para trás o servidor de STT segurando a porta e o
        patch do jogo aplicado na memória. O gancho do console dá os poucos segundos que o
        encerramento limpo precisa.

        A referência do callback fica no objeto de propósito: coletá-la deixaria o Windows
        chamando memória já liberada."""
        if sys.platform != "win32":
            return
        import ctypes

        def handle(event: int) -> int:
            if event not in CONSOLE_CLOSE_EVENTS:
                return 0
            log.info("janela do terminal fechada; encerrando o mod")
            self.shutdown()
            return 1

        callback = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_uint)(handle)
        try:
            installed = bool(ctypes.windll.kernel32.SetConsoleCtrlHandler(callback, True))
        except (AttributeError, OSError):
            return
        if installed:
            self._console_handler = callback

    def _install_fallback_handler(self, number: int) -> None:
        try:
            previous = signal.getsignal(number)
            signal.signal(number, lambda *_: self._request_stop())
        except (ValueError, OSError, TypeError):
            log.debug("sem handler de sinal para %s nesta thread", number)
            return
        self._previous_signals.append((number, previous))

    def _restore_signal_handlers(self) -> None:
        loop = self._loop
        if loop is not None and not loop.is_closed():
            for name in ("SIGINT", "SIGTERM"):
                number = getattr(signal, name, None)
                if number is not None:
                    try:
                        loop.remove_signal_handler(number)
                    except (NotImplementedError, RuntimeError, ValueError):
                        continue
        for number, previous in self._previous_signals:
            try:
                signal.signal(number, previous)
            except (ValueError, OSError, TypeError):
                continue
        self._previous_signals.clear()
        self._restore_console_close_handler()

    def _restore_console_close_handler(self) -> None:
        """Cada volta do reinício instala o gancho de novo; sem soltar o anterior eles se
        empilham, um por aplicação já morta."""
        callback, self._console_handler = self._console_handler, None
        if callback is None:
            return
        import ctypes

        try:
            ctypes.windll.kernel32.SetConsoleCtrlHandler(callback, False)
        except (AttributeError, OSError):
            return


class _BootLog:
    """Guarda os ``ComponentChanged`` do boot para a interface, que só assina o barramento depois."""

    def __init__(self, bus: EventBus) -> None:
        self._events: list[ComponentChanged] = []
        self._unsubscribe = bus.subscribe(self._record)

    def _record(self, event: Any) -> None:
        if isinstance(event, ComponentChanged):
            self._events.append(event)

    def close(self) -> tuple[ComponentChanged, ...]:
        self._unsubscribe()
        return tuple(self._events)


def _with_overrides(config: AppConfig, inject_enabled: bool | None, open_browser: bool | None) -> AppConfig:
    sections: dict[str, Any] = {}
    if inject_enabled is not None and config.inject.enabled != inject_enabled:
        sections["inject"] = config.inject.model_copy(update={"enabled": inject_enabled})
    if open_browser is not None and config.ui.open_browser != open_browser:
        sections["ui"] = config.ui.model_copy(update={"open_browser": open_browser})
    return config.replace(**sections) if sections else config


def _load_packs(config: AppConfig, paths: ProjectPaths) -> LanguagePackSet:
    try:
        return LanguagePackSet.load(paths.packs_dir, config.languages.enabled, config.languages.primary)
    except LanguagePackError as exc:
        available, problems = LanguagePackSet.available(paths.packs_dir)
        codes = ", ".join(pack.code for pack in available) or "(nenhum)"
        detail = "; ".join(problems)
        raise LanguagePackError(
            f"{exc}\nPacotes válidos em {paths.packs_dir}: {codes}"
            + (f"\nPacotes com problema: {detail}" if detail else "")
        ) from exc


def _build_llm(config: AppConfig, bus: EventBus) -> Any:
    """Ligado na config mas quebrado vira status ``error``, não ``off``: o cinza de ``off``
    é indistinguível de desligado de propósito e o usuário fica sem fallback sem saber."""
    try:
        fallback = build_llm_fallback(config.intent.llm, bus)
    except CyhmoError as exc:
        bus.publish(LogLine(level="error", message=f"fallback LLM ligado mas indisponível: {exc}", source=LOG_SOURCE))
        bus.publish(ComponentChanged(component="llm", status="error", detail=str(exc)))
        log.error("fallback LLM ligado na config mas indisponível: %s", exc)
        return None
    if fallback is None:
        bus.publish(ComponentChanged(component="llm", status="off", detail="desabilitado na config"))
        return None
    detail = f"{config.intent.llm.provider}: {config.intent.llm.model or '(modelo padrão)'}"
    bus.publish(ComponentChanged(component="llm", status="ready", detail=detail))
    return fallback


def _connect_pine(config: AppConfig, bus: EventBus) -> PineClient:
    """O cliente sobrevive à falha da primeira conexão: ele reconecta sozinho a cada pedido,
    e a camada 3 pede o ponteiro da gramática a ``state.polling_hz``.

    Descartar o cliente aqui era o que tornava a conectividade uma escolha de TIPO feita no
    boot — quem abrisse o PCSX2 depois do mod ficava em dry-run pela sessão inteira, e não
    havia como saber disso sem reiniciar.
    """
    client = PineClient.from_config(config.pine)
    client.on_link_change = lambda connected, detail: _publish_link(bus, connected, detail)
    try:
        client.connect()
    except PineConnectError as exc:
        log.warning("PCSX2 ainda não respondeu em %s:%s: %s", config.pine.host, config.pine.port, exc)
    return client


def _publish_link(bus: EventBus, connected: bool, detail: str) -> None:
    """Chamado pelo cliente PINE só quando o elo muda de estado, de qualquer thread."""
    message = (
        f"conectado ao PCSX2 em {detail}"
        if connected
        else f"sem conexão com o PCSX2 — o mod reconecta sozinho quando o emulador abrir ({detail})"
    )
    bus.publish(ComponentChanged(component="pine", status="ready" if connected else "off", detail=message))
    bus.publish(LogLine(level="info" if connected else "warning", message=message, source=LOG_SOURCE))


def _load_recipe(paths: ProjectPaths, config: AppConfig, bus: EventBus) -> WriteRecipe | None:
    try:
        recipe = WriteRecipe.load(paths.recipe)
        recipe.validate_serial(config.pine.expected_serial)
    except InjectionError as exc:
        message = f"receita de injeção inutilizável ({paths.recipe}): {exc} — seguindo sem injeção (dry-run)"
        bus.publish(LogLine(level="warning", message=message, source=LOG_SOURCE))
        bus.publish(ComponentChanged(component="inject", status="error", detail=message))
        log.warning(message)
        return None
    return recipe



def _intent_detail(config: AppConfig, packs: LanguagePackSet) -> str:
    backend = config.intent.embedding_backend
    model = config.intent.embedding_model if backend == "sentence_transformers" else backend
    return f"{model}; idiomas {', '.join(packs.codes)} (primário {packs.primary.code})"


def _inject_detail(config: AppConfig, client: PineClient | None) -> str:
    """Sem tempo verbal: este detalhe é publicado uma vez, no boot, e o elo com o emulador
    muda depois. Quem mostra o elo ao vivo é o componente ``pine``."""
    if client is None or not config.inject.enabled:
        return "dry-run: nada é escrito na memória do jogo"
    return f"escrita ativa quando o jogo estiver conectado (patch {config.pine.patch_mode})"


def _state_detail(config: AppConfig, client: PineClient | None) -> str:
    if client is not None:
        return f"gramática viva a {config.state.polling_hz} Hz"
    if config.state.grammar_seed:
        return f"gramática semente: {config.state.grammar_seed}"
    return "sem fonte de gramática"


def _stt_detail(config: AppConfig, transcriber: Transcriber) -> str:
    return f"{config.stt.engine}: {transcriber.model_name}"
