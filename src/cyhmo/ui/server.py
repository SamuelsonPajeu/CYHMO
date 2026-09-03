"""Servidor HTTP/WebSocket local da interface.

Roda no mesmo loop asyncio do pipeline; chamadas potencialmente lentas vão para
thread e o fluxo de eventos usa a fila descarta-o-antigo do barramento."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import socket
import threading
import webbrowser
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from cyhmo.domain.errors import CyhmoError, UiServerError
from cyhmo.ui.viewmodel import DEFAULT_GRAMMAR_LIMIT, UiViewModel

if TYPE_CHECKING:
    from fastapi import FastAPI, WebSocket

log = logging.getLogger("cyhmo.ui.server")

GRACEFUL_SHUTDOWN_S = 2.0

STATIC_DIR = Path(__file__).resolve().parent / "static"
MIC_TEST_MAX_SECONDS = 30.0
DEFAULT_HOTKEY_TIMEOUT = 10.0
HOTKEY_CAPTURE_MAX_SECONDS = 60.0


class ConfigPatchBody(BaseModel):
    patch: dict[str, Any]


class TextBody(BaseModel):
    text: str = ""


class CommandBody(BaseModel):
    key: str
    args: dict[str, Any] = Field(default_factory=dict)


class InjectBody(BaseModel):
    text: str | None = None
    commands: list[CommandBody] | None = None


class MicTestBody(BaseModel):
    seconds: float = Field(default=2.0, gt=0.0, le=MIC_TEST_MAX_SECONDS)


class LanguageCodeBody(BaseModel):
    code: str


class ModelBody(BaseModel):
    model: str = ""


class CalibrationBody(BaseModel):
    dataset: str = ""
    spontaneous: str = ""
    grid: str = ""


class HotkeyCaptureBody(BaseModel):
    timeout_s: float = Field(default=DEFAULT_HOTKEY_TIMEOUT, gt=0.0, le=HOTKEY_CAPTURE_MAX_SECONDS)


def _publish_route_annotations(**symbols: Any) -> None:
    """FastAPI resolve as anotações das rotas pelo namespace do módulo, não pelo local do import tardio."""
    globals().update(symbols)


def create_app(viewmodel: UiViewModel, static_dir: Path | None = None) -> "FastAPI":
    from fastapi import FastAPI, Query, Request, WebSocket
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles

    _publish_route_annotations(WebSocket=WebSocket, Request=Request)
    static_dir = static_dir or STATIC_DIR
    app = FastAPI(title="CYHMO", docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def always_revalidate_the_view(request: Request, call_next: Any) -> Any:
        """Sem ``Cache-Control`` o navegador aplica cache heurístico e pode servir a
        View antiga sem perguntar — foi como uma atualização do mod deixou um
        navegador com o JS velho e a página em branco. ``no-cache`` obriga a
        revalidar; o ETag mantém a resposta em 304, então não custa nada."""
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    @app.exception_handler(CyhmoError)
    async def expected_error(_request: Request, exc: CyhmoError) -> JSONResponse:
        return JSONResponse({"error": str(exc)}, status_code=400)

    @app.exception_handler(Exception)
    async def unexpected_error(_request: Request, exc: Exception) -> JSONResponse:
        log.exception("erro inesperado na interface")
        return JSONResponse({"error": f"erro interno na interface: {exc}"}, status_code=500)

    @app.get("/api/state")
    async def get_state() -> dict[str, Any]:
        return viewmodel.snapshot()

    @app.get("/api/config")
    async def get_config() -> dict[str, Any]:
        snapshot = viewmodel.snapshot()
        return {"config": snapshot["config"], "schema": snapshot["config_schema"]}

    @app.put("/api/config")
    async def put_config(body: ConfigPatchBody) -> dict[str, Any]:
        return viewmodel.apply_config_patch(body.patch)

    @app.get("/api/devices")
    async def get_devices(refresh: int = Query(default=0)) -> list[dict[str, Any]]:
        reader = viewmodel.refresh_devices if refresh else viewmodel.devices
        return await asyncio.to_thread(reader)

    @app.get("/api/i18n")
    async def get_i18n() -> dict[str, Any]:
        return viewmodel.i18n_bundle()

    @app.get("/api/languages")
    async def get_languages(refresh: int = Query(default=0)) -> dict[str, Any]:
        if refresh:
            return await asyncio.to_thread(viewmodel.refresh_languages)
        return viewmodel.snapshot()["languages"]

    @app.get("/api/languages/registry")
    async def get_language_registry() -> dict[str, Any]:
        return await asyncio.to_thread(viewmodel.language_registry)

    @app.post("/api/languages/install")
    async def post_language_install(body: LanguageCodeBody) -> dict[str, Any]:
        return await asyncio.to_thread(viewmodel.install_language_pack, body.code)

    @app.get("/api/llm")
    async def get_llm() -> dict[str, Any]:
        return await asyncio.to_thread(viewmodel.llm_status)

    @app.post("/api/llm/pull")
    async def post_llm_pull(body: ModelBody) -> dict[str, Any]:
        return await asyncio.to_thread(viewmodel.llm_pull, body.model)

    @app.post("/api/llm/pull/cancel")
    async def post_llm_pull_cancel() -> dict[str, Any]:
        return await asyncio.to_thread(viewmodel.llm_pull_cancel)

    @app.post("/api/llm/delete")
    async def post_llm_delete(body: ModelBody) -> dict[str, Any]:
        return await asyncio.to_thread(viewmodel.llm_delete_model, body.model)

    @app.get("/api/calibration")
    async def get_calibration() -> dict[str, Any]:
        return await asyncio.to_thread(viewmodel.calibration_status)

    @app.post("/api/calibration/start")
    async def post_calibration_start(body: CalibrationBody) -> dict[str, Any]:
        return await asyncio.to_thread(viewmodel.calibration_start, body.dataset, body.spontaneous, body.grid)

    @app.post("/api/calibration/cancel")
    async def post_calibration_cancel() -> dict[str, Any]:
        return await asyncio.to_thread(viewmodel.calibration_cancel)

    @app.post("/api/listen/start")
    async def listen_start() -> dict[str, Any]:
        return {"listening": await asyncio.to_thread(viewmodel.start_listening)}

    @app.post("/api/listen/stop")
    async def listen_stop() -> dict[str, Any]:
        return {"listening": await asyncio.to_thread(viewmodel.stop_listening)}

    @app.post("/api/restart")
    async def post_restart() -> dict[str, Any]:
        """A resposta sai antes do encerramento: quem pediu precisa saber que foi aceito."""
        return viewmodel.request_restart()

    @app.post("/api/exit")
    async def post_exit() -> dict[str, Any]:
        return viewmodel.request_exit()

    @app.get("/api/update")
    async def get_update() -> dict[str, Any]:
        return viewmodel.update_status()

    @app.post("/api/update/check")
    async def post_update_check() -> dict[str, Any]:
        return viewmodel.update_check()

    @app.post("/api/update/install")
    async def post_update_install() -> dict[str, Any]:
        """A resposta sai antes do download: quem pediu precisa saber que foi aceito."""
        return viewmodel.update_install()

    @app.post("/api/update/skip")
    async def post_update_skip() -> dict[str, Any]:
        return await asyncio.to_thread(viewmodel.update_skip)

    @app.post("/api/update/postpone")
    async def post_update_postpone() -> dict[str, Any]:
        return viewmodel.update_postpone()

    @app.post("/api/interpret")
    async def post_interpret(body: TextBody) -> dict[str, Any]:
        return await asyncio.to_thread(viewmodel.interpret_text, body.text)

    @app.post("/api/inject")
    async def post_inject(body: InjectBody) -> dict[str, Any]:
        if body.commands:
            commands = [command.model_dump() for command in body.commands]
            return await asyncio.to_thread(viewmodel.inject_commands, commands)
        if body.text and body.text.strip():
            return await asyncio.to_thread(viewmodel.inject_text, body.text.strip())
        raise CyhmoError("informe 'text' ou 'commands' para injetar")

    @app.post("/api/mic-test")
    async def post_mic_test(body: MicTestBody) -> dict[str, Any]:
        return await asyncio.to_thread(viewmodel.mic_test, body.seconds)

    @app.post("/api/hotkey/capture")
    async def post_hotkey_capture(body: HotkeyCaptureBody) -> dict[str, Any]:
        return await asyncio.to_thread(viewmodel.capture_hotkey, body.timeout_s)

    @app.get("/api/grammar")
    async def get_grammar(
        q: str = Query(default=""), limit: int = Query(default=DEFAULT_GRAMMAR_LIMIT, ge=1, le=5000)
    ) -> dict[str, Any]:
        entries = viewmodel.grammar_entries(q, limit)
        return {"entries": entries, "count": len(entries), "query": q, "limit": limit}

    @app.get("/api/injector")
    async def get_injector() -> dict[str, Any]:
        return await asyncio.to_thread(viewmodel.injector_status)

    @app.websocket("/ws")
    async def websocket_events(websocket: WebSocket) -> None:
        await _stream_events(websocket, viewmodel)

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(static_dir / "index.html", media_type="text/html")

    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    return app


async def _stream_events(websocket: "WebSocket", viewmodel: UiViewModel) -> None:
    await websocket.accept()
    queue = viewmodel.bus.attach_queue(asyncio.get_running_loop())
    try:
        await websocket.send_json({"kind": "snapshot", "state": viewmodel.snapshot(queue.dropped)})
        pump = asyncio.create_task(_pump_events(websocket, queue))
        watch = asyncio.create_task(_wait_for_disconnect(websocket))
        done, pending = await asyncio.wait({pump, watch}, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        for task in (*pending, *done):
            with contextlib.suppress(BaseException):
                await task
    except asyncio.CancelledError:
        log.debug("WebSocket cancelado pelo encerramento do mod")
    except Exception as exc:
        log.debug("cliente WebSocket encerrou: %s", exc)
    finally:
        queue.close()


async def _pump_events(websocket: "WebSocket", queue: Any) -> None:
    dropped_seen = queue.dropped
    while True:
        event = await queue.get()
        await websocket.send_json(event.to_dict())
        if queue.dropped != dropped_seen:
            dropped_seen = queue.dropped
            await websocket.send_json({"kind": "dropped", "count": dropped_seen})


async def _wait_for_disconnect(websocket: "WebSocket") -> None:
    """Recarregar a página fecha o socket: é encerramento normal, não erro a propagar."""
    from starlette.websockets import WebSocketDisconnect

    with contextlib.suppress(WebSocketDisconnect):
        while True:
            await websocket.receive_text()


async def _serve_until_stopped(
    server: Any, serving: "asyncio.Task[None]", stop_event: asyncio.Event | None
) -> None:
    if stop_event is None:
        await serving
        return
    waiter = asyncio.create_task(stop_event.wait())
    try:
        done, _pending = await asyncio.wait({serving, waiter}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        waiter.cancel()
    if serving in done:
        await serving
        return
    await _graceful_stop(server, serving)


async def _graceful_stop(server: Any, serving: "asyncio.Task[None]") -> None:
    server.should_exit = True
    with contextlib.suppress(BaseException):
        await asyncio.wait_for(asyncio.shield(serving), GRACEFUL_SHUTDOWN_S)


def ensure_port_free(host: str, port: int) -> None:
    """O uvicorn responde a porta ocupada com ``sys.exit`` de dentro da task, o que vira
    um traceback ilegível; a checagem antecipada troca isso por uma instrução."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind((host, port))
    except OSError as exc:
        raise UiServerError(
            f"a porta {port} em {host} já está ocupada ({exc.strerror or exc}). "
            "Quase sempre é outra sessão do mod ainda aberta: feche-a "
            f"(no Windows, `Get-NetTCPConnection -LocalPort {port} -State Listen`) "
            "ou escolha outra porta em ui.port no config.toml."
        ) from exc
    finally:
        probe.close()


async def serve(
    app: "FastAPI",
    host: str,
    port: int,
    open_browser: bool,
    stop_event: asyncio.Event | None = None,
) -> None:
    """Sobe o uvicorn no loop atual, sem instalar handlers de sinal (o integrador cuida do Ctrl+C).

    ``stop_event`` deixa o uvicorn se encerrar sozinho: cancelar a task no meio derruba
    lifespan e WebSocket na marra e enche o log de traceback."""
    import uvicorn

    ensure_port_free(host, port)

    class QuietServer(uvicorn.Server):
        def capture_signals(self) -> contextlib.AbstractContextManager[None]:
            return contextlib.nullcontext()

    server = QuietServer(uvicorn.Config(app, host=host, port=port, log_level="warning", loop="asyncio"))
    serving = asyncio.create_task(server.serve())
    while not server.started and not serving.done():
        await asyncio.sleep(0.05)
    if open_browser and server.started:
        url = f"http://{host}:{port}/"
        threading.Thread(target=webbrowser.open, args=(url,), name="cyhmo-ui-browser", daemon=True).start()
    try:
        await _serve_until_stopped(server, serving, stop_event)
    except asyncio.CancelledError:
        await _graceful_stop(server, serving)
        raise
    except SystemExit as exc:
        raise UiServerError(
            f"a interface não subiu em {host}:{port} (uvicorn encerrou com código {exc.code}). "
            "Confira ui.host e ui.port no config.toml."
        ) from exc
