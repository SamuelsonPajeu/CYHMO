"""Servidor HTTP/WebSocket local da interface.

Roda no mesmo loop asyncio do pipeline; chamadas potencialmente lentas vão para
thread e o fluxo de eventos usa a fila descarta-o-antigo do barramento.

A interface é local mas o navegador que a abre não é: qualquer página aberta em outra
aba fala com 127.0.0.1. Três guardas cuidam disso, e as três estão em ``_deny_reason``:
o ``Host`` precisa ser um nome de loopback conhecido, o que fecha o DNS rebinding; o
``Origin``, quando existe, precisa casar com o ``Host``, o que fecha o CSRF; e toda rota
``/api`` exige o token da sessão, que só existe dentro do HTML servido aqui.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import logging
import secrets
import socket
import threading
import webbrowser
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from pydantic import BaseModel, Field

from cyhmo.domain.errors import CyhmoError, UiServerError
from cyhmo.ui.viewmodel import DEFAULT_GRAMMAR_LIMIT, UiViewModel

if TYPE_CHECKING:
    from fastapi import FastAPI, Request, WebSocket

log = logging.getLogger("cyhmo.ui.server")

GRACEFUL_SHUTDOWN_S = 2.0

STATIC_DIR = Path(__file__).resolve().parent / "static"
MIC_TEST_MAX_SECONDS = 30.0
DEFAULT_HOTKEY_TIMEOUT = 10.0
HOTKEY_CAPTURE_MAX_SECONDS = 60.0

DEFAULT_UI_HOST = "127.0.0.1"
DEFAULT_UI_PORT = 8765
LOOPBACK_NAMES = ("127.0.0.1", "localhost", "[::1]")
TOKEN_META_NAME = "cyhmo-token"
TOKEN_HEADER = "x-cyhmo-token"
TOKEN_QUERY = "token"
TOKEN_BYTES = 32
HEAD_MARKER = "<head>"
API_PREFIX = "/api/"
WS_POLICY_VIOLATION = 1008
FORBIDDEN = 403

CODE_BAD_HOST = "bad_host"
CODE_BAD_ORIGIN = "bad_origin"
CODE_STALE_TOKEN = "stale_token"

# default-src 'self' em vez de 'none' por segurança de operação: um recurso que eu não
# tenha previsto continua carregando da própria origem em vez de sumir da tela. O que
# fecha de verdade está nas outras diretivas. 'unsafe-inline' em style-src é exigido
# pelo atributo style que a View escreve nas barras de nível e progresso.
CONTENT_SECURITY_POLICY = (
    "default-src 'self'; "
    "img-src 'self' data:; "
    "style-src 'self' 'unsafe-inline'; "
    "object-src 'none'; "
    "base-uri 'none'; "
    "form-action 'none'; "
    "frame-ancestors 'none'"
)

REFUSED_HOST = (
    "esta interface só atende pelo endereço local do mod. O pedido chegou com Host {host!r}, "
    "que não é um deles — quase sempre é uma página de outro site tentando falar com o mod."
)
REFUSED_ORIGIN = (
    "pedido recusado: veio da origem {origin!r}, que não é a interface do mod. "
    "Abra o painel pelo endereço que o CYHMO imprime no terminal."
)
REFUSED_TOKEN = (
    "esta aba está com o token de uma sessão antiga do mod. Recarregue a página (F5) para "
    "pegar o token da sessão atual."
)


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


def is_loopback(host: str) -> bool:
    return host.strip().strip("[]").casefold() in {"127.0.0.1", "localhost", "::1"}


def allowed_authorities(host: str, port: int) -> frozenset[str] | None:
    """Os valores de ``Host`` que esta interface aceita, ou ``None`` para não conferir.

    Fora do loopback o mod não tem como saber por qual nome o navegador chega, e quem
    mudou ``ui.host`` já decidiu expor a interface; a checagem sai de cena e a defesa fica
    com o token e com a igualdade Origin/Host, que não dependem do endereço."""
    if not is_loopback(host):
        return None
    names = [f"{name}:{port}" for name in LOOPBACK_NAMES]
    if port == 80:
        names += list(LOOPBACK_NAMES)
    return frozenset(name.casefold() for name in names)


def index_with_token(static_dir: Path, token: str) -> str:
    """O token da sessão viaja num ``<meta>`` do próprio HTML.

    Num ``<script>`` embutido ele obrigaria a CSP a aceitar script inline, que é
    justamente o que ela existe para barrar. E como nenhuma outra origem consegue ler o
    corpo desta resposta, servir o token aqui não o entrega a ninguém."""
    source = static_dir / "index.html"
    try:
        html = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise UiServerError(f"não consegui ler {source}: {exc}") from exc
    position = html.find(HEAD_MARKER)
    if position < 0:
        raise UiServerError(f"{source} não tem <head>; a interface não pode ser servida com segurança")
    cut = position + len(HEAD_MARKER)
    return f'{html[:cut]}\n  <meta name="{TOKEN_META_NAME}" content="{token}">{html[cut:]}'


def _deny_reason(
    path: str, headers: Any, authorities: frozenset[str] | None, token: str, presented: str
) -> tuple[str, str] | None:
    """Motivo pelo qual o pedido não deve ser atendido, ou ``None`` quando pode passar."""
    host_header = str(headers.get("host", "")).casefold()
    if authorities is not None and host_header not in authorities:
        return REFUSED_HOST.format(host=headers.get("host", "")), CODE_BAD_HOST
    origin = headers.get("origin")
    if origin and urlsplit(str(origin)).netloc.casefold() != host_header:
        return REFUSED_ORIGIN.format(origin=origin), CODE_BAD_ORIGIN
    if not path.startswith(API_PREFIX):
        return None
    if not hmac.compare_digest(presented, token):
        return REFUSED_TOKEN, CODE_STALE_TOKEN
    return None


def _harden(response: Any, path: str) -> Any:
    """Cabeçalhos que valem para toda resposta, inclusive as recusadas.

    Sem ``Cache-Control`` o navegador aplica cache heurístico e pode servir a View antiga
    sem perguntar — foi como uma atualização do mod deixou um navegador com o JS velho e a
    página em branco. ``no-cache`` obriga a revalidar; o ETag mantém a resposta em 304,
    então não custa nada."""
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("Content-Security-Policy", CONTENT_SECURITY_POLICY)
    if path == "/" or path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


def create_app(
    viewmodel: UiViewModel,
    static_dir: Path | None = None,
    host: str = DEFAULT_UI_HOST,
    port: int = DEFAULT_UI_PORT,
) -> "FastAPI":
    from fastapi import FastAPI, Query, Request, WebSocket
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles

    _publish_route_annotations(WebSocket=WebSocket, Request=Request)
    static_dir = static_dir or STATIC_DIR
    app = FastAPI(title="CYHMO", docs_url=None, redoc_url=None, openapi_url=None)

    token = secrets.token_urlsafe(TOKEN_BYTES)
    authorities = allowed_authorities(host, port)
    index_html = index_with_token(static_dir, token)
    app.state.cyhmo_token = token
    if authorities is None:
        log.warning(
            "ui.host = %r não é loopback: a interface fica alcançável pela rede e a checagem de Host "
            "não pode ser aplicada. O token da sessão continua sendo exigido.",
            host,
        )

    @app.middleware("http")
    async def guard_local_origin(request: Request, call_next: Any) -> Any:
        denial = _deny_reason(
            request.url.path,
            request.headers,
            authorities,
            token,
            request.headers.get(TOKEN_HEADER, ""),
        )
        if denial is not None:
            message, code = denial
            log.warning("pedido recusado em %s: %s", request.url.path, code)
            return _harden(JSONResponse({"error": message, "code": code}, status_code=FORBIDDEN), request.url.path)
        return _harden(await call_next(request), request.url.path)

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

    @app.get("/api/stt/models")
    async def get_stt_models() -> dict[str, Any]:
        return await asyncio.to_thread(viewmodel.whisper_status)

    @app.post("/api/stt/models/download")
    async def post_stt_model_download(body: ModelBody) -> dict[str, Any]:
        return await asyncio.to_thread(viewmodel.whisper_download, body.model)

    @app.post("/api/stt/models/download/cancel")
    async def post_stt_model_download_cancel() -> dict[str, Any]:
        return await asyncio.to_thread(viewmodel.whisper_download_cancel)

    @app.post("/api/stt/models/delete")
    async def post_stt_model_delete(body: ModelBody) -> dict[str, Any]:
        return await asyncio.to_thread(viewmodel.whisper_delete, body.model)

    @app.post("/api/stt/gpu/install")
    async def post_stt_gpu_install() -> dict[str, Any]:
        return await asyncio.to_thread(viewmodel.whisper_gpu_install)

    @app.post("/api/stt/gpu/install/cancel")
    async def post_stt_gpu_install_cancel() -> dict[str, Any]:
        return await asyncio.to_thread(viewmodel.whisper_gpu_install_cancel)

    @app.post("/api/stt/gpu/remove")
    async def post_stt_gpu_remove() -> dict[str, Any]:
        return await asyncio.to_thread(viewmodel.whisper_gpu_remove)

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
        """O WebSocket não passa por CORS: sem esta guarda, qualquer aba aberta receberia o
        snapshot com a configuração e o histórico do que o jogador falou. O token vem na
        query porque o navegador não deixa a página escolher cabeçalhos aqui."""
        denial = _deny_reason(
            API_PREFIX,
            websocket.headers,
            authorities,
            token,
            websocket.query_params.get(TOKEN_QUERY, ""),
        )
        if denial is not None:
            log.warning("WebSocket recusado: %s", denial[1])
            await websocket.close(code=WS_POLICY_VIOLATION)
            return
        await _stream_events(websocket, viewmodel)

    @app.get("/")
    async def index() -> HTMLResponse:
        return HTMLResponse(index_html)

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
