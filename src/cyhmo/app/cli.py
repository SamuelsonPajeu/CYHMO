"""Linha de comando do mod: run, setup, devices, languages, config, interpret, say,
status, models, pnach, doctor, calibrate e update."""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

from cyhmo.app.boot_progress import BootProgress
from cyhmo.app.bootstrap import EXIT_MODE_RESTART, EXIT_MODE_UPDATE, Application
from cyhmo.app.calibration_job import DEFAULT_GRID, parse_grid
from cyhmo.app.doctor import describe_orphan_patch, run_doctor
from cyhmo.config.loader import DEFAULT_CONFIG_NAME, ConfigStore, render_config, write_default_config
from cyhmo.config.schema import ProjectPaths
from cyhmo.domain.contracts import CommandRef, GameState, InjectResult, Interpretation, Transcript
from cyhmo.domain.errors import (
    AudioDeviceError,
    ConfigError,
    CyhmoError,
    GrammarUnavailableError,
    InjectionError,
)
from cyhmo.inject.recipe import WriteRecipe
from cyhmo.intent.language_packs import LanguagePackSet

PROGRAM = "cyhmo"
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_BLOCKED = 2
EXIT_USAGE = 3
EXIT_UPDATED = 10
ARGPARSE_USAGE_EXIT = 2
TOP_K_SHOWN = 5
GRAMMAR_PREVIEW = 8


def _use_utf8_streams() -> None:
    """Saída redirecionada no Windows usa a página de código do sistema, e um pacote de
    idioma com CJK derruba o comando. O executável congelado ignora PYTHONUTF8, então a
    escolha do encoding precisa morar aqui."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def main(argv: Sequence[str] | None = None) -> int:
    _use_utf8_streams()
    parser = _build_parser()
    try:
        args = parser.parse_args(None if argv is None else list(argv))
    except argparse.ArgumentError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_USAGE
    except SystemExit as exc:
        return EXIT_USAGE if exc.code == ARGPARSE_USAGE_EXIT else int(exc.code or EXIT_OK)
    if getattr(args, "handler", None) is None:
        parser.print_help()
        return EXIT_USAGE
    return _dispatch(args)


def _dispatch(args: argparse.Namespace) -> int:
    try:
        return args.handler(args)
    except KeyboardInterrupt:
        return EXIT_OK
    except CyhmoError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_ERROR
    except Exception as exc:
        if getattr(args, "debug", False):
            traceback.print_exc()
        else:
            print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
            print("rode com --debug para o traceback", file=sys.stderr)
        return EXIT_ERROR


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=PROGRAM, description="Mod de comandos por voz para Lifeline (PS2/PCSX2).")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--debug", action="store_true", help="mostra o traceback completo em caso de erro")
    common.add_argument("--config", type=Path, default=None, metavar="ARQ", help="caminho do config.toml")
    subparsers = parser.add_subparsers(dest="command")
    for register in (
        _add_run,
        _add_setup,
        _add_devices,
        _add_languages,
        _add_config,
        _add_interpret,
        _add_say,
        _add_status,
        _add_models,
        _add_pnach,
        _add_doctor,
        _add_calibrate,
        _add_update,
    ):
        register(subparsers, common)
    return parser


def _add_run(subparsers: Any, common: argparse.ArgumentParser) -> None:
    command = subparsers.add_parser("run", parents=[common], help="roda o mod (captura, interpreta e injeta)")
    command.add_argument("--headless", action="store_true", help="não sobe a interface web")
    command.add_argument("--record", action="store_true", help="grava o .wav de cada enunciado")
    command.add_argument("--no-inject", action="store_true", help="interpreta e mostra, mas não escreve no jogo")
    command.add_argument("--no-browser", action="store_true", help="não abre o navegador (a aba já está aberta)")
    command.set_defaults(handler=_cmd_run)


def _add_setup(subparsers: Any, common: argparse.ArgumentParser) -> None:
    command = subparsers.add_parser("setup", parents=[common], help="baixa o backend de transcrição (whisper.cpp)")
    command.add_argument("--skip-model", action="store_true", help="instala só o executável, sem o peso ggml")
    command.add_argument("--force", action="store_true", help="rebaixa mesmo se já estiver instalado")
    command.set_defaults(handler=_cmd_setup)


def _add_devices(subparsers: Any, common: argparse.ArgumentParser) -> None:
    command = subparsers.add_parser("devices", parents=[common], help="lista os dispositivos de captura")
    command.set_defaults(handler=_cmd_devices)


def _add_languages(subparsers: Any, common: argparse.ArgumentParser) -> None:
    command = subparsers.add_parser("languages", parents=[common], help="lista os pacotes de idioma")
    command.set_defaults(handler=_cmd_languages)


def _add_config(subparsers: Any, common: argparse.ArgumentParser) -> None:
    command = subparsers.add_parser("config", parents=[common], help="cria ou mostra o config.toml")
    mode = command.add_mutually_exclusive_group()
    mode.add_argument("--init", action="store_true", help="grava um config.toml com os defaults comentados")
    mode.add_argument("--show", action="store_true", help="imprime a configuração efetiva (padrão)")
    command.add_argument("--force", action="store_true", help="com --init, sobrescreve um arquivo existente")
    command.set_defaults(handler=_cmd_config)


def _add_interpret(subparsers: Any, common: argparse.ArgumentParser) -> None:
    command = subparsers.add_parser("interpret", parents=[common], help="interpreta um texto sem falar nem injetar")
    command.add_argument("text", metavar="TEXTO", help="o que o jogador diria")
    command.add_argument("--mode", default=None, help="modo de jogo assumido (battle, normal, dialog...)")
    command.add_argument("--enemies", type=int, default=None, help="quantidade de inimigos em cena")
    command.add_argument(
        "--grammar",
        type=Path,
        default=None,
        metavar="ARQ",
        help="usa a gramática deste arquivo (YAML com 'entries:') em vez da viva",
    )
    command.set_defaults(handler=_cmd_interpret)


def _add_say(subparsers: Any, common: argparse.ArgumentParser) -> None:
    command = subparsers.add_parser("say", parents=[common], help="injeta um comando literal no jogo")
    command.add_argument("text", metavar="TEXTO", nargs="+", help="comando exatamente como o jogo o conhece")
    command.add_argument("--force", action="store_true", help="injeta mesmo fora da gramática ativa")
    command.set_defaults(handler=_cmd_say)


def _add_status(subparsers: Any, common: argparse.ArgumentParser) -> None:
    command = subparsers.add_parser("status", parents=[common], help="mostra o estado do ASR, do injetor e da gramática")
    command.set_defaults(handler=_cmd_status)


def _add_models(subparsers: Any, common: argparse.ArgumentParser) -> None:
    command = subparsers.add_parser("models", parents=[common], help="verifica a integridade dos modelos baixados")
    command.add_argument("--lock", action="store_true", help="regrava o models.lock com os hashes atuais")
    command.add_argument("--quarantine", action="store_true", help="isola arquivos corrompidos em vez de só apontar")
    command.set_defaults(handler=_cmd_models)


def _add_pnach(subparsers: Any, common: argparse.ArgumentParser) -> None:
    command = subparsers.add_parser("pnach", parents=[common], help="gera o .pnach do patch para a pasta cheats do PCSX2")
    command.add_argument("--out", type=Path, default=None, metavar="DIR", help="pasta de destino (padrão: ./cheats)")
    command.add_argument("--print", dest="to_stdout", action="store_true", help="mostra o conteúdo sem gravar")
    command.set_defaults(handler=_cmd_pnach)


def _add_doctor(subparsers: Any, common: argparse.ArgumentParser) -> None:
    command = subparsers.add_parser("doctor", parents=[common], help="diagnostica o ambiente e aponta o que falta")
    command.set_defaults(handler=_cmd_doctor)


def _add_calibrate(subparsers: Any, common: argparse.ArgumentParser) -> None:
    command = subparsers.add_parser("calibrate", parents=[common], help="varre limiares de aceite e rejeição")
    command.add_argument("--dataset", type=Path, required=True, metavar="ARQ", help="casos rotulados")
    command.add_argument("--spontaneous", type=Path, required=True, metavar="ARQ", help="frases que não são comandos")
    command.add_argument("--grid", default=DEFAULT_GRID, help='pares "aceite,rejeição" separados por espaço')
    command.add_argument("--out", type=Path, default=None, metavar="ARQ", help="grava o relatório JSON")
    command.set_defaults(handler=_cmd_calibrate)


def _add_update(subparsers: Any, common: argparse.ArgumentParser) -> None:
    command = subparsers.add_parser("update", parents=[common], help="mostra a versão instalada e a última release")
    command.add_argument("--install", action="store_true", help="baixa e instala a última release por cima desta")
    command.set_defaults(handler=_cmd_update)


def _cmd_run(args: argparse.Namespace) -> int:
    """O reinício pedido pela interface reconstrói a aplicação inteira, relendo o
    config.toml — é o que faz uma mudança de configuração valer sem voltar ao terminal.

    A aba do navegador só é aberta na primeira volta: no reinício ela já está aberta e
    reconecta sozinha, então abrir de novo daria duas janelas para a mesma interface."""
    open_browser: bool | None = False if args.no_browser else None
    while True:
        progress = BootProgress.to_console()
        progress.opening()
        application = Application.build(
            config_path=args.config,
            headless=args.headless,
            record=args.record,
            debug=args.debug,
            inject_enabled=False if args.no_inject else None,
            open_browser=open_browser,
            progress=progress,
        )
        try:
            asyncio.run(application.run())
        finally:
            application.shutdown()
        if application.exit_mode == EXIT_MODE_UPDATE:
            return _apply_update(application.paths)
        if application.exit_mode != EXIT_MODE_RESTART:
            return EXIT_OK
        open_browser = False
        print("reiniciando com a configuração atual...", file=sys.stderr)


def _apply_update(paths: ProjectPaths) -> int:
    """A troca dos arquivos acontece com o mod encerrado, e quem carrega o código novo é o
    processo seguinte — por isso um código de saída para o lançador, e não outra volta do
    laço aqui dentro."""
    from cyhmo.update.installer import apply_pending

    version = apply_pending(paths.base_dir, paths.data_dir)
    print(f"CYHMO atualizado para {version}; encerrando para o lançador reabrir.", file=sys.stderr)
    return EXIT_UPDATED


def _cmd_setup(args: argparse.Namespace) -> int:
    """Não abre o PCSX2 nem carrega modelo de embeddings: só baixa e extrai arquivos."""
    from cyhmo.stt.whisper_setup import ensure_backend

    store = ConfigStore.open(args.config)
    report = ensure_backend(
        store.config,
        store.paths,
        skip_model=args.skip_model,
        force=args.force,
        report=lambda message: print(message),
    )
    for action in report.actions:
        print(action)
    if not report.ready and not args.skip_model:
        print("o backend continua incompleto; o mod vai usar o faster-whisper", file=sys.stderr)
        return EXIT_ERROR
    print("backend de transcrição pronto." if report.ready else "executável pronto; falta o modelo.")
    return EXIT_OK


def _cmd_devices(args: argparse.Namespace) -> int:
    from cyhmo.stt import format_devices

    print(format_devices(_capture_devices()))
    return EXIT_OK


def _cmd_languages(args: argparse.Namespace) -> int:
    store = ConfigStore.open(args.config)
    languages = store.config.languages
    packs, problems = LanguagePackSet.available(store.paths.packs_dir)
    print(f"pacotes em {store.paths.packs_dir}")
    for pack in packs:
        summary = pack.to_summary()
        marks = [
            mark
            for mark, on in (
                ("habilitado", pack.code in languages.enabled),
                ("primário", pack.code == languages.primary),
            )
            if on
        ]
        suffix = f" — {', '.join(marks)}" if marks else ""
        print(
            f"  {pack.code:<8} {pack.name} (STT {pack.stt_language}, "
            f"{summary['command_examples']} exemplos, {summary['lexicon']} termos){suffix}"
        )
    for problem in problems:
        print(f"  ! {problem}")
    missing = [code for code in languages.enabled if code not in {pack.code for pack in packs}]
    if missing:
        print(f"habilitados sem arquivo: {', '.join(missing)}")
    return EXIT_ERROR if missing or problems else EXIT_OK


def _cmd_config(args: argparse.Namespace) -> int:
    if not args.init:
        print(render_config(ConfigStore.open(args.config).config))
        return EXIT_OK
    path = (args.config or Path.cwd() / DEFAULT_CONFIG_NAME).resolve()
    if path.exists() and not args.force:
        raise ConfigError(f"{path} já existe; use --force para sobrescrever")
    write_default_config(path)
    print(f"config.toml gravado em {path}")
    return EXIT_OK


def _cmd_models(args: argparse.Namespace) -> int:
    from cyhmo.app.models_lock import LOCK_NAME, ModelsLock

    store = ConfigStore.open(args.config)
    models_dir, lock_path = store.paths.models_dir, store.paths.base_dir / LOCK_NAME
    if args.lock:
        lock = ModelsLock.from_directory(models_dir)
        lock.write(lock_path)
        print(f"{lock_path} gravado com {len(lock.files)} arquivo(s).")
        return EXIT_OK
    verdicts = ModelsLock.load(lock_path).verify(models_dir, quarantine=args.quarantine)
    for verdict in verdicts:
        suffix = f" — {verdict.detail}" if verdict.detail else ""
        print(f"[{'OK' if verdict.ok else verdict.status.upper()}] {verdict.path}{suffix}")
    failed = [verdict for verdict in verdicts if not verdict.ok]
    print(f"\n{len(verdicts) - len(failed)}/{len(verdicts)} arquivo(s) íntegro(s).")
    return EXIT_ERROR if failed else EXIT_OK


def _cmd_pnach(args: argparse.Namespace) -> int:
    """Não abre o PCSX2 nem carrega modelo: só precisa da receita, que é um arquivo."""
    from cyhmo.inject import WriteRecipe, render_pnach, write_pnach

    store = ConfigStore.open(args.config)
    recipe = WriteRecipe.load(store.paths.recipe)
    if args.to_stdout:
        print(render_pnach(recipe), end="")
        return EXIT_OK
    target = write_pnach(recipe, args.out or store.paths.base_dir / "cheats")
    print(f"{target} gravado.")
    print("Copie para a pasta 'cheats' do PCSX2 e ligue Settings → Advanced → Enable Cheats.")
    print(f'Depois troque pine.patch_mode para "pnach" no config.toml (hoje: "{store.config.pine.patch_mode}").')
    return EXIT_OK


def _cmd_interpret(args: argparse.Namespace) -> int:
    application = Application.build(config_path=args.config, headless=True, debug=args.debug)
    try:
        state = _interpret_state(application, args)
        transcript = Transcript(
            text=args.text,
            lang=application.packs.stt_language,
            confidence=1.0,
            t_speech_end=time.perf_counter(),
        )
        _print_interpretation(application.interpreter.interpret(transcript, state), state)
    finally:
        application.shutdown()
    return EXIT_OK


def _cmd_say(args: argparse.Namespace) -> int:
    text = " ".join(args.text).strip()
    if not text:
        raise ConfigError("informe o comando a injetar, ex.: cyhmo say Walk")
    application = Application.build(config_path=args.config, headless=True, debug=args.debug)
    try:
        _require_pine(application, "injetar")
        application.state_service.refresh_now()
        if not args.force and not application.state_service.accepts(text):
            print(
                f"{text!r} não está na gramática ativa desta cena; nada foi injetado. "
                "Use --force para escrever mesmo assim (o jogo provavelmente vai ignorar).",
                file=sys.stderr,
            )
            return EXIT_BLOCKED
        if args.force:
            print("--force: portão de gramática ignorado", file=sys.stderr)
        result = _say_injector(application, args.force).inject([CommandRef(key=text)])
        _print_inject_result(text, result)
        return EXIT_OK if result.ok else EXIT_ERROR
    finally:
        application.shutdown()


def _cmd_status(args: argparse.Namespace) -> int:
    application = Application.build(config_path=args.config, headless=True, debug=args.debug)
    try:
        _require_pine(application, "ler o estado do jogo")
        application.state_service.refresh_now()
        _print_injector_status(application)
        _print_grammar_status(application.state_service.read_state())
    finally:
        application.shutdown()
    return EXIT_OK


def _cmd_doctor(args: argparse.Namespace) -> int:
    return run_doctor(args.config, sys.stdout)


def _cmd_calibrate(args: argparse.Namespace) -> int:
    from cyhmo.intent import build_interpreter
    from cyhmo.intent.calibration import format_report, load_dataset, load_spontaneous, run_calibration, to_json
    from cyhmo.intent.embedders import build_embedder

    store = ConfigStore.open(args.config)
    config, paths = store.config, store.paths
    packs = LanguagePackSet.load(paths.packs_dir, config.languages.enabled, config.languages.primary)
    embedder = build_embedder(config.intent, paths.models_dir)

    def make_interpreter(accept: float, reject: float) -> Any:
        tuned = config.replace(
            intent=config.intent.model_copy(update={"accept_threshold": accept, "reject_threshold": reject})
        )
        return build_interpreter(tuned, paths, packs=packs, embedder=embedder)

    report = run_calibration(
        make_interpreter, load_dataset(args.dataset), load_spontaneous(args.spontaneous), parse_grid(args.grid)
    )
    print(format_report(report))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(to_json(report), encoding="utf-8")
        print(f"relatório em {args.out}")
    return EXIT_OK


def _cmd_update(args: argparse.Namespace) -> int:
    """Aqui a troca é imediata: nenhuma camada do mod está no ar para atrapalhar."""
    from cyhmo.pipeline.bus import EventBus
    from cyhmo.update.installer import apply_pending
    from cyhmo.update.service import UpdateService

    store = ConfigStore.open(args.config)
    service = UpdateService(store, store.paths, EventBus())
    status = service.check_now()
    print(f"versão instalada: {status['current']}")
    print(f"última release: {status['latest']} ({status['url']})")
    if not status["available"]:
        print("nada a atualizar.")
        return EXIT_OK
    if not args.install:
        print("rode 'cyhmo update --install' para instalar esta versão.")
        return EXIT_OK
    print(f"baixando o pacote da versão {status['latest']}...")
    service.download_now()
    version = apply_pending(store.paths.base_dir, store.paths.data_dir)
    print(f"CYHMO atualizado para {version}. Abra o mod de novo para usar a versão nova.")
    return EXIT_OK


def _capture_devices() -> Any:
    from cyhmo.stt import list_capture_devices

    try:
        return list_capture_devices()
    except ImportError as exc:
        raise AudioDeviceError(
            f"biblioteca de áudio indisponível ({exc}); instale as dependências do mod para listar microfones"
        ) from exc


def _require_pine(application: Application, action: str) -> None:
    """O cliente agora sobrevive desconectado, para reatar sozinho quando o emulador abrir.
    Num comando de uma tirada só não há o que reatar: sem ``is_connected``, esta mensagem
    acionável virava código morto e o usuário recebia a espera de reconexão no lugar dela."""
    client = application.pine_client
    if client is not None and client.is_connected:
        return
    pine = application.config.pine
    raise InjectionError(
        f"PCSX2 não encontrado em {pine.host}:{pine.port}, impossível {action}. "
        "Abra o emulador com o jogo rodando e o PINE habilitado (Settings → Advanced)."
    )


def _say_injector(application: Application, force: bool) -> Any:
    if not force:
        return application.injector
    from cyhmo.inject import build_injector

    return build_injector(application.config, application.paths, application.pine_client, bus=application.bus)


def _interpret_state(application: Application, args: argparse.Namespace) -> GameState:
    live = application.state_service
    live.refresh_now()
    state = live.read_state()
    entries = _grammar_from_file(args.grammar) if args.grammar is not None else state.grammar
    if not entries:
        raise GrammarUnavailableError(
            "sem gramática para interpretar: abra o PCSX2 com o jogo numa cena com comandos, "
            "passe --grammar ARQ ou configure state.grammar_seed no config.toml"
        )
    enemies = None if args.enemies is None else tuple({"index": index + 1} for index in range(args.enemies))
    return GameState(
        mode=args.mode or state.mode,
        can_talk=True,
        enemies=enemies,
        grammar=tuple(entries),
        grammar_stale=state.grammar_stale and args.grammar is None,
    )


def _grammar_from_file(path: Path) -> tuple[str, ...]:
    from cyhmo.state.grammar_source import load_grammar_entries

    return tuple(load_grammar_entries(path))


def _print_interpretation(interpretation: Interpretation, state: GameState) -> None:
    print(f"texto normalizado: {interpretation.normalized_text or '(vazio)'}")
    print(f"contexto: modo {state.mode}, {len(state.grammar or ())} comandos na gramática")
    commands = " + ".join(_format_command(command) for command in interpretation.commands) or "(nenhum)"
    print(f"comandos: {commands}")
    print(
        f"método: {interpretation.method} | confiança: {interpretation.confidence:.3f} | "
        f"motivo: {interpretation.reason or '—'} | latência: {interpretation.latency_ms:.1f} ms"
    )
    if not interpretation.candidates:
        return
    print("top-k:")
    for position, candidate in enumerate(interpretation.candidates[:TOP_K_SHOWN], start=1):
        print(
            f"  {position}. {candidate.key:<24} {candidate.score:.4f}  "
            f"exemplo: {candidate.matched_example!r} [{candidate.example_lang}]"
        )


def _format_command(command: CommandRef) -> str:
    return f"{command.key} {command.args}" if command.args else command.key


def _print_inject_result(text: str, result: InjectResult) -> None:
    if not result.ok:
        print(f"falhou ao injetar {text!r}: {result.error}", file=sys.stderr)
        return
    print(f"injetado {text!r} em {result.latency_ms:.1f} ms")
    oracle = result.payload_echo.get("oracle")
    if not isinstance(oracle, dict):
        print("oráculo: não verificado")
        return
    verdict = "casou" if oracle.get("matched") else "NÃO casou"
    print(f"oráculo: {verdict} (texto lido: {oracle.get('matched_text')!r}, ids aceitos: {oracle.get('accepted_ids')})")


def _print_injector_status(application: Application) -> None:
    reader = getattr(application.injector, "read_status", None)
    if not callable(reader):
        print("injetor: dry-run (inject.enabled = false); nada é escrito no jogo")
        return
    status = reader()
    print(
        f"ASR: estado {status['asr_state']}, id {status['word_id']}, "
        f"{status['word_count']} palavra(s), pendente {status['pending']}, escutando {status['listening']}"
    )
    print(f"palavras no slot: {status['words'] or '(vazio)'}")
    print(f"portão do pad: {_describe_gate(status, application.recipe)}")
    oracle = status.get("oracle")
    if isinstance(oracle, dict):
        print(f"oráculo: texto {oracle.get('matched_text')!r}, ids aceitos {oracle.get('accepted_ids')}")


def _describe_gate(status: dict[str, Any], recipe: WriteRecipe | None) -> str:
    """Órfão e patcheado são estados diferentes — colapsá-los escondeu o patch preso —, e o
    desfecho do órfão depende da política ``code_patch.on_orphan`` da receita."""
    if not status.get("gate_orphan") or recipe is None:
        return "PATCHED" if status["gate_patched"] else "original"
    policy = describe_orphan_patch(recipe.ascii_words.code_patch.on_orphan)
    return f"PATCHED órfão de uma execução anterior — {policy}"


def _print_grammar_status(state: GameState) -> None:
    entries = state.grammar or ()
    pointer = state.raw.get("pointer")
    location = "—" if not isinstance(pointer, int) else f"0x{pointer:08X}"
    print(
        f"gramática: {len(entries)} comandos, ponteiro {location}, "
        f"{'stale' if state.grammar_stale else 'atual'}, modo {state.mode}"
    )
    if entries:
        print("amostra: " + ", ".join(entries[:GRAMMAR_PREVIEW]) + ("..." if len(entries) > GRAMMAR_PREVIEW else ""))


