"""Diagnóstico de primeiro uso: só leitura — nunca escreve na memória do jogo.

A lista de checagens vive em ``_Doctor.probes``; a contagem exibida sai dela, para
não haver um número no texto que envelhece sozinho."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TextIO

from cyhmo.config.loader import ConfigStore
from cyhmo.config.schema import AppConfig, ProjectPaths
from cyhmo.domain.errors import CyhmoError
from cyhmo.inject.pine import STATUS_NAMES, STATUS_RUNNING, PineClient, PineConnectError, PineError
from cyhmo.inject.recipe import WriteRecipe

MINIMUM_PYTHON = (3, 11)
NAME_WIDTH = 46
EE_PROBE_ADDRESS = 0x00100000

Probe = Callable[[], tuple[bool, str, str]]

NO_CONFIG = (False, "config.toml não carregou", "corrija a checagem 'config.toml carrega e valida' antes das demais")
NO_PINE = (
    False,
    "sem conexão PINE",
    "resolva a checagem 'socket PINE conecta' acima — ela diz qual dos dois problemas é o seu",
)
NO_RECIPE = (False, "receita de injeção indisponível", "resolva a checagem 'receita de injeção e serial'")

PINE_REFUSED_HINT = "PCSX2 aberto? PINE habilitado em Settings → Advanced? Slot 28011?"
PINE_UNRESPONSIVE_HINT = (
    "reinicie o PCSX2 — o jogo em si continua rodando, quem parou de responder é o servidor PINE; "
    "isso acontece quando muitas conexões ficam presas do lado do emulador"
)
PINE_DROPPED_HINT = "a conexão PINE caiu no meio do diagnóstico; " + PINE_UNRESPONSIVE_HINT
# Duas causas com o mesmo sintoma, e a segunda manda olhar para o lugar oposto: o PINE
# fica com o PRIMEIRO PCSX2 que abriu, então com dois abertos o mod conversa com o que
# não tem jogo nenhum — e ele responde certinho "sem jogo", com o Lifeline rodando na
# tela ao lado. Sem esta segunda frase a dica mandava carregar um jogo já carregado.
SECOND_INSTANCE_HINT = (
    "Se o jogo JÁ está rodando, provavelmente há mais de um PCSX2 aberto: o PINE atende pelo "
    "primeiro que subiu, e o mod está falando com o vazio. Feche todos e abra um só, com o jogo."
)
NO_GAME_HINT = (
    "nenhum jogo rodando no PCSX2 que atende nesta porta: carregue o Lifeline e tire a emulação "
    f"da pausa. {SECOND_INSTANCE_HINT}"
)
MEMORY_UNAVAILABLE_HINT = (
    "o PCSX2 respondeu, mas recusou a leitura: o jogo está rodando? o endereço pode não valer nesta versão"
)
NO_GRAMMAR_HINT = (
    "o jogo está no menu, numa cutscene ou carregando: entre numa cena com comandos e rode de novo. "
    "Se continuar assim dentro do jogo, o endereço de contexto mudou nesta versão"
)


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str
    hint: str = ""


def describe_orphan_patch(on_orphan: str) -> str:
    """Prometer cura automática sob ``code_patch.on_orphan: abort`` faria o usuário esperar
    por algo que o injetor recusa a fazer, enquanto toda injeção falha."""
    if on_orphan == "adopt":
        return "recuperável: o mod adota o patch e restaura o valor original na próxima injeção"
    return (
        "não recuperável sozinho: com code_patch.on_orphan = abort na receita o mod recusa o patch órfão "
        "e toda injeção falha — troque a política para adopt ou recarregue a cena no jogo para o próprio "
        "jogo reescrever a instrução"
    )


def _pine_failure(exc: PineError, live_channel_hint: str) -> tuple[bool, str, str]:
    """Canal caído e recusa do PCSX2 pedem instruções opostas: perguntar pelo jogo quando quem caiu
    foi a conexão manda o usuário investigar o lugar errado — e as checagens anteriores da lista já
    responderam essa pergunta."""
    return False, str(exc), PINE_DROPPED_HINT if isinstance(exc, PineConnectError) else live_channel_hint


def run_doctor(config_path: Path | None = None, stream: TextIO = sys.stdout) -> int:
    doctor = _Doctor(config_path)
    probes = doctor.probes()
    results: list[CheckResult] = []
    try:
        for index, (name, probe) in enumerate(probes, start=1):
            result = _evaluate(name, probe)
            results.append(result)
            _render(stream, index, len(probes), result)
    finally:
        doctor.close()
    passed = sum(1 for result in results if result.ok)
    verdict = "ambiente pronto" if passed == len(results) else "corrija os itens marcados FALHOU acima"
    print(f"\n{passed}/{len(results)} checagens passaram — {verdict}.", file=stream)
    return 0 if passed == len(results) else 1


class _Doctor:
    """Estado compartilhado entre as checagens: config, caminhos, cliente PINE e receita."""

    def __init__(self, config_path: Path | None) -> None:
        self._config_path = config_path
        self._config: AppConfig | None = None
        self._paths: ProjectPaths | None = None
        self._client: PineClient | None = None
        self._recipe: WriteRecipe | None = None
        self._serial = ""

    def probes(self) -> tuple[tuple[str, Probe], ...]:
        return (
            ("Python >= 3.11 e venv", self._check_python),
            ("config.toml carrega e valida", self._check_config),
            ("pacotes de idioma", self._check_language_packs),
            ("dispositivo de captura", self._check_capture_device),
            ("modelos presentes", self._check_models),
            ("backend whisper.cpp", self._check_whisper_cpp),
            ("integridade dos modelos (models.lock)", self._check_models_lock),
            ("socket PINE conecta", self._check_pine_socket),
            ("versão do PCSX2 (MsgVersion)", self._check_version),
            ("VM rodando (MsgStatus)", self._check_status),
            ("serial do jogo (MsgID)", self._check_serial),
            ("título do jogo (MsgTitle)", self._check_title),
            ("leitura da RAM do EE (MsgRead32)", self._check_memory_read),
            ("receita de injeção e serial", self._check_recipe),
            ("endereço do patch íntegro", self._check_patch_address),
            ("gramática ativa legível", self._check_grammar),
            ("anexo semântico", self._check_annex),
            ("fallback LLM", self._check_llm),
            ("textos da interface", self._check_interface_texts),
        )

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def _check_python(self) -> tuple[bool, str, str]:
        version = ".".join(str(part) for part in sys.version_info[:3])
        in_venv = sys.prefix != sys.base_prefix
        detail = f"Python {version} " + ("dentro do venv" if in_venv else "FORA de um venv")
        if sys.version_info < MINIMUM_PYTHON:
            required = ".".join(str(part) for part in MINIMUM_PYTHON)
            return False, detail, f"instale Python {required}+ e rode install.ps1"
        hint = "" if in_venv else "recomendado rodar dentro do venv do mod (.venv)"
        return True, detail, hint

    def _check_config(self) -> tuple[bool, str, str]:
        try:
            store = ConfigStore.open(self._config_path)
        except CyhmoError as exc:
            return False, str(exc), "corrija a chave apontada acima no config.toml"
        self._config = store.config
        self._paths = store.paths
        return True, f"{store.path} (config_version {store.config.config_version})", ""

    def _check_language_packs(self) -> tuple[bool, str, str]:
        if self._config is None or self._paths is None:
            return NO_CONFIG
        from cyhmo.intent.language_packs import LanguagePackSet

        packs, problems = LanguagePackSet.available(self._paths.packs_dir)
        codes = {pack.code for pack in packs}
        languages = self._config.languages
        missing = [code for code in languages.enabled if code not in codes]
        detail = f"{len(packs)} pacote(s) em {self._paths.packs_dir}: " + (", ".join(sorted(codes)) or "(nenhum)")
        if problems:
            detail += " | com problema: " + "; ".join(problems)
        if missing:
            return False, detail, f"pacotes habilitados sem arquivo: {missing}"
        if languages.primary not in codes:
            return False, detail, f"languages.primary {languages.primary!r} não tem pacote válido"
        return True, detail + f" | primário {languages.primary}", ""

    def _check_capture_device(self) -> tuple[bool, str, str]:
        if self._config is None:
            return NO_CONFIG
        from cyhmo import stt

        try:
            devices = stt.list_capture_devices()
        except ImportError as exc:
            return False, f"biblioteca de áudio indisponível: {exc}", "instale as dependências: pip install sounddevice"
        device = stt.resolve_device(self._config.audio.device, devices)
        return True, f"{device.label} — {device.sample_rate} Hz, {device.channels} canal(is)", ""

    def _check_models(self) -> tuple[bool, str, str]:
        if self._config is None or self._paths is None:
            return NO_CONFIG
        config = self._config
        wanted = (
            ("STT", "stt", config.stt.engine != "fake"),
            ("embeddings", "embeddings", config.intent.embedding_backend != "hashing"),
        )
        parts: list[str] = []
        missing: list[str] = []
        for label, subdir, required in wanted:
            directory = self._paths.models_dir / subdir
            if not required:
                parts.append(f"{label}: backend local, sem download")
            elif _has_files(directory):
                parts.append(f"{label}: {directory}")
            else:
                missing.append(str(directory))
        if missing:
            detail = " | ".join(parts + [f"ausentes: {', '.join(missing)}"])
            return False, detail, "os modelos são baixados no primeiro 'cyhmo run'"
        return True, " | ".join(parts), ""

    def _check_whisper_cpp(self) -> tuple[bool, str, str]:
        if self._config is None or self._paths is None:
            return NO_CONFIG
        from cyhmo.stt.whisper_gpu import effective_binary

        settings = self._config.stt.whisper_cpp
        if self._config.stt.engine != "whisper-cpp":
            return True, f"não usado (engine = {self._config.stt.engine})", ""
        binary, use_gpu = effective_binary(
            self._paths.base_dir, settings.binary, settings.gpu_binary, settings.use_gpu
        )
        missing = [str(path) for path in (binary, (self._paths.base_dir / settings.model)) if not path.is_file()]
        if missing:
            return (
                False,
                f"ausentes: {', '.join(missing)}",
                "rode `CYHMO.cmd setup` para baixá-los; até lá o mod usa o faster-whisper, mais lento",
            )
        where = "GPU (CUDA)" if use_gpu else "CPU"
        if settings.use_gpu and not use_gpu:
            return (
                True,
                f"{Path(settings.model).name} em CPU — o build com GPU não está instalado",
                "instale-o em Configurações › Modelo de reconhecimento, ou desligue stt.whisper_cpp.use_gpu",
            )
        return True, f"{Path(settings.model).name} em {where}", ""

    def _check_models_lock(self) -> tuple[bool, str, str]:
        if self._paths is None:
            return NO_CONFIG
        from cyhmo.app.models_lock import LOCK_NAME, ModelsLock, ModelsLockError

        lock_path = self._paths.base_dir / LOCK_NAME
        if not lock_path.is_file():
            return True, f"{LOCK_NAME} ausente — nada a conferir", "gere com 'cyhmo models --lock'"
        try:
            verdicts = ModelsLock.load(lock_path).verify(self._paths.models_dir)
        except ModelsLockError as exc:
            return False, str(exc), "regrave o manifesto com 'cyhmo models --lock'"
        if not verdicts:
            return True, f"{LOCK_NAME} sem arquivos listados", ""
        broken = [verdict for verdict in verdicts if not verdict.ok]
        if broken:
            detail = "; ".join(f"{verdict.path}: {verdict.status}" for verdict in broken)
            return False, detail, "baixe o modelo de novo e rode 'cyhmo models --quarantine'"
        return True, f"{len(verdicts)} arquivo(s) conferem com {LOCK_NAME}", ""

    def _check_pine_socket(self) -> tuple[bool, str, str]:
        """Aceitar o socket não prova que o servidor PINE responde: só uma ida e volta separa a
        conexão recusada do emulador que aceita e emudece — confundi-los acusava 'PCSX2 fechado'
        com o PCSX2 aberto e rodando.

        Resposta de falha também é resposta: chamá-la de mudez negava a si mesma e escondia das
        checagens seguintes um canal que está vivo.

        O texto da exceção é a dica em outras palavras — repetido aqui, mandava o usuário ler
        duas vezes a mesma instrução. O detalhe fica com o que foi medido; a dica, com o que fazer."""
        if self._config is None:
            return NO_CONFIG
        pine = self._config.pine
        endpoint = f"{pine.host}:{pine.port}"
        client = PineClient.from_config(pine)
        try:
            client.connect()
        except PineError as exc:
            return False, str(exc), PINE_REFUSED_HINT
        try:
            client.version()
        except PineConnectError:
            client.close()
            waited = pine.timeout_ms / 1000.0
            detail = f"{endpoint} aceitou a conexão mas não respondeu ao MsgVersion dentro de {waited:.1f} s"
            return False, detail, PINE_UNRESPONSIVE_HINT
        except PineError as exc:
            self._client = client
            return True, f"conectado em {endpoint}, servidor PINE respondeu — MsgVersion falhou: {exc}", ""
        self._client = client
        return True, f"conectado em {endpoint}, servidor PINE respondendo", ""

    def _check_version(self) -> tuple[bool, str, str]:
        return self._ask("versão", lambda client: client.version(), "PCSX2 respondeu sem versão")

    def _check_status(self) -> tuple[bool, str, str]:
        if self._client is None:
            return NO_PINE
        try:
            status = self._client.status()
        except PineError as exc:
            return _pine_failure(exc, NO_GAME_HINT)
        name = STATUS_NAMES.get(status, f"desconhecido ({status})")
        if status != STATUS_RUNNING:
            return (
                False,
                f"VM {name}",
                f"inicie o jogo no PCSX2 e tire a emulação da pausa. {SECOND_INSTANCE_HINT}",
            )
        return True, f"VM {name}", ""

    def _check_serial(self) -> tuple[bool, str, str]:
        if self._client is None or self._config is None:
            return NO_PINE
        try:
            self._serial = self._client.serial().strip()
        except PineError as exc:
            return _pine_failure(exc, NO_GAME_HINT)
        if not self._serial:
            return False, "PCSX2 respondeu sem serial", NO_GAME_HINT
        expected = self._config.pine.expected_serial.strip().upper()
        if self._serial.upper() != expected:
            return False, f"jogo rodando: {self._serial}", f"esperado {expected} (Lifeline NTSC-U)"
        return True, self._serial, ""

    def _check_title(self) -> tuple[bool, str, str]:
        return self._ask("título", lambda client: client.title(), "PCSX2 respondeu sem título")

    def _check_memory_read(self) -> tuple[bool, str, str]:
        if self._client is None:
            return NO_PINE
        try:
            value = self._client.read32(EE_PROBE_ADDRESS)
        except PineError as exc:
            return _pine_failure(exc, MEMORY_UNAVAILABLE_HINT)
        return True, f"0x{EE_PROBE_ADDRESS:08X} = 0x{value:08X}", ""

    def _check_recipe(self) -> tuple[bool, str, str]:
        if self._config is None or self._paths is None:
            return NO_CONFIG
        try:
            recipe = WriteRecipe.load(self._paths.recipe)
        except CyhmoError as exc:
            return False, str(exc), "aponte pine.recipe para config/write_recipe.yaml"
        self._recipe = recipe
        reference = self._serial or self._config.pine.expected_serial
        try:
            recipe.validate_serial(reference)
        except CyhmoError as exc:
            return False, str(exc), "use a receita derivada para a sua versão do jogo"
        return True, f"{self._paths.recipe.name}: v{recipe.version}, serial {recipe.serial}, CRC {recipe.crc}", ""

    def _check_patch_address(self) -> tuple[bool, str, str]:
        """Ler o valor patcheado só é problema no modo runtime: no modo pnach ele é o esperado."""
        if self._recipe is None or self._config is None:
            return NO_RECIPE
        if self._client is None:
            return NO_PINE
        patch = self._recipe.ascii_words.code_patch
        try:
            current = self._client.read32(patch.addr)
        except PineError as exc:
            return _pine_failure(exc, MEMORY_UNAVAILABLE_HINT)
        reading = f"0x{patch.addr:08X} = 0x{current:08X}"
        if current == patch.original:
            return True, reading, ""
        if current == patch.patched:
            if self._config.pine.patch_mode == "pnach":
                return True, f"{reading} — patch permanente do .pnach ativo", ""
            return (
                False,
                f"{reading} — patch órfão de uma execução anterior",
                describe_orphan_patch(patch.on_orphan),
            )
        return (
            False,
            f"{reading}, esperado 0x{patch.original:08X}",
            "endereços mudaram ou o jogo é de outra versão; confira a receita de injeção",
        )

    def _check_grammar(self) -> tuple[bool, str, str]:
        """``PineGrammarSource`` devolve ``None`` tanto para cena sem gramática quanto para canal
        morto — engolir o erro é o certo no runtime, onde o mod não pode emudecer, mas aqui as duas
        causas mandam o usuário investigar lugares opostos: uma é a cena, a outra é a engenharia
        reversa, que está correta.

        Por isso o ponteiro de contexto é lido direto, sem a fonte: antes da extração, para pegar o
        canal que morreu nas checagens anteriores, e de novo quando ela vem vazia, porque esta é a
        checagem de I/O mais pesada da lista — a que mais tem chance de o servidor PINE emudecer no
        meio dela. Custa duas leituras de 32 bits."""
        if self._recipe is None:
            return NO_RECIPE
        if self._client is None:
            return NO_PINE
        from cyhmo.state.grammar_source import PineGrammarSource
        from cyhmo.state.service import ROUTE_LABELS

        context_pointer = self._recipe.grammar.context_pointer
        source = PineGrammarSource(self._client, self._recipe.grammar)
        try:
            snapshot = source.read(self._client.read32(context_pointer))
            if snapshot is None or not snapshot.entries:
                pointer = self._client.read32(context_pointer)
                return False, f"nenhum comando extraído (ponteiro 0x{pointer:08X})", NO_GRAMMAR_HINT
        except PineError as exc:
            return _pine_failure(exc, MEMORY_UNAVAILABLE_HINT)
        preview = ", ".join(snapshot.entries[:5])
        route = ROUTE_LABELS.get(snapshot.source, snapshot.source)
        return True, f"{snapshot.size} comandos ativos via {route}: {preview}...", ""

    def _check_annex(self) -> tuple[bool, str, str]:
        if self._config is None or self._paths is None:
            return NO_CONFIG
        if not self._config.intent.annex:
            return True, "anexo desligado (intent.annex vazio)", ""
        from cyhmo.intent.annex import Annex

        path = self._paths.annex
        if not path.is_file():
            return False, f"anexo não encontrado: {path}", "aponte intent.annex para um arquivo válido ou deixe vazio"
        annex = Annex.load(path)
        return True, f"{path.name}: {annex.size} comandos com exemplos", ""

    def _check_llm(self) -> tuple[bool, str, str]:
        """Ligado e quebrado é o pior caso: o mod cala nos comandos ambíguos e nada avisa."""
        if self._config is None:
            return NO_CONFIG
        llm = self._config.intent.llm
        if not llm.enabled:
            return True, "desligado (intent.llm.enabled = false)", ""
        from cyhmo.intent.llm.factory import build_llm_provider

        try:
            build_llm_provider(llm)
        except CyhmoError as exc:
            return False, str(exc), "preencha intent.llm.model no config.toml ou desligue intent.llm.enabled"
        if llm.provider == "anthropic":
            import os

            if not os.environ.get(llm.api_key_env):
                return False, f"variável {llm.api_key_env} não definida", f"exporte {llm.api_key_env} com a chave da API"
            return True, f"anthropic: {llm.model}", ""
        if llm.provider in ("ollama", "openai_compat"):
            return _probe_llm_endpoint(llm.endpoint, llm.provider, llm.model)
        return True, f"{llm.provider}: {llm.model}", ""

    def _check_interface_texts(self) -> tuple[bool, str, str]:
        """Locale faltando ou com chave a menos deixa a interface mostrando o nome da chave."""
        if self._config is None:
            return NO_CONFIG
        from cyhmo.ui.i18n import available_languages, flatten, load_locale, resolve_language

        ui = self._config.ui
        language = resolve_language(ui.language, self._config.languages.primary)
        try:
            keys = {code: set(flatten(load_locale(code))) for code in available_languages()}
        except CyhmoError as exc:
            return False, str(exc), "reinstale o mod: os textos da interface vêm com ele"
        reference = keys[language]
        incomplete = [code for code, other in keys.items() if other != reference]
        detail = f"modo {ui.mode}, idioma {language} ({len(reference)} textos)"
        if incomplete:
            return False, f"{detail} | divergentes: {', '.join(sorted(incomplete))}", "algum locale foi editado à mão"
        return True, detail, ""

    def _ask(self, label: str, call: Callable[[PineClient], str], empty_message: str) -> tuple[bool, str, str]:
        if self._client is None:
            return NO_PINE
        try:
            answer = call(self._client).strip()
        except PineError as exc:
            return _pine_failure(exc, NO_GAME_HINT)
        if not answer:
            return False, empty_message, NO_GAME_HINT
        return True, f"{label}: {answer}", ""


def _probe_llm_endpoint(endpoint: str, provider: str, model: str) -> tuple[bool, str, str]:
    import socket
    from urllib.parse import urlsplit

    parts = urlsplit(endpoint)
    port = parts.port or (443 if parts.scheme == "https" else 80)
    host = parts.hostname or "127.0.0.1"
    try:
        with socket.create_connection((host, port), timeout=2.0):
            return True, f"{provider}: {model} em {host}:{port}", ""
    except OSError as exc:
        return False, f"{host}:{port} não responde ({exc})", f"suba o servidor de {provider} ou desligue intent.llm.enabled"


def _evaluate(name: str, probe: Probe) -> CheckResult:
    try:
        ok, detail, hint = probe()
    except CyhmoError as exc:
        return CheckResult(name, False, str(exc), "")
    except Exception as exc:
        return CheckResult(name, False, f"{type(exc).__name__}: {exc}", "falha inesperada nesta checagem")
    return CheckResult(name, ok, detail, hint)


def _render(stream: TextIO, index: int, total: int, result: CheckResult) -> None:
    label = f"{result.name} "
    dots = "." * max(3, NAME_WIDTH - len(label))
    print(f"[{index:2d}/{total}] {label}{dots} {'OK' if result.ok else 'FALHOU'}", file=stream)
    if result.detail:
        print(f"         {result.detail}", file=stream)
    if not result.ok and result.hint:
        print(f"         dica: {result.hint}", file=stream)


def _has_files(directory: Path) -> bool:
    return directory.is_dir() and any(entry.is_file() for entry in directory.rglob("*"))

