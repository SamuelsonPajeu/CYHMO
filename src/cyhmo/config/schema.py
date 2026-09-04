"""Esquema do ``config.toml``.

Toda chave tem default; chave desconhecida ou valor fora da faixa aborta com
mensagem apontando a chave. Segredos nunca entram aqui:
o arquivo guarda apenas o NOME da variável de ambiente.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CONFIG_VERSION = 1


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class AudioConfig(StrictModel):
    device: str | int | None = "default"
    sample_rate: int = Field(default=16_000, ge=8_000, le=48_000)
    block_ms: int = Field(default=20, ge=10, le=100)


class VadConfig(StrictModel):
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    tail_ms: int = Field(default=250, ge=50, le=2_000)
    min_utterance_ms: int = Field(default=200, ge=0, le=5_000)
    max_utterance_s: float = Field(default=15.0, gt=0.0, le=60.0)


class ActivationConfig(StrictModel):
    mode: Literal["ptt", "vad"] = "ptt"
    ptt_hotkey: str = "right ctrl"
    pre_roll_ms: int = Field(default=300, ge=0, le=2_000)
    vad: VadConfig = Field(default_factory=VadConfig)


class WhisperCppConfig(StrictModel):
    """Backend padrão de transcrição. O executável e o peso vêm de ``cyhmo setup``.

    ``use_gpu`` nasce falso porque o build que o setup instala é só CPU: ligá-lo sobre esse
    build não muda onde a conta é feita, só tira o ``-ng`` da linha de comando. Quem tem
    placa NVIDIA instala o build cuBLAS pelo painel, e ele vai para ``gpu_binary`` — uma
    pasta própria, para que voltar à CPU seja trocar a opção e não baixar tudo de novo.
    Sem esse build no lugar, ``use_gpu`` ligado cai de volta na CPU e avisa no log."""

    binary: str = "whisper_cpp/whisper-server.exe"
    gpu_binary: str = "whisper_cpp/cuda/whisper-server.exe"
    model: str = "models/stt/ggml/ggml-small.bin"
    host: str = "127.0.0.1"
    port: int = Field(default=8178, ge=1, le=65_535)
    threads: int = Field(default=8, ge=1, le=128)
    use_gpu: bool = False
    flash_attn: bool = True
    audio_ctx: int = Field(default=512, ge=0, le=1_500)
    auto_start: bool = True
    timeout_ms: int = Field(default=15_000, ge=500, le=120_000)


class SttConfig(StrictModel):
    """``engine`` é whisper-cpp por medição: ~0,7 s por comando contra ~1,9 s do
    faster-whisper na mesma CPU. O faster-whisper continua como rede de segurança — se o
    servidor do whisper.cpp não subir, o mod troca sozinho e avisa no log."""

    engine: Literal["faster-whisper", "whisper-cpp", "fake"] = "whisper-cpp"
    model: str = "small"
    compute_type: Literal["auto", "int8", "int8_float16", "int8_float32", "float16", "float32"] = "int8"
    device: Literal["auto", "cpu", "cuda"] = "cpu"
    language: str = "pack"
    beam_size: int = Field(default=1, ge=1, le=10)
    warm_up: bool = True
    cpu_threads: int = Field(default=0, ge=0, le=128)
    temperature_fallback: bool = False
    hotwords: bool = True
    max_hotwords: int = Field(default=48, ge=0, le=256)
    silence_gate_ratio: float = Field(default=0.125, ge=0.0, le=1.0)
    whisper_cpp: WhisperCppConfig = Field(default_factory=WhisperCppConfig)

    @field_validator("language")
    @classmethod
    def _language(cls, language: str) -> str:
        language = language.strip().lower()
        if language in {"pack", "auto"} or len(language) in (2, 3):
            return language
        raise ValueError("stt.language deve ser 'pack', 'auto' ou um código ISO curto (ex.: 'pt')")


class LanguagesConfig(StrictModel):
    packs_dir: str = "languages"
    enabled: list[str] = Field(default_factory=lambda: ["pt-BR", "en"])
    primary: str = "pt-BR"
    registry_url: str = "https://raw.githubusercontent.com/SamuelsonPajeu/CYHMO/main/languages/index.json"

    @model_validator(mode="after")
    def _primary_enabled(self) -> "LanguagesConfig":
        if not self.enabled:
            raise ValueError("languages.enabled não pode ser vazio")
        if self.primary not in self.enabled:
            raise ValueError(f"languages.primary {self.primary!r} precisa estar em languages.enabled {self.enabled}")
        return self


class LlmConfig(StrictModel):
    """``keep_alive`` mantém o modelo residente no Ollama: medido em 2026-08-26, a chamada
    quente custa ~600 ms e a fria ~6,5 s, quase toda em carregamento do disco — sem ele
    qualquer intervalo entre comandos faz a chamada seguinte estourar o timeout.

    ``max_output_tokens`` é pequeno de propósito: o assistente responde o TEXTO do comando,
    não JSON, e a entrada mais longa da gramática cabe com folga em 16 tokens."""

    enabled: bool = False
    mode: Literal["fallback", "pair"] = "fallback"
    provider: Literal["ollama", "openai_compat", "anthropic", "fake"] = "ollama"
    model: str = ""
    endpoint: str = "http://127.0.0.1:11434"
    api_key_env: str = "CYHMO_LLM_API_KEY"
    timeout_ms: int = Field(default=800, ge=100, le=60_000)
    keep_alive: str = "30m"
    max_output_tokens: int = Field(default=16, ge=1, le=256)
    warm_up: bool = True
    in_battle: bool = False
    prompt_top_k: int = Field(default=20, ge=3, le=60)

    @field_validator("api_key_env")
    @classmethod
    def _env_name_not_secret(cls, name: str) -> str:
        if not name or " " in name or name.startswith(("sk-", "sk_ant", "key-")) or len(name) > 64:
            raise ValueError("intent.llm.api_key_env deve ser o NOME de uma variável de ambiente, não a chave")
        return name


class IntentConfig(StrictModel):
    embedding_backend: Literal["sentence_transformers", "hashing"] = "sentence_transformers"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    accept_threshold: float = Field(default=0.78, ge=-1.0, le=1.0)
    confident_threshold: float = Field(default=0.92, ge=-1.0, le=1.0)
    accept_margin: float = Field(default=0.05, ge=0.0, le=1.0)
    reject_threshold: float = Field(default=0.55, ge=-1.0, le=1.0)
    accept_threshold_no_context: float = Field(default=0.85, ge=-1.0, le=1.0)
    stale_grammar_penalty: float = Field(default=0.05, ge=0.0, le=0.5)
    top_k: int = Field(default=3, ge=1, le=10)
    max_query_variants: int = Field(default=3, ge=1, le=8)
    annex: str = "catalog/commands.yaml"
    observed_vocab: str = "data/observed_vocab.yaml"
    embedding_cache: str = "data/embeddings"
    llm: LlmConfig = Field(default_factory=LlmConfig)

    @model_validator(mode="after")
    def _thresholds_ordered(self) -> "IntentConfig":
        if not self.reject_threshold < self.accept_threshold:
            raise ValueError(
                f"intent.reject_threshold ({self.reject_threshold}) deve ser menor que "
                f"intent.accept_threshold ({self.accept_threshold})"
            )
        return self


class StateConfig(StrictModel):
    polling_hz: int = Field(default=15, ge=1, le=30)
    grammar_seed: str = ""
    infer_mode_from_grammar: bool = True


class PineConfig(StrictModel):
    """``connect_timeout_ms`` é bem maior que ``timeout_ms`` de propósito: o SO leva ~2 s
    para recusar uma conexão em 127.0.0.1, e cortar antes disso faz o PCSX2 FECHADO chegar
    como timeout e ser acusado de travado — a mensagem errada no caso mais comum de todos."""

    host: str = "127.0.0.1"
    port: int = Field(default=28011, ge=1, le=65_535)
    timeout_ms: int = Field(default=1_000, ge=50, le=30_000)
    connect_timeout_ms: int = Field(default=3_000, ge=50, le=30_000)
    auto_reconnect: bool = True
    recipe: str = "config/write_recipe.yaml"
    patch_mode: Literal["runtime", "pnach"] = "runtime"
    patch_hold_ms: int = Field(default=250, ge=0, le=5_000)
    verify_oracle: bool = True
    expected_serial: str = "SLUS-20848"


class InjectConfig(StrictModel):
    """``confidence`` é a nota de reconhecimento que o mod entrega ao jogo (0..9999). Ela não
    é derivada do casamento de intenção: o mod substitui o reconhecedor e não mede pronúncia,
    então converter score de embeddings em nota de pronúncia seria inventar o número. Só o
    tutorial exibe essa nota, mas o jogo a compara com um limiar próprio (3800 no NTSC-U) e
    abaixo dele responde "Failed in recognition"."""

    enabled: bool = True
    require_can_talk: bool = False
    confidence: int = Field(default=8000, ge=0, le=9999)


class UiConfig(StrictModel):
    """``mode`` só muda por edição do arquivo: o modo avançado expõe telas que
    confundem quem só quer jogar, e não deve ser alcançável por clique acidental."""

    host: str = "127.0.0.1"
    port: int = Field(default=8765, ge=1, le=65_535)
    open_browser: bool = True
    mode: Literal["basic", "advanced"] = "basic"
    language: Literal["auto", "pt-BR", "en"] = "auto"
    theme: Literal["auto", "dark", "light"] = "auto"
    log_level: Literal["debug", "info", "warning", "error"] = "info"
    log_file: str = "data/logs/cyhmo.log"


class UpdateConfig(StrictModel):
    """A checagem só lê a API pública de releases do GitHub; nada da máquina é enviado.

    ``skipped_version`` é escrito pelo botão "pular esta versão" da interface: o aviso
    some para aquela versão e volta sozinho na próxima release."""

    check_on_start: bool = True
    repository: str = "SamuelsonPajeu/CYHMO"
    skipped_version: str = ""

    @field_validator("repository")
    @classmethod
    def _owner_and_name(cls, repository: str) -> str:
        repository = repository.strip().strip("/")
        if repository.count("/") != 1 or not all(part.strip() for part in repository.split("/")):
            raise ValueError('update.repository deve ser "dono/repositorio", como "SamuelsonPajeu/CYHMO"')
        return repository


class DebugConfig(StrictModel):
    save_audio: bool = False
    audio_dir: str = "data/debug_audio"
    telemetry_dir: str = "data/logs"


class AppConfig(StrictModel):
    config_version: int = Field(default=CONFIG_VERSION, ge=1)
    models_dir: str = "models"
    data_dir: str = "data"
    audio: AudioConfig = Field(default_factory=AudioConfig)
    activation: ActivationConfig = Field(default_factory=ActivationConfig)
    stt: SttConfig = Field(default_factory=SttConfig)
    languages: LanguagesConfig = Field(default_factory=LanguagesConfig)
    intent: IntentConfig = Field(default_factory=IntentConfig)
    state: StateConfig = Field(default_factory=StateConfig)
    pine: PineConfig = Field(default_factory=PineConfig)
    inject: InjectConfig = Field(default_factory=InjectConfig)
    ui: UiConfig = Field(default_factory=UiConfig)
    update: UpdateConfig = Field(default_factory=UpdateConfig)
    debug: DebugConfig = Field(default_factory=DebugConfig)

    @field_validator("config_version")
    @classmethod
    def _supported_version(cls, version: int) -> int:
        if version > CONFIG_VERSION:
            raise ValueError(
                f"config_version {version} é mais novo que o suportado ({CONFIG_VERSION}); atualize o mod"
            )
        return version

    def paths(self, base_dir: Path) -> "ProjectPaths":
        return ProjectPaths.from_config(self, base_dir)

    def replace(self, **sections: BaseModel) -> "AppConfig":
        return self.model_copy(update=sections, deep=True)


@dataclass(frozen=True)
class ProjectPaths:
    """Caminhos absolutos derivados da config, relativos à pasta do ``config.toml``."""

    base_dir: Path
    models_dir: Path
    data_dir: Path
    packs_dir: Path
    annex: Path
    observed_vocab: Path
    embedding_cache: Path
    recipe: Path
    log_file: Path
    audio_dir: Path
    telemetry_dir: Path
    grammar_seed: Path | None

    @classmethod
    def from_config(cls, config: AppConfig, base_dir: Path) -> "ProjectPaths":
        base_dir = base_dir.resolve()

        def resolve(relative: str) -> Path:
            return (base_dir / relative).resolve()

        return cls(
            base_dir=base_dir,
            models_dir=resolve(config.models_dir),
            data_dir=resolve(config.data_dir),
            packs_dir=resolve(config.languages.packs_dir),
            annex=resolve(config.intent.annex),
            observed_vocab=resolve(config.intent.observed_vocab),
            embedding_cache=resolve(config.intent.embedding_cache),
            recipe=resolve(config.pine.recipe),
            log_file=resolve(config.ui.log_file),
            audio_dir=resolve(config.debug.audio_dir),
            telemetry_dir=resolve(config.debug.telemetry_dir),
            grammar_seed=resolve(config.state.grammar_seed) if config.state.grammar_seed else None,
        )

    def ensure_directories(self) -> None:
        for directory in (
            self.models_dir,
            self.data_dir,
            self.embedding_cache,
            self.log_file.parent,
            self.audio_dir,
            self.telemetry_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
