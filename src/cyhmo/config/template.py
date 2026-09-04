"""Modelo default do ``config.toml``."""

CONFIG_TEMPLATE = """\
# config.toml — CYHMO
# Gerado com os defaults. Edite à vontade: chave desconhecida ou valor inválido
# aborta a inicialização com a chave apontada. Caminhos são relativos
# a esta pasta.

config_version = {config_version}
models_dir = {models_dir}            # modelos baixados (STT, embeddings)
data_dir = {data_dir}                # estado do usuário: vocabulário observado, cache, logs

[audio]
device = {audio.device}              # "default", trecho do nome ou índice numérico
sample_rate = {audio.sample_rate}    # alvo do pipeline; captura reamostra se preciso
block_ms = {audio.block_ms}          # tamanho do bloco de captura

[activation]
mode = {activation.mode}             # "ptt" (segurar tecla) | "vad"
ptt_hotkey = {activation.ptt_hotkey} # tecla do push-to-talk;
pre_roll_ms = {activation.pre_roll_ms}  # áudio ANTES do gatilho;

[activation.vad]
threshold = {activation.vad.threshold}
tail_ms = {activation.vad.tail_ms}                 # silêncio que encerra a fala
min_utterance_ms = {activation.vad.min_utterance_ms}   # abaixo disso: descarta (ruído)
max_utterance_s = {activation.vad.max_utterance_s}     # teto duro

[stt]
engine = {stt.engine}                # "whisper-cpp" (padrão, ~3x mais rápido) | "faster-whisper" (rede de segurança) | "fake" (debug)
model = {stt.model}                  # do faster-whisper: tiny | base | small | distil-small.en ...
compute_type = {stt.compute_type}    # int8: CPU; float16: GPU NVIDIA
device = {stt.device}                # "cpu" | "cuda" | "auto"
language = {stt.language}            # "pack" (idioma do pacote primário) | "auto" | código ISO
beam_size = {stt.beam_size}
warm_up = {stt.warm_up}
cpu_threads = {stt.cpu_threads}       # 0 = automático; recomendado = 8
temperature_fallback = {stt.temperature_fallback}  # redecodifica quando o decoder falha;
hotwords = {stt.hotwords}             # ancora o decoder no vocabulário do pacote primário
max_hotwords = {stt.max_hotwords}     # âncora curta demais faz o silêncio virar comando; longa demais puxa a fala e custa latência
silence_gate_ratio = {stt.silence_gate_ratio}  # descarta enunciado abaixo desta fração da SUA fala típica; 0 desliga

[stt.whisper_cpp]                    # backend padrão; "cyhmo setup" baixa o executável e o modelo
binary = {stt.whisper_cpp.binary}    # instalado por "cyhmo setup"; nada é compilado
gpu_binary = {stt.whisper_cpp.gpu_binary}  # build cuBLAS, instalado pelo painel; sem ele use_gpu cai na CPU
model = {stt.whisper_cpp.model}      # modelo no formato ggml
host = {stt.whisper_cpp.host}
port = {stt.whisper_cpp.port}
threads = {stt.whisper_cpp.threads}
use_gpu = {stt.whisper_cpp.use_gpu}  # exige placa NVIDIA e o build de gpu_binary; o painel cuida dos dois
flash_attn = {stt.whisper_cpp.flash_attn}
audio_ctx = {stt.whisper_cpp.audio_ctx}  # 0 = janela cheia; reduzir acelera e arrisca cortar fala longa
auto_start = {stt.whisper_cpp.auto_start}  # o mod sobe e derruba o servidor junto com a sessão
timeout_ms = {stt.whisper_cpp.timeout_ms}

[languages]                          # pacotes de idioma (dados em languages/*.yaml)
# Na primeira execução estes dois vieram do idioma do Windows, com inglês de reserva;
# daqui em diante quem manda é o que está escrito aqui.
packs_dir = {languages.packs_dir}
enabled = {languages.enabled}        # pacotes carregados; todos participam do matching
primary = {languages.primary}        # define STT e as tabelas de numerais/partes/direções
registry_url = {languages.registry_url}  # índice de pacotes que a interface oferece para baixar

[intent]
embedding_backend = {intent.embedding_backend}   # "sentence_transformers" | "hashing" (debug)
embedding_model = {intent.embedding_model}
accept_threshold = {intent.accept_threshold}     # calibrado p/ paraphrase-multilingual-mpnet-base-v2
confident_threshold = {intent.confident_threshold}   # score sozinho, sem margem
accept_margin = {intent.accept_margin}           # margem exigida ENTRE accept e confident; empate vai ao fallback
reject_threshold = {intent.reject_threshold}     # abaixo disso rejeita sem nem chamar o LLM
accept_threshold_no_context = {intent.accept_threshold_no_context}
stale_grammar_penalty = {intent.stale_grammar_penalty}  # elevação do limiar com gramática stale
top_k = {intent.top_k}
max_query_variants = {intent.max_query_variants}  # original + traduções lexicais consultadas por enunciado
annex = {intent.annex}               # anexo semântico opcional; "" desliga
observed_vocab = {intent.observed_vocab}
embedding_cache = {intent.embedding_cache}   # cache por string, persistido

[intent.llm]                         # assistente opcional; exige o Ollama (ou outro provedor) instalado
enabled = {intent.llm.enabled}
mode = {intent.llm.mode}             # "fallback" (só quando o matching não decide) | "pair" (arbitra cada enunciado)
# no modo par o assistente é chamado uma vez por SEGMENTO, não por enunciado: "anda, para e corre" custa
# 3 chamadas em série, cada uma podendo gastar o timeout_ms inteiro
provider = {intent.llm.provider}     # "ollama" | "openai_compat" | "anthropic"
model = {intent.llm.model}
endpoint = {intent.llm.endpoint}
api_key_env = {intent.llm.api_key_env}   # NOME da variável de ambiente com a chave
timeout_ms = {intent.llm.timeout_ms}   # prazo duro; se passar, injeta o palpite do matcher
keep_alive = {intent.llm.keep_alive}   # o Ollama segura o modelo por esse tempo; quente responde em ~50 ms, frio custa ~6,5 s e estoura o timeout
max_output_tokens = {intent.llm.max_output_tokens}
warm_up = {intent.llm.warm_up}       # warm up do modelo no boot do mod
in_battle = {intent.llm.in_battle}   # suprimido em combate
prompt_top_k = {intent.llm.prompt_top_k}   # só o top-k vai ao prompt

[state]
polling_hz = {state.polling_hz}
grammar_seed = {state.grammar_seed}  # arquivo usado só até a memória do jogo responder; "" desliga
infer_mode_from_grammar = {state.infer_mode_from_grammar}   # batalha/diálogo inferidos do vocabulário

[pine]
host = {pine.host}
port = {pine.port}
timeout_ms = {pine.timeout_ms}       # espera por resposta de um pedido na conexão já aberta
connect_timeout_ms = {pine.connect_timeout_ms}  # maior de propósito: o SO leva ~2 s para recusar em 127.0.0.1; o PCSX2 fechado parece travado
auto_reconnect = {pine.auto_reconnect}
recipe = {pine.recipe}
patch_mode = {pine.patch_mode}       # "runtime" (o mod aplica e restaura) | "pnach"
patch_hold_ms = {pine.patch_hold_ms}
verify_oracle = {pine.verify_oracle}
expected_serial = {pine.expected_serial}

[inject]
enabled = {inject.enabled}           # false = não escreve no jogo (debug)
require_can_talk = {inject.require_can_talk}   # exige que a cena aceite fala; ainda não é lido do jogo, mantenha false
confidence = {inject.confidence}          # nota do reconhecimento, serve de patch pro tutorial

[ui]
host = {ui.host}
port = {ui.port}
open_browser = {ui.open_browser}
mode = {ui.mode}                     # "basic" | "advanced"
language = {ui.language}             # "auto" (languages.primary) | "pt-BR" | "en"
theme = {ui.theme}                   # "auto" | "dark" | "light"
log_level = {ui.log_level}
log_file = {ui.log_file}

[update]                             # avisa quando sai release nova e instala pela interface
check_on_start = {update.check_on_start}   # consulta as releases do GitHub no boot; false desliga o aviso
repository = {update.repository}     # "dono/repositorio" de onde vem o pacote portátil
skipped_version = {update.skipped_version}   # escrito pelo "pular esta versão"; "" = nenhuma ignorada

[debug]
save_audio = {debug.save_audio}      # grava cada enunciado em .wav
audio_dir = {debug.audio_dir}
telemetry_dir = {debug.telemetry_dir}
"""
