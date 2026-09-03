<div align="center">

# CYHMO

*Can you hear me, operator?*

**Português** · [English](README.en.md)

![Windows 10 ou 11](https://img.shields.io/badge/Windows-10%20ou%2011-29434C?style=flat-square&labelColor=0E1A21&logo=windows&logoColor=56E8E0)
[![PCSX2 2.6.3+](https://img.shields.io/badge/PCSX2-2.6.3-29434C?style=flat-square&labelColor=0E1A21)](https://pcsx2.net/)
![Python 3.11+](https://img.shields.io/badge/Python-3.11+-29434C?style=flat-square&labelColor=0E1A21&logo=python&logoColor=56E8E0)
[![Licença MIT](https://img.shields.io/badge/licen%C3%A7a-MIT-29434C?style=flat-square&labelColor=0E1A21)](LICENSE)
[![Baixar](https://img.shields.io/badge/baixar-CYHMO__portable.zip-56E8E0?style=flat-square&labelColor=0E1A21)](https://github.com/SamuelsonPajeu/CYHMO/releases)

<img src="docs/screenshot-panel.png" width="880"
     alt="Painel do CYHMO no tema escuro: a frase falada em português, o comando correspondente enviado ao jogo e a tabela dos últimos comandos.">

</div>

**Lifeline** (*Operator's Side*, PS2, 2003) é jogado inteiramente por voz: você fala e
a Rio obedece. O reconhecedor original só entende inglês, erra muito, e é o motivo de o
jogo ter fama de injogável.

O CYHMO troca esse reconhecedor por um moderno. Você fala **no seu idioma**, o mod
transcreve na sua máquina, descobre qual comando do jogo corresponde ao que você disse e
escreve esse comando na memória do emulador.

O vocabulário não é uma lista fixa: a cada troca de cena o mod lê, da memória do jogo, os
comandos que **aquela cena** aceita.

---

## Antes de começar

| | |
|---|---|
| Sistema | Windows 10 ou 11 |
| Emulador | [PCSX2](https://pcsx2.net/) (2.6.3) |
| Jogo | **sua própria cópia** do Lifeline **NTSC-U** (serial `SLUS-20848`) — outras regiões não funcionam |
| Microfone | |

**Ajuste o PCSX2:**

1. `Tools → Show Advanced Settings` → ative a opção e confirme o diálogo.
2. `Settings → Advanced → PINE Settings` → marque **Enable**, deixe o slot em **28011**.
3. `Settings → Controller → USB Port 1` → configure um microfone no Player 1 para habilitar o reconhecimento nativo do jogo.

---

## Hardware

Por padrão o reconhecimento é processado na **CPU**.

| | Mínimo | Recomendado |
|---|---|---|
| CPU | 4 núcleos / 8 threads recentes | 8 núcleos |
| RAM | 8 GB | 16 GB |
| Disco | 4 GB livres | 6 GB livres, em SSD |
| Placa de vídeo | - | - |

**Backends**

Nos meus testes o whisper.cpp se saiu melhor com menor latência.

* whisper.cpp — **padrão**
* faster-whisper

### Assistente LLM

O assistente é **opcional e desligado por padrão**. Ele só é executado quando o reconhecimento
fica em dúvida, e roda localmente pelo Ollama. Ele soma a estes requisitos:

| | Mínimo | Recomendado |
|---|---|---|
| Modelo | `qwen2.5:1.5b` (~1 GB) | `qwen2.5:3b` (~2 GB) |
| VRAM livre | ~6 GB no total | 8 GB ou mais |
| RAM | 16 GB | 32 GB, se rodar o modelo na CPU |

Custo por comando, medido: **~25 ms** na GPU e **~215 ms** na CPU. Sem GPU o assistente
continua utilizável, mas come um quinto do orçamento de latência.

> **Atenção:** esses requisitos não levam em conta os recursos que o jogo usa
> durante a execução no emulador.

---

## Instalação

Requer **Python 3.11+** instalado: [Python](https://www.python.org/downloads/) | [uv](https://docs.astral.sh/uv/getting-started/installation/).

Baixe `CYHMO_portable.zip` em
[Releases](https://github.com/SamuelsonPajeu/CYHMO/releases), extraia os arquivos em uma pasta e
execute **`CYHMO.cmd`**. Na primeira execução ele baixa as dependências e leva alguns minutos.

Ou

<details>
<summary>Clone o projeto</summary>

<br>

```powershell
git clone https://github.com/SamuelsonPajeu/CYHMO.git
cd CYHMO
.\install.ps1
.\.venv\Scripts\python.exe -m cyhmo run
```

</details>

---

## Jogar

1. Abra o PCSX2 e carregue o jogo (a ordem não importa; o mod espera o emulador).
2. Abra o CYHMO. Ele sobe a interface em <http://127.0.0.1:8765> e abre o navegador.
3. **Segure <kbd>Ctrl</kbd> direito, fale o comando, solte.**

A aba **Cheats** lista
tudo que a cena atual aceita. Clicar num item envia o comando direto ao jogo.

**Se algo não funcionar,** abra um terminal na pasta do CYHMO e rode o diagnóstico: ele
confere ambiente, emulador, jogo, microfone e modelos.

```powershell
.\CYHMO.cmd doctor
```

---

## Problema com o tutorial

O reconhecimento de voz do tutorial é levemente diferente do resto do jogo, o mod até tem suporte e processa múltiplos comandos e os dispara, mas não é muito bom e consistente nisso, por isso recomendo que use o reconhecedor nativo do próprio jogo para passar o tutorial. (Por sorte é a parte que o jogo funciona melhor :D)

---

## Avisos legais

- O mod **exige uma cópia própria e legítima** do jogo. Ele **não fornece, não distribui e
  não indica onde obter** o jogo — nem ISO, nem executável. O vocabulário que o mod usa
  para jogar sai da memória da **sua** cópia, na sua máquina, a cada troca de cena, e não
  é redistribuído. A única exceção está em `datasets/grammars/exploration.yaml`: 62 nomes
  de comando de uma cena, usados apenas como fixture de calibração.
- Projeto independente, **sem afiliação** com a Konami, a SCEJ/Sony ou a equipe do PCSX2.
  Todas as marcas pertencem aos seus donos.
- O guia de Lifeline de **Daniel Engel** no GameFAQs foi usado como referência de
  pesquisa:
  <https://gamefaqs.gamespot.com/ps2/561643-lifeline/faqs>
- Os modelos baixados na primeira execução mantêm as licenças de seus autores e não são
  redistribuídos: Whisper (OpenAI, MIT) e as conversões `faster-whisper` da Systran,
  whisper.cpp / ggml (MIT), Silero VAD (MIT), `paraphrase-multilingual-mpnet-base-v2`
  (Apache-2.0).
- Código deste repositório sob licença **[MIT](LICENSE)**. Ela cobre o código e os pacotes
  de idioma, e não concede nada sobre o jogo, o guia, o emulador ou os modelos.
