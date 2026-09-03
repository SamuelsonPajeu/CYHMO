"""Extração pura do vocabulário de um blob de gramática ``VGP 2.00``.

Sem socket: recebe bytes e devolve as frases que o matcher nativo aceita, mais as
rejeitadas pelo filtro anti-ruído (para log — token derrubado por engano some em silêncio).

Duas rotas, nesta ordem de preferência:

``parse_records`` lê a **estrutura declarada** do blob (contagem, tabela de registros,
pool de strings) e devolve os limites exatos de cada string. É a fonte de verdade.

``extract_vocabulary`` varre os bytes procurando texto e adivinha os limites. Só serve
quando a estrutura não valida — por exemplo num blob cujo pool é japonês.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass

EE_RAM_SIZE = 0x02000000
PHYSICAL_MASK = 0x01FFFFFF
GRAMMAR_MAGIC = b"VGP 2.00"
DEFAULT_GAP = 0x400
MIN_PHRASE_LENGTH = 2
MAX_PHRASE_LENGTH = 40
MAX_ABBREVIATION_LENGTH = 4
MIN_PACKED_RUN = 12
VOWELS = frozenset("aeiouyAEIOUY")
TOKEN_EXTRA_CHARS = frozenset("'-?")
JUNK_MARKERS = ("=", ":", "\\", "_", "dOUT", "Dummy", "dummy", "BID_", "System")
RUN_PATTERN = re.compile(rb"[A-Za-z][ -~]*")
PACKED_BOUNDARY = re.compile(r"(?<=[a-z0-9?'.])(?=[A-Z])")

HEADER_SIZE = 0x10
RECORD_SIZE = 0x10
RECORD_OFFSET_FIELD = 0x08
MAX_RECORDS = 4096


@dataclass(frozen=True)
class GrammarExtraction:
    """``end_offset`` é o fim (relativo ao blob) da última frase aceita antes do corte por gap."""

    entries: tuple[str, ...]
    rejected: tuple[str, ...]
    end_offset: int
    base_address: int = 0

    @property
    def end_address(self) -> int:
        return self.base_address + self.end_offset


@dataclass(frozen=True)
class GrammarLayout:
    """Cabeçalho do blob: quantos registros a tabela tem e onde o pool de strings termina."""

    record_count: int
    pool_end: int

    @property
    def table_end(self) -> int:
        return HEADER_SIZE + self.record_count * RECORD_SIZE

    @property
    def required_bytes(self) -> int:
        return self.pool_end


def read_layout(head: bytes, magic: bytes = GRAMMAR_MAGIC) -> GrammarLayout | None:
    """Valida o cabeçalho com o mínimo de bytes possível, para o chamador saber quanto ler."""
    if len(head) < HEADER_SIZE or not head.startswith(magic):
        return None
    count, pool_end = struct.unpack_from("<2I", head, len(magic))
    layout = GrammarLayout(count, pool_end)
    if not 0 < count <= MAX_RECORDS or not layout.table_end <= pool_end:
        return None
    return layout


def parse_records(blob: bytes, base_address: int = 0, magic: bytes = GRAMMAR_MAGIC) -> GrammarExtraction | None:
    """Lê o vocabulário pelos limites que o próprio blob declara.

    Cada registro da tabela guarda o deslocamento da sua string; o texto vai daí até o
    byte nulo **ou** até o começo da próxima string. Sem essa segunda condição, comando
    cujo tamanho é múltiplo de 8 sai grudado no seguinte — o pool alinha em 8 bytes e
    omite o terminador quando ele não cabe ("Sit down", "Vending machines").

    ``None`` quando a estrutura não confere: aí o chamador cai na varredura por bytes.
    """
    layout = read_layout(blob, magic)
    if layout is None or layout.pool_end > len(blob):
        return None
    offsets: set[int] = set()
    for index in range(layout.record_count):
        field = HEADER_SIZE + index * RECORD_SIZE + RECORD_OFFSET_FIELD
        offset = struct.unpack_from("<I", blob, field)[0]
        if not layout.table_end <= offset < layout.pool_end:
            return None
        offsets.add(offset)
    ordered = sorted(offsets)
    entries: list[str] = []
    rejected: list[str] = []
    for position, offset in enumerate(ordered):
        stop = ordered[position + 1] if position + 1 < len(ordered) else layout.pool_end
        text = _pool_string(blob[offset:stop])
        if text is None:
            continue
        target = entries if looks_like_vocabulary(text) else rejected
        if text not in target:
            target.append(text)
    return GrammarExtraction(tuple(entries), tuple(rejected), layout.pool_end, base_address)


def _pool_string(slot: bytes) -> str | None:
    terminator = slot.find(b"\x00")
    raw = slot if terminator < 0 else slot[:terminator]
    try:
        return raw.decode("ascii")
    except UnicodeDecodeError:
        return None


def looks_like_vocabulary(text: str) -> bool:
    """Filtro mínimo da rota estruturada: o pool só tem vocabulário, então basta descartar
    artefato de build e marcador sem letra. O filtro linguístico de ``looks_like_phrase``
    seria destrutivo aqui — derrubaria entradas legítimas com ponto ("Sarah...",
    "I get it. kind of") que a varredura por bytes nunca chega a ver."""
    if not text or not text.isprintable() or not any(char.isalpha() for char in text):
        return False
    return not any(marker in text for marker in JUNK_MARKERS)


def physical(addr: int) -> int:
    """O jogo guarda ponteiros nos espelhos 0x2…/0x3… da RAM; o PINE só enxerga 0x0…–0x01FFFFFF."""
    return addr & PHYSICAL_MASK


def normalize_entry(text: str) -> str:
    return text.strip().lower()


def looks_like_command(token: str) -> bool:
    """Filtro conservador: o ruído fonético do blob não tem vogal ou tem maiúscula no meio."""
    if token.isdigit():
        return True
    if not token or not token[0].isalpha():
        return False
    if _is_short_acronym(token) or _is_abbreviation(token):
        return True
    if any(char.isupper() for char in token[1:]):
        return False
    if not any(char in VOWELS for char in token):
        return False
    return all(char.isalpha() or char in TOKEN_EXTRA_CHARS for char in token)


def _is_short_acronym(token: str) -> bool:
    return 2 <= len(token) <= 3 and token.isalnum() and token.isupper()


def _is_abbreviation(token: str) -> bool:
    body = token[:-1]
    return (
        token.endswith(".")
        and 1 <= len(body) <= MAX_ABBREVIATION_LENGTH
        and body.isalpha()
        and body[0].isupper()
        and body[1:] == body[1:].lower()
    )


def looks_like_phrase(text: str) -> bool:
    if not MIN_PHRASE_LENGTH <= len(text) <= MAX_PHRASE_LENGTH:
        return False
    if any(marker in text for marker in JUNK_MARKERS):
        return False
    tokens = text.split()
    return bool(tokens) and all(looks_like_command(token) for token in tokens)


def split_packed(run: str) -> list[tuple[int, str]]:
    """Nem toda frase do blob termina em byte nulo: há trechos com comandos colados
    ("Sit downMicrophone checkLook around"), e o filtro derrubava o trecho inteiro —
    comandos reais sumiam em silêncio.

    A fronteira minúscula→Maiúscula sem espaço não ocorre dentro de um comando legítimo
    ("Go back", "Zoom In", "Hi Rio" sempre têm espaço), mas ocorre no ruído fonético do
    L&H ("imDT", "ZlQ"). Por isso o corte só vale com prova de que o trecho é de
    comandos: todos os pedaços passam no filtro e pelo menos um tem mais de uma
    palavra — coisa que o ruído nunca tem.
    """
    whole = _trimmed(run, 0)
    if len(run) < MIN_PACKED_RUN:
        return whole
    pieces = _boundary_pieces(run)
    if len(pieces) < 2 or not all(looks_like_phrase(text) for _, text in pieces):
        return whole
    return pieces if any(" " in text for _, text in pieces) else whole


def _boundary_pieces(run: str) -> list[tuple[int, str]]:
    pieces: list[tuple[int, str]] = []
    start = 0
    for boundary in [match.start() for match in PACKED_BOUNDARY.finditer(run)] + [len(run)]:
        pieces.extend(_trimmed(run[start:boundary], start))
        start = boundary
    return pieces


def _trimmed(raw: str, offset: int) -> list[tuple[int, str]]:
    text = raw.strip()
    return [(offset + raw.index(text[0]), text)] if text else []


def extract_vocabulary(blob: bytes, base_address: int = 0, gap: int = DEFAULT_GAP) -> GrammarExtraction:
    """As frases ficam num bloco contíguo no início do blob; o primeiro salto maior que
    ``gap`` marca o fim das palavras — o resto é dado fonético."""
    entries: list[str] = []
    rejected: list[str] = []
    seen: set[str] = set()
    last_offset: int | None = None
    end_offset = 0
    for match in RUN_PATTERN.finditer(blob):
        run = match.group().decode("ascii", errors="replace")
        offset = match.start()
        stop = False
        for position, piece in split_packed(run):
            if not looks_like_phrase(piece):
                rejected.append(piece)
                continue
            start = offset + position
            if last_offset is not None and start - last_offset > gap:
                stop = True
                break
            last_offset = start
            end_offset = start + len(piece)
            if piece not in seen:
                seen.add(piece)
                entries.append(piece)
        if stop:
            break
    return GrammarExtraction(tuple(entries), tuple(rejected), end_offset, base_address)
