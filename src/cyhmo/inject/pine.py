"""Cliente PINE do mod.

Framing e opcodes validados contra o PCSX2 v2.6.3; aqui entram escrita, lotes de
leitura/escrita e reconexão com backoff.
"""

from __future__ import annotations

import logging
import socket
import struct
import threading
import time
from typing import Callable, Iterator, Sequence, TypeVar

from cyhmo.config.schema import PineConfig
from cyhmo.domain.errors import CyhmoError

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 28011

MSG_READ8 = 0x00
MSG_READ16 = 0x01
MSG_READ32 = 0x02
MSG_READ64 = 0x03
MSG_WRITE8 = 0x04
MSG_WRITE16 = 0x05
MSG_WRITE32 = 0x06
MSG_WRITE64 = 0x07
MSG_VERSION = 0x08
MSG_SAVESTATE = 0x09
MSG_LOADSTATE = 0x0A
MSG_TITLE = 0x0B
MSG_ID = 0x0C
MSG_UUID = 0x0D
MSG_GAMEVERSION = 0x0E
MSG_STATUS = 0x0F

STATUS_RUNNING = 0
STATUS_NAMES = {0: "rodando", 1: "pausado", 2: "desligado (sem jogo)"}

READ_SIZE_BY_OPCODE = {MSG_READ8: 1, MSG_READ16: 2, MSG_READ32: 4, MSG_READ64: 8}
READ_OPCODE_BY_SIZE = {8: MSG_READ64, 4: MSG_READ32, 2: MSG_READ16, 1: MSG_READ8}
WRITE_OPCODE_BY_SIZE = {1: MSG_WRITE8, 2: MSG_WRITE16, 4: MSG_WRITE32, 8: MSG_WRITE64}
VALID_WIDTHS = (8, 16, 32, 64)

# Tetos do servidor PINE do PCSX2 (MAX_IPC_SIZE e MAX_IPC_RETURN_SIZE em pcsx2/PINE.cpp).
# O contador de resposta começa nos 5 bytes de cabeçalho e cada comando é conferido ANTES de
# escrever: estourar o teto não devolve dados parciais, devolve 0xFF para o pacote inteiro.
PINE_REQUEST_LIMIT_BYTES = 650_000
PINE_REPLY_LIMIT_BYTES = 450_000
REQUEST_HEADER_BYTES = 4
REPLY_HEADER_BYTES = 5
OPCODE_BYTES = 1
ADDRESS_BYTES = 4

BATCH_MAX_BYTES = 256 * 1024
RESPONSE_MAX_BYTES = 64 * 1024 * 1024
BACKOFF_INITIAL_S = 0.5
BACKOFF_MAX_S = 8.0
BACKOFF_REFUSED_MAX_S = 1.0
BATCH_REFUSALS_BEFORE_FALLBACK = 3

CONNECT_STAGE = "ao pedido de conexão"
REQUEST_STAGE = "ao pedido na conexão já aberta"

MemoryRead = tuple[int, int]
MemoryWrite = tuple[int, int, int]
T = TypeVar("T")

log = logging.getLogger("cyhmo.inject.pine")


class PineError(CyhmoError, RuntimeError):
    """Erro de protocolo ou resposta de falha do PINE."""


class PineConnectError(PineError):
    """Socket PINE indisponível: recusado, caído ou em espera de reconexão."""


def parse_string(data: bytes) -> str:
    if len(data) < 4:
        raise PineError(f"resposta de string curta demais ({len(data)} bytes).")
    (length,) = struct.unpack("<I", data[:4])
    return data[4 : 4 + length].rstrip(b"\x00").decode("utf-8", errors="replace")


def width_to_bytes(width_bits: int) -> int:
    if width_bits not in VALID_WIDTHS:
        raise ValueError(f"largura inválida: {width_bits} bits (esperava 8, 16, 32 ou 64)")
    return width_bits // 8


def encode_read(addr: int, width_bits: int) -> bytes:
    return struct.pack("<BI", READ_OPCODE_BY_SIZE[width_to_bytes(width_bits)], addr)


def encode_write(addr: int, width_bits: int, value: int) -> bytes:
    size = width_to_bytes(width_bits)
    if not 0 <= value < (1 << width_bits):
        raise ValueError(f"valor 0x{value:X} não cabe em {width_bits} bits")
    return struct.pack("<BI", WRITE_OPCODE_BY_SIZE[size], addr) + value.to_bytes(size, "little")


def decompose_range(addr: int, nbytes: int) -> Iterator[MemoryRead]:
    """Cobre a faixa com leituras de 64 bits e uma cauda menor."""
    cursor = addr
    remaining = nbytes
    for size in (8, 4, 2, 1):
        while remaining >= size:
            yield cursor, size * 8
            cursor += size
            remaining -= size


def split_into_packets(items: Sequence[T], cost: Callable[[T], tuple[int, int]]) -> Iterator[list[T]]:
    """Fatia a sequência em pacotes que cabem nos dois buffers do PINE ao mesmo tempo.

    Pacote grande demais não devolve dados parciais: o emulador recusa o pacote inteiro com
    0xFF, que o cliente lê como falta de suporte a lote e responde caindo para uma requisição
    por operação — 65 mil round-trips para meio megabyte. Respeitar o teto aqui é o que impede
    esse regime; ``cost`` devolve quantos bytes o item gasta no pedido e na resposta.

    Item que sozinho estoura vai sozinho: recusar aqui trocaria o erro do emulador por um
    nosso e esconderia qual endereço falhou.
    """
    request_budget = PINE_REQUEST_LIMIT_BYTES - REQUEST_HEADER_BYTES
    reply_budget = PINE_REPLY_LIMIT_BYTES - REPLY_HEADER_BYTES
    group: list[T] = []
    request = reply = 0
    for item in items:
        asked, answered = cost(item)
        if group and (request + asked > request_budget or reply + answered > reply_budget):
            yield group
            group, request, reply = [], 0, 0
        group.append(item)
        request += asked
        reply += answered
    if group:
        yield group


def read_cost(read: MemoryRead) -> tuple[int, int]:
    return OPCODE_BYTES + ADDRESS_BYTES, width_to_bytes(read[1])


def write_cost(write: MemoryWrite) -> tuple[int, int]:
    return OPCODE_BYTES + ADDRESS_BYTES + width_to_bytes(write[1]), 0


class PineClient:
    """Cliente TCP do PINE: leituras/escritas avulsas, lotes e faixas, com reconexão."""

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        timeout: float = 5.0,
        connect_timeout: float = 3.0,
        auto_reconnect: bool = True,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self.connect_timeout = connect_timeout
        self.auto_reconnect = auto_reconnect
        self._sock: socket.socket | None = None
        self._batch_supported: bool | None = None
        self._batch_refusals = 0
        self._connection_count = 0
        self._retry_delay_s = BACKOFF_INITIAL_S
        self._retry_not_before = 0.0
        self._lock = threading.RLock()
        self._connect_lock = threading.RLock()
        self.on_link_change: Callable[[bool, str], None] | None = None
        self._link_reported: bool | None = None

    @classmethod
    def from_config(cls, config: PineConfig) -> "PineClient":
        return cls(
            host=config.host,
            port=config.port,
            timeout=config.timeout_ms / 1000.0,
            connect_timeout=config.connect_timeout_ms / 1000.0,
            auto_reconnect=config.auto_reconnect,
        )

    @property
    def is_connected(self) -> bool:
        return self._sock is not None

    @property
    def connection_count(self) -> int:
        return self._connection_count

    @property
    def batch_supported(self) -> bool | None:
        return self._batch_supported

    def connect(self) -> None:
        """``connect_timeout`` tem de ser maior que o RST do SO — medido em ~2,04 s no loopback
        desta máquina. Abaixo disso a porta fechada nunca vira ``ConnectionRefusedError``: ela
        chega como timeout e o PCSX2 fechado é acusado de travado, com a dica de reiniciar um
        emulador que nem está aberto.

        O aperto de mão fica **fora** do lock das transações, sob um lock só dele. Com o
        emulador fechado — estado normal desde que o mod passou a esperar o PCSX2 aparecer —
        cada tentativa segura a linha pelos ~2 s do RST, e a camada 3 tenta a cada segundo:
        dentro do lock das transações isso bloquearia a injeção por quase todo esse tempo.
        """
        with self._connect_lock:
            self.close()
            try:
                sock = socket.create_connection((self.host, self.port), timeout=self.connect_timeout)
            except TimeoutError as exc:
                failure = self._unresponsive_error(CONNECT_STAGE, self.connect_timeout, exc)
                self._fail_connection(BACKOFF_MAX_S, str(failure))
                raise failure from exc
            except OSError as exc:
                self._fail_connection(BACKOFF_REFUSED_MAX_S, f"PCSX2 não encontrado em {self.host}:{self.port}")
                raise PineConnectError(
                    f"não conectou em {self.host}:{self.port} ({exc}). "
                    "O PCSX2 está aberto, com PINE habilitado nesse slot?"
                ) from exc
            sock.settimeout(self.timeout)
            with self._lock:
                self._sock = sock
                self._connection_count += 1

    def _fail_connection(self, max_delay: float, detail: str) -> None:
        with self._lock:
            self._schedule_retry(max_delay)
        self._report_link(False, detail)

    def close(self) -> None:
        with self._lock:
            if self._sock is None:
                return
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def reconnect(self) -> None:
        """Passa pelo mesmo portão do ``ensure_connected``: chamar ``connect`` direto devolvia
        ao emulador o socket que o backoff tinha acabado de negar, três linhas antes.
        """
        self.close()
        self.ensure_connected()

    def ensure_connected(self) -> None:
        """Reabre a conexão se caiu, respeitando o backoff para não martelar um emulador fechado.

        A decisão é tomada sob o lock das transações, mas o aperto de mão acontece sob o lock
        de conexão: as camadas 3 e 4 continuam abrindo um socket só — o perdedor da corrida
        encontra a conexão pronta e desiste — sem que a camada 4 espere o RST do emulador
        fechado para conseguir falar.
        """
        with self._lock:
            if self._sock is not None:
                return
            if not self.auto_reconnect:
                raise PineConnectError("não conectado — chame connect() (auto_reconnect desligado).")
            waiting = self._retry_not_before - time.monotonic()
            if waiting > 0:
                raise PineConnectError(
                    f"PCSX2 não encontrado ({self.host}:{self.port}); "
                    f"nova tentativa de conexão em {waiting:.1f} s."
                )
        with self._connect_lock:
            with self._lock:
                if self._sock is not None:
                    return
            self.connect()

    def __enter__(self) -> "PineClient":
        self.ensure_connected()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _schedule_retry(self, max_delay: float = BACKOFF_MAX_S) -> None:
        """O teto muda com a causa porque o custo muda: a conexão recusada é resposta imediata
        do SO e não prende socket nenhum no emulador, então esperar os 8 s da mudez por ela só
        cegaria o jogador durante o reinício do PCSX2 que a mudez acabou de pedir.
        """
        delay = min(self._retry_delay_s, max_delay)
        self._retry_not_before = time.monotonic() + delay
        self._retry_delay_s = min(delay * 2, max_delay)

    def _clear_retry(self) -> None:
        """Só uma resposta bem formada prova que a outra ponta está viva.

        Zerar o backoff em ``connect`` desarmava a escalada justamente no caso pior: quando o
        PCSX2 aceita o socket e não responde, todo TCP conecta e a espera nunca crescia.

        Pelo mesmo motivo o elo só é anunciado como vivo aqui: anunciá-lo no ``connect`` fazia
        a interface piscar verde↔vermelho contra um emulador que aceita a conexão e emudece —
        cada ciclo é uma transição de verdade, e a pílula alternava a cada ~2 s.
        """
        self._retry_delay_s = BACKOFF_INITIAL_S
        self._retry_not_before = 0.0
        self._report_link(True, f"{self.host}:{self.port}")

    def _drop_connection(self, reason: str) -> None:
        """Sem agendar a espera aqui, a camada 3 abre um socket novo a cada tique e o PCSX2
        acumula conexões em CLOSE_WAIT até parar de responder de vez."""
        self.close()
        self._schedule_retry()
        self._report_link(False, reason)

    def _report_link(self, connected: bool, detail: str) -> None:
        """Só a transição é anunciada: com o emulador fechado o cliente tenta reconectar a cada
        segundo, e avisar em toda tentativa afogaria o console do jogador."""
        if self._link_reported == connected or self.on_link_change is None:
            return
        self._link_reported = connected
        self.on_link_change(connected, detail)

    def _unresponsive_error(self, stage: str, timeout: float, exc: BaseException) -> PineConnectError:
        """Uma mensagem só para os dois estágios mudos: o SYN sem resposta e o pedido sem
        resposta são o mesmo entupimento em momentos diferentes, e o remédio é o mesmo.
        """
        return PineConnectError(
            f"o PCSX2 não respondeu {stage} em {self.host}:{self.port} dentro de "
            f"{timeout:.1f} s ({exc}). O servidor PINE dele costuma travar assim depois de "
            "acumular conexões não recicladas — reinicie o PCSX2."
        )

    def _recv_exact(self, sock: socket.socket, n: int) -> bytes:
        chunks = bytearray()
        while len(chunks) < n:
            chunk = sock.recv(n - len(chunks))
            if not chunk:
                raise ConnectionError("PCSX2 fechou a conexão no meio da resposta.")
            chunks += chunk
        return bytes(chunks)

    def request(self, payload: bytes) -> bytes:
        """Envia um pacote (um ou vários pares opcode+args) e devolve os dados da resposta.

        Serializado por lock: as camadas 3 e 4 compartilham a mesma instância, e dois
        ``sendall``/``recv`` intercalados no mesmo socket perdem o framing da resposta.

        Toda falha de transporte fecha o socket e agenda o backoff: reusar a conexão depois de
        um timeout leria a resposta atrasada como cabeçalho da próxima, e o PINE não tem
        identificador de requisição para ressincronizar.
        """
        with self._lock:
            self.ensure_connected()
            sock = self._sock
            if sock is None:
                raise PineConnectError("não conectado.")
            try:
                sock.sendall(struct.pack("<I", 4 + len(payload)) + payload)
                (size,) = struct.unpack("<I", self._recv_exact(sock, 4))
                if size < 5 or size > RESPONSE_MAX_BYTES:
                    raise PineError(f"tamanho de resposta sem sentido ({size}); framing perdido.")
                body = self._recv_exact(sock, size - 4)
            except PineError as exc:
                self._drop_connection(str(exc))
                raise
            except TimeoutError as exc:
                failure = self._unresponsive_error(REQUEST_STAGE, self.timeout, exc)
                self._drop_connection(str(failure))
                raise failure from exc
            except OSError as exc:
                self._drop_connection(f"conexão PINE perdida ({exc})")
                raise PineConnectError(f"conexão PINE perdida ({exc}); o PCSX2 fechou?") from exc
            self._clear_retry()
        result, data = body[0], body[1:]
        if result != 0x00:
            raise PineError(
                f"PCSX2 respondeu 'falha' (0x{result:02X}). Se foi comando de memória/jogo, "
                "provavelmente nenhum jogo está rodando ou o endereço é inválido."
            )
        return data

    def _command(self, opcode: int, args: bytes = b"") -> bytes:
        return self.request(struct.pack("<B", opcode) + args)

    def version(self) -> str:
        return parse_string(self._command(MSG_VERSION))

    def title(self) -> str:
        return parse_string(self._command(MSG_TITLE))

    def serial(self) -> str:
        return parse_string(self._command(MSG_ID))

    def status(self) -> int:
        (value,) = struct.unpack("<I", self._command(MSG_STATUS))
        return value

    def status_name(self) -> str:
        value = self.status()
        return STATUS_NAMES.get(value, f"desconhecido ({value})")

    def _read_value(self, addr: int, width_bits: int) -> int:
        size = width_to_bytes(width_bits)
        data = self.request(encode_read(addr, width_bits))
        if len(data) != size:
            raise PineError(f"leitura de {size} byte(s) em 0x{addr:08X} devolveu {len(data)} byte(s).")
        return int.from_bytes(data, "little")

    def read8(self, addr: int) -> int:
        return self._read_value(addr, 8)

    def read16(self, addr: int) -> int:
        return self._read_value(addr, 16)

    def read32(self, addr: int) -> int:
        return self._read_value(addr, 32)

    def read64(self, addr: int) -> int:
        return self._read_value(addr, 64)

    def write8(self, addr: int, value: int) -> None:
        self.request(encode_write(addr, 8, value))

    def write16(self, addr: int, value: int) -> None:
        self.request(encode_write(addr, 16, value))

    def write32(self, addr: int, value: int) -> None:
        self.request(encode_write(addr, 32, value))

    def write64(self, addr: int, value: int) -> None:
        self.request(encode_write(addr, 64, value))

    def _run_batch(self, packet: bytes, sequential: Callable[[], bytes]) -> bytes:
        """Tenta o pacote em lote; se o emulador o recusar, cai para comando a comando.

        Se o modo sequencial também falhar, a recusa era de uma operação real (não do
        lote), e o suporte a lote continua desconhecido. Canal caído não é recusa de lote:
        cair para o modo sequencial ali reconectaria por fora do backoff, mantendo a
        tempestade de sockets que ele existe para conter.
        """
        with self._lock:
            if self._batch_supported is False:
                return sequential()
            try:
                data = self.request(packet)
            except PineConnectError:
                raise
            except PineError as exc:
                if self._batch_supported:
                    raise
                return self._recover_from_batch_refusal(exc, sequential)
            self._batch_supported = True
            self._batch_refusals = 0
            return data

    def _recover_from_batch_refusal(self, exc: PineError, sequential: Callable[[], bytes]) -> bytes:
        """Só conta a recusa que se provou ser do lote — aquela em que o modo sequencial
        funcionou. Erro real de operação (endereço fora da RAM numa varredura) recusa lote e
        sequencial igualmente e não pode desligar o lote de um emulador que o suporta.
        """
        self.reconnect()
        log.warning("lote PINE recusado (%s); tentando comando a comando.", exc)
        data = sequential()
        self._batch_refusals += 1
        if self._batch_refusals >= BATCH_REFUSALS_BEFORE_FALLBACK:
            self._batch_supported = False
            log.warning("lote PINE desligado após %d recusas seguidas; seguindo comando a comando.", self._batch_refusals)
        else:
            log.info("recusa de lote %d de %d antes de desligar o lote.", self._batch_refusals, BATCH_REFUSALS_BEFORE_FALLBACK)
        return data

    def _read_sequential(self, reads: Sequence[MemoryRead]) -> bytes:
        chunks = bytearray()
        for addr, width_bits in reads:
            chunks += self._read_value(addr, width_bits).to_bytes(width_to_bytes(width_bits), "little")
        return bytes(chunks)

    def _read_batch(self, reads: Sequence[MemoryRead]) -> bytes:
        return b"".join(self._read_packet(group) for group in split_into_packets(reads, read_cost))

    def _read_packet(self, reads: Sequence[MemoryRead]) -> bytes:
        packet = b"".join(encode_read(addr, width_bits) for addr, width_bits in reads)
        expected = sum(width_to_bytes(width_bits) for _, width_bits in reads)
        data = self._run_batch(packet, lambda: self._read_sequential(reads))
        if len(data) != expected:
            raise PineError(f"lote pediu {expected} bytes mas recebeu {len(data)}.")
        return data

    def read_many(self, reads: Sequence[MemoryRead]) -> list[int]:
        """Lê vários valores ``(addr, largura_em_bits)`` em um único pacote."""
        if not reads:
            return []
        data = self._read_batch(reads)
        values: list[int] = []
        offset = 0
        for _, width_bits in reads:
            size = width_to_bytes(width_bits)
            values.append(int.from_bytes(data[offset : offset + size], "little"))
            offset += size
        return values

    def _write_sequential(self, writes: Sequence[MemoryWrite]) -> bytes:
        for addr, width_bits, value in writes:
            self.request(encode_write(addr, width_bits, value))
        return b""

    def write_many(self, writes: Sequence[MemoryWrite]) -> None:
        """Escreve vários ``(addr, largura_em_bits, valor)`` na ordem dada, no menor número de
        pacotes que cabe no buffer de pedido do PINE — uma injeção inteira ainda vai em um só."""
        for group in split_into_packets(writes, write_cost):
            self._write_packet(group)

    def _write_packet(self, writes: Sequence[MemoryWrite]) -> None:
        packet = b"".join(encode_write(addr, width_bits, value) for addr, width_bits, value in writes)
        self._run_batch(packet, lambda: self._write_sequential(writes))

    def read_cstring(self, addr: int, limit: int = 64) -> str:
        raw = self.read_range(addr, limit)
        return raw.split(b"\x00", 1)[0].decode("ascii", errors="replace")

    def _read_block(self, addr: int, nbytes: int) -> bytes:
        return self._read_batch(list(decompose_range(addr, nbytes)))

    def iter_range(
        self,
        start: int,
        size: int,
        block: int = 65536,
        progress: Callable[[int, int], None] | None = None,
    ) -> Iterator[bytes]:
        if size < 0:
            raise ValueError(f"size negativo ({size}).")
        if block <= 0:
            raise ValueError(f"block deve ser positivo (recebi {block}).")
        cursor = start
        end = start + size
        done = 0
        while cursor < end:
            n = min(block, end - cursor, BATCH_MAX_BYTES)
            data = self._read_block(cursor, n)
            cursor += n
            done += n
            if progress is not None:
                progress(done, size)
            yield data

    def read_range(self, start: int, size: int, block: int = 65536) -> bytes:
        return b"".join(self.iter_range(start, size, block))
