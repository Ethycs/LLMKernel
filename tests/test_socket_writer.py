"""Contract tests for :mod:`llm_kernel.socket_writer` (RFC-008 §2).

Exercises:

* :func:`parse_address` over the four address shapes (UDS unprefixed,
  ``unix:`` prefixed, ``tcp:`` prefixed, ``pipe:`` prefixed which is V1
  ``NotImplementedError``).
* :class:`SocketWriter.write_frame` over a real ``socket.socketpair``
  to verify newline framing and byte-clean JSON.
* Multi-threaded interleaving prevention: 1000 frames from N threads
  reach the receiver in N-frame chunks split cleanly on newlines.
* :meth:`SocketWriter.close` idempotency and post-close drops.
"""

from __future__ import annotations

import json
import socket
import threading
from typing import List

import pytest

from llm_kernel.socket_writer import SocketWriter, parse_address


# ---------------------------------------------------------------------------
# parse_address
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="POSIX-only")
def test_parse_address_unix_unprefixed() -> None:
    family, target = parse_address("/tmp/llmnb-foo.sock")
    assert family == socket.AF_UNIX
    assert target == "/tmp/llmnb-foo.sock"


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="POSIX-only")
def test_parse_address_unix_prefixed() -> None:
    family, target = parse_address("unix:/tmp/llmnb-foo.sock")
    assert family == socket.AF_UNIX
    assert target == "/tmp/llmnb-foo.sock"


def test_parse_address_tcp() -> None:
    family, target = parse_address("tcp:127.0.0.1:54321")
    assert family == socket.AF_INET
    assert target == ("127.0.0.1", 54321)


def test_parse_address_tcp_bad_port() -> None:
    with pytest.raises(ValueError):
        parse_address("tcp:127.0.0.1:notaport")


def test_parse_address_tcp_missing_port() -> None:
    with pytest.raises(ValueError):
        parse_address("tcp:127.0.0.1")


def test_parse_address_pipe_not_implemented_v1() -> None:
    """Windows named-pipe transport is queued for I-T-W; V1 raises."""
    with pytest.raises(NotImplementedError):
        parse_address(r"pipe:\\.\pipe\llmnb-foo")


def test_parse_address_empty_raises() -> None:
    with pytest.raises(ValueError):
        parse_address("")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tcp_pair() -> "tuple[socket.socket, socket.socket]":
    """Return a connected (client, server-accepted) TCP pair on loopback.

    A portable substitute for ``socket.socketpair`` which is POSIX-only
    in this codebase's matrix; the SocketWriter accepts any connected
    stream socket so the underlying transport is irrelevant.
    """
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    accepted: list = []

    def _accept() -> None:
        sock, _addr = listener.accept()
        accepted.append(sock)

    t = threading.Thread(target=_accept, daemon=True)
    t.start()
    client.connect(("127.0.0.1", port))
    t.join(timeout=5)
    listener.close()
    assert accepted, "accept never returned"
    return client, accepted[0]


def _drain_until(sock: socket.socket, expected_lines: int, timeout: float = 10.0) -> List[bytes]:
    """Read newline-delimited lines from ``sock`` until ``expected_lines`` reached."""
    sock.settimeout(timeout)
    buf = bytearray()
    lines: List[bytes] = []
    while len(lines) < expected_lines:
        try:
            chunk = sock.recv(8192)
        except socket.timeout:
            break
        if not chunk:
            break
        buf.extend(chunk)
        while True:
            nl = buf.find(b"\n")
            if nl < 0:
                break
            lines.append(bytes(buf[:nl]))
            del buf[: nl + 1]
    return lines


# ---------------------------------------------------------------------------
# write_frame: shape + framing
# ---------------------------------------------------------------------------


def test_write_frame_appends_newline() -> None:
    client, server = _tcp_pair()
    writer = SocketWriter()
    # Splice the writer onto the existing connected client without
    # ``connect`` so we don't need a real listening server fixture.
    writer._sock = client  # type: ignore[attr-defined]
    writer.write_frame({"type": "smoke", "payload": {"v": 1}})
    writer.close()
    server.settimeout(5.0)
    data = server.recv(8192)
    assert data.endswith(b"\n")
    parsed = json.loads(data.rstrip(b"\n"))
    assert parsed == {"type": "smoke", "payload": {"v": 1}}
    server.close()


def test_write_frame_roundtrips_otlp_shape() -> None:
    """OTLP/JSON-shaped LogRecord round-trips with intValue strings preserved."""
    client, server = _tcp_pair()
    writer = SocketWriter()
    writer._sock = client  # type: ignore[attr-defined]
    record = {
        "timeUnixNano": "1745588938412000000",
        "severityNumber": 9,
        "body": {"stringValue": "kernel.ready"},
        "attributes": [
            {"key": "logger.name", "value": {"stringValue": "llm_kernel.boot"}},
            {"key": "code.lineno", "value": {"intValue": "42"}},
        ],
    }
    writer.write_frame(record)
    writer.close()
    server.settimeout(5.0)
    data = server.recv(8192)
    line = data.rstrip(b"\n")
    parsed = json.loads(line)
    assert parsed == record
    server.close()


def test_write_frame_drops_when_not_connected() -> None:
    """An unconnected writer drops frames silently rather than raising."""
    writer = SocketWriter()
    writer.write_frame({"type": "x", "payload": {}})  # MUST NOT raise
    assert not writer.is_connected()


def test_close_is_idempotent() -> None:
    writer = SocketWriter()
    writer.close()
    writer.close()  # Second call is a no-op.
    assert not writer.is_connected()


def test_write_frame_drops_unencodable_silently() -> None:
    """Un-serializable payloads are dropped, not crashes."""
    client, server = _tcp_pair()
    writer = SocketWriter()
    writer._sock = client  # type: ignore[attr-defined]

    class _Unserializable:
        pass

    # Falls through ``_json_default`` which returns ``repr(...)`` --
    # ``json.dumps`` does NOT raise. Verify the wire still gets a frame.
    writer.write_frame({"value": _Unserializable()})
    writer.close()
    server.settimeout(5.0)
    data = server.recv(8192)
    parsed = json.loads(data.rstrip(b"\n"))
    assert "value" in parsed
    server.close()


# ---------------------------------------------------------------------------
# Multi-threaded interleaving prevention
# ---------------------------------------------------------------------------


def test_concurrent_writes_do_not_interleave() -> None:
    """1000 frames from N threads MUST split cleanly on the receiver side.

    Each frame includes a ``thread`` and ``seq`` attribute; the receiver
    parses every line and asserts the parsed dict carries both fields.
    A single byte-interleaved frame would yield a JSON parse error.
    """
    n_threads = 8
    per_thread = 125  # 8 * 125 = 1000 frames
    client, server = _tcp_pair()
    writer = SocketWriter()
    writer._sock = client  # type: ignore[attr-defined]

    def _producer(thread_id: int) -> None:
        for seq in range(per_thread):
            writer.write_frame({
                "thread": thread_id,
                "seq": seq,
                # Pad the body to make interleaving more likely if the
                # lock weren't held -- TCP send buffers are >= 4 KiB.
                "padding": "x" * 200,
            })

    # Drain on a background thread so producers don't block on a full
    # TCP send buffer (1000 × ~250B exceeds default Windows SO_SNDBUF;
    # without a concurrent reader, sendall stalls and join() hangs).
    drained: List[bytes] = []
    drain_done = threading.Event()

    def _bg_drain() -> None:
        try:
            drained.extend(_drain_until(server, n_threads * per_thread, timeout=20.0))
        finally:
            drain_done.set()

    drain_thread = threading.Thread(target=_bg_drain, daemon=True)
    drain_thread.start()

    producers = [
        threading.Thread(target=_producer, args=(i,), daemon=True)
        for i in range(n_threads)
    ]
    for p in producers:
        p.start()
    for p in producers:
        p.join()
    writer.close()  # half-closes; server sees EOF after last byte.
    drain_thread.join(timeout=25.0)
    server.close()
    lines = drained

    assert len(lines) == n_threads * per_thread, (
        f"expected {n_threads * per_thread} lines, got {len(lines)}"
    )
    parsed_per_thread: dict = {i: [] for i in range(n_threads)}
    for line in lines:
        # If interleaving had occurred, this would raise.
        parsed = json.loads(line)
        parsed_per_thread[parsed["thread"]].append(parsed["seq"])

    for thread_id in range(n_threads):
        seqs = parsed_per_thread[thread_id]
        assert len(seqs) == per_thread
        # In-thread order is preserved (writes inside one thread are
        # naturally serialized).
        assert seqs == sorted(seqs), f"thread {thread_id} out of order: {seqs}"


def test_close_after_concurrent_writes_safe() -> None:
    client, _server = _tcp_pair()
    writer = SocketWriter()
    writer._sock = client  # type: ignore[attr-defined]

    def _producer() -> None:
        for i in range(50):
            writer.write_frame({"i": i})

    threads = [threading.Thread(target=_producer, daemon=True) for _ in range(4)]
    for t in threads:
        t.start()
    writer.close()  # close concurrently with active writers
    for t in threads:
        t.join(timeout=5)
    # If close races with writes, the worst-case is a logged drop --
    # never a crash or a corrupted frame.
