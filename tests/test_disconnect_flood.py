"""Hidden tests: the pty-mode kernel survives its client going away.

When the extension side of the RFC-008 socket closes while the kernel is
still emitting, the kernel must drop what it can no longer deliver, keep
its stderr quiet, shut down on its own with the documented exit code, and
still produce the final ``notebook.metadata`` snapshot. A client that stays
connected keeps receiving every forwarded log record and the snapshot
exactly as before -- the fix must not mute, filter or demote anything.

The in-process tests boot the kernel the way the facade consumers do
(``PtyTransport.prepare`` + ``Kernel(kernel_shell=...)`` on a private loop
thread). The subprocess tests run ``python -m llm_kernel pty-mode`` so the
captured stderr is the real process stderr.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pytest

PY = sys.executable
SESSION_ID = "hidden-disconnect-session"
#: Upper bound for the kernel's whole stderr after a mid-stream disconnect
#: (banner + at most a couple of notices). The flood is thousands of lines.
MAX_STDERR_LINES = 40
#: How many "sendall failed"-style notices are tolerated for one disconnect.
MAX_SENDALL_NOTICES = 2
#: Thread names the tests own; never counted as kernel leaks.
_OWN_THREAD_PREFIXES = ("hidden-receiver", "hidden-pty-loop", "asyncio_", "pydevd")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _env(**extra: str) -> Dict[str, str]:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    # No kernel-owned proxy: declare an external endpoint that is never
    # contacted so no port is opened and no mitmdump/litellm is needed.
    env.setdefault("LLMKERNEL_LITELLM_ENDPOINT_URL", "http://127.0.0.1:1/v1")
    env.pop("LLMKERNEL_PTY_MODE", None)
    env.update(extra)
    return env


class _Receiver:
    """Extension-side end of the RFC-008 socket: accept one kernel, collect frames."""

    def __init__(self) -> None:
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind(("127.0.0.1", 0))
        self.server.listen(1)
        self.server.settimeout(30.0)
        self.port = self.server.getsockname()[1]
        self.frames: List[Dict[str, Any]] = []
        self.eof = threading.Event()
        self.conn: Optional[socket.socket] = None
        self._thread = threading.Thread(target=self._loop, name="hidden-receiver", daemon=True)
        self._thread.start()

    @property
    def address(self) -> str:
        return f"tcp:127.0.0.1:{self.port}"

    def _loop(self) -> None:
        try:
            conn, _ = self.server.accept()
        except OSError:
            self.eof.set()
            return
        self.conn = conn
        conn.settimeout(0.2)
        buf = bytearray()
        while True:
            try:
                chunk = conn.recv(65536)
            except socket.timeout:
                continue
            except OSError:
                break
            if not chunk:
                break
            buf.extend(chunk)
            while True:
                nl = buf.find(b"\n")
                if nl < 0:
                    break
                line = bytes(buf[:nl])
                del buf[: nl + 1]
                if line.strip():
                    try:
                        self.frames.append(json.loads(line.decode("utf-8")))
                    except ValueError:
                        pass
        self.eof.set()

    def wait_for(self, predicate: Callable[[Dict[str, Any]], bool], timeout: float = 30.0) -> Optional[Dict[str, Any]]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for frame in list(self.frames):
                if predicate(frame):
                    return frame
            time.sleep(0.05)
        return None

    def send(self, payload: Dict[str, Any]) -> None:
        assert self.conn is not None
        self.conn.sendall((json.dumps(payload) + "\n").encode("utf-8"))

    def close(self) -> None:
        """Drop the client: close the accepted connection and the listener."""
        for s in (self.conn, self.server):
            try:
                if s is not None:
                    s.close()
            except OSError:
                pass
        self._thread.join(timeout=5.0)


def _body(frame: Dict[str, Any]) -> str:
    return str((frame.get("body") or {}).get("stringValue", ""))


def _event_name(frame: Dict[str, Any]) -> Optional[str]:
    for attr in frame.get("attributes") or []:
        if attr.get("key") == "event.name":
            return (attr.get("value") or {}).get("stringValue")
    return None


def _attr(frame: Dict[str, Any], key: str) -> Optional[str]:
    for attr in frame.get("attributes") or []:
        if attr.get("key") == key:
            return (attr.get("value") or {}).get("stringValue")
    return None


def _is_ready(frame: Dict[str, Any]) -> bool:
    return _event_name(frame) == "kernel.ready"


def _is_dispatcher_started(frame: Dict[str, Any]) -> bool:
    return "dispatcher started" in _body(frame)


def _is_shutdown_snapshot(frame: Dict[str, Any]) -> bool:
    payload = frame.get("payload")
    return (
        frame.get("type") == "notebook.metadata"
        and isinstance(payload, dict)
        and payload.get("trigger") == "shutdown"
    )


def _stderr_stats(text: str) -> Dict[str, int]:
    return {
        "lines": len(text.splitlines()),
        "logging_error_blocks": text.count("--- Logging error ---"),
        "sendall_notices": text.count("sendall failed"),
        "tracebacks": text.count("Traceback (most recent call last)"),
    }


def _assert_quiet_stderr(text: str) -> None:
    stats = _stderr_stats(text)
    excerpt = text[:1500]
    assert "Logging error" not in text, f"stderr carries logging-error blocks: {stats}\n{excerpt}"
    assert stats["tracebacks"] == 0, f"stderr carries tracebacks: {stats}\n{excerpt}"
    assert stats["lines"] <= MAX_STDERR_LINES, (
        f"stderr is {stats['lines']} lines after the client went away (limit {MAX_STDERR_LINES}): {stats}\n{excerpt}"
    )
    assert stats["sendall_notices"] <= MAX_SENDALL_NOTICES, (
        f"{stats['sendall_notices']} send-failure notices for one disconnect (limit {MAX_SENDALL_NOTICES})\n{excerpt}"
    )


def _kernel_threads() -> List[str]:
    return sorted(
        t.name for t in threading.enumerate()
        if not t.name.startswith(_OWN_THREAD_PREFIXES)
    )


def _wait_threads_back_to(baseline: List[str], timeout: float = 10.0) -> List[str]:
    """Return the kernel threads still alive beyond ``baseline`` after ``timeout``."""
    deadline = time.monotonic() + timeout
    while True:
        leaked = [n for n in _kernel_threads() if n not in baseline]
        if not leaked or time.monotonic() > deadline:
            return leaked
        time.sleep(0.1)


# --------------------------------------------------------------------------
# subprocess boot
# --------------------------------------------------------------------------
def _spawn_pty(tmp_path: Path, receiver: _Receiver) -> "tuple[subprocess.Popen, Path]":
    env = _env(LLMKERNEL_IPC_SOCKET=receiver.address, LLMKERNEL_SESSION_ID=SESSION_ID)
    stderr_path = tmp_path / "kernel.stderr"
    with stderr_path.open("wb") as stderr_fh:
        proc = subprocess.Popen(
            [PY, "-m", "llm_kernel", "pty-mode"], cwd=str(tmp_path), env=env,
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=stderr_fh,
        )
    return proc, stderr_path


def _wait_booted(receiver: _Receiver) -> None:
    assert receiver.wait_for(_is_ready, timeout=60.0) is not None, "no kernel.ready frame"
    assert receiver.wait_for(_is_dispatcher_started, timeout=30.0) is not None, (
        "the dispatcher never reported started on the data plane"
    )


def _disconnect_and_wait(proc: subprocess.Popen, receiver: _Receiver, timeout: float = 60.0) -> "tuple[int, float]":
    """Close the client mid-stream; return (exit code, seconds until exit)."""
    # Let a few more records flow so the close lands while the kernel emits.
    time.sleep(0.5)
    t0 = time.monotonic()
    receiver.close()
    try:
        rc = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
        raise AssertionError(f"kernel did not exit within {timeout}s after the client went away")
    return rc, time.monotonic() - t0


def test_subprocess_disconnect_keeps_stderr_quiet(tmp_path: Path) -> None:
    """Peer closes mid-stream: the kernel's stderr stays a handful of lines
    with no logging-error blocks and at most a couple of send-failure notices."""
    receiver = _Receiver()
    proc, stderr_path = _spawn_pty(tmp_path, receiver)
    try:
        _wait_booted(receiver)
        _disconnect_and_wait(proc, receiver)
    finally:
        receiver.close()
        if proc.poll() is None:
            proc.kill()
    _assert_quiet_stderr(stderr_path.read_text(errors="replace"))


def test_subprocess_exits_on_its_own_after_peer_eof(tmp_path: Path) -> None:
    """Peer EOF is the RFC-008 step-6 shutdown path: the process exits by
    itself, promptly, with exit code 0 (the code the path gives today)."""
    receiver = _Receiver()
    proc, stderr_path = _spawn_pty(tmp_path, receiver)
    try:
        _wait_booted(receiver)
        rc, elapsed = _disconnect_and_wait(proc, receiver)
    finally:
        receiver.close()
        if proc.poll() is None:
            proc.kill()
    assert rc == 0, f"exit code {rc}; stderr tail:\n{stderr_path.read_text(errors='replace')[-1500:]}"
    assert elapsed < 30.0, f"kernel took {elapsed:.1f}s to exit after peer EOF"


# --------------------------------------------------------------------------
# in-process boot (facade consumer style)
# --------------------------------------------------------------------------
@pytest.fixture
def pty_kernel(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Boot one pty-mode kernel in-process against a loopback receiver.

    Yields a dict with the facade ``kernel``, the ``transport``, the
    ``receiver``, the ``loop`` the kernel runs on, the pre-boot kernel
    thread names and a ``shutdown()`` callable that runs ``Kernel.shutdown``
    on the loop and returns how long it took (seconds).
    """
    from llm_kernel.kernel_facade import Kernel
    from llm_kernel.transports import pty_mode

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LLMKERNEL_LITELLM_ENDPOINT_URL", "http://127.0.0.1:1/v1")
    monkeypatch.delenv(pty_mode.ENV_PTY_MODE, raising=False)
    saved_signals = {
        sig: signal.getsignal(sig)
        for sig in (signal.SIGINT, signal.SIGTERM)
        if hasattr(signal, sig.name)
    }
    root_handlers = list(logging.getLogger().handlers)
    threads_before = _kernel_threads()
    receiver = _Receiver()
    transport = pty_mode.PtyTransport(address=receiver.address, session_id=SESSION_ID, pty_flag=False)
    kernel = None
    loop: Optional[asyncio.AbstractEventLoop] = None
    loop_thread: Optional[threading.Thread] = None
    state: Dict[str, Any] = {}

    def _shutdown(timeout: float = 30.0) -> float:
        assert kernel is not None and loop is not None
        t0 = time.monotonic()
        asyncio.run_coroutine_threadsafe(kernel.shutdown(), loop).result(timeout=timeout)
        return time.monotonic() - t0

    try:
        rc = transport.prepare()
        assert rc == 0, f"pty transport failed to prepare: rc={rc!r}"
        loop = asyncio.new_event_loop()
        loop_thread = threading.Thread(target=loop.run_forever, name="hidden-pty-loop", daemon=True)
        loop_thread.start()
        kernel = Kernel(workdir=tmp_path, kernel_shell=transport.kernel_shell, loop=loop)

        async def _start() -> None:
            await kernel.start()
            await kernel.bind_transport(transport)

        asyncio.run_coroutine_threadsafe(_start(), loop).result(timeout=60.0)
        assert receiver.wait_for(_is_ready, timeout=30.0) is not None, "no ready handshake on the data plane"
        state.update(
            kernel=kernel, transport=transport, receiver=receiver, loop=loop,
            metadata_writer=kernel.metadata_writer, threads_before=threads_before,
            shutdown=_shutdown,
        )
        yield state
    finally:
        if kernel is not None and loop is not None:
            try:
                asyncio.run_coroutine_threadsafe(kernel.shutdown(), loop).result(timeout=30.0)
            except Exception:
                pass
            loop.call_soon_threadsafe(loop.stop)
            if loop_thread is not None:
                loop_thread.join(timeout=5.0)
            if not loop.is_running():
                loop.close()
        transport.close()
        receiver.close()
        for sig, prev in saved_signals.items():
            try:
                signal.signal(sig, prev)
            except (ValueError, OSError):
                pass
        root = logging.getLogger()
        for h in list(root.handlers):
            if h not in root_handlers:
                root.removeHandler(h)


def _wait_read_loop_ended(transport: Any, timeout: float = 15.0) -> None:
    thread = getattr(transport, "_read_thread", None)
    if thread is not None:
        thread.join(timeout=timeout)
        assert not thread.is_alive(), "the socket read loop did not notice the peer going away"


def _emit_after_disconnect() -> None:
    """A few records the kernel can no longer deliver (what the heartbeat,
    the dispatcher and the shutdown path produce in the field)."""
    log = logging.getLogger("llm_kernel.transports.pty_mode")
    for i in range(3):
        log.info("post-disconnect record %d", i)
        time.sleep(0.05)


def test_inprocess_disconnect_keeps_stderr_quiet(pty_kernel: Dict[str, Any], capfd: pytest.CaptureFixture[str]) -> None:
    """Same disconnect through the facade boot: nothing but the banner and at
    most a couple of notices reaches stderr while the kernel keeps emitting
    and then shuts down."""
    receiver: _Receiver = pty_kernel["receiver"]
    receiver.close()
    _wait_read_loop_ended(pty_kernel["transport"])
    _emit_after_disconnect()
    pty_kernel["shutdown"]()
    _emit_after_disconnect()
    _assert_quiet_stderr(capfd.readouterr().err)


def test_inprocess_shutdown_completes_and_leaves_no_threads(pty_kernel: Dict[str, Any]) -> None:
    """After the peer goes, ``Kernel.shutdown`` returns promptly and no
    dispatcher / heartbeat / autosave / read-loop thread survives it."""
    receiver: _Receiver = pty_kernel["receiver"]
    receiver.close()
    _wait_read_loop_ended(pty_kernel["transport"])
    _emit_after_disconnect()
    elapsed = pty_kernel["shutdown"](timeout=30.0)
    assert elapsed < 15.0, f"Kernel.shutdown took {elapsed:.1f}s after the peer went away"
    pty_kernel["transport"].close()
    leaked = _wait_threads_back_to(pty_kernel["threads_before"])
    assert leaked == [], f"kernel threads still alive after shutdown: {leaked}"


def test_inprocess_final_snapshot_still_taken_once_after_disconnect(pty_kernel: Dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """The peer-EOF shutdown path still runs end to end: exactly one
    ``notebook.metadata`` snapshot with trigger ``shutdown`` is built."""
    writer = pty_kernel["metadata_writer"]
    calls: List[str] = []
    real_snapshot = writer.snapshot

    def _counting_snapshot(trigger: str = "save", *args: Any, **kwargs: Any) -> Any:
        calls.append(trigger)
        return real_snapshot(trigger, *args, **kwargs)

    monkeypatch.setattr(writer, "snapshot", _counting_snapshot)
    receiver: _Receiver = pty_kernel["receiver"]
    receiver.close()
    _wait_read_loop_ended(pty_kernel["transport"])
    _emit_after_disconnect()
    pty_kernel["shutdown"]()
    assert calls.count("shutdown") == 1, f"expected exactly one shutdown snapshot, got triggers {calls}"
    last = writer.take_last_envelope()
    assert last is not None, "the metadata writer built no envelope at all"
    kind = last.get("message_type") or last.get("type")
    assert kind == "notebook.metadata", f"last envelope: {last!r}"
    assert (last.get("payload") or {}).get("trigger") == "shutdown", f"last envelope: {last!r}"


# --------------------------------------------------------------------------
# regression guard: a connected client still gets everything
# --------------------------------------------------------------------------
def test_connected_client_still_receives_every_log_record(pty_kernel: Dict[str, Any]) -> None:
    """With the client connected, forwarded log records keep arriving as
    OTLP frames with the right severity, whatever logger or text produced
    them -- including the socket writer's own logger and its failure text.
    The final snapshot is still the last envelope on the wire."""
    receiver: _Receiver = pty_kernel["receiver"]
    token = f"probe-{os.getpid()}-{time.time_ns()}"

    probes = [
        ("llm_kernel.transports.pty_mode", logging.INFO, 9, f"{token} info from the transport"),
        ("llm_kernel.capture.socket_writer", logging.WARNING, 13, f"{token} warning from the writer logger"),
        (
            "llm_kernel.capture.socket_writer", logging.WARNING, 13,
            f"SocketWriter.write_frame: sendall failed ({token}); frame dropped",
        ),
        ("llm_kernel.supervision.agent_supervisor", logging.ERROR, 17, f"{token} error from the supervisor"),
    ]
    for name, level, _sev, text in probes:
        logging.getLogger(name).log(level, text)

    for name, level, sev, text in probes:
        frame = receiver.wait_for(lambda f, text=text: _body(f) == text, timeout=10.0)
        assert frame is not None, f"record {text!r} from logger {name!r} never reached the client"
        assert frame.get("severityNumber") == sev, f"severityNumber {frame.get('severityNumber')!r} != {sev} for {text!r}"
        assert frame.get("severityText") == logging.getLevelName(level), frame
        assert _attr(frame, "logger.name") == name, frame
        assert _attr(frame, "llmnb.kernel.session_id") == SESSION_ID, frame

    # Orderly shutdown while connected: the snapshot still goes out, once, last.
    pty_kernel["shutdown"]()
    assert receiver.wait_for(_is_shutdown_snapshot, timeout=15.0) is not None, "no shutdown snapshot on the wire"
    time.sleep(0.3)
    snapshots = [f for f in receiver.frames if _is_shutdown_snapshot(f)]
    assert len(snapshots) == 1, f"expected exactly one shutdown snapshot, got {len(snapshots)}"
    typed = [f for f in receiver.frames if "type" in f]
    assert typed and _is_shutdown_snapshot(typed[-1]), f"last typed envelope: {typed[-1] if typed else None!r}"
    assert _is_ready(receiver.frames[0]), "kernel.ready must still be the first frame"
