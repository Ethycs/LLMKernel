"""Shared helpers for the conformance suite (no dependency on other tests).

Every helper here drives the kernel through the surfaces the project
documents: the ``Kernel`` facade, the process entry points
(``pty_mode.main`` / ``boot_kernel``, ``serve_mode``, the uvicorn ``app``),
the RFC-006 wire (newline-delimited JSON on a socket) and the public
``llm_kernel.wire.versions`` source.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Dict, Iterable, List, Optional

from llm_kernel.wire import versions as _versions

#: Generous budget for anything that crosses a thread or a socket.
WAIT_SECONDS: float = 15.0


# ---------------------------------------------------------------------------
# Polling
# ---------------------------------------------------------------------------


def wait_until(predicate: Callable[[], Any], timeout: float = WAIT_SECONDS,
               interval: float = 0.02) -> Any:
    """Poll ``predicate`` until it returns a truthy value; return that value.

    Returns the last (falsy) value on timeout so the caller's assert shows it.
    """
    deadline = time.monotonic() + timeout
    value = predicate()
    while not value and time.monotonic() < deadline:
        time.sleep(interval)
        value = predicate()
    return value


def run(coro: Any) -> Any:
    """Run one coroutine on a fresh loop (the suite never relies on an
    asyncio pytest plugin, so it runs the same under any configuration)."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# RFC-005 / RFC-006 hydrate payloads
# ---------------------------------------------------------------------------


def current_kernel_block() -> Dict[str, str]:
    """What a notebook saved by *this* kernel records under
    ``config.volatile.kernel``: exactly what the ready frame announces."""
    block = {key: value for key, value in _versions.RFC_VERSIONS.items()}
    block["kernel_version"] = _versions.KERNEL_VERSION
    return block


def bump(version: str, part: str) -> str:
    major, minor, patch = (int(x) for x in version.split("."))
    if part == "major":
        return f"{major + 1}.0.0"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def hydrate_snapshot(kernel_block: Dict[str, Any], *,
                     config: Optional[Dict[str, Any]] = None,
                     cells: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """A minimal persisted ``metadata.rts`` snapshot (RFC-005)."""
    cfg: Dict[str, Any] = dict(config or {})
    cfg.setdefault("volatile", {})["kernel"] = dict(kernel_block)
    snap: Dict[str, Any] = {
        "schema_version": _versions.SCHEMA_VERSION,
        "config": cfg,
        "drift_log": [],
    }
    if cells is not None:
        snap["cells"] = cells
    return snap


def hydrate_envelope(kernel_block: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
    """RFC-006 Family F ``notebook.metadata mode:"hydrate"`` envelope."""
    return {
        "type": "notebook.metadata",
        "payload": {
            "mode": "hydrate",
            "snapshot": hydrate_snapshot(kernel_block, **kwargs),
            "trigger": "open",
        },
    }


def envelope_type(frame: Dict[str, Any]) -> Optional[str]:
    """RFC-006 v2 thin envelopes carry ``type``; the dispatcher's RFC-003
    envelope (what serve mode forwards verbatim) carries ``message_type``."""
    if not isinstance(frame, dict):
        return None
    return frame.get("type") or frame.get("message_type")


def is_hydrate_complete(frame: Dict[str, Any]) -> bool:
    payload = frame.get("payload") if isinstance(frame, dict) else None
    return (
        isinstance(payload, dict)
        and envelope_type(frame) == "notebook.metadata"
        and payload.get("trigger") == "hydrate_complete"
    )


def hydrate_complete_snapshot(frames: Iterable[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for frame in frames:
        if is_hydrate_complete(frame):
            return frame["payload"]["snapshot"]
    return None


def kernel_drift(snapshot: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """``{rfc_key: drift_event}`` for every ``config.volatile.kernel.*`` entry."""
    out: Dict[str, Dict[str, Any]] = {}
    for event in (snapshot or {}).get("drift_log") or []:
        path = str(event.get("field_path", ""))
        prefix = "config.volatile.kernel."
        if path.startswith(prefix):
            out[path[len(prefix):]] = event
    return out


# ---------------------------------------------------------------------------
# OTLP helpers (RFC-008 ready frame)
# ---------------------------------------------------------------------------


def otlp_attrs(record: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for attr in record.get("attributes") or []:
        value = attr.get("value") or {}
        out[attr.get("key")] = next(iter(value.values()), None) if value else None
    return out


def is_ready_frame(frame: Dict[str, Any]) -> bool:
    body = frame.get("body") if isinstance(frame, dict) else None
    return isinstance(body, dict) and body.get("stringValue") == "kernel.ready"


# ---------------------------------------------------------------------------
# A wire peer: what the extension is to pty-mode (RFC-008 §4)
# ---------------------------------------------------------------------------


class WirePeer:
    """TCP listener the kernel connects to; collects newline-JSON frames."""

    def __init__(self) -> None:
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind(("127.0.0.1", 0))
        self.server.listen(1)
        self.server.settimeout(WAIT_SECONDS * 2)
        self.frames: List[Dict[str, Any]] = []
        self.raw_lines: List[bytes] = []
        self.conn: Optional[socket.socket] = None
        self.connected = threading.Event()
        self.eof = threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, name="conformance-peer", daemon=True)
        self._thread.start()

    @property
    def address(self) -> str:
        return f"tcp:127.0.0.1:{self.server.getsockname()[1]}"

    def _loop(self) -> None:
        try:
            self.conn, _ = self.server.accept()
        except OSError:
            self.eof.set()
            return
        self.connected.set()
        self.conn.settimeout(0.25)
        buf = bytearray()
        while True:
            try:
                chunk = self.conn.recv(65536)
            except socket.timeout:
                continue
            except OSError:
                break
            if not chunk:
                break
            buf.extend(chunk)
            while (nl := buf.find(b"\n")) >= 0:
                line = bytes(buf[:nl])
                del buf[: nl + 1]
                if not line.strip():
                    continue
                with self._lock:
                    self.raw_lines.append(line)
                    try:
                        self.frames.append(json.loads(line.decode("utf-8")))
                    except ValueError:
                        pass
        self.eof.set()

    def snapshot(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self.frames)

    def send(self, frame: Dict[str, Any]) -> None:
        assert self.connected.wait(WAIT_SECONDS), "kernel never connected to the peer"
        assert self.conn is not None
        self.conn.sendall((json.dumps(frame) + "\n").encode("utf-8"))

    def wait_for(self, predicate: Callable[[Dict[str, Any]], bool],
                 timeout: float = WAIT_SECONDS) -> Optional[Dict[str, Any]]:
        def _find() -> Optional[Dict[str, Any]]:
            for frame in self.snapshot():
                if predicate(frame):
                    return frame
            return None
        return wait_until(_find, timeout)

    def close(self) -> None:
        for sock in (self.conn, self.server):
            if sock is None:
                continue
            try:
                sock.close()
            except OSError:
                pass
        self._thread.join(timeout=5.0)


def has_log_body(fragment: str) -> Callable[[Dict[str, Any]], bool]:
    def _pred(frame: Dict[str, Any]) -> bool:
        body = frame.get("body") if isinstance(frame, dict) else None
        return isinstance(body, dict) and fragment in str(body.get("stringValue", ""))
    return _pred


# ---------------------------------------------------------------------------
# A serve-mode client (PLAN-S5.0.3 TCP handshake)
# ---------------------------------------------------------------------------


class LineSocket:
    """Client side of a newline-JSON TCP connection."""

    def __init__(self, host: str, port: int, timeout: float = WAIT_SECONDS) -> None:
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.buf = bytearray()

    def send(self, frame: Dict[str, Any]) -> None:
        self.sock.sendall((json.dumps(frame) + "\n").encode("utf-8"))

    def recv(self, timeout: float = WAIT_SECONDS) -> Optional[Dict[str, Any]]:
        deadline = time.monotonic() + timeout
        while True:
            nl = self.buf.find(b"\n")
            if nl >= 0:
                line = bytes(self.buf[:nl])
                del self.buf[: nl + 1]
                if line.strip():
                    return json.loads(line.decode("utf-8"))
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            self.sock.settimeout(remaining)
            try:
                chunk = self.sock.recv(65536)
            except (socket.timeout, TimeoutError):
                return None
            if not chunk:
                return None
            self.buf.extend(chunk)

    def recv_until(self, predicate: Callable[[Dict[str, Any]], bool],
                   timeout: float = WAIT_SECONDS) -> Optional[Dict[str, Any]]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            frame = self.recv(timeout=max(0.05, deadline - time.monotonic()))
            if frame is None:
                return None
            if predicate(frame):
                return frame
        return None

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


def handshake_request(token: str, *, wire_version: Optional[str] = None) -> Dict[str, Any]:
    return {
        "type": "kernel.handshake",
        "payload": {
            "wire_version": wire_version or _versions.WIRE_VERSION,
            "client_name": "conformance",
            "client_version": _versions.CLIENT_VERSION,
            "capabilities": _versions.kernel_capabilities(),
            "auth": {"scheme": "bearer", "token": token},
        },
    }


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def connect_with_retry(host: str, port: int, timeout: float = 30.0) -> LineSocket:
    deadline = time.monotonic() + timeout
    last: Optional[BaseException] = None
    while time.monotonic() < deadline:
        try:
            return LineSocket(host, port)
        except OSError as exc:
            last = exc
            time.sleep(0.1)
    raise AssertionError(f"could not connect to {host}:{port}: {last!r}")


# ---------------------------------------------------------------------------
# Subprocess entry points (``python -m llm_kernel <subcommand>``)
# ---------------------------------------------------------------------------


def kernel_subprocess_env(**extra: str) -> Dict[str, str]:
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("LLMKERNEL_", "PYTEST_")) and k != "LLMNB_OPERATOR_PIN"}
    # External proxy lifecycle: no kernel-owned proxy in conformance runs.
    env["LLMKERNEL_LITELLM_ENDPOINT_URL"] = "http://127.0.0.1:1/v1"
    env.update(extra)
    return env


def spawn_kernel(args: List[str], env: Dict[str, str], cwd: str) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-m", "llm_kernel", *args],
        env=env, cwd=cwd,
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )


def stop_process(proc: subprocess.Popen, sig: Optional[int] = None,
                 timeout: float = WAIT_SECONDS) -> tuple:
    import signal as _signal

    if proc.poll() is None:
        proc.send_signal(sig if sig is not None else _signal.SIGTERM)
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate(timeout=5.0)
    return proc.returncode, out, err


# ---------------------------------------------------------------------------
# Facade instrumentation
# ---------------------------------------------------------------------------


class FacadeLedger:
    """Counts ``Kernel`` constructions / starts / shutdowns (monkeypatched)."""

    def __init__(self, monkeypatch: Any) -> None:
        from llm_kernel import kernel_facade

        self.instances: List[Any] = []
        self.starts = 0
        self.shutdowns = 0
        cls = kernel_facade.Kernel
        original_init = cls.__init__
        original_start = cls.start
        original_shutdown = cls.shutdown
        ledger = self

        def __init__(k: Any, *args: Any, **kwargs: Any) -> None:
            ledger.instances.append(k)
            original_init(k, *args, **kwargs)

        async def start(k: Any) -> None:
            ledger.starts += 1
            await original_start(k)

        async def shutdown(k: Any) -> None:
            ledger.shutdowns += 1
            await original_shutdown(k)

        monkeypatch.setattr(cls, "__init__", __init__)
        monkeypatch.setattr(cls, "start", start)
        monkeypatch.setattr(cls, "shutdown", shutdown)


class RecordingShell:
    """A kernel-shaped egress (``Kernel(kernel_shell=...)``) that records
    every frame the dispatcher emits and lets the test deliver inbound
    frames to the registered Comm target, the way a transport does."""

    class _Session:
        def __init__(self, shell: "RecordingShell") -> None:
            self._shell = shell

        def send(self, _socket: Any, msg_type: str, **kwargs: Any) -> None:
            content = kwargs.get("content") or {}
            data = (content.get("data") or {}) if isinstance(content, dict) else {}
            run = data.get("application/vnd.rts.run+json") if isinstance(data, dict) else None
            self._shell.record({"msg_type": msg_type, "run": run})

    class _Comm:
        def __init__(self, shell: "RecordingShell", name: str) -> None:
            self.name = name
            self._shell = shell
            self.on_msg_cb: Optional[Callable[[Dict[str, Any]], None]] = None

        def on_msg(self, cb: Callable[[Dict[str, Any]], None]) -> None:
            self.on_msg_cb = cb

        def send(self, data: Dict[str, Any]) -> None:
            self._shell.record(data)

        def close(self) -> None:
            self.on_msg_cb = None

    class _CommManager:
        def __init__(self, shell: "RecordingShell") -> None:
            self._shell = shell
            self.comms: Dict[str, "RecordingShell._Comm"] = {}

        def register_target(self, name: str, callback: Callable[..., None]) -> None:
            comm = RecordingShell._Comm(self._shell, name)
            self.comms[name] = comm
            callback(comm, {"content": {"data": {}}})

        def unregister_target(self, name: str, _cb: Callable[..., None]) -> None:
            self.comms.pop(name, None)

    class _Shell:
        def __init__(self, shell: "RecordingShell") -> None:
            self.comm_manager = RecordingShell._CommManager(shell)

    def __init__(self) -> None:
        self.frames: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self.session = RecordingShell._Session(self)
        self.iopub_socket = "<conformance-iopub>"
        self.shell = RecordingShell._Shell(self)
        self._parent_header: Dict[str, Any] = {}

    def record(self, frame: Dict[str, Any]) -> None:
        with self._lock:
            self.frames.append(frame)

    def snapshot(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self.frames)

    def deliver(self, frame: Dict[str, Any]) -> None:
        comm = self.shell.comm_manager.comms.get(_versions.COMM_TARGET)
        assert comm is not None and comm.on_msg_cb is not None, (
            "dispatcher never registered its RFC-006 comm target"
        )
        comm.on_msg_cb({"content": {"data": frame}})


def thread_names_started_since(before: Iterable[int]) -> List[str]:
    before = set(before)
    return sorted(
        t.name for t in threading.enumerate()
        if t.ident not in before and t is not threading.main_thread()
        and t.is_alive()
        and not t.name.startswith(("pytest", "pydevd", "asyncio_", "ThreadPoolExecutor",
                                   "conformance-"))
    )


def run_serve_main_with_client(monkeypatch: Any, client: Callable[[int], Any], *,
                               token: str = "conformance-token",
                               extra_args: Optional[List[str]] = None) -> tuple:
    """Run ``serve_mode.main`` on the calling (main) thread while ``client``
    (given the port) runs on a worker thread; SIGINT stops serve once the
    client returns. Returns ``(rc, client_result)``."""
    import signal as _signal

    from llm_kernel.transports import serve_mode

    monkeypatch.setenv("CONFORMANCE_SERVE_TOKEN", token)
    port = free_port()
    result: Dict[str, Any] = {}

    def _client() -> None:
        try:
            result["value"] = client(port)
        except BaseException as exc:  # surfaced by the caller
            result["error"] = exc
        finally:
            os.kill(os.getpid(), _signal.SIGINT)

    saved = (_signal.getsignal(_signal.SIGINT), _signal.getsignal(_signal.SIGTERM))
    worker = threading.Thread(target=_client, name="conformance-serve-client", daemon=True)
    worker.start()
    try:
        rc = serve_mode.main([
            "--transport", "tcp", "--bind", f"127.0.0.1:{port}",
            "--auth-token-env", "CONFORMANCE_SERVE_TOKEN", *(extra_args or []),
        ])
    finally:
        _signal.signal(_signal.SIGINT, saved[0])
        _signal.signal(_signal.SIGTERM, saved[1])
        worker.join(WAIT_SECONDS)
    if "error" in result:
        raise result["error"]
    return rc, result.get("value")
