"""Facade commits 6-8: the process entry points are consumers of ``Kernel``.

In-process checks (no subprocess) that pty-mode, serve and the uvicorn app
each run exactly one facade kernel and leave no subsystem threads behind.
"""
from __future__ import annotations

import json
import os
import signal
import socket
import threading
import time
from typing import Any, Dict, List

import pytest

from llm_kernel import kernel_facade


class _Counter:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.starts = 0
        self.shutdowns = 0
        original_start = kernel_facade.Kernel.start
        original_shutdown = kernel_facade.Kernel.shutdown

        async def start(k: kernel_facade.Kernel) -> None:
            self.starts += 1
            await original_start(k)

        async def shutdown(k: kernel_facade.Kernel) -> None:
            self.shutdowns += 1
            await original_shutdown(k)

        monkeypatch.setattr(kernel_facade.Kernel, "start", start)
        monkeypatch.setattr(kernel_facade.Kernel, "shutdown", shutdown)


def _thread_idents() -> set:
    return {t.ident for t in threading.enumerate()}


def _new_threads(before: set) -> List[str]:
    """Threads alive now that were not alive at ``before`` (other tests in the
    same process may leak their own; only this test's leftovers count)."""
    return sorted(
        t.name for t in threading.enumerate()
        if t.ident not in before and t is not threading.main_thread()
        and not t.name.startswith(("pytest", "pydevd", "asyncio_", "ThreadPoolExecutor"))
    )


class _Receiver:
    def __init__(self) -> None:
        self.server = socket.socket()
        self.server.bind(("127.0.0.1", 0))
        self.server.listen(1)
        self.server.settimeout(15.0)
        self.frames: List[Dict[str, Any]] = []
        self.conn: Any = None
        self.eof = threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()

    @property
    def address(self) -> str:
        return f"tcp:127.0.0.1:{self.server.getsockname()[1]}"

    def _loop(self) -> None:
        try:
            self.conn, _ = self.server.accept()
        except OSError:
            self.eof.set()
            return
        self.conn.settimeout(0.5)
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
                line = bytes(buf[:nl]); del buf[: nl + 1]
                if line.strip():
                    self.frames.append(json.loads(line.decode("utf-8")))
        self.eof.set()

    def close(self) -> None:
        for s in (self.conn, self.server):
            try:
                if s is not None:
                    s.close()
            except OSError:
                pass


@pytest.fixture(autouse=True)
def _external_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    # External proxy lifecycle: no kernel-owned proxy process in tests.
    monkeypatch.setenv("LLMKERNEL_LITELLM_ENDPOINT_URL", "http://127.0.0.1:1/v1")


def test_pty_mode_main_runs_one_facade_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    from llm_kernel.transports import pty_mode

    before = _thread_idents()
    counter = _Counter(monkeypatch)
    receiver = _Receiver()
    monkeypatch.setenv(pty_mode.ENV_IPC_SOCKET, receiver.address)
    monkeypatch.setenv(pty_mode.ENV_SESSION_ID, "facade-pty")
    monkeypatch.delenv(pty_mode.ENV_PTY_MODE, raising=False)
    rc: Dict[str, int] = {}
    t = threading.Thread(target=lambda: rc.setdefault("rc", pty_mode.main([])), daemon=True)
    t.start()
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and not any("dispatcher started" in str(f.get("body", {}).get("stringValue", "")) for f in list(receiver.frames)):
            time.sleep(0.05)
        receiver.conn.sendall((json.dumps({"type": "kernel.shutdown_request", "payload": {}}) + "\n").encode())
        assert receiver.eof.wait(15)
        t.join(15)
    finally:
        receiver.close()
    assert rc.get("rc") == 0
    assert (counter.starts, counter.shutdowns) == (1, 1)
    typed = [f for f in receiver.frames if "type" in f]
    assert typed and typed[-1]["type"] == "notebook.metadata" and typed[-1]["payload"]["trigger"] == "shutdown"
    assert _new_threads(before) == []


def test_serve_main_runs_one_facade_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    from llm_kernel.transports import serve_mode

    before = _thread_idents()
    counter = _Counter(monkeypatch)
    monkeypatch.setenv("LLMNB_AUTH_TOKEN", "facade-token")
    saved = (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM))
    threading.Timer(1.5, lambda: os.kill(os.getpid(), signal.SIGINT)).start()
    try:
        rc = serve_mode.main(["--transport", "tcp", "--bind", "127.0.0.1:0", "--auth-token-env", "LLMNB_AUTH_TOKEN"])
    finally:
        signal.signal(signal.SIGINT, saved[0])
        signal.signal(signal.SIGTERM, saved[1])
    assert rc == 0
    assert (counter.starts, counter.shutdowns) == (1, 1)
    assert _new_threads(before) == []


def test_app_lifespan_owns_one_facade_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    from starlette.testclient import TestClient

    from llm_kernel import app as app_module

    before = _thread_idents()
    counter = _Counter(monkeypatch)
    receiver = _Receiver()
    monkeypatch.setenv("LLMKERNEL_IPC_SOCKET", receiver.address)
    monkeypatch.setenv("LLMKERNEL_SESSION_ID", "facade-app")
    monkeypatch.delenv("LLMKERNEL_PTY_MODE", raising=False)
    try:
        with TestClient(app_module.app) as client:
            during = client.get("/health").json()
            assert isinstance(app_module.app.state.kernel, kernel_facade.Kernel)
        after = TestClient(app_module.app).get("/health").json()
    finally:
        receiver.close()
    assert during["kernel_session_id"] == "facade-app"
    assert after["kernel_session_id"] is None
    assert (counter.starts, counter.shutdowns) == (1, 1)
    assert not hasattr(app_module, "_state")
    assert _new_threads(before) == []
