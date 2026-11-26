"""Every process entry point runs on exactly one ``Kernel`` facade.

Sources: ``docs/06 - Roadmaps/PLAN-kernel-facade.md`` §"Entry-point
refactor" and commits 6-8 (pty-mode, serve and the FastAPI lifespan become
facade consumers; behaviour preserved, smokes stay green), RFC-008 §4
(ready frame first on the data plane, final ``notebook.metadata`` snapshot
on shutdown, exit codes 2/3), BSP-004 (``app.state.kernel``, ``/health``).
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List

import pytest

from llm_kernel import kernel_facade
from llm_kernel.wire import versions as _versions

from tests.conformance._support import (
    FacadeLedger,
    WAIT_SECONDS,
    WirePeer,
    connect_with_retry,
    free_port,
    handshake_request,
    is_ready_frame,
    kernel_subprocess_env,
    thread_names_started_since,
    wait_until,
)


def _typed(frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [f for f in frames if isinstance(f, dict) and "type" in f]


def _shutdown_request() -> Dict[str, Any]:
    return {"type": "kernel.shutdown_request", "payload": {}}


# -- pty-mode (commit 6) ----------------------------------------------------


def test_pty_mode_main_runs_one_facade_kernel_over_the_wire(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from llm_kernel.transports import pty_mode

    monkeypatch.chdir(tmp_path)
    before = [t.ident for t in threading.enumerate()]
    ledger = FacadeLedger(monkeypatch)
    peer = WirePeer()
    monkeypatch.setenv(pty_mode.ENV_IPC_SOCKET, peer.address)
    monkeypatch.setenv(pty_mode.ENV_SESSION_ID, "conformance-pty")
    rc: Dict[str, int] = {}
    runner = threading.Thread(
        target=lambda: rc.setdefault("rc", pty_mode.main([])),
        name="conformance-pty-main", daemon=True,
    )
    runner.start()
    try:
        assert wait_until(lambda: ledger.starts == 1)
        # Frames sent before the read loop binds wait in the socket buffer.
        peer.send(_shutdown_request())
        assert peer.eof.wait(WAIT_SECONDS), "kernel never closed the socket"
        runner.join(WAIT_SECONDS)
    finally:
        peer.close()
    assert rc.get("rc") == 0
    assert len(ledger.instances) == 1
    assert (ledger.starts, ledger.shutdowns) == (1, 1)
    frames = peer.snapshot()
    # RFC-008 §4: the ready handshake is the first data-plane frame.
    assert frames and is_ready_frame(frames[0]), frames[:2]
    # The facade's shutdown emitted the final snapshot through the
    # transport-supplied egress, before the socket closed.
    typed = _typed(frames)
    assert typed[-1]["type"] == "notebook.metadata"
    assert typed[-1]["payload"]["trigger"] == "shutdown"
    assert not ledger.instances[0].started
    assert wait_until(lambda: not thread_names_started_since(before), timeout=5.0), (
        thread_names_started_since(before)
    )


def test_pty_mode_main_without_socket_env_exits_2_before_any_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_kernel.transports import pty_mode

    ledger = FacadeLedger(monkeypatch)
    assert pty_mode.main([]) == 2
    assert ledger.instances == [] and ledger.starts == 0


def test_pty_mode_main_connect_failure_exits_3_before_any_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_kernel.transports import pty_mode

    ledger = FacadeLedger(monkeypatch)
    monkeypatch.setenv(pty_mode.ENV_IPC_SOCKET, f"tcp:127.0.0.1:{free_port()}")
    assert pty_mode.main([]) == 3
    assert ledger.instances == [] and ledger.starts == 0


def test_pty_boot_kernel_is_a_facade_lifetime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """BSP-004 sync boot helpers run one facade kernel; the state's
    subsystems are the facade's, and shutdown_kernel tears it down after
    the final snapshot reached the wire."""
    from llm_kernel.transports import pty_mode

    ledger = FacadeLedger(monkeypatch)
    peer = WirePeer()
    monkeypatch.setenv(pty_mode.ENV_IPC_SOCKET, peer.address)
    monkeypatch.setenv(pty_mode.ENV_SESSION_ID, "conformance-boot")
    state = pty_mode.boot_kernel(workdir=tmp_path)
    try:
        assert isinstance(state, dict), state
        facade = state["facade"]
        assert isinstance(facade, kernel_facade.Kernel)
        assert facade.started
        assert ledger.instances == [facade] and ledger.starts == 1
        assert state["dispatcher"] is facade.dispatcher
        assert state["metadata_writer"] is facade.metadata_writer
        assert state["supervisor"] is facade.supervisor
        assert state["tracker"] is facade.run_tracker
        assert state["session_id"] == "conformance-boot"
        assert facade.workdir == tmp_path
        assert facade.list_agents() == []
    finally:
        pty_mode.shutdown_kernel(state)
    assert peer.eof.wait(WAIT_SECONDS)
    peer.close()
    assert ledger.shutdowns == 1
    assert not facade.started
    with pytest.raises(RuntimeError):
        facade.list_agents()
    typed = _typed(peer.snapshot())
    assert typed and typed[-1]["payload"]["trigger"] == "shutdown"
    assert is_ready_frame(peer.snapshot()[0])


def test_pty_boot_kernel_without_socket_returns_2(monkeypatch: pytest.MonkeyPatch) -> None:
    from llm_kernel.transports import pty_mode

    ledger = FacadeLedger(monkeypatch)
    assert pty_mode.boot_kernel() == 2
    assert ledger.instances == []


# -- serve (commit 7) -------------------------------------------------------


def test_serve_boot_helper_owns_one_started_facade(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from llm_kernel.transports import serve_mode

    ledger = FacadeLedger(monkeypatch)
    state = serve_mode._boot_kernel_for_serve("none", workdir=tmp_path)  # noqa: SLF001
    try:
        kernel = state["kernel"]
        assert isinstance(kernel, kernel_facade.Kernel) and kernel.started
        assert ledger.instances == [kernel]
        assert state["dispatcher"] is kernel.dispatcher
        assert state["metadata_writer"] is kernel.metadata_writer
        assert state["supervisor"] is kernel.supervisor
        assert state["tracker"] is kernel.run_tracker
        assert state["server"] is None  # --proxy none
    finally:
        serve_mode._teardown_serve_kernel(state)  # noqa: SLF001
    assert (ledger.starts, ledger.shutdowns) == (1, 1)
    assert not kernel.started


def test_serve_main_runs_one_facade_kernel_and_serves_the_wire(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """``serve --transport tcp`` boots one facade kernel, answers the
    handshake from it, and shuts it down once on SIGINT."""
    from llm_kernel.transports import serve_mode

    if threading.current_thread() is not threading.main_thread():
        pytest.skip("serve installs signal handlers; needs the main thread")
    monkeypatch.chdir(tmp_path)
    before = [t.ident for t in threading.enumerate()]
    ledger = FacadeLedger(monkeypatch)
    monkeypatch.setenv("CONFORMANCE_SERVE_TOKEN", "conformance-token")
    port = free_port()
    seen: Dict[str, Any] = {}

    def client() -> None:
        try:
            conn = connect_with_retry("127.0.0.1", port)
            try:
                conn.send(handshake_request("conformance-token"))
                seen["handshake"] = conn.recv()
                seen["kernels_during"] = (len(ledger.instances), ledger.starts)
            finally:
                conn.close()
        finally:
            os.kill(os.getpid(), signal.SIGINT)

    saved = (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM))
    worker = threading.Thread(target=client, name="conformance-serve-client", daemon=True)
    worker.start()
    try:
        rc = serve_mode.main([
            "--transport", "tcp", "--bind", f"127.0.0.1:{port}",
            "--auth-token-env", "CONFORMANCE_SERVE_TOKEN",
        ])
    finally:
        signal.signal(signal.SIGINT, saved[0])
        signal.signal(signal.SIGTERM, saved[1])
        worker.join(WAIT_SECONDS)
    assert rc == 0
    assert seen.get("kernels_during") == (1, 1)
    payload = (seen.get("handshake") or {}).get("payload") or {}
    assert "error" not in payload, payload
    assert payload["kernel_version"] == _versions.KERNEL_VERSION
    assert payload["wire_version"] == _versions.WIRE_VERSION
    assert (ledger.starts, ledger.shutdowns) == (1, 1)
    assert wait_until(lambda: not thread_names_started_since(before), timeout=5.0), (
        thread_names_started_since(before)
    )


def test_serve_main_without_token_exits_2_before_any_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_kernel.transports import serve_mode

    ledger = FacadeLedger(monkeypatch)
    monkeypatch.delenv("CONFORMANCE_MISSING_TOKEN", raising=False)
    rc = serve_mode.main([
        "--transport", "tcp", "--bind", "127.0.0.1:0",
        "--auth-token-env", "CONFORMANCE_MISSING_TOKEN",
    ])
    assert rc == 2
    assert ledger.instances == []


# -- the uvicorn app (commit 8) --------------------------------------------


def test_app_lifespan_owns_one_facade_kernel_on_app_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """PLAN commit 8: the lifespan body is ``async with Kernel(...) as k:
    app.state.kernel = k``; the module-level ``_state`` dict is gone and
    ``/health`` reads the kernel from ``request.app.state``."""
    from starlette.testclient import TestClient

    from llm_kernel import app as app_module

    monkeypatch.chdir(tmp_path)
    ledger = FacadeLedger(monkeypatch)
    peer = WirePeer()
    monkeypatch.setenv("LLMKERNEL_IPC_SOCKET", peer.address)
    monkeypatch.setenv("LLMKERNEL_SESSION_ID", "conformance-app")
    try:
        with TestClient(app_module.app) as client:
            kernel = app_module.app.state.kernel
            assert isinstance(kernel, kernel_facade.Kernel) and kernel.started
            during = client.get("/health")
        assert peer.eof.wait(WAIT_SECONDS)
        after = TestClient(app_module.app).get("/health").json()
    finally:
        peer.close()
    assert during.status_code == 200
    assert during.json() == {
        "status": "ok",
        "kernel_session_id": "conformance-app",
        "kernel_version": _versions.KERNEL_VERSION,
    }
    assert after["kernel_session_id"] is None
    assert app_module.app.state.kernel is None
    assert not hasattr(app_module, "_state")
    assert ledger.instances == [kernel]
    assert (ledger.starts, ledger.shutdowns) == (1, 1)
    frames = peer.snapshot()
    assert frames and is_ready_frame(frames[0])
    typed = _typed(frames)
    assert typed and typed[-1]["payload"]["trigger"] == "shutdown"


def test_app_lifespan_keeps_the_exit_code_2_without_a_socket(tmp_path: Path) -> None:
    """app.py: wire-level boot failures keep the legacy exit codes (2 when
    the socket env var is missing) -- the process exits hard."""
    code = (
        "from starlette.testclient import TestClient\n"
        "from llm_kernel import app as m\n"
        "with TestClient(m.app):\n"
        "    pass\n"
    )
    env = kernel_subprocess_env()
    proc = subprocess.run(
        [sys.executable, "-c", code], env=env, cwd=str(tmp_path),
        capture_output=True, timeout=60,
    )
    assert proc.returncode == 2, proc.stderr.decode("utf-8", "replace")[-2000:]
    assert b"LLMKERNEL_IPC_SOCKET" in proc.stderr
