"""
LLM Kernel Entry Point

This module provides the entry point for running the LLM kernel.

Subcommand dispatch (additive across Stage 2 tracks):
    ``python -m llm_kernel``                          -> launch IPython kernel app
    ``python -m llm_kernel mcp-server ...``           -> RFC-001 MCP server (B1)
    ``python -m llm_kernel litellm-proxy ...``        -> RFC-002 LiteLLM proxy (B5)
    ``python -m llm_kernel paper-telephone-smoke``    -> kernel-only paper-telephone
                                                         smoke (Track B3)
    ``python -m llm_kernel agent-supervisor-smoke``   -> spawn one Claude Code agent
                                                         end-to-end (Track B4); requires
                                                         ANTHROPIC_API_KEY in env
    ``python -m llm_kernel metadata-writer-smoke``    -> single-process MetadataWriter
                                                         end-to-end smoke (RFC-005 / RFC-006
                                                         Family F).  No network required.
    ``python -m llm_kernel pty-mode``                 -> RFC-008 §3 + §4 PTY+socket kernel
                                                         entry point. Reads
                                                         ``LLMKERNEL_IPC_SOCKET`` from env.
    ``python -m llm_kernel pty-mode-smoke``           -> RFC-008 end-to-end smoke against a
                                                         UDS server in the same process.
                                                         No node-pty required.

The historical top-level imports below are wrapped in a try/except so that
the subcommands can still be dispatched in a kernel environment that does
not ship the full notebook dependency stack.
"""

import sys

try:
    from ipykernel.kernelapp import IPKernelApp
    from .kernel import LLMKernel

    class LLMKernelApp(IPKernelApp):
        """Application for launching the LLM kernel."""

        kernel_class = LLMKernel

        def initialize(self, argv=None):
            super().initialize(argv)
            # Additional initialization if needed
except ImportError:
    # Notebook-stack deps unavailable; the IPython-kernel main() will fail
    # if invoked, but the mcp-server subcommand can still run.
    LLMKernelApp = None  # type: ignore[assignment]


def main():
    """Main entry point for the LLM kernel."""
    if LLMKernelApp is None:
        raise RuntimeError(
            "LLM kernel notebook dependencies are not installed; "
            "only the 'mcp-server' subcommand is available in this environment."
        )
    app = LLMKernelApp.instance()
    app.initialize()
    app.start()


def _run_litellm_proxy(argv: list) -> None:
    """Track B5 dispatch: spawn the RFC-002 LiteLLM proxy and block on signal.

    Invoked by ``python -m llm_kernel litellm-proxy --port 0 --host 127.0.0.1``.
    The proxy runs uvicorn in a daemon thread; this entrypoint blocks the
    main thread on a signal.pause-equivalent until SIGTERM/SIGINT.
    """
    import argparse
    import os
    import signal
    import threading

    from . import litellm_proxy as _proxy

    parser = argparse.ArgumentParser(prog="llm_kernel litellm-proxy")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args(argv)

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    server = _proxy.LiteLLMProxyServer(api_key=api_key, host=args.host, port=args.port)
    server.start()
    print(server.base_url(), flush=True)

    stop_event = threading.Event()

    def _stop(signum: int, frame: object) -> None:  # pragma: no cover
        stop_event.set()

    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _stop)
    if hasattr(signal, "SIGINT"):
        signal.signal(signal.SIGINT, _stop)
    try:
        stop_event.wait()
    finally:
        server.stop()


def _run_paper_telephone_smoke() -> int:
    """Track B3 dispatch: kernel-only paper-telephone smoke test.

    Spins up the LiteLLM proxy on an ephemeral port, instantiates a
    :class:`CustomMessageDispatcher` backed by a stub kernel that prints
    every IOPub message as JSONL, points a :class:`RunTracker` at it,
    and synthetically calls ``OperatorBridgeServer._handle_notify`` to
    drive a full ``run.start`` -> ``run.complete`` cycle. Asserts the
    cycle landed on stdout. No Claude Code or VS Code required.
    """
    import asyncio
    import json
    import uuid as _uuid
    from unittest.mock import MagicMock

    from . import litellm_proxy as _proxy
    from .custom_messages import CustomMessageDispatcher
    from .mcp_server import OperatorBridgeServer
    from .run_tracker import RunTracker

    emitted: list = []

    class _PrintSession:
        def send(self, _sock, msg_type, **kwargs):
            payload = {"msg_type": msg_type, "content": kwargs.get("content")}
            print(json.dumps(payload, default=str), flush=True)
            emitted.append(msg_type)

    kernel = MagicMock()
    kernel.session = _PrintSession()
    kernel.iopub_socket = MagicMock()

    class _CommMgr:
        def register_target(self, n, cb): ...
        def unregister_target(self, n, cb): ...

    kernel.shell.comm_manager = _CommMgr()
    kernel._parent_header = {}

    server = _proxy.LiteLLMProxyServer(api_key="smoke", host="127.0.0.1", port=0)
    server.start()
    try:
        dispatcher = CustomMessageDispatcher(kernel)
        tracker = RunTracker(
            trace_id=str(_uuid.uuid4()), sink=dispatcher,
            agent_id="smoke", zone_id="smoke",
        )
        bridge = OperatorBridgeServer(
            agent_id="smoke", zone_id="smoke", run_tracker=tracker,
        )
        rid = tracker.start_run(
            name="notify", run_type="tool", inputs={"observation": "hello", "importance": "info"},
        )
        asyncio.run(bridge._handle_notify({"observation": "hello", "importance": "info"}))
        tracker.complete_run(rid, outputs={"acknowledged": True})
    finally:
        server.stop()

    if "display_data" not in emitted or "update_display_data" not in emitted:
        print("FAIL: missing run.start or run.complete IOPub", flush=True)
        return 1
    print("OK: paper-telephone smoke complete", flush=True)
    return 0


def _run_agent_supervisor_smoke() -> int:
    """Track B4 dispatch: spawn one Claude Code agent end-to-end.

    Brings up the LiteLLM proxy on an ephemeral port, instantiates a
    :class:`CustomMessageDispatcher` + :class:`RunTracker` +
    :class:`AgentSupervisor`, then calls ``supervisor.spawn`` against
    the configured ``ANTHROPIC_API_KEY``. Waits up to 60s for the
    agent process to finish; PASS if a ``notify`` and a
    ``report_completion`` are observed in the run-tracker.

    The argv this assembles is the RFC-002 v1.0.0 best-guess; deviations
    from real Claude Code CLI behavior land as RFC-002 v1.0.1 amendments
    once the operator runs the R2-prototype.
    """
    import json
    import os
    import tempfile
    import time
    import uuid as _uuid
    from pathlib import Path
    from unittest.mock import MagicMock

    # Load secrets from a project-root .env if python-dotenv is available.
    # Walks up from CWD; first .env found wins. Safe no-op if missing.
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    from . import litellm_proxy as _proxy
    from . import anthropic_passthrough as _passthrough
    from .agent_supervisor import AgentSupervisor
    from .custom_messages import CustomMessageDispatcher
    from .run_tracker import RunTracker

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "FAIL: ANTHROPIC_API_KEY missing from env "
            "(set it directly or drop a .env at the project root)",
            flush=True,
        )
        return 1

    class _Sink:
        def send(self, *args, **kwargs):  # noqa: D401, ANN001
            pass

    kernel = MagicMock()
    kernel.session = _Sink()
    kernel.iopub_socket = MagicMock()

    class _CommMgr:
        def register_target(self, n, cb): ...
        def unregister_target(self, n, cb): ...

    kernel.shell.comm_manager = _CommMgr()
    kernel._parent_header = {}

    # Pick the proxy: LLMKERNEL_USE_PASSTHROUGH=1 selects the transparent
    # Anthropic passthrough at /v1/* (works under OAuth — model-resolution
    # preflights reach the real API). Default is the LiteLLM proxy
    # (only honors --bare/API-key auth flows).
    use_passthrough = os.environ.get("LLMKERNEL_USE_PASSTHROUGH") == "1"
    dispatcher = CustomMessageDispatcher(MagicMock())  # placeholder
    tracker = RunTracker(
        trace_id=str(_uuid.uuid4()), sink=dispatcher,
        agent_id="smoke", zone_id="smoke",
    )
    if use_passthrough:
        from . import anthropic_passthrough as _pt
        server = _pt.AnthropicPassthroughServer(
            run_tracker=tracker, host="127.0.0.1", port=0,
        )
    else:
        server = _proxy.LiteLLMProxyServer(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            host="127.0.0.1", port=0,
        )
    server.start()
    try:
        # Re-bind dispatcher to the real MagicMock kernel.
        dispatcher = CustomMessageDispatcher(kernel)
        tracker = RunTracker(
            trace_id=str(_uuid.uuid4()), sink=dispatcher,
            agent_id="smoke", zone_id="smoke",
        )
        # If we have a passthrough server, swap in the new tracker.
        if use_passthrough:
            server.run_tracker = tracker  # type: ignore[attr-defined]
        # Instantiate the supervisor directly. `attach_agent_supervisor`
        # uses ``getattr(kernel, ATTR, None)`` for idempotency, but the
        # MagicMock kernel above auto-creates attributes — which would
        # short-circuit and return a MagicMock instead of a real
        # supervisor. Smoke harnesses bypass the attach helper.
        supervisor = AgentSupervisor(
            run_tracker=tracker, dispatcher=dispatcher,
            litellm_endpoint_url=server.base_url(),
        )
        # Use a debug-friendly dir so the operator can inspect spawn artifacts
        # (mcp-config.json, system-prompt.txt, kernel.stderr.<id>.log) after
        # the run. Cleaned up on the next smoke invocation.
        debug_root = Path(".run-smoke")
        if debug_root.exists():
            import shutil
            shutil.rmtree(debug_root, ignore_errors=True)
        debug_root.mkdir(exist_ok=True)
        work_dir = debug_root
        # Pin a small model for the smoke. Default opus-4-7 with 1M
        # context creates ~19k cache-creation tokens per call (~$0.12);
        # haiku-4-5 keeps the per-run cost in single-cent territory.
        # Default to OAuth (Claude Code's keychain). Set
        # LLMKERNEL_USE_BARE=1 to force ANTHROPIC_API_KEY auth instead.
        # When the passthrough is selected, force set_base_url=True so
        # Claude Code routes ALL /v1/* through our proxy (which handles
        # model-resolution preflights cleanly, unlike LiteLLM).
        handle = supervisor.spawn(
            zone_id="smoke", agent_id="alpha",
            task="Use the notify tool to greet the operator. Then call report_completion.",
            work_dir=work_dir,
            model=os.environ.get(
                "LLMKERNEL_SMOKE_MODEL", "claude-haiku-4-5-20251001",
            ),
            use_bare=os.environ.get("LLMKERNEL_USE_BARE") == "1",
            set_base_url=True if use_passthrough else None,
        )
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline and handle.poll() is None:
            time.sleep(0.5)
        if handle.poll() is None:
            handle.terminate()
            print("FAIL: agent did not exit within 60s", flush=True)
            _dump_debug_artifacts(debug_root, "alpha")
            return 1
        # If passthrough is on, flush the mitm log into the tracker so
        # we can also report on intercepted Anthropic API calls.
        if use_passthrough and hasattr(server, "flush_into_tracker"):
            try:
                rows = server.flush_into_tracker(tracker)  # type: ignore[union-attr]
                req_count = sum(1 for r in rows if r.get("kind") == "request")
                if req_count:
                    print(
                        f"INFO: passthrough intercepted {req_count} Anthropic "
                        f"API call(s); see {server.log_file}",  # type: ignore[union-attr]
                        flush=True,
                    )
                else:
                    print(
                        "WARN: passthrough intercepted 0 calls "
                        "(agent may not have routed through us)",
                        flush=True,
                    )
            except Exception as exc:  # noqa: BLE001
                print(f"WARN: mitm log flush raised: {exc}", flush=True)
        names = {r.name for r in tracker.iter_runs()}
        if "notify" in names and "report_completion" in names:
            print("OK: agent emitted notify + report_completion", flush=True)
            return 0
        print(
            "FAIL: missing tool runs; observed=" + json.dumps(sorted(names)),
            flush=True,
        )
        _dump_debug_artifacts(debug_root, "alpha")
        return 1
    finally:
        server.stop()


def _dump_debug_artifacts(work_dir, agent_id: str) -> None:  # noqa: ANN001
    """Print spawn artifact contents to stdout so the operator can debug."""
    from pathlib import Path
    spawn_dir = Path(work_dir) / ".run" / agent_id
    print(f"\n--- DEBUG: {spawn_dir} ---", flush=True)
    if not spawn_dir.exists():
        print("(no spawn dir)", flush=True)
        return
    for name in ("mcp-config.json", "system-prompt.txt",
                 f"kernel.stderr.{agent_id}.log"):
        path = spawn_dir / name
        if path.exists():
            print(f"\n--- {name} ({path.stat().st_size} bytes) ---", flush=True)
            try:
                txt = path.read_text(encoding="utf-8", errors="replace")
                # Truncate very long stderr to last 2KB to keep output manageable.
                if len(txt) > 2048:
                    txt = "...[truncated to last 2048 chars]...\n" + txt[-2048:]
                print(txt, flush=True)
            except OSError as exc:
                print(f"(read failed: {exc})", flush=True)


def _run_metadata_writer_smoke() -> int:
    """RFC-005 / RFC-006 dispatch: the metadata writer end-to-end smoke.

    Spins up a tracker + dispatcher + ``MetadataWriter`` against a stub
    kernel and:

    1. Emits a few runs (open + close, simulating real activity).
    2. Triggers a manual ``snapshot`` and prints the captured Family F
       envelope.
    3. Verifies forbidden-secret rejection by feeding a config with an
       ``api_key`` field and asserting :class:`SecretRejected` raises.

    Exits 0 on PASS, 1 on any failure.  No network, no subprocess.
    """
    import json
    import uuid as _uuid
    from typing import Any, Dict, List
    from unittest.mock import MagicMock

    from .custom_messages import CustomMessageDispatcher
    from .metadata_writer import MetadataWriter, SecretRejected
    from .run_tracker import RunTracker

    # Stub kernel mirroring the IPython surface the dispatcher uses.
    sent: List[tuple] = []

    class _Session:
        def send(self, _sock: Any, msg_type: str, **kwargs: Any) -> None:
            sent.append((msg_type, kwargs))

    class _CommMgr:
        def register_target(self, _n: str, _cb: Any) -> None: ...
        def unregister_target(self, _n: str, _cb: Any) -> None: ...

    kernel = MagicMock()
    kernel.session = _Session()
    kernel.iopub_socket = MagicMock()
    kernel.shell.comm_manager = _CommMgr()
    kernel._parent_header = {}

    dispatcher = CustomMessageDispatcher(kernel)
    dispatcher.start()
    tracker = RunTracker(
        trace_id=str(_uuid.uuid4()), sink=dispatcher,
        agent_id="smoke", zone_id="smoke",
    )
    writer = MetadataWriter(
        dispatcher=dispatcher, run_tracker=tracker,
        autosave_interval_sec=999.0,
        session_id="smoke-session",
    )

    # Step 1: emit a few runs simulating real activity.
    for _ in range(3):
        rid = tracker.start_run(
            name="notify", run_type="tool",
            inputs={"observation": "smoke", "importance": "info"},
        )
        tracker.complete_run(rid, outputs={"acknowledged": True})

    # Step 2: trigger a snapshot manually.
    snapshot = writer.snapshot(trigger="save")
    envelope = writer.take_last_envelope()
    if envelope is None or envelope["message_type"] != "notebook.metadata":
        print(
            "FAIL: writer did not emit notebook.metadata; "
            f"last_envelope={envelope}",
            flush=True,
        )
        return 1
    print("metadata-writer-smoke: captured Family F envelope:", flush=True)
    print(json.dumps({
        "type": envelope["message_type"],
        "snapshot_version": envelope["payload"]["snapshot_version"],
        "trigger": envelope["payload"]["trigger"],
        "schema_version": envelope["payload"]["snapshot"]["schema_version"],
        "session_id": envelope["payload"]["snapshot"]["session_id"],
        "run_count": len(envelope["payload"]["snapshot"]["event_log"]["runs"]),
    }, indent=2), flush=True)

    # Step 3: forbidden-secret rejection.  Crafted config carries
    # ``api_key`` (case-insensitive forbidden); the writer MUST raise
    # ``SecretRejected`` and MUST NOT log the value.
    rejected = False
    try:
        writer.update_config(
            recoverable={"kernel": {"api_key": "DO_NOT_LOG"}},
            volatile={},
        )
    except SecretRejected as exc:
        if "DO_NOT_LOG" in str(exc):
            print(
                "FAIL: SecretRejected leaked the offending VALUE",
                flush=True,
            )
            return 1
        rejected = True
    if not rejected:
        print(
            "FAIL: forbidden api_key was NOT rejected by metadata writer",
            flush=True,
        )
        return 1

    print("OK: metadata-writer-smoke complete", flush=True)
    return 0


def _run_pty_mode_smoke() -> int:
    """RFC-008 dispatch: end-to-end pty-mode smoke against a UDS server.

    Spins up a server in this process (UDS on POSIX, loopback TCP
    fallback on Windows / when ``AF_UNIX`` is unavailable), spawns the
    kernel as a subprocess in pty-mode, waits for the ready handshake,
    sends a ``cell_execute`` operator-action envelope, and asserts a
    Family A run.start span arrives back. No node-pty required: the
    kernel's behavior is identical regardless of whether stdin is a real
    PTY or a pipe; the termios setup is a no-op when stdin isn't a TTY.
    """
    import json
    import os
    import socket as _socket
    import subprocess
    import sys
    import tempfile
    import threading
    import time
    import uuid as _uuid

    session_id = str(_uuid.uuid4())

    # Allocate a transport address. UDS on POSIX, TCP loopback on
    # Windows or when AF_UNIX isn't available (RFC-008 §2 fallback).
    use_uds = hasattr(_socket, "AF_UNIX") and sys.platform != "win32"
    if use_uds:
        tmpdir = tempfile.mkdtemp(prefix="llmnb-pty-smoke-")
        sock_path = os.path.join(tmpdir, f"llmnb-{session_id}.sock")
        family = _socket.AF_UNIX
        bind_target = sock_path
        env_address = sock_path  # unprefixed = unix per RFC-008 §2
    else:
        tmpdir = None
        family = _socket.AF_INET
        # Bind to an ephemeral port; read back the actual port assigned.
        probe = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        bind_target = ("127.0.0.1", port)
        env_address = f"tcp:127.0.0.1:{port}"

    server = _socket.socket(family, _socket.SOCK_STREAM)
    server.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
    server.bind(bind_target)
    if use_uds:
        os.chmod(bind_target, 0o600)
    server.listen(1)
    server.settimeout(30.0)

    received_frames: list = []
    accepted_holder: dict = {}

    def _accept_and_read() -> None:
        try:
            conn, _addr = server.accept()
        except _socket.timeout:
            return
        accepted_holder["conn"] = conn
        conn.settimeout(30.0)
        buf = bytearray()
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            try:
                chunk = conn.recv(4096)
            except _socket.timeout:
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
                if not line.strip():
                    continue
                try:
                    received_frames.append(json.loads(line.decode("utf-8")))
                except ValueError:
                    pass

    reader = threading.Thread(target=_accept_and_read, daemon=True)
    reader.start()

    env = dict(os.environ)
    env["LLMKERNEL_IPC_SOCKET"] = env_address
    env["LLMKERNEL_PTY_MODE"] = "0"  # we're not under a real PTY
    env["LLMKERNEL_SESSION_ID"] = session_id
    # Keep PYTHONPATH so the subprocess finds llm_kernel without install.
    env.setdefault("PYTHONPATH", os.pathsep.join(sys.path))

    proc = subprocess.Popen(
        [sys.executable, "-m", "llm_kernel", "pty-mode"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )

    failure = None
    ready_seen = False
    try:
        # Wait for ready handshake. Snapshot the list before iterating
        # so a concurrent ``append`` from the reader thread never raises.
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            for frame in list(received_frames):
                attrs = frame.get("attributes") or []
                if isinstance(attrs, list):
                    for pair in attrs:
                        if (
                            isinstance(pair, dict)
                            and pair.get("key") == "event.name"
                            and pair.get("value", {}).get("stringValue") == "kernel.ready"
                        ):
                            ready_seen = True
                            break
                if ready_seen:
                    break
            if ready_seen:
                break
            time.sleep(0.1)
        if not ready_seen:
            failure = "no ready handshake within 30s"
        else:
            # Send a heartbeat.extension envelope -- catalogued in RFC-006
            # so it survives validate_envelope; no handler is registered
            # so the dispatcher logs "no registered handlers; dropped"
            # and continues. We then assert the kernel process is still
            # alive (parse path didn't crash).
            conn = accepted_holder.get("conn")
            if conn is not None:
                try:
                    conn.sendall(
                        (json.dumps({
                            "type": "heartbeat.extension",
                            "payload": {"sequence": 1, "elapsed_ms": 0},
                        }) + "\n").encode("utf-8")
                    )
                except OSError as exc:
                    failure = f"failed to send envelope: {exc}"
                else:
                    time.sleep(0.5)
                    if proc.poll() is not None:
                        failure = (
                            f"kernel exited prematurely after inbound "
                            f"envelope (exit={proc.returncode})"
                        )
    finally:
        # Unconditional teardown: close the socket FIRST so the kernel's
        # read loop sees EOF and exits cleanly through its finally
        # block (final ``notebook.metadata`` snapshot + ``writer.close``).
        # Closing the socket is portable; SIGTERM via ``Popen.terminate``
        # is uncatchable on Windows (TerminateProcess), so we don't
        # rely on it for clean shutdown.
        try:
            conn = accepted_holder.get("conn")
            if conn is not None:
                try:
                    conn.shutdown(_socket.SHUT_RDWR)
                except OSError:
                    pass
                conn.close()
        except OSError:
            pass
        try:
            server.close()
        except OSError:
            pass
        # Wait briefly for the kernel to exit on socket EOF; fall back to
        # terminate -> kill if it doesn't.
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except (subprocess.TimeoutExpired, OSError):
                try:
                    proc.kill()
                    proc.wait(timeout=5)
                except (subprocess.TimeoutExpired, OSError):
                    pass
        if use_uds and tmpdir is not None:
            import shutil as _shutil
            _shutil.rmtree(tmpdir, ignore_errors=True)

    if failure is not None:
        stderr_tail = (proc.stderr.read() or b"").decode("utf-8", errors="replace")
        print(f"FAIL: {failure}", flush=True)
        print(f"--- kernel stderr ---\n{stderr_tail}", flush=True)
        return 1
    print(
        f"OK: pty-mode-smoke complete; received {len(received_frames)} frame(s)",
        flush=True,
    )
    return 0


if __name__ == '__main__':
    # Load secrets from a project-root .env if python-dotenv is available.
    # Uses find_dotenv() to walk up from CWD until a .env is found. Safe
    # no-op if missing. Applied for ALL subcommands so subprocess-spawned
    # kernels (e.g. the extension's PtyKernelClient launching ``pty-mode``
    # with cwd=workspaceFolders[0]) pick up ANTHROPIC_API_KEY without the
    # parent process having to set it.
    try:
        from dotenv import find_dotenv as _find_dotenv, load_dotenv as _load_dotenv
        _dotenv_path = _find_dotenv(usecwd=True)
        if _dotenv_path:
            _load_dotenv(_dotenv_path)
    except ImportError:
        pass

    # Additive subcommand dispatch:
    #   ``mcp-server``               -> Track B1 RFC-001 MCP server
    #   ``litellm-proxy``            -> Track B5 RFC-002 LiteLLM proxy
    #   ``paper-telephone-smoke``    -> Track B3 kernel-only smoke
    #   ``agent-supervisor-smoke``   -> Track B4 single-agent end-to-end smoke
    #   ``metadata-writer-smoke``    -> RFC-005 / RFC-006 Family F smoke
    #   ``pty-mode``                 -> RFC-008 PTY+socket kernel for extension
    if len(sys.argv) > 1 and sys.argv[1] == "mcp-server":
        from . import mcp_server as _mcp_server

        _mcp_server.main(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == "litellm-proxy":
        _run_litellm_proxy(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == "paper-telephone-smoke":
        sys.exit(_run_paper_telephone_smoke())
    elif len(sys.argv) > 1 and sys.argv[1] == "agent-supervisor-smoke":
        sys.exit(_run_agent_supervisor_smoke())
    elif len(sys.argv) > 1 and sys.argv[1] == "metadata-writer-smoke":
        sys.exit(_run_metadata_writer_smoke())
    elif len(sys.argv) > 1 and sys.argv[1] == "pty-mode":
        # Pre-BSP-004 sync dispatch path — restored after a regression
        # in the BSP-004 uvicorn dispatch broke test #4 (live /spawn).
        # Symptom: _run_read_loop exited ~1.4s after agent_spawn_returned
        # because subprocess.Popen-from-thread-pool-worker had different
        # handle-lifecycle behavior than from main thread on Windows.
        # The BSP-004 scaffolding (app.py, boot_kernel, shutdown_kernel,
        # _async_serve_socket) stays in pty_mode.py for a future
        # uvicorn-with-correct-thread-model attempt; the entry point
        # uses the synchronous main() which is the known-good path.
        from .pty_mode import main as _pty_main
        sys.exit(_pty_main(sys.argv[2:]))
    elif len(sys.argv) > 1 and sys.argv[1] == "pty-mode-smoke":
        sys.exit(_run_pty_mode_smoke())
    else:
        main()
