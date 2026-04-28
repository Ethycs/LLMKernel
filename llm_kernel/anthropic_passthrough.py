"""mitmproxy-based Anthropic API passthrough with run-tracker logging.

V2 of the passthrough — replaces the FastAPI/httpx homegrown forwarder
with a thin wrapper around ``mitmdump``. mitmproxy handles all the
hop-by-hop / gzip / streaming corner cases correctly out of the box;
this module just spawns it in reverse-proxy mode, points it at
``https://api.anthropic.com``, and reads the JSONL log produced by the
companion addon (:mod:`llm_kernel._mitm_addon`) into the run-tracker.

Path namespacing
----------------

The proxy listens on ``http://host:port`` and rewrites incoming paths
straight onto ``https://api.anthropic.com``. The kernel can register
custom paths under ``/v$name/*`` by composing additional addons or
launching multiple instances; for V1 the only mode is ``"1"`` which
maps to upstream ``/v1`` 1-to-1.

Why subprocess instead of embedded
----------------------------------

mitmproxy's :class:`DumpMaster` runs in its own asyncio event loop and
has signal-handler logic that conflicts with non-main-thread embedding
on Windows. Spawning ``mitmdump`` as a subprocess sidesteps both
issues, and the addon's JSONL output is a stable interface for cross-
process communication.

Run-tracker integration
-----------------------

After ``stop()``, ``flush_into_tracker(tracker)`` reads the JSONL log
and emits one ``run_type=llm`` record per request/response pair, with
streaming chunks recorded as ``run.event(event_type="token")`` events.
Synchronous; called by the smoke before the proxy's temp dir is torn
down.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import sys  # noqa: F401  # kept for tests that monkeypatch sys.executable
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from . import zone_control

if TYPE_CHECKING:  # pragma: no cover
    from .run_tracker import RunTracker

logger: logging.Logger = logging.getLogger("llm_kernel.anthropic_passthrough")

#: Default upstream the reverse-proxy targets. Override via constructor.
UPSTREAM_BASE: str = "https://api.anthropic.com"

#: Bundled mitmproxy addon path; resolved relative to this module so
#: the supervisor can pass ``-s <abs path>`` to ``mitmdump``.
_ADDON_PATH: Path = Path(__file__).parent / "_mitm_addon.py"


def _allocate_port() -> int:
    """Find an available loopback port via :class:`socket.socket`."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]
    finally:
        sock.close()


class AnthropicPassthroughServer:
    """Run-tracker-instrumented mitmdump reverse proxy for Anthropic.

    Mirrors the surface of :class:`LiteLLMProxyServer` (start/stop/
    base_url) so the agent supervisor can swap one for the other.
    """

    def __init__(
        self, run_tracker: Optional["RunTracker"] = None,
        host: str = "127.0.0.1", port: int = 0,
        upstream_base: str = UPSTREAM_BASE,
        log_file: Optional[Path] = None,
    ) -> None:
        self.run_tracker = run_tracker
        self.host: str = host
        self.bound_port: int = port if port > 0 else _allocate_port()
        self.upstream_base: str = upstream_base.rstrip("/")
        if log_file is None:
            tmp = Path(tempfile.mkdtemp(prefix="mitm-log-"))
            log_file = tmp / "flows.jsonl"
        self.log_file: Path = Path(log_file)
        self._proc: Optional[subprocess.Popen] = None

    def base_url(self, mode: str = "1") -> str:
        """Return ``http://{host}:{port}`` for an agent's ANTHROPIC_BASE_URL.

        Empirical: the Anthropic Python SDK (and Claude Code) append a
        literal ``/v1/messages`` to ``base_url`` rather than using
        ``urljoin`` semantics. So ``base_url`` MUST NOT contain a
        ``/v1`` suffix; otherwise the agent emits requests for
        ``/v1/v1/messages`` (observed during R2-prototype runs).

        Future ``/v$name/*`` mounts will be exposed via a separate
        accessor — for V1 we only support the default mode "1" which
        passes through to upstream ``/v1/*``.
        """
        del mode
        return f"http://{self.host}:{self.bound_port}"

    def start(self) -> None:
        """Spawn mitmdump in reverse-proxy mode; block until it accepts."""
        if not _ADDON_PATH.exists():  # pragma: no cover - sanity
            raise RuntimeError(f"mitm addon missing: {_ADDON_PATH}")
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        # Truncate any prior log so the post-run reader gets only our run.
        self.log_file.write_text("", encoding="utf-8")

        env = dict(os.environ)
        env["MITM_LOG_FILE"] = str(self.log_file)
        # Resolve the actual mitmdump executable via RFC-009 §4.2 discovery
        # (env var override > PATH > pixi env probe). On Windows mitmdump
        # is a pip-installed entry-point exe at
        # ``<env>/Scripts/mitmdump.exe``, which the Extension Host's PATH
        # does not include when it spawns the kernel; the pixi probe is
        # what lets us survive that. Caller catches the RuntimeError
        # below and turns it into K12 (`pty_mode_proxy_start_failed`).
        mitmdump_bin = zone_control.locate_mitmdump_bin()
        if mitmdump_bin is None:
            raise RuntimeError(
                "mitmdump executable not found (set LLMNB_MITM_BIN, "
                "install mitmproxy, or run inside a pixi kernel env)"
            )
        argv = [
            mitmdump_bin,
            "--mode", f"reverse:{self.upstream_base}",
            "--listen-host", self.host,
            "--listen-port", str(self.bound_port),
            "-s", str(_ADDON_PATH),
        ]
        logger.info("starting mitmdump on %s:%d -> %s",
                    self.host, self.bound_port, self.upstream_base)
        self._proc = subprocess.Popen(
            argv, env=env, stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
        )
        # Poll until the listener accepts; mitmdump usually binds in <2s.
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                stderr = self._proc.stderr.read() if self._proc.stderr else ""
                raise RuntimeError(
                    f"mitmdump exited code={self._proc.returncode}; "
                    f"stderr={stderr[:500]}"
                )
            try:
                with socket.create_connection((self.host, self.bound_port),
                                              timeout=0.5):
                    return
            except OSError:
                time.sleep(0.1)
        self.stop()
        raise RuntimeError("mitmdump failed to start within 15s")

    def stop(self) -> None:
        """Terminate mitmdump; SIGTERM-equivalent then 3s grace then kill."""
        if self._proc is None:
            return
        try:
            self._proc.terminate()
        except OSError:  # pragma: no cover
            pass
        try:
            self._proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            try:
                self._proc.kill()
            except OSError:  # pragma: no cover
                pass
            try:
                self._proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:  # pragma: no cover
                logger.warning("mitmdump did not exit after SIGKILL")
        self._proc = None

    def flush_into_tracker(
        self, tracker: Optional["RunTracker"] = None,
    ) -> List[Dict[str, Any]]:
        """Read the addon's JSONL log; emit RunTracker events; return rows.

        Pairs ``request``/``response`` records by ``correlation_id`` and
        opens one OTLP span per pair (``llmnb.run_type = "llm"``).  The
        request URL/method/auth surface as inputs (folded into
        ``input.value``); ``gen_ai.system = "anthropic"`` and the model
        name when it can be parsed from the request path.  Errors
        (connection / TLS) close the span with
        ``status.code = STATUS_CODE_ERROR``.  Returns the raw list of
        rows read for the smoke harness's own assertions.
        """
        target = tracker or self.run_tracker
        rows: List[Dict[str, Any]] = []
        if not self.log_file.exists():
            return rows
        with self.log_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
        if target is None:
            return rows
        # Pair request + response by correlation_id (mitm addon's
        # per-flow id; distinct from the OTLP spanId we allocate).
        pending: Dict[str, str] = {}
        for row in rows:
            cid = row.get("correlation_id")
            if not cid:
                continue
            if row["kind"] == "request":
                run_id = target.start_run(
                    name=f"anthropic-passthrough:{row['method']} {row.get('path', '/')}",
                    run_type="llm",
                    inputs={
                        "method": row.get("method"),
                        "url": row.get("url"),
                        "host": row.get("host"),
                        "auth": row.get("auth"),
                        "content_type": row.get("content_type"),
                        "body_bytes": row.get("body_bytes"),
                        "stream": row.get("stream"),
                    },
                    metadata={
                        "gen_ai.system": "anthropic",
                        "http.method": row.get("method"),
                        "http.url": row.get("url"),
                        "mitm.correlation_id": cid,
                        "mitm.ts": row.get("ts"),
                    },
                )
                pending[cid] = run_id
            elif row["kind"] == "response":
                run_id = pending.pop(cid, None)
                if run_id is None:
                    continue
                status = row.get("status_code")
                ok = isinstance(status, int) and 200 <= status < 400
                target.complete_run(
                    run_id, outputs={
                        "status_code": status,
                        "body_bytes": row.get("body_bytes"),
                        "content_type": row.get("content_type"),
                        "duration_ms": row.get("duration_ms"),
                    },
                    status="STATUS_CODE_OK" if ok else "STATUS_CODE_ERROR",
                )
            elif row["kind"] == "error":
                run_id = pending.pop(cid, None)
                if run_id is None:
                    continue
                target.fail_run(
                    run_id,
                    error={
                        "exception.type": "AnthropicPassthroughError",
                        "exception.message": str(row.get("message", "")),
                    },
                )
        return rows


__all__ = [
    "AnthropicPassthroughServer", "UPSTREAM_BASE",
]
