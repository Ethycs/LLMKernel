"""BSP-004 — kernel runtime under uvicorn.

Boots the existing kernel subsystems on uvicorn's asyncio event loop. The
RFC-008 PTY+socket transport is unchanged; the socket reader runs as an
asyncio task instead of a thread (see :func:`pty_mode._async_serve_socket`).
A ``/health`` route is exposed for liveness probes; no other HTTP routes
in V1 (the data plane is RFC-008 socket envelopes).
"""

from __future__ import annotations

import asyncio
import os
import sys
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI

from . import _diagnostics, pty_mode

#: Module-level holder for the kernel state so /health and future routes
#: can read it without pulling app.state typing gymnastics. Set on
#: lifespan startup; cleared on shutdown.
_state: Dict[str, Any] = {}
_serve_task: Optional[asyncio.Task] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """BSP-004 §4 boot sequence: boot subsystems, schedule async socket
    reader as a task, yield to uvicorn, run cleanup on shutdown.
    """
    global _serve_task
    _diagnostics.mark("app_lifespan_startup_begin")
    state_or_rc = pty_mode.boot_kernel()
    if isinstance(state_or_rc, int):
        _diagnostics.mark("app_lifespan_startup_failed", rc=state_or_rc)
        sys.stderr.write(
            f"LLMKernel app: kernel boot failed with rc={state_or_rc}; exiting\n"
        )
        sys.stderr.flush()
        # Force the process to exit; uvicorn won't otherwise propagate
        # this. The exit code semantics match the legacy pty_mode.main.
        os._exit(state_or_rc)

    _state.update(state_or_rc)
    _serve_task = asyncio.create_task(pty_mode._async_serve_socket(_state))
    _diagnostics.mark("app_lifespan_startup_done", session_id=_state.get("session_id"))

    try:
        yield
    finally:
        _diagnostics.mark("app_lifespan_shutdown_begin")
        # Signal the async serve loop to stop
        done = _state.get("async_done")
        if done is not None:
            done.set()
        if _serve_task is not None:
            try:
                await asyncio.wait_for(_serve_task, timeout=2.0)
            except asyncio.TimeoutError:
                _serve_task.cancel()
            except Exception:  # pragma: no cover — best-effort cleanup
                pass
        # Run the legacy shutdown sequence
        try:
            pty_mode.shutdown_kernel(_state)
        except Exception:
            sys.stderr.write("LLMKernel app: shutdown_kernel raised\n")
            sys.stderr.flush()
        _state.clear()
        _diagnostics.mark("app_lifespan_shutdown_done")


app = FastAPI(lifespan=lifespan, title="LLMKernel", version="1.0.0")


@app.get("/health")
def health() -> Dict[str, Any]:
    """Liveness probe. Returns 200 with the kernel's session_id when the
    subsystems are attached; uvicorn treats lifespan-startup failures as
    a non-200 startup outcome before this route ever serves.
    """
    return {
        "status": "ok",
        "kernel_session_id": _state.get("session_id"),
        "kernel_version": getattr(pty_mode, "KERNEL_VERSION", "unknown"),
    }
