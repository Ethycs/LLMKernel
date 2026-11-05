"""BSP-004 — kernel runtime under uvicorn, on the Kernel facade.

The FastAPI lifespan owns exactly one :class:`~llm_kernel.kernel_facade.Kernel`
(PLAN-kernel-facade commit 8). The RFC-008 PTY+socket transport is the same
:class:`~llm_kernel.transports.pty_mode.PtyTransport` the pty-mode
subcommand uses; here it is bound on uvicorn's event loop. ``/health`` reads
the kernel from ``app.state`` — there is no module-level state dict.
"""
from __future__ import annotations

import os
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, Request

from llm_kernel._internal import _diagnostics
from llm_kernel.kernel_facade import Kernel
from llm_kernel.transports import pty_mode
from llm_kernel.wire import versions as _versions


@asynccontextmanager
async def lifespan(app: FastAPI):
    """BSP-004 §4 boot sequence on the facade: bring the wire up, start the
    kernel, bind the transport, yield to uvicorn, tear down in reverse.

    Wire-level boot failures keep the legacy exit-code semantics (2 no
    socket env var, 3 socket connect failed, 11/12 proxy K11/K12): uvicorn
    would not otherwise propagate them, so the process exits hard.
    """
    _diagnostics.mark("app_lifespan_startup_begin")
    address = os.environ.get(pty_mode.ENV_IPC_SOCKET, "")
    pty_flag = os.environ.get(pty_mode.ENV_PTY_MODE) == "1"
    session_id = os.environ.get(pty_mode.ENV_SESSION_ID) or str(uuid.uuid4())
    if not address:
        sys.stderr.write(
            f"LLMKernel app: {pty_mode.ENV_IPC_SOCKET} env var is required\n"
        )
        sys.stderr.flush()
        _diagnostics.mark("app_lifespan_startup_failed", rc=2)
        os._exit(2)
    transport = pty_mode.PtyTransport(
        address=address, session_id=session_id, pty_flag=pty_flag,
    )
    rc = transport.prepare()
    if rc != 0:
        _diagnostics.mark("app_lifespan_startup_failed", rc=rc)
        sys.stderr.write(
            f"LLMKernel app: kernel boot failed with rc={rc}; exiting\n"
        )
        sys.stderr.flush()
        os._exit(rc)
    app.state.kernel = None
    app.state.session_id = session_id
    try:
        async with Kernel(
            workdir=Path.cwd(), kernel_shell=transport.kernel_shell,
        ) as kernel:
            app.state.kernel = kernel
            await kernel.bind_transport(transport)
            _diagnostics.mark("app_lifespan_startup_done", session_id=session_id)
            try:
                yield
            finally:
                _diagnostics.mark("app_lifespan_shutdown_begin")
    finally:
        # The socket closes only after Kernel.shutdown() emitted the final
        # notebook.metadata snapshot through it.
        app.state.kernel = None
        transport.close()
        _diagnostics.mark("app_lifespan_shutdown_done")


app = FastAPI(lifespan=lifespan, title="LLMKernel", version=_versions.KERNEL_VERSION)


@app.get("/health")
def health(request: Request) -> Dict[str, Any]:
    """Liveness probe. Reports the kernel's session id while the facade is
    attached to ``app.state``; uvicorn treats lifespan-startup failures as
    a non-200 startup outcome before this route ever serves.
    """
    kernel = getattr(request.app.state, "kernel", None)
    attached = kernel is not None and kernel.started
    return {
        "status": "ok",
        "kernel_session_id": (
            getattr(request.app.state, "session_id", None) if attached else None
        ),
        "kernel_version": _versions.KERNEL_VERSION,
    }
