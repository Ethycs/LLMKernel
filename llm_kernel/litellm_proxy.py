"""LLMKernel LiteLLM proxy server (Stage 2 Track B5).

Production OpenAI/Anthropic-compatible HTTP proxy at ``ANTHROPIC_BASE_URL``
for every Claude Code agent (RFC-002 § "API base URL configuration").
LiteLLM is the layer-2 stable abstraction (DR-0016); this module is the
kernel's mediation seam in front of it. Sources: RFC-002 (contract),
RFC-003 (envelopes), RFC-004 (failure surface), and the prototype at
``_ingest/prototypes/r2-prototype/stub_litellm_proxy.py``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import socket
import threading
import time
import uuid
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Optional

import httpx
import litellm
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

if TYPE_CHECKING:  # pragma: no cover
    from .run_tracker import RunTracker  # type: ignore[import-not-found]

logger: logging.Logger = logging.getLogger("llm_kernel.litellm_proxy")

#: Static V1 model catalogue surfaced by ``GET /v1/models``. The proxy does
#: NOT use this for routing; LiteLLM resolves the actual provider downstream.
V1_MODELS: List[Dict[str, Any]] = [
    {"id": "claude-sonnet-4-6", "object": "model", "owned_by": "anthropic"},
    {"id": "claude-opus-4-7", "object": "model", "owned_by": "anthropic"},
]


def _allocate_free_port() -> int:
    """Bind a transient socket to port 0 and read back the OS-assigned port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _to_dict(obj: Any) -> Dict[str, Any]:
    """Best-effort conversion of LiteLLM responses to JSON-able dicts."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    for attr in ("model_dump", "dict", "to_dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return fn()  # type: ignore[no-any-return]
            except TypeError:
                pass
    try:
        return json.loads(json.dumps(obj, default=lambda o: getattr(o, "__dict__", str(o))))
    except (TypeError, ValueError):
        return {"_repr": repr(obj)}


class LiteLLMProxyServer:
    """OpenAI/Anthropic-compatible LiteLLM proxy hosted in-process.

    Each Claude Code agent's ``ANTHROPIC_BASE_URL`` points at this proxy on
    a loopback ephemeral port. Every model call is forwarded through
    ``litellm.acompletion`` and logged as an RFC-003 run record via the
    run-tracker. V1: passthrough auth (no bearer validation).
    """

    def __init__(
        self,
        api_key: str,
        run_tracker: Optional["RunTracker"] = None,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        self._api_key: str = api_key
        self._run_tracker: Optional["RunTracker"] = run_tracker
        self._host: str = host
        self._configured_port: int = port
        self.bound_port: Optional[int] = None
        self._app: FastAPI = self._build_app()
        self._server: Optional[uvicorn.Server] = None
        self._thread: Optional[threading.Thread] = None
        self._ready_event: threading.Event = threading.Event()

    def _build_app(self) -> FastAPI:
        """Construct the FastAPI app with the two RFC-002 routes."""
        app = FastAPI(title="llm_kernel.litellm_proxy", version="1.0.0")

        @app.on_event("startup")
        async def _on_startup() -> None:  # pragma: no cover
            self._ready_event.set()
            logger.info("litellm_proxy ready on %s:%s", self._host, self.bound_port)

        @app.get("/v1/models")
        async def list_models() -> Dict[str, Any]:
            """Return the static V1 model list for pre-spawn health checks."""
            return {"object": "list", "data": V1_MODELS}

        @app.post("/v1/messages")
        async def messages(request: Request) -> Any:
            """Anthropic Messages API entrypoint; forwards via LiteLLM."""
            return await self._handle_messages(request)

        return app

    async def _handle_messages(self, request: Request) -> Any:
        """Forward an Anthropic Messages request through LiteLLM."""
        try:
            body: Dict[str, Any] = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise HTTPException(status_code=400, detail=f"invalid JSON body: {exc}")

        model: Optional[str] = body.get("model")
        if not model:
            raise HTTPException(status_code=400, detail="'model' is required")

        run_id: str = str(uuid.uuid4())
        stream: bool = bool(body.get("stream"))

        # TODO(V1.5): validate HMAC bearer (RFC-002 § "Authentication"); V1
        # passes the inbound Authorization header through unchanged.
        # B3 note: the proxy never talks to the dispatcher directly; it
        # calls run_tracker.start_run / event / complete_run, and the
        # run-tracker's sink (the Track B3 CustomMessageDispatcher in
        # production, a list-sink in tests) does the IOPub emission.
        self._tracker_call(
            "start_run", run_id=run_id, name=f"litellm:{model}", run_type="llm",
            inputs=body, metadata={"endpoint": "v1/messages", "stream": stream},
        )
        logger.info("messages run_id=%s model=%s stream=%s", run_id, model, stream)

        kwargs: Dict[str, Any] = dict(body)
        kwargs["model"] = model if "/" in model else f"anthropic/{model}"
        kwargs.setdefault("api_key", self._api_key)

        try:
            result: Any = await litellm.acompletion(**kwargs)
        except Exception as exc:
            return self._fail(run_id, exc)

        if stream:
            return StreamingResponse(
                self._stream_through(run_id, result), media_type="text/event-stream"
            )

        outputs = _to_dict(result)
        self._complete(run_id, outputs, status="success")
        logger.info("messages run_id=%s complete", run_id)
        return JSONResponse(content=outputs)

    async def _stream_through(
        self, run_id: str, result: AsyncIterator[Any]
    ) -> AsyncIterator[bytes]:
        """Tee streamed chunks to the run-tracker; yield them unchanged."""
        try:
            async for chunk in result:
                data = _to_dict(chunk)
                logger.debug("stream chunk run_id=%s", run_id)
                self._tracker_call(
                    "event", run_id=run_id, event_type="token", data={"chunk": data}
                )
                yield (b"data: " + json.dumps(data).encode("utf-8") + b"\n\n")
        except Exception as exc:
            self._fail(run_id, exc)
            err = {"type": type(exc).__name__, "message": str(exc)}
            yield (b"data: " + json.dumps({"type": "error", "error": err}).encode("utf-8") + b"\n\n")
            return
        self._complete(run_id, {"streamed": True}, status="success")
        logger.info("messages run_id=%s stream complete", run_id)

    def _tracker_call(self, method: str, **kwargs: Any) -> None:
        """Invoke a run-tracker method, swallowing exceptions defensively."""
        if self._run_tracker is None:
            return
        try:
            getattr(self._run_tracker, method)(**kwargs)
        except Exception:  # pragma: no cover
            logger.exception("run_tracker.%s raised; continuing", method)

    def _complete(self, run_id: str, outputs: Dict[str, Any], status: str) -> None:
        """Forward run completion to the run-tracker."""
        self._tracker_call("complete_run", run_id=run_id, outputs=outputs, status=status)

    def _fail(self, run_id: str, exc: BaseException) -> JSONResponse:
        """Close the run with status=error and return an Anthropic-shaped body."""
        err = {
            "type": type(exc).__name__,
            "message": str(exc),
            "code": getattr(exc, "status_code", "litellm_error"),
        }
        logger.warning("messages run_id=%s error=%s", run_id, err)
        self._tracker_call(
            "complete_run", run_id=run_id, outputs={}, status="error", error=err
        )
        body = {"type": "error", "error": {"type": err["type"], "message": err["message"]}}
        status = err["code"] if isinstance(err["code"], int) else 502
        return JSONResponse(status_code=status, content=body)

    def start(self) -> None:
        """Start uvicorn in a daemon thread; block until ready."""
        self.bound_port = self._configured_port or _allocate_free_port()
        config = uvicorn.Config(
            self._app, host=self._host, port=self.bound_port,
            log_level="warning", lifespan="on", loop="asyncio",
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(
            target=lambda: asyncio.run(self._server.serve()),  # type: ignore[union-attr]
            name="llm_kernel.litellm_proxy", daemon=True,
        )
        self._thread.start()
        if not self._ready_event.wait(timeout=10.0):
            raise RuntimeError("litellm_proxy failed to become ready within 10s")

    def stop(self) -> None:
        """Graceful shutdown; SIGTERM-then-3s-grace-then-cancel pattern."""
        if self._server is None:
            return
        self._server.should_exit = True
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if self._thread is None or not self._thread.is_alive():
                break
            time.sleep(0.05)
        if self._thread is not None and self._thread.is_alive():
            self._server.force_exit = True
            self._thread.join(timeout=2.0)

    def base_url(self) -> str:
        """Return the URL Claude Code's ``ANTHROPIC_BASE_URL`` MUST point at."""
        if self.bound_port is None:
            raise RuntimeError("litellm_proxy not started; bound_port is unset")
        return f"http://{self._host}:{self.bound_port}/v1"

    def health(self) -> bool:
        """Return True iff ``GET /v1/models`` answers 200."""
        if self.bound_port is None:
            return False
        try:
            resp = httpx.get(self.base_url() + "/models", timeout=2.0)
            return resp.status_code == 200
        except httpx.HTTPError:
            return False
