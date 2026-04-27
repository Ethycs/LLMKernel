"""Tests for the Stage 2 Track B5 LiteLLM proxy module.

Uses :class:`fastapi.testclient.TestClient` so no real network or real
provider calls are made. ``litellm.acompletion`` is patched per-test.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List

import pytest
from fastapi.testclient import TestClient

from llm_kernel import litellm_proxy as proxy_module
from llm_kernel.litellm_proxy import LiteLLMProxyServer


class StubRunTracker:
    """List-sink double for the Track B2 RunTracker.

    Records every method invocation as ``(method_name, kwargs_dict)`` so
    tests can assert on the RFC-003 envelope sequence the proxy emits.
    Post-OTLP refactor (R1-K) ``start_run`` returns a 16-lowercase-hex
    span id (the same shape :class:`RunTracker.start_run` returns).
    """

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        self._next_id = 0

    def start_run(self, **kwargs: Any) -> str:
        self._next_id += 1
        span_id = f"{self._next_id:016x}"
        self.calls.append({"method": "start_run", "span_id": span_id, **kwargs})
        return span_id

    def event(self, **kwargs: Any) -> None:
        self.calls.append({"method": "event", **kwargs})

    def complete_run(self, **kwargs: Any) -> None:
        self.calls.append({"method": "complete_run", **kwargs})

    def fail_run(self, **kwargs: Any) -> None:
        self.calls.append({"method": "fail_run", **kwargs})

    def methods(self) -> List[str]:
        return [c["method"] for c in self.calls]


def _build_server(tracker: StubRunTracker | None = None) -> LiteLLMProxyServer:
    """Construct a server without actually binding a port."""
    server = LiteLLMProxyServer(api_key="test-key", run_tracker=tracker)
    server.bound_port = 0  # signal "do not call start()" but base_url() still works
    return server


def _client(server: LiteLLMProxyServer) -> TestClient:
    """Wrap the server's FastAPI app in a TestClient (no network)."""
    return TestClient(server._app)  # noqa: SLF001 — test access is intentional


# ---------------------------------------------------------------------------
# health / models
# ---------------------------------------------------------------------------


def test_health_endpoint_returns_models() -> None:
    """``GET /v1/models`` MUST return 200 with a non-empty data list."""
    client = _client(_build_server())
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    assert isinstance(body["data"], list) and len(body["data"]) > 0
    assert all("id" in m for m in body["data"])


# ---------------------------------------------------------------------------
# request validation
# ---------------------------------------------------------------------------


def test_messages_endpoint_validates_required_fields() -> None:
    """Missing ``model`` MUST surface as a 4xx error response."""
    client = _client(_build_server())
    resp = client.post("/v1/messages", json={"messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code >= 400
    assert resp.status_code < 500
    body = resp.json()
    assert "detail" in body or "error" in body


# ---------------------------------------------------------------------------
# happy path: non-streaming
# ---------------------------------------------------------------------------


def test_messages_endpoint_emits_run_start_and_complete(monkeypatch: pytest.MonkeyPatch) -> None:
    """A successful non-stream call MUST emit ``start_run`` then ``complete_run``."""
    canned: Dict[str, Any] = {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "hello operator"}],
        "model": "claude-sonnet-4-6",
        "stop_reason": "end_turn",
    }

    async def _fake_acompletion(**kwargs: Any) -> Dict[str, Any]:
        return canned

    monkeypatch.setattr(proxy_module.litellm, "acompletion", _fake_acompletion)

    tracker = StubRunTracker()
    client = _client(_build_server(tracker))
    resp = client.post(
        "/v1/messages",
        json={
            "model": "claude-sonnet-4-6",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status_code == 200
    assert resp.json() == canned
    assert tracker.methods() == ["start_run", "complete_run"]
    start = tracker.calls[0]
    end = tracker.calls[1]
    assert start["name"] == "litellm:claude-sonnet-4-6"
    # OTLP refactor: run_type rides on the metadata dict so it lands on
    # the span as ``llmnb.run_type``; the legacy top-level field is gone.
    assert start["run_type"] == "llm"
    # OTel canonical status codes per OTLP/JSON spec.
    assert end["status"] == "STATUS_CODE_OK"
    # ``run_id`` arg the proxy forwards to ``complete_run`` MUST equal
    # the spanId start_run returned (same OTLP span across the lifecycle).
    assert end["run_id"] == start["span_id"]


# ---------------------------------------------------------------------------
# streaming pass-through
# ---------------------------------------------------------------------------


def test_streaming_response_passes_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """Streaming MUST flow all chunks through and emit one event per chunk."""
    chunks: List[Dict[str, Any]] = [
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "h"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "i"}},
        {"type": "message_stop"},
    ]

    async def _fake_stream() -> AsyncIterator[Dict[str, Any]]:
        for c in chunks:
            yield c

    async def _fake_acompletion(**kwargs: Any) -> AsyncIterator[Dict[str, Any]]:
        return _fake_stream()

    monkeypatch.setattr(proxy_module.litellm, "acompletion", _fake_acompletion)

    tracker = StubRunTracker()
    client = _client(_build_server(tracker))
    with client.stream(
        "POST",
        "/v1/messages",
        json={
            "model": "claude-sonnet-4-6",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    ) as resp:
        assert resp.status_code == 200
        body = b"".join(resp.iter_bytes())

    body_text = body.decode("utf-8")
    for chunk in chunks:
        assert json.dumps(chunk) in body_text

    methods = tracker.methods()
    assert methods[0] == "start_run"
    assert methods.count("event") == 3
    assert methods[-1] == "complete_run"
    # OTel canonical status code for a clean stream completion.
    assert tracker.calls[-1]["status"] == "STATUS_CODE_OK"


# ---------------------------------------------------------------------------
# error path
# ---------------------------------------------------------------------------


def test_litellm_error_emits_run_error_envelope(monkeypatch: pytest.MonkeyPatch) -> None:
    """A LiteLLM-side raise MUST emit ``complete_run(status="error")`` and return Anthropic-shaped error."""

    class FakeAPIError(Exception):
        status_code = 503

        def __init__(self) -> None:
            super().__init__("upstream provider down")

    async def _fake_acompletion(**kwargs: Any) -> Any:
        raise FakeAPIError()

    monkeypatch.setattr(proxy_module.litellm, "acompletion", _fake_acompletion)

    tracker = StubRunTracker()
    client = _client(_build_server(tracker))
    resp = client.post(
        "/v1/messages",
        json={
            "model": "claude-sonnet-4-6",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status_code == 503
    body = resp.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "FakeAPIError"
    assert "upstream provider down" in body["error"]["message"]

    # The proxy now closes errored runs via fail_run (which emits the
    # OTel exception semconv attributes) instead of complete_run with
    # an error envelope.
    methods = tracker.methods()
    assert methods == ["start_run", "fail_run"]
    assert tracker.calls[-1]["error"]["exception.type"] == "FakeAPIError"
    assert "upstream provider down" in tracker.calls[-1]["error"]["exception.message"]
