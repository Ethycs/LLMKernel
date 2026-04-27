"""Track B3 / RFC-006 v2 contract tests for the custom-message dispatcher.

Exercises :class:`llm_kernel.custom_messages.CustomMessageDispatcher`
against a stub kernel mirroring the IPython surface the dispatcher
reads (``iopub_socket``, ``session.send``,
``shell.comm_manager.register_target``).

Post-RFC-006 supersession the wire form is:

* Family A (run lifecycle) emits ONLY the OTLP span at
  ``application/vnd.rts.run+json`` -- no envelope MIME alongside.
* Comm target is ``llmnb.rts.v2`` exclusively.
* Comm payloads are the thin v2 envelope ``{type, payload,
  correlation_id?}``; the receiver-side ``message_type`` form is
  reconstructed from the thin envelope by the dispatcher.
"""

from __future__ import annotations

import logging
import secrets
import threading
import uuid
from typing import Any, Callable, Dict, List, Tuple
from unittest.mock import MagicMock

import pytest

from llm_kernel.custom_messages import (
    DEFAULT_BUFFER_SIZE, DEFAULT_COMM_TARGET, MIME_ENVELOPE, MIME_RUN,
    CustomMessageDispatcher,
)
from llm_kernel.run_envelope import DIRECTION_EXTENSION_TO_KERNEL, make_envelope


class StubSession:
    def __init__(self) -> None:
        self.sent: List[Tuple[str, Dict[str, Any]]] = []
        self._lock = threading.Lock()

    def send(self, socket: Any, msg_type: str, **kwargs: Any) -> None:
        with self._lock:
            self.sent.append((msg_type, dict(kwargs)))


class StubCommManager:
    def __init__(self) -> None:
        self.targets: Dict[str, Callable[[Any, Dict[str, Any]], None]] = {}

    def register_target(self, name: str, callback: Callable[..., None]) -> None:
        self.targets[name] = callback

    def unregister_target(self, name: str, callback: Callable[..., None]) -> None:
        self.targets.pop(name, None)


class StubKernel:
    """Mirror the surface :class:`CustomMessageDispatcher` reads."""

    def __init__(self) -> None:
        self.session = StubSession()
        self.iopub_socket = MagicMock(name="iopub_socket")
        self.shell = MagicMock()
        self.shell.comm_manager = StubCommManager()
        self._parent_header: Dict[str, Any] = {}


def _run_env(message_type: str, run_id: str | None = None) -> Dict[str, Any]:
    """Build a minimally-valid OTLP-shaped Family A envelope."""
    rid = run_id or secrets.token_hex(8)
    if message_type == "run.start":
        payload: Dict[str, Any] = {
            "spanId": rid, "traceId": secrets.token_hex(16),
            "parentSpanId": None, "name": "notify",
            "kind": "SPAN_KIND_INTERNAL",
            "startTimeUnixNano": "1745588938412000000",
            "endTimeUnixNano": None,
            "attributes": [
                {"key": "llmnb.run_type", "value": {"stringValue": "tool"}},
            ],
            "events": [], "links": [],
            "status": {"code": "STATUS_CODE_UNSET", "message": ""},
        }
    elif message_type == "run.event":
        payload = {
            "spanId": rid,
            "event": {
                "timeUnixNano": "1745588938412000000",
                "name": "log",
                "attributes": [],
            },
        }
    else:
        payload = {
            "spanId": rid,
            "endTimeUnixNano": "1745588938412000000",
            "status": {"code": "STATUS_CODE_OK", "message": ""},
            "attributes": [],
        }
    return make_envelope(message_type, payload, correlation_id=rid)


def _layout_env() -> Dict[str, Any]:
    return make_envelope(
        "layout.update",
        {"snapshot_version": 1, "tree": {"id": "root", "type": "workspace", "children": []}},
        correlation_id=str(uuid.uuid4()),
    )


def _new() -> tuple[CustomMessageDispatcher, StubKernel]:
    kernel = StubKernel()
    return CustomMessageDispatcher(kernel), kernel


def test_run_start_emits_display_data_with_display_id() -> None:
    """RFC-006 §1: Family A emits ONLY ``application/vnd.rts.run+json``.

    The legacy v1 envelope MIME is deprecated and MUST NOT be present
    in fresh kernel output.  The OTLP span itself is self-describing.
    """
    dispatcher, kernel = _new()
    rid = secrets.token_hex(8)
    dispatcher.emit(_run_env("run.start", rid))
    assert len(kernel.session.sent) == 1
    msg_type, kwargs = kernel.session.sent[0]
    assert msg_type == "display_data"
    content = kwargs["content"]
    # display_id is the OTLP spanId, used by Jupyter to update the cell.
    assert content["transient"] == {"display_id": rid}
    assert MIME_RUN in content["data"]
    # RFC-006 §1: dual MIME emission was deprecated at v2.0 and
    # removed before v2.1; the v2 kernel emits ONLY the OTLP span.
    assert MIME_ENVELOPE not in content["data"]
    assert content["data"][MIME_RUN]["spanId"] == rid


def test_default_comm_target_is_v2() -> None:
    """RFC-006 §2: the Comm target name is ``llmnb.rts.v2``.

    Major-version bumps (v2 -> v3) change the target name as the
    handshake mechanism per RFC-006 §9 "Cross-family invariants".
    """
    assert DEFAULT_COMM_TARGET == "llmnb.rts.v2"


def test_run_event_emits_update_display_data() -> None:
    dispatcher, kernel = _new()
    rid = secrets.token_hex(8)
    dispatcher.emit(_run_env("run.start", rid))
    dispatcher.emit(_run_env("run.event", rid))
    assert [m for m, _ in kernel.session.sent] == ["display_data", "update_display_data"]
    assert kernel.session.sent[1][1]["content"]["transient"] == {"display_id": rid}


def test_run_complete_emits_update_display_data() -> None:
    dispatcher, kernel = _new()
    rid = secrets.token_hex(8)
    dispatcher.emit(_run_env("run.start", rid))
    dispatcher.emit(_run_env("run.complete", rid))
    msg_type, kwargs = kernel.session.sent[-1]
    assert msg_type == "update_display_data"
    payload = kwargs["content"]["data"][MIME_RUN]
    # OTel canonical status code for a successful close.
    assert payload["status"]["code"] == "STATUS_CODE_OK"
    assert payload["spanId"] == rid


def test_unknown_message_type_when_no_comm_buffers_then_flushes() -> None:
    """Buffered envelopes flush on Comm attach as RFC-006 §3 thin v2 form."""
    dispatcher, kernel = _new()
    dispatcher.start()
    env = _layout_env()
    dispatcher.emit(env)
    assert kernel.session.sent == []  # no Comm yet
    comm = MagicMock(name="comm")
    kernel.shell.comm_manager.targets[DEFAULT_COMM_TARGET](comm, {"content": {}})
    comm.send.assert_called_once()
    sent = comm.send.call_args[0][0]
    # RFC-006 §3 thin envelope shape: {type, payload, correlation_id?}.
    # ``layout.update`` is not a request/response pair so
    # correlation_id is omitted on egress.
    assert sent["type"] == "layout.update"
    assert sent["payload"] == env["payload"]
    assert "correlation_id" not in sent
    assert "direction" not in sent
    assert "timestamp" not in sent
    assert "rfc_version" not in sent
    comm.on_msg.assert_called_once()


def test_request_response_pair_preserves_correlation_id() -> None:
    """RFC-006 §5 / §3: ``agent_graph.*`` keep ``correlation_id`` on egress."""
    dispatcher, kernel = _new()
    dispatcher.start()
    cid = str(uuid.uuid4())
    env = make_envelope(
        "agent_graph.response",
        {"nodes": [], "edges": [], "truncated": False},
        correlation_id=cid,
    )
    comm = MagicMock(name="comm")
    kernel.shell.comm_manager.targets[DEFAULT_COMM_TARGET](comm, {"content": {}})
    dispatcher.emit(env)
    sent = comm.send.call_args[0][0]
    assert sent["type"] == "agent_graph.response"
    assert sent["correlation_id"] == cid


def test_notebook_metadata_family_f_envelope_emits() -> None:
    """RFC-006 §8 Family F: ``notebook.metadata`` rides the Comm thin form."""
    dispatcher, kernel = _new()
    dispatcher.start()
    env = make_envelope(
        "notebook.metadata",
        {
            "mode": "snapshot",
            "snapshot_version": 1,
            "snapshot": {
                "schema_version": "1.0.0",
                "session_id": "abc",
                "event_log": {"version": 1, "runs": []},
            },
            "trigger": "save",
        },
        correlation_id="abc:1",
    )
    comm = MagicMock(name="comm")
    kernel.shell.comm_manager.targets[DEFAULT_COMM_TARGET](comm, {"content": {}})
    dispatcher.emit(env)
    sent = comm.send.call_args[0][0]
    assert sent["type"] == "notebook.metadata"
    assert sent["payload"]["mode"] == "snapshot"
    assert sent["payload"]["trigger"] == "save"
    assert sent["payload"]["snapshot_version"] == 1


def test_buffer_overflow_drops_oldest_with_warning(caplog: pytest.LogCaptureFixture) -> None:
    dispatcher, _ = _new()
    with caplog.at_level(logging.WARNING, logger="llm_kernel.custom_messages"):
        for _ in range(DEFAULT_BUFFER_SIZE + 2):
            dispatcher.emit(_layout_env())
    assert len(dispatcher._buffer) == DEFAULT_BUFFER_SIZE  # noqa: SLF001
    assert sum("buffer overflow" in r.getMessage() for r in caplog.records) >= 2


def test_inbound_routing_to_registered_handler_thin_v2() -> None:
    """Inbound thin v2 form ``{type, payload}`` routes to the right handler."""
    dispatcher, kernel = _new()
    dispatcher.start()
    received: List[Dict[str, Any]] = []
    dispatcher.register_handler("operator.action", received.append)
    kernel.shell.comm_manager.targets[DEFAULT_COMM_TARGET](
        MagicMock(), {"content": {}}
    )
    # RFC-006 §3 thin v2 envelope: type + payload (correlation_id
    # optional and irrelevant for non-paired messages).
    inbound = {
        "type": "operator.action",
        "payload": {
            "action_type": "approval_response",
            "parameters": {"request_id": "r"},
        },
    }
    dispatcher._on_comm_msg({"content": {"data": inbound}})  # noqa: SLF001
    assert len(received) == 1
    assert received[0]["payload"]["action_type"] == "approval_response"


def test_inbound_legacy_v1_envelope_still_accepted() -> None:
    """Receivers MAY still accept the legacy v1 envelope during transition."""
    dispatcher, kernel = _new()
    dispatcher.start()
    received: List[Dict[str, Any]] = []
    dispatcher.register_handler("operator.action", received.append)
    kernel.shell.comm_manager.targets[DEFAULT_COMM_TARGET](
        MagicMock(), {"content": {}}
    )
    inbound = make_envelope(
        "operator.action",
        {"action_type": "approval_response", "parameters": {"request_id": "r"}},
        correlation_id=str(uuid.uuid4()),
        direction=DIRECTION_EXTENSION_TO_KERNEL,
    )
    dispatcher._on_comm_msg({"content": {"data": inbound}})  # noqa: SLF001
    assert len(received) == 1
    assert received[0]["payload"]["action_type"] == "approval_response"


def test_inbound_unknown_type_is_dropped_with_warning(caplog: pytest.LogCaptureFixture) -> None:
    dispatcher, _ = _new()
    seen: List[Dict[str, Any]] = []
    dispatcher.register_handler("operator.action", seen.append)
    with caplog.at_level(logging.WARNING, logger="llm_kernel.custom_messages"):
        # RFC-006 W4: inbound type values outside the catalog are
        # logged and dropped (V1 fail-closed).
        dispatcher._on_comm_msg(  # noqa: SLF001
            {"content": {"data": {"type": "not.a.real.type", "payload": {}}}}
        )
    assert seen == []
    assert any(
        "validation" in r.getMessage() or "no registered handlers" in r.getMessage()
        for r in caplog.records
    )


def test_emit_validates_envelope() -> None:
    dispatcher, kernel = _new()
    with pytest.raises(ValueError):
        dispatcher.emit({"message_type": "run.start"})
    assert kernel.session.sent == []


def test_concurrent_emits_serialize() -> None:
    dispatcher, kernel = _new()
    n = 16
    barrier = threading.Barrier(n)

    def worker() -> None:
        barrier.wait()
        dispatcher.emit(_run_env("run.start"))

    threads = [threading.Thread(target=worker) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(kernel.session.sent) == n
    assert all(m == "display_data" for m, _ in kernel.session.sent)


def test_register_handler_rejects_unknown_type() -> None:
    dispatcher, _ = _new()
    with pytest.raises(ValueError):
        dispatcher.register_handler("not.a.real.type", lambda env: None)
