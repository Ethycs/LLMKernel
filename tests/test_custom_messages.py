"""Stage 2 Track B3 contract tests for the custom-message dispatcher.

Exercises :class:`llm_kernel.custom_messages.CustomMessageDispatcher`
against a stub kernel mirroring the IPython surface the dispatcher
reads (``iopub_socket``, ``session.send``,
``shell.comm_manager.register_target``).
"""

from __future__ import annotations

import logging
import threading
import uuid
from typing import Any, Callable, Dict, List, Tuple
from unittest.mock import MagicMock

import pytest

from llm_kernel.custom_messages import (
    DEFAULT_BUFFER_SIZE, MIME_ENVELOPE, MIME_RUN, CustomMessageDispatcher,
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
    rid = run_id or str(uuid.uuid4())
    if message_type == "run.start":
        payload: Dict[str, Any] = {
            "id": rid, "trace_id": str(uuid.uuid4()), "parent_run_id": None,
            "name": "notify", "run_type": "tool",
            "start_time": "2026-04-25T14:32:18.412Z",
            "inputs": {}, "tags": [], "metadata": {},
        }
    elif message_type == "run.event":
        payload = {"run_id": rid, "event_type": "log", "data": {}, "timestamp": "x"}
    else:
        payload = {"run_id": rid, "end_time": "x", "outputs": {}, "error": None, "status": "success"}
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
    dispatcher, kernel = _new()
    rid = str(uuid.uuid4())
    dispatcher.emit(_run_env("run.start", rid))
    assert len(kernel.session.sent) == 1
    msg_type, kwargs = kernel.session.sent[0]
    assert msg_type == "display_data"
    content = kwargs["content"]
    assert content["transient"] == {"display_id": rid}
    assert MIME_RUN in content["data"] and MIME_ENVELOPE in content["data"]
    assert content["data"][MIME_RUN]["id"] == rid


def test_run_event_emits_update_display_data() -> None:
    dispatcher, kernel = _new()
    rid = str(uuid.uuid4())
    dispatcher.emit(_run_env("run.start", rid))
    dispatcher.emit(_run_env("run.event", rid))
    assert [m for m, _ in kernel.session.sent] == ["display_data", "update_display_data"]
    assert kernel.session.sent[1][1]["content"]["transient"] == {"display_id": rid}


def test_run_complete_emits_update_display_data() -> None:
    dispatcher, kernel = _new()
    rid = str(uuid.uuid4())
    dispatcher.emit(_run_env("run.start", rid))
    dispatcher.emit(_run_env("run.complete", rid))
    msg_type, kwargs = kernel.session.sent[-1]
    assert msg_type == "update_display_data"
    payload = kwargs["content"]["data"][MIME_RUN]
    assert payload["status"] == "success" and payload["run_id"] == rid


def test_unknown_message_type_when_no_comm_buffers_then_flushes() -> None:
    dispatcher, kernel = _new()
    dispatcher.start()
    env = _layout_env()
    dispatcher.emit(env)
    assert kernel.session.sent == []  # no Comm yet
    comm = MagicMock(name="comm")
    kernel.shell.comm_manager.targets["llmnb.rts.v1"](comm, {"content": {}})
    comm.send.assert_called_once_with(env)
    comm.on_msg.assert_called_once()


def test_buffer_overflow_drops_oldest_with_warning(caplog: pytest.LogCaptureFixture) -> None:
    dispatcher, _ = _new()
    with caplog.at_level(logging.WARNING, logger="llm_kernel.custom_messages"):
        for _ in range(DEFAULT_BUFFER_SIZE + 2):
            dispatcher.emit(_layout_env())
    assert len(dispatcher._buffer) == DEFAULT_BUFFER_SIZE  # noqa: SLF001
    assert sum("buffer overflow" in r.getMessage() for r in caplog.records) >= 2


def test_inbound_routing_to_registered_handler() -> None:
    dispatcher, kernel = _new()
    dispatcher.start()
    received: List[Dict[str, Any]] = []
    dispatcher.register_handler("operator.action", received.append)
    kernel.shell.comm_manager.targets["llmnb.rts.v1"](MagicMock(), {"content": {}})
    inbound = make_envelope(
        "operator.action",
        {"action_type": "approval_response", "parameters": {"request_id": "r"}},
        correlation_id=str(uuid.uuid4()), direction=DIRECTION_EXTENSION_TO_KERNEL,
    )
    dispatcher._on_comm_msg({"content": {"data": inbound}})  # noqa: SLF001
    assert len(received) == 1
    assert received[0]["payload"]["action_type"] == "approval_response"


def test_inbound_unknown_type_is_dropped_with_warning(caplog: pytest.LogCaptureFixture) -> None:
    dispatcher, _ = _new()
    seen: List[Dict[str, Any]] = []
    dispatcher.register_handler("operator.action", seen.append)
    with caplog.at_level(logging.WARNING, logger="llm_kernel.custom_messages"):
        dispatcher._on_comm_msg(  # noqa: SLF001
            {"content": {"data": {"message_type": "not.a.real.type"}}}
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
