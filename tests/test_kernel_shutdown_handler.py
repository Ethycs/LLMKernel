"""K-CM G4: ``kernel.shutdown_request`` inbound handler.

Per RFC-006 §7.1 + RFC-008 §4 step 6 the extension's graceful-shutdown
signal is the ``kernel.shutdown_request`` envelope.  The dispatcher's
inbound handler MUST set the host-bound ``threading.Event`` so the read
loop in ``pty_mode`` exits cleanly into its final-snapshot finally
block.  The reason field is informational only.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Dict, List
from unittest.mock import MagicMock

import pytest

from llm_kernel.custom_messages import DEFAULT_COMM_TARGET, CustomMessageDispatcher


class _StubCommManager:
    def __init__(self) -> None:
        self.targets: Dict[str, Callable[..., None]] = {}

    def register_target(self, name: str, callback: Callable[..., None]) -> None:
        self.targets[name] = callback

    def unregister_target(self, name: str, callback: Callable[..., None]) -> None:
        self.targets.pop(name, None)


class _StubKernel:
    def __init__(self) -> None:
        self.session = MagicMock()
        self.iopub_socket = MagicMock(name="iopub_socket")
        self.shell = MagicMock()
        self.shell.comm_manager = _StubCommManager()
        self._parent_header: Dict[str, Any] = {}


def _make_dispatcher() -> tuple[CustomMessageDispatcher, _StubKernel]:
    kernel = _StubKernel()
    # Long heartbeat so the test never observes a heartbeat envelope.
    return CustomMessageDispatcher(kernel, heartbeat_interval_sec=300.0), kernel


def test_shutdown_request_sets_bound_event() -> None:
    """The handler MUST set the bound shutdown event."""
    dispatcher, kernel = _make_dispatcher()
    shutdown_event = threading.Event()
    dispatcher.set_shutdown_event(shutdown_event)
    dispatcher.start()
    try:
        # Inbound envelopes arrive via the registered Comm target.
        kernel.shell.comm_manager.targets[DEFAULT_COMM_TARGET](
            MagicMock(), {"content": {}}
        )
        dispatcher._on_comm_msg(  # noqa: SLF001
            {"content": {"data": {
                "type": "kernel.shutdown_request",
                "payload": {"reason": "operator_close"},
            }}}
        )
        assert shutdown_event.is_set()
    finally:
        dispatcher.stop()


def test_shutdown_request_logs_reason_with_event_name(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Reason MUST be logged with ``event.name=kernel.shutdown_requested``."""
    dispatcher, kernel = _make_dispatcher()
    shutdown_event = threading.Event()
    dispatcher.set_shutdown_event(shutdown_event)
    dispatcher.start()
    try:
        kernel.shell.comm_manager.targets[DEFAULT_COMM_TARGET](
            MagicMock(), {"content": {}}
        )
        with caplog.at_level(logging.INFO, logger="llm_kernel.custom_messages"):
            dispatcher._on_comm_msg(  # noqa: SLF001
                {"content": {"data": {
                    "type": "kernel.shutdown_request",
                    "payload": {"reason": "extension_deactivate"},
                }}}
            )
        # The structured log carries the event.name + reason in extra.
        matched: List[logging.LogRecord] = [
            r for r in caplog.records
            if getattr(r, "event.name", None) == "kernel.shutdown_requested"
        ]
        assert matched, "expected one record with event.name kernel.shutdown_requested"
        assert getattr(matched[0], "llmnb.shutdown_reason", None) == "extension_deactivate"
    finally:
        dispatcher.stop()


def test_shutdown_request_without_bound_event_warns_and_returns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """No bound event -> warn and rely on EOF fallback (RFC-006 §7.1)."""
    dispatcher, kernel = _make_dispatcher()
    # No set_shutdown_event call.
    dispatcher.start()
    try:
        kernel.shell.comm_manager.targets[DEFAULT_COMM_TARGET](
            MagicMock(), {"content": {}}
        )
        with caplog.at_level(logging.WARNING, logger="llm_kernel.custom_messages"):
            dispatcher._on_comm_msg(  # noqa: SLF001
                {"content": {"data": {
                    "type": "kernel.shutdown_request",
                    "payload": {"reason": "restart"},
                }}}
            )
        assert any(
            "EOF fallback" in r.getMessage() for r in caplog.records
        )
    finally:
        dispatcher.stop()


def test_shutdown_request_handler_is_registered_via_start() -> None:
    """``start()`` MUST register the kernel.shutdown_request handler."""
    dispatcher, _ = _make_dispatcher()
    dispatcher.start()
    try:
        # ``register_handler`` is the canonical surface; the built-in
        # registration installs at least one handler under this type.
        with dispatcher._lock:  # noqa: SLF001
            handlers = dispatcher._handlers.get("kernel.shutdown_request", [])  # noqa: SLF001
        assert handlers, "kernel.shutdown_request handler not auto-registered"
    finally:
        dispatcher.stop()
