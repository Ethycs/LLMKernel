"""V1 Kernel Gap Closure — K-CM gaps G4 and G5 (consolidated).

Closes the two K-CM (kernel custom-messages dispatcher) gaps from the
V1 Kernel Gap Closure plan:

* **G4** — RFC-006 §7.1 ``kernel.shutdown_request`` MUST set the
  host-bound ``threading.Event`` so the read loop in ``pty_mode``
  exits into its final-snapshot ``finally`` block. Socket-EOF
  remains the V1.0.0 fallback per RFC-006 §7.1.
* **G5** — RFC-006 §6 ``operator.action`` with
  ``action_type == "drift_acknowledged"`` MUST flip
  ``operator_acknowledged = True`` on the matching
  ``metadata_writer._drift_log[*]`` entry by calling
  ``MetadataWriter.acknowledge_drift(field_path, detected_at)``.
  Both wire paths (the direct route and the BSP-003 §3 ``submit_intent``
  envelope with ``intent_kind: "acknowledge_drift"``) round-trip.

Atom anchors: ``docs/atoms/protocols/operator-action.md``,
``docs/atoms/protocols/submit-intent-envelope.md``,
``docs/atoms/contracts/metadata-writer.md``.
"""

from __future__ import annotations

import logging
import threading
import uuid
from typing import Any, Callable, Dict, List, Tuple
from unittest.mock import MagicMock

import pytest

from llm_kernel.custom_messages import DEFAULT_COMM_TARGET, CustomMessageDispatcher
from llm_kernel.mcp_server import OperatorBridgeServer
from llm_kernel.metadata_writer import MetadataWriter


# ---------------------------------------------------------------------------
# Test fixtures: minimal stub kernel + dispatcher factory
# ---------------------------------------------------------------------------


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


def _make_dispatcher() -> Tuple[CustomMessageDispatcher, _StubKernel]:
    kernel = _StubKernel()
    # Long heartbeat so the test never observes a heartbeat envelope.
    return (
        CustomMessageDispatcher(kernel, heartbeat_interval_sec=300.0),
        kernel,
    )


class _MockMetadataWriter:
    """Mock writer recording acknowledge_drift calls (K-MW surface)."""

    def __init__(self, return_value: bool = True) -> None:
        self.calls: List[Tuple[str, str]] = []
        self.return_value: bool = return_value

    def acknowledge_drift(self, field_path: str, detected_at: str) -> bool:
        self.calls.append((field_path, detected_at))
        return self.return_value


def _drift_envelope(field_path: str, detected_at: str) -> Dict[str, Any]:
    """Build an outer ``operator.action`` envelope per the atom shape."""
    return {
        "type": "operator.action",
        "payload": {
            "action_type": "drift_acknowledged",
            "parameters": {
                "field_path": field_path,
                "detected_at": detected_at,
            },
        },
    }


# ---------------------------------------------------------------------------
# G4 — kernel.shutdown_request handler
# ---------------------------------------------------------------------------


def test_kernel_shutdown_request_sets_shutdown_event() -> None:
    """G4: arrival of ``kernel.shutdown_request`` MUST set the bound event.

    Per RFC-006 §7.1 + RFC-008 §4 step 6, the dispatcher's inbound
    handler signals the host-bound ``threading.Event`` so the read
    loop exits cleanly into its final-snapshot ``finally`` block.
    """
    dispatcher, kernel = _make_dispatcher()
    shutdown_event = threading.Event()
    dispatcher.set_shutdown_event(shutdown_event)
    dispatcher.start()
    try:
        # Open a Comm so handlers are reachable.
        kernel.shell.comm_manager.targets[DEFAULT_COMM_TARGET](
            MagicMock(), {"content": {}}
        )
        dispatcher._on_comm_msg(  # noqa: SLF001
            {"content": {"data": {
                "type": "kernel.shutdown_request",
                "payload": {"reason": "user_requested"},
            }}}
        )
        assert shutdown_event.is_set(), (
            "kernel.shutdown_request handler did not set the bound event"
        )
    finally:
        dispatcher.stop()


def test_kernel_shutdown_request_with_reason_logs_marker(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """G4: ``reason`` MUST be logged with ``event.name=kernel.shutdown_requested``.

    Per RFC-006 §7.1 the reason is informational; we surface it via a
    structured log record so tape capture / OTLP data-plane retains it.
    """
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
                    "payload": {"reason": "fatal_error"},
                }}}
            )
        matched: List[logging.LogRecord] = [
            r for r in caplog.records
            if getattr(r, "event.name", None) == "kernel.shutdown_requested"
        ]
        assert matched, (
            "expected one record with event.name=kernel.shutdown_requested"
        )
        assert getattr(matched[0], "llmnb.shutdown_reason", None) == "fatal_error"
    finally:
        dispatcher.stop()


# ---------------------------------------------------------------------------
# G5 — drift_acknowledged write-back (direct route + submit_intent route)
# ---------------------------------------------------------------------------


def test_drift_acknowledged_calls_writer_acknowledge_drift() -> None:
    """G5: ``drift_acknowledged`` MUST call ``MetadataWriter.acknowledge_drift``.

    With the matching ``field_path`` and ``detected_at`` from the
    envelope per RFC-006 §6 / RFC-005 drift_log.
    """
    bridge = OperatorBridgeServer(
        agent_id="test-agent",
        zone_id="test-zone",
        trace_id=str(uuid.uuid4()),
    )
    writer = _MockMetadataWriter(return_value=True)
    bridge.metadata_writer = writer
    bridge._route_operator_action(_drift_envelope(  # noqa: SLF001
        field_path="config.volatile.kernel.model_default",
        detected_at="2026-04-28T12:34:56.789Z",
    ))
    assert writer.calls == [(
        "config.volatile.kernel.model_default",
        "2026-04-28T12:34:56.789Z",
    )]


def test_drift_acknowledged_with_unknown_path_returns_safe_noop(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """G5: unknown ``field_path`` MUST NOT crash; writer returns False; log records.

    The writer's ``acknowledge_drift`` is a documented safe-no-op on
    miss (returns ``False``); the dispatcher's INFO log MUST still
    fire with ``matched=False`` so operators see the attempt.
    """
    bridge = OperatorBridgeServer(
        agent_id="test-agent",
        zone_id="test-zone",
        trace_id=str(uuid.uuid4()),
    )
    # The real K-MW writer with no matching entry is the canonical
    # "unknown path" surface; ``acknowledge_drift`` returns False
    # without raising per metadata_writer.py:715.
    writer = MetadataWriter(autosave_interval_sec=999.0)
    writer.append_drift_event(
        field_path="config.volatile.kernel.model_default",
        previous_value="x", current_value="y", severity="info",
    )
    bridge.metadata_writer = writer
    with caplog.at_level(logging.INFO, logger="llm_kernel.mcp"):
        # No exception MUST escape.
        bridge._route_operator_action(_drift_envelope(  # noqa: SLF001
            field_path="config.volatile.kernel.does.not.exist",
            detected_at="2026-04-28T12:34:56.789Z",
        ))
    matching = [
        r for r in caplog.records
        if "drift_acknowledged" in r.getMessage()
    ]
    assert matching, "expected operator.drift_acknowledged log record"
    assert any(getattr(r, "matched", None) is False for r in matching), (
        "expected matched=False on unknown-path write-back"
    )


def test_drift_acknowledged_via_submit_intent_envelope() -> None:
    """G5 alt path: BSP-003 ``submit_intent`` MUST round-trip to acknowledge_drift.

    Per ``docs/atoms/protocols/submit-intent-envelope.md`` and
    ``MetadataWriter._BSP003_INTENT_KINDS``, ``acknowledge_drift`` is
    a registered intent kind; submitting via the intent envelope MUST
    flip the same ``operator_acknowledged`` flag the direct route
    flips.
    """
    writer = MetadataWriter(autosave_interval_sec=999.0)
    event = writer.append_drift_event(
        field_path="config.volatile.kernel.rfc_001_version",
        previous_value="1.0.0",
        current_value="1.1.0",
        severity="warn",
    )
    detected_at = event["detected_at"]

    result = writer.submit_intent({
        "type": "operator.action",
        "payload": {
            "action_type": "zone_mutate",
            "intent_kind": "acknowledge_drift",
            "intent_id": str(uuid.uuid4()),
            "parameters": {
                "field_path": "config.volatile.kernel.rfc_001_version",
                "detected_at": detected_at,
            },
        },
    })
    assert result["applied"] is True, (
        f"submit_intent did not apply acknowledge_drift: {result!r}"
    )
    assert result["error_code"] is None
    snap = writer.snapshot()
    matched = [
        e for e in snap["drift_log"]
        if e["field_path"] == "config.volatile.kernel.rfc_001_version"
    ]
    assert matched and matched[0]["operator_acknowledged"] is True, (
        "submit_intent route did not flip operator_acknowledged"
    )
