"""K-CM Family F: inbound ``notebook.metadata`` ``mode:"hydrate"`` handler.

Per RFC-006 §8 (v2.0.2) the extension ships the persisted
``metadata.rts`` snapshot to the kernel on file-open.  The dispatcher's
inbound handler MUST:

1. Validate single-shot per RFC-006 §9 hydrate exclusivity (a second
   hydrate envelope is rejected with a wire-failure LogRecord).
2. Call ``MetadataWriter.hydrate(snapshot)``.
3. Drive ``DriftDetector.compare(...)`` and append events back to the
   writer.
4. Call ``AgentSupervisor.respawn_from_config(...)``.
5. Emit a ``mode:"snapshot"`` ``trigger:"hydrate_complete"`` confirmation.
"""

from __future__ import annotations

import logging
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


def _make_snapshot() -> Dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "session_id": "sess-1",
        "snapshot_version": 7,
        "config": {
            "version": 1,
            "recoverable": {
                "kernel": {"blob_threshold_bytes": 65536},
                "agents": [
                    {"agent_id": "alpha", "zone_id": "z1", "task": "demo"},
                ],
                "mcp_servers": [],
            },
            "volatile": {
                "kernel": {"rfc_006_version": "2.0.0"},
                "agents": [],
                "mcp_servers": [],
            },
        },
        "event_log": {"version": 1, "runs": []},
        "layout": {
            "version": 1,
            "tree": {"id": "root", "type": "workspace", "children": []},
        },
        "agents": {"version": 1, "nodes": [], "edges": []},
        "blobs": {},
        "drift_log": [],
    }


def _new() -> tuple[CustomMessageDispatcher, _StubKernel]:
    kernel = _StubKernel()
    return CustomMessageDispatcher(kernel, heartbeat_interval_sec=300.0), kernel


def _wire_collaborators(
    dispatcher: CustomMessageDispatcher,
    snapshot_for_confirmation: Dict[str, Any],
    drift_events: List[Dict[str, Any]] | None = None,
) -> tuple[MagicMock, MagicMock, MagicMock]:
    writer = MagicMock()
    writer.snapshot.return_value = snapshot_for_confirmation
    supervisor = MagicMock()
    supervisor.respawn_from_config.return_value = {"alpha": "running"}
    detector = MagicMock()
    detector.compare.return_value = drift_events or []
    dispatcher.set_metadata_writer(writer)
    dispatcher.set_agent_supervisor(supervisor)
    dispatcher.set_drift_detector(detector)
    dispatcher.set_current_volatile_provider(
        lambda: {"rfc_006_version": "2.0.0"}
    )
    return writer, supervisor, detector


def test_hydrate_calls_writer_and_supervisor_and_emits_confirmation() -> None:
    """End-to-end: hydrate -> writer.hydrate + supervisor.respawn -> echo."""
    dispatcher, kernel = _new()
    confirmation_snapshot = _make_snapshot()
    confirmation_snapshot["snapshot_version"] = 8
    writer, supervisor, _ = _wire_collaborators(
        dispatcher, confirmation_snapshot,
    )

    dispatcher.start()
    try:
        comm = MagicMock(name="comm")
        kernel.shell.comm_manager.targets[DEFAULT_COMM_TARGET](
            comm, {"content": {}}
        )
        snap = _make_snapshot()
        dispatcher._on_comm_msg(  # noqa: SLF001
            {"content": {"data": {
                "type": "notebook.metadata",
                "payload": {
                    "mode": "hydrate",
                    "snapshot_version": snap["snapshot_version"],
                    "snapshot": snap,
                    "trigger": "open",
                },
            }}}
        )
        writer.hydrate.assert_called_once_with(snap)
        supervisor.respawn_from_config.assert_called_once_with(
            snap["config"]["recoverable"]["agents"],
        )

        # Confirmation envelope: notebook.metadata mode=snapshot
        # trigger=hydrate_complete with the post-hydrate snapshot.
        confirmations = [
            args[0][0] for args in comm.send.call_args_list
            if isinstance(args[0][0], dict)
            and args[0][0].get("type") == "notebook.metadata"
            and args[0][0].get("payload", {}).get("trigger") == "hydrate_complete"
        ]
        assert len(confirmations) == 1
        payload = confirmations[0]["payload"]
        assert payload["mode"] == "snapshot"
        assert payload["snapshot_version"] == 8
        assert payload["snapshot"]["schema_version"] == "1.0.0"
    finally:
        dispatcher.stop()


def test_hydrate_drives_drift_detector_with_current_volatile() -> None:
    """DriftDetector.compare receives persisted snapshot + current_kernel kwarg."""
    dispatcher, kernel = _new()
    drift_event = {
        "detected_at": "2026-04-26T10:00:00.000Z",
        "field_path": "config.volatile.kernel.rfc_006_version",
        "previous_value": "2.0.0",
        "current_value": "2.0.1",
        "severity": "warn",
        "operator_acknowledged": False,
    }
    writer, _, detector = _wire_collaborators(
        dispatcher, _make_snapshot(), drift_events=[drift_event],
    )

    dispatcher.start()
    try:
        comm = MagicMock(name="comm")
        kernel.shell.comm_manager.targets[DEFAULT_COMM_TARGET](
            comm, {"content": {}}
        )
        snap = _make_snapshot()
        dispatcher._on_comm_msg(  # noqa: SLF001
            {"content": {"data": {
                "type": "notebook.metadata",
                "payload": {
                    "mode": "hydrate",
                    "snapshot_version": snap["snapshot_version"],
                    "snapshot": snap,
                    "trigger": "open",
                },
            }}}
        )
        detector.compare.assert_called_once()
        args, kwargs = detector.compare.call_args
        assert args[0] is snap
        assert kwargs.get("current_kernel") == {"rfc_006_version": "2.0.0"}
        # Drift events get appended through writer.append_drift_event.
        writer.append_drift_event.assert_called_once()
    finally:
        dispatcher.stop()


def test_second_hydrate_is_rejected_with_wire_failure_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """RFC-006 §9 hydrate exclusivity: at most one hydrate per session."""
    dispatcher, kernel = _new()
    writer, supervisor, _ = _wire_collaborators(
        dispatcher, _make_snapshot(),
    )

    dispatcher.start()
    try:
        kernel.shell.comm_manager.targets[DEFAULT_COMM_TARGET](
            MagicMock(), {"content": {}}
        )
        snap = _make_snapshot()
        envelope_data = {
            "type": "notebook.metadata",
            "payload": {
                "mode": "hydrate",
                "snapshot_version": snap["snapshot_version"],
                "snapshot": snap,
                "trigger": "open",
            },
        }
        dispatcher._on_comm_msg({"content": {"data": envelope_data}})  # noqa: SLF001
        # Writer/supervisor were each called once on the first hydrate.
        assert writer.hydrate.call_count == 1
        assert supervisor.respawn_from_config.call_count == 1

        with caplog.at_level(logging.WARNING, logger="llm_kernel.custom_messages"):
            dispatcher._on_comm_msg({"content": {"data": envelope_data}})  # noqa: SLF001
        # No additional hydrate / respawn calls.
        assert writer.hydrate.call_count == 1
        assert supervisor.respawn_from_config.call_count == 1
        # A wire-failure record is emitted.
        wire_fail_records = [
            r for r in caplog.records
            if "hydrate exclusivity" in r.getMessage()
            or getattr(r, "event.name", None) == "wire-failure"
        ]
        assert wire_fail_records
    finally:
        dispatcher.stop()


def test_patch_mode_is_rejected_with_wire_failure_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """RFC-006 §8: V1 rejects ``mode:"patch"``."""
    dispatcher, kernel = _new()
    _wire_collaborators(dispatcher, _make_snapshot())
    dispatcher.start()
    try:
        kernel.shell.comm_manager.targets[DEFAULT_COMM_TARGET](
            MagicMock(), {"content": {}}
        )
        with caplog.at_level(logging.WARNING, logger="llm_kernel.custom_messages"):
            dispatcher._on_comm_msg(  # noqa: SLF001
                {"content": {"data": {
                    "type": "notebook.metadata",
                    "payload": {"mode": "patch", "patch": []},
                }}}
            )
        assert any(
            "patch" in r.getMessage() and "rejected" in r.getMessage()
            for r in caplog.records
        )
    finally:
        dispatcher.stop()


def test_inbound_snapshot_mode_is_logged_and_ignored(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The kernel never receives ``mode:"snapshot"`` inbound; log + ignore."""
    dispatcher, kernel = _new()
    writer, supervisor, _ = _wire_collaborators(
        dispatcher, _make_snapshot(),
    )
    dispatcher.start()
    try:
        kernel.shell.comm_manager.targets[DEFAULT_COMM_TARGET](
            MagicMock(), {"content": {}}
        )
        with caplog.at_level(logging.INFO, logger="llm_kernel.custom_messages"):
            dispatcher._on_comm_msg(  # noqa: SLF001
                {"content": {"data": {
                    "type": "notebook.metadata",
                    "payload": {"mode": "snapshot", "snapshot": {}},
                }}}
            )
        writer.hydrate.assert_not_called()
        supervisor.respawn_from_config.assert_not_called()
        assert any(
            "logged-and-ignored" in r.getMessage() for r in caplog.records
        )
    finally:
        dispatcher.stop()
