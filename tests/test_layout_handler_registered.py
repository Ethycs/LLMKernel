"""K-CM Family B: ``layout.edit`` inbound handler registration.

Per RFC-006 §4 the kernel applies layout edits and echoes a fresh
``layout.update``.  The dispatcher consumes K-MW's
:meth:`MetadataWriter.apply_layout_edit` +
:meth:`MetadataWriter.emit_layout_update` interface (locked in the
mega-round brief).  Tests mock the writer to keep the slice isolated.
"""

from __future__ import annotations

from typing import Any, Callable, Dict
from unittest.mock import MagicMock

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


def _new() -> tuple[CustomMessageDispatcher, _StubKernel]:
    kernel = _StubKernel()
    return CustomMessageDispatcher(kernel, heartbeat_interval_sec=300.0), kernel


def test_layout_edit_calls_writer_and_emits_layout_update() -> None:
    """layout.edit -> apply_layout_edit -> emit_layout_update -> Comm send."""
    dispatcher, kernel = _new()
    writer = MagicMock()
    writer.apply_layout_edit.return_value = 17  # new snapshot_version
    new_layout_payload = {
        "snapshot_version": 17,
        "tree": {"id": "root", "type": "workspace", "children": []},
    }
    writer.emit_layout_update.return_value = new_layout_payload
    dispatcher.set_metadata_writer(writer)

    dispatcher.start()
    try:
        comm = MagicMock(name="comm")
        kernel.shell.comm_manager.targets[DEFAULT_COMM_TARGET](
            comm, {"content": {}}
        )
        # Inbound edit envelope (extension -> kernel).
        dispatcher._on_comm_msg(  # noqa: SLF001
            {"content": {"data": {
                "type": "layout.edit",
                "payload": {
                    "operation": "rename_node",
                    "parameters": {"node_id": "n1", "new_name": "renamed"},
                },
            }}}
        )

        writer.apply_layout_edit.assert_called_once_with(
            operation="rename_node",
            parameters={"node_id": "n1", "new_name": "renamed"},
        )
        writer.emit_layout_update.assert_called_once_with()

        # Echo: a Comm send with type=layout.update and the writer's
        # returned payload (RFC-006 §3 thin v2 form).
        assert comm.send.call_count >= 1
        sent_layouts = [
            args[0][0] for args in comm.send.call_args_list
            if isinstance(args[0][0], dict) and args[0][0].get("type") == "layout.update"
        ]
        assert len(sent_layouts) == 1
        assert sent_layouts[0]["payload"] == new_layout_payload
    finally:
        dispatcher.stop()


def test_layout_edit_without_writer_drops_with_warning() -> None:
    """No writer bound -> drop and continue (RFC-006 W4 fail-closed style)."""
    dispatcher, kernel = _new()
    dispatcher.start()
    try:
        comm = MagicMock(name="comm")
        kernel.shell.comm_manager.targets[DEFAULT_COMM_TARGET](
            comm, {"content": {}}
        )
        dispatcher._on_comm_msg(  # noqa: SLF001
            {"content": {"data": {
                "type": "layout.edit",
                "payload": {"operation": "add_zone", "parameters": {}},
            }}}
        )
        # Without a writer, no layout.update is emitted.
        sent_types = [
            args[0][0].get("type") for args in comm.send.call_args_list
            if isinstance(args[0][0], dict)
        ]
        assert "layout.update" not in sent_types
    finally:
        dispatcher.stop()


def test_layout_edit_handler_is_registered_via_start() -> None:
    """``start()`` MUST register the layout.edit handler."""
    dispatcher, _ = _new()
    dispatcher.start()
    try:
        with dispatcher._lock:  # noqa: SLF001
            handlers = dispatcher._handlers.get("layout.edit", [])  # noqa: SLF001
        assert handlers, "layout.edit handler not auto-registered"
    finally:
        dispatcher.stop()
