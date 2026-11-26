from __future__ import annotations

import asyncio
import fnmatch
import json
import logging
import os
import re
import signal
import sys
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ipykernel.kernelapp import IPKernelApp
from ipykernel.kernelbase import Kernel as JupyterKernelBase, kernel_protocol_version

from llm_kernel.capture.custom_messages import MIME_RUN
from llm_kernel.capture.metadata_writer._serialize import SecretRejected
from llm_kernel.capture.run_envelope import (
    DIRECTION_EXTENSION_TO_KERNEL,
    make_envelope,
)
from llm_kernel.capture.socket_writer import sanitize_outbound_record
from llm_kernel.kernel_facade import Kernel as FacadeKernel
from llm_kernel.magics.magic_generators import GeneratorError, dispatch_generator
from llm_kernel.mcp.auth_handlers import apply_auth_command
from llm_kernel.notebook import notebook_format
from llm_kernel.notebook.cell_manager import (
    CellManager,
    CellManagerPreconditionError,
)
from llm_kernel.notebook.file_actions import apply_export
from llm_kernel.text.cell_text import (
    CellParseError,
    ParsedCell,
    parse_cell,
    split_at_breaks,
)
from llm_kernel.text.magic_registry import (
    CELL_MAGICS,
    LINE_MAGICS,
    is_reserved_name,
    parse_kv_args,
)
from llm_kernel.wire import versions as _versions

logger: logging.Logger = logging.getLogger("llm_kernel.jupyter_kernel")

ENV_WORKDIR: str = "LLMNB_WORKDIR"
ENV_NOTEBOOK_PATH: str = "LLMNB_NOTEBOOK_PATH"
ENV_AUTOSAVE_NOTEBOOK: str = "LLMNB_AUTOSAVE_NOTEBOOK"
ENV_UNATTENDED: str = "LLMNB_UNATTENDED"
ENV_PROXY_MODE: str = "LLMNB_JUPYTER_PROXY"
ENV_SPAWN_SEND_INITIAL_TURN: str = "LLMNB_SPAWN_SEND_INITIAL_TURN"

MIME_LLMNB: str = "application/vnd.llmnb+json"
MIME_LLMNB_RESULT: str = "application/vnd.llmnb.result+json"

K20_UNBOUND_AGENT_CELL: str = "K20"
K23_UNKNOWN_AGENT: str = "K23"
K32_RESERVED_MAGIC_NAME: str = "K32"
K36_CELL_CONTAMINATED: str = "K36"
K42_VALIDATION_FAILED: str = "K42"
K43_SECRET_REJECTED: str = "K43"

_RESERVED_INERT_KINDS: frozenset[str] = frozenset(
    {"tool", "artifact", "control", "native"}
)
_BODY_EXPECTING_MAGICS: frozenset[str] = frozenset(
    {"agent", "markdown", "scratch", "checkpoint", "expand"}
)
_AUTH_SUBCOMMANDS: Tuple[str, ...] = ("set", "rotate", "off", "verify", "status")

_CELL_MAGIC_KWARGS: Dict[str, Tuple[str, ...]] = {
    "spawn": ("task:", "endpoint:", "model:"),
    "agent": (),
    "markdown": (),
    "scratch": (),
    "checkpoint": ("covers:",),
    "endpoint": ("provider:", "model:", "api_key_env:"),
    "compare": ("endpoints:",),
    "section": ("title:", "id:"),
    "template": (),
    "expand": (),
    "import": ("file:", "format:"),
    "export": ("path:", "format:", "overwrite:"),
    "tool": (),
    "artifact": (),
    "native": (),
}

_MAGIC_DOCS: Dict[str, Dict[str, str]] = {
    "@@agent": {
        "syntax": "@@agent <agent_id>\\n<message>",
        "summary": "Bind the cell to <agent_id> and send the body as a continuation turn.",
        "errors": "K20 (unbound agent), K23 (unknown agent), K42 (empty message)",
    },
    "@@spawn": {
        "syntax": '@@spawn <agent_id> task:"<description>" [endpoint:<name>] [model:<id>]',
        "summary": "Spawn an agent in the active zone and bind the cell to it.",
        "errors": "K32 (reserved magic name as agent_id), K42 (missing agent_id or task)",
    },
    "@@markdown": {
        "syntax": "@@markdown\\n<markdown body>",
        "summary": "Operator prose cell. Never binds to an agent; renders as Markdown.",
        "errors": "K30 (multiple cell kinds)",
    },
    "@@scratch": {
        "syntax": "@@scratch\\n<notes>",
        "summary": "Operator scratchpad cell excluded from default ContextPacker walks.",
        "errors": "K30 (multiple cell kinds)",
    },
    "@@checkpoint": {
        "syntax": "@@checkpoint [covers:[c_1,c_2]]\\n<summary>",
        "summary": "Operator-authored summary substituting for covered turns in agent context.",
        "errors": "K30 (multiple cell kinds)",
    },
    "@@endpoint": {
        "syntax": "@@endpoint <name> [provider:<p>] [model:<m>] [api_key_env:<ENV>]",
        "summary": "Register a named model endpoint in metadata.rts.config.endpoints.",
        "errors": "K42 (missing name), K43 (inline secret rejected; use api_key_env)",
    },
    "@@section": {
        "syntax": '@@section <title> [id:"sec_..."]',
        "summary": "Create a notebook section via an overlay commit (create_section op).",
        "errors": "K42 (missing title), K90-K95 (overlay rejection)",
    },
    "@@template": {
        "syntax": "@@template <name> [key:value ...]",
        "summary": "Generator magic: instantiate metadata.rts.config.templates[<name>] as cells with provenance.",
        "errors": "K30 (missing/unknown template or unresolved placeholder), K3I, K3J",
    },
    "@@expand": {
        "syntax": "@@expand\\n<cell 1>\\n@@break\\n<cell 2>",
        "summary": "Generator magic: split body on @@break and insert cells with generated_by provenance.",
        "errors": "K30 (empty or invalid fragment), K3I, K3J",
    },
    "@@import": {
        "syntax": '@@import <relpath> [format:"llmnb"|"magic"|"ipynb"]',
        "summary": "Generator magic: import cells from a workspace .llmnb, .magic, or .ipynb file.",
        "errors": "K30 (missing file, traversal outside workspace, cycle), K3O (parse/format error)",
    },
    "@@export": {
        "syntax": '@@export path:"<relpath>" [format:"llmnb"|"magic"|"ipynb"] [overwrite:true|false]',
        "summary": "Atomically serialize the current notebook state to a workspace file.",
        "errors": "K3M (outside workspace), K3N (exists without overwrite:true), K3O (I/O or format error)",
    },
    "@@compare": {
        "syntax": "@@compare endpoints:a,b\\n<prompt>",
        "summary": "Stub (V1.5+): compare one prompt across multiple declared endpoints.",
        "errors": "K42 (not yet implemented in V1)",
    },
    "@@tool": {
        "syntax": "@@tool\\n<body>",
        "summary": "Reserved cell kind (V2+): preserved verbatim and rendered inert in V1.",
        "errors": "None (forward-compatible reserved kind)",
    },
    "@@artifact": {
        "syntax": "@@artifact\\n<body>",
        "summary": "Reserved cell kind (V2+): ArtifactRef lens cell; inert in V1.",
        "errors": "None (forward-compatible reserved kind)",
    },
    "@@native": {
        "syntax": "@@native\\n<body>",
        "summary": "Reserved cell kind (V2+): runtime directive cell; inert in V1.",
        "errors": "None (forward-compatible reserved kind)",
    },
    "@@break": {
        "syntax": "@@break",
        "summary": "Cell separator consumed by split_at_breaks; resets cell-local flags.",
        "errors": "None",
    },
    "@pin": {
        "syntax": "@pin",
        "summary": "Add 'pinned' to cell.flags so ContextPacker retains this cell.",
        "errors": "None",
    },
    "@unpin": {
        "syntax": "@unpin",
        "summary": "Remove 'pinned' from cell.flags.",
        "errors": "None",
    },
    "@exclude": {
        "syntax": "@exclude",
        "summary": "Add 'excluded' to cell.flags so ContextPacker omits this cell.",
        "errors": "None",
    },
    "@include": {
        "syntax": "@include",
        "summary": "Remove 'excluded' from cell.flags.",
        "errors": "None",
    },
    "@mark": {
        "syntax": "@mark <kind>",
        "summary": "Change the cell's parsed kind to <kind>.",
        "errors": "K34 (unknown or incompatible target kind)",
    },
    "@affinity": {
        "syntax": "@affinity <endpoint1,endpoint2,...>",
        "summary": "Record preferred endpoint stack on the cell.",
        "errors": "None",
    },
    "@handoff": {
        "syntax": "@handoff <agent_id>",
        "summary": "Switch the active continuation agent binding for subsequent cells to <agent_id>.",
        "errors": "K42 (missing agent_id)",
    },
    "@status": {
        "syntax": "@status",
        "summary": "Display live kernel versions, zone state, active agents, hash mode, and queue metrics.",
        "errors": "None",
    },
    "@auth": {
        "syntax": "@auth set <pin> | rotate <new_pin> | off | verify | status",
        "summary": "Manage HMAC cell-magic injection defense (pin length >= 8; never stored in notebook).",
        "errors": "K38 (pin too short/invalid), K39 (collision), K3B (fingerprint mismatch), K3X (unknown subcommand)",
    },
    "@branch": {
        "syntax": "@branch <new_agent_id> [from:<source_agent>] [at:<turn_id>]",
        "summary": "Fork an agent at its head turn or a specified ancestor turn.",
        "errors": "K23 (unknown source agent), K32 (reserved name), K42 (invalid arguments)",
    },
    "@revert": {
        "syntax": "@revert <target_turn_id> [agent:<agent_id>]",
        "summary": "Move an agent's head_turn_id back to <target_turn_id> and stop its live process.",
        "errors": "K23 (unknown agent), K42 (missing or unknown target_turn_id)",
    },
    "@stop": {
        "syntax": "@stop [<agent_id>]",
        "summary": "Cleanly stop an agent's live process while preserving session state for resume.",
        "errors": "K23 (unknown agent), K42 (missing agent_id)",
    },
}


class _KernelExecutionError(RuntimeError):
    def __init__(self, k_code: str, message: str) -> None:
        super().__init__(f"{k_code}: {message}")
        self.k_code = k_code
        self.reason = message


class _JupyterWireSanitizer:
    def __init__(self) -> None:
        self._hash_mode_provider: Optional[Callable[[], bool]] = None
        self._k36_handler: Optional[
            Callable[[Dict[str, Any], Dict[str, Any]], None]
        ] = None

    def set_hash_mode_provider(
        self, provider: Optional[Callable[[], bool]]
    ) -> None:
        self._hash_mode_provider = provider

    def set_k36_handler(
        self,
        handler: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]],
    ) -> None:
        self._k36_handler = handler

    def enabled(self) -> bool:
        provider = self._hash_mode_provider
        if provider is None:
            return False
        try:
            return bool(provider())
        except Exception:
            return False

    def sanitize_frame(self, record: Dict[str, Any]) -> Dict[str, Any]:
        if not self.enabled() or not isinstance(record, dict):
            return record
        if record.get("type") == "notebook.metadata":
            return self._sanitize_metadata_frame(record)
        sanitized = sanitize_outbound_record(record)
        if sanitized is not record and self._k36_handler is not None:
            try:
                self._k36_handler(record, sanitized)
            except Exception:
                logger.exception("K36 handler raised during Jupyter wire sanitize")
        return sanitized

    def _sanitize_metadata_frame(self, record: Dict[str, Any]) -> Dict[str, Any]:
        payload = record.get("payload")
        if not isinstance(payload, dict):
            return record
        snapshot = payload.get("snapshot")
        if not isinstance(snapshot, dict):
            return record
        cells = snapshot.get("cells")
        saved_texts: Dict[str, str] = {}
        if isinstance(cells, dict):
            for cid, crec in cells.items():
                if isinstance(crec, dict) and isinstance(crec.get("text"), str):
                    saved_texts[cid] = crec["text"]
        sanitized = sanitize_outbound_record(record)
        if sanitized is record:
            return record
        if saved_texts:
            s_payload = dict(sanitized.get("payload") or {})
            s_snapshot = dict(s_payload.get("snapshot") or {})
            s_cells = dict(s_snapshot.get("cells") or {})
            for cid, orig_text in saved_texts.items():
                if cid in s_cells and isinstance(s_cells[cid], dict):
                    restored = dict(s_cells[cid])
                    restored["text"] = orig_text
                    s_cells[cid] = restored
            s_snapshot["cells"] = s_cells
            s_payload["snapshot"] = s_snapshot
            sanitized = dict(sanitized)
            sanitized["payload"] = s_payload
        return sanitized


class _JupyterKernelSession:
    def __init__(
        self,
        owner: "LLMNBKernel",
        sanitizer: _JupyterWireSanitizer,
        send_lock: threading.RLock,
    ) -> None:
        self._owner = owner
        self._writer = sanitizer
        self._send_lock = send_lock

    def send(
        self,
        stream: Any,
        msg_type: str,
        content: Optional[Dict[str, Any]] = None,
        parent: Optional[Dict[str, Any]] = None,
        ident: Any = None,
        buffers: Any = None,
        track: bool = False,
        header: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        if self._owner._silent_execution:
            return None
        session = self._owner.session
        if session is None or stream is None:
            return None
        out_content = content
        if isinstance(content, dict) and self._writer.enabled():
            data = content.get("data")
            if isinstance(data, dict) and MIME_RUN in data and isinstance(data[MIME_RUN], dict):
                sanitized_span = self._writer.sanitize_frame(data[MIME_RUN])
                if sanitized_span is not data[MIME_RUN]:
                    new_data = dict(data)
                    new_data[MIME_RUN] = sanitized_span
                    out_content = dict(content)
                    out_content["data"] = new_data
            else:
                out_content = self._writer.sanitize_frame(content)
        effective_parent = parent if parent else self._owner._current_parent_header()
        with self._send_lock:
            return session.send(
                stream,
                msg_type,
                content=out_content,
                parent=effective_parent,
                ident=ident,
                buffers=buffers,
                track=track,
                header=header,
                metadata=metadata,
            )


class _ClientComm:
    def __init__(
        self,
        *,
        comm_id: str,
        target_name: str,
        manager: "_JupyterCommManager",
    ) -> None:
        self.comm_id = comm_id
        self.target_name = target_name
        self._manager = manager
        self._closed = False

    def send(
        self,
        data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        buffers: Any = None,
    ) -> None:
        if self._closed:
            return
        self._manager._publish_comm_msg(
            "comm_msg",
            comm_id=self.comm_id,
            data=data if data is not None else {},
            metadata=metadata,
            buffers=buffers,
        )

    def close(
        self,
        data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        buffers: Any = None,
        deleting: bool = False,
    ) -> None:
        if self._closed:
            return
        self._closed = True
        self._manager._remove_client_comm(self.comm_id)
        if not deleting:
            self._manager._publish_comm_msg(
                "comm_close",
                comm_id=self.comm_id,
                data=data if data is not None else {},
                metadata=metadata,
                buffers=buffers,
            )


class _MultiplexedTargetComm:
    def __init__(self, target_name: str, manager: "_JupyterCommManager") -> None:
        self.target_name = target_name
        self._manager = manager
        self._on_msg: Optional[Callable[[Dict[str, Any]], None]] = None
        self._on_close: Optional[Callable[[Dict[str, Any]], None]] = None

    def on_msg(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        self._on_msg = callback

    def on_close(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        self._on_close = callback

    def send(
        self,
        data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        buffers: Any = None,
    ) -> None:
        payload = data if isinstance(data, dict) else {}
        self._manager.broadcast_target(
            self.target_name, payload, metadata=metadata, buffers=buffers,
        )

    def close(
        self,
        data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        buffers: Any = None,
    ) -> None:
        self._manager.close_target(
            self.target_name, data=data, metadata=metadata, buffers=buffers,
        )

    def deliver(self, msg: Dict[str, Any]) -> None:
        cb = self._on_msg
        if cb is not None:
            cb(msg)


class _JupyterCommManager:
    def __init__(
        self,
        owner: "LLMNBKernel",
        sanitizer: _JupyterWireSanitizer,
        send_lock: threading.RLock,
    ) -> None:
        self.kernel = owner
        self._sanitizer = sanitizer
        self._send_lock = send_lock
        self._lock = threading.RLock()
        self.targets: Dict[str, Callable[[Any, Dict[str, Any]], None]] = {}
        self.comms: Dict[str, _ClientComm] = {}
        self._mux_by_target: Dict[str, _MultiplexedTargetComm] = {}
        self._target_comms: Dict[str, Dict[str, _ClientComm]] = {}

    def register_target(
        self, target_name: str, callback: Callable[[Any, Dict[str, Any]], None]
    ) -> None:
        with self._lock:
            self.targets[target_name] = callback

    def unregister_target(
        self, target_name: str, _callback: Callable[[Any, Dict[str, Any]], None]
    ) -> None:
        with self._lock:
            self.targets.pop(target_name, None)
            self._mux_by_target.pop(target_name, None)

    def has_open_target(self, target_name: str) -> bool:
        with self._lock:
            return bool(self._target_comms.get(target_name))

    def _publish_comm_msg(
        self,
        msg_type: str,
        *,
        comm_id: str,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        buffers: Any = None,
        parent: Optional[Dict[str, Any]] = None,
    ) -> None:
        session = self.kernel.session
        stream = self.kernel.iopub_socket
        if session is None or stream is None:
            return
        safe_data = self._sanitizer.sanitize_frame(data) if isinstance(data, dict) else data
        content: Dict[str, Any] = {"comm_id": comm_id, "data": safe_data}
        effective_parent = parent if parent is not None else self.kernel._current_parent_header()
        ident = f"comm-{comm_id}".encode("ascii", errors="replace")
        with self._send_lock:
            session.send(
                stream,
                msg_type,
                content=content,
                metadata=metadata or {},
                parent=effective_parent,
                ident=ident,
                buffers=buffers,
            )

    def _remove_client_comm(self, comm_id: str) -> None:
        with self._lock:
            client = self.comms.pop(comm_id, None)
            if client is not None:
                bucket = self._target_comms.get(client.target_name)
                if bucket is not None:
                    bucket.pop(comm_id, None)
                    if not bucket:
                        self._target_comms.pop(client.target_name, None)

    def broadcast_target(
        self,
        target_name: str,
        data: Dict[str, Any],
        *,
        metadata: Optional[Dict[str, Any]] = None,
        buffers: Any = None,
    ) -> None:
        with self._lock:
            recipients = list((self._target_comms.get(target_name) or {}).values())
        for client in recipients:
            client.send(data, metadata=metadata, buffers=buffers)

    def close_target(
        self,
        target_name: str,
        *,
        data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        buffers: Any = None,
    ) -> None:
        with self._lock:
            clients = list((self._target_comms.pop(target_name, None) or {}).values())
            self._mux_by_target.pop(target_name, None)
            for client in clients:
                self.comms.pop(client.comm_id, None)
        for client in clients:
            client.close(data=data, metadata=metadata, buffers=buffers)

    def comm_open(self, _stream: Any, _ident: Any, msg: Dict[str, Any]) -> None:
        content = msg.get("content") or {}
        comm_id = content.get("comm_id")
        target_name = content.get("target_name")
        if not isinstance(comm_id, str) or not comm_id:
            logger.warning("comm_open missing comm_id; dropped")
            return
        if not isinstance(target_name, str) or not target_name:
            self._publish_comm_msg(
                "comm_close", comm_id=comm_id, data={}, parent=msg,
            )
            return
        with self._lock:
            callback = self.targets.get(target_name)
            if callback is None:
                refuse = True
                first_attach = False
                mux = None
            else:
                refuse = False
                client = _ClientComm(
                    comm_id=comm_id, target_name=target_name, manager=self,
                )
                self.comms[comm_id] = client
                bucket = self._target_comms.setdefault(target_name, {})
                first_attach = target_name not in self._mux_by_target
                if first_attach:
                    mux = _MultiplexedTargetComm(target_name, self)
                    self._mux_by_target[target_name] = mux
                else:
                    mux = self._mux_by_target[target_name]
                bucket[comm_id] = client
        if refuse:
            logger.warning(
                "comm_open refused for unregistered target %r (comm_id=%s)",
                target_name, comm_id,
            )
            self._publish_comm_msg(
                "comm_close",
                comm_id=comm_id,
                data={"error": f"unregistered comm target: {target_name}"},
                parent=msg,
            )
            return
        if first_attach and callback is not None and mux is not None:
            try:
                callback(mux, msg)
            except Exception:
                logger.exception("comm_open callback raised for target %r", target_name)
                self.close_target(target_name)

    def comm_msg(self, _stream: Any, _ident: Any, msg: Dict[str, Any]) -> None:
        content = msg.get("content") or {}
        comm_id = content.get("comm_id")
        if not isinstance(comm_id, str) or not comm_id:
            logger.warning("comm_msg missing comm_id; dropped")
            return
        with self._lock:
            client = self.comms.get(comm_id)
            mux = (
                self._mux_by_target.get(client.target_name)
                if client is not None
                else None
            )
        if client is None or mux is None:
            logger.warning("comm_msg for unknown comm_id=%r; dropped", comm_id)
            return
        try:
            mux.deliver(msg)
        except Exception:
            logger.exception("comm_msg handler raised for comm_id=%s", comm_id)

    def comm_close(self, _stream: Any, _ident: Any, msg: Dict[str, Any]) -> None:
        content = msg.get("content") or {}
        comm_id = content.get("comm_id")
        if not isinstance(comm_id, str) or not comm_id:
            return
        with self._lock:
            client = self.comms.pop(comm_id, None)
            if client is None:
                return
            bucket = self._target_comms.get(client.target_name)
            if bucket is not None:
                bucket.pop(comm_id, None)
                remaining = len(bucket)
                if remaining == 0:
                    self._target_comms.pop(client.target_name, None)
            else:
                remaining = 0
            client._closed = True
            mux = self._mux_by_target.get(client.target_name) if remaining == 0 else None
            if remaining == 0:
                self._mux_by_target.pop(client.target_name, None)
        if remaining == 0 and mux is not None:
            dispatcher = self.kernel._get_dispatcher_or_none()
            if dispatcher is not None:
                with dispatcher._lock:
                    if dispatcher._active_comm is mux:
                        dispatcher._active_comm = None
            if mux._on_close is not None:
                try:
                    mux._on_close(msg)
                except Exception:
                    logger.exception("comm_close callback raised")


class _JupyterShellAdapter:
    def __init__(self, comm_manager: _JupyterCommManager) -> None:
        self.comm_manager = comm_manager
        self.user_ns: Dict[str, Any] = {}


class _JupyterKernelShell:
    def __init__(self, owner: "LLMNBKernel") -> None:
        self._owner = owner
        self._send_lock = threading.RLock()
        self.sanitizer = _JupyterWireSanitizer()
        self.session = _JupyterKernelSession(owner, self.sanitizer, self._send_lock)
        self.comm_manager = _JupyterCommManager(owner, self.sanitizer, self._send_lock)
        self.shell = _JupyterShellAdapter(self.comm_manager)

    @property
    def iopub_socket(self) -> Any:
        return self._owner.iopub_socket

    @property
    def _parent_header(self) -> Dict[str, Any]:
        return self._owner._current_parent_header()


@dataclass
class _HistoryEntry:
    session: int
    line_num: int
    raw_input: str
    output: str


@dataclass
class _SegmentOutcome:
    cell_id: str
    kind: str
    summary: str
    data: Dict[str, Any]
    markdown_display: Optional[str] = None
    warnings: Tuple[str, ...] = ()


_AUTH_PIN_REDACT_RE: re.Pattern[str] = re.compile(
    r"^(@(?:[0-9a-fA-F]+:)?auth\s+(?:set|rotate)\s+)\S+(.*)$"
)


def _redact_auth_pins(text: str) -> str:
    if "auth" not in text:
        return text
    lines = text.splitlines()
    changed = False
    out: List[str] = []
    for ln in lines:
        m = _AUTH_PIN_REDACT_RE.match(ln)
        if m is not None:
            out.append(f"{m.group(1)}<redacted>{m.group(2)}")
            changed = True
        else:
            out.append(ln)
    if not changed:
        return text
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def _mint_section_id(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")[:24]
    tail = uuid.uuid4().hex[:6]
    return f"sec_{slug + '_' if slug else ''}{tail}"


def _is_expand_cell(code: str) -> bool:
    for ln in code.splitlines():
        s = ln.strip()
        if not s:
            continue
        return bool(re.match(r"^@@(?:[0-9a-fA-F]+:)?expand(?:\s|$)", s))
    return False


def _parse_cell_or_expand(text: str, parser_context: Any) -> ParsedCell:
    if not _is_expand_cell(text):
        return parse_cell(text, parser_context=parser_context)
    lines = text.splitlines()
    first_idx = next(i for i, ln in enumerate(lines) if ln.strip())
    header = parse_cell(lines[first_idx], parser_context=parser_context)
    if header.kind != "expand" or header.kind_was_default:
        header.body = "\n".join(lines[first_idx + 1:])
        return header
    header.body = "\n".join(lines[first_idx + 1:])
    return header


def _has_unclosed_quote(line: str) -> bool:
    quote: Optional[str] = None
    escaped = False
    for ch in line:
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if quote is None:
            if ch in ('"', "'"):
                quote = ch
        elif ch == quote:
            quote = None
    return quote is not None


def _has_unclosed_fence(text: str) -> bool:
    fence_count = 0
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            fence_count += 1
    return (fence_count % 2) == 1


class LLMNBKernel(JupyterKernelBase):
    implementation = "llmnb"
    language = "llmnb"

    help_links = [
        {"text": "LLMNB Magic Vocabulary", "url": "https://llmnb.dev/docs/atoms/concepts/magic"},
        {"text": "RFC-005 .llmnb File Format", "url": "https://llmnb.dev/docs/rfcs/RFC-005"},
        {"text": "RFC-006 Wire Format (v2)", "url": "https://llmnb.dev/docs/rfcs/RFC-006"},
    ]

    def __init__(
        self,
        *,
        workdir: Optional[Path] = None,
        notebook_path: Optional[Path] = None,
        autosave_notebook: Optional[bool] = None,
        facade_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        raw_workdir = workdir or os.environ.get(ENV_WORKDIR) or Path.cwd()
        self._workdir: Path = Path(raw_workdir).resolve()
        self._workdir.mkdir(parents=True, exist_ok=True)

        raw_nb = notebook_path or os.environ.get(ENV_NOTEBOOK_PATH)
        self._notebook_path: Optional[Path] = (
            Path(raw_nb).resolve() if raw_nb else None
        )
        if autosave_notebook is None:
            self._autosave_notebook: bool = (
                os.environ.get(ENV_AUTOSAVE_NOTEBOOK, "1") == "1"
            )
        else:
            self._autosave_notebook = bool(autosave_notebook)

        self._facade_config: Dict[str, Any] = dict(facade_config or {})
        self._zone_id: str = os.environ.get("LLMKERNEL_ZONE_ID", "default")
        self._silent_execution: bool = False
        self._active_agent_id: Optional[str] = None
        self._executing_agent_id: Optional[str] = None
        self._session_number: int = 1
        self._history: List[_HistoryEntry] = []
        self._shutdown_event = threading.Event()
        self._shutdown_watcher: Optional[threading.Thread] = None
        self._proxy_server: Any = None
        self._pending_restart: bool = False

        self._shell_bridge = _JupyterKernelShell(self)
        self.comm_manager = self._shell_bridge.comm_manager
        for msg_type in ("comm_open", "comm_msg", "comm_close"):
            self.shell_handlers[msg_type] = getattr(self.comm_manager, msg_type)

        self._facade_loop: Optional[asyncio.AbstractEventLoop] = None
        self._facade_thread: Optional[threading.Thread] = None
        self._facade: Optional[FacadeKernel] = None
        self._boot_facade()

        if self._notebook_path is not None and self._notebook_path.exists():
            self._hydrate_from_notebook_file(self._notebook_path)

    # -- Dynamic version/info properties (single-source versions.py) ----

    @property
    def implementation_version(self) -> str:
        return _versions.KERNEL_VERSION

    @property
    def language_version(self) -> str:
        return _versions.SCHEMA_VERSION

    @property
    def language_info(self) -> Dict[str, Any]:
        return {
            "name": "llmnb",
            "version": _versions.SCHEMA_VERSION,
            "mimetype": MIME_LLMNB,
            "file_extension": ".llmnb",
            "pygments_lexer": "markdown",
            "codemirror_mode": "markdown",
        }

    @property
    def banner(self) -> str:
        return (
            f"LLMNB Kernel v{_versions.KERNEL_VERSION} "
            f"(schema {_versions.SCHEMA_VERSION}, wire {_versions.WIRE_VERSION}, "
            f"comm {_versions.COMM_TARGET})"
        )

    @property
    def kernel_info(self) -> Dict[str, Any]:
        return {
            "protocol_version": kernel_protocol_version,
            "implementation": self.implementation,
            "implementation_version": self.implementation_version,
            "language_info": self.language_info,
            "banner": self.banner,
            "help_links": list(self.help_links),
            "agents": self._collect_agent_summaries(),
            "llmnb": {
                "kernel_version": _versions.KERNEL_VERSION,
                "package_version": _versions.PACKAGE_VERSION,
                "wire_version": _versions.WIRE_VERSION,
                "schema_version": _versions.SCHEMA_VERSION,
                "comm_target": _versions.COMM_TARGET,
                "capabilities": _versions.kernel_capabilities(),
                "rfc_versions": _versions.rfc_versions(),
                "zone_id": self._zone_id,
            },
        }

    @property
    def facade(self) -> FacadeKernel:
        if (self._facade is None or not self._facade.started) and self._pending_restart:
            self._pending_restart = False
            self._boot_facade()
        if self._facade is None or not self._facade.started:
            raise RuntimeError("LLMNBKernel facade is not running")
        return self._facade

    # -- Facade lifecycle -----------------------------------------------

    def _boot_facade(self) -> None:
        self._shutdown_event.clear()
        config = dict(self._facade_config)

        proxy_mode = os.environ.get(ENV_PROXY_MODE, "none").strip().lower()
        if proxy_mode == "litellm":
            from llm_kernel.providers import litellm_proxy as _proxy_mod

            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            server = _proxy_mod.LiteLLMProxyServer(
                api_key=api_key, host="127.0.0.1", port=0,
            )
            server.start()
            self._proxy_server = server
            config.setdefault("litellm_endpoint_url", server.base_url())

        loop = asyncio.new_event_loop()
        loop_thread = threading.Thread(
            target=loop.run_forever,
            name="llmnb-jupyter-facade-loop",
            daemon=True,
        )
        loop_thread.start()

        facade = FacadeKernel(
            workdir=self._workdir,
            config=config,
            loop=loop,
            kernel_shell=self._shell_bridge,
        )
        try:
            asyncio.run_coroutine_threadsafe(facade.start(), loop).result(timeout=60.0)
        except BaseException:
            self._stop_proxy()
            loop.call_soon_threadsafe(loop.stop)
            loop_thread.join(timeout=5.0)
            if not loop.is_running():
                loop.close()
            raise

        if proxy_mode == "passthrough":
            from llm_kernel.providers import anthropic_passthrough as _pt_mod

            server = _pt_mod.AnthropicPassthroughServer(
                run_tracker=facade.run_tracker, host="127.0.0.1", port=0,
            )
            server.start()
            self._proxy_server = server

        facade.dispatcher.set_shutdown_event(self._shutdown_event)
        facade.operator_bridge.stdin_responder = self._stdin_operator_responder
        self._shell_bridge.shell.user_ns["__llmnb_cell_manager__"] = CellManager(
            facade.metadata_writer
        )

        self._facade_loop = loop
        self._facade_thread = loop_thread
        self._facade = facade
        self._start_shutdown_watcher()

    def _start_shutdown_watcher(self) -> None:
        def _watch() -> None:
            self._shutdown_event.wait()
            facade = self._facade
            if facade is None or not facade.started:
                return
            io_loop = getattr(self, "io_loop", None)
            if io_loop is not None:
                try:
                    io_loop.add_callback(self._handle_comm_shutdown_signal)
                except Exception:
                    pass

        watcher = threading.Thread(
            target=_watch,
            name="llmnb-jupyter-shutdown-watch",
            daemon=True,
        )
        watcher.start()
        self._shutdown_watcher = watcher

    def _handle_comm_shutdown_signal(self) -> None:
        self._teardown_facade(persist_notebook=True)
        control_stream = getattr(self, "control_stream", None)
        if control_stream is not None and getattr(control_stream, "io_loop", None) is not None:
            control_stream.io_loop.add_callback(control_stream.io_loop.stop)
        io_loop = getattr(self, "io_loop", None)
        if io_loop is not None:
            io_loop.add_callback(io_loop.stop)

    def _stop_proxy(self) -> None:
        server = self._proxy_server
        self._proxy_server = None
        if server is not None:
            try:
                server.stop()
            except Exception:
                logger.exception("LLMNBKernel: proxy stop raised")

    def _teardown_facade(self, *, persist_notebook: bool = True) -> None:
        facade = self._facade
        loop = self._facade_loop
        loop_thread = self._facade_thread
        watcher = self._shutdown_watcher

        self._facade = None
        self._facade_loop = None
        self._facade_thread = None
        self._shutdown_watcher = None

        final_snapshot: Optional[Dict[str, Any]] = None
        if facade is not None and facade.started and persist_notebook and self._notebook_path is not None and self._autosave_notebook:
            try:
                final_snapshot = facade.metadata_writer.snapshot(trigger="save")
            except Exception:
                logger.exception("LLMNBKernel: pre-shutdown notebook snapshot failed")

        self._stop_proxy()

        if facade is not None and loop is not None:
            try:
                asyncio.run_coroutine_threadsafe(facade.shutdown(), loop).result(
                    timeout=30.0
                )
            except Exception:
                logger.exception("LLMNBKernel: facade shutdown raised")
            loop.call_soon_threadsafe(loop.stop)
            if loop_thread is not None:
                loop_thread.join(timeout=5.0)
            if not loop.is_running():
                loop.close()

        self._shutdown_event.set()
        if watcher is not None and watcher is not threading.current_thread():
            watcher.join(timeout=2.0)

        self.comm_manager.close_target(_versions.COMM_TARGET)

        if final_snapshot is not None and self._notebook_path is not None:
            self._persist_notebook_file(self._notebook_path, final_snapshot)

    def _get_dispatcher_or_none(self) -> Optional[Any]:
        facade = self._facade
        if facade is None or not facade.started:
            return None
        return facade.dispatcher

    def _current_parent_header(self) -> Dict[str, Any]:
        try:
            parent = self.get_parent("shell")
            if isinstance(parent, dict):
                return parent
        except Exception:
            pass
        return {}

    # -- Notebook file hydrate + atomic persist -------------------------

    def _hydrate_from_notebook_file(self, path: Path) -> None:
        try:
            fmt = notebook_format.detect_format(path)
            raw = path.read_text(encoding="utf-8")
            if fmt == "llmnb":
                doc = json.loads(raw)
            elif fmt == "magic":
                doc = notebook_format.magic_to_llmnb(raw)
            elif fmt == "ipynb":
                doc = notebook_format.ipynb_to_llmnb(json.loads(raw))
            else:
                logger.warning("LLMNBKernel: unrecognized notebook format at %s", path)
                return
            rts = (doc.get("metadata") or {}).get("rts")
            if not isinstance(rts, dict):
                return
            self.facade.dispatcher._handle_notebook_metadata(
                make_envelope(
                    "notebook.metadata",
                    {"mode": "hydrate", "snapshot": rts, "trigger": "jupyter.open"},
                    correlation_id=str(uuid.uuid4()),
                    direction=DIRECTION_EXTENSION_TO_KERNEL,
                )
            )
            cells = rts.get("cells") or {}
            if isinstance(cells, dict):
                for cid in self.facade.metadata_writer.get_cell_layout_order():
                    rec = cells.get(cid)
                    if isinstance(rec, dict) and isinstance(rec.get("bound_agent_id"), str):
                        self._active_agent_id = rec["bound_agent_id"]
        except Exception:
            logger.exception("LLMNBKernel: failed to hydrate from %s", path)

    def _persist_notebook_file(self, path: Path, snapshot: Dict[str, Any]) -> None:
        try:
            existing_meta: Dict[str, Any] = {}
            existing_cells: List[Dict[str, Any]] = []
            if path.exists():
                try:
                    loaded = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        if isinstance(loaded.get("metadata"), dict):
                            existing_meta = dict(loaded["metadata"])
                        if isinstance(loaded.get("cells"), list):
                            existing_cells = list(loaded["cells"])
                except Exception:
                    pass
            existing_meta.setdefault(
                "kernelspec", notebook_format.llmnb_kernelspec_metadata()
            )
            existing_meta.setdefault(
                "language_info", notebook_format.llmnb_language_info_metadata()
            )
            existing_meta["rts"] = snapshot
            cells_out = self._materialize_nbformat_cells(snapshot) or existing_cells
            doc = {
                "nbformat": 4,
                "nbformat_minor": 5,
                "metadata": existing_meta,
                "cells": cells_out,
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f"{path.name}.tmp")
            tmp.write_text(
                json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            os.replace(tmp, path)
        except Exception:
            logger.exception("LLMNBKernel: failed to persist notebook to %s", path)

    @staticmethod
    def _materialize_nbformat_cells(snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
        cells_map = snapshot.get("cells")
        if not isinstance(cells_map, dict) or not cells_map:
            return []
        layout_tree = (snapshot.get("layout") or {}).get("tree")
        ordered = notebook_format._layout_walk_ids(layout_tree)
        for cid in cells_map:
            if cid not in ordered:
                ordered.append(cid)
        out: List[Dict[str, Any]] = []
        for cid in ordered:
            rec = cells_map.get(cid)
            if not isinstance(rec, dict):
                continue
            text = rec.get("text", "")
            kind = rec.get("kind") or "agent"
            outputs = [
                o for o in (rec.get("outputs") or [])
                if isinstance(o, dict) and "output_type" in o
            ]
            is_md = kind == "markdown"
            out.append(
                {
                    "cell_type": "markdown" if is_md else "code",
                    "id": cid,
                    "source": text,
                    "metadata": {"rts": {"cell": {"kind": kind}}},
                    **({} if is_md else {"outputs": outputs, "execution_count": None}),
                }
            )
        return out

    # -- Stdin socket approval bridge -----------------------------------

    def _stdin_operator_responder(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        if self.comm_manager.has_open_target(_versions.COMM_TARGET):
            return None
        if not getattr(self, "_allow_stdin", False):
            if os.environ.get(ENV_UNATTENDED) == "1":
                if tool_name in ("request_approval", "propose"):
                    return {"decision": "reject", "scope_granted": "one_shot"}
                if tool_name == "ask":
                    return {"answer": ""}
            return None
        prompt = self._format_stdin_prompt(tool_name, arguments)
        try:
            raw_reply = self.raw_input(prompt)
        except Exception:
            logger.exception("LLMNBKernel: raw_input failed for tool %s", tool_name)
            return None
        answer = (raw_reply or "").strip()
        if tool_name == "ask":
            return {"answer": answer}
        if tool_name == "clarify":
            options = arguments.get("options") or []
            valid_ids = {
                str(opt.get("id"))
                for opt in options
                if isinstance(opt, dict) and opt.get("id") is not None
            }
            if answer in valid_ids:
                return {"selected_id": answer}
            default_id = arguments.get("default_id")
            if default_id:
                return {"selected_id": str(default_id)}
            if options and isinstance(options[0], dict) and options[0].get("id"):
                return {"selected_id": str(options[0]["id"])}
            return {"selected_id": answer or "default"}
        lowered = answer.lower()
        if lowered in ("y", "yes", "approve", "approved", "allow", "ok"):
            decision = "approve"
        elif lowered in ("n", "no", "reject", "rejected", "deny"):
            decision = "reject"
        else:
            decision = "defer"
        return {"decision": decision, "scope_granted": "one_shot"}

    @staticmethod
    def _format_stdin_prompt(tool_name: str, arguments: Dict[str, Any]) -> str:
        if tool_name == "ask":
            q = arguments.get("question") or arguments.get("prompt") or "Agent question"
            return f"[llmnb ask] {q}: "
        if tool_name == "clarify":
            q = arguments.get("question") or "Select option"
            opts = arguments.get("options") or []
            labels = ", ".join(
                str(o.get("id")) for o in opts if isinstance(o, dict) and o.get("id")
            )
            return f"[llmnb clarify] {q} ({labels}): "
        if tool_name == "propose":
            summary = (
                arguments.get("summary")
                or arguments.get("description")
                or arguments.get("action")
                or "Agent proposal"
            )
            return f"[llmnb propose] {summary} [approve/reject/defer]: "
        summary = (
            arguments.get("action")
            or arguments.get("reason")
            or arguments.get("summary")
            or "Approval requested"
        )
        return f"[llmnb approval] {summary} [approve/reject/defer]: "

    # -- Jupyter Shell: execute_request ---------------------------------

    def do_execute(
        self,
        code: str,
        silent: bool,
        store_history: bool = True,
        user_expressions: Optional[Dict[str, str]] = None,
        allow_stdin: bool = False,
        *,
        cell_meta: Optional[Dict[str, Any]] = None,
        cell_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        prev_silent = self._silent_execution
        prev_allow_stdin = getattr(self, "_allow_stdin", False)
        self._silent_execution = bool(silent)
        self._allow_stdin = bool(allow_stdin)
        try:
            return self._execute_impl(
                code=code or "",
                silent=bool(silent),
                store_history=bool(store_history),
                user_expressions=user_expressions,
                cell_meta=cell_meta,
                cell_id=cell_id,
            )
        finally:
            self._silent_execution = prev_silent
            self._allow_stdin = prev_allow_stdin
            self._executing_agent_id = None

    def _execute_impl(
        self,
        *,
        code: str,
        silent: bool,
        store_history: bool,
        user_expressions: Optional[Dict[str, str]],
        cell_meta: Optional[Dict[str, Any]],
        cell_id: Optional[str],
    ) -> Dict[str, Any]:
        exec_count = int(self.execution_count)
        if not code.strip():
            return {
                "status": "ok",
                "execution_count": exec_count,
                "payload": [],
                "user_expressions": self._evaluate_user_expressions(user_expressions),
            }

        base_cell_id = self._resolve_cell_id(cell_id, cell_meta, exec_count)
        has_break = (
            not _is_expand_cell(code)
            and any(line.strip() == "@@break" for line in code.splitlines())
        )
        segments = split_at_breaks(code) if has_break else [code]
        if not segments:
            return {
                "status": "ok",
                "execution_count": exec_count,
                "payload": [],
                "user_expressions": self._evaluate_user_expressions(user_expressions),
            }

        outcomes: List[_SegmentOutcome] = []
        for idx, segment in enumerate(segments):
            seg_cell_id = (
                base_cell_id if len(segments) == 1 else f"{base_cell_id}:{idx}"
            )
            try:
                outcome = self._execute_segment(seg_cell_id, segment, silent=silent)
                outcomes.append(outcome)
            except _KernelExecutionError as err:
                if store_history and not silent:
                    self._history.append(
                        _HistoryEntry(
                            session=self._session_number,
                            line_num=exec_count,
                            raw_input=_redact_auth_pins(code),
                            output=f"{err.k_code}: {err.reason}",
                        )
                    )
                tb = [f"{err.k_code}: {err.reason}"]
                if not silent:
                    self._send_iopub(
                        "error",
                        {
                            "ename": err.k_code,
                            "evalue": err.reason,
                            "traceback": tb,
                        },
                    )
                return {
                    "status": "error",
                    "execution_count": exec_count,
                    "ename": err.k_code,
                    "evalue": err.reason,
                    "traceback": tb,
                }

        summaries = [o.summary for o in outcomes if o.summary]
        combined_summary = "\n".join(summaries)

        if not silent:
            for outcome in outcomes:
                for warning in outcome.warnings:
                    self._send_iopub(
                        "stream", {"name": "stderr", "text": f"{warning}\n"}
                    )
                if outcome.markdown_display is not None:
                    self._send_iopub(
                        "display_data",
                        {
                            "data": {
                                "text/markdown": outcome.markdown_display,
                                "text/plain": outcome.markdown_display,
                            },
                            "metadata": {},
                        },
                    )
            if combined_summary:
                result_payload = (
                    outcomes[0].data
                    if len(outcomes) == 1
                    else {"segments": [o.data for o in outcomes]}
                )
                self._send_iopub(
                    "execute_result",
                    {
                        "execution_count": exec_count,
                        "data": {
                            "text/plain": combined_summary,
                            MIME_LLMNB_RESULT: result_payload,
                        },
                        "metadata": {},
                    },
                )

        if store_history and not silent:
            self._history.append(
                _HistoryEntry(
                    session=self._session_number,
                    line_num=exec_count,
                    raw_input=_redact_auth_pins(code),
                    output=combined_summary,
                )
            )

        return {
            "status": "ok",
            "execution_count": exec_count,
            "payload": [],
            "user_expressions": self._evaluate_user_expressions(user_expressions),
        }

    def _resolve_cell_id(
        self,
        cell_id: Optional[str],
        cell_meta: Optional[Dict[str, Any]],
        exec_count: int,
    ) -> str:
        if isinstance(cell_id, str) and cell_id.strip():
            return cell_id.strip()
        if isinstance(cell_meta, dict):
            for key in ("cellId", "cell_id"):
                val = cell_meta.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()
            rts_cell = (cell_meta.get("rts") or {}).get("cell")
            if isinstance(rts_cell, dict) and isinstance(rts_cell.get("id"), str):
                return rts_cell["id"]
        return f"cell-{exec_count}"

    def _send_iopub(self, msg_type: str, content: Dict[str, Any]) -> None:
        self._shell_bridge.session.send(
            self.iopub_socket, msg_type, content=content
        )

    def _execute_segment(
        self, cell_id: str, text: str, *, silent: bool
    ) -> _SegmentOutcome:
        facade = self.facade
        writer = facade.metadata_writer
        tracker = facade.run_tracker
        supervisor = facade.supervisor

        if writer.is_cell_contaminated(cell_id):
            self._emit_error_span(
                cell_id=cell_id,
                name="cell.execute",
                k_code=K36_CELL_CONTAMINATED,
                reason=f"cell {cell_id!r} is flagged contaminated; reset before re-executing",
            )
            raise _KernelExecutionError(
                K36_CELL_CONTAMINATED,
                f"cell {cell_id!r} is flagged contaminated; reset before re-executing",
            )

        ctx = writer.parser_context()
        try:
            parsed = _parse_cell_or_expand(text, ctx)
        except CellParseError as exc:
            self._emit_error_span(
                cell_id=cell_id,
                name="cell.parse",
                k_code=exc.code,
                reason=exc.reason,
            )
            raise _KernelExecutionError(exc.code, exc.reason) from exc

        warnings: List[str] = [
            f"{em.get('code', 'K35')}: {em.get('reason', '')}"
            for em in parsed.k_class_emissions
        ]

        prior = writer.get_cell_record(cell_id) or {}
        section_id = self._find_cell_section_id(writer, cell_id)
        span_meta: Dict[str, Any] = {
            "cell_id": cell_id,
            "cell_kind": parsed.kind,
        }
        if section_id:
            span_meta["section_id"] = section_id
        if parsed.k_class_emissions:
            span_meta["k_class_emissions"] = [
                em.get("code", "") for em in parsed.k_class_emissions
            ]

        run_id = tracker.start_run(
            name=f"cell.{parsed.kind}",
            run_type="chain",
            inputs={"cell_id": cell_id, "kind": parsed.kind},
            tags=[f"zone:{self._zone_id}", f"kind:{parsed.kind}"],
            metadata=span_meta,
        )
        writer.set_cell_execution_state(cell_id, True)
        persisted_text = _redact_auth_pins(text)
        try:
            writer.update_cell_record(
                cell_id,
                text=persisted_text,
                kind=parsed.kind,
                flags=parsed.flags,
                append_to_layout=True,
            )

            line_notes, line_data, pending_handoff, extra_cell_fields = (
                self._apply_line_magics(
                    cell_id=cell_id,
                    parsed=parsed,
                    writer=writer,
                    supervisor=supervisor,
                    prior_record=prior,
                )
            )
            if extra_cell_fields:
                writer.update_cell_record(
                    cell_id,
                    flags=parsed.flags,
                    extra=extra_cell_fields,
                    append_to_layout=False,
                )

            kind_summary, kind_data, md_display, bound_agent, extra_warnings = (
                self._dispatch_cell_kind(
                    cell_id=cell_id,
                    text=text,
                    parsed=parsed,
                    prior_record=prior,
                    writer=writer,
                    supervisor=supervisor,
                    has_line_magics=bool(parsed.line_magics or parsed.flags),
                )
            )
            warnings.extend(extra_warnings)

            if pending_handoff is not None:
                self._active_agent_id = pending_handoff

            summary_parts = [s for s in (*line_notes, kind_summary) if s]
            full_summary = "\n".join(summary_parts)

            merged_data: Dict[str, Any] = {
                "cell_id": cell_id,
                "kind": parsed.kind,
                "flags": sorted(parsed.flags),
                **line_data,
                **kind_data,
            }
            if bound_agent is not None:
                merged_data["bound_agent_id"] = bound_agent
            if pending_handoff is not None:
                merged_data["handoff_agent_id"] = pending_handoff
            if warnings:
                merged_data["warnings"] = list(warnings)

            tracker.complete_run(run_id, outputs=merged_data)
            closed_span = tracker.get_run(run_id).model_dump()

            cell_outputs: List[Dict[str, Any]] = [
                {
                    "output_type": "display_data",
                    "data": {MIME_RUN: closed_span},
                    "metadata": {},
                }
            ]
            if md_display is not None:
                cell_outputs.append(
                    {
                        "output_type": "display_data",
                        "data": {
                            "text/markdown": md_display,
                            "text/plain": md_display,
                        },
                        "metadata": {},
                    }
                )
            if full_summary and not silent:
                cell_outputs.append(
                    {
                        "output_type": "execute_result",
                        "execution_count": int(self.execution_count),
                        "data": {
                            "text/plain": full_summary,
                            MIME_LLMNB_RESULT: merged_data,
                        },
                        "metadata": {},
                    }
                )

            writer.update_cell_record(
                cell_id,
                kind=parsed.kind,
                bound_agent_id=bound_agent,
                outputs=cell_outputs,
                flags=parsed.flags,
                append_to_layout=False,
            )

            return _SegmentOutcome(
                cell_id=cell_id,
                kind=parsed.kind,
                summary=full_summary,
                data=merged_data,
                markdown_display=md_display,
                warnings=tuple(warnings),
            )
        except _KernelExecutionError as err:
            tracker.fail_run(
                run_id,
                {
                    "exception.type": err.k_code,
                    "exception.message": err.reason,
                },
            )
            raise
        except Exception as exc:
            err = _KernelExecutionError(K42_VALIDATION_FAILED, str(exc))
            tracker.fail_run(
                run_id,
                {
                    "exception.type": err.k_code,
                    "exception.message": err.reason,
                },
            )
            raise err from exc
        finally:
            writer.set_cell_execution_state(cell_id, False)

    def _emit_error_span(
        self, *, cell_id: str, name: str, k_code: str, reason: str
    ) -> None:
        tracker = self.facade.run_tracker
        writer = self.facade.metadata_writer
        rid = tracker.start_run(
            name=name,
            run_type="chain",
            inputs={"cell_id": cell_id},
            tags=[f"zone:{self._zone_id}", f"k_code:{k_code}"],
            metadata={"cell_id": cell_id, "k_code": k_code},
        )
        tracker.fail_run(
            rid,
            {"exception.type": k_code, "exception.message": reason},
        )

    @staticmethod
    def _find_cell_section_id(writer: Any, cell_id: str) -> Optional[str]:
        rec = writer.get_cell_record(cell_id)
        if isinstance(rec, dict) and isinstance(rec.get("section_id"), str):
            return rec["section_id"]
        return None

    # -- Line-magic execution -------------------------------------------

    def _apply_line_magics(
        self,
        *,
        cell_id: str,
        parsed: ParsedCell,
        writer: Any,
        supervisor: Any,
        prior_record: Dict[str, Any],
    ) -> Tuple[List[str], Dict[str, Any], Optional[str], Dict[str, Any]]:
        notes: List[str] = []
        data: Dict[str, Any] = {}
        pending_handoff: Optional[str] = None
        extra_fields: Dict[str, Any] = {}

        default_agent = (
            parsed.args.get("agent_id")
            or prior_record.get("bound_agent_id")
            or self._active_agent_id
        )

        for magic_name, raw_args in parsed.line_magics:
            args_str = (raw_args or "").strip()

            if magic_name == "affinity":
                stack = [s.strip() for s in args_str.split(",") if s.strip()]
                if not stack:
                    raise _KernelExecutionError(
                        K42_VALIDATION_FAILED,
                        "@affinity requires at least one endpoint name",
                    )
                extra_fields["affinity_stack"] = stack
                data["affinity_stack"] = stack
                notes.append(f"affinity: {', '.join(stack)}")

            elif magic_name == "handoff":
                pos, named = parse_kv_args(args_str)
                target = pos[0] if pos else named.get("agent") or named.get("to")
                if not isinstance(target, str) or not target:
                    raise _KernelExecutionError(
                        K42_VALIDATION_FAILED,
                        "@handoff requires a target agent_id",
                    )
                if is_reserved_name(target):
                    raise _KernelExecutionError(
                        K32_RESERVED_MAGIC_NAME,
                        f"reserved magic name {target!r} cannot be an agent_id",
                    )
                pending_handoff = target
                extra_fields["inherit_context_from"] = default_agent
                notes.append(f"handoff -> {target}")

            elif magic_name == "status":
                status_info = self._build_status_dict()
                data["status_info"] = status_info
                agent_count = len(status_info["agents"])
                notes.append(
                    f"llmnb v{status_info['kernel_version']} | "
                    f"zone={status_info['zone_id']} | "
                    f"agents={agent_count} | "
                    f"cells={status_info['cell_count']} | "
                    f"snapshot_version={status_info['snapshot_version']} | "
                    f"hash_mode={'on' if status_info['magic_hash_enabled'] else 'off'}"
                )

            elif magic_name == "auth":
                sub = args_str.split(None, 1)[0].lower() if args_str else ""
                if sub == "status":
                    enabled = bool(writer.get_config_setting("magic_hash_enabled"))
                    fp = writer.get_config_setting("magic_pin_fingerprint")
                    pin_loaded = writer.get_operator_pin() is not None
                    auth_info = {
                        "subcommand": "status",
                        "magic_hash_enabled": enabled,
                        "magic_pin_fingerprint": fp,
                        "pin_loaded": pin_loaded,
                    }
                    data["auth"] = auth_info
                    notes.append(
                        f"auth: hash_mode={'on' if enabled else 'off'} "
                        f"pin_loaded={pin_loaded}"
                    )
                else:
                    cm = CellManager(writer)
                    result = apply_auth_command(
                        args_str, writer=writer, cell_manager=cm
                    )
                    if not result.ok:
                        raise _KernelExecutionError(
                            result.code or K42_VALIDATION_FAILED,
                            result.reason or "@auth command failed",
                        )
                    data["auth"] = {
                        "subcommand": sub,
                        "ok": True,
                        "details": dict(result.details or {}),
                    }
                    notes.append(f"auth {sub}: ok")

            elif magic_name == "branch":
                pos, named = parse_kv_args(args_str)
                source = (
                    named.get("from")
                    or named.get("source")
                    or named.get("source_agent")
                )
                at_turn = named.get("at") or named.get("at_turn_id")
                new_id = named.get("new") or named.get("new_agent_id") or named.get("to")
                if len(pos) >= 2 and not source and not new_id:
                    source, new_id = pos[0], pos[1]
                elif len(pos) == 1 and not new_id:
                    new_id = pos[0]
                if not source:
                    source = default_agent
                if not isinstance(source, str) or not source or not isinstance(new_id, str) or not new_id:
                    raise _KernelExecutionError(
                        K42_VALIDATION_FAILED,
                        "usage: @branch <new_agent_id> [from:<source_agent>] [at:<turn_id>]",
                    )
                if is_reserved_name(new_id):
                    raise _KernelExecutionError(
                        K32_RESERVED_MAGIC_NAME,
                        f"reserved magic name {new_id!r} cannot be an agent_id",
                    )
                try:
                    supervisor.fork(source, at_turn, new_id)
                except KeyError as exc:
                    intent_res = writer.submit_intent(
                        {
                            "type": "operator.action",
                            "payload": {
                                "action_type": "zone_mutate",
                                "intent_kind": "fork_agent",
                                "intent_id": f"branch-{cell_id}-{uuid.uuid4().hex[:8]}",
                                "parameters": {
                                    "source_agent_id": source,
                                    "new_agent_id": new_id,
                                    **({"at_turn_id": at_turn} if at_turn else {}),
                                },
                            },
                        }
                    )
                    if not intent_res.get("applied"):
                        raise _KernelExecutionError(
                            K23_UNKNOWN_AGENT,
                            f"unknown source agent {source!r}",
                        ) from exc
                except ValueError as exc:
                    code = K32_RESERVED_MAGIC_NAME if "K32" in str(exc) else K42_VALIDATION_FAILED
                    raise _KernelExecutionError(code, str(exc)) from exc
                pending_handoff = new_id
                data["branched"] = {
                    "source_agent": source,
                    "new_agent_id": new_id,
                    "at_turn_id": at_turn,
                }
                notes.append(f"branched {source} -> {new_id}")

            elif magic_name == "revert":
                pos, named = parse_kv_args(args_str)
                agent_target = named.get("agent") or named.get("agent_id")
                turn_target = named.get("to") or named.get("turn") or named.get("target_turn_id")
                if len(pos) >= 2 and not agent_target and not turn_target:
                    agent_target, turn_target = pos[0], pos[1]
                elif len(pos) == 1 and not turn_target:
                    turn_target = pos[0]
                if not agent_target:
                    agent_target = default_agent
                if not isinstance(agent_target, str) or not agent_target or not isinstance(turn_target, str) or not turn_target:
                    raise _KernelExecutionError(
                        K42_VALIDATION_FAILED,
                        "usage: @revert <target_turn_id> [agent:<agent_id>]",
                    )
                try:
                    supervisor.revert(agent_target, turn_target)
                except KeyError as exc:
                    intent_res = writer.submit_intent(
                        {
                            "type": "operator.action",
                            "payload": {
                                "action_type": "zone_mutate",
                                "intent_kind": "move_agent_head",
                                "intent_id": f"revert-{cell_id}-{uuid.uuid4().hex[:8]}",
                                "parameters": {
                                    "agent_id": agent_target,
                                    "target_turn_id": turn_target,
                                },
                            },
                        }
                    )
                    if not intent_res.get("applied"):
                        err_code = intent_res.get("error_code") or K23_UNKNOWN_AGENT
                        err_reason = (
                            intent_res.get("error_reason")
                            or f"unknown agent {agent_target!r}"
                        )
                        raise _KernelExecutionError(err_code, err_reason) from exc
                except ValueError as exc:
                    raise _KernelExecutionError(K42_VALIDATION_FAILED, str(exc)) from exc
                data["reverted"] = {
                    "agent_id": agent_target,
                    "target_turn_id": turn_target,
                }
                notes.append(f"reverted {agent_target} -> {turn_target}")

            elif magic_name == "stop":
                pos, named = parse_kv_args(args_str)
                agent_target = (
                    pos[0]
                    if pos
                    else named.get("agent") or named.get("agent_id") or default_agent
                )
                if not isinstance(agent_target, str) or not agent_target:
                    raise _KernelExecutionError(
                        K42_VALIDATION_FAILED,
                        "usage: @stop [<agent_id>]",
                    )
                try:
                    supervisor.stop(agent_target)
                except KeyError as exc:
                    raise _KernelExecutionError(
                        K23_UNKNOWN_AGENT, f"unknown agent {agent_target!r}"
                    ) from exc
                except ValueError as exc:
                    raise _KernelExecutionError(K42_VALIDATION_FAILED, str(exc)) from exc
                data["stopped_agent_id"] = agent_target
                notes.append(f"stopped {agent_target}")

            elif magic_name == "mark":
                notes.append(f"marked kind={parsed.kind}")

        if parsed.flags and not notes and not parsed.body.strip() and parsed.kind_was_default:
            notes.append(f"flags: {', '.join(sorted(parsed.flags))}")

        return notes, data, pending_handoff, extra_fields

    # -- Cell-kind execution --------------------------------------------

    def _dispatch_cell_kind(
        self,
        *,
        cell_id: str,
        text: str,
        parsed: ParsedCell,
        prior_record: Dict[str, Any],
        writer: Any,
        supervisor: Any,
        has_line_magics: bool,
    ) -> Tuple[str, Dict[str, Any], Optional[str], Optional[str], List[str]]:
        kind = parsed.kind
        extra_warnings: List[str] = []

        if kind == "markdown":
            body = parsed.body
            return "", {}, (body if body.strip() else None), None, extra_warnings

        if kind == "scratch":
            return "", {"excluded_from_context": True}, None, None, extra_warnings

        if kind == "checkpoint":
            covers = list(parsed.args.get("covers") or [])
            summary_text = parsed.body.strip()
            writer.update_cell_record(
                cell_id,
                extra={"covers_cell_ids": covers, "summary_text": summary_text},
                append_to_layout=False,
            )
            label = (
                f"checkpoint recorded (covers {len(covers)} cell(s))"
                if covers
                else "checkpoint recorded"
            )
            return (
                label,
                {"covers_cell_ids": covers, "summary_text": summary_text},
                None,
                None,
                extra_warnings,
            )

        if kind == "endpoint":
            ep_name = parsed.args.get("endpoint_name")
            if not isinstance(ep_name, str) or not ep_name.strip():
                raise _KernelExecutionError(
                    K42_VALIDATION_FAILED,
                    "@@endpoint requires a name: @@endpoint <name> [provider:...] [model:...] [api_key_env:...]",
                )
            named = dict(parsed.args.get("named") or {})
            try:
                writer.set_config_endpoint(ep_name, named)
            except SecretRejected as exc:
                writer.delete_cell(cell_id)
                raise _KernelExecutionError(K43_SECRET_REJECTED, str(exc)) from exc
            return (
                f"registered endpoint {ep_name!r}",
                {"endpoint_name": ep_name, "endpoint": named},
                None,
                None,
                extra_warnings,
            )

        if kind == "section":
            title = parsed.args.get("title") or parsed.args.get("section_name")
            if not isinstance(title, str) or not title.strip():
                raise _KernelExecutionError(
                    K42_VALIDATION_FAILED,
                    '@@section requires a title: @@section <title> [id:"sec_..."]',
                )
            section_id = parsed.args.get("section_id") or _mint_section_id(title)
            intent_res = writer.submit_intent(
                {
                    "type": "operator.action",
                    "payload": {
                        "action_type": "zone_mutate",
                        "intent_kind": "apply_overlay_commit",
                        "intent_id": f"sec-{cell_id}-{uuid.uuid4().hex[:8]}",
                        "parameters": {
                            "operations": [
                                {
                                    "kind": "create_section",
                                    "section_id": section_id,
                                    "title": title,
                                }
                            ],
                            "message": f"@@section {title}",
                        },
                    },
                }
            )
            if not intent_res.get("applied"):
                raise _KernelExecutionError(
                    intent_res.get("error_code") or K42_VALIDATION_FAILED,
                    intent_res.get("error_reason") or "failed to create section",
                )
            writer.update_cell_record(
                cell_id,
                extra={"section_id": section_id, "title": title},
                append_to_layout=False,
            )
            resp = intent_res.get("response") or {}
            return (
                f"section {section_id!r} ({title})",
                {
                    "section_id": section_id,
                    "title": title,
                    "commit_id": resp.get("commit_id"),
                },
                f"# {title}",
                None,
                extra_warnings,
            )

        if parsed.is_generator or kind in ("template", "expand", "import"):
            cm = CellManager(writer)
            writer.set_cell_execution_state(cell_id, False)
            try:
                inserted = dispatch_generator(
                    cell_id=cell_id,
                    magic_name=kind,
                    args=parsed.args,
                    body=parsed.body,
                    writer=writer,
                    cell_manager=cm,
                )
            except GeneratorError as exc:
                raise _KernelExecutionError(exc.code, exc.reason) from exc
            except CellManagerPreconditionError as exc:
                raise _KernelExecutionError(exc.k_code, exc.reason) from exc
            finally:
                writer.set_cell_execution_state(cell_id, True)
            if kind == "import":
                pos = parsed.args.get("positional") or []
                src = pos[0] if pos else (parsed.args.get("named") or {}).get("file", "")
                summary = f"imported {len(inserted)} cells from {src}"
            else:
                summary = f"generated {len(inserted)} cells via @@{kind}"
            return (
                summary,
                {"inserted_cell_ids": list(inserted), "count": len(inserted)},
                None,
                None,
                extra_warnings,
            )

        if kind == "export":
            target_path = parsed.args.get("path")
            if not isinstance(target_path, str) or not target_path.strip():
                raise _KernelExecutionError(
                    K42_VALIDATION_FAILED,
                    '@@export requires path:"<relpath>"',
                )
            fmt = parsed.args.get("format")
            overwrite = bool(parsed.args.get("overwrite", False))
            snapshot = writer.snapshot(trigger="save")
            outcome = apply_export(
                cell_id=cell_id,
                path=target_path,
                format=fmt,
                overwrite=overwrite,
                notebook_state=snapshot,
                workspace_root=self._workdir,
            )
            if outcome.status != "ok":
                msg = outcome.message or "export failed"
                if outcome.cause and outcome.cause not in msg:
                    msg = f"{msg} (cause={outcome.cause})"
                raise _KernelExecutionError(
                    outcome.k_code or "K3O", msg
                )
            extra_warnings.extend(outcome.warnings)
            rel_or_abs = str(outcome.path)
            try:
                if outcome.path is not None:
                    rel_or_abs = str(outcome.path.relative_to(self._workdir))
            except ValueError:
                pass
            return (
                f"exported {outcome.cells_written} cells to {rel_or_abs}",
                {
                    "path": str(outcome.path) if outcome.path else target_path,
                    "format": outcome.format,
                    "cells_written": outcome.cells_written,
                },
                None,
                None,
                extra_warnings,
            )

        if kind == "spawn":
            return self._execute_spawn_cell(
                cell_id=cell_id,
                parsed=parsed,
                writer=writer,
                supervisor=supervisor,
            )

        if kind == "compare":
            raise _KernelExecutionError(
                K42_VALIDATION_FAILED,
                "@@compare is reserved for V1.5+ and not yet implemented",
            )

        if kind in _RESERVED_INERT_KINDS:
            return (
                f"[reserved kind: {kind}]",
                {"reserved": True},
                None,
                None,
                extra_warnings,
            )

        if kind == "agent":
            return self._execute_agent_cell(
                cell_id=cell_id,
                parsed=parsed,
                prior_record=prior_record,
                supervisor=supervisor,
                has_line_magics=has_line_magics,
            )

        raise _KernelExecutionError(
            K42_VALIDATION_FAILED, f"unsupported cell kind: {kind!r}"
        )

    def _execute_spawn_cell(
        self,
        *,
        cell_id: str,
        parsed: ParsedCell,
        writer: Any,
        supervisor: Any,
    ) -> Tuple[str, Dict[str, Any], Optional[str], Optional[str], List[str]]:
        agent_id = parsed.args.get("agent_id")
        if not isinstance(agent_id, str) or not agent_id.strip():
            raise _KernelExecutionError(
                K42_VALIDATION_FAILED,
                '@@spawn requires an agent_id: @@spawn <id> task:"..."',
            )
        if is_reserved_name(agent_id):
            raise _KernelExecutionError(
                K32_RESERVED_MAGIC_NAME,
                f"reserved magic name {agent_id!r} cannot be an agent_id",
            )
        task = parsed.args.get("task")
        body = parsed.body.strip()
        if not isinstance(task, str) or not task.strip():
            task = body
        if not task:
            raise _KernelExecutionError(
                K42_VALIDATION_FAILED,
                f'@@spawn {agent_id} requires task:"..." or a non-empty body',
            )

        named = parsed.args.get("named") or {}
        endpoint_name = parsed.args.get("endpoint")
        ep_cfg: Dict[str, Any] = {}
        if isinstance(endpoint_name, str) and endpoint_name:
            ep_cfg = writer.read_config_endpoints().get(endpoint_name, {})

        model = (
            named.get("model")
            or ep_cfg.get("model")
            or os.environ.get(
                "LLMKERNEL_DEFAULT_AGENT_MODEL", "claude-haiku-4-5-20251001"
            )
        )
        use_bare = os.environ.get("LLMKERNEL_USE_BARE") == "1"
        api_key_env = ep_cfg.get("api_key_env") or "ANTHROPIC_API_KEY"
        api_key = os.environ.get(api_key_env) if use_bare else None

        agent_work_dir = (self._workdir / ".llmnb-agents" / agent_id).resolve()
        agent_work_dir.mkdir(parents=True, exist_ok=True)

        self._executing_agent_id = agent_id
        try:
            handle = supervisor.spawn(
                zone_id=self._zone_id,
                agent_id=agent_id,
                task=task,
                work_dir=agent_work_dir,
                model=model,
                api_key=api_key,
                use_bare=use_bare,
            )
        except Exception as exc:
            msg = str(exc)
            code = K32_RESERVED_MAGIC_NAME if "K32" in msg else "K12"
            raise _KernelExecutionError(code, msg) from exc

        spawned_id = getattr(handle, "agent_id", agent_id) or agent_id
        self._active_agent_id = spawned_id

        initial_turn_text = (
            body
            if (parsed.args.get("task") and body)
            else (
                task
                if os.environ.get(ENV_SPAWN_SEND_INITIAL_TURN) == "1"
                else None
            )
        )
        turn_result: Optional[Dict[str, Any]] = None
        if initial_turn_text:
            try:
                turn_result = supervisor.send_user_turn(
                    spawned_id, initial_turn_text, cell_id=cell_id
                )
            except Exception as exc:
                raise _KernelExecutionError(K42_VALIDATION_FAILED, str(exc)) from exc

        result_data: Dict[str, Any] = {
            "agent_id": spawned_id,
            "zone_id": self._zone_id,
            "task": task,
            "model": model,
        }
        if endpoint_name:
            result_data["endpoint"] = endpoint_name
        if isinstance(turn_result, dict):
            result_data["initial_turn"] = turn_result
        return (
            f"spawned agent {spawned_id} (zone={self._zone_id})",
            result_data,
            None,
            spawned_id,
            [],
        )

    def _execute_agent_cell(
        self,
        *,
        cell_id: str,
        parsed: ParsedCell,
        prior_record: Dict[str, Any],
        supervisor: Any,
        has_line_magics: bool,
    ) -> Tuple[str, Dict[str, Any], Optional[str], Optional[str], List[str]]:
        body = parsed.body.strip()
        explicit_agent = parsed.args.get("agent_id")
        prior_agent = prior_record.get("bound_agent_id")

        if parsed.kind_was_default and not body:
            if has_line_magics:
                return "", {}, None, prior_agent, []
            if parsed.k_class_emissions:
                return "", {}, None, prior_agent, []

        if parsed.kind_was_default and parsed.k_class_emissions and not explicit_agent and not self._active_agent_id and not prior_agent:
            return "", {"degraded_to_body": True}, None, None, []

        agent_id = explicit_agent or prior_agent or self._active_agent_id
        if not isinstance(agent_id, str) or not agent_id.strip():
            raise _KernelExecutionError(
                K20_UNBOUND_AGENT_CELL,
                "agent cell has no bound agent; spawn an agent first or specify @@agent <id>",
            )
        if not body:
            raise _KernelExecutionError(
                K42_VALIDATION_FAILED,
                f"@@agent {agent_id} requires a non-empty message body",
            )

        self._executing_agent_id = agent_id
        try:
            turn_res = supervisor.send_user_turn(
                agent_id, body, cell_id=cell_id
            )
        except KeyError as exc:
            raise _KernelExecutionError(
                K23_UNKNOWN_AGENT, f"unknown agent {agent_id!r}"
            ) from exc
        except ValueError as exc:
            raise _KernelExecutionError(K42_VALIDATION_FAILED, str(exc)) from exc

        self._active_agent_id = agent_id
        out_data: Dict[str, Any] = {"agent_id": agent_id}
        if isinstance(turn_res, dict):
            out_data["turn"] = turn_res
        turn_id = turn_res.get("turn_id") if isinstance(turn_res, dict) else None
        suffix = f" (turn={turn_id})" if turn_id else ""
        return (
            f"sent turn to {agent_id}{suffix}",
            out_data,
            None,
            agent_id,
            [],
        )

    # -- Jupyter Shell: complete_request --------------------------------

    def do_complete(self, code: str, cursor_pos: int) -> Dict[str, Any]:
        pos = max(0, min(len(code), int(cursor_pos)))
        line_start = code.rfind("\n", 0, pos) + 1
        line_prefix = code[line_start:pos]

        matches, start_offset, types = self._compute_completions(line_prefix)
        cursor_start = line_start + start_offset
        return {
            "status": "ok",
            "matches": matches,
            "cursor_start": cursor_start,
            "cursor_end": pos,
            "metadata": {
                "_jupyter_types_experimental": [
                    {
                        "text": m,
                        "type": t,
                        "start": cursor_start,
                        "end": pos,
                    }
                    for m, t in zip(matches, types)
                ]
            },
        }

    def _compute_completions(
        self, line_prefix: str
    ) -> Tuple[List[str], int, List[str]]:
        if line_prefix.startswith("@@") and " " not in line_prefix:
            partial = line_prefix[2:]
            cands = sorted(f"@@{name}" for name in CELL_MAGICS if name.startswith(partial))
            return cands, 0, ["magic"] * len(cands)

        if line_prefix.startswith("@@"):
            m = re.match(r"^@@([A-Za-z_]\w*)\s+(.*)$", line_prefix)
            if m:
                magic_name, rest = m.group(1), m.group(2)
                token_match = re.search(r"(\S*)$", rest)
                token = token_match.group(1) if token_match else ""
                tok_start = len(line_prefix) - len(token)
                return self._complete_cell_magic_args(magic_name, rest, token, tok_start)

        if line_prefix.startswith("@") and " " not in line_prefix and ":" not in line_prefix:
            partial = line_prefix[1:]
            line_cands = [
                (f"@{name}", "magic")
                for name in sorted(LINE_MAGICS)
                if name.startswith(partial)
            ]
            agent_cands = [
                (f"@{aid}: ", "agent")
                for aid in self._known_agent_ids()
                if aid.startswith(partial) and not is_reserved_name(aid)
            ]
            combined = line_cands + agent_cands
            return (
                [c[0] for c in combined],
                0,
                [c[1] for c in combined],
            )

        if line_prefix.startswith("@"):
            m = re.match(r"^@([A-Za-z_]\w*)\s+(.*)$", line_prefix)
            if m:
                magic_name, rest = m.group(1), m.group(2)
                token_match = re.search(r"(\S*)$", rest)
                token = token_match.group(1) if token_match else ""
                tok_start = len(line_prefix) - len(token)
                return self._complete_line_magic_args(magic_name, token, tok_start)

        if line_prefix.startswith("/") and " " not in line_prefix:
            if "/spawn".startswith(line_prefix):
                return ["/spawn"], 0, ["magic"]

        return [], len(line_prefix), []

    def _complete_cell_magic_args(
        self, magic_name: str, rest: str, token: str, tok_start: int
    ) -> Tuple[List[str], int, List[str]]:
        if ":" in token:
            key, val_prefix = token.split(":", 1)
            val_start = tok_start + len(key) + 1
            if key == "endpoint":
                eps = [e for e in self._known_endpoint_names() if e.startswith(val_prefix)]
                return eps, val_start, ["value"] * len(eps)
            if key == "format":
                fmts = [f for f in ("llmnb", "magic", "ipynb") if f.startswith(val_prefix)]
                return fmts, val_start, ["value"] * len(fmts)
            if key == "overwrite":
                bools = [b for b in ("true", "false") if b.startswith(val_prefix)]
                return bools, val_start, ["value"] * len(bools)
            if key == "endpoints":
                last_comma = val_prefix.rfind(",")
                sub_prefix = val_prefix[last_comma + 1:]
                sub_start = val_start + last_comma + 1
                eps = [e for e in self._known_endpoint_names() if e.startswith(sub_prefix)]
                return eps, sub_start, ["value"] * len(eps)
            return [], len(rest), []

        prior_tokens = rest[: len(rest) - len(token)].split()
        results: List[Tuple[str, str]] = []
        if magic_name == "agent" and not prior_tokens:
            results.extend(
                (aid, "agent")
                for aid in self._known_agent_ids()
                if aid.startswith(token)
            )
        elif magic_name == "template" and not prior_tokens:
            results.extend(
                (t, "value")
                for t in self._known_template_names()
                if t.startswith(token)
            )
        for kw in _CELL_MAGIC_KWARGS.get(magic_name, ()):
            if kw.startswith(token):
                results.append((kw, "keyword"))
        return [r[0] for r in results], tok_start, [r[1] for r in results]

    def _complete_line_magic_args(
        self, magic_name: str, token: str, tok_start: int
    ) -> Tuple[List[str], int, List[str]]:
        if magic_name == "mark":
            kinds = sorted(
                k for k in CELL_MAGICS if k != "break" and k.startswith(token)
            )
            return kinds, tok_start, ["value"] * len(kinds)
        if magic_name == "auth":
            subs = [s for s in _AUTH_SUBCOMMANDS if s.startswith(token)]
            return subs, tok_start, ["keyword"] * len(subs)
        if magic_name in ("handoff", "stop"):
            agents = [a for a in self._known_agent_ids() if a.startswith(token)]
            return agents, tok_start, ["agent"] * len(agents)
        if magic_name == "branch":
            if token.startswith("from:"):
                sub = token[5:]
                agents = [a for a in self._known_agent_ids() if a.startswith(sub)]
                return agents, tok_start + 5, ["agent"] * len(agents)
            cands: List[Tuple[str, str]] = [
                (kw, "keyword") for kw in ("from:", "at:") if kw.startswith(token)
            ]
            return [c[0] for c in cands], tok_start, [c[1] for c in cands]
        if magic_name == "revert":
            if token.startswith("agent:"):
                sub = token[6:]
                agents = [a for a in self._known_agent_ids() if a.startswith(sub)]
                return agents, tok_start + 6, ["agent"] * len(agents)
            cands = [
                (kw, "keyword") for kw in ("agent:", "to:") if kw.startswith(token)
            ]
            return [c[0] for c in cands], tok_start, [c[1] for c in cands]
        if magic_name == "affinity":
            last_comma = token.rfind(",")
            sub = token[last_comma + 1:]
            sub_start = tok_start + last_comma + 1
            eps = [e for e in self._known_endpoint_names() if e.startswith(sub)]
            return eps, sub_start, ["value"] * len(eps)
        return [], tok_start + len(token), []

    # -- Jupyter Shell: inspect_request ---------------------------------

    def do_inspect(
        self,
        code: str,
        cursor_pos: int,
        detail_level: int = 0,
        omit_sections: Sequence[str] = (),
    ) -> Dict[str, Any]:
        token = self._extract_token_at(code, cursor_pos)
        if not token:
            return {"status": "ok", "found": False, "data": {}, "metadata": {}}

        doc = self._lookup_inspection(token, detail_level=int(detail_level))
        if doc is None:
            return {"status": "ok", "found": False, "data": {}, "metadata": {}}

        plain, markdown = doc
        return {
            "status": "ok",
            "found": True,
            "data": {"text/plain": plain, "text/markdown": markdown},
            "metadata": {},
        }

    @staticmethod
    def _extract_token_at(code: str, cursor_pos: int) -> str:
        if not code:
            return ""
        pos = max(0, min(len(code), int(cursor_pos)))
        allowed = set(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_@/-:"
        )
        start = pos
        while start > 0 and code[start - 1] in allowed:
            start -= 1
        end = pos
        while end < len(code) and code[end] in allowed:
            end += 1
        token = code[start:end].rstrip(":")
        if token == "/spawn":
            return "@@spawn"
        return token

    def _lookup_inspection(
        self, token: str, *, detail_level: int
    ) -> Optional[Tuple[str, str]]:
        magic_key: Optional[str] = None
        if token in _MAGIC_DOCS:
            magic_key = token
        elif f"@@{token}" in _MAGIC_DOCS and token in CELL_MAGICS:
            magic_key = f"@@{token}"
        elif f"@{token}" in _MAGIC_DOCS and token in LINE_MAGICS:
            magic_key = f"@{token}"

        if magic_key is not None:
            info = _MAGIC_DOCS[magic_key]
            bare = magic_key.lstrip("@")
            handler = (
                CELL_MAGICS.get(bare)
                if magic_key.startswith("@@")
                else LINE_MAGICS.get(bare)
            )
            status = getattr(handler, "status", "active")
            pending = getattr(handler, "pending_slice", "")
            status_label = f"{status} ({pending})" if pending else status
            plain_lines = [
                f"{magic_key} [{status_label}]",
                f"Syntax: {info['syntax']}",
                info["summary"],
            ]
            md_lines = [
                f"**`{magic_key}`** (`{status_label}`)",
                "",
                f"```\n{info['syntax']}\n```",
                "",
                info["summary"],
            ]
            if detail_level >= 1:
                plain_lines.append(f"Errors: {info['errors']}")
                md_lines.extend(["", f"**K-class errors:** {info['errors']}"])
            return "\n".join(plain_lines), "\n".join(md_lines)

        bare_agent = token[1:] if token.startswith("@") and not token.startswith("@@") else token
        agent_doc = self._inspect_agent(bare_agent, detail_level=detail_level)
        if agent_doc is not None:
            return agent_doc

        facade = self._facade
        if facade is None or not facade.started:
            return None
        writer = facade.metadata_writer

        endpoints = writer.read_config_endpoints()
        if token in endpoints:
            spec = endpoints[token]
            plain = f"Endpoint {token!r}: {json.dumps(spec, sort_keys=True)}"
            md = f"**Endpoint `{token}`**\n\n```json\n{json.dumps(spec, indent=2, sort_keys=True)}\n```"
            return plain, md

        templates = writer.read_config_templates()
        if token in templates:
            body = templates[token]
            plain = f"Template {token!r}\n{body}" if detail_level >= 1 else f"Template {token!r} ({len(body.splitlines())} line(s))"
            md = f"**Template `{token}`**\n\n```\n{body}\n```"
            return plain, md

        cell_rec = writer.get_cell_record(token)
        if isinstance(cell_rec, dict):
            return self._inspect_cell(token, cell_rec, writer, detail_level=detail_level)

        snap = writer.snapshot(trigger="inspect")
        sections = (snap.get("zone") or {}).get("sections") or {}
        if isinstance(sections, dict) and token in sections:
            sec = sections[token]
            title = sec.get("title", "")
            status = sec.get("status", "open")
            cells = sec.get("cell_range") or []
            plain = f"Section {token!r}: title={title!r} status={status} cells={len(cells)}"
            md = f"**Section `{token}`** — *{title}* (`{status}`, {len(cells)} cell(s))"
            return plain, md

        return None

    def _inspect_agent(
        self, agent_id: str, *, detail_level: int
    ) -> Optional[Tuple[str, str]]:
        facade = self._facade
        if facade is None or not facade.started:
            return None
        live_map = {a["agent_id"]: a for a in facade.list_agents() if "agent_id" in a}
        snap = facade.metadata_writer.snapshot(trigger="inspect")
        zone = snap.get("zone") or {}
        zone_agents = zone.get("agents") if isinstance(zone.get("agents"), dict) else {}

        live = live_map.get(agent_id)
        persisted = zone_agents.get(agent_id) if isinstance(zone_agents, dict) else None
        if live is None and not isinstance(persisted, dict):
            return None

        status = (
            (live or {}).get("status")
            or ((persisted or {}).get("session") or {}).get("status")
            or "idle"
        )
        zone_id = (live or {}).get("zone_id") or self._zone_id
        model = (live or {}).get("model") or ((persisted or {}).get("model")) or "unknown"
        turns = (persisted or {}).get("turns") if isinstance(persisted, dict) else []
        turns_list = turns if isinstance(turns, list) else []
        head_turn = (
            ((persisted or {}).get("session") or {}).get("head_turn_id")
            if isinstance(persisted, dict)
            else None
        )

        plain_lines = [
            f"Agent {agent_id} (zone={zone_id}, status={status}, model={model})",
            f"Turns: {len(turns_list)} | head_turn_id={head_turn}",
        ]
        md_lines = [
            f"**Agent `{agent_id}`**",
            f"- Zone: `{zone_id}`",
            f"- Status: `{status}`",
            f"- Model: `{model}`",
            f"- Turns: `{len(turns_list)}` (head: `{head_turn}`)",
        ]

        if detail_level >= 1:
            run_frames = zone.get("run_frames") if isinstance(zone.get("run_frames"), dict) else {}
            agent_frames = [
                rf for rf in run_frames.values()
                if isinstance(rf, dict)
                and (rf.get("agent_id") == agent_id or rf.get("executor_id") == agent_id)
            ]
            if agent_frames:
                latest = agent_frames[-1]
                manifest_id = latest.get("context_manifest_id") or latest.get("manifest_id")
                plain_lines.append(
                    f"Latest RunFrame: status={latest.get('status')} manifest={manifest_id}"
                )
                md_lines.append(
                    f"- Latest RunFrame: status=`{latest.get('status')}`, manifest=`{manifest_id}`"
                )
                manifests = (
                    zone.get("context_manifests")
                    if isinstance(zone.get("context_manifests"), dict)
                    else {}
                )
                if manifest_id and manifest_id in manifests:
                    mf = manifests[manifest_id]
                    inc = len(mf.get("included") or mf.get("included_turn_ids") or [])
                    exc = len(mf.get("excluded") or mf.get("excluded_turn_ids") or [])
                    plain_lines.append(f"ContextManifest {manifest_id}: included={inc} excluded={exc}")
                    md_lines.append(f"- ContextManifest `{manifest_id}`: included=`{inc}`, excluded=`{exc}`")

        return "\n".join(plain_lines), "\n".join(md_lines)

    def _inspect_cell(
        self,
        cell_id: str,
        record: Dict[str, Any],
        writer: Any,
        *,
        detail_level: int,
    ) -> Tuple[str, str]:
        kind = record.get("kind") or "agent"
        agent = record.get("bound_agent_id")
        pinned = bool(record.get("pinned"))
        excluded = bool(record.get("excluded"))
        contaminated = bool(record.get("contaminated"))
        plain_lines = [
            f"Cell {cell_id}: kind={kind} agent={agent} pinned={pinned} excluded={excluded} contaminated={contaminated}"
        ]
        md_lines = [
            f"**Cell `{cell_id}`**",
            f"- Kind: `{kind}`",
            f"- Bound agent: `{agent}`",
            f"- Flags: pinned=`{pinned}`, excluded=`{excluded}`, contaminated=`{contaminated}`",
        ]
        if detail_level >= 1:
            snap = writer.snapshot(trigger="inspect")
            zone = snap.get("zone") or {}
            frames = zone.get("run_frames") if isinstance(zone.get("run_frames"), dict) else {}
            matching = [
                rf for rf in frames.values()
                if isinstance(rf, dict) and rf.get("cell_id") == cell_id
            ]
            if matching:
                latest = matching[-1]
                plain_lines.append(
                    f"Latest RunFrame: run_id={latest.get('run_id')} status={latest.get('status')}"
                )
                md_lines.append(
                    f"- Latest RunFrame: `{latest.get('run_id')}` (`{latest.get('status')}`)"
                )
        return "\n".join(plain_lines), "\n".join(md_lines)

    # -- Jupyter Shell: is_complete_request -----------------------------

    def do_is_complete(self, code: str) -> Dict[str, Any]:
        if not code or not code.strip():
            return {"status": "incomplete", "indent": ""}

        lines = code.splitlines()
        non_blank = [ln for ln in lines if ln.strip()]
        first = non_blank[0]

        if first.startswith(("@", "/")) and _has_unclosed_quote(first):
            return {"status": "incomplete", "indent": ""}

        if _has_unclosed_fence(code):
            return {"status": "incomplete", "indent": ""}

        ctx = (
            self.facade.metadata_writer.parser_context()
            if (self._facade is not None and self._facade.started)
            else None
        )
        segments = (
            split_at_breaks(code)
            if (not _is_expand_cell(code) and any(ln.strip() == "@@break" for ln in lines))
            else [code]
        )
        last_parsed: Optional[ParsedCell] = None
        try:
            for seg in segments:
                last_parsed = _parse_cell_or_expand(seg, ctx)
        except CellParseError:
            return {"status": "invalid"}

        if (
            last_parsed is not None
            and not last_parsed.kind_was_default
            and last_parsed.kind in _BODY_EXPECTING_MAGICS
            and not last_parsed.body.strip()
        ):
            if code.endswith("\n\n") or (len(lines) >= 2 and not lines[-1].strip()):
                return {"status": "complete"}
            return {"status": "incomplete", "indent": ""}

        return {"status": "complete"}

    # -- Jupyter Shell: history_request ---------------------------------

    def do_history(
        self,
        hist_access_type: str,
        output: bool,
        raw: bool,
        session: Optional[int] = None,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        n: Optional[int] = None,
        pattern: Optional[str] = None,
        unique: bool = False,
    ) -> Dict[str, Any]:
        entries = list(self._history)

        if hist_access_type == "range":
            if session is not None and session > 0:
                entries = [e for e in entries if e.session == session]
            if start is not None:
                entries = [e for e in entries if e.line_num >= start]
            if stop is not None:
                entries = [e for e in entries if e.line_num < stop]
        elif hist_access_type == "tail":
            if n is not None and n >= 0:
                entries = entries[-n:] if n > 0 else []
        elif hist_access_type == "search":
            if pattern:
                entries = [
                    e for e in entries if fnmatch.fnmatchcase(e.raw_input, pattern)
                ]
            if unique:
                seen: set[str] = set()
                deduped: List[_HistoryEntry] = []
                for e in reversed(entries):
                    if e.raw_input not in seen:
                        seen.add(e.raw_input)
                        deduped.append(e)
                entries = list(reversed(deduped))
            if n is not None and n >= 0:
                entries = entries[-n:] if n > 0 else []

        formatted = [
            (e.session, e.line_num, (e.raw_input, e.output) if output else e.raw_input)
            for e in entries
        ]
        return {"status": "ok", "history": formatted}

    # -- Jupyter Control: interrupt_request + shutdown_request ----------

    def interrupt_agents(self) -> List[Dict[str, Any]]:
        facade = self._facade
        if facade is None or not facade.started:
            return []
        supervisor = facade.supervisor
        targets: List[str] = []
        if self._executing_agent_id:
            targets.append(self._executing_agent_id)
        for info in supervisor.list_agents():
            aid = info.get("agent_id")
            if isinstance(aid, str) and aid not in targets:
                targets.append(aid)
        results: List[Dict[str, Any]] = []
        for aid in targets:
            try:
                res = supervisor.interrupt(aid)
                results.append(
                    {"agent_id": aid, **(res if isinstance(res, dict) else {})}
                )
            except Exception:
                logger.exception("LLMNBKernel: interrupt raised for agent %s", aid)
        return results

    async def interrupt_request(
        self, stream: Any, ident: Any, parent: Dict[str, Any]
    ) -> None:
        if not self.session:
            return
        interrupted = self.interrupt_agents()
        content: Dict[str, Any] = {"status": "ok", "interrupted": interrupted}
        with self._shell_bridge._send_lock:
            self.session.send(stream, "interrupt_reply", content, parent, ident=ident)

    def do_shutdown(self, restart: bool) -> Dict[str, Any]:
        self._teardown_facade(persist_notebook=True)
        self._pending_restart = bool(restart)
        if restart:
            self._session_number += 1
            self._active_agent_id = None
            self._executing_agent_id = None
        return {"status": "ok", "restart": bool(restart)}

    # -- Introspection & user_expressions helpers -----------------------

    def _known_agent_ids(self) -> List[str]:
        return [a["id"] for a in self._collect_agent_summaries()]

    def _known_endpoint_names(self) -> List[str]:
        facade = self._facade
        if facade is None or not facade.started:
            return []
        return sorted(facade.metadata_writer.read_config_endpoints().keys())

    def _known_template_names(self) -> List[str]:
        facade = self._facade
        if facade is None or not facade.started:
            return []
        return sorted(facade.metadata_writer.read_config_templates().keys())

    def _collect_agent_summaries(self) -> List[Dict[str, Any]]:
        facade = self._facade
        if facade is None or not facade.started:
            return []
        by_id: Dict[str, Dict[str, Any]] = {}
        for info in facade.list_agents():
            aid = info.get("agent_id")
            if not isinstance(aid, str) or not aid:
                continue
            by_id[aid] = {
                "id": aid,
                "zone_id": info.get("zone_id", self._zone_id),
                "provider": info.get("provider", "claude-code"),
                "model": info.get("model", ""),
                "status": info.get("status", "running"),
            }
        try:
            snap = facade.metadata_writer.snapshot(trigger="inspect")
            zone_agents = (snap.get("zone") or {}).get("agents") or {}
            if isinstance(zone_agents, dict):
                for aid, rec in zone_agents.items():
                    if aid not in by_id and isinstance(rec, dict):
                        session = rec.get("session") or {}
                        by_id[aid] = {
                            "id": aid,
                            "zone_id": self._zone_id,
                            "provider": rec.get("provider", "claude-code"),
                            "model": rec.get("model", ""),
                            "status": session.get("status", "idle"),
                        }
        except Exception:
            pass
        return [by_id[k] for k in sorted(by_id)]

    def _build_status_dict(self) -> Dict[str, Any]:
        facade = self.facade
        writer = facade.metadata_writer
        snap = writer.snapshot(trigger="inspect")
        cells = snap.get("cells") or {}
        return {
            "kernel_version": _versions.KERNEL_VERSION,
            "package_version": _versions.PACKAGE_VERSION,
            "wire_version": _versions.WIRE_VERSION,
            "schema_version": _versions.SCHEMA_VERSION,
            "comm_target": _versions.COMM_TARGET,
            "zone_id": self._zone_id,
            "active_agent_id": self._active_agent_id,
            "snapshot_version": snap.get("snapshot_version", 0),
            "cell_count": len(cells) if isinstance(cells, dict) else 0,
            "agents": self._collect_agent_summaries(),
            "zones": facade.list_zones(),
            "magic_hash_enabled": bool(
                writer.get_config_setting("magic_hash_enabled")
            ),
            "event_log_queue_depth": writer.event_log_queue_depth(),
            "event_log_overflow_checkpoints": writer.overflow_checkpoint_count(),
            "drift_count": len(snap.get("drift_log") or []),
        }

    def _evaluate_user_expressions(
        self, user_expressions: Optional[Dict[str, str]]
    ) -> Dict[str, Any]:
        if not user_expressions:
            return {}
        results: Dict[str, Any] = {}
        for name, expr in user_expressions.items():
            try:
                val = self._resolve_expression((expr or "").strip())
                results[name] = {
                    "status": "ok",
                    "data": {
                        "text/plain": json.dumps(val, ensure_ascii=False, default=str),
                        "application/json": val,
                    },
                    "metadata": {},
                }
            except KeyError as exc:
                msg = str(exc)
                results[name] = {
                    "status": "error",
                    "ename": "KeyError",
                    "evalue": msg,
                    "traceback": [f"KeyError: {msg}"],
                }
        return results

    def _resolve_expression(self, expr: str) -> Any:
        if expr == "status":
            return self._build_status_dict()
        if expr == "agents":
            return self._collect_agent_summaries()
        if expr == "zones":
            return self.facade.list_zones()
        if expr == "active_agent":
            return self._active_agent_id
        if expr == "endpoints":
            return self.facade.metadata_writer.read_config_endpoints()
        if expr == "templates":
            return sorted(self.facade.metadata_writer.read_config_templates().keys())
        snap = self.facade.metadata_writer.snapshot(trigger="inspect")
        if expr == "snapshot_version":
            return snap.get("snapshot_version", 0)
        if expr == "drift":
            return list(snap.get("drift_log") or [])
        if expr == "sections":
            return dict((snap.get("zone") or {}).get("sections") or {})
        if expr == "cells":
            cells = snap.get("cells") or {}
            order = self.facade.metadata_writer.get_cell_layout_order()
            return [
                {
                    "id": cid,
                    "kind": (cells.get(cid) or {}).get("kind"),
                    "bound_agent_id": (cells.get(cid) or {}).get("bound_agent_id"),
                    "pinned": bool((cells.get(cid) or {}).get("pinned")),
                    "excluded": bool((cells.get(cid) or {}).get("excluded")),
                }
                for cid in order
                if cid in cells
            ]
        if expr.startswith("cell:"):
            cid = expr[5:]
            rec = self.facade.metadata_writer.get_cell_record(cid)
            if rec is None:
                raise KeyError(f"unknown cell: {cid}")
            return rec
        if expr.startswith("agent:"):
            aid = expr[6:]
            for summary in self._collect_agent_summaries():
                if summary["id"] == aid:
                    return summary
            raise KeyError(f"unknown agent: {aid}")
        raise KeyError(f"unknown llmnb expression: {expr}")


class LLMNBKernelApp(IPKernelApp):
    name = "llmnb-kernel"
    kernel_class = LLMNBKernel

    def init_signal(self) -> None:
        if (
            os.name == "nt"
            or threading.current_thread() is not threading.main_thread()
        ):
            return
        signal.signal(signal.SIGINT, self._handle_sigint)

    def _handle_sigint(self, _signum: int, _frame: Any) -> None:
        kernel = getattr(self, "kernel", None)
        if isinstance(kernel, LLMNBKernel):
            kernel.interrupt_agents()


def main(argv: Optional[List[str]] = None) -> int:
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    try:
        from dotenv import find_dotenv, load_dotenv

        dotenv_path = find_dotenv(usecwd=True)
        if dotenv_path:
            load_dotenv(dotenv_path)
    except ImportError:
        pass

    args = list(sys.argv[1:] if argv is None else argv)
    cleaned: List[str] = []
    idx = 0
    while idx < len(args):
        arg = args[idx]
        if arg == "--notebook" and idx + 1 < len(args):
            os.environ[ENV_NOTEBOOK_PATH] = args[idx + 1]
            idx += 2
            continue
        if arg.startswith("--notebook="):
            os.environ[ENV_NOTEBOOK_PATH] = arg.split("=", 1)[1]
            idx += 1
            continue
        if arg == "--workdir" and idx + 1 < len(args):
            os.environ[ENV_WORKDIR] = args[idx + 1]
            idx += 2
            continue
        if arg.startswith("--workdir="):
            os.environ[ENV_WORKDIR] = arg.split("=", 1)[1]
            idx += 1
            continue
        cleaned.append(arg)
        idx += 1
    LLMNBKernelApp.launch_instance(argv=cleaned)
    return 0


__all__ = [
    "ENV_AUTOSAVE_NOTEBOOK",
    "ENV_NOTEBOOK_PATH",
    "ENV_PROXY_MODE",
    "ENV_SPAWN_SEND_INITIAL_TURN",
    "ENV_UNATTENDED",
    "ENV_WORKDIR",
    "LLMNBKernel",
    "LLMNBKernelApp",
    "MIME_LLMNB",
    "MIME_LLMNB_RESULT",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
