"""Single source of truth for version strings: ``llm_kernel.wire.versions``.

Covers:

1. Every site that emits or checks a version derives it from the one
   source module (equality across the ready frame, both handshake
   payloads, the volatile provider, the metadata writer, the notebook
   format default, the health endpoint, the exported schema JSON, the
   MCP server, run envelopes and the driver's connect defaults).
2. The drift regression: a notebook that persisted exactly what the
   ready frame announced yields ZERO drift events against the volatile
   provider, and the detector compares every RFC key the source defines
   (not only rfc_001..003).
3. Runtime override / in-place mutation of the source propagates to the
   ready frame, the handshake payloads, the provider, the drift detector,
   the writer, the exporter and the driver -- without re-importing.
4. Static checks: no ``MAJOR.MINOR.PATCH`` string literal remains in the
   kernel (or driver) source outside the source module and the exported
   schema JSON; ``llm_kernel.wire`` stays stdlib-only.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import re
import socket
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

import llm_kernel
import llm_kernel.wire as wire
from llm_kernel._internal._attrs import decode_attrs
from llm_kernel.capture import custom_messages, run_envelope, run_tracker
from llm_kernel.capture import metadata_writer as mw_pkg
from llm_kernel.capture.drift_detector import DriftDetector
from llm_kernel.capture.metadata_writer import MetadataWriter, _serialize
from llm_kernel.notebook import notebook_format
from llm_kernel.transports import pty_mode, serve_mode
from llm_kernel.wire import export as wire_export
from llm_kernel.wire import tools as wire_tools
from llm_kernel.wire import version as wire_version
from llm_kernel.wire import versions

_SEMVER = re.compile(r"^\d+\.\d+\.\d+$")
_KERNEL_PKG = Path(llm_kernel.__file__).resolve().parent
_SOURCE_MODULE = _KERNEL_PKG / "wire" / "versions.py"
_SCHEMAS_DIR = _KERNEL_PKG / "wire" / "schemas"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FrameSink:
    """Minimal stand-in for SocketWriter: records written frames."""

    def __init__(self) -> None:
        self.frames: List[Dict[str, Any]] = []

    def write_frame(self, record: Dict[str, Any]) -> None:
        self.frames.append(record)


def _ready_frame_attrs() -> Dict[str, Any]:
    sink = _FrameSink()
    pty_mode._emit_ready_handshake(sink, "sess-test")  # noqa: SLF001
    assert len(sink.frames) == 1
    return decode_attrs(sink.frames[0]["attributes"])


def _ready_frame_rfc_versions() -> Dict[str, str]:
    """``rfc_NNN_version -> value`` exactly as the ready frame announced it."""
    attrs = _ready_frame_attrs()
    out: Dict[str, str] = {}
    for key, value in attrs.items():
        if key.startswith("llmnb.kernel.rfc_") and key.endswith("_version"):
            out[key[len("llmnb.kernel."):]] = value
    return out


def _persisted(volatile_kernel: Dict[str, Any]) -> Dict[str, Any]:
    """A minimal persisted ``metadata.rts`` with the given volatile kernel block."""
    return {
        "schema_version": versions.SCHEMA_VERSION,
        "config": {"volatile": {"kernel": dict(volatile_kernel)}},
        "event_log": {"runs": []},
        "zone": {"run_frames": {}},
        "agents": {"nodes": [], "edges": []},
    }


def _handshake_error_payload(code: str) -> Dict[str, Any]:
    sock = MagicMock()
    serve_mode._handshake_error(sock, code)  # noqa: SLF001
    sent = b"".join(call.args[0] for call in sock.sendall.call_args_list)
    return json.loads(sent.decode("utf-8").strip())["payload"]


def _health() -> Dict[str, Any]:
    from fastapi.testclient import TestClient
    from llm_kernel.app import app
    # No ``with`` block: the lifespan (which boots a real kernel) is not run.
    return TestClient(app).get("/health").json()


def _semver_literals(py_file: Path) -> List[str]:
    tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
    hits: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if _SEMVER.match(node.value):
                hits.append(f"{py_file}:{node.lineno}: {node.value!r}")
    return hits


# ---------------------------------------------------------------------------
# 1. Single-source equality across every site
# ---------------------------------------------------------------------------


def test_source_values_are_the_canonical_wire_values() -> None:
    """The source carries exactly what the ready frame emitted before the refactor."""
    assert versions.KERNEL_VERSION == "1.0.0"
    assert versions.WIRE_VERSION == "1.0.0"
    assert versions.SCHEMA_VERSION == "1.0.0"
    assert versions.COMM_TARGET == "llmnb.rts.v2"
    assert versions.PACKAGE_VERSION == "0.1.0"
    assert versions.CLIENT_VERSION == "0.1.0"
    assert versions.RFC_VERSIONS == {
        "rfc_001_version": "1.0.0",
        "rfc_002_version": "1.0.0",
        "rfc_003_version": "1.0.0",
        "rfc_004_version": "1.0.0",
        "rfc_005_version": "1.0.0",
        "rfc_006_version": "2.0.0",
        "rfc_007_version": "1.0.0",
        "rfc_008_version": "1.0.0",
    }
    assert versions.KERNEL_CAPABILITIES == [
        "family_a", "family_b", "family_c", "family_f", "family_g",
    ]


def test_public_names_forward_to_source() -> None:
    """Every pre-existing public name is importable where it was and reads the source."""
    assert pty_mode.KERNEL_VERSION == versions.KERNEL_VERSION
    assert pty_mode.RFC_VERSIONS is versions.RFC_VERSIONS
    assert serve_mode.KERNEL_VERSION == versions.KERNEL_VERSION
    assert serve_mode.KERNEL_CAPABILITIES is versions.KERNEL_CAPABILITIES
    assert serve_mode.WIRE_VERSION == versions.WIRE_VERSION
    assert serve_mode.WIRE_MAJOR == versions.semver_major(versions.WIRE_VERSION)
    assert wire.WIRE_VERSION == versions.WIRE_VERSION
    assert (wire.WIRE_MAJOR, wire.WIRE_MINOR, wire.WIRE_PATCH) == versions.wire_version_tuple()
    assert wire_version.wire_version_tuple() == versions.semver_tuple(versions.WIRE_VERSION)
    assert wire.KERNEL_VERSION == versions.KERNEL_VERSION
    assert wire.RFC_VERSIONS is versions.RFC_VERSIONS
    assert wire.KERNEL_CAPABILITIES is versions.KERNEL_CAPABILITIES
    assert mw_pkg.SCHEMA_VERSION == versions.SCHEMA_VERSION
    assert _serialize.SCHEMA_VERSION == versions.SCHEMA_VERSION
    assert run_envelope.RFC003_VERSION == versions.rfc_version(3)
    assert run_tracker.RFC003_VERSION == versions.rfc_version(3)
    assert custom_messages.DEFAULT_COMM_TARGET == versions.COMM_TARGET
    assert llm_kernel.__version__ == versions.PACKAGE_VERSION
    assert wire_tools._VER["default"] == versions.rfc_version(1)  # noqa: SLF001
    # Star-import / explicit-import paths still work for forwarded names.
    from llm_kernel.wire import WIRE_MAJOR, WIRE_VERSION  # noqa: F401
    from llm_kernel.capture.metadata_writer import SCHEMA_VERSION  # noqa: F401
    from llm_kernel.capture.run_tracker import RFC003_VERSION  # noqa: F401
    from llm_kernel.transports.pty_mode import KERNEL_VERSION, RFC_VERSIONS  # noqa: F401
    from llm_kernel.transports.serve_mode import KERNEL_CAPABILITIES  # noqa: F401


def test_mcp_server_rfc001_version_forwards_to_source() -> None:
    mcp_server = pytest.importorskip("llm_kernel.mcp.mcp_server")
    assert mcp_server.RFC_001_VERSION == versions.rfc_version(1)
    from llm_kernel.mcp.mcp_server import RFC_001_VERSION  # noqa: F401


def test_ready_frame_derives_from_source() -> None:
    attrs = _ready_frame_attrs()
    assert attrs["llmnb.kernel.version"] == versions.KERNEL_VERSION
    assert _ready_frame_rfc_versions() == versions.RFC_VERSIONS
    for key, value in versions.ready_frame_attrs().items():
        assert attrs[key] == value


def test_handshake_payloads_and_validation_derive_from_source() -> None:
    ok = serve_mode._handshake_response("sess-1")["payload"]  # noqa: SLF001
    assert ok["kernel_version"] == versions.KERNEL_VERSION
    assert ok["wire_version"] == versions.WIRE_VERSION
    assert ok["accepted_capabilities"] == versions.KERNEL_CAPABILITIES
    assert ok["accepted_capabilities"] is not versions.KERNEL_CAPABILITIES  # a copy

    err = _handshake_error_payload("auth_failed")
    assert err["kernel_version"] == versions.KERNEL_VERSION
    assert err["wire_version"] == versions.WIRE_VERSION
    assert err["error"] == "auth_failed"

    def _req(wire_ver: str) -> Dict[str, Any]:
        return {"type": "kernel.handshake", "payload": {
            "client_name": "t", "client_version": versions.CLIENT_VERSION,
            "wire_version": wire_ver, "transport": "tcp",
            "auth": {"scheme": "bearer", "token": "tok"},
        }}

    assert serve_mode._validate_handshake(  # noqa: SLF001
        _req(versions.WIRE_VERSION), expected_token="tok",
    ) == (None, [])
    major = versions.semver_major(versions.WIRE_VERSION)
    assert serve_mode._validate_handshake(  # noqa: SLF001
        _req(f"{major + 1}.0.0"), expected_token="tok",
    )[0] == "version_mismatch_major"
    assert serve_mode._validate_handshake(  # noqa: SLF001
        _req(f"{major}.99.0"), expected_token="tok",
    ) == (None, ["minor_version_skew"])


def test_volatile_provider_covers_every_rfc_key_from_source() -> None:
    provided = pty_mode._current_volatile()  # noqa: SLF001
    assert set(provided) == {"kernel"}
    kernel = provided["kernel"]
    assert kernel["kernel_version"] == versions.KERNEL_VERSION
    for key, value in versions.RFC_VERSIONS.items():
        assert kernel[key] == value
    assert kernel == versions.kernel_volatile()


def test_metadata_writer_stamps_and_checks_source_schema_version() -> None:
    writer = MetadataWriter(autosave_interval_sec=999.0)
    try:
        assert writer.snapshot()["schema_version"] == versions.SCHEMA_VERSION
        major = versions.semver_major(versions.SCHEMA_VERSION)
        with pytest.raises(ValueError, match="schema_version major"):
            writer.hydrate({"schema_version": f"{major + 1}.0.0"})
    finally:
        writer.stop()


def test_notebook_format_default_schema_version_from_source() -> None:
    doc = notebook_format.magic_to_llmnb("print(1)")
    assert doc["metadata"]["rts"]["schema_version"] == versions.SCHEMA_VERSION


def test_health_endpoint_and_fastapi_version_from_source() -> None:
    pytest.importorskip("fastapi")
    from llm_kernel.app import app
    assert app.version == versions.KERNEL_VERSION
    body = _health()
    assert body["kernel_version"] == versions.KERNEL_VERSION
    assert body["kernel_version"] != "unknown"


def test_run_envelope_and_tracker_use_source_rfc003_version() -> None:
    env = run_envelope.make_envelope(
        "run.start", {}, "0123456789abcdef", direction=run_envelope.DIRECTION_KERNEL_TO_EXTENSION,
    )
    assert env["rfc_version"] == versions.rfc_version(3)


def test_exported_schema_json_matches_source() -> None:
    for name in ("family-a", "family-b", "family-c", "family-f", "family-g",
                 "handshake.request", "handshake.response"):
        data = json.loads((_SCHEMAS_DIR / f"{name}.json").read_text(encoding="utf-8"))
        assert data["wire_version"] == versions.WIRE_VERSION, name
    for path in sorted(_SCHEMAS_DIR.glob("tool.*.input.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        rfc = data.get("properties", {}).get("_rfc_version", {})
        if "default" in rfc:
            assert rfc["default"] == versions.rfc_version(1), path.name
    # The in-memory builders agree with the on-disk JSON.
    fam = wire_export.family_schemas()
    assert fam["family-a"]["wire_version"] == versions.WIRE_VERSION
    assert wire_export.FAMILY_SCHEMAS["family-g"]["wire_version"] == versions.WIRE_VERSION
    assert wire_export.handshake_schemas()["handshake.response"]["wire_version"] == versions.WIRE_VERSION


# ---------------------------------------------------------------------------
# 2. Drift regression
# ---------------------------------------------------------------------------


def test_persisted_ready_frame_versions_produce_zero_drift() -> None:
    """The symptom: persisted == ready frame, current == provider -> no drift."""
    detector = DriftDetector()
    persisted_from_ready_frame = _ready_frame_rfc_versions()
    assert persisted_from_ready_frame  # sanity: the frame carried RFC versions

    # Provider output as the dispatcher hands it over (``{"kernel": {...}}``).
    events = detector.compare(
        _persisted(persisted_from_ready_frame),
        current_kernel=pty_mode._current_volatile(),  # noqa: SLF001
    )
    assert events == []

    # Flat ``config.volatile.kernel`` shape (what unit tests inject).
    events = detector.compare(
        _persisted(persisted_from_ready_frame),
        current_kernel=pty_mode._current_volatile()["kernel"],  # noqa: SLF001
    )
    assert events == []

    # Persisting the module attributes directly is the same thing.
    events = detector.compare(
        _persisted(dict(pty_mode.RFC_VERSIONS)),
        current_kernel=pty_mode._current_volatile(),  # noqa: SLF001
    )
    assert events == []


def test_drift_detector_compares_every_rfc_key_the_source_defines() -> None:
    """A mismatch on rfc_004..008 (previously never checked) is now reported."""
    detector = DriftDetector()
    for key in versions.RFC_VERSIONS:
        persisted = dict(versions.RFC_VERSIONS)
        major = versions.semver_major(persisted[key])
        persisted[key] = f"{major + 1}.0.0"
        events = detector.compare(
            _persisted(persisted),
            current_kernel=pty_mode._current_volatile(),  # noqa: SLF001
        )
        assert [e["field_path"] for e in events] == [f"config.volatile.kernel.{key}"], key
        assert events[0]["severity"] == "error"
        assert events[0]["current_value"] == versions.RFC_VERSIONS[key]

    # Minor skew on a late key is a warning, not an error.
    persisted = dict(versions.RFC_VERSIONS)
    major = versions.semver_major(persisted["rfc_008_version"])
    persisted["rfc_008_version"] = f"{major}.7.0"
    events = detector.compare(
        _persisted(persisted), current_kernel=versions.kernel_volatile(),
    )
    assert len(events) == 1 and events[0]["severity"] == "warn"


def test_dispatcher_hydrate_with_real_provider_and_detector_reports_no_drift() -> None:
    """End-to-end through the dispatcher: real DriftDetector + pty provider."""
    from llm_kernel.capture.custom_messages import CustomMessageDispatcher

    class _CommManager:
        def __init__(self) -> None:
            self.targets: Dict[str, Any] = {}

        def register_target(self, name: str, cb: Any) -> None:
            self.targets[name] = cb

        def unregister_target(self, name: str, cb: Any) -> None:
            self.targets.pop(name, None)

    kernel = MagicMock()
    kernel.shell.comm_manager = _CommManager()
    kernel._parent_header = {}
    dispatcher = CustomMessageDispatcher(kernel, heartbeat_interval_sec=300.0)
    writer = MagicMock()
    supervisor = MagicMock()
    supervisor.respawn_from_config.return_value = {}
    dispatcher.set_metadata_writer(writer)
    dispatcher.set_agent_supervisor(supervisor)
    dispatcher.set_drift_detector(DriftDetector())
    dispatcher.set_current_volatile_provider(pty_mode._current_volatile)  # noqa: SLF001

    snap = _persisted(_ready_frame_rfc_versions())
    snap.update({"session_id": "s", "snapshot_version": 1, "layout": {"tree": {}},
                 "blobs": {}, "drift_log": []})
    writer.snapshot.return_value = snap
    dispatcher.start()
    try:
        kernel.shell.comm_manager.targets[custom_messages.DEFAULT_COMM_TARGET](
            MagicMock(name="comm"), {"content": {}},
        )
        dispatcher._on_comm_msg({"content": {"data": {  # noqa: SLF001
            "type": "notebook.metadata",
            "payload": {"mode": "hydrate", "snapshot_version": 1,
                        "snapshot": snap, "trigger": "open"},
        }}})
        writer.hydrate.assert_called_once()
        writer.append_drift_event.assert_not_called()
    finally:
        dispatcher.stop()


# ---------------------------------------------------------------------------
# 3. Runtime override / in-place mutation propagates
# ---------------------------------------------------------------------------


def test_in_place_mutation_of_source_propagates_everywhere(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(versions.RFC_VERSIONS, "rfc_006_version", "3.1.4")
    monkeypatch.setitem(versions.RFC_VERSIONS, "rfc_009_version", "0.9.0")
    monkeypatch.setattr(versions, "KERNEL_VERSION", "9.8.7")
    versions.KERNEL_CAPABILITIES.append("family_z")
    try:
        # Ready frame
        attrs = _ready_frame_attrs()
        assert attrs["llmnb.kernel.version"] == "9.8.7"
        assert attrs["llmnb.kernel.rfc_006_version"] == "3.1.4"
        assert attrs["llmnb.kernel.rfc_009_version"] == "0.9.0"
        # Handshake (success + error) and the capability echo
        ok = serve_mode._handshake_response("s")["payload"]  # noqa: SLF001
        assert ok["kernel_version"] == "9.8.7"
        assert "family_z" in ok["accepted_capabilities"]
        assert _handshake_error_payload("kernel_busy")["kernel_version"] == "9.8.7"
        # Volatile provider
        kernel = pty_mode._current_volatile()["kernel"]  # noqa: SLF001
        assert kernel["kernel_version"] == "9.8.7"
        assert kernel["rfc_006_version"] == "3.1.4"
        assert kernel["rfc_009_version"] == "0.9.0"
        # Forwarded public names
        assert pty_mode.KERNEL_VERSION == "9.8.7"
        assert serve_mode.KERNEL_VERSION == "9.8.7"
        assert wire.KERNEL_VERSION == "9.8.7"
        assert pty_mode.RFC_VERSIONS["rfc_006_version"] == "3.1.4"
        assert "family_z" in serve_mode.KERNEL_CAPABILITIES
        # Health endpoint
        assert _health()["kernel_version"] == "9.8.7"
        # Drift: persisted == new ready frame -> still zero events; the
        # new rfc_009 key is compared too.
        detector = DriftDetector()
        assert detector.compare(
            _persisted(_ready_frame_rfc_versions()),
            current_kernel=pty_mode._current_volatile(),  # noqa: SLF001
        ) == []
        stale = _ready_frame_rfc_versions()
        stale["rfc_009_version"] = "1.0.0"
        events = detector.compare(
            _persisted(stale), current_kernel=pty_mode._current_volatile(),  # noqa: SLF001
        )
        assert [e["field_path"] for e in events] == ["config.volatile.kernel.rfc_009_version"]
        assert events[0]["severity"] == "error"
    finally:
        versions.KERNEL_CAPABILITIES.remove("family_z")


def test_wire_version_override_propagates_to_components_and_checks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(versions, "WIRE_VERSION", "2.5.1")
    assert (wire.WIRE_MAJOR, wire.WIRE_MINOR, wire.WIRE_PATCH) == (2, 5, 1)
    assert wire_version.WIRE_MAJOR == 2 and wire_version.wire_version_tuple() == (2, 5, 1)
    assert serve_mode.WIRE_MAJOR == 2 and serve_mode.WIRE_VERSION == "2.5.1"
    # Handshake validation now rejects the old major and emits the new string.
    req = {"type": "kernel.handshake", "payload": {
        "client_name": "t", "client_version": "0.0.1", "wire_version": "1.0.0",
        "transport": "tcp", "auth": {"scheme": "bearer", "token": "tok"},
    }}
    assert serve_mode._validate_handshake(req, expected_token="tok")[0] == "version_mismatch_major"  # noqa: SLF001
    assert serve_mode._handshake_response("s")["payload"]["wire_version"] == "2.5.1"  # noqa: SLF001
    # Event-log replay check reads the live major.
    from llm_kernel.capture.event_log import EventLogVersionMismatchError, _check_envelope_version
    with pytest.raises(EventLogVersionMismatchError):
        _check_envelope_version({"rfc_version": "1.0.0"})
    _check_envelope_version({"rfc_version": "2.5.1"})
    # Exporter stamps the live value.
    assert wire_export.family_schemas()["family-a"]["wire_version"] == "2.5.1"
    written = wire_export.export_schemas(tmp_path)
    fam = json.loads((tmp_path / "family-a.json").read_text(encoding="utf-8"))
    assert fam["wire_version"] == "2.5.1" and written


def test_schema_and_rfc003_override_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(versions, "SCHEMA_VERSION", "4.0.0")
    assert mw_pkg.SCHEMA_VERSION == "4.0.0" and _serialize.SCHEMA_VERSION == "4.0.0"
    writer = MetadataWriter(autosave_interval_sec=999.0)
    try:
        assert writer.snapshot()["schema_version"] == "4.0.0"
        with pytest.raises(ValueError, match="schema_version major"):
            writer.hydrate({"schema_version": "1.0.0"})
    finally:
        writer.stop()
    assert notebook_format.magic_to_llmnb("x = 1")["metadata"]["rts"]["schema_version"] == "4.0.0"

    monkeypatch.setitem(versions.RFC_VERSIONS, "rfc_003_version", "1.2.0")
    assert run_envelope.RFC003_VERSION == "1.2.0"
    assert run_tracker.RFC003_VERSION == "1.2.0"
    assert run_envelope.make_envelope("run.start", {}, "0123456789abcdef")["rfc_version"] == "1.2.0"


# ---------------------------------------------------------------------------
# Driver: connect defaults derive from the source (loopback responder)
# ---------------------------------------------------------------------------


def _loopback_handshake(
    monkeypatch: pytest.MonkeyPatch, respond: Dict[str, Any],
) -> Dict[str, Any]:
    """Run ``tcp.connect`` against a one-shot responder; return the request it sent."""
    tcp = pytest.importorskip("llm_client.transport.tcp")
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    seen: Dict[str, Any] = {}

    def _serve() -> None:
        conn, _ = srv.accept()
        with conn:
            buf = b""
            while b"\n" not in buf:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                buf += chunk
            seen.update(json.loads(buf.split(b"\n", 1)[0].decode("utf-8")))
            conn.sendall((json.dumps(respond) + "\n").encode("utf-8"))

    thread = threading.Thread(target=_serve, daemon=True)
    thread.start()
    try:
        conn = tcp.connect(bind=f"127.0.0.1:{port}", token="tok", timeout=5.0)
        conn.close()
    finally:
        thread.join(timeout=5.0)
        srv.close()
    return seen


def test_driver_connect_defaults_derive_from_source(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("llm_client.transport.tcp")
    response = {"type": "kernel.handshake", "payload": {
        "kernel_version": versions.KERNEL_VERSION,
        "wire_version": versions.WIRE_VERSION,
        "session_id": "s",
        "accepted_capabilities": list(versions.KERNEL_CAPABILITIES),
    }}
    req = _loopback_handshake(monkeypatch, response)["payload"]
    assert req["wire_version"] == versions.WIRE_VERSION
    assert req["client_version"] == versions.CLIENT_VERSION
    assert req["capabilities"] == versions.KERNEL_CAPABILITIES

    # Override the source: the driver's next connect follows it.
    monkeypatch.setattr(versions, "WIRE_VERSION", "7.0.0")
    monkeypatch.setattr(versions, "CLIENT_VERSION", "0.2.0")
    versions.KERNEL_CAPABILITIES.append("family_q")
    try:
        response["payload"]["wire_version"] = "7.0.0"
        response["payload"]["accepted_capabilities"] = list(versions.KERNEL_CAPABILITIES)
        req = _loopback_handshake(monkeypatch, response)["payload"]
        assert req["wire_version"] == "7.0.0"
        assert req["client_version"] == "0.2.0"
        assert "family_q" in req["capabilities"]
    finally:
        versions.KERNEL_CAPABILITIES.remove("family_q")


# ---------------------------------------------------------------------------
# 4. Static checks
# ---------------------------------------------------------------------------


def _package_py_files(root: Path) -> List[Path]:
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def test_no_semver_literal_outside_source_module_in_kernel() -> None:
    violations: List[str] = []
    for py_file in _package_py_files(_KERNEL_PKG):
        if py_file.resolve() == _SOURCE_MODULE:
            continue
        violations.extend(_semver_literals(py_file))
    assert not violations, (
        "MAJOR.MINOR.PATCH string literals outside llm_kernel/wire/versions.py:\n"
        + "\n".join(violations)
    )
    # And the source module itself does define them.
    assert _semver_literals(_SOURCE_MODULE)


def test_no_semver_literal_in_driver_package() -> None:
    spec = importlib.util.find_spec("llm_client")
    if spec is None or not spec.submodule_search_locations:
        pytest.skip("llm_client not importable from this environment")
    root = Path(next(iter(spec.submodule_search_locations)))
    violations: List[str] = []
    for py_file in _package_py_files(root):
        violations.extend(_semver_literals(py_file))
    assert not violations, (
        "MAJOR.MINOR.PATCH string literals in the driver (must read llm_kernel.wire.versions):\n"
        + "\n".join(violations)
    )


def test_previously_drifted_values_are_gone() -> None:
    """The provider's stale 1.0.1 / 2.0.2 copies no longer exist anywhere in the kernel."""
    offenders: List[str] = []
    for py_file in _package_py_files(_KERNEL_PKG):
        text = py_file.read_text(encoding="utf-8")
        for needle in ('"1.0.1"', '"2.0.2"'):
            if needle in text:
                offenders.append(f"{py_file}: {needle}")
    assert not offenders, offenders


def _import_error_guarded(tree: ast.AST) -> set[int]:
    """ids of import nodes inside a ``try`` whose handlers catch ImportError."""
    guarded: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        catches_import_error = False
        for handler in node.handlers:
            t = handler.type
            names = [t] if t is not None and not isinstance(t, ast.Tuple) else (t.elts if t else [])
            for n in names:
                if isinstance(n, ast.Name) and n.id in ("ImportError", "ModuleNotFoundError"):
                    catches_import_error = True
        if not catches_import_error:
            continue
        # Both the attempted import (try body) and the fallback import
        # (handler body) are optional by construction.
        scopes = list(node.body) + [stmt for h in node.handlers for stmt in h.body]
        for sub in scopes:
            for inner in ast.walk(sub):
                if isinstance(inner, (ast.Import, ast.ImportFrom)):
                    guarded.add(id(inner))
    return guarded


def test_wire_package_is_stdlib_only() -> None:
    """llm_kernel.wire (the driver's only import surface) imports stdlib + itself only.

    Imports inside ``try: ... except ImportError`` are optional accelerators
    (``jsonschema`` in tools.py, the ``typing_extensions`` fallback in
    families.py) and pre-date this module; they are tolerated.  Nothing in
    the package may import ``llm_kernel`` outside ``llm_kernel.wire``.
    """
    stdlib = set(getattr(sys, "stdlib_module_names", ()))
    assert stdlib, "sys.stdlib_module_names required (Python 3.10+)"
    violations: List[str] = []
    for py_file in _package_py_files(_KERNEL_PKG / "wire"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        guarded = _import_error_guarded(tree)
        for node in ast.walk(tree):
            if id(node) in guarded:
                continue
            names: List[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:  # relative import inside the package: fine
                    continue
                names = [node.module or ""]
            for name in names:
                top = name.split(".", 1)[0]
                if top == "llm_kernel":
                    if not (name == "llm_kernel.wire" or name.startswith("llm_kernel.wire.")):
                        violations.append(f"{py_file.name}: {name}")
                elif top not in stdlib and top != "__future__":
                    violations.append(f"{py_file.name}: {name}")
    assert not violations, violations
    # The source module in particular imports nothing from llm_kernel at all.
    src_tree = ast.parse(_SOURCE_MODULE.read_text(encoding="utf-8"))
    for node in ast.walk(src_tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            assert not any(m.startswith("llm_kernel") for m in mods), mods


def test_source_module_reloads_to_the_same_values() -> None:
    """Re-importing the source does not change identity semantics for forwarders."""
    before = dict(versions.RFC_VERSIONS)
    mod = importlib.import_module("llm_kernel.wire.versions")
    assert mod is versions
    assert mod.RFC_VERSIONS == before
