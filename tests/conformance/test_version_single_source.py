"""One source for every version the kernel puts on the wire.

Sources: ``llm_kernel/wire/versions.py`` (the single source: the ready
frame, the serve handshake and its validation, the drift provider, the
metadata writer's schema stamp, the notebook-format default, ``/health``,
the exported schemas and the driver defaults read *through* it at call
time; bumping a version is a one-line edit there),
``docs/04 - Reference/atoms/discipline/wire-as-public-api.md`` (WIRE_VERSION
and its components; major mismatch -> refused, minor skew -> warning),
RFC-008 §4 (``kernel.ready`` carries ``llmnb.kernel.version`` and
``llmnb.kernel.rfc_001_version`` .. ``rfc_008_version``).
"""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

import pytest

import llm_kernel
import llm_kernel.wire as wire
from llm_kernel.wire import version as wire_version
from llm_kernel.wire import versions as _versions

from tests.conformance._support import (
    WAIT_SECONDS,
    WirePeer,
    connect_with_retry,
    handshake_request,
    is_ready_frame,
    otlp_attrs,
    run_serve_main_with_client,
)

KERNEL_ROOT = Path(__file__).resolve().parents[2]
SEMVER = r"^\d+\.\d+\.\d+$"


def test_source_defines_every_wire_version() -> None:
    import re

    for name in ("KERNEL_VERSION", "WIRE_VERSION", "SCHEMA_VERSION",
                 "PACKAGE_VERSION", "CLIENT_VERSION"):
        assert re.match(SEMVER, getattr(_versions, name)), name
    assert list(_versions.RFC_VERSIONS) == [f"rfc_{n:03d}_version" for n in range(1, 9)]
    assert all(re.match(SEMVER, v) for v in _versions.RFC_VERSIONS.values())
    assert _versions.COMM_TARGET == "llmnb.rts.v2"
    assert _versions.kernel_capabilities() == list(_versions.KERNEL_CAPABILITIES)
    assert _versions.kernel_capabilities() is not _versions.KERNEL_CAPABILITIES
    assert _versions.rfc_versions() == dict(_versions.RFC_VERSIONS)
    assert _versions.rfc_version(6) == _versions.RFC_VERSIONS["rfc_006_version"]
    assert _versions.rfc_key(3) == "rfc_003_version"


def test_semver_parsing_is_strict() -> None:
    assert _versions.semver_tuple("12.3.4") == (12, 3, 4)
    assert _versions.semver_major("3.1.0") == 3
    for bad in ("1.0", "1.0.0-rc1", "v1.0.0", "", "a.b.c", None):
        with pytest.raises(ValueError):
            _versions.semver_tuple(bad)  # type: ignore[arg-type]


def test_wire_version_components_derive_from_the_string(monkeypatch: pytest.MonkeyPatch) -> None:
    """wire-as-public-api: WIRE_MAJOR/MINOR/PATCH never disagree with
    WIRE_VERSION; ``llm_kernel.wire`` forwards the live values."""
    assert wire_version.wire_version_tuple() == _versions.semver_tuple(_versions.WIRE_VERSION)
    monkeypatch.setattr(_versions, "WIRE_VERSION", "4.5.6")
    assert (wire_version.WIRE_MAJOR, wire_version.WIRE_MINOR, wire_version.WIRE_PATCH) == (4, 5, 6)
    assert wire_version.WIRE_VERSION == "4.5.6"
    assert (wire.WIRE_VERSION, wire.WIRE_MAJOR, wire.WIRE_MINOR, wire.WIRE_PATCH) == ("4.5.6", 4, 5, 6)
    assert wire.wire_version_tuple() == (4, 5, 6)


def test_public_wire_names_forward_live(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_versions, "KERNEL_VERSION", "8.8.8")
    monkeypatch.setattr(_versions, "SCHEMA_VERSION", "3.2.1")
    monkeypatch.setattr(_versions, "CLIENT_VERSION", "0.9.9")
    monkeypatch.setitem(_versions.RFC_VERSIONS, "rfc_007_version", "1.7.0")
    assert wire.KERNEL_VERSION == "8.8.8"
    assert wire.SCHEMA_VERSION == "3.2.1"
    assert wire.CLIENT_VERSION == "0.9.9"
    assert wire.RFC_VERSIONS["rfc_007_version"] == "1.7.0"
    assert wire.versions is _versions
    from llm_kernel.transports import pty_mode, serve_mode

    assert pty_mode.KERNEL_VERSION == "8.8.8"
    assert pty_mode.RFC_VERSIONS["rfc_007_version"] == "1.7.0"
    assert serve_mode.KERNEL_VERSION == "8.8.8"


def test_package_version_derives_from_the_source() -> None:
    assert llm_kernel.__version__ == _versions.PACKAGE_VERSION
    assert importlib.metadata.version("llm-kernel") == _versions.PACKAGE_VERSION


def _setup_py_version(root: Path, tmp_path: Path) -> str:
    """Execute ``setup.py`` with a stub ``setuptools`` capturing ``version``."""
    stub = tmp_path / "stub"
    stub.mkdir(exist_ok=True)
    (stub / "setuptools.py").write_text(
        "import json, sys\n"
        "def find_packages(*a, **k):\n    return []\n"
        "def setup(**kw):\n"
        "    sys.stdout.write(json.dumps({'version': kw.get('version')}))\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-S", str(root / "setup.py")],
        cwd=str(root), env={"PYTHONPATH": str(stub), "PATH": "/usr/bin:/bin"},
        capture_output=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    return json.loads(proc.stdout.decode("utf-8"))["version"]


def test_setup_py_reads_the_version_from_the_source(tmp_path: Path) -> None:
    """A fresh build takes the distribution version from
    ``llm_kernel/wire/versions.py`` -- never a stale literal or the text
    of an ``__version__ = ...`` expression."""
    assert _setup_py_version(KERNEL_ROOT, tmp_path) == _versions.PACKAGE_VERSION
    fake = tmp_path / "fake"
    (fake / "llm_kernel" / "wire").mkdir(parents=True)
    shutil.copy(KERNEL_ROOT / "setup.py", fake / "setup.py")
    for extra in ("requirements.txt", "README.md"):
        if (KERNEL_ROOT / extra).exists():
            shutil.copy(KERNEL_ROOT / extra, fake / extra)
    source = (KERNEL_ROOT / "llm_kernel" / "wire" / "versions.py").read_text(encoding="utf-8")
    source = source.replace(
        f'PACKAGE_VERSION: str = "{_versions.PACKAGE_VERSION}"',
        'PACKAGE_VERSION: str = "9.8.7"',
    )
    assert '"9.8.7"' in source
    (fake / "llm_kernel" / "wire" / "versions.py").write_text(source, encoding="utf-8")
    assert _setup_py_version(fake, tmp_path) == "9.8.7"


def test_ready_frame_reads_the_source_at_call_time(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """RFC-008 §4 ready frame + boot banner, under a runtime override."""
    from llm_kernel.transports import pty_mode

    monkeypatch.setattr(_versions, "KERNEL_VERSION", "5.4.3")
    monkeypatch.setitem(_versions.RFC_VERSIONS, "rfc_008_version", "1.8.0")
    peer = WirePeer()
    monkeypatch.setenv(pty_mode.ENV_IPC_SOCKET, peer.address)
    monkeypatch.setenv(pty_mode.ENV_SESSION_ID, "conformance-ready")
    state = pty_mode.boot_kernel(workdir=tmp_path)
    try:
        ready = peer.wait_for(is_ready_frame)
    finally:
        pty_mode.shutdown_kernel(state)
        peer.close()
    assert ready is not None and is_ready_frame(peer.snapshot()[0])
    attrs = otlp_attrs(ready)
    assert attrs["event.name"] == "kernel.ready"
    assert attrs["llmnb.kernel.version"] == "5.4.3"
    assert attrs["llmnb.kernel.session_id"] == "conformance-ready"
    for key, value in _versions.RFC_VERSIONS.items():
        assert attrs[f"llmnb.kernel.{key}"] == value
    assert attrs["llmnb.kernel.rfc_008_version"] == "1.8.0"
    expected = _versions.ready_frame_attrs()
    assert {k: attrs[k] for k in expected} == expected
    assert "LLMKernel pty-mode v5.4.3" in capsys.readouterr().err


def test_serve_handshake_reads_the_source_at_call_time(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """Handshake response + validation follow the live source:
    kernel_version, wire_version, accepted capabilities; a client whose
    wire major differs is refused, a minor skew is a warning."""
    import threading

    if threading.current_thread() is not threading.main_thread():
        pytest.skip("serve installs signal handlers; needs the main thread")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(_versions, "KERNEL_VERSION", "6.0.1")
    monkeypatch.setattr(_versions, "WIRE_VERSION", "2.3.0")
    monkeypatch.setattr(_versions, "KERNEL_CAPABILITIES", ["family_a", "family_x"])

    def client(port: int) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for label, request in (
            ("same", handshake_request("conformance-token", wire_version="2.3.0")),
            ("minor", handshake_request("conformance-token", wire_version="2.1.0")),
            ("major", handshake_request("conformance-token", wire_version="1.3.0")),
            ("auth", handshake_request("wrong-token", wire_version="2.3.0")),
        ):
            out[label] = _handshake_once(port, request)
        return out

    rc, out = run_serve_main_with_client(monkeypatch, client)
    assert rc == 0
    same = out["same"]
    assert "error" not in same
    assert same["kernel_version"] == "6.0.1"
    assert same["wire_version"] == "2.3.0"
    assert same["accepted_capabilities"] == ["family_a", "family_x"]
    assert out["minor"].get("warnings") == ["minor_version_skew"]
    assert "error" not in out["minor"]
    assert out["major"]["error"] == "version_mismatch_major"
    assert out["major"]["wire_version"] == "2.3.0"
    assert out["auth"]["error"] == "auth_failed"
    err = capsys.readouterr().err
    assert "llm_kernel serve v6.0.1" in err and "wire_version=2.3.0" in err


def _handshake_once(port: int, request: Dict[str, Any]) -> Dict[str, Any]:
    """One handshake; retried while the previous connection is still being
    released (V1 serves one connection at a time -> ``kernel_busy``)."""
    deadline = time.monotonic() + WAIT_SECONDS
    while True:
        conn = connect_with_retry("127.0.0.1", port)
        try:
            conn.send(request)
            payload = (conn.recv() or {}).get("payload") or {}
        finally:
            conn.close()
        if payload.get("error") != "kernel_busy" or time.monotonic() > deadline:
            return payload
        time.sleep(0.1)


def test_health_endpoint_reports_the_source_kernel_version(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from starlette.testclient import TestClient

    from llm_kernel import app as app_module

    assert app_module.app.version == _versions.KERNEL_VERSION
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(_versions, "KERNEL_VERSION", "7.1.2")
    peer = WirePeer()
    monkeypatch.setenv("LLMKERNEL_IPC_SOCKET", peer.address)
    try:
        with TestClient(app_module.app) as client:
            body = client.get("/health").json()
        ready = peer.wait_for(is_ready_frame)
    finally:
        peer.close()
    assert body["kernel_version"] == "7.1.2"
    assert ready is not None and otlp_attrs(ready)["llmnb.kernel.version"] == "7.1.2"


def test_metadata_snapshots_stamp_the_source_schema_version(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from llm_kernel.capture import metadata_writer as mw_pkg
    from llm_kernel.capture.metadata_writer import MetadataWriter

    writer = MetadataWriter(workspace_root=tmp_path, autosave_interval_sec=3600.0)
    try:
        assert writer.snapshot(trigger="save")["schema_version"] == _versions.SCHEMA_VERSION
        monkeypatch.setattr(_versions, "SCHEMA_VERSION", "1.9.0")
        assert writer.snapshot(trigger="save")["schema_version"] == "1.9.0"
        assert mw_pkg.SCHEMA_VERSION == "1.9.0"
    finally:
        writer.stop(emit_final=False)


def test_metadata_hydrate_checks_the_source_schema_major(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from llm_kernel.capture.metadata_writer import MetadataWriter

    major = _versions.semver_major(_versions.SCHEMA_VERSION)
    writer = MetadataWriter(workspace_root=tmp_path, autosave_interval_sec=3600.0)
    try:
        writer.hydrate({"schema_version": f"{major}.7.0", "config": {}})
        with pytest.raises(ValueError):
            writer.hydrate({"schema_version": f"{major + 1}.0.0", "config": {}})
        monkeypatch.setattr(_versions, "SCHEMA_VERSION", f"{major + 1}.0.0")
        writer.hydrate({"schema_version": f"{major + 1}.2.0", "config": {}})
    finally:
        writer.stop(emit_final=False)


def test_notebook_format_default_schema_version_is_the_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_kernel.notebook import notebook_format

    doc = notebook_format.magic_to_llmnb("@@scratch\nhello")
    rts = doc["metadata"]["rts"]
    assert rts["schema_version"] == _versions.SCHEMA_VERSION
    monkeypatch.setattr(_versions, "SCHEMA_VERSION", "1.4.0")
    doc = notebook_format.magic_to_llmnb("@@scratch\nhello")
    assert doc["metadata"]["rts"]["schema_version"] == "1.4.0"


def test_exported_schemas_carry_the_source_wire_version(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The committed ``wire/schemas/*.json`` mirror a fresh export, and an
    export carries the live WIRE_VERSION."""
    from llm_kernel.wire import export

    committed = KERNEL_ROOT / "llm_kernel" / "wire" / "schemas"
    fresh = tmp_path / "fresh"
    written = export.export_schemas(fresh)
    assert written
    for path in written:
        assert (committed / path.name).read_text(encoding="utf-8") == path.read_text(encoding="utf-8"), path.name
    for name in ("family-a.json", "handshake.request.json", "handshake.response.json"):
        assert json.loads((fresh / name).read_text(encoding="utf-8"))["wire_version"] == _versions.WIRE_VERSION
    monkeypatch.setattr(_versions, "WIRE_VERSION", "3.0.0")
    bumped = tmp_path / "bumped"
    export.export_schemas(bumped)
    for name in ("family-f.json", "handshake.response.json"):
        assert json.loads((bumped / name).read_text(encoding="utf-8"))["wire_version"] == "3.0.0"


def test_wire_versions_module_is_stdlib_only() -> None:
    """versions.py rules: no imports of other llm_kernel modules (it is
    importable by external drivers via ``llm_kernel.wire``)."""
    import ast

    tree = ast.parse((KERNEL_ROOT / "llm_kernel" / "wire" / "versions.py").read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    imported.discard("__future__")
    assert imported <= set(sys.stdlib_module_names), imported
