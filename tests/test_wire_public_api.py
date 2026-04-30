"""S5.0.3a lint check: llm_kernel.wire public API surface.

Per PLAN-S5.0.3 §8.1.  Asserts that the public surface is importable,
shaped correctly, and that ``_rfc_schemas`` aliases preserve identity.
"""

from __future__ import annotations

import importlib
import json
import re
import types
from pathlib import Path


# ---------------------------------------------------------------------------
# Import surface
# ---------------------------------------------------------------------------

def test_wire_package_importable() -> None:
    """``from llm_kernel.wire import ...`` succeeds without error."""
    from llm_kernel.wire import (  # noqa: F401
        WIRE_VERSION,
        Envelope,
        TOOL_CATALOG,
        validate_tool_input,
    )


def test_wire_version_is_semver() -> None:
    """WIRE_VERSION is a non-empty string matching semver shape."""
    from llm_kernel.wire import WIRE_VERSION
    assert isinstance(WIRE_VERSION, str)
    assert WIRE_VERSION, "WIRE_VERSION must not be empty"
    assert re.match(r"^\d+\.\d+\.\d+$", WIRE_VERSION), (
        f"WIRE_VERSION {WIRE_VERSION!r} does not match semver pattern"
    )


def test_wire_version_parts_are_integers() -> None:
    """WIRE_MAJOR, WIRE_MINOR, WIRE_PATCH are integers."""
    from llm_kernel.wire import WIRE_MAJOR, WIRE_MINOR, WIRE_PATCH
    assert isinstance(WIRE_MAJOR, int)
    assert isinstance(WIRE_MINOR, int)
    assert isinstance(WIRE_PATCH, int)


def test_wire_version_parts_consistent_with_version_string() -> None:
    """WIRE_MAJOR/MINOR/PATCH are consistent with WIRE_VERSION string."""
    from llm_kernel.wire import WIRE_VERSION, WIRE_MAJOR, WIRE_MINOR, WIRE_PATCH
    parts = [int(x) for x in WIRE_VERSION.split(".")]
    assert parts == [WIRE_MAJOR, WIRE_MINOR, WIRE_PATCH], (
        f"WIRE_VERSION {WIRE_VERSION!r} inconsistent with "
        f"({WIRE_MAJOR}, {WIRE_MINOR}, {WIRE_PATCH})"
    )


# ---------------------------------------------------------------------------
# TOOL_CATALOG alias identity
# ---------------------------------------------------------------------------

def test_tool_catalog_alias_identity() -> None:
    """wire.tools.TOOL_CATALOG is identical to _rfc_schemas.TOOL_CATALOG."""
    from llm_kernel.wire import tools as wire_tools
    import llm_kernel._rfc_schemas as rfc
    assert wire_tools.TOOL_CATALOG is rfc.TOOL_CATALOG


def test_rfc_schemas_validate_tool_input_alias() -> None:
    """_rfc_schemas.validate_tool_input delegates to wire.tools."""
    from llm_kernel.wire.tools import validate_tool_input as wire_fn
    from llm_kernel._rfc_schemas import validate_tool_input as rfc_fn
    assert wire_fn is rfc_fn


# ---------------------------------------------------------------------------
# Envelope / families
# ---------------------------------------------------------------------------

def test_envelope_is_union() -> None:
    """wire.families.Envelope is a proper Union (typing alias)."""
    from llm_kernel.wire import families
    env = families.Envelope
    # Union types expose __args__ in Python 3.8+
    assert hasattr(env, "__args__"), "Envelope must be a Union type with __args__"
    assert len(env.__args__) >= 5, "Envelope must cover at least 5 families"


def test_all_family_classes_importable() -> None:
    """All five family TypedDicts are importable from wire."""
    from llm_kernel.wire import (  # noqa: F401
        FamilyA_OperatorAction,
        FamilyB_LayoutEdit,
        FamilyC_AgentGraphCommand,
        FamilyF_NotebookSnapshot,
        FamilyG_Lifecycle,
    )


# ---------------------------------------------------------------------------
# JSON schema files
# ---------------------------------------------------------------------------

def _schemas_dir() -> Path:
    import llm_kernel.wire
    return Path(llm_kernel.wire.__file__).parent / "schemas"


def test_schemas_directory_exists() -> None:
    """wire/schemas/ directory exists."""
    assert _schemas_dir().is_dir(), f"Expected schemas dir at {_schemas_dir()}"


def test_all_schema_files_parseable() -> None:
    """Every .json file in wire/schemas/ is valid JSON."""
    schemas_dir = _schemas_dir()
    json_files = list(schemas_dir.glob("*.json"))
    assert json_files, "No JSON files found in wire/schemas/"
    for f in json_files:
        data = json.loads(f.read_text(encoding="utf-8"))
        assert isinstance(data, dict), f"{f.name}: expected JSON object"


def test_schema_file_count() -> None:
    """wire/schemas/ has the expected number of files.

    13 tools * 2 (input + output) + 5 families + 2 handshake (request +
    response, S5.0.3d) = 33 JSON files.
    """
    schemas_dir = _schemas_dir()
    json_files = list(schemas_dir.glob("*.json"))
    expected_tool_count = 13  # per RFC-001 / TOOL_CATALOG
    expected_family_count = 5  # A, B, C, F, G
    expected_handshake_count = 2  # handshake.request, handshake.response (S5.0.3d)
    expected_total = (
        expected_tool_count * 2 + expected_family_count + expected_handshake_count
    )
    assert len(json_files) == expected_total, (
        f"Expected {expected_total} schema files, found {len(json_files)}: "
        f"{sorted(f.name for f in json_files)}"
    )


def test_tool_schema_names_match_catalog() -> None:
    """Each tool in TOOL_CATALOG has corresponding input + output JSON files."""
    from llm_kernel.wire import TOOL_CATALOG
    schemas_dir = _schemas_dir()
    for tool_name in TOOL_CATALOG:
        for kind in ("input", "output"):
            fname = schemas_dir / f"tool.{tool_name}.{kind}.json"
            assert fname.exists(), f"Missing schema file: {fname.name}"


def test_handshake_schema_files_exist() -> None:
    """Handshake request + response schema files exist (S5.0.3d)."""
    schemas_dir = _schemas_dir()
    for kind in ("request", "response"):
        fname = schemas_dir / f"handshake.{kind}.json"
        assert fname.exists(), f"Missing schema file: {fname.name}"
        # Ensure it's a real JSON Schema with the kernel.handshake type const.
        data = json.loads(fname.read_text(encoding="utf-8"))
        assert data["properties"]["type"]["const"] == "kernel.handshake"


def test_family_schema_files_exist() -> None:
    """Family envelope schema files exist for A, B, C, F, G."""
    schemas_dir = _schemas_dir()
    for letter in ("a", "b", "c", "f", "g"):
        fname = schemas_dir / f"family-{letter}.json"
        assert fname.exists(), f"Missing family schema file: {fname.name}"
