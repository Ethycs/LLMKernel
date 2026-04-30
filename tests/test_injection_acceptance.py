"""Tests for ``MetadataWriter.accept_injection_risk`` (PLAN-S5.0.1c §3.11).

Covers the verbatim-format invariant, idempotency on re-call, the
validator's rejection of arbitrary text, round-trip preservation, and
the K3G one-shot emit semantics.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from llm_kernel._rfc_schemas import K_CLASS_REGISTRY
from llm_kernel.metadata_writer import MetadataWriter


_PREFIX: str = "The Operator Has Accepted Arbitrary Code Injection at "


@pytest.fixture
def writer(tmp_path) -> MetadataWriter:
    return MetadataWriter(workspace_root=tmp_path)


# --- format invariants ----------------------------------------------


def test_accept_injection_risk_writes_verbatim_prefix(writer) -> None:
    s = writer.accept_injection_risk()
    assert s.startswith(_PREFIX)


def test_accept_injection_risk_includes_iso8601_timestamp(writer) -> None:
    s = writer.accept_injection_risk()
    ts_part = s[len(_PREFIX):]
    # Trim trailing 'Z' for portability with fromisoformat on 3.10.
    normalized = ts_part[:-1] if ts_part.endswith("Z") else ts_part
    parsed = datetime.fromisoformat(normalized)
    assert parsed.year >= 2026


def test_accept_injection_risk_persisted_in_config(writer) -> None:
    s = writer.accept_injection_risk()
    assert writer._config.get("injection_acceptance") == s


def test_get_injection_acceptance_returns_none_initially(writer) -> None:
    assert writer.get_injection_acceptance() is None


def test_get_injection_acceptance_after_accept(writer) -> None:
    s = writer.accept_injection_risk()
    assert writer.get_injection_acceptance() == s


# --- idempotency ----------------------------------------------------


def test_accept_injection_risk_idempotent_returns_existing(writer) -> None:
    first = writer.accept_injection_risk()
    second = writer.accept_injection_risk()
    assert first == second  # exact match: original timestamp preserved


def test_accept_injection_risk_idempotent_no_overwrite(writer) -> None:
    first = writer.accept_injection_risk()
    # Even after time advances the second call must not re-stamp.
    import time
    time.sleep(0.005)
    second = writer.accept_injection_risk()
    assert second == first
    assert writer._config["injection_acceptance"] == first


# --- validator rejects non-verbatim values --------------------------


def test_validator_rejects_arbitrary_text() -> None:
    bad = "Operator says: anything goes"
    assert MetadataWriter._validate_injection_acceptance(bad) is False


def test_validator_rejects_empty_string() -> None:
    assert MetadataWriter._validate_injection_acceptance("") is False


def test_validator_rejects_none() -> None:
    assert MetadataWriter._validate_injection_acceptance(None) is False


def test_validator_rejects_prefix_only() -> None:
    """Prefix without timestamp = invalid."""
    assert MetadataWriter._validate_injection_acceptance(_PREFIX) is False


def test_validator_rejects_prefix_plus_garbage() -> None:
    """Prefix + non-ISO8601 garbage = invalid."""
    assert MetadataWriter._validate_injection_acceptance(
        _PREFIX + "yesterday at noon"
    ) is False


def test_validator_accepts_constructed_form() -> None:
    """The exact form the writer constructs must round-trip."""
    s = _PREFIX + "2026-04-29T12:34:56.789Z"
    assert MetadataWriter._validate_injection_acceptance(s) is True


# --- get_injection_acceptance rejects smuggled values --------------


def test_get_injection_acceptance_rejects_smuggled_value(writer) -> None:
    """Direct config writes that bypass accept_injection_risk must
    not be returned by the accessor."""
    with writer._lock:
        writer._config["injection_acceptance"] = "I declare it accepted"
    assert writer.get_injection_acceptance() is None


# --- K3G registry entry ---------------------------------------------


def test_K3G_in_kclass_registry() -> None:
    assert "K3G" in K_CLASS_REGISTRY
    entry = K_CLASS_REGISTRY["K3G"]
    assert entry["name"] == "operator_accepted_injection_persisted"


def test_K3C_K3D_K3E_K3F_in_kclass_registry() -> None:
    for code in ("K3C", "K3D", "K3E", "K3F"):
        assert code in K_CLASS_REGISTRY
        assert "name" in K_CLASS_REGISTRY[code]
        assert "description" in K_CLASS_REGISTRY[code]


# --- round-trip via writer dirty flag --------------------------------


def test_accept_injection_risk_marks_writer_dirty(writer) -> None:
    # Reset dirty flag.
    writer._dirty = False
    writer.accept_injection_risk()
    assert writer._dirty is True


def test_accept_injection_risk_idempotent_does_not_re_dirty(writer) -> None:
    writer.accept_injection_risk()
    writer._dirty = False
    writer.accept_injection_risk()
    # Idempotent re-call returns existing string and does NOT set
    # dirty (no mutation occurred).
    assert writer._dirty is False
