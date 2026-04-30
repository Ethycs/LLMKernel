"""Tests for the S5.0.1b schema additions on ``MetadataWriter``.

Covers the duck-typed methods S5.0.1a's contamination detector
forward-referenced — ``get_config_setting``, ``set_hash_mode``,
``flag_cells_contaminated_by_agent`` — plus the per-cell
``contaminated`` / ``contamination_log`` round-trip through the
snapshot.
"""

from __future__ import annotations

import pytest

from llm_kernel.metadata_writer import MetadataWriter


@pytest.fixture
def writer(tmp_path) -> MetadataWriter:
    return MetadataWriter(workspace_root=tmp_path)


# --- get_config_setting --------------------------------------------


def test_get_config_setting_defaults(writer) -> None:
    """magic_hash_enabled defaults to False; fingerprint to None."""
    assert writer.get_config_setting("magic_hash_enabled") is False
    assert writer.get_config_setting("magic_pin_fingerprint") is None


def test_get_config_setting_unknown_key(writer) -> None:
    assert writer.get_config_setting("nonexistent_key") is None


def test_get_config_setting_returns_set_value(writer) -> None:
    writer.set_hash_mode(enabled=True, fingerprint="abc1234567890def")
    assert writer.get_config_setting("magic_hash_enabled") is True
    assert writer.get_config_setting("magic_pin_fingerprint") == \
        "abc1234567890def"


# --- set_hash_mode atomic ------------------------------------------


def test_set_hash_mode_atomic_pair(writer) -> None:
    writer.set_hash_mode(enabled=True, fingerprint="fp_a_a_a_a_a_a_a")
    assert writer.get_config_setting("magic_hash_enabled") is True
    assert writer.get_config_setting("magic_pin_fingerprint") == \
        "fp_a_a_a_a_a_a_a"


def test_set_hash_mode_disable_clears_fingerprint(writer) -> None:
    writer.set_hash_mode(enabled=True, fingerprint="some_fp_xxxxxxxx")
    writer.set_hash_mode(enabled=False, fingerprint=None)
    assert writer.get_config_setting("magic_hash_enabled") is False
    assert writer.get_config_setting("magic_pin_fingerprint") is None


def test_set_hash_mode_rejects_bad_types(writer) -> None:
    with pytest.raises(TypeError):
        writer.set_hash_mode(enabled="yes", fingerprint=None)  # type: ignore
    with pytest.raises(TypeError):
        writer.set_hash_mode(enabled=True, fingerprint=123)  # type: ignore


# --- flag_cells_contaminated_by_agent ------------------------------


def test_flag_cells_contaminated_finds_bound_cells(writer) -> None:
    """Only cells whose bound_agent_id matches get flagged."""
    writer._cells["c1"] = {"kind": "agent", "bound_agent_id": "alpha"}
    writer._cells["c2"] = {"kind": "agent", "bound_agent_id": "beta"}
    writer._cells["c3"] = {"kind": "agent", "bound_agent_id": "alpha"}
    flagged = writer.flag_cells_contaminated_by_agent(
        agent_id="alpha", line="@@spawn evil",
        source="agent_emit:stdout", layer="plain",
    )
    assert set(flagged) == {"c1", "c3"}


def test_flag_cells_contaminated_appends_to_log(writer) -> None:
    """Each call appends an entry to contamination_log."""
    writer._cells["c1"] = {"kind": "agent", "bound_agent_id": "alpha"}
    writer.flag_cells_contaminated_by_agent(
        agent_id="alpha", line="@@spawn evil",
        source="agent_emit:stdout", layer="plain",
    )
    writer.flag_cells_contaminated_by_agent(
        agent_id="alpha", line="@@deadbeef:spawn",
        source="agent_emit:stdout", layer="hashed_emission_ban",
    )
    cell = writer._cells["c1"]
    assert cell["contaminated"] is True
    log = cell["contamination_log"]
    assert len(log) == 2
    assert log[0]["layer"] == "plain"
    assert log[1]["layer"] == "hashed_emission_ban"
    # Each entry has the required fields.
    for entry in log:
        assert "detected_at" in entry
        assert "line" in entry
        assert "reason" in entry
        assert "layer" in entry


def test_flag_cells_contaminated_truncates_long_line(writer) -> None:
    """Lines >256 chars are truncated to keep notebook size bounded."""
    writer._cells["c1"] = {"kind": "agent", "bound_agent_id": "alpha"}
    long_line = "x" * 1000
    writer.flag_cells_contaminated_by_agent(
        agent_id="alpha", line=long_line,
        source="agent_emit:stdout", layer="plain",
    )
    log = writer._cells["c1"]["contamination_log"]
    assert len(log[0]["line"]) <= 256


def test_flag_cells_contaminated_no_match_returns_empty(writer) -> None:
    """Agent-id with no bound cells returns []."""
    flagged = writer.flag_cells_contaminated_by_agent(
        agent_id="nobody", line="x", source="x", layer="plain",
    )
    assert flagged == []


def test_flag_cells_contaminated_round_trips_through_snapshot(
    writer,
) -> None:
    """The contaminated flag + log persist into the next snapshot."""
    writer._cells["c1"] = {"kind": "agent", "bound_agent_id": "alpha"}
    writer.flag_cells_contaminated_by_agent(
        agent_id="alpha", line="@@spawn evil",
        source="agent_emit:stdout", layer="plain",
    )
    snap = writer._build_snapshot()
    cells = snap.get("cells", {})
    assert cells["c1"]["contaminated"] is True
    assert isinstance(cells["c1"]["contamination_log"], list)
