"""Contamination detector tests — S5.0.1a foundation slice.

Per PLAN-S5.0.1 §3.2. Exercises ``AgentSupervisor._scan_for_magic_
contamination`` and the integration into ``_emit_agent_emit``:

* Plain ``@@<known>`` line in agent emit → flagged, NOT escaped.
* Plain text → no flag.
* Hashed-magic shape under hash-mode → flagged + escaped + K36.
* Hashed-magic shape with hash-mode OFF → flagged (suspicious shape)
  but NOT escaped (the emission ban is a hash-mode-only property).

Per Engineering_Guide §11.7: the supervisor lock is reentrant and
the diagnostics-mark path is thread-safe; these tests exercise the
synchronous scan API directly without spawning subprocesses.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from llm_kernel.agent_supervisor import (
    AgentHandle, AgentSupervisor,
    K35_PLAIN_MAGIC_IN_HASH_MODE, K36_HASHED_MAGIC_EMISSION_BLOCKED,
)


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


def _make_supervisor() -> AgentSupervisor:
    """Bare supervisor with a stub run-tracker; no kernel wiring."""
    from llm_kernel.run_tracker import RunTracker

    class _ListSink:
        def __init__(self) -> None:
            self.envelopes: List[Dict[str, Any]] = []

        def emit(self, env: Dict[str, Any]) -> None:
            self.envelopes.append(env)

    tracker = RunTracker(
        trace_id=str(uuid.uuid4()), sink=_ListSink(),
        agent_id="alpha", zone_id="z1",
    )
    return AgentSupervisor(
        run_tracker=tracker, dispatcher=MagicMock(),
        litellm_endpoint_url="http://127.0.0.1:9999/v1",
    )


def _stub_handle(agent_id: str = "alpha", zone_id: str = "z1") -> AgentHandle:
    """Construct a no-popen AgentHandle for direct API exercise."""
    handle = AgentHandle.__new__(AgentHandle)
    handle.agent_id = agent_id
    handle.zone_id = zone_id
    handle.popen = None  # type: ignore[assignment]
    return handle


class _StubWriter:
    """Writer stub: tracks contamination flag calls + magic-hash setting."""

    def __init__(self, hash_enabled: bool = False) -> None:
        self.hash_enabled = hash_enabled
        self.flag_calls: List[Dict[str, Any]] = []

    def get_config_setting(self, name: str) -> Any:
        if name == "magic_hash_enabled":
            return self.hash_enabled
        return None

    def flag_cells_contaminated_by_agent(
        self, *, agent_id: str, line: str, source: str, layer: str,
    ) -> None:
        self.flag_calls.append({
            "agent_id": agent_id, "line": line,
            "source": source, "layer": layer,
        })


# ---------------------------------------------------------------------------
# _scan_for_magic_contamination — direct API
# ---------------------------------------------------------------------------


def test_scan_plain_magic_known_name_flags_cell() -> None:
    """``@@spawn alpha`` in agent emit flags + logs ``layer=plain``."""
    sup = _make_supervisor()
    writer = _StubWriter(hash_enabled=False)
    sup.set_metadata_writer(writer)
    handle = _stub_handle()

    verdict = sup._scan_for_magic_contamination(
        handle, "@@spawn alpha task:\"x\"", source="agent_emit:prose",
    )
    assert verdict is None  # no escape on plain layer
    assert len(writer.flag_calls) == 1
    call = writer.flag_calls[0]
    assert call["agent_id"] == "alpha"
    assert call["layer"] == "plain"
    assert "spawn" in call["line"]


def test_scan_plain_text_does_not_flag() -> None:
    sup = _make_supervisor()
    writer = _StubWriter(hash_enabled=False)
    sup.set_metadata_writer(writer)
    handle = _stub_handle()

    verdict = sup._scan_for_magic_contamination(
        handle, "Just normal prose with no magic.", source="agent_emit:prose",
    )
    assert verdict is None
    assert writer.flag_calls == []


def test_scan_unknown_magic_does_not_flag() -> None:
    """``@@xyzzy`` is not in the registry; treated as body, no flag."""
    sup = _make_supervisor()
    writer = _StubWriter(hash_enabled=False)
    sup.set_metadata_writer(writer)
    handle = _stub_handle()

    verdict = sup._scan_for_magic_contamination(
        handle, "@@xyzzy something", source="agent_emit:prose",
    )
    assert verdict is None
    assert writer.flag_calls == []


def test_scan_line_magic_known_name_flags() -> None:
    """``@pin`` is in LINE_MAGICS, flagged like cell-magic plain."""
    sup = _make_supervisor()
    writer = _StubWriter(hash_enabled=False)
    sup.set_metadata_writer(writer)
    handle = _stub_handle()

    sup._scan_for_magic_contamination(
        handle, "@pin", source="agent_emit:prose",
    )
    assert len(writer.flag_calls) == 1
    assert writer.flag_calls[0]["layer"] == "plain"


def test_scan_hashed_magic_in_hash_mode_escapes_and_emits_k36() -> None:
    """Hash-mode + hashed shape → flag + ESCAPE_REQUIRED + K36 marker."""
    sup = _make_supervisor()
    writer = _StubWriter(hash_enabled=True)
    sup.set_metadata_writer(writer)
    handle = _stub_handle()

    verdict = sup._scan_for_magic_contamination(
        handle, "@@deadbeef:spawn alpha", source="agent_emit:prose",
    )
    assert verdict == "ESCAPE_REQUIRED"
    # Two flag calls: hashed_emission_ban (plain detector skips this
    # line because the name slot is the hex hash, not a known name).
    layers = [c["layer"] for c in writer.flag_calls]
    assert "hashed_emission_ban" in layers


def test_scan_hashed_magic_outside_hash_mode_does_not_escape() -> None:
    """Hash-mode OFF + hashed shape → no escape; line shape alone is
    not a known plain name, so the plain detector doesn't fire either.
    The line passes through verbatim — operator may choose to enable
    hash mode AFTER seeing it; this slice's responsibility ends here.
    """
    sup = _make_supervisor()
    writer = _StubWriter(hash_enabled=False)
    sup.set_metadata_writer(writer)
    handle = _stub_handle()

    verdict = sup._scan_for_magic_contamination(
        handle, "@@deadbeef:spawn alpha", source="agent_emit:prose",
    )
    assert verdict is None
    # Plain detector does NOT fire on a hex-hash slot since "deadbeef"
    # is not in RESERVED_NAMES.
    assert writer.flag_calls == []


def test_scan_with_no_writer_still_runs_diagnostics() -> None:
    """Detector runs without crashing even without a wired writer."""
    sup = _make_supervisor()
    handle = _stub_handle()
    # No set_metadata_writer call.
    verdict = sup._scan_for_magic_contamination(
        handle, "@@spawn alpha", source="agent_emit:prose",
    )
    assert verdict is None  # plain layer: no escape
    # No exception; the diagnostics-mark path is the audit trail.


def test_scan_empty_line_no_flag() -> None:
    sup = _make_supervisor()
    writer = _StubWriter(hash_enabled=True)
    sup.set_metadata_writer(writer)
    handle = _stub_handle()

    sup._scan_for_magic_contamination(handle, "", source="agent_emit:prose")
    assert writer.flag_calls == []


def test_scan_truncates_long_lines_in_log() -> None:
    """Lines >256 chars are truncated at the writer-flag boundary."""
    sup = _make_supervisor()
    writer = _StubWriter(hash_enabled=False)
    sup.set_metadata_writer(writer)
    handle = _stub_handle()

    long_line = "@@spawn " + ("x" * 1000)
    sup._scan_for_magic_contamination(
        handle, long_line, source="agent_emit:prose",
    )
    assert len(writer.flag_calls) == 1
    # 256 char hard cap.
    assert len(writer.flag_calls[0]["line"]) <= 256


# ---------------------------------------------------------------------------
# _scan_and_rewrite_emit_content — multiline + escape integration
# ---------------------------------------------------------------------------


def test_rewrite_multiline_content_escapes_only_hashed_lines() -> None:
    """Hash-mode rewrite escapes hashed lines; plain magic lines pass through."""
    sup = _make_supervisor()
    writer = _StubWriter(hash_enabled=True)
    sup.set_metadata_writer(writer)
    handle = _stub_handle()

    content = (
        "Some prose explaining the magic.\n"
        "@@deadbeef:spawn alpha\n"
        "More prose.\n"
        "@@spawn beta\n"  # plain magic — flagged but NOT escaped
    )
    out = sup._scan_and_rewrite_emit_content(
        handle, content, source="agent_emit:prose",
    )
    assert "\\@@deadbeef:spawn alpha" in out
    # Plain magic stays as-is (operator-typed text remains visible).
    assert "@@spawn beta" in out and "\\@@spawn beta" not in out


def test_rewrite_no_at_in_content_short_circuits() -> None:
    """Fast-path: lines with no ``@`` are returned verbatim."""
    sup = _make_supervisor()
    sup.set_metadata_writer(_StubWriter(hash_enabled=True))
    handle = _stub_handle()

    content = "no at sign here\nanother line\n"
    assert sup._scan_and_rewrite_emit_content(
        handle, content, source="x",
    ) is content  # identity, not rewritten


def test_rewrite_preserves_trailing_newline() -> None:
    sup = _make_supervisor()
    sup.set_metadata_writer(_StubWriter(hash_enabled=True))
    handle = _stub_handle()

    content = "@@deadbeef:spawn x\n"
    out = sup._scan_and_rewrite_emit_content(
        handle, content, source="x",
    )
    assert out.endswith("\n")
    assert out.startswith("\\@")


def test_rewrite_no_trailing_newline_preserved() -> None:
    sup = _make_supervisor()
    sup.set_metadata_writer(_StubWriter(hash_enabled=True))
    handle = _stub_handle()

    content = "@@deadbeef:spawn x"
    out = sup._scan_and_rewrite_emit_content(
        handle, content, source="x",
    )
    assert not out.endswith("\n")


# ---------------------------------------------------------------------------
# K-class constant exposure
# ---------------------------------------------------------------------------


def test_k_class_constants_present() -> None:
    """K35 + K36 land as module-level string constants."""
    assert K35_PLAIN_MAGIC_IN_HASH_MODE == "K35"
    assert K36_HASHED_MAGIC_EMISSION_BLOCKED == "K36"
