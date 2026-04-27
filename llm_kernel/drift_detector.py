"""LLMKernel drift detector (RFC-005 §`metadata.rts.drift_log`).

Runs on file load.  Compares the persisted ``config.volatile.*``
substructure against the current environment, scans
``event_log.runs[*]`` for in-progress spans (and truncates them with a
drift event), and re-evaluates ``agents.nodes[*].properties.status``
against the current agent process state.  Detected drift is appended
to ``metadata.rts.drift_log`` and surfaced to the operator via the
extension's drift surface; this module produces the events but does
not emit them onto the wire (the metadata writer does that on the
next snapshot).

What this module provides:

* :class:`DriftDetector` -- the single class wired into the kernel's
  file-load path.  Accepts the persisted snapshot, is told the
  current-environment values, and returns a list of detected drift
  events.
* :func:`truncate_in_progress_spans` -- the per-RFC-005 truncation
  pass for in-progress spans (``endTimeUnixNano: null`` AND
  ``status.code: STATUS_CODE_UNSET``).  Mutates the spans in place
  AND returns one drift event per truncation.
* Severity classification helpers per RFC-005 §"`metadata.rts.
  drift_log`" :class:`Severity`: ``info`` for benign drift,
  ``warn`` for behavior-affecting drift, ``error`` for
  resume-blocking drift (RFC major mismatch, MCP server gone, model
  unavailable).
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger: logging.Logger = logging.getLogger("llm_kernel.drift_detector")

#: Severities per RFC-005 §"`metadata.rts.drift_log`".
SEVERITY_INFO: str = "info"
SEVERITY_WARN: str = "warn"
SEVERITY_ERROR: str = "error"

#: OTel UNSET status code; used to recognize in-progress spans.
_STATUS_UNSET: str = "STATUS_CODE_UNSET"
#: OTel ERROR status code; used as the truncation status.
_STATUS_ERROR: str = "STATUS_CODE_ERROR"


def _utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO 8601 with millisecond precision."""
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _major_minor(version: str) -> Tuple[int, int]:
    """Return ``(major, minor)`` parsed from a semver string.

    Returns ``(0, 0)`` for malformed input so the comparison degrades
    gracefully (callers see a "warn" via minor mismatch rather than
    a crash).  The third component (patch) is intentionally ignored:
    RFC-005 §"Resume-time RFC version check" only distinguishes major
    (error) vs. minor (warn).
    """
    try:
        parts = version.split(".")
        return int(parts[0]), int(parts[1])
    except (AttributeError, IndexError, ValueError):
        return (0, 0)


def _classify_version_drift(prev: str, current: str) -> str:
    """Classify a semver drift as ``info|warn|error`` per RFC-005."""
    if prev == current:
        return SEVERITY_INFO
    pmaj, pmin = _major_minor(prev)
    cmaj, cmin = _major_minor(current)
    if pmaj != cmaj:
        return SEVERITY_ERROR
    if pmin != cmin:
        return SEVERITY_WARN
    return SEVERITY_WARN


def truncate_in_progress_spans(
    runs: List[Dict[str, Any]],
    *,
    detected_at: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Truncate in-progress spans in place per RFC-005 §"In-progress spans".

    For each span with ``endTimeUnixNano is None`` AND
    ``status.code == STATUS_CODE_UNSET``: stamp ``endTimeUnixNano``
    to wall-clock now (decimal nanoseconds, JSON-string per OTLP),
    set ``status.code = STATUS_CODE_ERROR`` with message ``"kernel
    restart truncated"``, and produce one drift event with
    ``severity: "info"`` per RFC-005's classification.

    Returns the list of generated drift events (one per truncated
    span).  Callers append these to ``metadata.rts.drift_log``.
    """
    timestamp_iso = detected_at or _utc_now_iso()
    end_ns = str(time.time_ns())
    events: List[Dict[str, Any]] = []
    for idx, span in enumerate(runs):
        if not isinstance(span, dict):
            continue
        if span.get("endTimeUnixNano") is not None:
            continue
        status = span.get("status") or {}
        if status.get("code") != _STATUS_UNSET:
            continue
        prev_status_str = status.get("code", _STATUS_UNSET)
        span["endTimeUnixNano"] = end_ns
        span["status"] = {
            "code": _STATUS_ERROR,
            "message": "kernel restart truncated",
        }
        events.append({
            "detected_at": timestamp_iso,
            "field_path": f"event_log.runs[{idx}].status",
            "previous_value": prev_status_str,
            "current_value": f"{_STATUS_ERROR} (kernel restart truncated)",
            "severity": SEVERITY_INFO,
            "operator_acknowledged": False,
        })
    return events


class DriftDetector:
    """Compute drift events between persisted state and current environment.

    Usage::

        detector = DriftDetector()
        drift = detector.compare(persisted_snapshot, current_env, current_agents)
        # drift is a list of dicts ready to append to metadata.rts.drift_log

    The detector is stateless across calls; every comparison takes the
    persisted file's snapshot as input and the current-environment
    values as input, and returns a fresh list of drift events.
    """

    def compare(
        self,
        persisted: Dict[str, Any],
        *,
        current_kernel: Optional[Dict[str, Any]] = None,
        current_agents: Optional[Iterable[Dict[str, Any]]] = None,
        current_mcp_servers: Optional[Iterable[Dict[str, Any]]] = None,
        current_agent_status: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """Return the drift events for one persisted-vs-current comparison.

        ``persisted`` is the ``metadata.rts`` dict loaded from disk.
        ``current_kernel`` / ``current_agents`` / ``current_mcp_servers``
        carry the current-environment values for the volatile fields
        RFC-005 enumerates; pass ``None`` to skip a category.
        ``current_agent_status`` is a mapping of ``agent_id`` to
        currently-observed status (``"idle"`` / ``"busy"`` /
        ``"crashed"`` / etc.).

        Order of detection: kernel-volatile fields, per-agent volatile
        fields, MCP-server transports, in-progress span truncation,
        agent-process status drift.  Matches RFC-005's enumeration
        order so the operator sees the most-impactful drift first.
        """
        events: List[Dict[str, Any]] = []
        events.extend(self._kernel_volatile(persisted, current_kernel))
        events.extend(self._agent_volatile(persisted, current_agents))
        events.extend(self._mcp_volatile(persisted, current_mcp_servers))
        events.extend(self._in_progress_spans(persisted))
        events.extend(self._agent_status(persisted, current_agent_status))
        return events

    # -- Kernel volatile fields --------------------------------------

    def _kernel_volatile(
        self, persisted: Dict[str, Any],
        current: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Compare ``config.volatile.kernel.*`` field by field."""
        if current is None:
            return []
        prev_kernel = (
            persisted.get("config", {})
            .get("volatile", {})
            .get("kernel", {})
        )
        events: List[Dict[str, Any]] = []
        # Plain string fields where any change is a warn.
        for key in ("model_default", "passthrough_mode"):
            prev = prev_kernel.get(key)
            curr = current.get(key)
            if prev is not None and curr is not None and prev != curr:
                events.append(_make_drift(
                    field_path=f"config.volatile.kernel.{key}",
                    previous=prev, current=curr, severity=SEVERITY_WARN,
                ))
        # Semver fields where major mismatch escalates to error.
        for key in ("rfc_001_version", "rfc_002_version", "rfc_003_version"):
            prev = prev_kernel.get(key)
            curr = current.get(key)
            if prev is None or curr is None:
                continue
            if prev == curr:
                continue
            severity = _classify_version_drift(str(prev), str(curr))
            events.append(_make_drift(
                field_path=f"config.volatile.kernel.{key}",
                previous=prev, current=curr, severity=severity,
            ))
        return events

    # -- Per-agent volatile fields -----------------------------------

    def _agent_volatile(
        self, persisted: Dict[str, Any],
        current: Optional[Iterable[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """Compare ``config.volatile.agents[*]`` field by field by agent_id."""
        if current is None:
            return []
        prev_agents: List[Dict[str, Any]] = (
            persisted.get("config", {})
            .get("volatile", {})
            .get("agents", [])
        ) or []
        current_by_id: Dict[str, Dict[str, Any]] = {}
        for entry in current:
            aid = entry.get("agent_id")
            if isinstance(aid, str):
                current_by_id[aid] = entry
        events: List[Dict[str, Any]] = []
        for idx, prev_entry in enumerate(prev_agents):
            aid = prev_entry.get("agent_id")
            if not isinstance(aid, str):
                continue
            curr_entry = current_by_id.get(aid)
            if curr_entry is None:
                # Agent disappeared from the current environment.
                events.append(_make_drift(
                    field_path=f"config.volatile.agents[{idx}].agent_id",
                    previous=aid, current=None, severity=SEVERITY_WARN,
                ))
                continue
            for key in (
                "model", "system_prompt_template_id", "system_prompt_hash",
            ):
                prev = prev_entry.get(key)
                curr = curr_entry.get(key)
                if prev is None or curr is None:
                    continue
                if prev != curr:
                    events.append(_make_drift(
                        field_path=(
                            f"config.volatile.agents[{idx}].{key}"
                        ),
                        previous=prev, current=curr,
                        severity=SEVERITY_WARN,
                    ))
        return events

    # -- MCP server volatile fields ----------------------------------

    def _mcp_volatile(
        self, persisted: Dict[str, Any],
        current: Optional[Iterable[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """Compare ``config.volatile.mcp_servers[*].transport``."""
        if current is None:
            return []
        prev_servers: List[Dict[str, Any]] = (
            persisted.get("config", {})
            .get("volatile", {})
            .get("mcp_servers", [])
        ) or []
        current_by_id: Dict[str, Dict[str, Any]] = {}
        for entry in current:
            sid = entry.get("server_id")
            if isinstance(sid, str):
                current_by_id[sid] = entry
        events: List[Dict[str, Any]] = []
        for idx, prev_entry in enumerate(prev_servers):
            sid = prev_entry.get("server_id")
            if not isinstance(sid, str):
                continue
            curr_entry = current_by_id.get(sid)
            if curr_entry is None:
                # MCP server disappeared -- resume-blocking per RFC-005.
                events.append(_make_drift(
                    field_path=(
                        f"config.volatile.mcp_servers[{idx}].server_id"
                    ),
                    previous=sid, current=None, severity=SEVERITY_ERROR,
                ))
                continue
            prev = prev_entry.get("transport")
            curr = curr_entry.get("transport")
            if prev is None or curr is None:
                continue
            if prev != curr:
                events.append(_make_drift(
                    field_path=(
                        f"config.volatile.mcp_servers[{idx}].transport"
                    ),
                    previous=prev, current=curr, severity=SEVERITY_WARN,
                ))
        return events

    # -- In-progress span truncation ---------------------------------

    def _in_progress_spans(
        self, persisted: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Truncate in-progress spans in the persisted event log.

        Mutates ``persisted["event_log"]["runs"]`` in place per
        RFC-005 §"In-progress spans" so callers can persist the
        normalized form.
        """
        runs: List[Dict[str, Any]] = (
            persisted.get("event_log", {}).get("runs", [])
        ) or []
        return truncate_in_progress_spans(runs)

    # -- Agent process status ----------------------------------------

    def _agent_status(
        self, persisted: Dict[str, Any],
        current_status: Optional[Dict[str, str]],
    ) -> List[Dict[str, Any]]:
        """Compare ``agents.nodes[*].properties.status`` against current."""
        if current_status is None:
            return []
        nodes: List[Dict[str, Any]] = (
            persisted.get("agents", {}).get("nodes", [])
        ) or []
        events: List[Dict[str, Any]] = []
        for node in nodes:
            if not isinstance(node, dict):
                continue
            node_id = node.get("id", "")
            if node.get("type") != "agent":
                continue
            # Node IDs follow ``agent:<agent_id>`` per RFC-005.
            agent_id = (
                node_id.split("agent:", 1)[1]
                if isinstance(node_id, str) and node_id.startswith("agent:")
                else node_id
            )
            prev = (node.get("properties") or {}).get("status")
            curr = current_status.get(agent_id)
            if prev is None or curr is None:
                continue
            if prev != curr:
                events.append(_make_drift(
                    field_path=(
                        f"agents.nodes[{node_id}].properties.status"
                    ),
                    previous=prev, current=curr, severity=SEVERITY_INFO,
                ))
                # The agent graph reflects current reality on load.
                node.setdefault("properties", {})["status"] = curr
        return events


def _make_drift(
    *, field_path: str, previous: Any, current: Any, severity: str,
) -> Dict[str, Any]:
    """Build a single drift event with the canonical key order."""
    return {
        "detected_at": _utc_now_iso(),
        "field_path": field_path,
        "previous_value": previous,
        "current_value": current,
        "severity": severity,
        "operator_acknowledged": False,
    }


__all__ = [
    "DriftDetector",
    "SEVERITY_ERROR", "SEVERITY_INFO", "SEVERITY_WARN",
    "truncate_in_progress_spans",
]
