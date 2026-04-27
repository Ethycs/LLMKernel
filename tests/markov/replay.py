"""RFC-004 replay harness — three modes per the RFC.

Split from :mod:`harness` to honor the per-file size cap. Provides
:class:`ReplayMode` (LIVE | DRY | PARTIAL) and :class:`ReplayHarness`
(the driver that consumes a captured envelope log and produces a
:class:`harness.RunResult`).

V1 mode semantics:

* **DRY** — pure state simulation. No kernel-side calls; the captured
  envelopes are folded into the final state via :func:`fold_state`.
* **LIVE** — V1 echoes the captured envelopes (no real model calls).
  ``TODO(T2-future)``: re-drive a fresh
  :class:`llm_kernel.mcp_server.OperatorBridgeServer` so live replay
  produces NEW outputs against the recorded inputs.
* **PARTIAL** — filter the captured envelopes by a selector predicate
  before folding. Useful for replaying one cell, one agent, or one
  correlation_id family.
"""

from __future__ import annotations

import enum
from typing import Any, Callable, Dict, List, Optional

from .harness import RunResult, fold_state


class ReplayMode(enum.Enum):
    """Replay modes per RFC-004 §"Replay harness modes"."""

    LIVE = "live"
    """Re-drive a fresh kernel-side stack from the recorded envelopes.

    V1 placeholder: echoes the captured envelopes; full LIVE replay is
    deferred (TODO(T2-future)).
    """

    DRY = "dry"
    """State simulation only — fold the log without invoking handlers."""

    PARTIAL = "partial"
    """Replay only envelopes matching ``selector``."""


class ReplayHarness:
    """Replay a captured envelope log under one of the three modes.

    Constructor validates that PARTIAL mode is paired with a selector
    predicate; missing it raises :class:`ValueError` so the contract is
    explicit at the call site, not deep inside :meth:`run`.
    """

    def __init__(
        self,
        events: List[Dict[str, Any]],
        mode: ReplayMode,
        selector: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> None:
        if mode == ReplayMode.PARTIAL and selector is None:
            raise ValueError(
                "PARTIAL replay requires a selector predicate; "
                "see RFC-004 §'Replay harness modes' for the contract."
            )
        self.events: List[Dict[str, Any]] = list(events)
        self.mode: ReplayMode = mode
        self.selector: Optional[Callable[[Dict[str, Any]], bool]] = selector

    def run(self) -> RunResult:
        """Apply the mode and return a :class:`RunResult`."""
        if self.mode == ReplayMode.PARTIAL:
            assert self.selector is not None  # validated in __init__
            filtered = [e for e in self.events if self.selector(e)]
            return RunResult(
                events=filtered, errors=[],
                final_state=fold_state(filtered),
                wall_clock_ms=0.0, zone_lifetimes=[], run_tracker=None,
            )
        # LIVE and DRY both fold the captured log in V1.
        return RunResult(
            events=list(self.events), errors=[],
            final_state=fold_state(self.events),
            wall_clock_ms=0.0, zone_lifetimes=[], run_tracker=None,
        )


__all__ = ["ReplayHarness", "ReplayMode"]
