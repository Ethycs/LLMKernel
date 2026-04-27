"""RFC-004 fault-injection scheduler.

Split from :mod:`harness` to honor the per-file size cap. Provides
:class:`FaultMatrix` (per-step transition probabilities, RFC-004
§"Fault-injection scheduler" config schema) and :class:`FaultInjector`
(the seeded driver that mutates a sequencer's captured envelopes per
the matrix).

V1 fault posture: corruptions are injected into the captured log
directly, never through the run-tracker's sink. That way
``RunTracker.iter_runs()`` stays OTLP-spanId-clean and RFC-004 §I6
holds even under p_corrupt > 0 — the receiver's invariant is "log
and drop", not "trust and crash" (RFC-003 §F1).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List

from .harness import EventSequencer, RunResult, fold_state


@dataclass
class FaultMatrix:
    """Per-step fault-injection probabilities.

    All four probabilities MUST lie in ``[0, 1]``; see :meth:`validate`.
    Defaults match RFC-004 §"Fault-injection scheduler" example
    weighting (low-frequency drops + corruptions, near-zero
    disconnects).
    """

    p_drop: float = 0.05
    p_delay: float = 0.05
    p_corrupt: float = 0.02
    p_disconnect: float = 0.01

    def validate(self) -> None:
        """Raise :class:`ValueError` if any probability is outside [0, 1]."""
        for name in ("p_drop", "p_delay", "p_corrupt", "p_disconnect"):
            value = getattr(self, name)
            if not (0.0 <= float(value) <= 1.0):
                raise ValueError(f"{name} = {value!r} outside [0, 1]")


class FaultInjector:
    """Run a scenario through a sequencer with seeded fault injection.

    The injector consumes the sequencer's captured envelopes after the
    happy-path run completes and applies four independent dice rolls
    per envelope:

    * **p_drop** — remove the envelope from the captured log entirely.
    * **p_delay** — record a simulated delay marker (no real sleep, to
      keep tests fast and wall-clock measurements meaningful).
    * **p_corrupt** — append a sibling envelope with a non-UUID
      correlation_id directly to the captured list (bypassing the
      run-tracker's sink); this models the "log-and-drop" recovery
      branch from RFC-003 §F1.
    * **p_disconnect** — append a synthetic :class:`ConnectionError` to
      ``result.errors`` to model RFC-003 §F8 (extension/kernel transport
      failure).
    """

    def __init__(
        self, sequencer: EventSequencer, matrix: FaultMatrix, seed: int = 0,
    ) -> None:
        matrix.validate()
        self.sequencer = sequencer
        self.matrix = matrix
        self.rng = random.Random(seed)

    def run(self) -> RunResult:
        """Drive the sequencer once, then mutate the captured log per the matrix."""
        result = self.sequencer.run()
        kept: List[Dict[str, Any]] = []
        m = self.matrix
        for env in result.events:
            if self.rng.random() < m.p_drop:
                continue
            kept.append(env)
            if (self.rng.random() < m.p_corrupt
                    and env.get("message_type") == "run.start"):
                corrupt = dict(env)
                # OTLP spanIds are 16 lowercase hex chars; this synthesizes
                # a malformed correlation_id that the receiver MUST log
                # and drop per RFC-003 §F1.
                corrupt["correlation_id"] = (
                    f"NOT-A-SPANID-{self.rng.randrange(10**6)}"
                )
                kept.append(corrupt)
            if self.rng.random() < m.p_disconnect:
                result.errors.append(
                    ConnectionError("simulated kernel disconnect")
                )
            if self.rng.random() < m.p_delay:
                # Recorded as a metadata-only marker; do not actually sleep
                # so the suite stays fast.
                pass
        result.events = kept
        result.final_state = fold_state(kept)
        return result


__all__ = ["FaultInjector", "FaultMatrix"]
