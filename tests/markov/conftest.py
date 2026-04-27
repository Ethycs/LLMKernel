"""Pytest fixtures local to the RFC-004 Markov harness.

Kept separate from ``vendor/LLMKernel/conftest.py`` (which the prompt
says we MUST NOT edit) so this package stays self-contained.

Fixtures here are deliberately minimal: the harness builds its own
:class:`~.harness.EventSequencer` per test. The only shared concern is
the Hypothesis profile selection and a deterministic RNG seed when a
test wants one.
"""

from __future__ import annotations

import os
import random

import pytest


@pytest.fixture(autouse=True)
def _deterministic_random() -> None:
    """Seed the global :mod:`random` to keep non-Hypothesis tests reproducible."""
    random.seed(int(os.environ.get("LLMKERNEL_MARKOV_SEED", "4242")))
