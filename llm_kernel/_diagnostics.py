"""Kernel-side diagnostic markers for E2E test introspection.

Test-only escape hatch: when ``LLMNB_E2E_MARKER_FILE`` is set, the kernel
appends one-line JSON records to that file at strategic lifecycle
points. Tests read the marker file to determine where a hung run
actually got stuck.

In production (no env var set), :func:`mark` is a tight no-op — the
file open and JSON encode are skipped. Zero overhead when not testing.

Cf. ``Testing.md`` §6 (Known limits) — closing the Tier 4 diagnostic
gap that made the live-cell e2e test impossible to debug.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

_LOCK: threading.Lock = threading.Lock()
_ENV_VAR: str = "LLMNB_E2E_MARKER_FILE"
#: Default fallback marker location when the env var isn't set. Used by
#: F5 launches (where the operator never sets the env var) so the
#: marker trail is still discoverable by inspecting the work_dir.
_FALLBACK_PATH_FACTORY = lambda: os.path.join(
    os.environ.get("LLMKERNEL_WORK_DIR_HINT", os.getcwd()),
    ".llmnb-kernel-markers.jsonl",
)


def mark(stage: str, **kw: Any) -> None:
    """Append one JSON line documenting where in the kernel lifecycle we are.

    Schema: ``{"ts": <unix>, "stage": "<name>", **kw}``. ``stage`` is a
    short snake_case name (``boot``, ``connect_socket``, ``ready_emitted``,
    ``agent_spawn_received``, ``supervisor_spawn_returned``, etc.).

    Writes to ``$LLMNB_E2E_MARKER_FILE`` if set (test mode), else falls
    back to ``<cwd>/.llmnb-kernel-markers.jsonl`` so F5 launches still
    leave a trail. The lock serializes writes from multiple threads so
    JSON lines never interleave mid-record.
    """
    target = os.environ.get(_ENV_VAR) or _FALLBACK_PATH_FACTORY()
    record = {"ts": time.time(), "stage": stage, **kw}
    try:
        line = json.dumps(record, default=str) + "\n"
    except (TypeError, ValueError):
        line = json.dumps({"ts": time.time(), "stage": stage, "_encode_error": True}) + "\n"
    try:
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        with _LOCK:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(line)
                fh.flush()
    except OSError:
        # Diagnostic markers MUST NEVER crash the kernel. Swallow.
        pass
