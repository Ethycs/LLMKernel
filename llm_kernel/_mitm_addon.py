"""mitmproxy addon: logs every flow as JSONL for the kernel run-tracker.

Loaded by ``mitmdump -s _mitm_addon.py`` from
:class:`MitmProxyServer`. Reads the output path from
``MITM_LOG_FILE`` env var. Each line is a single JSON record; the
record types are ``"request"`` (emitted at request time) and
``"response"`` (emitted after the upstream replies, with status code
and body length).

The addon is intentionally minimal — heavy lifting (correlation_ids,
RFC-003 envelopes, streaming chunks) lives in the parent supervisor
process, which reads this file after the agent run completes.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


_LOG_PATH: Optional[Path] = None
_LOG_LOCK: threading.Lock = threading.Lock()


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _redact_bearer(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    if not value.lower().startswith("bearer "):
        return value[:8] + "..."
    token = value[7:]
    if len(token) <= 12:
        return value[:7] + "***"
    return f"Bearer {token[:8]}...{token[-4:]}"


def _write(record: Dict[str, Any]) -> None:
    global _LOG_PATH
    if _LOG_PATH is None:
        env_path = os.environ.get("MITM_LOG_FILE")
        if not env_path:
            return
        _LOG_PATH = Path(env_path)
    try:
        line = json.dumps(record, default=str) + "\n"
        with _LOG_LOCK:
            with _LOG_PATH.open("a", encoding="utf-8") as fh:
                fh.write(line)
    except OSError:
        # Don't let logging errors disrupt the proxy flow.
        pass


def request(flow: Any) -> None:  # noqa: ANN001 - mitmproxy types are dynamic
    """Hook fired when mitmproxy receives a request from the agent."""
    flow.metadata["llmnb_correlation_id"] = str(uuid.uuid4())
    flow.metadata["llmnb_t0"] = time.monotonic()
    req = flow.request
    _write({
        "kind": "request",
        "ts": _utc_iso(),
        "correlation_id": flow.metadata["llmnb_correlation_id"],
        "method": req.method,
        "url": req.pretty_url,
        "path": req.path,
        "host": req.host,
        "auth": _redact_bearer(req.headers.get("authorization")),
        "content_type": req.headers.get("content-type"),
        "body_bytes": len(req.raw_content or b""),
        "stream": "text/event-stream"
                   in (req.headers.get("accept") or "").lower(),
    })


def response(flow: Any) -> None:  # noqa: ANN001
    """Hook fired after the upstream response has been received in full."""
    cid = flow.metadata.get("llmnb_correlation_id")
    t0 = flow.metadata.get("llmnb_t0")
    duration_ms: Optional[float] = None
    if t0 is not None:
        duration_ms = (time.monotonic() - t0) * 1000.0
    resp = flow.response
    _write({
        "kind": "response",
        "ts": _utc_iso(),
        "correlation_id": cid,
        "status_code": resp.status_code if resp else None,
        "content_type": resp.headers.get("content-type") if resp else None,
        "body_bytes": len(resp.raw_content or b"") if resp else 0,
        "duration_ms": duration_ms,
    })


def error(flow: Any) -> None:  # noqa: ANN001
    """Hook fired on transport-level errors (TLS, DNS, connection refused)."""
    cid = flow.metadata.get("llmnb_correlation_id")
    err = flow.error
    _write({
        "kind": "error",
        "ts": _utc_iso(),
        "correlation_id": cid,
        "message": str(err) if err else "unknown",
    })
