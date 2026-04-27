"""drift_acknowledged routes to ``MetadataWriter.acknowledge_drift`` (RFC-005).

K-MCP V1 mega-round: when an ``operator.action`` envelope arrives with
``action_type == "drift_acknowledged"``, the MCP server MUST locate
the kernel's :class:`MetadataWriter` (via the
``__llmnb_metadata_writer__`` user_ns slot, the
``_llmnb_metadata_writer`` kernel attribute, or a directly-set
``bridge.metadata_writer``) and call its ``acknowledge_drift``
method.  The interface contract is owned by K-MW; this test mocks
the method and asserts the call shape.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Tuple

import pytest

from llm_kernel.mcp_server import OperatorBridgeServer


class _MockMetadataWriter:
    """Minimal stand-in for K-MW's MetadataWriter.acknowledge_drift surface."""

    def __init__(self, return_value: bool = True) -> None:
        self.calls: List[Tuple[str, str]] = []
        self.return_value: bool = return_value

    def acknowledge_drift(self, field_path: str, detected_at: str) -> bool:
        self.calls.append((field_path, detected_at))
        return self.return_value


@pytest.fixture
def bridge() -> OperatorBridgeServer:
    return OperatorBridgeServer(
        agent_id="test-agent",
        zone_id="test-zone",
        trace_id=str(uuid.uuid4()),
    )


def _envelope(params: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "operator.action",
        "payload": {
            "action_type": "drift_acknowledged",
            "parameters": params,
        },
    }


def test_drift_acknowledged_calls_metadata_writer(
    bridge: OperatorBridgeServer,
) -> None:
    """drift_acknowledged MUST call acknowledge_drift with field_path/detected_at."""
    writer = _MockMetadataWriter(return_value=True)
    bridge.metadata_writer = writer
    bridge._route_operator_action(_envelope({
        "field_path": "config.volatile.kernel.rfc_001_version",
        "detected_at": "2026-04-26T12:34:56.789Z",
    }))
    assert writer.calls == [(
        "config.volatile.kernel.rfc_001_version",
        "2026-04-26T12:34:56.789Z",
    )]


def test_drift_acknowledged_via_user_ns(
    bridge: OperatorBridgeServer,
) -> None:
    """The user_ns slot MUST be probed when ``bridge.metadata_writer`` is None."""
    writer = _MockMetadataWriter()

    class _Shell:
        def __init__(self) -> None:
            self.user_ns: Dict[str, Any] = {"__llmnb_metadata_writer__": writer}

    class _Kernel:
        def __init__(self) -> None:
            self.shell = _Shell()

    bridge.kernel = _Kernel()
    bridge._route_operator_action(_envelope({
        "field_path": "config.volatile.agents",
        "detected_at": "2026-04-26T12:34:56.789Z",
    }))
    assert len(writer.calls) == 1
    assert writer.calls[0][0] == "config.volatile.agents"


def test_drift_acknowledged_via_kernel_attribute(
    bridge: OperatorBridgeServer,
) -> None:
    """The ``_llmnb_metadata_writer`` kernel attribute MUST be probed."""
    writer = _MockMetadataWriter()

    class _Kernel:
        def __init__(self) -> None:
            self._llmnb_metadata_writer = writer

    bridge.kernel = _Kernel()
    bridge._route_operator_action(_envelope({
        "field_path": "config.recoverable.kernel",
        "detected_at": "2026-04-26T01:02:03Z",
    }))
    assert len(writer.calls) == 1


def test_drift_acknowledged_no_writer_logs_warning(
    bridge: OperatorBridgeServer,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """No attached writer MUST surface a WARN log; no exception."""
    with caplog.at_level(logging.WARNING, logger="llm_kernel.mcp"):
        bridge._route_operator_action(_envelope({
            "field_path": "config.volatile.kernel",
            "detected_at": "2026-04-26T01:02:03Z",
        }))
    assert any(
        "MetadataWriter" in rec.getMessage() for rec in caplog.records
    )


def test_drift_acknowledged_logs_match_result(
    bridge: OperatorBridgeServer,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The post-call INFO log MUST carry the matched result."""
    writer = _MockMetadataWriter(return_value=False)
    bridge.metadata_writer = writer
    with caplog.at_level(logging.INFO, logger="llm_kernel.mcp"):
        bridge._route_operator_action(_envelope({
            "field_path": "config.volatile.kernel",
            "detected_at": "2026-04-26T01:02:03Z",
        }))
    assert any(
        "drift_acknowledged" in rec.getMessage() for rec in caplog.records
    )
    matching = [r for r in caplog.records if "drift_acknowledged" in r.getMessage()]
    assert any(getattr(r, "matched", None) is False for r in matching)
