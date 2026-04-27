"""Contract tests for :mod:`llm_kernel._otlp_log_handler` (RFC-008 §7).

Exercises:

* Python ``logging`` level -> OTel ``severityNumber`` mapping.
* OTLP/JSON wire shape: ``timeUnixNano``, ``observedTimeUnixNano`` as
  string ints, ``body.stringValue`` carrying the formatted message,
  ``attributes`` encoded with ``logger.name`` / ``code.function`` /
  ``code.lineno``.
* Optional ``extra_attributes`` (e.g., ``llmnb.kernel.session_id``)
  appearing on every emitted record.
* Defensive fallback: an unserializable LogRecord doesn't crash the
  kernel main loop.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import pytest

from llm_kernel._attrs import decode_attrs
from llm_kernel._otlp_log_handler import SEVERITY_MAP, OtlpDataPlaneHandler


class _CaptureWriter:
    """Stand-in :class:`SocketWriter` that buffers frames in-memory."""

    def __init__(self) -> None:
        self.frames: List[Dict[str, Any]] = []

    def write_frame(self, record: Dict[str, Any]) -> None:
        self.frames.append(record)


def _make_record(level: int, msg: str = "hello", logger_name: str = "llm_kernel.test") -> logging.LogRecord:
    """Construct a LogRecord with deterministic fields for assertions."""
    record = logging.LogRecord(
        name=logger_name,
        level=level,
        pathname="/src/llm_kernel/test.py",
        lineno=42,
        msg=msg,
        args=None,
        exc_info=None,
        func="test_func",
    )
    return record


# ---------------------------------------------------------------------------
# Severity mapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("level,expected", [
    (logging.DEBUG, 5),
    (logging.INFO, 9),
    (logging.WARNING, 13),
    (logging.ERROR, 17),
    (logging.CRITICAL, 21),
])
def test_severity_mapping(level: int, expected: int) -> None:
    """Each Python level maps to the RFC-008 §7 OTel severityNumber."""
    assert SEVERITY_MAP[level] == expected
    writer = _CaptureWriter()
    handler = OtlpDataPlaneHandler(writer)
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.emit(_make_record(level))
    assert len(writer.frames) == 1
    assert writer.frames[0]["severityNumber"] == expected


def test_unknown_level_defaults_to_info() -> None:
    """Levels outside the standard table fall through to 9 (INFO)."""
    writer = _CaptureWriter()
    handler = OtlpDataPlaneHandler(writer)
    handler.setFormatter(logging.Formatter("%(message)s"))
    record = _make_record(15)  # between INFO and WARNING; unmapped
    handler.emit(record)
    assert writer.frames[0]["severityNumber"] == 9


# ---------------------------------------------------------------------------
# Wire shape
# ---------------------------------------------------------------------------


def test_emit_produces_otlp_log_record_shape() -> None:
    """The frame carries every required RFC-008 §7 field."""
    writer = _CaptureWriter()
    handler = OtlpDataPlaneHandler(writer)
    handler.setFormatter(logging.Formatter("%(name)s - %(message)s"))
    handler.emit(_make_record(logging.INFO, msg="kernel.ready"))
    assert len(writer.frames) == 1
    frame = writer.frames[0]
    assert isinstance(frame["timeUnixNano"], str)
    assert int(frame["timeUnixNano"]) > 0
    assert isinstance(frame["observedTimeUnixNano"], str)
    assert int(frame["observedTimeUnixNano"]) > 0
    assert frame["severityNumber"] == 9
    assert frame["severityText"] == "INFO"
    assert frame["body"]["stringValue"] == "llm_kernel.test - kernel.ready"
    attrs = decode_attrs(frame["attributes"])
    assert attrs["logger.name"] == "llm_kernel.test"
    assert attrs["code.function"] == "test_func"
    assert attrs["code.lineno"] == 42


def test_extra_attributes_attached_to_every_record() -> None:
    """Constructor ``extra_attributes`` ride on every emitted record."""
    writer = _CaptureWriter()
    handler = OtlpDataPlaneHandler(
        writer, extra_attributes={"llmnb.kernel.session_id": "abc-123"},
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.emit(_make_record(logging.INFO, msg="m1"))
    handler.emit(_make_record(logging.WARNING, msg="m2"))
    for frame in writer.frames:
        attrs = decode_attrs(frame["attributes"])
        assert attrs["llmnb.kernel.session_id"] == "abc-123"


def test_attribute_encoding_uses_otlp_anyvalue_shape() -> None:
    """Each attribute is ``{key, value: <AnyValue>}`` per OTLP/JSON spec."""
    writer = _CaptureWriter()
    handler = OtlpDataPlaneHandler(writer)
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.emit(_make_record(logging.INFO))
    attrs = writer.frames[0]["attributes"]
    assert isinstance(attrs, list)
    for pair in attrs:
        assert set(pair.keys()) == {"key", "value"}
        assert isinstance(pair["value"], dict)
        # Exactly one of stringValue / intValue / boolValue / etc is set.
        assert any(
            k in pair["value"] for k in (
                "stringValue", "intValue", "boolValue", "doubleValue",
            )
        )


def test_emit_does_not_raise_on_writer_failure() -> None:
    """A writer that raises is swallowed by ``handleError`` semantics."""

    class _RaisingWriter:
        def write_frame(self, _record: Dict[str, Any]) -> None:
            raise RuntimeError("simulated socket failure")

    handler = OtlpDataPlaneHandler(_RaisingWriter())  # type: ignore[arg-type]
    handler.setFormatter(logging.Formatter("%(message)s"))
    # Stub out handleError so we don't write to stderr during pytest.
    handler.handleError = lambda _record: None  # type: ignore[method-assign]
    handler.emit(_make_record(logging.ERROR))  # MUST NOT raise


def test_logger_namespace_propagation() -> None:
    """Records logged via the standard logger reach our handler."""
    writer = _CaptureWriter()
    handler = OtlpDataPlaneHandler(writer)
    handler.setFormatter(logging.Formatter("%(message)s"))

    log = logging.getLogger("llm_kernel._otlp_log_handler.test")
    log.setLevel(logging.DEBUG)
    log.addHandler(handler)
    try:
        log.info("via standard logger")
        log.warning("warning via standard logger")
    finally:
        log.removeHandler(handler)
    assert len(writer.frames) == 2
    assert writer.frames[0]["body"]["stringValue"] == "via standard logger"
    assert writer.frames[1]["severityText"] == "WARNING"
