"""OTLP/JSON attribute encode/decode helpers.

OTLP/JSON encodes attributes as a list of ``{key, value}`` objects where
``value`` is an AnyValue tagged-union (``stringValue`` / ``intValue`` /
``boolValue`` / ``doubleValue`` / ``arrayValue`` / ``kvlistValue`` /
``bytesValue``).  This module hides that wire shape so callers can pass
plain Python dicts and round-trip them losslessly.

Reference: opentelemetry-proto OTLP/JSON encoding spec (``AnyValue``).
``intValue`` is a JSON STRING to preserve 64-bit precision; on decode
we convert back to ``int``.
"""

from __future__ import annotations

from typing import Any, Dict, List


def _encode_value(value: Any) -> Dict[str, Any]:
    """Encode a single Python value as an OTLP AnyValue object.

    Bools are checked before ints because ``isinstance(True, int)`` is
    True in Python.  Lists become ``arrayValue.values`` (each element
    re-encoded recursively).  Dicts become ``kvlistValue.values`` (a
    list of ``{key, value}`` AnyValue pairs).  ``None`` round-trips as
    an empty AnyValue (``{}``) per OTLP convention.
    """
    if value is None:
        return {}
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, str):
        return {"stringValue": value}
    if isinstance(value, bytes):
        # OTLP bytesValue is base64-encoded per JSON encoding spec.
        import base64
        return {"bytesValue": base64.b64encode(value).decode("ascii")}
    if isinstance(value, (list, tuple)):
        return {"arrayValue": {"values": [_encode_value(v) for v in value]}}
    if isinstance(value, dict):
        return {
            "kvlistValue": {
                "values": [
                    {"key": str(k), "value": _encode_value(v)}
                    for k, v in value.items()
                ]
            }
        }
    # Fallback: stringify unknown types so attribute encoding is total.
    return {"stringValue": str(value)}


def _decode_value(any_value: Dict[str, Any]) -> Any:
    """Decode one OTLP AnyValue object back to a plain Python value."""
    if not any_value:
        return None
    if "stringValue" in any_value:
        return any_value["stringValue"]
    if "intValue" in any_value:
        # OTLP encodes 64-bit ints as JSON strings; restore to int.
        raw = any_value["intValue"]
        try:
            return int(raw)
        except (TypeError, ValueError):
            return raw
    if "boolValue" in any_value:
        return bool(any_value["boolValue"])
    if "doubleValue" in any_value:
        return float(any_value["doubleValue"])
    if "bytesValue" in any_value:
        import base64
        try:
            return base64.b64decode(any_value["bytesValue"])
        except (ValueError, TypeError):
            return any_value["bytesValue"]
    if "arrayValue" in any_value:
        values = any_value["arrayValue"].get("values") or []
        return [_decode_value(v) for v in values]
    if "kvlistValue" in any_value:
        values = any_value["kvlistValue"].get("values") or []
        return {pair["key"]: _decode_value(pair.get("value", {})) for pair in values}
    # Unknown AnyValue shape: pass through verbatim so callers can inspect.
    return any_value


def encode_attrs(d: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Encode a plain Python dict as an OTLP/JSON attribute list.

    Returns ``[{"key": k, "value": <AnyValue>}, ...]``.  Keys are
    coerced to strings; values are encoded by :func:`_encode_value`.
    """
    if not d:
        return []
    return [{"key": str(k), "value": _encode_value(v)} for k, v in d.items()]


def decode_attrs(lst: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Decode an OTLP/JSON attribute list back to a plain Python dict.

    The inverse of :func:`encode_attrs`.  Unknown keys without a
    ``value`` field are skipped silently (defensive: external producers
    might emit malformed attribute lists).
    """
    out: Dict[str, Any] = {}
    if not lst:
        return out
    for pair in lst:
        if not isinstance(pair, dict):
            continue
        key = pair.get("key")
        if not isinstance(key, str):
            continue
        out[key] = _decode_value(pair.get("value", {}))
    return out


__all__ = ["decode_attrs", "encode_attrs"]
