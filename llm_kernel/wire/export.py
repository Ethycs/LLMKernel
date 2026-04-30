"""JSON Schema export for llm_kernel.wire.

Usage::

    python -m llm_kernel.wire.export [out_dir]

Writes one JSON file per tool schema (input + output) and one per
family envelope shape.  Output filenames:

    tool.<name>.input.json
    tool.<name>.output.json
    family-a.json
    family-b.json
    family-c.json
    family-f.json
    family-g.json

CI regenerates and diffs against the in-tree ``wire/schemas/`` to
detect drift between Python validators and emitted JSON.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

from .tools import TOOL_CATALOG
from .version import WIRE_VERSION


def _family_schema(
    family_letter: str,
    type_value: str,
    description: str,
) -> Dict[str, Any]:
    """Build a minimal JSON Schema stub for a family envelope shape.

    The full payload shape is kind-discriminated and validated by
    wire.tools validators; this export captures the top-level envelope
    structure (``type`` + ``payload``) as the public schema contract.
    """
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": f"Family {family_letter.upper()} envelope",
        "description": description,
        "wire_version": WIRE_VERSION,
        "type": "object",
        "required": ["type", "payload"],
        "additionalProperties": True,
        "properties": {
            "type": {"type": "string", "const": type_value} if type_value else {"type": "string"},
            "payload": {"type": "object"},
            "correlation_id": {"type": "string"},
        },
    }


FAMILY_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "family-a": _family_schema(
        "a", "operator.action",
        "Family A: operator-action envelopes (RFC-006 §1). "
        "Rides on IOPub display_data / update_display_data.",
    ),
    "family-b": _family_schema(
        "b", "layout.edit",
        "Family B: layout edits (RFC-006 §4). "
        "Extension -> kernel; kernel echoes layout.update.",
    ),
    "family-c": _family_schema(
        "c", "agent_graph.command",
        "Family C: agent-graph commands (RFC-006 §5). "
        "correlation_id required for request/response pairing.",
    ),
    "family-f": _family_schema(
        "f", "notebook.metadata",
        "Family F: notebook metadata snapshots / patches / hydrate (RFC-006 §8). "
        "Bidirectional in v2.0.2.",
    ),
    "family-g": _family_schema(
        "g", "",
        "Family G: lifecycle envelopes (RFC-006 §7). "
        "Includes kernel.shutdown_request, heartbeat.kernel, kernel.handshake (V1.5+).",
    ),
}


def export_schemas(out_dir: Path) -> list[Path]:
    """Export all schemas to ``out_dir``. Returns list of written paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # Tool schemas
    for tool_name, (input_schema, output_schema, _desc) in TOOL_CATALOG.items():
        for kind, schema in (("input", input_schema), ("output", output_schema)):
            fname = out_dir / f"tool.{tool_name}.{kind}.json"
            fname.write_text(
                json.dumps(schema, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            written.append(fname)

    # Family envelope schemas
    for name, schema in FAMILY_SCHEMAS.items():
        fname = out_dir / f"{name}.json"
        fname.write_text(
            json.dumps(schema, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        written.append(fname)

    return written


def main() -> None:
    """Entry point: ``python -m llm_kernel.wire.export [out_dir]``."""
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "schemas"
    written = export_schemas(out_dir)
    print(f"Exported {len(written)} schema file(s) to {out_dir}")


if __name__ == "__main__":
    main()
