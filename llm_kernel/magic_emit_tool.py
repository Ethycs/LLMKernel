"""Privileged-agent magic-emit MCP tool handler -- PLAN-S5.0.4 §3.1.

Per [discipline/certified-magic-emitter](../docs/atoms/discipline/certified-magic-emitter.md)
clause 1, an agent the operator has granted ``magic_emit_privileges``
may invoke the kernel-side ``emit_magic_cell`` MCP tool to produce a
new cell on the operator's behalf. The agent never *types* magic; it
*invokes* it -- the kernel handler composes the canonical cell text,
validates the privilege grant, and dispatches through Cell Manager's
structural-write surface (clause 2) with ``generated_by: <agent_id>``
provenance (clause 4).

Public surface:

* :func:`emit_magic_cell` -- the tool entry point. Returns
  ``{"cell_id": <new_id>}`` on success, raises :class:`MagicEmitError`
  with K3K (``unprivileged_agent_magic_emit``) when the privilege
  grant is missing.

This module is structurally certified per clause 2: it never writes
to ``cells[<id>].outputs`` and never appends to text outside
``CellManager.insert_cells_with_provenance``. The lint pass in the
companion test walks this module's source asserting the rule.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:  # pragma: no cover
    from .cell_manager import CellManager
    from .metadata_writer import MetadataWriter

from .wire.tools import K3K_UNPRIVILEGED_AGENT_MAGIC_EMIT


__all__ = (
    "MagicEmitError",
    "K3K_UNPRIVILEGED_AGENT_MAGIC_EMIT",
    "emit_magic_cell",
)


class MagicEmitError(ValueError):
    """Raised by :func:`emit_magic_cell` on privilege / input failures.

    Carries a K-code (K3K when the privilege grant is missing) and a
    human-readable reason. The wire dispatcher catches this and
    re-wraps as a structured tool-call rejection so the agent's
    JSON-RPC response is typed.
    """

    def __init__(self, code: str, reason: str) -> None:
        super().__init__(f"{code}: {reason}")
        self.code: str = code
        self.reason: str = reason


def _utc_now_iso() -> str:
    """ISO8601 UTC with trailing ``Z`` -- matches generator dispatcher."""
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _format_args(args: Optional[Dict[str, Any]]) -> str:
    """Render the ``args`` dict as the magic-line tail.

    PLAN-S5.0.4 §4 schema: ``args: { <name>: <string_value> }`` --
    every value is a string per the tool's input schema. We emit
    ``k=v`` pairs joined by single spaces. Empty / None ``args``
    returns the empty string (caller composes ``"@@<name>"`` with no
    tail).
    """
    if not args:
        return ""
    if not isinstance(args, dict):
        return ""
    pieces = []
    for key, value in args.items():
        if not isinstance(key, str) or not key:
            continue
        # Coerce non-string values defensively; the wire validator
        # restricts to str but the kernel-side dispatcher is the
        # final trust boundary.
        sval = value if isinstance(value, str) else str(value)
        pieces.append(f"{key}={sval}")
    return " ".join(pieces)


def emit_magic_cell(
    *,
    agent_id: str,
    zone_id: str,
    name: str,
    args: Dict[str, Any],
    body: Optional[str],
    position: Dict[str, Any],
    writer: "MetadataWriter",
    cell_manager: "CellManager",
    promoted_from_stream: bool = False,
) -> Dict[str, str]:
    """Privileged-agent magic-emit entry point.

    PLAN-S5.0.4 §3.1 / §4. Behaviour:

    1. Validate ``agent_id`` / ``zone_id`` / ``name`` / ``position``.
    2. Look up the privilege grant via
       :meth:`MetadataWriter.has_magic_emit_privilege`. Reject K3K if
       no covering entry.
    3. Compose canonical cell text via :func:`emit_magic_line` (handles
       hash mode by stamping ``@@<HMAC(pin, name)>:<name>`` when the
       operator's pin is set).
    4. Dispatch through
       :meth:`CellManager.insert_cells_with_provenance` with
       ``generated_by=<agent_id>``, ``promoted_from_stream`` per the
       caller's request.
    5. Return ``{"cell_id": <new_id>}``.

    ``body`` is appended after the magic-line on its own line(s) when
    non-empty/None -- this is the cell-magic body for ``@@expand``-
    style emissions. ``args`` keys become ``k=v`` tail tokens on the
    magic line.

    The function never writes to ``cells[<id>].outputs`` or to
    ``cells[<id>].text`` directly -- every mutation flows through the
    Cell Manager structural-write API per certified-magic-emitter
    clause 2.
    """
    # --- Argument validation -------------------------------------------------
    if not isinstance(agent_id, str) or not agent_id:
        raise MagicEmitError(
            K3K_UNPRIVILEGED_AGENT_MAGIC_EMIT,
            "agent_id required",
        )
    if not isinstance(zone_id, str) or not zone_id:
        raise MagicEmitError(
            K3K_UNPRIVILEGED_AGENT_MAGIC_EMIT,
            "zone_id required",
        )
    if not isinstance(name, str) or not name:
        raise MagicEmitError(
            K3K_UNPRIVILEGED_AGENT_MAGIC_EMIT,
            "magic name required",
        )
    if not isinstance(position, dict):
        raise MagicEmitError(
            K3K_UNPRIVILEGED_AGENT_MAGIC_EMIT,
            "position must be a dict {after_cell_id: <id>}",
        )
    after_cell_id = position.get("after_cell_id")
    if not isinstance(after_cell_id, str) or not after_cell_id:
        raise MagicEmitError(
            K3K_UNPRIVILEGED_AGENT_MAGIC_EMIT,
            "position.after_cell_id required",
        )

    # --- Privilege validation (clause 1) ------------------------------------
    has_grant_fn = getattr(writer, "has_magic_emit_privilege", None)
    if not callable(has_grant_fn):
        # Defense-in-depth: a writer without the privilege API cannot
        # authorize anything. This is the same shape as an empty
        # grants list.
        raise MagicEmitError(
            K3K_UNPRIVILEGED_AGENT_MAGIC_EMIT,
            f"no privilege grant for agent_id={agent_id!r} "
            f"zone_id={zone_id!r} magic={name!r}",
        )
    if not has_grant_fn(agent_id=agent_id, zone_id=zone_id, magic_name=name):
        raise MagicEmitError(
            K3K_UNPRIVILEGED_AGENT_MAGIC_EMIT,
            f"no privilege grant for agent_id={agent_id!r} "
            f"zone_id={zone_id!r} magic={name!r}",
        )

    # --- Compose canonical cell text (clause 3) -----------------------------
    from .magic_hash import emit_magic_line

    hash_enabled = bool(writer.get_config_setting("magic_hash_enabled"))
    pin: Optional[str] = None
    if hash_enabled:
        pin_getter = getattr(writer, "get_operator_pin", None)
        if callable(pin_getter):
            pin = pin_getter()
    args_tail = _format_args(args)
    magic_line = emit_magic_line(
        name, args_tail, hash_enabled=hash_enabled, pin=pin,
    )
    if isinstance(body, str) and body:
        magic_text = f"{magic_line}\n{body}"
    else:
        magic_text = magic_line

    # --- Structural write (clause 2 + 4) ------------------------------------
    new_ids = cell_manager.insert_cells_with_provenance(
        after_cell_id=after_cell_id,
        magic_texts=[magic_text],
        generated_by=agent_id,
        generated_at=_utc_now_iso(),
        promoted_from_stream=promoted_from_stream,
    )
    if not new_ids:
        # ``insert_cells_with_provenance`` is atomic; an empty result
        # means the dispatcher rejected. Surface a typed error.
        raise MagicEmitError(
            K3K_UNPRIVILEGED_AGENT_MAGIC_EMIT,
            f"cell manager did not insert any cells for {name!r}",
        )
    return {"cell_id": new_ids[0]}
