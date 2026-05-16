"""Operator-action intent handlers (PLAN-S5.0.4 §3.3+).

Per RFC-006 §6 ``operator.action`` envelopes carry an
``action_type`` discriminator that the :mod:`mcp_server`'s
``_route_operator_action`` dispatcher splits on. For action types
that warrant non-trivial logic (vs. a one-shot logger.info call),
the handler implementation lives here as a focused module so the
dispatcher stays a thin routing table.

V1 module: :mod:`.promote_stream_magic` -- the recovery affordance
for privileged-agent stream emissions per PLAN-S5.0.4.
"""

from __future__ import annotations

__all__ = ("promote_stream_magic",)
