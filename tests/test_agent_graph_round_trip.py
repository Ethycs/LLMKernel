"""K-MW Family C agent-graph state machine (RFC-006 §5).

Covers :meth:`MetadataWriter.apply_agent_graph_command` mutations
(``upsert_node | remove_node | upsert_edge | remove_edge``) and queries
(``neighbors | paths | subgraph | full_snapshot``) and the response
shape per RFC-006 §"Family C".
"""

from __future__ import annotations

from llm_kernel.metadata_writer import MetadataWriter


def _new_writer() -> MetadataWriter:
    return MetadataWriter(autosave_interval_sec=999.0)


def _seed(writer: MetadataWriter) -> None:
    writer.apply_agent_graph_command(
        command="upsert_node",
        parameters={
            "node": {"id": "agent:alpha", "type": "agent",
                     "properties": {"status": "idle"}},
        },
    )
    writer.apply_agent_graph_command(
        command="upsert_node",
        parameters={
            "node": {"id": "zone-refactor", "type": "zone", "properties": {}},
        },
    )
    writer.apply_agent_graph_command(
        command="upsert_node",
        parameters={
            "node": {"id": "tool:notify", "type": "tool", "properties": {}},
        },
    )
    writer.apply_agent_graph_command(
        command="upsert_edge",
        parameters={"edge": {
            "source": "agent:alpha", "target": "zone-refactor",
            "kind": "in_zone",
        }},
    )
    writer.apply_agent_graph_command(
        command="upsert_edge",
        parameters={"edge": {
            "source": "agent:alpha", "target": "tool:notify",
            "kind": "has_tool",
        }},
    )


def test_upsert_node_mutation_returns_ok_and_version() -> None:
    """Mutation responses carry ``ok`` and a fresh ``snapshot_version``."""
    writer = _new_writer()
    response = writer.apply_agent_graph_command(
        command="upsert_node",
        parameters={"node": {"id": "agent:alpha", "type": "agent"}},
    )
    assert response["ok"] is True
    assert response["snapshot_version"] >= 1


def test_upsert_node_idempotent_for_same_payload() -> None:
    """Re-upserting the same node still bumps version (last-writer-wins)."""
    writer = _new_writer()
    r1 = writer.apply_agent_graph_command(
        command="upsert_node",
        parameters={"node": {"id": "agent:alpha", "type": "agent"}},
    )
    r2 = writer.apply_agent_graph_command(
        command="upsert_node",
        parameters={"node": {"id": "agent:alpha", "type": "agent",
                             "properties": {"status": "busy"}}},
    )
    assert r1["ok"] and r2["ok"]
    snap = writer.apply_agent_graph_command(
        command="full_snapshot", parameters={},
    )
    nodes = [n for n in snap["nodes"] if n["id"] == "agent:alpha"]
    assert len(nodes) == 1
    assert nodes[0]["properties"]["status"] == "busy"


def test_upsert_node_rejects_invalid_payload() -> None:
    """Missing id / type returns ok=False without raising."""
    writer = _new_writer()
    r1 = writer.apply_agent_graph_command(
        command="upsert_node",
        parameters={"node": {"id": "agent:alpha"}},  # no type
    )
    r2 = writer.apply_agent_graph_command(
        command="upsert_node",
        parameters={"node": {"type": "agent"}},  # no id
    )
    assert r1["ok"] is False
    assert r2["ok"] is False


def test_upsert_edge_requires_both_endpoints_to_exist() -> None:
    """Edge upsert rejects when either endpoint is not in nodes[]."""
    writer = _new_writer()
    writer.apply_agent_graph_command(
        command="upsert_node",
        parameters={"node": {"id": "a", "type": "agent"}},
    )
    response = writer.apply_agent_graph_command(
        command="upsert_edge",
        parameters={"edge": {"source": "a", "target": "b", "kind": "spawned"}},
    )
    assert response["ok"] is False


def test_remove_node_drops_incident_edges() -> None:
    """Removing a node also removes edges referencing it."""
    writer = _new_writer()
    _seed(writer)
    response = writer.apply_agent_graph_command(
        command="remove_node",
        parameters={"node_id": "agent:alpha"},
    )
    assert response["ok"] is True
    snap = writer.apply_agent_graph_command(
        command="full_snapshot", parameters={},
    )
    assert all(n["id"] != "agent:alpha" for n in snap["nodes"])
    assert all(
        e["source"] != "agent:alpha" and e["target"] != "agent:alpha"
        for e in snap["edges"]
    )


def test_remove_edge() -> None:
    """remove_edge drops a specific (source, target, kind) triple."""
    writer = _new_writer()
    _seed(writer)
    response = writer.apply_agent_graph_command(
        command="remove_edge",
        parameters={
            "source": "agent:alpha", "target": "tool:notify", "kind": "has_tool",
        },
    )
    assert response["ok"] is True
    snap = writer.apply_agent_graph_command(
        command="full_snapshot", parameters={},
    )
    kinds = {(e["source"], e["target"], e["kind"]) for e in snap["edges"]}
    assert ("agent:alpha", "tool:notify", "has_tool") not in kinds


def test_full_snapshot_returns_all_nodes_and_edges() -> None:
    """full_snapshot returns the entire graph; not truncated."""
    writer = _new_writer()
    _seed(writer)
    snap = writer.apply_agent_graph_command(
        command="full_snapshot", parameters={},
    )
    assert snap["truncated"] is False
    ids = {n["id"] for n in snap["nodes"]}
    assert ids == {"agent:alpha", "zone-refactor", "tool:notify"}
    kinds = {(e["source"], e["target"], e["kind"]) for e in snap["edges"]}
    assert ("agent:alpha", "zone-refactor", "in_zone") in kinds
    assert ("agent:alpha", "tool:notify", "has_tool") in kinds


def test_neighbors_returns_one_hop() -> None:
    """neighbors with hops=1 returns just the immediate neighborhood."""
    writer = _new_writer()
    _seed(writer)
    response = writer.apply_agent_graph_command(
        command="neighbors",
        parameters={"query_type": "neighbors", "node_id": "agent:alpha", "hops": 1},
    )
    ids = {n["id"] for n in response["nodes"]}
    assert ids == {"agent:alpha", "zone-refactor", "tool:notify"}
    assert response["truncated"] is False


def test_neighbors_with_edge_filter() -> None:
    """edge_filters narrows the traversal to specific edge kinds."""
    writer = _new_writer()
    _seed(writer)
    response = writer.apply_agent_graph_command(
        command="neighbors",
        parameters={
            "node_id": "agent:alpha", "hops": 1,
            "edge_filters": ["in_zone"],
        },
    )
    ids = {n["id"] for n in response["nodes"]}
    assert ids == {"agent:alpha", "zone-refactor"}


def test_paths_finds_route() -> None:
    """paths returns nodes+edges along the discovered route."""
    writer = _new_writer()
    _seed(writer)
    response = writer.apply_agent_graph_command(
        command="paths",
        parameters={
            "source": "tool:notify", "target": "zone-refactor", "hops": 4,
        },
    )
    ids = {n["id"] for n in response["nodes"]}
    assert "agent:alpha" in ids
    assert "zone-refactor" in ids
    assert "tool:notify" in ids


def test_subgraph_returns_only_requested_ids() -> None:
    """subgraph returns nodes and edges constrained to the given IDs."""
    writer = _new_writer()
    _seed(writer)
    response = writer.apply_agent_graph_command(
        command="subgraph",
        parameters={"node_ids": ["agent:alpha", "tool:notify"]},
    )
    ids = {n["id"] for n in response["nodes"]}
    assert ids == {"agent:alpha", "tool:notify"}
    # The in_zone edge is excluded because zone-refactor is not in the set.
    kinds = {(e["source"], e["target"], e["kind"]) for e in response["edges"]}
    assert kinds == {("agent:alpha", "tool:notify", "has_tool")}


def test_unknown_command_returns_ok_false_no_raise() -> None:
    """Unknown commands return a mutation-style ok=False payload."""
    writer = _new_writer()
    response = writer.apply_agent_graph_command(
        command="not_a_real_command", parameters={},
    )
    assert response["ok"] is False
    assert "snapshot_version" in response
