"""K-MW Family B layout state machine (RFC-006 §4 / RFC-005 §layout).

Covers :meth:`MetadataWriter.apply_layout_edit` for each operation
(``add_zone | remove_node | move_node | rename_node |
update_render_hints``) and :meth:`MetadataWriter.emit_layout_update`
returning the new ``{snapshot_version, tree}`` payload.
"""

from __future__ import annotations

from llm_kernel.metadata_writer import MetadataWriter


def _new_writer() -> MetadataWriter:
    return MetadataWriter(autosave_interval_sec=999.0)


def test_apply_layout_edit_add_zone_increments_version() -> None:
    """Adding a zone under root mutates the tree and bumps version."""
    writer = _new_writer()
    initial = writer.emit_layout_update()["snapshot_version"]
    new_version = writer.apply_layout_edit(
        operation="add_zone",
        parameters={
            "node_spec": {
                "id": "zone-refactor", "type": "zone",
                "render_hints": {"color": "#4a90e2"},
            },
        },
    )
    assert new_version == initial + 1
    payload = writer.emit_layout_update()
    assert payload["snapshot_version"] == new_version
    children = payload["tree"]["children"]
    assert len(children) == 1
    assert children[0]["id"] == "zone-refactor"
    assert children[0]["render_hints"]["color"] == "#4a90e2"


def test_apply_layout_edit_add_zone_under_named_parent() -> None:
    """``new_parent_id`` routes the new node under that parent."""
    writer = _new_writer()
    writer.apply_layout_edit(
        operation="add_zone",
        parameters={"node_spec": {"id": "zone-a", "type": "zone"}},
    )
    writer.apply_layout_edit(
        operation="add_zone",
        parameters={
            "new_parent_id": "zone-a",
            "node_spec": {"id": "src/file.rs", "type": "file"},
        },
    )
    tree = writer.emit_layout_update()["tree"]
    zone = tree["children"][0]
    assert zone["id"] == "zone-a"
    assert zone["children"][0]["id"] == "src/file.rs"


def test_apply_layout_edit_rejects_duplicate_id() -> None:
    """ID uniqueness is enforced; duplicate insert is a no-op."""
    writer = _new_writer()
    v1 = writer.apply_layout_edit(
        operation="add_zone",
        parameters={"node_spec": {"id": "zone-a", "type": "zone"}},
    )
    v2 = writer.apply_layout_edit(
        operation="add_zone",
        parameters={"node_spec": {"id": "zone-a", "type": "zone"}},
    )
    assert v2 == v1  # version unchanged on rejection
    tree = writer.emit_layout_update()["tree"]
    assert len(tree["children"]) == 1


def test_apply_layout_edit_remove_node() -> None:
    """remove_node deletes the named subtree."""
    writer = _new_writer()
    writer.apply_layout_edit(
        operation="add_zone",
        parameters={"node_spec": {"id": "zone-a", "type": "zone"}},
    )
    v_before = writer.emit_layout_update()["snapshot_version"]
    v_after = writer.apply_layout_edit(
        operation="remove_node",
        parameters={"node_id": "zone-a"},
    )
    assert v_after == v_before + 1
    assert writer.emit_layout_update()["tree"]["children"] == []


def test_apply_layout_edit_remove_unknown_node_is_noop() -> None:
    """Removing a non-existent node leaves state untouched."""
    writer = _new_writer()
    v_before = writer.emit_layout_update()["snapshot_version"]
    v_after = writer.apply_layout_edit(
        operation="remove_node",
        parameters={"node_id": "nope"},
    )
    assert v_after == v_before


def test_apply_layout_edit_move_node() -> None:
    """move_node re-parents an existing node."""
    writer = _new_writer()
    writer.apply_layout_edit(
        operation="add_zone",
        parameters={"node_spec": {"id": "zone-a", "type": "zone"}},
    )
    writer.apply_layout_edit(
        operation="add_zone",
        parameters={"node_spec": {"id": "zone-b", "type": "zone"}},
    )
    writer.apply_layout_edit(
        operation="add_zone",
        parameters={
            "new_parent_id": "zone-a",
            "node_spec": {"id": "src/file.rs", "type": "file"},
        },
    )
    writer.apply_layout_edit(
        operation="move_node",
        parameters={"node_id": "src/file.rs", "new_parent_id": "zone-b"},
    )
    tree = writer.emit_layout_update()["tree"]
    zone_a = next(c for c in tree["children"] if c["id"] == "zone-a")
    zone_b = next(c for c in tree["children"] if c["id"] == "zone-b")
    assert zone_a["children"] == []
    assert len(zone_b["children"]) == 1
    assert zone_b["children"][0]["id"] == "src/file.rs"


def test_apply_layout_edit_move_into_descendant_is_rejected() -> None:
    """Cannot move a node under one of its own descendants."""
    writer = _new_writer()
    writer.apply_layout_edit(
        operation="add_zone",
        parameters={"node_spec": {"id": "zone-a", "type": "zone"}},
    )
    writer.apply_layout_edit(
        operation="add_zone",
        parameters={
            "new_parent_id": "zone-a",
            "node_spec": {"id": "zone-a-child", "type": "zone"},
        },
    )
    v_before = writer.emit_layout_update()["snapshot_version"]
    v_after = writer.apply_layout_edit(
        operation="move_node",
        parameters={"node_id": "zone-a", "new_parent_id": "zone-a-child"},
    )
    assert v_after == v_before


def test_apply_layout_edit_rename_node() -> None:
    """rename_node updates the id and bumps the version."""
    writer = _new_writer()
    writer.apply_layout_edit(
        operation="add_zone",
        parameters={"node_spec": {"id": "zone-a", "type": "zone"}},
    )
    v_before = writer.emit_layout_update()["snapshot_version"]
    v_after = writer.apply_layout_edit(
        operation="rename_node",
        parameters={"node_id": "zone-a", "new_name": "zone-renamed"},
    )
    assert v_after == v_before + 1
    tree = writer.emit_layout_update()["tree"]
    assert tree["children"][0]["id"] == "zone-renamed"


def test_apply_layout_edit_rename_to_existing_id_is_rejected() -> None:
    """Cannot rename to a name that already exists in the tree."""
    writer = _new_writer()
    writer.apply_layout_edit(
        operation="add_zone",
        parameters={"node_spec": {"id": "zone-a", "type": "zone"}},
    )
    writer.apply_layout_edit(
        operation="add_zone",
        parameters={"node_spec": {"id": "zone-b", "type": "zone"}},
    )
    v_before = writer.emit_layout_update()["snapshot_version"]
    v_after = writer.apply_layout_edit(
        operation="rename_node",
        parameters={"node_id": "zone-a", "new_name": "zone-b"},
    )
    assert v_after == v_before


def test_apply_layout_edit_update_render_hints() -> None:
    """update_render_hints merges new hints into the node."""
    writer = _new_writer()
    writer.apply_layout_edit(
        operation="add_zone",
        parameters={
            "node_spec": {
                "id": "zone-a", "type": "zone",
                "render_hints": {"color": "#000"},
            },
        },
    )
    v_before = writer.emit_layout_update()["snapshot_version"]
    v_after = writer.apply_layout_edit(
        operation="update_render_hints",
        parameters={
            "node_id": "zone-a",
            "render_hints": {"label": "refactor", "color": "#fff"},
        },
    )
    assert v_after == v_before + 1
    tree = writer.emit_layout_update()["tree"]
    hints = tree["children"][0]["render_hints"]
    assert hints["color"] == "#fff"
    assert hints["label"] == "refactor"


def test_apply_layout_edit_unknown_operation_is_noop() -> None:
    """Unknown operation leaves state and version untouched; no raise."""
    writer = _new_writer()
    v_before = writer.emit_layout_update()["snapshot_version"]
    v_after = writer.apply_layout_edit(
        operation="not_a_real_op",
        parameters={},
    )
    assert v_after == v_before


def test_emit_layout_update_returns_a_deep_copy() -> None:
    """Mutating the returned payload's tree does not affect writer state."""
    writer = _new_writer()
    writer.apply_layout_edit(
        operation="add_zone",
        parameters={"node_spec": {"id": "zone-a", "type": "zone"}},
    )
    payload = writer.emit_layout_update()
    payload["tree"]["children"].clear()
    assert writer.emit_layout_update()["tree"]["children"][0]["id"] == "zone-a"
