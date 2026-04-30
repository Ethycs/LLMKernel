"""Hash-mode generator tests — PLAN-S5.0.2 §5.4."""

from __future__ import annotations

import os

import pytest

from llm_kernel.cell_manager import CellManager
from llm_kernel.magic_generators import (
    GeneratorError,
    K3I_GENERATOR_HANDLER_PRODUCED_INVALID_HASH,
    _with_optional_hmac,
    dispatch_generator,
)
from llm_kernel.magic_hash import magic_hash, magic_pin_fingerprint
from llm_kernel.metadata_writer import MetadataWriter


@pytest.fixture
def hashmode_writer(tmp_path, monkeypatch) -> MetadataWriter:
    """Writer with hash mode on + LLMNB_OPERATOR_PIN env var set."""
    pin = "test-pin-1234"
    monkeypatch.setenv("LLMNB_OPERATOR_PIN", pin)
    w = MetadataWriter(workspace_root=tmp_path)
    w.set_hash_mode(True, magic_pin_fingerprint(pin))
    w.set_cell_text("c_gen", "@@template greet")
    return w


@pytest.fixture
def cell_manager(hashmode_writer) -> CellManager:
    return CellManager(hashmode_writer)


def test_with_optional_hmac_no_pin_returns_unchanged() -> None:
    line = "@@scratch hello"
    assert _with_optional_hmac(line, None, "scratch") == line


def test_with_optional_hmac_stamps_with_pin() -> None:
    pin = "test-pin"
    out = _with_optional_hmac("@@scratch body", pin, "scratch")
    expected_hash = magic_hash(pin, "scratch")
    assert out.startswith(f"@@{expected_hash}:scratch")
    assert "body" in out


def test_with_optional_hmac_idempotent_on_already_hashed() -> None:
    pin = "test-pin"
    h = magic_hash(pin, "scratch")
    line = f"@@{h}:scratch body"
    assert _with_optional_hmac(line, pin, "scratch") == line


def test_dispatcher_validates_hashes_when_pin_set(
    hashmode_writer, cell_manager,
) -> None:
    """Generator output that includes a valid HMAC passes dispatcher
    validation; the cells are inserted normally."""
    pin = os.environ["LLMNB_OPERATOR_PIN"]
    h = magic_hash(pin, "scratch")
    body = f"@@{h}:scratch hello"
    hashmode_writer.set_config_template("greet", body)
    args = {"positional": ["greet"], "named": {}}
    new_ids = dispatch_generator(
        "c_gen", "template", args, "", hashmode_writer, cell_manager,
    )
    assert len(new_ids) == 1


def test_invalid_hmac_raises_K3I(hashmode_writer, cell_manager) -> None:
    """Handler emitting a fragment with bad HMAC trips K3I atomically."""
    # Manually craft a template with a bad hash (12 hex chars but not
    # valid for any registered name keyed against the test pin).
    bad_body = "@@deadbeefdead:scratch hello"
    hashmode_writer.set_config_template("bad", bad_body)
    args = {"positional": ["bad"], "named": {}}
    with pytest.raises(GeneratorError) as exc:
        dispatch_generator(
            "c_gen", "template", args, "", hashmode_writer, cell_manager,
        )
    assert exc.value.code == K3I_GENERATOR_HANDLER_PRODUCED_INVALID_HASH
    # Atomic: no cells inserted apart from the seeded c_gen.
    assert list(hashmode_writer._cells.keys()) == ["c_gen"]


def test_emission_ban_does_not_trip_on_generator_output(
    hashmode_writer, cell_manager,
) -> None:
    """Generator output goes through Cell Manager (structural write),
    NOT through agent stdout — so the emission ban (which scans
    stdout) never sees it."""
    pin = os.environ["LLMNB_OPERATOR_PIN"]
    h = magic_hash(pin, "scratch")
    body = f"@@{h}:scratch ok"
    hashmode_writer.set_config_template("safe", body)
    args = {"positional": ["safe"], "named": {}}
    new_ids = dispatch_generator(
        "c_gen", "template", args, "", hashmode_writer, cell_manager,
    )
    rec = hashmode_writer.get_cell_record(new_ids[0])
    # The hashed-magic shape survives intact in the cell text — the
    # emission ban only escapes lines coming from the agent stdout
    # path; generator output is structural and trusted.
    assert "scratch" in rec["text"]


def test_get_operator_pin_off_returns_none(tmp_path) -> None:
    """When hash mode is off, get_operator_pin returns None even with
    LLMNB_OPERATOR_PIN set."""
    os.environ["LLMNB_OPERATOR_PIN"] = "anything"
    try:
        w = MetadataWriter(workspace_root=tmp_path)
        # hash mode defaults to off
        assert w.get_operator_pin() is None
    finally:
        os.environ.pop("LLMNB_OPERATOR_PIN", None)


def test_get_operator_pin_on_returns_env_value(tmp_path) -> None:
    os.environ["LLMNB_OPERATOR_PIN"] = "secret-pin"
    try:
        w = MetadataWriter(workspace_root=tmp_path)
        w.set_hash_mode(True, magic_pin_fingerprint("secret-pin"))
        assert w.get_operator_pin() == "secret-pin"
    finally:
        os.environ.pop("LLMNB_OPERATOR_PIN", None)
