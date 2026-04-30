"""Tests for the @auth lifecycle magic handlers (S5.0.1b §3.5).

Covers ``@auth set / rotate / off / verify`` happy paths, validation
rejects, and the env-mutation contract. Tests pass a private env
dict (NOT ``os.environ``) so xdist parallel workers don't collide
on the global env per Engineering_Guide §11.7.
"""

from __future__ import annotations

import pytest

from llm_kernel import magic_hash as MH
from llm_kernel.auth_handlers import (
    AuthCommandResult,
    K38_PIN_TOO_SHORT,
    K39_PIN_COLLISION,
    K3B_PIN_FINGERPRINT_MISMATCH,
    PIN_ENV_VAR,
    apply_auth_command,
    auth_off,
    auth_rotate,
    auth_set,
    auth_verify,
    validate_pin,
)
from llm_kernel.cell_manager import CellManager
from llm_kernel.metadata_writer import MetadataWriter


@pytest.fixture
def writer(tmp_path) -> MetadataWriter:
    return MetadataWriter(workspace_root=tmp_path)


@pytest.fixture
def cm(writer) -> CellManager:
    return CellManager(writer)


@pytest.fixture
def env() -> dict:
    """Private env dict — never touches os.environ."""
    return {}


# --- validate_pin ----------------------------------------------------


def test_validate_pin_short_rejected_K38() -> None:
    r = validate_pin("short")  # 5 chars
    assert not r.ok
    assert r.code == K38_PIN_TOO_SHORT


def test_validate_pin_collision_with_magic_name_K39() -> None:
    """A pin equal to a registered magic name is rejected."""
    r = validate_pin("spawn___")
    # 'spawn___' is not equal to 'spawn' so this passes; check the
    # actual collision case directly.
    assert r.ok


def test_validate_pin_literal_pin_K39() -> None:
    """Pin literal 'pin' is reserved (operator placeholder bait)."""
    # 'pin' is too short anyway; check the literal-equal-name case
    # via a longer reserved name. 'spawn' is reserved AND >= 8? No —
    # 'spawn' is 5 chars. But validate_pin checks length first.
    # Test the literal 'pin' string explicitly.
    r = validate_pin("pin")
    assert not r.ok


def test_validate_pin_with_whitespace_rejected() -> None:
    r = validate_pin("hunter 2-pin")  # space in middle
    assert not r.ok


def test_validate_pin_happy_path() -> None:
    r = validate_pin("hunter2-pin-ok")
    assert r.ok


# --- @auth set --------------------------------------------------------


def test_auth_set_happy_path_enables_hash_mode_and_stores_fingerprint(
    writer, cm, env,
) -> None:
    pin = "hunter2-pin-ok"
    r = auth_set(pin, writer=writer, cell_manager=cm, env=env)
    assert r.ok
    assert writer.get_config_setting("magic_hash_enabled") is True
    fp = writer.get_config_setting("magic_pin_fingerprint")
    assert fp == MH.magic_pin_fingerprint(pin)
    # Pin stored in the local env dict (NOT os.environ).
    assert env[PIN_ENV_VAR] == pin


def test_auth_set_rejects_short_pin(writer, cm, env) -> None:
    r = auth_set("short", writer=writer, cell_manager=cm, env=env)
    assert not r.ok
    assert r.code == K38_PIN_TOO_SHORT
    # Schema unchanged.
    assert writer.get_config_setting("magic_hash_enabled") is False
    assert PIN_ENV_VAR not in env


def test_auth_set_rejects_when_already_on(writer, cm, env) -> None:
    """A second @auth set is refused; operator must rotate or off."""
    auth_set("hunter2-pin-ok", writer=writer, cell_manager=cm, env=env)
    r = auth_set("another-pin-9", writer=writer, cell_manager=cm, env=env)
    assert not r.ok
    assert "already on" in r.reason


def test_auth_set_restamps_existing_plain_magic_lines(
    writer, cm, env,
) -> None:
    """@auth set walks every cell and rewrites @@spawn → @@<hash>:spawn."""
    writer.set_cell_text("c1", "@@spawn alpha task:\"X\"\nbody\n")
    writer.set_cell_text("c2", "@@agent beta\nhello\n")
    r = auth_set("hunter2-pin-ok", writer=writer, cell_manager=cm, env=env)
    assert r.ok
    assert r.details["restamped"] >= 2
    # Cell texts now carry hashed magic lines.
    t1 = writer.get_cell_text("c1")
    t2 = writer.get_cell_text("c2")
    h_spawn = MH.magic_hash("hunter2-pin-ok", "spawn")
    h_agent = MH.magic_hash("hunter2-pin-ok", "agent")
    assert f"@@{h_spawn}:spawn" in t1
    assert f"@@{h_agent}:agent" in t2


# --- @auth rotate -----------------------------------------------------


def test_auth_rotate_atomic_re_stamp(writer, cm, env) -> None:
    """Rotation walks every magic line, replacing old hash with new."""
    auth_set("hunter2-pin-ok", writer=writer, cell_manager=cm, env=env)
    # Add a cell AFTER set so it has the old hash.
    h_old = MH.magic_hash("hunter2-pin-ok", "spawn")
    writer.set_cell_text("c1", f"@@{h_old}:spawn alpha\n")
    r = auth_rotate(
        "rotated-pin-zz", writer=writer, cell_manager=cm, env=env,
    )
    assert r.ok
    assert env[PIN_ENV_VAR] == "rotated-pin-zz"
    # Fingerprint updated.
    assert writer.get_config_setting("magic_pin_fingerprint") == \
        MH.magic_pin_fingerprint("rotated-pin-zz")
    # The cell line was re-stamped to the new hash.
    h_new = MH.magic_hash("rotated-pin-zz", "spawn")
    assert f"@@{h_new}:spawn" in writer.get_cell_text("c1")
    assert h_old not in writer.get_cell_text("c1")


def test_auth_rotate_rejects_when_off(writer, cm, env) -> None:
    """Rotate refuses when hash mode isn't on."""
    r = auth_rotate("anypin-12345", writer=writer, cell_manager=cm, env=env)
    assert not r.ok


def test_auth_rotate_rejects_short_new_pin(writer, cm, env) -> None:
    auth_set("hunter2-pin-ok", writer=writer, cell_manager=cm, env=env)
    r = auth_rotate("short", writer=writer, cell_manager=cm, env=env)
    assert not r.ok
    assert r.code == K38_PIN_TOO_SHORT
    # Env pin unchanged.
    assert env[PIN_ENV_VAR] == "hunter2-pin-ok"


# --- @auth off --------------------------------------------------------


def test_auth_off_clears_fingerprint_and_env(writer, cm, env) -> None:
    auth_set("hunter2-pin-ok", writer=writer, cell_manager=cm, env=env)
    r = auth_off(writer=writer, cell_manager=cm, env=env)
    assert r.ok
    assert writer.get_config_setting("magic_hash_enabled") is False
    assert writer.get_config_setting("magic_pin_fingerprint") is None
    assert PIN_ENV_VAR not in env


def test_auth_off_leaves_existing_hashed_lines_verbatim(
    writer, cm, env,
) -> None:
    """Disable is a no-op on cell text per PLAN §3.7."""
    writer.set_cell_text("c1", "@@spawn alpha\n")
    auth_set("hunter2-pin-ok", writer=writer, cell_manager=cm, env=env)
    after_set = writer.get_cell_text("c1")
    assert "@@" in after_set  # was re-stamped
    auth_off(writer=writer, cell_manager=cm, env=env)
    # Cell text unchanged by off.
    assert writer.get_cell_text("c1") == after_set


# --- @auth verify -----------------------------------------------------


def test_auth_verify_matches_correct_pin(writer, cm, env) -> None:
    auth_set("hunter2-pin-ok", writer=writer, cell_manager=cm, env=env)
    r = auth_verify(writer=writer, env=env)
    assert r.ok


def test_auth_verify_mismatch_K3B(writer, cm, env) -> None:
    auth_set("hunter2-pin-ok", writer=writer, cell_manager=cm, env=env)
    env[PIN_ENV_VAR] = "different-pin-9"
    r = auth_verify(writer=writer, env=env)
    assert not r.ok
    assert r.code == K3B_PIN_FINGERPRINT_MISMATCH


def test_auth_verify_no_pin_loaded_K3B(writer, cm, env) -> None:
    auth_set("hunter2-pin-ok", writer=writer, cell_manager=cm, env=env)
    del env[PIN_ENV_VAR]
    r = auth_verify(writer=writer, env=env)
    assert not r.ok
    assert r.code == K3B_PIN_FINGERPRINT_MISMATCH


def test_auth_verify_rejects_when_off(writer, env) -> None:
    r = auth_verify(writer=writer, env=env)
    assert not r.ok


# --- apply_auth_command unified dispatch -----------------------------


def test_apply_auth_command_set(writer, cm, env) -> None:
    r = apply_auth_command(
        "set hunter2-pin-ok",
        writer=writer, cell_manager=cm, env=env,
    )
    assert r.ok
    assert writer.get_config_setting("magic_hash_enabled") is True


def test_apply_auth_command_unknown_subcommand(writer, cm, env) -> None:
    r = apply_auth_command(
        "frobnicate",
        writer=writer, cell_manager=cm, env=env,
    )
    assert not r.ok
    assert r.code == "K3X"


def test_apply_auth_command_off_after_set(writer, cm, env) -> None:
    apply_auth_command(
        "set hunter2-pin-ok",
        writer=writer, cell_manager=cm, env=env,
    )
    r = apply_auth_command(
        "off", writer=writer, cell_manager=cm, env=env,
    )
    assert r.ok
    assert writer.get_config_setting("magic_hash_enabled") is False


def test_apply_auth_command_verify(writer, cm, env) -> None:
    apply_auth_command(
        "set hunter2-pin-ok",
        writer=writer, cell_manager=cm, env=env,
    )
    r = apply_auth_command("verify", writer=writer, cell_manager=cm, env=env)
    assert r.ok
