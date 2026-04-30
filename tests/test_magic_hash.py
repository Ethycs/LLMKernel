"""Unit tests for ``llm_kernel.magic_hash`` — S5.0.1a foundation slice.

Covers:

* :func:`magic_hash` determinism + length truncation + different-pin
  divergence + bad-arg rejection.
* :func:`magic_pin_fingerprint` salt usage + irreversibility.
* :func:`looks_like_hashed_magic` / :func:`looks_like_plain_magic`
  pattern detectors.
* :func:`validate_hashed_magic` constant-time hash check + name-set
  guard.
* :func:`strip_hash_from_line` / :func:`strip_hashes_from_text` —
  bidirectional strip + idempotence + name-set guard.
* :func:`escape_leading_at` — escape + idempotence.

Per Engineering_Guide §11.7 these tests are pure (no threads, no
filesystem, no env mutation); the pytest-xdist worker boundary is
trivial.
"""

from __future__ import annotations

import re

import pytest

from llm_kernel import magic_hash as MH


# --- magic_hash ------------------------------------------------------


def test_magic_hash_deterministic() -> None:
    """Same pin + name → same hash, every call."""
    a = MH.magic_hash("hunter2-pin", "spawn")
    b = MH.magic_hash("hunter2-pin", "spawn")
    assert a == b
    # Default length is 8 hex chars = 4 bytes.
    assert len(a) == 8


def test_magic_hash_length_truncation() -> None:
    """``length`` controls the truncated hex prefix length."""
    h6 = MH.magic_hash("pin", "spawn", length=6)
    h8 = MH.magic_hash("pin", "spawn", length=8)
    h12 = MH.magic_hash("pin", "spawn", length=12)
    h64 = MH.magic_hash("pin", "spawn", length=64)
    assert len(h6) == 6
    assert len(h8) == 8
    assert len(h12) == 12
    # SHA-256 hex is 64 chars; longer requests truncate to 64.
    assert len(h64) == 64
    # Prefixes are consistent across length choices (same HMAC input).
    assert h12.startswith(h8) and h8.startswith(h6)


def test_magic_hash_different_pins_diverge() -> None:
    """Different pins produce different hashes for the same magic name."""
    a = MH.magic_hash("pin-one-1234567890", "spawn")
    b = MH.magic_hash("pin-two-0987654321", "spawn")
    assert a != b


def test_magic_hash_different_names_diverge() -> None:
    """Different magic names diverge under the same pin."""
    a = MH.magic_hash("pin", "spawn")
    b = MH.magic_hash("pin", "agent")
    assert a != b


def test_magic_hash_unicode_pin_and_name() -> None:
    """UTF-8 encoding is exercised for non-ASCII pins + names."""
    a = MH.magic_hash("русский", "spawn")
    b = MH.magic_hash("русский", "spawn")
    assert a == b
    # Hex is always ASCII even with non-ASCII inputs.
    assert all(c in "0123456789abcdef" for c in a)


def test_magic_hash_long_magic_name() -> None:
    """Long names hash without truncation issues."""
    long = "x" * 1024
    a = MH.magic_hash("pin", long)
    b = MH.magic_hash("pin", long)
    assert a == b and len(a) == 8


def test_magic_hash_empty_pin_allowed_at_primitive() -> None:
    """Empty pin is permitted at the primitive layer (caller validates)."""
    # The pin lifecycle magic in slice 5.0.1b will reject empty pins
    # via K38 (magic_pin_too_short). The primitive itself must remain
    # pure — same input must produce same output regardless of caller-
    # side policy.
    a = MH.magic_hash("", "spawn")
    b = MH.magic_hash("", "spawn")
    assert a == b


def test_magic_hash_rejects_non_string() -> None:
    """Type errors surface at the primitive boundary."""
    with pytest.raises(TypeError):
        MH.magic_hash(123, "spawn")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        MH.magic_hash("pin", b"spawn")  # type: ignore[arg-type]


def test_magic_hash_rejects_non_positive_length() -> None:
    """``length`` must be a positive integer."""
    with pytest.raises(ValueError):
        MH.magic_hash("pin", "spawn", length=0)
    with pytest.raises(ValueError):
        MH.magic_hash("pin", "spawn", length=-1)


# --- magic_pin_fingerprint -------------------------------------------


def test_magic_pin_fingerprint_deterministic() -> None:
    """Same pin → same fingerprint."""
    a = MH.magic_pin_fingerprint("hunter2-pin")
    b = MH.magic_pin_fingerprint("hunter2-pin")
    assert a == b
    assert len(a) == 16
    assert all(c in "0123456789abcdef" for c in a)


def test_magic_pin_fingerprint_uses_salt() -> None:
    """Fingerprint != raw SHA-256 (the salt is mixed in)."""
    import hashlib
    pin = "hunter2-pin"
    raw = hashlib.sha256(pin.encode("utf-8")).hexdigest()[:16]
    salted = MH.magic_pin_fingerprint(pin)
    # The salt MUST shift the fingerprint away from the raw hash.
    assert raw != salted


def test_magic_pin_fingerprint_different_pins_diverge() -> None:
    """Different pins → different fingerprints."""
    assert (
        MH.magic_pin_fingerprint("pin-one-12345")
        != MH.magic_pin_fingerprint("pin-two-67890")
    )


def test_magic_pin_fingerprint_rejects_non_string() -> None:
    with pytest.raises(TypeError):
        MH.magic_pin_fingerprint(None)  # type: ignore[arg-type]


# --- looks_like_hashed_magic -----------------------------------------


def test_looks_like_hashed_magic_positive_cases() -> None:
    """Canonical hashed-magic shapes are detected."""
    assert MH.looks_like_hashed_magic("@@deadbeef:spawn alpha")
    assert MH.looks_like_hashed_magic("@@deadbe:spawn")  # 6-char hash
    assert MH.looks_like_hashed_magic("@a1b2c3d4:pin")  # line-magic sigil
    assert MH.looks_like_hashed_magic("@@01234567:checkpoint covers:c_3")


def test_looks_like_hashed_magic_negative_cases() -> None:
    """Plain text and plain magics do NOT match."""
    assert not MH.looks_like_hashed_magic("@@spawn alpha")
    assert not MH.looks_like_hashed_magic("normal prose")
    assert not MH.looks_like_hashed_magic("")
    assert not MH.looks_like_hashed_magic("\\@@deadbeef:spawn")  # escaped
    # A name with leading uppercase in the hash slot is not a hex hash.
    assert not MH.looks_like_hashed_magic("@@DEADBEEF:spawn")


def test_looks_like_hashed_magic_rejects_non_string() -> None:
    assert not MH.looks_like_hashed_magic(None)  # type: ignore[arg-type]


# --- looks_like_plain_magic ------------------------------------------


def test_looks_like_plain_magic_positive_cases() -> None:
    known = {"spawn", "agent", "pin", "exclude"}
    assert MH.looks_like_plain_magic("@@spawn alpha", known)
    assert MH.looks_like_plain_magic("@pin", known)
    assert MH.looks_like_plain_magic("@@agent beta", known)
    # Trailing colon for arg-bearing magics is acceptable.
    assert MH.looks_like_plain_magic("@@spawn:alpha", known)


def test_looks_like_plain_magic_unknown_name_is_negative() -> None:
    """An unknown ``@@xyzzy`` does not flag — operator may type literal text."""
    assert not MH.looks_like_plain_magic("@@xyzzy", {"spawn"})


def test_looks_like_plain_magic_hashed_form_does_not_match() -> None:
    """Hashed magics DON'T match the plain detector (separate layer)."""
    assert not MH.looks_like_plain_magic("@@deadbeef:spawn", {"spawn"})


def test_looks_like_plain_magic_accepts_iterable() -> None:
    """Caller may pass any iterable, not just a set."""
    assert MH.looks_like_plain_magic("@@spawn", ["spawn", "agent"])


# --- validate_hashed_magic -------------------------------------------


def test_validate_hashed_magic_correct_hash() -> None:
    """Correct hash + known name → (True, name)."""
    pin = "operator-pin-123"
    h = MH.magic_hash(pin, "spawn")
    line = f"@@{h}:spawn alpha task:\"x\""
    ok, recovered = MH.validate_hashed_magic(line, pin, {"spawn", "agent"})
    assert ok is True and recovered == "spawn"


def test_validate_hashed_magic_wrong_pin() -> None:
    """A line hashed with one pin fails validation under another pin."""
    h = MH.magic_hash("pin-A", "spawn")
    line = f"@@{h}:spawn"
    ok, recovered = MH.validate_hashed_magic(line, "pin-B", {"spawn"})
    assert ok is False and recovered is None


def test_validate_hashed_magic_unknown_name() -> None:
    """A name not in ``known_names`` fails even with a valid-shape hash."""
    pin = "p"
    h = MH.magic_hash(pin, "fakemagic")
    line = f"@@{h}:fakemagic"
    ok, recovered = MH.validate_hashed_magic(line, pin, {"spawn"})
    assert ok is False and recovered is None


def test_validate_hashed_magic_non_matching_line() -> None:
    """Plain text returns (False, None)."""
    ok, recovered = MH.validate_hashed_magic("hello", "p", {"spawn"})
    assert ok is False and recovered is None


# --- strip_hash_from_line / strip_hashes_from_text -------------------


def test_strip_hash_from_line_basic() -> None:
    known = {"spawn"}
    assert (
        MH.strip_hash_from_line("@@deadbeef:spawn alpha task:\"x\"", known)
        == "@@spawn alpha task:\"x\""
    )
    # Line-magic sigil round-trip.
    assert MH.strip_hash_from_line("@a1b2c3d4:pin", {"pin"}) == "@pin"


def test_strip_hash_from_line_unknown_name_passthrough() -> None:
    """An unknown name in the hashed shape is NOT stripped (defense)."""
    line = "@@deadbeef:fakemagic args"
    assert MH.strip_hash_from_line(line, {"spawn"}) == line


def test_strip_hash_from_line_passthrough_on_plain_text() -> None:
    assert MH.strip_hash_from_line("plain text", {"spawn"}) == "plain text"
    assert MH.strip_hash_from_line("@@spawn alpha", {"spawn"}) == "@@spawn alpha"


def test_strip_hash_from_line_idempotent() -> None:
    """Stripping a stripped line is a no-op."""
    known = {"spawn"}
    once = MH.strip_hash_from_line("@@deadbeef:spawn x", known)
    twice = MH.strip_hash_from_line(once, known)
    assert once == twice


def test_strip_hashes_from_text_multiline() -> None:
    """Every matching line is stripped; non-matching lines pass through."""
    known = {"spawn", "checkpoint"}
    text = (
        "@@deadbeef:spawn alpha\n"
        "body line\n"
        "@@feedface:checkpoint covers:c_3\n"
        "more body\n"
    )
    expected = (
        "@@spawn alpha\n"
        "body line\n"
        "@@checkpoint covers:c_3\n"
        "more body\n"
    )
    assert MH.strip_hashes_from_text(text, known) == expected


def test_strip_hashes_from_text_preserves_no_trailing_newline() -> None:
    """Text without a trailing newline round-trips without one added."""
    known = {"spawn"}
    text = "@@deadbeef:spawn x\nbody"
    out = MH.strip_hashes_from_text(text, known)
    assert out == "@@spawn x\nbody"
    assert not out.endswith("\n")


def test_strip_hashes_from_text_empty() -> None:
    assert MH.strip_hashes_from_text("", {"spawn"}) == ""


# --- escape_leading_at -----------------------------------------------


def test_escape_leading_at_basic() -> None:
    """Leading ``@`` becomes ``\\@``; rest preserved verbatim."""
    assert MH.escape_leading_at("@@deadbeef:spawn x") == "\\@@deadbeef:spawn x"
    assert MH.escape_leading_at("@pin") == "\\@pin"


def test_escape_leading_at_idempotent() -> None:
    """Already-escaped lines are unchanged."""
    once = MH.escape_leading_at("@@x")
    twice = MH.escape_leading_at(once)
    assert once == twice == "\\@@x"


def test_escape_leading_at_no_op_on_normal_text() -> None:
    assert MH.escape_leading_at("hello world") == "hello world"
    assert MH.escape_leading_at(" @indented") == " @indented"
    assert MH.escape_leading_at("") == ""


def test_escape_leading_at_rejects_non_string() -> None:
    # Non-string input falls through unchanged (defensive — sanitizer
    # callers always pass strings, but the helper must not raise on
    # synthetic edge cases).
    assert MH.escape_leading_at(None) is None  # type: ignore[arg-type]


# --- pattern shape sanity --------------------------------------------


def test_hashed_magic_line_pattern_compiled() -> None:
    """The exported pattern is a compiled regex with the documented shape."""
    assert isinstance(MH.HASHED_MAGIC_LINE, re.Pattern)
    m = MH.HASHED_MAGIC_LINE.match("@@deadbeef:spawn args")
    assert m is not None and m.group(1) == "deadbeef" and m.group(2) == "spawn"
