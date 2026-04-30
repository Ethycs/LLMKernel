"""tests/test_serve_subcommand.py — `python -m llm_kernel serve` argv parsing (S5.0.3d).

Unit tests for the ``serve`` subcommand argparse + token-discovery path.
Full TCP integration is exercised in the outer repo's
``tests/test_tcp_transport.py`` (subprocess kernel boot).
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from llm_kernel.serve_mode import _build_parser, _parse_bind, main


def test_parse_bind_default() -> None:
    host, port = _parse_bind("127.0.0.1:7474")
    assert host == "127.0.0.1"
    assert port == 7474


def test_parse_bind_zero_port() -> None:
    host, port = _parse_bind("127.0.0.1:0")
    assert host == "127.0.0.1"
    assert port == 0


def test_parse_bind_external() -> None:
    host, port = _parse_bind("0.0.0.0:8080")
    assert host == "0.0.0.0"
    assert port == 8080


@pytest.mark.parametrize("bad", [
    "no-colon",
    ":1234",
    "host:not-a-port",
    "host:99999",
    "host:-1",
])
def test_parse_bind_rejects_malformed(bad: str) -> None:
    with pytest.raises(ValueError):
        _parse_bind(bad)


def test_argparse_defaults() -> None:
    parser = _build_parser()
    args = parser.parse_args([])
    assert args.bind == "127.0.0.1:7474"
    assert args.auth_token_env == "LLMNB_AUTH_TOKEN"
    assert args.transport == "tcp"
    assert args.proxy == "none"


def test_argparse_token_env_override() -> None:
    parser = _build_parser()
    args = parser.parse_args(["--auth-token-env", "CI_TOKEN"])
    assert args.auth_token_env == "CI_TOKEN"


def test_argparse_does_not_accept_token_on_argv() -> None:
    """Sanity: there is NO ``--token`` flag (would leak via `ps`)."""
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--token", "secret"])


def test_main_returns_2_when_token_env_unset(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("LLMNB_AUTH_TOKEN", raising=False)
    rc = main(["--bind", "127.0.0.1:0", "--auth-token-env", "LLMNB_AUTH_TOKEN"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "LLMNB_AUTH_TOKEN" in err


def test_main_returns_2_on_malformed_bind(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("LLMNB_AUTH_TOKEN", "any-token")
    rc = main(["--bind", "garbage"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "bind" in err.lower() or "host:port" in err.lower()
