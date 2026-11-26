from __future__ import annotations

import asyncio
import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import pytest
from jupyter_client.session import Session

from llm_kernel.capture.custom_messages import MIME_RUN
from llm_kernel.jupyter_kernel import (
    LLMNBKernel,
    MIME_LLMNB,
    MIME_LLMNB_RESULT,
)
from llm_kernel.install import get_llmnb_kernel_spec, install_llmnb_kernel
from llm_kernel.notebook.notebook_format import (
    LLMNB_KERNEL_NAME,
    ipynb_to_llmnb,
    llmnb_kernelspec_metadata,
    llmnb_language_info_metadata,
    llmnb_to_ipynb,
    magic_to_llmnb,
)
from llm_kernel.text.magic_hash import emit_magic_line
from llm_kernel.wire import versions as _versions


class _RecordingSession(Session):
    def __init__(self) -> None:
        super().__init__()
        self.sent: List[Tuple[Any, str, Dict[str, Any]]] = []
        self._rec_lock = threading.Lock()

    def send(
        self,
        stream: Any,
        msg_or_type: Any,
        content: Any = None,
        parent: Any = None,
        ident: Any = None,
        buffers: Any = None,
        track: bool = False,
        header: Any = None,
        metadata: Any = None,
    ) -> Dict[str, Any]:
        msg_type = str(msg_or_type)
        record = {
            "content": content,
            "parent": parent,
            "ident": ident,
            "metadata": metadata,
        }
        with self._rec_lock:
            self.sent.append((stream, msg_type, record))
        return {"msg_type": msg_type, "content": content}

    def messages_of_type(self, msg_type: str) -> List[Dict[str, Any]]:
        with self._rec_lock:
            return [rec["content"] for _s, mt, rec in self.sent if mt == msg_type]

    def clear(self) -> None:
        with self._rec_lock:
            self.sent.clear()


class _FakeSupervisor:
    def __init__(self, writer: Any) -> None:
        self._writer = writer
        self._agents: Dict[str, Dict[str, Any]] = {}
        self.turns: List[Dict[str, Any]] = []
        self.interrupted: List[str] = []
        self.stopped: List[str] = []
        self.reverted: List[Tuple[str, str]] = []
        self.forked: List[Tuple[str, Any, str]] = []
        self.terminated: bool = False

    def set_metadata_writer(self, writer: Any) -> None:
        self._writer = writer

    def spawn(
        self,
        *,
        zone_id: str,
        agent_id: str,
        task: str,
        work_dir: Path,
        model: str | None = None,
        api_key: str | None = None,
        use_bare: bool = False,
        **_kwargs: Any,
    ) -> Any:
        self._agents[agent_id] = {
            "agent_id": agent_id,
            "zone_id": zone_id,
            "task": task,
            "model": model or "test-model",
            "provider": "claude-code",
            "status": "running",
            "work_dir": str(work_dir),
        }
        return SimpleNamespace(agent_id=agent_id, zone_id=zone_id)

    def send_user_turn(
        self,
        agent_id: str,
        text: str,
        cell_id: str | None = None,
        *,
        inject_notebook_prefix: bool = True,
    ) -> Dict[str, Any]:
        if agent_id not in self._agents:
            raise KeyError(agent_id)
        if not isinstance(text, str) or not text.strip():
            raise ValueError("empty turn text")
        turn_id = f"turn_{len(self.turns) + 1}"
        entry = {
            "agent_id": agent_id,
            "text": text,
            "cell_id": cell_id,
            "turn_id": turn_id,
            "status": "sent",
        }
        self.turns.append(entry)
        return entry

    def fork(
        self, source_agent_id: str, at_turn_id: str | None, new_agent_id: str
    ) -> Any:
        if source_agent_id not in self._agents:
            raise KeyError(source_agent_id)
        src = self._agents[source_agent_id]
        self._agents[new_agent_id] = {
            **src,
            "agent_id": new_agent_id,
            "status": "idle",
        }
        self.forked.append((source_agent_id, at_turn_id, new_agent_id))
        return SimpleNamespace(agent_id=new_agent_id)

    def revert(self, agent_id: str, target_turn_id: str) -> None:
        if agent_id not in self._agents:
            raise KeyError(agent_id)
        self.reverted.append((agent_id, target_turn_id))

    def stop(self, agent_id: str) -> None:
        if agent_id not in self._agents:
            raise KeyError(agent_id)
        self._agents[agent_id]["status"] = "idle"
        self.stopped.append(agent_id)

    def interrupt(self, agent_id: str) -> Dict[str, Any]:
        self.interrupted.append(agent_id)
        if agent_id not in self._agents:
            return {"status": "unknown"}
        return {"status": "interrupted"}

    def list_agents(self) -> List[Dict[str, Any]]:
        return [dict(v) for v in self._agents.values()]

    def respawn_from_config(
        self, config_recoverable_agents: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        return {
            a["agent_id"]: "skipped"
            for a in config_recoverable_agents
            if isinstance(a, dict) and "agent_id" in a
        }

    def terminate_all(self, grace_seconds: float = 10.0) -> None:
        self.terminated = True
        self._agents.clear()


@pytest.fixture
def jupyter_kernel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("LLMNB_OPERATOR_PIN", raising=False)
    monkeypatch.setenv("LLMKERNEL_LITELLM_ENDPOINT_URL", "http://127.0.0.1:1/v1")
    rec_session = _RecordingSession()
    kernel = LLMNBKernel(
        workdir=tmp_path,
        session=rec_session,
        iopub_socket="<test-iopub>",
    )
    kernel.execution_count = 1
    fake_sup = _FakeSupervisor(kernel.facade.metadata_writer)
    kernel.facade._supervisor = fake_sup  # noqa: SLF001
    kernel.facade.dispatcher.set_agent_supervisor(fake_sup)
    kernel.facade.operator_bridge.agent_supervisor = fake_sup
    try:
        yield kernel, rec_session, fake_sup
    finally:
        kernel.do_shutdown(False)


def _exec(
    kernel: LLMNBKernel,
    code: str,
    *,
    silent: bool = False,
    store_history: bool = True,
    user_expressions: Dict[str, str] | None = None,
    allow_stdin: bool = False,
    cell_id: str | None = None,
) -> Dict[str, Any]:
    res = kernel.do_execute(
        code,
        silent=silent,
        store_history=store_history,
        user_expressions=user_expressions,
        allow_stdin=allow_stdin,
        cell_id=cell_id,
    )
    if not silent:
        kernel.execution_count += 1
    return res


def test_kernel_info_reflects_live_versions_and_agents(
    jupyter_kernel, monkeypatch: pytest.MonkeyPatch
) -> None:
    kernel, _sess, _sup = jupyter_kernel
    info = kernel.kernel_info
    assert info["implementation"] == "llmnb"
    assert info["implementation_version"] == _versions.KERNEL_VERSION
    assert info["language_info"]["name"] == "llmnb"
    assert info["language_info"]["mimetype"] == MIME_LLMNB
    assert info["language_info"]["file_extension"] == ".llmnb"
    assert info["llmnb"]["comm_target"] == _versions.COMM_TARGET
    assert info["llmnb"]["wire_version"] == _versions.WIRE_VERSION
    assert info["llmnb"]["schema_version"] == _versions.SCHEMA_VERSION
    assert info["llmnb"]["rfc_versions"] == _versions.rfc_versions()
    assert info["agents"] == []

    _exec(kernel, '@@spawn alpha task:"design api"')
    info_after = kernel.kernel_info
    assert [a["id"] for a in info_after["agents"]] == ["alpha"]
    assert info_after["agents"][0]["provider"] == "claude-code"

    monkeypatch.setattr(_versions, "KERNEL_VERSION", "9.8.7")
    assert kernel.kernel_info["implementation_version"] == "9.8.7"
    assert kernel.kernel_info["llmnb"]["kernel_version"] == "9.8.7"


def test_execute_spawn_continue_handoff_branch_revert_stop(jupyter_kernel) -> None:
    kernel, sess, sup = jupyter_kernel

    r1 = _exec(kernel, '@@endpoint fast provider:anthropic model:claude-sonnet-4-6 api_key_env:ANTHROPIC_API_KEY', cell_id="c1")
    assert r1["status"] == "ok"

    r2 = _exec(kernel, '@@spawn alpha endpoint:fast task:"build parser"', cell_id="c2")
    assert r2["status"] == "ok"
    assert sup._agents["alpha"]["model"] == "claude-sonnet-4-6"

    opens = [
        m for m in sess.messages_of_type("display_data")
        if MIME_RUN in (m.get("data") or {})
    ]
    closes = [
        m for m in sess.messages_of_type("update_display_data")
        if MIME_RUN in (m.get("data") or {})
    ]
    assert opens and closes
    assert opens[-1]["transient"]["display_id"] == opens[-1]["data"][MIME_RUN]["spanId"]
    assert closes[-1]["data"][MIME_RUN]["status"]["code"] == "STATUS_CODE_OK"
    assert closes[-1]["data"][MIME_RUN]["endTimeUnixNano"] is not None

    r3 = _exec(kernel, "follow up without explicit @@agent header", cell_id="c3")
    assert r3["status"] == "ok"
    assert sup.turns[-1]["agent_id"] == "alpha"
    assert sup.turns[-1]["text"] == "follow up without explicit @@agent header"

    r4 = _exec(kernel, '@@spawn beta task:"write tests"', cell_id="c4")
    assert r4["status"] == "ok"

    r5 = _exec(kernel, "@@agent alpha\n@handoff beta\nfinish spec", cell_id="c5")
    assert r5["status"] == "ok"
    assert sup.turns[-1]["agent_id"] == "alpha"

    r6 = _exec(kernel, "now write the tests", cell_id="c6")
    assert r6["status"] == "ok"
    assert sup.turns[-1]["agent_id"] == "beta"

    r7 = _exec(kernel, "@branch gamma from:alpha at:turn_1", cell_id="c7")
    assert r7["status"] == "ok"
    assert sup.forked == [("alpha", "turn_1", "gamma")]

    r8 = _exec(kernel, "@revert turn_1 agent:alpha", cell_id="c8")
    assert r8["status"] == "ok"
    assert sup.reverted == [("alpha", "turn_1")]

    r9 = _exec(kernel, "@stop beta", cell_id="c9")
    assert r9["status"] == "ok"
    assert sup.stopped == ["beta"]

    r10 = _exec(kernel, '@alpha: legacy shorthand works', cell_id="c10")
    assert r10["status"] == "ok"
    assert sup.turns[-1]["agent_id"] == "alpha"
    assert sup.turns[-1]["text"] == "legacy shorthand works"


def test_execute_markdown_scratch_checkpoint_section_generators_export(
    jupyter_kernel, tmp_path: Path
) -> None:
    kernel, sess, _sup = jupyter_kernel

    assert _exec(kernel, '@@section "Overview" id:"sec_overview"', cell_id="s1")["status"] == "ok"
    assert _exec(kernel, "@@markdown\n## Notes\nSome text", cell_id="m1")["status"] == "ok"
    md_msgs = [
        m for m in sess.messages_of_type("display_data")
        if "text/markdown" in (m.get("data") or {})
    ]
    assert any("## Notes" in m["data"]["text/markdown"] for m in md_msgs)

    assert _exec(kernel, "@@scratch\n@pin\nscratch content", cell_id="sc1")["status"] == "ok"
    assert _exec(kernel, "@@checkpoint covers:[m1,sc1]\nSummary of progress", cell_id="cp1")["status"] == "ok"

    assert _exec(kernel, "@@expand\n@@scratch\ngen one\n@@break\n@@scratch\ngen two", cell_id="ex1")["status"] == "ok"

    kernel.facade.metadata_writer.set_config_template("greet", "@@scratch\nhello $who")
    assert _exec(kernel, "@@template greet who:world", cell_id="tp1")["status"] == "ok"

    assert _exec(kernel, '@@export path:"saved.llmnb"', cell_id="exp1")["status"] == "ok"
    exported = tmp_path / "saved.llmnb"
    assert exported.exists()
    exported_doc = json.loads(exported.read_text(encoding="utf-8"))
    assert exported_doc["metadata"]["kernelspec"]["name"] == "llmnb"
    assert "sec_overview" in exported_doc["metadata"]["rts"]["zone"]["sections"]

    no_overwrite = _exec(kernel, '@@export path:"saved.llmnb"', cell_id="exp2")
    assert no_overwrite["status"] == "error"
    assert no_overwrite["ename"] == "K3N"

    assert _exec(kernel, '@@export path:"saved.llmnb" overwrite:true', cell_id="exp3")["status"] == "ok"

    outside = _exec(kernel, '@@export path:"../escape.llmnb"', cell_id="exp4")
    assert outside["status"] == "error"
    assert outside["ename"] == "K3M"

    other = tmp_path / "snippet.magic"
    other.write_text("@@scratch\nimported line\n", encoding="utf-8")
    imp = _exec(kernel, "@@import snippet.magic", cell_id="imp1")
    assert imp["status"] == "ok"


def test_execute_k_class_errors_and_reserved_kinds(jupyter_kernel) -> None:
    kernel, sess, _sup = jupyter_kernel

    r_unknown = _exec(kernel, "@@nonexistent_magic foo")
    assert r_unknown["status"] == "error"
    assert r_unknown["ename"] == "K31"

    r_multi = _exec(kernel, "@@scratch\n@@markdown\ntwo kinds")
    assert r_multi["status"] == "error"
    assert r_multi["ename"] == "K30"

    r_mark = _exec(kernel, "@mark bogus_kind\nbody")
    assert r_mark["status"] == "error"
    assert r_mark["ename"] == "K34"

    r_reserved_id = _exec(kernel, '@@spawn pin task:"collide"')
    assert r_reserved_id["status"] == "error"
    assert r_reserved_id["ename"] == "K32"

    r_unbound = _exec(kernel, "orphan prose with no spawned agent")
    assert r_unbound["status"] == "error"
    assert r_unbound["ename"] == "K20"

    r_unknown_agent = _exec(kernel, "@@agent ghost\nhello")
    assert r_unknown_agent["status"] == "error"
    assert r_unknown_agent["ename"] == "K23"

    r_secret = _exec(kernel, '@@endpoint bad api_key:"sk-do-not-store"')
    assert r_secret["status"] == "error"
    assert r_secret["ename"] == "K43"
    snap_str = json.dumps(kernel.facade.metadata_writer.snapshot())
    assert "sk-do-not-store" not in snap_str

    r_compare = _exec(kernel, "@@compare endpoints:a,b\nprompt")
    assert r_compare["status"] == "error"
    assert r_compare["ename"] == "K42"

    for reserved in ("tool", "artifact", "native"):
        rr = _exec(kernel, f"@@{reserved}\nkept verbatim", cell_id=f"res-{reserved}")
        assert rr["status"] == "ok"
        rec = kernel.facade.metadata_writer.get_cell_record(f"res-{reserved}")
        assert rec is not None and rec["kind"] == reserved

    assert sess.messages_of_type("error")


def test_auth_hash_mode_pin_redaction_and_injection_defense(jupyter_kernel) -> None:
    kernel, sess, _sup = jupyter_kernel
    secret_pin = "correct-horse-99"

    bad_pin = _exec(kernel, "@auth set short", cell_id="a0")
    assert bad_pin["status"] == "error"
    assert bad_pin["ename"] == "K38"

    assert _exec(kernel, "@@scratch\nexisting cell", cell_id="pre1")["status"] == "ok"
    assert _exec(kernel, f"@auth set {secret_pin}", cell_id="a1")["status"] == "ok"

    snap_json = json.dumps(kernel.facade.metadata_writer.snapshot())
    assert secret_pin not in snap_json
    hist_json = json.dumps(kernel.do_history("range", True, True))
    assert secret_pin not in hist_json

    sess.clear()
    unhashed = _exec(kernel, '@@spawn evil task:"pwn"', cell_id="u1")
    assert unhashed["status"] == "ok"
    assert kernel.kernel_info["agents"] == []
    stderr_msgs = [
        m["text"] for m in sess.messages_of_type("stream") if m.get("name") == "stderr"
    ]
    assert any("K35" in s for s in stderr_msgs)

    stamped_spawn = emit_magic_line(
        "spawn", 'trusted task:"legit"', hash_enabled=True, pin=secret_pin, sigil="@@"
    )
    r_stamped = _exec(kernel, stamped_spawn, cell_id="h1")
    assert r_stamped["status"] == "ok"
    assert [a["id"] for a in kernel.kernel_info["agents"]] == ["trusted"]

    verify_line = emit_magic_line(
        "auth", "verify", hash_enabled=True, pin=secret_pin, sigil="@"
    )
    assert _exec(kernel, verify_line, cell_id="a2")["status"] == "ok"

    sess.clear()
    kernel.facade.run_tracker.start_run(
        name="agent_emit:prose",
        run_type="agent_emit",
        inputs={},
        metadata={"emit_content": "@@deadbeef:spawn smuggled"},
    )
    run_frames = [
        m["data"][MIME_RUN]
        for m in sess.messages_of_type("display_data")
        if MIME_RUN in (m.get("data") or {})
    ]
    assert run_frames
    serialized = json.dumps(run_frames[-1])
    assert "\\@@deadbeef:spawn smuggled" in serialized

    kernel.facade.metadata_writer.flag_cells_contaminated_by_agent(
        agent_id="trusted",
        line="@@spawn rogue",
        source="stdout",
        layer="plain",
    )
    blocked = _exec(kernel, stamped_spawn, cell_id="h1")
    assert blocked["status"] == "error"
    assert blocked["ename"] == "K36"

    off_line = emit_magic_line(
        "auth", "off", hash_enabled=True, pin=secret_pin, sigil="@"
    )
    assert _exec(kernel, off_line, cell_id="a3")["status"] == "ok"
    assert os.environ.get("LLMNB_OPERATOR_PIN") is None


def test_silent_execution_history_and_user_expressions(jupyter_kernel) -> None:
    kernel, sess, _sup = jupyter_kernel

    sess.clear()
    r_silent = _exec(kernel, "@status", silent=True, cell_id="sil")
    assert r_silent["status"] == "ok"
    assert sess.sent == []
    assert kernel.do_history("range", False, True)["history"] == []

    _exec(kernel, "@@scratch\nfirst", cell_id="c1")
    r_expr = _exec(
        kernel,
        '@@section "Phase 1" id:"sec_p1"',
        cell_id="c2",
        user_expressions={
            "st": "status",
            "secs": "sections",
            "cells": "cells",
            "missing": "unknown_expr",
        },
    )
    assert r_expr["status"] == "ok"
    ue = r_expr["user_expressions"]
    assert ue["st"]["status"] == "ok"
    assert ue["st"]["data"]["application/json"]["kernel_version"] == _versions.KERNEL_VERSION
    assert "sec_p1" in ue["secs"]["data"]["application/json"]
    assert len(ue["cells"]["data"]["application/json"]) >= 2
    assert ue["missing"]["status"] == "error"

    hist_all = kernel.do_history("range", False, True)["history"]
    assert len(hist_all) == 2
    hist_tail = kernel.do_history("tail", True, True, n=1)["history"]
    assert len(hist_tail) == 1
    assert isinstance(hist_tail[0][2], tuple)
    hist_search = kernel.do_history("search", False, True, pattern="@@scratch*")["history"]
    assert len(hist_search) == 1


def test_complete_inspect_and_is_complete(jupyter_kernel) -> None:
    kernel, _sess, _sup = jupyter_kernel
    _exec(kernel, "@@endpoint primary provider:anthropic model:claude-sonnet-4-6", cell_id="ep1")
    _exec(kernel, '@@spawn alpha task:"assist"', cell_id="sp1")
    _exec(kernel, '@@section "Design" id:"sec_design"', cell_id="sec1")
    kernel.facade.metadata_writer.set_config_template("audit", "@@scratch\ncheck")

    c_cell = kernel.do_complete("@@sp", 4)
    assert "@@spawn" in c_cell["matches"]
    assert c_cell["cursor_start"] == 0

    c_line = kernel.do_complete("@p", 2)
    assert "@pin" in c_line["matches"]

    c_agent_short = kernel.do_complete("@al", 3)
    assert "@alpha: " in c_agent_short["matches"]

    c_agent_arg = kernel.do_complete("@@agent al", 10)
    assert "alpha" in c_agent_arg["matches"]

    c_ep = kernel.do_complete("@@spawn beta endpoint:pr", 24)
    assert "primary" in c_ep["matches"]

    c_fmt = kernel.do_complete('@@export path:"a" format:ip', 27)
    assert c_fmt["matches"] == ["ipynb"]

    c_auth = kernel.do_complete("@auth v", 7)
    assert c_auth["matches"] == ["verify"]

    c_mark = kernel.do_complete("@mark sc", 8)
    assert c_mark["matches"] == ["scratch"]

    c_tpl = kernel.do_complete("@@template au", 13)
    assert c_tpl["matches"] == ["audit"]

    ins_magic = kernel.do_inspect("@@spawn alpha", 3, detail_level=1)
    assert ins_magic["found"] is True
    assert "K32" in ins_magic["data"]["text/plain"]
    assert "text/markdown" in ins_magic["data"]

    ins_agent = kernel.do_inspect("@@agent alpha", 10, detail_level=1)
    assert ins_agent["found"] is True
    assert "alpha" in ins_agent["data"]["text/plain"]

    ins_ep = kernel.do_inspect("primary", 3, detail_level=0)
    assert ins_ep["found"] is True
    assert "claude-sonnet-4-6" in ins_ep["data"]["text/plain"]

    ins_sec = kernel.do_inspect("sec_design", 4, detail_level=0)
    assert ins_sec["found"] is True
    assert "Design" in ins_sec["data"]["text/plain"]

    ins_cell = kernel.do_inspect("sp1", 2, detail_level=0)
    assert ins_cell["found"] is True
    assert "spawn" in ins_cell["data"]["text/plain"]

    ins_miss = kernel.do_inspect("totally_unknown_symbol", 5)
    assert ins_miss["found"] is False

    assert kernel.do_is_complete("")["status"] == "incomplete"
    assert kernel.do_is_complete('@@spawn alpha task:"unclosed')["status"] == "incomplete"
    assert kernel.do_is_complete("@@markdown\n```python\nx = 1")["status"] == "incomplete"
    assert kernel.do_is_complete("@@agent alpha")["status"] == "incomplete"
    assert kernel.do_is_complete("@@agent alpha\n\n")["status"] == "complete"
    assert kernel.do_is_complete("@@agent alpha\ndo work")["status"] == "complete"
    assert kernel.do_is_complete('@@spawn alpha task:"done"')["status"] == "complete"
    assert kernel.do_is_complete("@@bogus_magic")["status"] == "invalid"


def test_multiclient_comm_protocol_hydrate_drift_and_zone_mutate(
    jupyter_kernel,
) -> None:
    kernel, sess, _sup = jupyter_kernel
    cm = kernel.comm_manager

    cm.comm_open(
        None,
        None,
        {"content": {"comm_id": "bad-1", "target_name": "llmnb.rts.v999", "data": {}}},
    )
    assert "bad-1" not in cm.comms
    assert any(m["comm_id"] == "bad-1" for m in sess.messages_of_type("comm_close"))

    cm.comm_open(
        None,
        None,
        {"content": {"comm_id": "c-one", "target_name": _versions.COMM_TARGET, "data": {}}},
    )
    cm.comm_open(
        None,
        None,
        {"content": {"comm_id": "c-two", "target_name": _versions.COMM_TARGET, "data": {}}},
    )
    assert set(cm.comms.keys()) == {"c-one", "c-two"}

    sess.clear()
    hydrate_snapshot = {
        "schema_version": _versions.SCHEMA_VERSION,
        "session_id": "11111111-1111-4111-8111-111111111111",
        "created_at": "2026-01-01T00:00:00.000Z",
        "config": {
            "version": 1,
            "recoverable": {"kernel": {}, "agents": [], "mcp_servers": []},
            "volatile": {"kernel": {"rfc_006_version": "1.0.0"}, "agents": [], "mcp_servers": []},
        },
        "layout": {"version": 1, "tree": {"id": "root", "type": "workspace", "children": []}},
        "agents": {"version": 1, "nodes": [], "edges": []},
        "event_log": {"version": 1, "runs": []},
        "blobs": {},
        "drift_log": [],
    }
    cm.comm_msg(
        None,
        None,
        {
            "content": {
                "comm_id": "c-one",
                "data": {
                    "type": "notebook.metadata",
                    "payload": {"mode": "hydrate", "snapshot": hydrate_snapshot},
                },
            }
        },
    )
    comm_msgs = sess.messages_of_type("comm_msg")
    recipients = {
        m["comm_id"]
        for m in comm_msgs
        if (m.get("data") or {}).get("type") == "notebook.metadata"
        and ((m["data"].get("payload") or {}).get("trigger") == "hydrate_complete")
    }
    assert recipients == {"c-one", "c-two"}
    drift_log = kernel.facade.metadata_writer.snapshot()["drift_log"]
    assert any("rfc_006_version" in d["field_path"] for d in drift_log)

    cm.comm_msg(
        None,
        None,
        {
            "content": {
                "comm_id": "c-two",
                "data": {
                    "type": "operator.action",
                    "payload": {
                        "action_type": "zone_mutate",
                        "intent_kind": "apply_overlay_commit",
                        "intent_id": "comm-sec-1",
                        "parameters": {
                            "operations": [
                                {
                                    "kind": "create_section",
                                    "section_id": "sec_from_comm",
                                    "title": "From Comm",
                                }
                            ],
                            "message": "comm section",
                        },
                    },
                },
            }
        },
    )
    sections = kernel.facade.metadata_writer.snapshot()["zone"]["sections"]
    assert "sec_from_comm" in sections

    cm.comm_close(None, None, {"content": {"comm_id": "c-one", "data": {}}})
    assert set(cm.comms.keys()) == {"c-two"}
    sess.clear()
    cm.comm_msg(
        None,
        None,
        {
            "content": {
                "comm_id": "c-two",
                "data": {
                    "type": "agent_graph.query",
                    "correlation_id": "22222222-2222-4222-8222-222222222222",
                    "payload": {"query_type": "list_agents"},
                },
            }
        },
    )
    replies = [
        m for m in sess.messages_of_type("comm_msg")
        if (m.get("data") or {}).get("type") == "agent_graph.response"
    ]
    assert len(replies) == 1
    assert replies[0]["comm_id"] == "c-two"
    assert replies[0]["data"]["correlation_id"] == "22222222-2222-4222-8222-222222222222"


def test_stdin_approval_and_control_interrupt(
    jupyter_kernel, monkeypatch: pytest.MonkeyPatch
) -> None:
    kernel, sess, sup = jupyter_kernel
    _exec(kernel, '@@spawn alpha task:"work"', cell_id="s1")

    kernel._allow_stdin = True
    prompts: List[str] = []
    monkeypatch.setattr(
        kernel, "raw_input", lambda prompt="": (prompts.append(prompt) or "approve")
    )
    res = asyncio.run(
        kernel.facade.operator_bridge._handle_request_approval({"action": "delete tmp"})
    )
    assert res == {"decision": "approve"}
    assert prompts and "delete tmp" in prompts[0]

    monkeypatch.setattr(kernel, "raw_input", lambda prompt="": "42")
    ask_res = asyncio.run(
        kernel.facade.operator_bridge._handle_ask({"question": "What is the answer?"})
    )
    assert ask_res == {"answer": "42"}
    kernel._allow_stdin = False

    sess.clear()
    asyncio.run(kernel.interrupt_request("<ctrl>", [b"id"], {"header": {"msg_id": "i1"}}))
    assert "alpha" in sup.interrupted
    ireplies = sess.messages_of_type("interrupt_reply")
    assert ireplies and ireplies[0]["status"] == "ok"


def test_notebook_hydrate_persist_and_no_leaked_threads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LLMKERNEL_LITELLM_ENDPOINT_URL", "http://127.0.0.1:1/v1")
    nb_path = tmp_path / "session.llmnb"
    seed = magic_to_llmnb("@@scratch\nseeded cell\n")
    seed["metadata"]["custom_user_key"] = {"keep": True}
    nb_path.write_text(json.dumps(seed, indent=2), encoding="utf-8")

    before = {t.ident for t in threading.enumerate()}
    rec = _RecordingSession()
    kernel = LLMNBKernel(
        workdir=tmp_path,
        notebook_path=nb_path,
        autosave_notebook=True,
        session=rec,
        iopub_socket="<iopub>",
    )
    kernel.execution_count = 1
    _exec(kernel, '@@section "Added" id:"sec_added"', cell_id="new-sec")
    shutdown_reply = kernel.do_shutdown(False)
    assert shutdown_reply == {"status": "ok", "restart": False}

    leaked = [
        t.name
        for t in threading.enumerate()
        if t.ident not in before
        and t is not threading.main_thread()
        and not t.name.startswith(("pytest", "pydevd", "asyncio_", "ThreadPoolExecutor"))
    ]
    assert leaked == []

    persisted = json.loads(nb_path.read_text(encoding="utf-8"))
    assert persisted["metadata"]["custom_user_key"] == {"keep": True}
    assert persisted["metadata"]["kernelspec"] == llmnb_kernelspec_metadata()
    assert persisted["metadata"]["language_info"] == llmnb_language_info_metadata()
    assert "sec_added" in persisted["metadata"]["rts"]["zone"]["sections"]
    assert "cell-0" in persisted["metadata"]["rts"]["cells"]


def test_kernelspec_installer_and_l2_format_metadata(tmp_path: Path) -> None:
    spec = get_llmnb_kernel_spec(python_executable="/usr/bin/python3")
    assert spec["language"] == "llmnb"
    assert spec["display_name"] == "LLMNB"
    assert spec["interrupt_mode"] == "message"
    assert "{connection_file}" in spec["argv"]
    assert spec["metadata"]["llmnb"]["comm_target"] == _versions.COMM_TARGET

    prefix = tmp_path / "prefix"
    dest = install_llmnb_kernel(prefix=str(prefix), kernel_name=LLMNB_KERNEL_NAME)
    kjson = Path(dest) / "kernel.json"
    assert kjson.exists()
    on_disk = json.loads(kjson.read_text(encoding="utf-8"))
    assert on_disk["language"] == "llmnb"

    llmnb = magic_to_llmnb('@@section "Title"\n@@break\n@@scratch\nbody\n')
    assert llmnb["metadata"]["kernelspec"]["name"] == "llmnb"
    assert llmnb["metadata"]["language_info"]["name"] == "llmnb"
    ipynb = llmnb_to_ipynb(llmnb)
    assert ipynb["metadata"]["kernelspec"]["name"] == "llmnb"
    round_tripped = ipynb_to_llmnb(ipynb)
    cells = list(round_tripped["metadata"]["rts"]["cells"].values())
    assert [c["kind"] for c in cells] == ["section", "scratch"]
    assert not cells[1]["text"].startswith("@@scratch\n@@scratch")
