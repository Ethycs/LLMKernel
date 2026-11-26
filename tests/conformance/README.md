# Conformance suite — merged kernel features vs. their documentation

Self-contained (own `conftest.py` + `_support.py`; nothing imported from
sibling test modules). Every test drives a documented surface: the `Kernel`
facade, the process entry points, the RFC-006 wire, `llm_kernel.wire.versions`.

| File | Feature | Documentation checked |
|---|---|---|
| `test_facade_contract.py` | `Kernel` facade lifecycle, delegators, event log, transports, failure, threadsafety | `docs/04 - Reference/kernel/architecture.md`, `embedding.md`, `docs/06 - Roadmaps/PLAN-kernel-facade.md` |
| `test_entrypoints_consume_facade.py` | pty-mode (`main`, `boot_kernel`), `serve`, the uvicorn app each run exactly one facade kernel | PLAN-kernel-facade §"Entry-point refactor", commits 6–8; RFC-008 §4; BSP-004 |
| `test_drift_every_start_path.py` | drift on hydrate in every start path (facade, legacy hooks, pty in-process + subprocess, serve helper + subprocess, app) against one live source | RFC-005 §drift_log / §"Resume-time RFC version check", `atoms/contracts/drift-detector.md`, RFC-006 §8/§9, `llm_kernel/wire/versions.py` |
| `test_version_single_source.py` | ready frame, handshake, `/health`, banners, schema stamp, exported schemas, `__version__`, `setup.py` all read `wire/versions.py` | `llm_kernel/wire/versions.py`, `atoms/discipline/wire-as-public-api.md`, RFC-008 §4 |
| `test_magic_defense_enforced.py` | contamination detector + wire sanitizer bound in every start path, pin-aware parsing everywhere, pin isolation, `emit_magic_cell` privilege, file-sourced trust quarantine | `atoms/discipline/magic-injection-defense.md`, PLAN-S5.0.1, PLAN-S5.0.4, `atoms/discipline/certified-magic-emitter.md` |

The notebook-repo (driver) side lives in `tests/conformance/` at the repo root.

Run (from the repo root):

```bash
pixi run -e kernel python -m pytest vendor/LLMKernel/tests/conformance -q
pixi run -e kernel python -m pytest tests/conformance -q
```
