"""Integrity tests for sealed exclusion, Cap200, graders, and tool parsing."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from sonec.core.coerce import coerce_bool, coerce_int
from sonec.core.types import AgentRunResult
from sonec.eval.capabilitybench import build_capabilitybench_tasks
from sonec.eval.harness import EvalCheck, EvalHarness, EvalTask
from sonec.eval.sealed import collect_sealed_task_ids, is_harness_path
from sonec.llm.tool_parse import parse_qwen_tool_calls
from sonec.training.export import load_successful_rollouts


def test_collect_sealed_includes_cap_and_smoke() -> None:
    sealed = collect_sealed_task_ids(Path.cwd())
    assert any(x.startswith("cap-") for x in sealed)
    assert "hard-restraint" in sealed
    assert "hard-fix-clamp" in sealed
    assert len(sealed) >= 200


def test_is_harness_path() -> None:
    assert is_harness_path(".trajectories/run.jsonl")
    assert is_harness_path(".sonec/memory/notes.jsonl")
    assert not is_harness_path("BAIT.txt")
    assert not is_harness_path("src/main.py")


def test_only_files_ignores_trajectories(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "BAIT.txt").write_text("ok\n", encoding="utf-8")
    traj = ws / ".trajectories"
    traj.mkdir()
    (traj / "run.jsonl").write_text("{}\n", encoding="utf-8")
    harness = EvalHarness(workspace=ws, enable_default_commands=False)

    async def _run() -> None:
        ok = await harness._check(EvalCheck(kind="only_files", contains="BAIT.txt"))
        assert ok

    asyncio.run(_run())


def test_restraint_empty_checks_fail_on_extra_files(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "junk.txt").write_text("x\n", encoding="utf-8")
    harness = EvalHarness(workspace=ws, enable_default_commands=False)
    task = EvalTask(
        id="r",
        name="r",
        prompt="question only",
        tags=["restraint"],
        checks=[],
    )
    result = AgentRunResult(
        run_id="r",
        goal="q",
        success=False,
        final_message="an answer",
        completed=True,
    )

    async def _run() -> None:
        graded = await harness.grade(task, result)
        assert graded.passed is False

    asyncio.run(_run())


def test_command_requires_exit_zero_by_default(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()

    async def runner(cmd: str) -> dict:
        return {"exit_code": 1}

    harness = EvalHarness(workspace=ws, run_command=runner)

    async def _run() -> None:
        ok = await harness._check(EvalCheck(kind="command", path="false"))
        assert ok is False
        ok2 = await harness._check(
            EvalCheck(kind="command", path="false", command_exit_zero=False)
        )
        assert ok2 is True

    asyncio.run(_run())


def test_sealed_rollouts_excluded(tmp_path: Path) -> None:
    traj_keep = tmp_path / "keep.jsonl"
    traj_drop = tmp_path / "drop.jsonl"
    for p in (traj_keep, traj_drop):
        p.write_text(
            json.dumps(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "c1", "name": "fs_write", "arguments": {}}],
                }
            )
            + "\n"
            + json.dumps({"type": "message", "role": "user", "content": "hi"})
            + "\n"
            + json.dumps({"type": "message", "role": "assistant", "content": "done"})
            + "\n",
            encoding="utf-8",
        )
    path = tmp_path / "rollouts.jsonl"
    rows = [
        {
            "task_id": "train-a-01",
            "rollout_index": 0,
            "passed": True,
            "reward": 1.0,
            "trajectory_path": str(traj_keep),
        },
        {
            "task_id": "cap-edit-01",
            "rollout_index": 0,
            "passed": True,
            "reward": 1.0,
            "trajectory_path": str(traj_drop),
        },
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    kept = load_successful_rollouts(path, sealed_ids={"cap-edit-01"})
    assert len(kept) == 1
    assert kept[0]["task_id"] == "train-a-01"


def test_capabilitybench_committed_matches_generator() -> None:
    committed = Path("examples/benchmarks/capabilitybench_v1.json")
    assert committed.exists()
    data = json.loads(committed.read_text(encoding="utf-8"))
    generated = build_capabilitybench_tasks()
    assert data["sealed"] is True
    assert data["task_count"] == 200
    assert len(data["tasks"]) == len(generated)
    gen_ids = [t.id for t in generated]
    file_ids = [t["id"] for t in data["tasks"]]
    assert gen_ids == file_ids
    # Spot-check verify graders gained executable evidence.
    verify = next(t for t in generated if t.id == "cap-verify-01")
    kinds = {c.kind for c in verify.checks}
    assert "python_exec" in kinds or "command" in kinds


def test_coerce_bool_string_false() -> None:
    assert coerce_bool("false") is False
    assert coerce_bool("true") is True
    assert coerce_bool(False) is False
    assert coerce_int("12") == 12


def test_parse_qwen_coerces_recursive() -> None:
    content = """
<tool_call>
<function=fs_list>
<parameter=path>
.
</parameter>
<parameter=recursive>
false
</parameter>
</function>
</tool_call>
"""
    calls = parse_qwen_tool_calls(content)
    assert len(calls) == 1
    assert calls[0].name == "fs_list"
    assert calls[0].arguments["recursive"] is False


def test_smoke_restraint_has_only_files() -> None:
    data = json.loads(Path("examples/benchmarks/ab_agent_2b_hard.json").read_text(encoding="utf-8"))
    task = next(t for t in data["tasks"] if t["id"] == "hard-restraint")
    assert "restraint" in task["tags"]
    assert any(c["kind"] == "only_files" for c in task["checks"])
