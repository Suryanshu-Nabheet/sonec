"""Agent reward shaping tests."""

from __future__ import annotations

from pathlib import Path

from sonec.eval.harness import EvalCheck, EvalTask
from sonec.training.rewards import compute_agent_reward


def _write_traj(path: Path, tool_names: list[str]) -> None:
    lines = []
    for i, name in enumerate(tool_names):
        lines.append(
            '{"type":"message","role":"assistant","content":"","tool_calls":'
            f'[{{"id":"c{i}","name":"{name}","arguments":{{}}}}]}}'
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_fail_is_zero(tmp_path: Path) -> None:
    traj = tmp_path / "t.jsonl"
    _write_traj(traj, ["fs_list", "fs_list", "fs_list"])
    task = EvalTask(
        id="t",
        name="t",
        prompt="Create notes/a.txt",
        checks=[EvalCheck(kind="file_exists", path="notes/a.txt")],
    )
    reward, meta = compute_agent_reward(
        passed=False, trajectory_path=str(traj), task=task, details=["fail file_exists"]
    )
    assert reward == 0.0
    assert meta["failure"] in {"wrong_path_or_content", "no_write", "explore_forever"}


def test_pass_penalizes_explore(tmp_path: Path) -> None:
    traj = tmp_path / "t.jsonl"
    _write_traj(traj, ["fs_list", "fs_list", "fs_list", "fs_list", "fs_list", "fs_write"])
    task = EvalTask(
        id="t",
        name="t",
        prompt="Create a.txt",
        checks=[EvalCheck(kind="file_exists", path="a.txt")],
    )
    reward, meta = compute_agent_reward(
        passed=True, trajectory_path=str(traj), task=task, details=[]
    )
    assert reward < 1.0
    assert "explore_before_write" in meta["penalties"]


def test_verify_tag_alone_does_not_require_terminal(tmp_path: Path) -> None:
    traj = tmp_path / "t.jsonl"
    _write_traj(traj, ["fs_write"])
    task = EvalTask(
        id="t",
        name="t",
        prompt="Create a.txt",
        tags=["verify"],
        checks=[EvalCheck(kind="file_exists", path="a.txt")],
    )
    reward, meta = compute_agent_reward(
        passed=True, trajectory_path=str(traj), task=task, details=[]
    )
    assert reward == 1.0
    assert "unfinished_verify" not in meta["penalties"]


def test_command_check_expects_verify(tmp_path: Path) -> None:
    traj = tmp_path / "t.jsonl"
    _write_traj(traj, ["fs_write"])
    task = EvalTask(
        id="t",
        name="t",
        prompt="Create and run",
        checks=[
            EvalCheck(kind="file_exists", path="a.txt"),
            EvalCheck(kind="command", path="python a.txt"),
        ],
    )
    reward, meta = compute_agent_reward(
        passed=True, trajectory_path=str(traj), task=task, details=[]
    )
    assert reward < 1.0
    assert "unfinished_verify" in meta["penalties"]


def test_clean_write_is_one(tmp_path: Path) -> None:
    traj = tmp_path / "t.jsonl"
    _write_traj(traj, ["fs_write"])
    task = EvalTask(
        id="t",
        name="t",
        prompt="Create a.txt",
        checks=[EvalCheck(kind="file_exists", path="a.txt")],
    )
    reward, meta = compute_agent_reward(
        passed=True, trajectory_path=str(traj), task=task, details=[]
    )
    assert reward == 1.0
    assert meta["penalties"] == []
