"""Leaderboard ranking + arms catalog tests."""

from __future__ import annotations

import json
from pathlib import Path

from sonec.eval.compare import ArmSpec
from sonec.eval.harness import BenchmarkReport, EvalResult
from sonec.eval.leaderboard import load_arms, rank_arms
from sonec.training.grpo_lite import _group_advantages


def test_group_advantages() -> None:
    assert _group_advantages([1.0, 0.0, 1.0, 0.0]) == [0.5, -0.5, 0.5, -0.5]


def test_load_arms_catalog() -> None:
    path = Path("configs/leaderboard/arms_2b.json")
    arms = load_arms(path)
    assert len(arms) == 4
    assert any(a.kind == "lora" for a in arms)
    assert all(a.base_url.endswith("/v1") or "/v1" in a.base_url for a in arms)
    # Strict 2B — no 1B / 1.5B / 3B+ peers in catalog.
    for a in arms:
        if a.kind == "lora":
            continue
        assert ":2b" in a.model.lower() or a.model.endswith("2b")
        assert "1.5" not in a.model
        assert ":1b" not in a.model
        assert ":3b" not in a.model


def test_hard_suite_exists() -> None:
    path = Path("examples/benchmarks/ab_agent_2b_hard.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert len(data["tasks"]) >= 8
    ids = {t["id"] for t in data["tasks"]}
    assert "hard-fix-clamp" in ids
    assert "hard-restraint" in ids


def test_rank_arms_prefers_lora_on_tie() -> None:
    arms = [
        ArmSpec("base", "http://127.0.0.1:11434/v1", "qwen", kind="base"),
        ArmSpec("sonec", "http://127.0.0.1:8080/v1", "mlx", kind="lora"),
    ]

    def _rep(name: str) -> BenchmarkReport:
        return BenchmarkReport(
            name=name,
            results=[
                EvalResult(
                    task_id="t",
                    passed=True,
                    score=1.0,
                    details=[],
                    duration_s=1.0,
                )
            ],
            pass_rate=1.0,
            mean_duration_s=1.0,
            mean_score=1.0,
            passed=1,
            total=1,
        )

    summary = rank_arms({"base": _rep("base"), "sonec": _rep("sonec")}, arms)
    assert summary.winner == "sonec"
    assert summary.arms[0]["name"] == "sonec"


def test_arms_catalog_json_valid() -> None:
    data = json.loads(Path("configs/leaderboard/arms_2b.json").read_text(encoding="utf-8"))
    assert "arms" in data and "catalog" in data
    names = {a["name"] for a in data["arms"]}
    assert "sonec" in names


def test_load_cached_arm(tmp_path: Path) -> None:
    from sonec.eval.leaderboard import _load_cached_arm

    report = BenchmarkReport(
        name="x",
        results=[
            EvalResult(task_id="t", passed=True, score=1.0, details=[], duration_s=0.5)
        ],
        pass_rate=1.0,
        mean_duration_s=0.5,
        mean_score=1.0,
        passed=1,
        total=1,
    )
    path = tmp_path / "arm_x.json"
    report.save(path)
    loaded = _load_cached_arm(path)
    assert loaded is not None
    assert loaded.pass_rate == 1.0
    assert _load_cached_arm(tmp_path / "missing.json") is None
