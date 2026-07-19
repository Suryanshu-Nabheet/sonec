"""CapabilityBench + light GRPO guards."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sonec.eval.capabilitybench import (
    CATEGORIES,
    build_capabilitybench_tasks,
    write_capabilitybench,
)
from sonec.training.grpo_lite import run_grpo_lite


def test_capabilitybench_shape() -> None:
    tasks = build_capabilitybench_tasks()
    assert len(tasks) == 200
    assert len(CATEGORIES) == 10
    by_diff = {d: 0 for d in ("easy", "medium", "hard")}
    by_cat: dict[str, int] = {}
    ids: set[str] = set()
    for t in tasks:
        assert t.id not in ids
        ids.add(t.id)
        assert t.id.startswith("cap-")
        by_diff[t.difficulty] += 1
        cat = t.id.split("-")[1]
        by_cat[cat] = by_cat.get(cat, 0) + 1
        assert t.tags, t.id
        assert t.difficulty in by_diff
    assert by_diff == {"easy": 70, "medium": 70, "hard": 60}
    assert all(v == 20 for v in by_cat.values())
    assert set(by_cat) == {c for c, _ in CATEGORIES}


def test_capabilitybench_write(tmp_path: Path) -> None:
    path = write_capabilitybench(tmp_path / "capabilitybench_v1.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["sealed"] is True
    assert data["task_count"] == 200
    assert len(data["tasks"]) == 200
    assert data["by_difficulty"]["easy"] == 70


def test_grpo_refuses_heavy_live(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="too heavy"):
        run_grpo_lite(root=tmp_path, group_size=8, train_n=24, live=True, sft_iters=10)
    with pytest.raises(ValueError, match="too heavy"):
        run_grpo_lite(root=tmp_path, group_size=2, train_n=32, live=True, sft_iters=10)


def test_capability_ids_never_collide_trainbench() -> None:
    from sonec.eval.trainbench import build_trainbench_tasks

    cap = {t.id for t in build_capabilitybench_tasks()}
    train = {t.id for t in build_trainbench_tasks(n=120)}
    assert not (cap & train)
