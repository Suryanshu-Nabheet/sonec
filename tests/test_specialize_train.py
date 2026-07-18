"""Specialize training + trainbench tests."""

from __future__ import annotations

from pathlib import Path

from sonec.eval.trainbench import build_trainbench_tasks
from sonec.training.specialize import (
    assemble_sft_corpus,
    generate_gold_agent_examples,
    prepare_rollout_fuel,
)
from sonec.training.pipeline import DatasetGenerator


def test_trainbench_size() -> None:
    tasks = build_trainbench_tasks(n=60)
    assert len(tasks) == 60
    assert all(t.id.startswith("train-") for t in tasks)


def test_gold_curriculum() -> None:
    gen = DatasetGenerator("t")
    n = generate_gold_agent_examples(gen, n=24)
    assert n == 24
    assert len(gen.manifest().examples) == 24


def test_prepare_fuel_and_corpus(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "examples" / "benchmarks").mkdir(parents=True)
    fuel = prepare_rollout_fuel(tmp_path / "fuel", group_size=1, train_n=6)
    assert fuel.exists()
    paths = assemble_sft_corpus(rollouts_jsonl=fuel, out_dir=tmp_path / "corpus", gold_n=12)
    assert paths["mlx_train"].exists()
    stats = (tmp_path / "corpus" / "corpus_stats.json").read_text(encoding="utf-8")
    assert "examples" in stats
