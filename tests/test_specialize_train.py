"""Specialize training + trainbench tests."""

from __future__ import annotations

from pathlib import Path

from sonec.eval.trainbench import build_trainbench_tasks
from sonec.training.pipeline import DatasetGenerator
from sonec.training.specialize import (
    assemble_sft_corpus,
    generate_gold_agent_examples,
    prepare_rollout_fuel,
)


def test_trainbench_size() -> None:
    tasks = build_trainbench_tasks(n=60)
    assert len(tasks) == 60
    assert all(t.id.startswith("train-") for t in tasks)


def test_trainbench_pyutil_pkggreet_kinds() -> None:
    tasks = build_trainbench_tasks(n=100)
    pyutil = [t for t in tasks if t.id.startswith("train-pyutil-")]
    pkggreet = [t for t in tasks if t.id.startswith("train-pkggreet-")]
    cli = [t for t in tasks if t.id.startswith("train-cli-")]
    clamp = [t for t in tasks if t.id.startswith("train-clamp-")]
    assert pyutil, "expected train-pyutil-* curriculum"
    assert pkggreet, "expected train-pkggreet-* curriculum"
    assert cli, "expected train-cli-* curriculum"
    assert clamp, "expected train-clamp-* curriculum"
    assert all(any(c.path == "src/util.py" for c in t.checks) for t in pyutil)
    assert all(any(c.path == "pkg/core.py" for c in t.checks) for t in pkggreet)
    # Must never collide with sealed A/B ids.
    sealed = {"py-util-main", "pkg-greet", "hard-fix-clamp", "hard-py-cli"}
    assert all(t.id not in sealed for t in tasks)


def test_gold_curriculum_openai_tool_calls() -> None:
    gen = DatasetGenerator("t")
    n = generate_gold_agent_examples(gen, n=24)
    assert n == 24
    examples = gen.manifest().examples
    assert len(examples) == 24
    toolish = [e for e in examples if any(s.tool_calls for s in e.trajectory)]
    assert toolish, "gold must include structured tool_calls"
    for e in toolish:
        for step in e.trajectory:
            assert not step.content.startswith("Calling tool")
            assert "<tool_call>" not in step.content


def test_identity_examples() -> None:
    from sonec.training.specialize import generate_identity_examples

    gen = DatasetGenerator("id")
    assert generate_identity_examples(gen, n=4) == 4
    for e in gen.manifest().examples:
        text = " ".join(s.content for s in e.trajectory)
        assert "Suryanshu Nabheet" in text
        assert "sonec" in text.lower()
        assert "Cursor" not in text or "not Cursor" in text
        # Must not claim to be the base weight lineage by name in the answer.
        assert "I am Qwen" not in text
        assert "I am Cursor" not in text


def test_gold_zero_skips() -> None:
    gen = DatasetGenerator("t")
    assert generate_gold_agent_examples(gen, n=0) == 0
    assert gen.manifest().examples == []


def test_prepare_fuel_and_corpus_mock(tmp_path: Path, monkeypatch) -> None:
    """Unit test uses mock fuel; production defaults to live."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "examples" / "benchmarks").mkdir(parents=True)
    fuel = prepare_rollout_fuel(
        tmp_path / "fuel", group_size=1, train_n=6, live=False
    )
    assert fuel.exists()
    # Mock fuel may yield few/no tool_calls — seed with gold for assemble
    paths = assemble_sft_corpus(rollouts_jsonl=fuel, out_dir=tmp_path / "corpus", gold_n=12)
    assert paths["mlx_train"].exists()
    text = paths["mlx_train"].read_text(encoding="utf-8")
    assert "tool_calls" in text
    assert "Calling tool" not in text
