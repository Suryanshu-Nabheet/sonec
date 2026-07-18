"""Eval and training tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from sonec.app import build_agent
from sonec.core.config import load_settings
from sonec.core.types import AgentRunResult, Message, Role
from sonec.eval.harness import (
    EvalCheck,
    EvalHarness,
    EvalTask,
    mock_provider_for_task,
)
from sonec.llm.provider import MockProvider
from sonec.training.pipeline import DatasetGenerator, TrainingPipeline


@pytest.mark.asyncio
async def test_eval_harness_grades_files(tmp_path: Path) -> None:
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "hello.txt").write_text("hello sonec\n", encoding="utf-8")
    harness = EvalHarness(workspace=tmp_path)
    task = EvalTask(
        id="t1",
        name="note",
        prompt="n/a",
        checks=[
            EvalCheck(kind="file_exists", path="notes/hello.txt"),
            EvalCheck(kind="file_contains", path="notes/hello.txt", contains="hello sonec"),
        ],
    )
    graded = await harness.grade(
        task,
        AgentRunResult(run_id="r", goal="g", success=True, final_message="ok"),
    )
    assert graded.passed
    assert graded.score == 1.0


@pytest.mark.asyncio
async def test_eval_suite_mock_agent(tmp_path: Path) -> None:
    tasks_path = tmp_path / "tasks.json"
    tasks_path.write_text(
        """[
          {"id": "x", "name": "x", "prompt": "Say hi", "checks": []}
        ]""",
        encoding="utf-8",
    )
    settings = load_settings(workspace=tmp_path, provider="mock")
    harness = EvalHarness(workspace=tmp_path)
    tasks = EvalHarness.load_tasks(tasks_path)

    def factory(task: EvalTask):
        del task
        provider = MockProvider([Message(role=Role.ASSISTANT, content="hi")])
        agent, *_ = build_agent(settings=settings, provider=provider, persist_memory=False)
        return agent

    report = await harness.run_suite(tasks, factory)
    assert report.pass_rate == 1.0


@pytest.mark.asyncio
async def test_smoke_benchmark_suite(tmp_path: Path) -> None:
    suite = Path(__file__).resolve().parents[1] / "examples" / "benchmarks" / "smoke.json"
    settings = load_settings(workspace=tmp_path, provider="mock")
    harness = EvalHarness(workspace=tmp_path)
    tasks = EvalHarness.load_tasks(suite)

    def factory(task: EvalTask):
        provider = mock_provider_for_task(task)
        agent, *_ = build_agent(settings=settings, provider=provider, persist_memory=False)
        return agent

    report = await harness.run_suite(tasks, factory, name="smoke")
    assert report.total == 8
    assert report.pass_rate == 1.0
    assert report.mean_score == 1.0


def test_dataset_export(tmp_path: Path) -> None:
    gen = DatasetGenerator("test")
    gen.synthesize_smoke_examples()
    manifest = gen.manifest()
    pipeline = TrainingPipeline(tmp_path)
    jsonl = pipeline.export_jsonl(manifest)
    config = pipeline.write_config()
    assert jsonl.exists()
    assert config.exists()
    assert len(manifest.examples) == 2
