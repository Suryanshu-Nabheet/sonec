"""Eval and training tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from sonec.app import build_runtime
from sonec.core.config import load_settings
from sonec.core.types import AgentRunResult, Message, Role
from sonec.eval.harness import (
    EvalCheck,
    EvalHarness,
    EvalTask,
    mock_provider_for_task,
)
from sonec.eval.sonecbench import build_sonecbench_tasks
from sonec.harness.versioning import CORE_TOOL_NAMES, HARNESS_VERSION
from sonec.llm.provider import MockProvider
from sonec.training.pipeline import DatasetGenerator, TrainingPipeline
from sonec.training.rollouts import run_rollouts_sync


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
    agent = AgentRunResult(
        run_id="r", goal="g", success=False, final_message="ok", completed=True
    )
    graded = await harness.grade(task, agent)
    assert graded.passed
    assert agent.success is True
    assert agent.evidence_success is True


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
        agent, *_ = build_runtime(
            settings=settings, provider=provider, persist_memory=False, log_dir=tmp_path / "t"
        )
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
        agent, *_ = build_runtime(
            settings=settings, provider=provider, persist_memory=False, log_dir=tmp_path / "t"
        )
        return agent

    report = await harness.run_suite(tasks, factory, name="smoke")
    assert report.total == 8
    assert report.pass_rate == 1.0


def test_sonecbench_size() -> None:
    tasks = build_sonecbench_tasks()
    assert len(tasks) >= 50
    ids = {t.id for t in tasks}
    assert len(ids) == len(tasks)


def test_rollout_factory(tmp_path: Path) -> None:
    tasks = build_sonecbench_tasks()[:2]
    records = run_rollouts_sync(tasks, tmp_path / "rollouts", group_size=2)
    assert len(records) == 4
    assert all(r.harness_version == HARNESS_VERSION for r in records)
    assert all(r.tool_schema_hash for r in records)
    assert (tmp_path / "rollouts" / "rollouts.jsonl").exists()


def test_core_tool_freeze() -> None:
    assert "fs_read" in CORE_TOOL_NAMES
    assert "terminal_run" in CORE_TOOL_NAMES
    assert "memory_note" not in CORE_TOOL_NAMES


def test_dataset_export(tmp_path: Path) -> None:
    gen = DatasetGenerator("test")
    gen.synthesize_smoke_examples()
    manifest = gen.manifest()
    pipeline = TrainingPipeline(tmp_path)
    jsonl = pipeline.export_jsonl(manifest)
    config = pipeline.write_config()
    assert jsonl.exists()
    assert config.exists()
