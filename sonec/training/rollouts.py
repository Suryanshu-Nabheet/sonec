"""Rollout factory — graded trajectories at scale (Phase 2)."""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sonec.app import build_runtime
from sonec.core.config import load_settings
from sonec.eval.harness import EvalHarness, EvalTask, mock_provider_for_task
from sonec.harness.versioning import HARNESS_VERSION
from sonec.llm.provider import LLMProvider, MockProvider
from sonec.training.rewards import compute_agent_reward


@dataclass
class RolloutRecord:
    task_id: str
    prompt: str
    harness_version: str
    tool_schema_hash: str
    model_id: str
    reward: float
    passed: bool
    trajectory_path: str
    failure_class: str
    group_id: str
    rollout_index: int
    duration_s: float
    details: list[str] = field(default_factory=list)
    reward_meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "prompt": self.prompt,
            "harness_version": self.harness_version,
            "tool_schema_hash": self.tool_schema_hash,
            "model_id": self.model_id,
            "reward": self.reward,
            "passed": self.passed,
            "trajectory_path": self.trajectory_path,
            "failure_class": self.failure_class,
            "group_id": self.group_id,
            "rollout_index": self.rollout_index,
            "duration_s": self.duration_s,
            "details": self.details,
            "reward_meta": self.reward_meta,
        }


def classify_failure(details: list[str], *, completed: bool, timed_out: bool = False) -> str:
    if timed_out:
        return "timeout"
    if not details:
        return "unknown"
    joined = " ".join(details).lower()
    if "pass " in joined and "fail " not in joined:
        return "none"
    if not completed:
        return "tool_loop"
    if "file_exists" in joined or "file_contains" in joined:
        return "wrong_localization"
    if "command" in joined:
        return "skipped_verify"
    return "over_edit_or_incomplete"


class RolloutFactory:
    """Snapshot → isolated workspace → harness run → grade → tear down.

    Uses temp directories for local isolation. Docker/VM hooks are optional when
    `docker` is available; Phase 2 exit gate requires graded JSONL regardless.
    """

    def __init__(
        self,
        *,
        output_dir: Path,
        group_size: int = 8,
        use_mock: bool = True,
        provider: LLMProvider | None = None,
        provider_name: str = "local",
        model: str | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.group_size = group_size
        self.use_mock = use_mock
        self.provider = provider
        self.provider_name = provider_name
        self.model = model
        self.records_path = self.output_dir / "rollouts.jsonl"
        # Fresh file each factory run so live/oracle fuels do not mix.
        if self.records_path.exists():
            self.records_path.unlink()

    def _isolate_workspace(self, seed: Path | None = None) -> Path:
        root = Path(tempfile.mkdtemp(prefix="sonec-rollout-"))
        if seed and seed.exists():
            for item in seed.iterdir():
                dest = root / item.name
                if item.is_dir():
                    shutil.copytree(item, dest)
                else:
                    shutil.copy2(item, dest)
        return root

    async def rollout_one(
        self,
        task: EvalTask,
        *,
        group_id: str,
        rollout_index: int,
    ) -> RolloutRecord:
        ws = self._isolate_workspace()
        traj_dir = self.output_dir / "trajectories" / task.id
        traj_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        try:
            if self.use_mock and self.provider is None:
                settings = load_settings(workspace=ws, provider="mock")
                llm: LLMProvider | None = mock_provider_for_task(task)
            elif self.provider is not None:
                overrides: dict[str, object] = {
                    "workspace": ws,
                    "provider": self.provider_name,
                }
                if self.model:
                    overrides["model"] = self.model
                settings = load_settings(**overrides)
                llm = self.provider
            else:
                overrides = {"workspace": ws, "provider": self.provider_name}
                if self.model:
                    overrides["model"] = self.model
                settings = load_settings(**overrides)
                from sonec.llm.provider import create_provider

                llm = create_provider(settings)
            runtime, _, _, registry = build_runtime(
                settings=settings,
                provider=llm,
                persist_memory=False,
                log_dir=traj_dir,
                enable_phase_hints=False,
                goal_for_prompt=task.prompt,
            )
            del registry
            harness = EvalHarness(workspace=ws)
            harness.apply_seeds(task)
            agent_result = await runtime.run(task.prompt)
            graded = await harness.grade(task, agent_result)
            failure = classify_failure(
                graded.details,
                completed=agent_result.completed,
            )
            traj_files = sorted(traj_dir.glob("*.jsonl"))
            traj_path = str(traj_files[-1]) if traj_files else ""
            reward, reward_meta = compute_agent_reward(
                passed=graded.passed,
                trajectory_path=traj_path,
                task=task,
                details=graded.details,
            )
            if not graded.passed and reward_meta.get("failure"):
                failure = str(reward_meta["failure"])
            record = RolloutRecord(
                task_id=task.id,
                prompt=task.prompt,
                harness_version=agent_result.harness_version or HARNESS_VERSION,
                tool_schema_hash=agent_result.tool_schema_hash,
                model_id=agent_result.model_id,
                reward=reward,
                passed=graded.passed,
                trajectory_path=traj_path,
                failure_class=failure if not graded.passed else "none",
                group_id=group_id,
                rollout_index=rollout_index,
                duration_s=time.perf_counter() - started,
                details=graded.details,
                reward_meta=reward_meta,
            )
            with self.records_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record.to_json()) + "\n")
            return record
        finally:
            shutil.rmtree(ws, ignore_errors=True)

    async def rollout_group(self, task: EvalTask, *, group_id: str | None = None) -> list[RolloutRecord]:
        gid = group_id or f"g_{task.id}"
        # Sequential for determinism in mock; parallel ok for live later.
        records: list[RolloutRecord] = []
        for i in range(self.group_size):
            # Fresh mock provider each time
            records.append(await self.rollout_one(task, group_id=gid, rollout_index=i))
        return records

    async def run_tasks(
        self,
        tasks: list[EvalTask],
        *,
        group_size: int | None = None,
    ) -> list[RolloutRecord]:
        if group_size is not None:
            self.group_size = group_size
        all_records: list[RolloutRecord] = []
        for task in tasks:
            all_records.extend(await self.rollout_group(task))
        return all_records


def run_rollouts_sync(
    tasks: list[EvalTask],
    output_dir: Path,
    *,
    group_size: int = 2,
    use_mock: bool = True,
    provider_name: str = "local",
    model: str | None = None,
) -> list[RolloutRecord]:
    factory = RolloutFactory(
        output_dir=output_dir,
        group_size=group_size,
        use_mock=use_mock,
        provider=None,
        provider_name=provider_name,
        model=model,
    )
    return asyncio.run(factory.run_tasks(tasks, group_size=group_size))
