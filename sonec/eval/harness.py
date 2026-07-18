"""Evaluation harness and benchmarking."""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from sonec.agent.runtime import AgentRuntime
from sonec.core.errors import EvalError
from sonec.core.types import AgentRunResult, utc_now


class EvalCheck(BaseModel):
    kind: str
    path: str | None = None
    contains: str | None = None
    command_exit_zero: bool | None = None


class EvalTask(BaseModel):
    id: str
    name: str
    prompt: str
    checks: list[EvalCheck] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    timeout_s: float = 300.0


class EvalResult(BaseModel):
    task_id: str
    passed: bool
    score: float
    details: list[str] = Field(default_factory=list)
    duration_s: float = 0.0
    agent: AgentRunResult | None = None
    created_at: str = Field(default_factory=lambda: utc_now().isoformat())


class BenchmarkReport(BaseModel):
    name: str
    results: list[EvalResult]
    pass_rate: float
    mean_duration_s: float


AgentFactory = Callable[[], Awaitable[AgentRuntime] | AgentRuntime]


class EvalHarness:
    """Runs tasks against an agent and scores deterministic checks."""

    def __init__(
        self,
        *,
        workspace: Path,
        run_command: Callable[[str], Awaitable[dict[str, Any]]] | None = None,
    ) -> None:
        self.workspace = workspace.resolve()
        self.run_command = run_command

    @staticmethod
    def load_tasks(path: Path) -> list[EvalTask]:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "tasks" in data:
            data = data["tasks"]
        if not isinstance(data, list):
            raise EvalError("Eval task file must be a list or {tasks: [...]}")
        return [EvalTask.model_validate(item) for item in data]

    async def grade(self, task: EvalTask, agent_result: AgentRunResult) -> EvalResult:
        details: list[str] = []
        passed_checks = 0
        for check in task.checks:
            ok = await self._check(check)
            if ok:
                passed_checks += 1
                details.append(f"PASS {check.kind}: {check.path or check.contains or ''}")
            else:
                details.append(f"FAIL {check.kind}: {check.path or check.contains or ''}")
        total = max(len(task.checks), 1)
        # If no checks, fall back to agent success
        if not task.checks:
            score = 1.0 if agent_result.success else 0.0
            passed = agent_result.success
            details.append("No checks defined; used agent success flag")
        else:
            score = passed_checks / total
            passed = passed_checks == total
        return EvalResult(
            task_id=task.id,
            passed=passed,
            score=score,
            details=details,
            agent=agent_result,
        )

    async def _check(self, check: EvalCheck) -> bool:
        if check.kind == "file_exists":
            if not check.path:
                return False
            return (self.workspace / check.path).exists()
        if check.kind == "file_contains":
            if not check.path or check.contains is None:
                return False
            path = self.workspace / check.path
            if not path.exists():
                return False
            return check.contains in path.read_text(encoding="utf-8", errors="replace")
        if check.kind == "command":
            if self.run_command is None or not check.path:
                return False
            result = await self.run_command(check.path)
            if check.command_exit_zero:
                return result.get("exit_code") == 0
            return True
        raise EvalError(f"Unknown check kind: {check.kind}")

    async def run_task(self, task: EvalTask, agent: AgentRuntime) -> EvalResult:
        started = time.perf_counter()
        agent_result = await agent.run(task.prompt)
        graded = await self.grade(task, agent_result)
        graded.duration_s = time.perf_counter() - started
        return graded

    async def run_suite(
        self,
        tasks: list[EvalTask],
        agent_factory: Callable[[], AgentRuntime],
        *,
        name: str = "suite",
    ) -> BenchmarkReport:
        results: list[EvalResult] = []
        for task in tasks:
            agent = agent_factory()
            results.append(await self.run_task(task, agent))
        pass_rate = sum(1 for r in results if r.passed) / max(len(results), 1)
        mean_duration = sum(r.duration_s for r in results) / max(len(results), 1)
        return BenchmarkReport(
            name=name,
            results=results,
            pass_rate=pass_rate,
            mean_duration_s=mean_duration,
        )
