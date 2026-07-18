"""Evaluation harness and benchmarking — deterministic grading for agentic SE."""

from __future__ import annotations

import ast
import json
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field

from sonec.core.errors import EvalError
from sonec.core.types import AgentRunResult, Message, Role, ToolCall, utc_now
from sonec.llm.provider import MockProvider


class EvalCheck(BaseModel):
    kind: str
    path: str | None = None
    contains: str | None = None
    command_exit_zero: bool | None = None
    pattern: str | None = None


class EvalTask(BaseModel):
    id: str
    name: str
    prompt: str
    checks: list[EvalCheck] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    timeout_s: float = 300.0
    difficulty: str = "easy"
    seed_files: dict[str, str] = Field(default_factory=dict)

class EvalResult(BaseModel):
    task_id: str
    passed: bool
    score: float
    details: list[str] = Field(default_factory=list)
    duration_s: float = 0.0
    difficulty: str = "easy"
    tags: list[str] = Field(default_factory=list)
    agent: AgentRunResult | None = None
    created_at: str = Field(default_factory=lambda: utc_now().isoformat())


class BenchmarkReport(BaseModel):
    name: str
    results: list[EvalResult]
    pass_rate: float
    mean_duration_s: float
    mean_score: float = 0.0
    passed: int = 0
    total: int = 0
    by_difficulty: dict[str, float] = Field(default_factory=dict)
    by_tag: dict[str, float] = Field(default_factory=dict)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")


class RunnableAgent(Protocol):
    async def run(self, goal: str) -> AgentRunResult: ...


class EvalHarness:
    """Runs tasks against an agent and scores deterministic checks."""

    def __init__(
        self,
        *,
        workspace: Path,
        run_command: Callable[[str], Awaitable[dict[str, Any]]] | None = None,
    ) -> None:
        self.workspace = workspace.resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
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
            label = check.path or check.contains or check.pattern or ""
            if ok:
                passed_checks += 1
                details.append(f"PASS {check.kind}: {label}")
            else:
                details.append(f"FAIL {check.kind}: {label}")
        total = max(len(task.checks), 1)
        if not task.checks:
            # Question-only / no-edit: environment evidence is "completed turn".
            score = 1.0 if agent_result.completed else 0.0
            passed = bool(agent_result.completed)
            details.append("No file checks; evidence=completed turn")
        else:
            score = passed_checks / total
            passed = passed_checks == total
        # Success is environment evidence only — never model self-report.
        agent_result.evidence_success = passed
        agent_result.success = passed
        return EvalResult(
            task_id=task.id,
            passed=passed,
            score=score,
            details=details,
            difficulty=task.difficulty,
            tags=list(task.tags),
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
        if check.kind == "file_not_contains":
            if not check.path or check.contains is None:
                return False
            path = self.workspace / check.path
            if not path.exists():
                return False
            return check.contains not in path.read_text(encoding="utf-8", errors="replace")
        if check.kind == "python_parses":
            if not check.path:
                return False
            path = self.workspace / check.path
            if not path.exists():
                return False
            try:
                ast.parse(path.read_text(encoding="utf-8"))
                return True
            except SyntaxError:
                return False
        if check.kind == "command":
            if self.run_command is None or not check.path:
                return False
            result = await self.run_command(check.path)
            if check.command_exit_zero:
                return result.get("exit_code") == 0
            return True
        raise EvalError(f"Unknown check kind: {check.kind}")

    def apply_seeds(self, task: EvalTask) -> None:
        """Materialize held-out seed files before the agent runs."""
        for rel, content in task.seed_files.items():
            path = self.workspace / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

    async def run_task(self, task: EvalTask, agent: RunnableAgent) -> EvalResult:
        started = time.perf_counter()
        self.apply_seeds(task)
        agent_result = await agent.run(task.prompt)
        graded = await self.grade(task, agent_result)
        graded.duration_s = time.perf_counter() - started
        return graded

    async def run_suite(
        self,
        tasks: list[EvalTask],
        agent_factory: Callable[[EvalTask], RunnableAgent],
        *,
        name: str = "suite",
    ) -> BenchmarkReport:
        results: list[EvalResult] = []
        for task in tasks:
            agent = agent_factory(task)
            try:
                results.append(await self.run_task(task, agent))
            except Exception as exc:  # noqa: BLE001 — live eval must continue arms
                results.append(
                    EvalResult(
                        task_id=task.id,
                        passed=False,
                        score=0.0,
                        details=[f"ERROR: {exc}"],
                        difficulty=task.difficulty,
                        tags=list(task.tags),
                    )
                )
        return build_report(name, results)


def build_report(name: str, results: list[EvalResult]) -> BenchmarkReport:
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    pass_rate = passed / max(total, 1)
    mean_duration = sum(r.duration_s for r in results) / max(total, 1)
    mean_score = sum(r.score for r in results) / max(total, 1)

    by_diff: dict[str, list[bool]] = {}
    by_tag: dict[str, list[bool]] = {}
    for r in results:
        by_diff.setdefault(r.difficulty, []).append(r.passed)
        for tag in r.tags:
            by_tag.setdefault(tag, []).append(r.passed)

    def rate(vals: list[bool]) -> float:
        return sum(1 for v in vals if v) / max(len(vals), 1)

    return BenchmarkReport(
        name=name,
        results=results,
        pass_rate=pass_rate,
        mean_duration_s=mean_duration,
        mean_score=mean_score,
        passed=passed,
        total=total,
        by_difficulty={k: rate(v) for k, v in sorted(by_diff.items())},
        by_tag={k: rate(v) for k, v in sorted(by_tag.items())},
    )


def mock_provider_for_task(task: EvalTask) -> MockProvider:
    """Build a scripted provider that satisfies deterministic file checks offline.

    Used for CI / harness self-tests. Live runs use the real model + tools.
    """
    targets: dict[str, dict[str, Any]] = {}
    for check in task.checks:
        if not check.path:
            continue
        if check.kind not in {
            "file_exists",
            "file_contains",
            "file_not_contains",
            "python_parses",
        }:
            continue
        meta = targets.setdefault(
            check.path,
            {"contains_all": [], "not_contains": [], "exists": True, "python": False},
        )
        if check.kind == "file_contains" and check.contains is not None:
            meta["contains_all"].append(check.contains)
        if check.kind == "file_not_contains" and check.contains is not None:
            meta["not_contains"].append(check.contains)
        if check.kind == "python_parses":
            meta["python"] = True
        meta["exists"] = True

    scripted: list[Message] = []
    for index, (path, meta) in enumerate(targets.items(), start=1):
        contains_all: list[str] = list(meta.get("contains_all") or [])
        not_contains_list: list[str] = list(meta.get("not_contains") or [])
        if meta.get("python"):
            primary = next((c for c in contains_all if c.startswith("def ")), None)
            if primary:
                name = primary.removeprefix("def ").split("(")[0].strip() or "main"
                content = f"def {name}() -> str:\n    return 'hello'\n"
            elif contains_all:
                joined = "\n".join(f"# {c}" for c in contains_all)
                content = f"{joined}\ndef main() -> None:\n    return None\n"
            else:
                content = "def main() -> None:\n    return None\n"
            for needle in contains_all:
                if needle not in content:
                    content = f"# {needle}\n" + content
        elif path.endswith(".json"):
            # Merge all required substrings into a JSON-ish document.
            blob = " ".join(contains_all) if contains_all else "sonec"
            content = "{\n"
            for i, needle in enumerate(contains_all or ["sonec"]):
                content += f'  "k{i}": "{needle}",\n'
            content += f'  "note": "{blob}"\n}}\n'
        elif contains_all:
            # Ensure every required substring appears.
            content = "\n".join(contains_all) + "\n"
        else:
            content = "sonec\n"
        for banned in not_contains_list:
            if banned in content:
                content = content.replace(banned, "")
            if not content.strip():
                content = "clean\n"
        scripted.append(
            Message(
                role=Role.ASSISTANT,
                content=None,
                tool_calls=[
                    ToolCall(
                        id=f"bench_{index}",
                        name="fs_write",
                        arguments={"path": path, "content": content},
                    )
                ],
            )
        )

    scripted.append(
        Message(
            role=Role.ASSISTANT,
            content=f"Benchmark task `{task.id}` completed with verification evidence.",
        )
    )
    return MockProvider(
        scripted,
        default=Message(role=Role.ASSISTANT, content="Benchmark phase complete."),
    )
