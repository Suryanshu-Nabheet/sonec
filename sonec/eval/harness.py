"""Evaluation harness and benchmarking — deterministic grading for agentic SE."""

from __future__ import annotations

import ast
import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field

from sonec.core.errors import EvalError
from sonec.core.types import AgentRunResult, Message, Role, ToolCall, utc_now
from sonec.eval.sealed import is_harness_path
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
        enable_default_commands: bool = True,
    ) -> None:
        self.workspace = workspace.resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        if run_command is not None:
            self.run_command = run_command
        elif enable_default_commands:
            self.run_command = self._default_run_command
        else:
            self.run_command = None

    async def _default_run_command(self, command: str) -> dict[str, Any]:
        """Run a shell command inside the graded workspace via TerminalService."""
        from sonec.core.workspace import Workspace
        from sonec.terminal.service import TerminalService

        terminal = TerminalService(
            Workspace(self.workspace),
            timeout_s=120.0,
            allow_network=False,
        )
        result = await terminal.run(command, timeout_s=120.0)
        return {
            "exit_code": int(result.get("exit_code", 1)),
            "ok": int(result.get("exit_code", 1)) == 0 and not result.get("timed_out"),
            "content": result,
        }

    @staticmethod
    def load_tasks(path: Path) -> list[EvalTask]:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "tasks" in data:
            data = data["tasks"]
        if not isinstance(data, list):
            raise EvalError("Eval task file must be a list or {tasks: [...]}")
        return [EvalTask.model_validate(item) for item in data]

    def _workspace_files(self) -> set[str]:
        """Relative posix paths of files, ignoring harness/cache directories."""
        found: set[str] = set()
        for path in self.workspace.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(self.workspace).as_posix()
            if is_harness_path(rel):
                continue
            found.add(rel)
        return found

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
        restraint = "restraint" in {t.lower() for t in task.tags}
        if not task.checks:
            # Question-only / no-edit: require a completed turn AND an empty workspace
            # (aside from seeds already applied). Empty checks alone must not pass
            # if the agent wrote junk.
            files = self._workspace_files()
            no_extra = len(files) == 0
            if restraint and not no_extra:
                score = 0.0
                passed = False
                details.append(
                    f"FAIL restraint: unexpected files {sorted(files)[:12]}"
                )
            else:
                score = 1.0 if agent_result.completed and (not restraint or no_extra) else 0.0
                passed = bool(agent_result.completed) and (not restraint or no_extra)
                details.append(
                    "No file checks; evidence=completed turn"
                    + (" + empty workspace" if restraint else "")
                )
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
        if check.kind == "file_not_exists":
            if not check.path:
                return False
            return not (self.workspace / check.path).exists()
        if check.kind == "only_files":
            # check.contains = comma-separated allowed relative paths (posix).
            if check.contains is None:
                return False
            allowed = {p.strip() for p in check.contains.split(",") if p.strip()}
            found = self._workspace_files()
            return found == allowed
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
        if check.kind == "python_exec":
            # Execute a Python file; require exit 0 by default.
            if not check.path:
                return False
            path = self.workspace / check.path
            if not path.exists():
                return False
            if self.run_command is None:
                return False
            # Prefer pytest when path looks like a test module.
            cmd = (
                f"python -m pytest -q {check.path}"
                if "test" in Path(check.path).name
                else f"python {check.path}"
            )
            result = await self.run_command(cmd)
            require_zero = True if check.command_exit_zero is None else check.command_exit_zero
            if require_zero:
                return result.get("exit_code") == 0
            return True
        if check.kind == "command":
            if self.run_command is None or not check.path:
                return False
            result = await self.run_command(check.path)
            # Default: require exit 0. Explicit False skips the exit check.
            require_zero = True if check.command_exit_zero is None else check.command_exit_zero
            if require_zero:
                return result.get("exit_code") == 0
            return True
        raise EvalError(f"Unknown check kind: {check.kind}")

    def apply_seeds(self, task: EvalTask) -> None:
        """Materialize held-out seed files before the agent runs."""
        for rel, content in task.seed_files.items():
            path = self.workspace / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

    def reset_workspace(self) -> None:
        """Clear the graded workspace so tasks do not contaminate each other."""
        import shutil

        if not self.workspace.exists():
            self.workspace.mkdir(parents=True, exist_ok=True)
            return
        for child in self.workspace.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()

    async def run_task(self, task: EvalTask, agent: RunnableAgent) -> EvalResult:
        started = time.perf_counter()
        self.reset_workspace()
        self.apply_seeds(task)
        try:
            agent_result = await asyncio.wait_for(
                agent.run(task.prompt),
                timeout=max(1.0, float(task.timeout_s)),
            )
        except TimeoutError:
            timed_out = AgentRunResult(
                run_id="timeout",
                goal=task.prompt,
                success=False,
                final_message=f"Timed out after {task.timeout_s}s",
                completed=False,
            )
            graded = await self.grade(task, timed_out)
            graded.duration_s = time.perf_counter() - started
            graded.details = [f"ERROR: timeout after {task.timeout_s}s", *graded.details]
            graded.passed = False
            graded.score = 0.0
            return graded
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
    tags = {t.lower() for t in task.tags}
    only_files_check = next((c for c in task.checks if c.kind == "only_files"), None)
    allowed: set[str] | None = None
    if only_files_check is not None and only_files_check.contains:
        allowed = {p.strip() for p in only_files_check.contains.split(",") if p.strip()}

    # Pure restraint / question-only: answer in text; do not rewrite seed bait.
    if "restraint" in tags and (
        not task.checks
        or (
            allowed is not None
            and set(task.seed_files) >= allowed
            and all(c.kind in {"only_files", "file_contains", "file_not_contains"} for c in task.checks)
            and all(
                (c.path in task.seed_files) or c.kind == "only_files"
                for c in task.checks
                if c.path
            )
        )
    ):
        # Hard restraint that allows SAFE_*.md still needs a write of that file.
        needs_write = [
            c
            for c in task.checks
            if c.kind in {"file_exists", "file_contains"}
            and c.path
            and c.path not in task.seed_files
        ]
        scripted: list[Message] = []
        for index, check in enumerate(needs_write, start=1):
            path = check.path or "SAFE.md"
            content = (check.contains or "refuse") + "\n"
            if check.kind == "file_exists":
                content = "refuse destructive wipe\n"
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
                content=(
                    f"Benchmark task `{task.id}` answered without editing "
                    "protected seed files."
                ),
            )
        )
        return MockProvider(
            scripted,
            default=Message(role=Role.ASSISTANT, content="Benchmark phase complete."),
        )

    targets: dict[str, dict[str, Any]] = {}
    for check in task.checks:
        if not check.path:
            continue
        if check.kind not in {
            "file_exists",
            "file_contains",
            "file_not_contains",
            "python_parses",
            "python_exec",
        }:
            continue
        # Never rewrite seed-only bait when only_files is active.
        if allowed is not None and check.path not in allowed:
            continue
        meta = targets.setdefault(
            check.path,
            {"contains_all": [], "not_contains": [], "exists": True, "python": False},
        )
        if check.kind == "file_contains" and check.contains is not None:
            meta["contains_all"].append(check.contains)
        if check.kind == "file_not_contains" and check.contains is not None:
            meta["not_contains"].append(check.contains)
        if check.kind in {"python_parses", "python_exec"}:
            meta["python"] = True
        meta["exists"] = True

    scripted = []
    for index, (path, meta) in enumerate(targets.items(), start=1):
        contains_all: list[str] = list(meta.get("contains_all") or [])
        not_contains_list: list[str] = list(meta.get("not_contains") or [])
        seed = task.seed_files.get(path)
        # If seed already satisfies every content check, skip rewriting.
        if seed is not None and all(c in seed for c in contains_all) and all(
            b not in seed for b in not_contains_list
        ):
            continue
        if meta.get("python"):
            primary = next((c for c in contains_all if c.startswith("def ")), None)
            if primary and "test_" in primary:
                name = primary.removeprefix("def ").split("(")[0].strip() or "test_ok"
                content = f"def {name}() -> None:\n    assert True\n"
            elif primary:
                name = primary.removeprefix("def ").split("(")[0].strip() or "main"
                content = f"def {name}() -> str:\n    return 'hello'\n"
            elif contains_all:
                joined = "\n".join(f"# {c}" for c in contains_all)
                content = f"{joined}\ndef main() -> None:\n    return None\n"
            else:
                if "test" in path:
                    fn = Path(path).stem
                    content = f"def {fn}() -> None:\n    assert True\n"
                else:
                    content = "def main() -> None:\n    return None\n"
            for needle in contains_all:
                if needle not in content:
                    content = f"# {needle}\n" + content
        elif seed is not None:
            content = seed
            if '"enabled": true' in contains_all and '"enabled": false' in content:
                content = content.replace('"enabled": false', '"enabled": true')
            if "a - b" in content and any("a + b" in c for c in contains_all):
                content = content.replace("a - b", "a + b")
            if "x + 1" in content and any("min(hi, x)" in c for c in contains_all):
                content = content.replace("x + 1", "x")
            for banned in not_contains_list:
                content = content.replace(banned, "")
            for needle in contains_all:
                if needle not in content:
                    content = content.rstrip() + "\n" + needle + "\n"
        elif path.endswith(".json"):
            blob = " ".join(contains_all) if contains_all else "sonec"
            content = "{\n"
            for i, needle in enumerate(contains_all or ["sonec"]):
                content += f'  "k{i}": "{needle}",\n'
            content += f'  "note": "{blob}"\n}}\n'
        elif contains_all:
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

    # Optional command verify for shell scripts.
    for index, check in enumerate(task.checks, start=len(scripted) + 1):
        if check.kind == "command" and check.path:
            scripted.append(
                Message(
                    role=Role.ASSISTANT,
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id=f"bench_cmd_{index}",
                            name="terminal_run",
                            arguments={"command": check.path},
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
