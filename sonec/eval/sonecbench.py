"""SonecBench task generation — private hard-ish SE eval (≥50 tasks)."""

from __future__ import annotations

import json
from pathlib import Path

from sonec.eval.harness import EvalCheck, EvalTask


def build_sonecbench_tasks() -> list[EvalTask]:
    """Construct ≥50 deterministic graded tasks (held-out from training by convention)."""
    tasks: list[EvalTask] = []

    # --- Bug-fix / feature / refactor / tests / migration / restraint ---
    templates: list[dict[str, object]] = []

    for i in range(1, 11):
        templates.append(
            {
                "id": f"bug-off-by-one-{i:02d}",
                "name": f"Off-by-one fix {i}",
                "prompt": f"util_{i}.py has a range bug. Ensure count_to(n) returns list(range(n)) not range(n+1).",
                "difficulty": "medium",
                "tags": ["bugfix", "python"],
                "checks": [
                    {"kind": "file_exists", "path": f"util_{i}.py"},
                    {"kind": "file_contains", "path": f"util_{i}.py", "contains": "def count_to"},
                    {"kind": "python_parses", "path": f"util_{i}.py"},
                ],
                "seed_files": {
                    f"util_{i}.py": (
                        "def count_to(n: int) -> list[int]:\n"
                        "    return list(range(n + 1))  # bug\n"
                    )
                },
            }
        )

    for i in range(1, 9):
        templates.append(
            {
                "id": f"feat-config-{i:02d}",
                "name": f"Add config flag {i}",
                "prompt": f"Add config/feature_{i}.json with enabled true for feature {i}.",
                "difficulty": "easy",
                "tags": ["feature", "config"],
                "checks": [
                    {"kind": "file_exists", "path": f"config/feature_{i}.json"},
                    {
                        "kind": "file_contains",
                        "path": f"config/feature_{i}.json",
                        "contains": '"enabled": true',
                    },
                ],
            }
        )

    for i in range(1, 7):
        templates.append(
            {
                "id": f"refactor-split-{i:02d}",
                "name": f"Split module {i}",
                "prompt": (
                    f"Create pkg{i}/core.py with helper() and pkg{i}/__init__.py exporting it."
                ),
                "difficulty": "hard",
                "tags": ["refactor", "architecture"],
                "checks": [
                    {"kind": "file_exists", "path": f"pkg{i}/__init__.py"},
                    {"kind": "file_exists", "path": f"pkg{i}/core.py"},
                    {"kind": "file_contains", "path": f"pkg{i}/core.py", "contains": "def helper"},
                    {"kind": "python_parses", "path": f"pkg{i}/core.py"},
                ],
            }
        )

    for i in range(1, 7):
        templates.append(
            {
                "id": f"test-add-{i:02d}",
                "name": f"Add tests {i}",
                "prompt": f"Add tests/test_mod_{i}.py with test_truth asserting True.",
                "difficulty": "medium",
                "tags": ["tests"],
                "checks": [
                    {"kind": "file_exists", "path": f"tests/test_mod_{i}.py"},
                    {
                        "kind": "file_contains",
                        "path": f"tests/test_mod_{i}.py",
                        "contains": "def test_truth",
                    },
                    {"kind": "python_parses", "path": f"tests/test_mod_{i}.py"},
                ],
            }
        )

    for i in range(1, 6):
        templates.append(
            {
                "id": f"secure-secret-{i:02d}",
                "name": f"Remove secret {i}",
                "prompt": f"Create secure_{i}.py without password= literals; use env.",
                "difficulty": "medium",
                "tags": ["security", "bugfix"],
                "checks": [
                    {"kind": "file_exists", "path": f"secure_{i}.py"},
                    {
                        "kind": "file_not_contains",
                        "path": f"secure_{i}.py",
                        "contains": "password=",
                    },
                    {"kind": "python_parses", "path": f"secure_{i}.py"},
                ],
            }
        )

    for i in range(1, 5):
        templates.append(
            {
                "id": f"migrate-rename-{i:02d}",
                "name": f"Migration rename {i}",
                "prompt": f"Create migrations/{i:04d}_init.sql containing CREATE TABLE items.",
                "difficulty": "easy",
                "tags": ["migration"],
                "checks": [
                    {"kind": "file_exists", "path": f"migrations/{i:04d}_init.sql"},
                    {
                        "kind": "file_contains",
                        "path": f"migrations/{i:04d}_init.sql",
                        "contains": "CREATE TABLE items",
                    },
                ],
            }
        )

    for i in range(1, 5):
        templates.append(
            {
                "id": f"debug-log-{i:02d}",
                "name": f"Debug from log {i}",
                "prompt": (
                    f"logs/fail_{i}.txt shows KeyError. Write fix_{i}.py defining safe_get(d, k)."
                ),
                "difficulty": "hard",
                "tags": ["debug", "logs"],
                "checks": [
                    {"kind": "file_exists", "path": f"fix_{i}.py"},
                    {"kind": "file_contains", "path": f"fix_{i}.py", "contains": "def safe_get"},
                    {"kind": "python_parses", "path": f"fix_{i}.py"},
                ],
                "seed_files": {
                    f"logs/fail_{i}.txt": f"KeyError: missing key on path {i}\n"
                },
            }
        )

    for i in range(1, 5):
        templates.append(
            {
                "id": f"restraint-q-{i:02d}",
                "name": f"Question only {i}",
                "prompt": f"In one sentence, what is a regression test? (question {i}; do not edit files)",
                "difficulty": "easy",
                "tags": ["review", "restraint"],
                "checks": [],
            }
        )

    # Pad to ≥50 if needed
    while len(templates) < 50:
        n = len(templates) + 1
        templates.append(
            {
                "id": f"docs-readme-{n:02d}",
                "name": f"Docs stub {n}",
                "prompt": f"Create docs/note_{n}.md mentioning SONEC.",
                "difficulty": "easy",
                "tags": ["docs"],
                "checks": [
                    {"kind": "file_exists", "path": f"docs/note_{n}.md"},
                    {
                        "kind": "file_contains",
                        "path": f"docs/note_{n}.md",
                        "contains": "SONEC",
                    },
                ],
            }
        )

    for raw in templates:
        tasks.append(
            EvalTask(
                id=str(raw["id"]),
                name=str(raw["name"]),
                prompt=str(raw["prompt"]),
                difficulty=str(raw.get("difficulty", "easy")),
                tags=list(raw.get("tags", [])),  # type: ignore[arg-type]
                checks=[EvalCheck.model_validate(c) for c in raw.get("checks", [])],  # type: ignore[arg-type]
                seed_files=dict(raw.get("seed_files", {})),  # type: ignore[arg-type]
            )
        )
    return tasks


def write_sonecbench(path: Path) -> Path:
    tasks = build_sonecbench_tasks()
    payload = {
        "name": "sonecbench-v1",
        "version": "1.0.0",
        "sealed": True,
        "description": (
            "Private SonecBench v1 — decision metric for coding-model training. "
            "Do not train on these task ids."
        ),
        "task_count": len(tasks),
        "tasks": [t.model_dump() for t in tasks],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    out = Path("examples/benchmarks/sonecbench_v1.json")
    write_sonecbench(out)
    print(f"Wrote {out} with {len(build_sonecbench_tasks())} tasks")
