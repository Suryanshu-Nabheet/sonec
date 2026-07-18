"""TrainBench — large training-only task set (never overlap sealed eval ids)."""

from __future__ import annotations

import json
from pathlib import Path

from sonec.eval.harness import EvalCheck, EvalTask


def build_trainbench_tasks(*, n: int = 120) -> list[EvalTask]:
    """Synthetic but realistic SE tasks for SFT/RL fuel only.

    Curriculum kinds: exact nested path → config → bug → tests → ts →
    multi-file pkg → verify → restraint.
    """
    tasks: list[EvalTask] = []
    for i in range(1, n + 1):
        kind = i % 8
        if kind == 0:
            tasks.append(
                EvalTask(
                    id=f"train-path-{i:04d}",
                    name=f"Exact path write {i}",
                    prompt=(
                        f"Create notes/hello_{i}.txt containing exactly: hello sonec {i}"
                    ),
                    difficulty="easy",
                    tags=["train", "exact_path"],
                    checks=[
                        EvalCheck(kind="file_exists", path=f"notes/hello_{i}.txt"),
                        EvalCheck(
                            kind="file_contains",
                            path=f"notes/hello_{i}.txt",
                            contains=f"hello sonec {i}",
                        ),
                    ],
                )
            )
        elif kind == 1:
            tasks.append(
                EvalTask(
                    id=f"train-feat-{i:04d}",
                    name=f"Feature flag {i}",
                    prompt=f"Create config/flag_{i}.json with \"enabled\": true.",
                    difficulty="easy",
                    tags=["train", "feature"],
                    checks=[
                        EvalCheck(kind="file_exists", path=f"config/flag_{i}.json"),
                        EvalCheck(
                            kind="file_contains",
                            path=f"config/flag_{i}.json",
                            contains='"enabled": true',
                        ),
                    ],
                )
            )
        elif kind == 2:
            tasks.append(
                EvalTask(
                    id=f"train-bug-{i:04d}",
                    name=f"Bug fix {i}",
                    prompt=f"Create fix_{i}.py with def answer() returning {i}.",
                    difficulty="medium",
                    tags=["train", "bugfix"],
                    checks=[
                        EvalCheck(kind="file_exists", path=f"fix_{i}.py"),
                        EvalCheck(
                            kind="file_contains",
                            path=f"fix_{i}.py",
                            contains="def answer",
                        ),
                        EvalCheck(kind="python_parses", path=f"fix_{i}.py"),
                    ],
                )
            )
        elif kind == 3:
            tasks.append(
                EvalTask(
                    id=f"train-test-{i:04d}",
                    name=f"Add test {i}",
                    prompt=f"Create tests/test_t{i}.py with def test_ok asserting True.",
                    difficulty="medium",
                    tags=["train", "tests"],
                    checks=[
                        EvalCheck(kind="file_exists", path=f"tests/test_t{i}.py"),
                        EvalCheck(
                            kind="file_contains",
                            path=f"tests/test_t{i}.py",
                            contains="def test_ok",
                        ),
                        EvalCheck(kind="python_parses", path=f"tests/test_t{i}.py"),
                    ],
                )
            )
        elif kind == 4:
            tasks.append(
                EvalTask(
                    id=f"train-ts-{i:04d}",
                    name=f"TS util {i}",
                    prompt=(
                        f"Create src/util_{i}.ts exporting function util{i}(): "
                        f"number returning {i}."
                    ),
                    difficulty="medium",
                    tags=["train", "typescript"],
                    checks=[
                        EvalCheck(kind="file_exists", path=f"src/util_{i}.ts"),
                        EvalCheck(
                            kind="file_contains",
                            path=f"src/util_{i}.ts",
                            contains=f"util{i}",
                        ),
                    ],
                )
            )
        elif kind == 5:
            tasks.append(
                EvalTask(
                    id=f"train-pkg-{i:04d}",
                    name=f"Package scaffold {i}",
                    prompt=(
                        f"Create pkg{i}/__init__.py and pkg{i}/core.py with "
                        f"def helper_{i}() returning {i}."
                    ),
                    difficulty="medium",
                    tags=["train", "multifile"],
                    checks=[
                        EvalCheck(kind="file_exists", path=f"pkg{i}/__init__.py"),
                        EvalCheck(kind="file_exists", path=f"pkg{i}/core.py"),
                        EvalCheck(
                            kind="file_contains",
                            path=f"pkg{i}/core.py",
                            contains=f"def helper_{i}",
                        ),
                        EvalCheck(kind="python_parses", path=f"pkg{i}/core.py"),
                    ],
                )
            )
        elif kind == 6:
            tasks.append(
                EvalTask(
                    id=f"train-verify-{i:04d}",
                    name=f"Verify script {i}",
                    prompt=(
                        f"Create scripts/check_{i}.sh echoing PASS_{i} and "
                        f"VERIFY_{i}.md referencing ./scripts/check_{i}.sh"
                    ),
                    difficulty="medium",
                    tags=["train", "verify"],
                    checks=[
                        EvalCheck(kind="file_exists", path=f"scripts/check_{i}.sh"),
                        EvalCheck(
                            kind="file_contains",
                            path=f"scripts/check_{i}.sh",
                            contains=f"PASS_{i}",
                        ),
                        EvalCheck(
                            kind="file_contains",
                            path=f"VERIFY_{i}.md",
                            contains=f"./scripts/check_{i}.sh",
                        ),
                    ],
                )
            )
        else:
            tasks.append(
                EvalTask(
                    id=f"train-q-{i:04d}",
                    name=f"Restraint {i}",
                    prompt=(
                        f"In one sentence, what is continuous integration? "
                        f"(q{i}; do not edit files)"
                    ),
                    difficulty="easy",
                    tags=["train", "restraint"],
                    checks=[],
                )
            )
    return tasks


def write_trainbench(path: Path, *, n: int = 120) -> Path:
    tasks = build_trainbench_tasks(n=n)
    payload = {
        "name": "trainbench-v1",
        "version": "1.0.0",
        "sealed": False,
        "training_only": True,
        "description": "Training-only fuel. Do not use as decision eval.",
        "task_count": len(tasks),
        "tasks": [t.model_dump() for t in tasks],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
