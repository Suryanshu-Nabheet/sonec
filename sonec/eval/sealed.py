"""Sealed eval suite IDs — never training fuel."""

from __future__ import annotations

import json
from pathlib import Path

# Suites whose task ids must never enter SFT / RFT / GRPO fuel.
SEALED_SUITE_RELATIVE: tuple[str, ...] = (
    "examples/benchmarks/sonecbench_v1.json",
    "examples/benchmarks/worldbench_v1.json",
    "examples/benchmarks/ab_agent_v1.json",
    "examples/benchmarks/ab_agent_2b_hard.json",
    "examples/benchmarks/capabilitybench_v1.json",
)

# Harness / agent paths that must not count as workspace evidence for only_files.
HARNESS_IGNORE_DIR_NAMES: frozenset[str] = frozenset(
    {
        ".trajectories",
        ".sonec",
        ".git",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".venv",
        "venv",
        "node_modules",
    }
)


def collect_sealed_task_ids(root: Path | None = None) -> set[str]:
    """Load all sealed task ids from committed suite JSON files."""
    base = root or Path.cwd()
    sealed: set[str] = set()
    for rel in SEALED_SUITE_RELATIVE:
        suite = base / rel
        if not suite.is_file():
            continue
        data = json.loads(suite.read_text(encoding="utf-8"))
        for task in data.get("tasks") or []:
            tid = task.get("id")
            if tid:
                sealed.add(str(tid))
    return sealed


def is_harness_path(rel_posix: str) -> bool:
    """True if a workspace-relative path is harness noise (logs, caches)."""
    parts = Path(rel_posix).parts
    return any(part in HARNESS_IGNORE_DIR_NAMES for part in parts)
