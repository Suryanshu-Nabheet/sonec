"""Agent-loop reward shaping for rejection / RFT ranking.

Primary signal remains grader pass (0/1). Penalties break ties among
winners and annotate failures for the data engine.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sonec.eval.harness import EvalTask

EXPLORE_TOOLS = frozenset(
    {
        "fs_list",
        "fs_read",
        "fs_search",
        "git_status",
        "git_diff",
        "git_log",
        "git_branch",
        "index_build",
        "index_search",
        "index_symbols",
        "memory_search",
        "skills_list",
        "skills_load",
        "rules_list",
        "rules_load",
    }
)
WRITE_TOOLS = frozenset({"fs_write", "fs_edit"})
VERIFY_TOOLS = frozenset({"terminal_run"})

# Max explore tool calls allowed before first write without penalty.
MAX_EXPLORE_BEFORE_WRITE = 4


def _tool_names_in_order(trajectory_path: str) -> list[str]:
    path = Path(trajectory_path)
    if not path.exists():
        return []
    names: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("type") != "message" or event.get("role") != "assistant":
            continue
        for call in event.get("tool_calls") or []:
            name = call.get("name") or (call.get("function") or {}).get("name")
            if name:
                names.append(str(name))
    return names


def analyze_tool_trace(trajectory_path: str) -> dict[str, Any]:
    names = _tool_names_in_order(trajectory_path)
    explore_before_write = 0
    saw_write = False
    for name in names:
        if name in WRITE_TOOLS:
            saw_write = True
            break
        if name in EXPLORE_TOOLS:
            explore_before_write += 1
    return {
        "tool_names": names,
        "n_tools": len(names),
        "saw_write": saw_write,
        "explore_before_write": explore_before_write,
        "saw_verify": any(n in VERIFY_TOOLS for n in names),
    }


def task_expects_files(task: EvalTask) -> bool:
    return any(c.kind in {"file_exists", "file_contains", "python_parses"} for c in task.checks)


def task_expects_verify(task: EvalTask) -> bool:
    """True only when graders require a live command / python_exec check.

    Cap200 category tag ``verify`` alone is not enough — those tasks are often
    file-evidence graded. Penalizing passers without ``terminal_run`` then
    mis-ranks good trajectories.
    """
    return any(c.kind in {"command", "python_exec"} for c in task.checks)


def compute_agent_reward(
    *,
    passed: bool,
    trajectory_path: str,
    task: EvalTask,
    details: list[str] | None = None,
) -> tuple[float, dict[str, Any]]:
    """Return (reward, meta). Passed-only fuel uses reward==1.0 filter elsewhere.

    Among passers, penalties lower reward for ranking (rejection sampling).
    Failures always get 0.0.
    """
    meta = analyze_tool_trace(trajectory_path)
    meta["passed"] = passed
    details = details or []
    joined = " ".join(details).lower()

    if not passed:
        if "file_exists" in joined or "file_contains" in joined:
            meta["failure"] = "wrong_path_or_content"
        elif task_expects_files(task) and not meta["saw_write"]:
            meta["failure"] = "no_write"
        elif meta["explore_before_write"] > MAX_EXPLORE_BEFORE_WRITE and not meta["saw_write"]:
            meta["failure"] = "explore_forever"
        else:
            meta["failure"] = "incomplete"
        return 0.0, meta

    reward = 1.0
    penalties: list[str] = []

    if task_expects_files(task) and not meta["saw_write"]:
        # Passed without tools is rare; still flag.
        reward -= 0.5
        penalties.append("no_write")

    if meta["explore_before_write"] > MAX_EXPLORE_BEFORE_WRITE:
        over = meta["explore_before_write"] - MAX_EXPLORE_BEFORE_WRITE
        reward -= min(0.4, 0.1 * over)
        penalties.append("explore_before_write")

    if task_expects_verify(task) and not meta["saw_verify"]:
        reward -= 0.2
        penalties.append("unfinished_verify")

    meta["penalties"] = penalties
    meta["reward"] = max(0.0, round(reward, 4))
    return meta["reward"], meta
