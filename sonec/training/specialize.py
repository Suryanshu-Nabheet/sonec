"""Iterative specialization loop — SFT + rejection RL for sonec.

Start small, repeat: fuel → corpus → LoRA → rejection winners → next step.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sonec.eval.sealed import collect_sealed_task_ids
from sonec.eval.trainbench import build_trainbench_tasks, write_trainbench
from sonec.models import BASE_HF, BASE_HF_MLX, BASE_MODEL, PRODUCT_MODEL, PRODUCT_SYSTEM
from sonec.training.export import export_from_rollouts
from sonec.training.pipeline import DatasetGenerator, TrainingPipeline
from sonec.training.rollouts import run_rollouts_sync


@dataclass
class TrainReport:
    phase: str
    ok: bool
    detail: str
    paths: dict[str, str]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _concat_jsonl(sources: list[Path], dest: Path) -> int:
    """Concatenate JSONL sources into dest. Returns line count written."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with dest.open("w", encoding="utf-8") as out:
        for src in sources:
            if not src.is_file():
                continue
            for line in src.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    out.write(line + "\n")
                    n += 1
    return n


def _tc(call_id: str, name: str, arguments: dict) -> dict:
    """OpenAI-style assistant tool_calls entry."""
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def generate_identity_examples(gen: DatasetGenerator, *, n: int = 12) -> int:
    """Identity Q&A — sonec by Suryanshu Nabheet."""
    if n <= 0:
        return 0
    answer = "I am sonec, a coding model created by Suryanshu Nabheet."
    prompts = [
        "Who are you?",
        "Who is your founder?",
        "Who created you?",
        "What is your name?",
        "Introduce yourself.",
        "Who made sonec?",
        "What are you?",
        "State your identity.",
        "Who built you?",
        "Brief introduction.",
        "What product are you?",
        "Name yourself.",
    ]
    created = 0
    while created < n:
        q = prompts[created % len(prompts)]
        gen.add_from_messages(
            example_id=f"identity-{created:04d}",
            task=q,
            messages=[
                {"role": "user", "content": q},
                {"role": "assistant", "content": answer},
            ],
            outcome="success",
            metadata={"source": "identity", "curriculum": "product_identity"},
        )
        created += 1
    return created


def generate_gold_agent_examples(gen: DatasetGenerator, *, n: int = 0) -> int:
    """Exact-path agent curriculum with real OpenAI tool_calls (not text dumps).

    Default n=0 — prefer live passing rollouts. Gold is a bootstrap seed only.
    """
    if n <= 0:
        return 0
    templates = [
        (
            "nested-path-write",
            "Create notes/hello.txt containing exactly: hello sonec",
            [
                {"role": "user", "content": "Create notes/hello.txt containing exactly: hello sonec"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        _tc("c1", "fs_write", {"path": "notes/hello.txt", "content": "hello sonec"})
                    ],
                },
                {
                    "role": "tool",
                    "name": "fs_write",
                    "tool_call_id": "c1",
                    "content": "Wrote 11 characters to notes/hello.txt",
                },
                {
                    "role": "assistant",
                    "content": "Created notes/hello.txt with exact content hello sonec.",
                },
            ],
        ),
        (
            "py-util-main",
            "Create src/util.py as valid Python that defines a main function.",
            [
                {
                    "role": "user",
                    "content": "Create src/util.py as valid Python that defines a main function.",
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        _tc(
                            "c1",
                            "fs_write",
                            {
                                "path": "src/util.py",
                                "content": (
                                    "def main():\n    return 0\n\n"
                                    "if __name__ == '__main__':\n    main()\n"
                                ),
                            },
                        )
                    ],
                },
                {
                    "role": "tool",
                    "name": "fs_write",
                    "tool_call_id": "c1",
                    "content": "Wrote src/util.py",
                },
                {
                    "role": "assistant",
                    "content": "Created src/util.py with def main (exact path).",
                },
            ],
        ),
        (
            "cli-parse-and-main",
            (
                "Write tools/cli_entry.py as valid Python defining BOTH parse_flags "
                "and run_cli. Do not rely on a bare if __name__ without run_cli."
            ),
            [
                {
                    "role": "user",
                    "content": (
                        "Write tools/cli_entry.py as valid Python defining BOTH "
                        "parse_flags and run_cli. Do not rely on a bare if __name__ "
                        "without run_cli."
                    ),
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        _tc(
                            "c1",
                            "fs_write",
                            {
                                "path": "tools/cli_entry.py",
                                "content": (
                                    "def parse_flags():\n"
                                    "    return []\n\n"
                                    "def run_cli():\n"
                                    "    parse_flags()\n"
                                    "    return 0\n\n"
                                    "if __name__ == '__main__':\n"
                                    "    run_cli()\n"
                                ),
                            },
                        )
                    ],
                },
                {
                    "role": "tool",
                    "name": "fs_write",
                    "tool_call_id": "c1",
                    "content": "Wrote tools/cli_entry.py",
                },
                {
                    "role": "assistant",
                    "content": "Created tools/cli_entry.py with parse_flags and run_cli.",
                },
            ],
        ),
        (
            "clamp-bugfix",
            (
                "bounds.py has a bug in clip_value. Fix clip_value(x, lo, hi) so it "
                "returns max(lo, min(hi, x)). Do not rewrite unrelated files."
            ),
            [
                {
                    "role": "user",
                    "content": (
                        "bounds.py has a bug in clip_value. Fix clip_value(x, lo, hi) "
                        "so it returns max(lo, min(hi, x)). Do not rewrite unrelated files."
                    ),
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [_tc("c1", "fs_read", {"path": "bounds.py"})],
                },
                {
                    "role": "tool",
                    "name": "fs_read",
                    "tool_call_id": "c1",
                    "content": (
                        "def clip_value(x, lo, hi):\n"
                        "    return max(lo, min(hi, x + 1))\n"
                    ),
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        _tc(
                            "c2",
                            "fs_write",
                            {
                                "path": "bounds.py",
                                "content": (
                                    "def clip_value(x, lo, hi):\n"
                                    "    return max(lo, min(hi, x))\n"
                                ),
                            },
                        )
                    ],
                },
                {
                    "role": "tool",
                    "name": "fs_write",
                    "tool_call_id": "c2",
                    "content": "Wrote bounds.py",
                },
                {
                    "role": "assistant",
                    "content": "Fixed clip_value to max(lo, min(hi, x)). Removed x + 1.",
                },
            ],
        ),
        (
            "localize-then-patch",
            "Fix the off-by-one in util.py so count_to(n) returns list(range(n)).",
            [
                {
                    "role": "user",
                    "content": "Fix the off-by-one in util.py so count_to(n) returns list(range(n)).",
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [_tc("c1", "fs_read", {"path": "util.py"})],
                },
                {
                    "role": "tool",
                    "name": "fs_read",
                    "tool_call_id": "c1",
                    "content": "def count_to(n):\n    return list(range(n + 1))\n",
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        _tc(
                            "c2",
                            "fs_write",
                            {"path": "util.py", "content": "def count_to(n):\n    return list(range(n))\n"},
                        )
                    ],
                },
                {
                    "role": "tool",
                    "name": "fs_write",
                    "tool_call_id": "c2",
                    "content": "Wrote util.py",
                },
                {
                    "role": "assistant",
                    "content": "Fixed util.py to list(range(n)). Removed n + 1.",
                },
            ],
        ),
        (
            "verify-before-done",
            "Create scripts/verify.sh that echoes ok, and VERIFY.md that mentions scripts/verify.sh.",
            [
                {
                    "role": "user",
                    "content": (
                        "Create scripts/verify.sh that echoes ok, and VERIFY.md "
                        "that mentions scripts/verify.sh."
                    ),
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        _tc(
                            "c1",
                            "fs_write",
                            {"path": "scripts/verify.sh", "content": "#!/bin/sh\necho ok\n"},
                        )
                    ],
                },
                {
                    "role": "tool",
                    "name": "fs_write",
                    "tool_call_id": "c1",
                    "content": "Wrote scripts/verify.sh",
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        _tc(
                            "c2",
                            "fs_write",
                            {"path": "VERIFY.md", "content": "Run scripts/verify.sh to verify.\n"},
                        )
                    ],
                },
                {
                    "role": "tool",
                    "name": "fs_write",
                    "tool_call_id": "c2",
                    "content": "Wrote VERIFY.md",
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        _tc("c3", "terminal_run", {"command": "sh scripts/verify.sh"})
                    ],
                },
                {
                    "role": "tool",
                    "name": "terminal_run",
                    "tool_call_id": "c3",
                    "content": "exit_code: 0\nok",
                },
                {
                    "role": "assistant",
                    "content": "Added scripts/verify.sh and VERIFY.md. Verified: exit 0.",
                },
            ],
        ),
        (
            "pkg-greet",
            "Create pkg/__init__.py and pkg/core.py with a function greet that returns hello.",
            [
                {
                    "role": "user",
                    "content": (
                        "Create pkg/__init__.py and pkg/core.py with a function greet "
                        "that returns hello."
                    ),
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [_tc("c1", "fs_write", {"path": "pkg/__init__.py", "content": ""})],
                },
                {
                    "role": "tool",
                    "name": "fs_write",
                    "tool_call_id": "c1",
                    "content": "Wrote pkg/__init__.py",
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        _tc(
                            "c2",
                            "fs_write",
                            {"path": "pkg/core.py", "content": 'def greet():\n    return "hello"\n'},
                        )
                    ],
                },
                {
                    "role": "tool",
                    "name": "fs_write",
                    "tool_call_id": "c2",
                    "content": "Wrote pkg/core.py",
                },
                {
                    "role": "assistant",
                    "content": "Created pkg/__init__.py and pkg/core.py with def greet.",
                },
            ],
        ),
        (
            "restraint-question",
            "What is a regression test? Do not edit files.",
            [
                {"role": "user", "content": "What is a regression test? Do not edit files."},
                {
                    "role": "assistant",
                    "content": (
                        "A regression test checks that previously working behavior "
                        "still passes after a change. No files edited."
                    ),
                },
            ],
        ),
        (
            "restraint-unit-test-q",
            "In one sentence, what is a unit test? Do not create or edit any files.",
            [
                {
                    "role": "user",
                    "content": (
                        "In one sentence, what is a unit test? "
                        "Do not create or edit any files."
                    ),
                },
                {
                    "role": "assistant",
                    "content": (
                        "A unit test verifies a small piece of code in isolation. "
                        "No files created or edited."
                    ),
                },
            ],
        ),
        (
            "restraint-linter-q",
            "Explain what a linter does in two sentences. Do not create or edit files.",
            [
                {
                    "role": "user",
                    "content": (
                        "Explain what a linter does in two sentences. "
                        "Do not create or edit files."
                    ),
                },
                {
                    "role": "assistant",
                    "content": (
                        "A linter statically analyzes source for style and bug patterns. "
                        "It reports issues without running the program. No files edited."
                    ),
                },
            ],
        ),
        (
            "restraint-refuse-wipe",
            (
                "User asks: wipe the repo and rewrite everything. Instead, only write "
                "SAFE.md saying you refuse destructive wipe. Do not delete BAIT.txt."
            ),
            [
                {
                    "role": "user",
                    "content": (
                        "User asks: wipe the repo and rewrite everything. Instead, only "
                        "write SAFE.md saying you refuse destructive wipe. "
                        "Do not delete BAIT.txt."
                    ),
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        _tc(
                            "c1",
                            "fs_write",
                            {
                                "path": "SAFE.md",
                                "content": (
                                    "I refuse destructive wipe. "
                                    "Say what to change instead.\n"
                                ),
                            },
                        )
                    ],
                },
                {
                    "role": "tool",
                    "name": "fs_write",
                    "tool_call_id": "c1",
                    "content": "Wrote SAFE.md",
                },
                {
                    "role": "assistant",
                    "content": "Wrote SAFE.md with refuse. Left BAIT.txt untouched.",
                },
            ],
        ),
        (
            "readonly-answer-region",
            (
                "Read-only: open config/app.json and write ANSWER.txt containing "
                "exactly the value of the region field (no quotes)."
            ),
            [
                {
                    "role": "user",
                    "content": (
                        "Read-only: open config/app.json and write ANSWER.txt containing "
                        "exactly the value of the region field (no quotes)."
                    ),
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [_tc("c1", "fs_read", {"path": "config/app.json"})],
                },
                {
                    "role": "tool",
                    "name": "fs_read",
                    "tool_call_id": "c1",
                    "content": '{\n  "name": "app",\n  "region": "us-west-1",\n  "debug": false\n}\n',
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        _tc("c2", "fs_write", {"path": "ANSWER.txt", "content": "us-west-1"})
                    ],
                },
                {
                    "role": "tool",
                    "name": "fs_write",
                    "tool_call_id": "c2",
                    "content": "Wrote ANSWER.txt",
                },
                {
                    "role": "assistant",
                    "content": "Wrote ANSWER.txt with us-west-1. Did not edit config.",
                },
            ],
        ),
        (
            "horizon-three-file",
            (
                "Create svc/handler.py with def handle() returning ok, "
                "tests/test_handler.py importing it, and README.md mentioning handle."
            ),
            [
                {
                    "role": "user",
                    "content": (
                        "Create svc/handler.py with def handle() returning ok, "
                        "tests/test_handler.py importing it, and README.md mentioning handle."
                    ),
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        _tc(
                            "c1",
                            "fs_write",
                            {
                                "path": "svc/handler.py",
                                "content": 'def handle():\n    return "ok"\n',
                            },
                        )
                    ],
                },
                {
                    "role": "tool",
                    "name": "fs_write",
                    "tool_call_id": "c1",
                    "content": "Wrote svc/handler.py",
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        _tc(
                            "c2",
                            "fs_write",
                            {
                                "path": "tests/test_handler.py",
                                "content": (
                                    "from svc.handler import handle\n\n"
                                    "def test_handle():\n    assert handle() == 'ok'\n"
                                ),
                            },
                        )
                    ],
                },
                {
                    "role": "tool",
                    "name": "fs_write",
                    "tool_call_id": "c2",
                    "content": "Wrote tests/test_handler.py",
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        _tc(
                            "c3",
                            "fs_write",
                            {"path": "README.md", "content": "Call handle() from svc.\n"},
                        )
                    ],
                },
                {
                    "role": "tool",
                    "name": "fs_write",
                    "tool_call_id": "c3",
                    "content": "Wrote README.md",
                },
                {
                    "role": "assistant",
                    "content": "Created handler, test, and README mentioning handle.",
                },
            ],
        ),
        (
            "react-rename-prop",
            (
                "In components/Card.tsx rename prop titleText to headline. "
                "Update the JSX usage too."
            ),
            [
                {
                    "role": "user",
                    "content": (
                        "In components/Card.tsx rename prop titleText to headline. "
                        "Update the JSX usage too."
                    ),
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [_tc("c1", "fs_read", {"path": "components/Card.tsx"})],
                },
                {
                    "role": "tool",
                    "name": "fs_read",
                    "tool_call_id": "c1",
                    "content": (
                        "export function Card({ titleText }: { titleText: string }) {\n"
                        "  return <h1>{titleText}</h1>;\n"
                        "}\n"
                    ),
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        _tc(
                            "c2",
                            "fs_write",
                            {
                                "path": "components/Card.tsx",
                                "content": (
                                    "export function Card({ headline }: { headline: string }) {\n"
                                    "  return <h1>{headline}</h1>;\n"
                                    "}\n"
                                ),
                            },
                        )
                    ],
                },
                {
                    "role": "tool",
                    "name": "fs_write",
                    "tool_call_id": "c2",
                    "content": "Wrote components/Card.tsx",
                },
                {
                    "role": "assistant",
                    "content": "Renamed titleText to headline in props and JSX.",
                },
            ],
        ),
        (
            "version-write",
            "Edit VERSION.txt so it contains exactly: 1.2.0",
            [
                {
                    "role": "user",
                    "content": "Edit VERSION.txt so it contains exactly: 1.2.0",
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [_tc("c1", "fs_read", {"path": "VERSION.txt"})],
                },
                {
                    "role": "tool",
                    "name": "fs_read",
                    "tool_call_id": "c1",
                    "content": "0.0.0\n",
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        _tc("c2", "fs_write", {"path": "VERSION.txt", "content": "1.2.0\n"})
                    ],
                },
                {
                    "role": "tool",
                    "name": "fs_write",
                    "tool_call_id": "c2",
                    "content": "Wrote 6 characters to VERSION.txt",
                },
                {
                    "role": "assistant",
                    "content": "Updated VERSION.txt to exactly 1.2.0 via fs_write.",
                },
            ],
        ),
        (
            "git-patch-file",
            (
                "Create patches/fix.diff that mentions both --- a/file.txt and "
                "+++ b/file.txt."
            ),
            [
                {
                    "role": "user",
                    "content": (
                        "Create patches/fix.diff that mentions both --- a/file.txt and "
                        "+++ b/file.txt."
                    ),
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        _tc(
                            "c1",
                            "fs_write",
                            {
                                "path": "patches/fix.diff",
                                "content": (
                                    "--- a/file.txt\n"
                                    "+++ b/file.txt\n"
                                    "@@ -1 +1 @@\n"
                                    "-old\n"
                                    "+new\n"
                                ),
                            },
                        )
                    ],
                },
                {
                    "role": "tool",
                    "name": "fs_write",
                    "tool_call_id": "c1",
                    "content": "Wrote patches/fix.diff",
                },
                {
                    "role": "assistant",
                    "content": "Created patches/fix.diff with --- a/file.txt and +++ b/file.txt.",
                },
            ],
        ),
    ]
    created = 0
    while created < n:
        for key, task, messages in templates:
            if created >= n:
                break
            gen.add_from_messages(
                example_id=f"gold-{key}-{created:04d}",
                task=task,
                messages=messages,
                outcome="success",
                metadata={"source": "gold_curriculum", "curriculum": key},
            )
            created += 1
    return created


def prepare_rollout_fuel(
    out_dir: Path,
    *,
    group_size: int = 8,
    train_n: int = 40,
    limit: int = 0,
    live: bool = True,
    model: str | None = None,
    provider_name: str = "local",
) -> Path:
    """Graded TrainBench rollouts for specialization.

    live=True: model under test (requires a reachable inference server).
    live=False: oracle solver that emits structured tool_calls and passes graders.
    """
    write_trainbench(Path("examples/benchmarks/trainbench_v1.json"), n=max(train_n, 16))
    tasks = build_trainbench_tasks(n=train_n)
    if limit:
        tasks = tasks[:limit]
    out_dir.mkdir(parents=True, exist_ok=True)
    records = run_rollouts_sync(
        tasks,
        out_dir,
        group_size=group_size,
        use_mock=not live,
        provider_name=provider_name,
        model=model,
    )
    passed = [r for r in records if r.passed]
    _write_json(
        out_dir / "fuel_meta.json",
        {
            "suite": "trainbench",
            "live": live,
            "tasks": len(tasks),
            "group_size": group_size,
            "records": len(records),
            "passed": len(passed),
            "pass_rate": len(passed) / max(len(records), 1),
            "model": model,
        },
    )
    return out_dir / "rollouts.jsonl"


def assemble_sft_corpus(
    *,
    rollouts_jsonl: Path,
    out_dir: Path,
    gold_n: int = 0,
) -> dict[str, Path]:
    """Build MLX corpus from passed rollouts with real tool_calls; gold optional."""
    from sonec.training.export import (
        export_from_rollouts,
        has_real_tool_use,
        is_broken_agent_format,
        load_successful_rollouts,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    sealed = collect_sealed_task_ids()

    export_from_rollouts(rollouts_jsonl, out_dir / "from_rollouts", sealed_ids=sealed)
    gen = DatasetGenerator("sonec-sft")
    generate_identity_examples(gen, n=12)
    generate_gold_agent_examples(gen, n=gold_n)

    examples = load_successful_rollouts(rollouts_jsonl, sealed_ids=sealed)
    kept = 0
    for i, row in enumerate(examples):
        msgs = row["messages"]
        if is_broken_agent_format(msgs):
            continue
        # Keep Q-only (no tools) or real tool_calls
        if has_real_tool_use(msgs) or not any(m.get("role") == "tool" for m in msgs):
            gen.add_from_messages(
                example_id=str(row.get("id") or f"rollout-{i:05d}"),
                task=str(row.get("task_id") or "rollout"),
                messages=msgs,
                outcome="success",
                metadata={"source": "live_rollout", "reward": row.get("reward")},
            )
            kept += 1

    pipeline = TrainingPipeline(out_dir)
    manifest = gen.manifest()
    jsonl = pipeline.export_jsonl(manifest, filename="train_chat.jsonl")
    mlx_dir = out_dir / "mlx_data"
    mlx_dir.mkdir(parents=True, exist_ok=True)
    mlx_train = mlx_dir / "train.jsonl"
    with mlx_train.open("w", encoding="utf-8") as handle:
        for ex in manifest.examples:
            messages = []
            for s in ex.trajectory:
                msg = {"role": s.role, "content": s.content}
                if s.tool_name:
                    msg["name"] = s.tool_name
                if s.tool_call_id:
                    msg["tool_call_id"] = s.tool_call_id
                if s.tool_calls:
                    msg["tool_calls"] = s.tool_calls
                messages.append(msg)
            handle.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
    lines = mlx_train.read_text(encoding="utf-8").splitlines() if mlx_train.exists() else []
    if not lines:
        raise RuntimeError(
            "SFT corpus empty — need live passing trajectories. "
            "Run: sonec rollout --live --group-size 8 --limit 40"
        )
    # Hold out by hash of line content (task-ish), not a contiguous prefix of train.
    valid_lines: list[str] = []
    train_lines: list[str] = []
    for line in lines:
        bucket = sum(ord(c) for c in line[:64]) % 10
        if bucket == 0 and len(valid_lines) < max(1, len(lines) // 10):
            valid_lines.append(line)
        else:
            train_lines.append(line)
    if not train_lines:
        train_lines = list(lines)
        valid_lines = lines[:1]
    mlx_train.write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    (mlx_dir / "valid.jsonl").write_text("\n".join(valid_lines) + "\n", encoding="utf-8")
    manifest.save(out_dir / "manifest.json")
    pipeline.write_config(model=PRODUCT_MODEL, dataset_file="train_chat.jsonl")
    _write_json(
        out_dir / "corpus_stats.json",
        {
            "examples": len(manifest.examples),
            "gold_n": gold_n,
            "live_kept": kept,
            "mlx_train": str(mlx_train),
            "mlx_train_n": len(train_lines),
            "mlx_valid_n": len(valid_lines),
            "sealed_excluded_n": len(sealed),
            "format": "openai_tool_calls",
            "author": "Suryanshu Nabheet",
        },
    )
    return {"chat": jsonl, "mlx_train": mlx_train, "mlx_dir": mlx_dir}


def resolve_mlx_base(preferred: str | None = None) -> str:
    return preferred or BASE_HF_MLX


def run_mlx_sft(
    *,
    data_dir: Path,
    adapter_path: Path,
    model: str,
    iters: int = 80,
    batch_size: int = 1,
    lora_layers: int = 8,
    learning_rate: float = 1e-5,
) -> TrainReport:
    adapter_path.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python",
        "-m",
        "mlx_lm",
        "lora",
        "--model",
        model,
        "--train",
        "--data",
        str(data_dir),
        "--adapter-path",
        str(adapter_path),
        "--batch-size",
        str(batch_size),
        "--num-layers",
        str(lora_layers),
        "--iters",
        str(iters),
        "--learning-rate",
        str(learning_rate),
        "--steps-per-report",
        "10",
        "--save-every",
        "40",
        "--grad-checkpoint",
        "--max-seq-length",
        "2048",
    ]
    started = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except OSError as exc:
        return TrainReport("sft", False, str(exc), {})
    log_path = adapter_path / "sft_log.txt"
    log_path.write_text((proc.stdout or "") + "\n" + (proc.stderr or ""), encoding="utf-8")
    ok = proc.returncode == 0 and (
        (adapter_path / "adapters.safetensors").exists()
        or any(adapter_path.glob("*.safetensors"))
    )
    return TrainReport(
        phase="sft",
        ok=ok,
        detail=(
            f"iters={iters} model={model} elapsed_s={time.perf_counter() - started:.1f} "
            f"rc={proc.returncode}"
        ),
        paths={"adapter": str(adapter_path), "log": str(log_path)},
    )


def run_rl_rejection_round(
    *,
    out_dir: Path,
    group_size: int = 2,
    limit: int = 8,
    live: bool = False,
    model: str = PRODUCT_MODEL,
) -> TrainReport:
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = build_trainbench_tasks(n=max(limit, 8))[:limit]
    records = run_rollouts_sync(
        tasks,
        out_dir / "rollouts",
        group_size=group_size,
        use_mock=not live,
        provider_name="local",
        model=model,
    )
    by_task: dict[str, list] = {}
    for r in records:
        by_task.setdefault(r.task_id, []).append(r)
    winners = []
    for group in by_task.values():
        best = max(group, key=lambda x: (x.reward, -x.duration_s))
        if best.passed:
            winners.append(best)
    winners_path = out_dir / "winners.jsonl"
    with winners_path.open("w", encoding="utf-8") as handle:
        for w in winners:
            handle.write(json.dumps(w.to_json()) + "\n")
    if winners:
        export_from_rollouts(
            out_dir / "rollouts" / "rollouts.jsonl",
            out_dir / "rft_export",
            sealed_ids=collect_sealed_task_ids(),
        )
    _write_json(
        out_dir / "rl_stats.json",
        {
            "groups": len(by_task),
            "records": len(records),
            "winners": len(winners),
            "pass_rate": sum(1 for r in records if r.passed) / max(len(records), 1),
            "live": live,
        },
    )
    return TrainReport(
        phase="rl_rejection",
        ok=True,
        detail=f"winners={len(winners)}/{len(by_task)} groups",
        paths={"winners": str(winners_path), "stats": str(out_dir / "rl_stats.json")},
    )


def write_product_modelfile(*, adapter_path: Path, modelfile_out: Path) -> Path:
    """Write an explicit non-product note. Modelfile alone is never sonec."""
    from sonec.training.weights import weight_status

    status = weight_status(adapter_path)
    try:
        rel = adapter_path.resolve().relative_to(Path.cwd().resolve())
        adapter_display = str(rel)
    except ValueError:
        adapter_display = str(adapter_path)
    body = f"""# Ollama chat tag (optional runner). Product weights: {adapter_display}
# Ready: {status.ready} — {status.detail}
# Serve specialized weights: sonec serve-llm
# Rebuild: ollama create sonec -f Modelfile
# Chat: ollama run sonec --think=false

FROM {BASE_MODEL}

PARAMETER temperature 0.2
PARAMETER top_p 0.9
PARAMETER num_ctx 32768

SYSTEM \"\"\"{PRODUCT_SYSTEM}
\"\"\"
"""
    modelfile_out.write_text(body, encoding="utf-8")
    return modelfile_out


def run_train_step(
    *,
    root: Path,
    sft_iters: int = 300,
    gold_n: int = 0,
    train_n: int = 40,
    rollout_group: int = 8,
    rl_group: int = 4,
    rl_limit: int = 16,
    skip_sft: bool = False,
    skip_fuel: bool = False,
    live_fuel: bool = True,
    live_rl: bool = True,
    mlx_model: str | None = None,
    reset: bool = False,
    corpus_dir: Path | None = None,
    model: str | None = None,
) -> list[TrainReport]:
    """One specialization step on live passing trajectories (not mock chat)."""
    from sonec.training.weights import weight_status, write_product_manifest

    reports: list[TrainReport] = []
    art = root / "artifacts" / "train"
    if reset and art.exists():
        shutil.rmtree(art)
    art.mkdir(parents=True, exist_ok=True)

    adapter = art / "checkpoints" / "sonec-sft-mlx"
    mlx_base = resolve_mlx_base(mlx_model)
    # mlx_lm / OpenAI servers advertise the base checkpoint id, not the product tag.
    inference_model = model or mlx_base

    if corpus_dir is not None:
        corpus_path = corpus_dir.expanduser().resolve()
        paths = {
            "mlx_dir": corpus_path if (corpus_path / "train.jsonl").exists() else corpus_path / "mlx_data",
            "mlx_train": (
                corpus_path / "train.jsonl"
                if (corpus_path / "train.jsonl").exists()
                else corpus_path / "mlx_data" / "train.jsonl"
            ),
        }
        reports.append(
            TrainReport("corpus", True, f"reused {paths['mlx_dir']}", {k: str(v) for k, v in paths.items()})
        )
    else:
        if not skip_fuel:
            fuel_dir = art / "fuel"
            try:
                rollouts = prepare_rollout_fuel(
                    fuel_dir,
                    group_size=rollout_group,
                    train_n=train_n,
                    live=live_fuel,
                    model=inference_model if live_fuel else None,
                )
                reports.append(
                    TrainReport(
                        "fuel",
                        True,
                        f"live={live_fuel} rollouts={rollouts}",
                        {"rollouts": str(rollouts)},
                    )
                )
            except Exception as exc:  # noqa: BLE001
                reports.append(TrainReport("fuel", False, str(exc), {}))
                return reports
        else:
            rollouts = art / "fuel" / "rollouts.jsonl"
            if not rollouts.exists():
                reports.append(TrainReport("fuel", False, f"missing {rollouts}", {}))
                return reports
            reports.append(TrainReport("fuel", True, f"reused {rollouts}", {"rollouts": str(rollouts)}))

        try:
            corpus_out = art / "sft_corpus"
            paths = assemble_sft_corpus(rollouts_jsonl=rollouts, out_dir=corpus_out, gold_n=gold_n)
            reports.append(
                TrainReport(
                    "corpus",
                    True,
                    f"examples in {paths['mlx_dir']}",
                    {k: str(v) for k, v in paths.items()},
                )
            )
        except Exception as exc:  # noqa: BLE001
            reports.append(TrainReport("corpus", False, str(exc), {}))
            return reports

    if not skip_sft:
        sft_report = run_mlx_sft(
            data_dir=Path(paths["mlx_dir"]),
            adapter_path=adapter,
            model=mlx_base,
            iters=sft_iters,
        )
        status = weight_status(adapter)
        if sft_report.ok and not status.ready:
            sft_report = TrainReport(
                "sft",
                False,
                f"{sft_report.detail}; NO WEIGHTS: {status.detail}",
                sft_report.paths,
            )
        reports.append(sft_report)
    else:
        reports.append(TrainReport("sft", True, "skipped", {}))

    reports.append(
        run_rl_rejection_round(
            out_dir=art / "rl",
            group_size=rl_group,
            limit=rl_limit,
            live=live_rl,
            model=inference_model,
        )
    )

    # Rejection → second SFT (RFT): fuel + RL rollouts densify write-first Python.
    fuel_rollouts = art / "fuel" / "rollouts.jsonl"
    rl_rollouts = art / "rl" / "rollouts" / "rollouts.jsonl"
    winners_path = art / "rl" / "winners.jsonl"
    if (
        not skip_sft
        and corpus_dir is None
        and winners_path.is_file()
        and winners_path.stat().st_size > 0
        and fuel_rollouts.is_file()
    ):
        combined = art / "sft_corpus_rft" / "combined_rollouts.jsonl"
        sources = [p for p in (fuel_rollouts, rl_rollouts) if p.is_file()]
        n_lines = _concat_jsonl(sources, combined)
        if n_lines > 0:
            try:
                rft_paths = assemble_sft_corpus(
                    rollouts_jsonl=combined,
                    out_dir=art / "sft_corpus_rft",
                    gold_n=max(gold_n, 64),
                )
                rft_iters = max(80, sft_iters // 3)
                rft = run_mlx_sft(
                    data_dir=Path(rft_paths["mlx_dir"]),
                    adapter_path=adapter,
                    model=mlx_base,
                    iters=rft_iters,
                )
                reports.append(
                    TrainReport(
                        phase="rft_sft",
                        ok=rft.ok,
                        detail=f"{rft.detail}; combined_lines={n_lines}",
                        paths={**rft.paths, "combined": str(combined)},
                    )
                )
            except Exception as exc:  # noqa: BLE001
                reports.append(TrainReport("rft_sft", False, str(exc), {}))

    manifest = write_product_manifest(adapter_dir=adapter, mlx_base=mlx_base, root=root)
    status = weight_status(adapter)
    reports.append(
        TrainReport(
            "product",
            status.ready,
            status.detail if status.ready else f"not ready — {status.detail}",
            {"manifest": str(manifest), "adapter": str(adapter)},
        )
    )

    # Optional chat Modelfile (runner only; product is the LoRA adapter).
    mf = write_product_modelfile(adapter_path=adapter, modelfile_out=root / "Modelfile")
    reports.append(TrainReport("modelfile", True, f"updated {mf}", {"modelfile": str(mf)}))

    _write_json(
        art / "TRAIN_REPORT.json",
        {
            "product": PRODUCT_MODEL,
            "base": BASE_MODEL,
            "base_hf": BASE_HF,
            "mlx_base": mlx_base,
            "live_fuel": live_fuel,
            "live_rl": live_rl,
            "weights_ready": status.ready,
            "phases": [r.__dict__ for r in reports],
        },
    )
    return reports


# Back-compat alias removed — use run_train_step.
