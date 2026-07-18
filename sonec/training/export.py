"""Convert graded rollouts / trajectories into trainer-ready shards.

Formats:
- chat_messages (OpenAI-style) — TRL / most trainers
- sharegpt — Axolotl conversation format
- mlx_chat — Apple Silicon mlx-lm finetune JSONL
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from sonec.harness.versioning import HARNESS_VERSION


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _trajectory_to_messages(traj_path: str) -> list[dict[str, Any]]:
    path = Path(traj_path)
    if not path.exists():
        return []
    messages: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("type") != "message":
            continue
        role = event.get("role")
        content = event.get("content")
        if role not in {"system", "user", "assistant", "tool"}:
            continue
        # Prefer textual content; include tool call summary if content empty.
        if content is None and event.get("tool_calls"):
            content = json.dumps({"tool_calls": event["tool_calls"]})
        if content is None:
            continue
        messages.append({"role": role, "content": str(content)})
    return messages


def load_successful_rollouts(
    rollouts_jsonl: Path,
    *,
    sealed_ids: set[str] | None = None,
    min_reward: float = 1.0,
) -> list[dict[str, Any]]:
    sealed = sealed_ids or set()
    out: list[dict[str, Any]] = []
    for row in _load_jsonl(rollouts_jsonl):
        if row.get("task_id") in sealed:
            continue
        if float(row.get("reward", 0)) < min_reward:
            continue
        if not row.get("passed"):
            continue
        messages = _trajectory_to_messages(str(row.get("trajectory_path") or ""))
        if len(messages) < 2:
            # Fallback: synthesize a minimal successful trajectory from prompt
            messages = [
                {
                    "role": "system",
                    "content": (
                        f"You are sonec v{HARNESS_VERSION}, a coding-specialist agent. "
                        "Use tools. Verify before done."
                    ),
                },
                {"role": "user", "content": str(row.get("prompt") or "")},
                {
                    "role": "assistant",
                    "content": "Task completed with environment verification evidence.",
                },
            ]
        out.append(
            {
                "id": f"{row.get('task_id')}_{row.get('rollout_index')}",
                "task_id": row.get("task_id"),
                "messages": messages,
                "reward": row.get("reward"),
                "harness_version": row.get("harness_version"),
                "tool_schema_hash": row.get("tool_schema_hash"),
                "model_id": row.get("model_id"),
            }
        )
    return out


def export_chat_jsonl(examples: list[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for ex in examples:
            handle.write(json.dumps(ex, ensure_ascii=False) + "\n")
    return path


def export_sharegpt_jsonl(examples: list[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for ex in examples:
            conversations = []
            for msg in ex["messages"]:
                role = msg["role"]
                if role == "system":
                    conversations.append({"from": "system", "value": msg["content"]})
                elif role == "user":
                    conversations.append({"from": "human", "value": msg["content"]})
                elif role == "assistant":
                    conversations.append({"from": "gpt", "value": msg["content"]})
                elif role == "tool":
                    conversations.append(
                        {"from": "tool", "value": msg["content"]}
                    )
            handle.write(
                json.dumps(
                    {"id": ex["id"], "conversations": conversations},
                    ensure_ascii=False,
                )
                + "\n"
            )
    return path


def export_mlx_jsonl(examples: list[dict[str, Any]], path: Path) -> Path:
    """mlx-lm chat finetune format: {\"messages\": [...]} per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for ex in examples:
            handle.write(
                json.dumps({"messages": ex["messages"]}, ensure_ascii=False) + "\n"
            )
    return path


def export_grpo_prompts(examples: list[dict[str, Any]], path: Path) -> Path:
    """Prompt-only shard for group-relative RL (one prompt per task)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    with path.open("w", encoding="utf-8") as handle:
        for ex in examples:
            tid = str(ex.get("task_id") or ex["id"])
            if tid in seen:
                continue
            seen.add(tid)
            user = next(
                (m["content"] for m in ex["messages"] if m["role"] == "user"),
                "",
            )
            handle.write(
                json.dumps(
                    {
                        "task_id": tid,
                        "prompt": user,
                        "harness_version": ex.get("harness_version", HARNESS_VERSION),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return path


FormatName = Literal["chat", "sharegpt", "mlx", "grpo_prompts"]


def export_from_rollouts(
    rollouts_jsonl: Path,
    output_dir: Path,
    *,
    formats: list[FormatName] | None = None,
    sealed_ids: set[str] | None = None,
) -> dict[str, Path]:
    formats = formats or ["chat", "sharegpt", "mlx", "grpo_prompts"]
    examples = load_successful_rollouts(rollouts_jsonl, sealed_ids=sealed_ids)
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "count": len(examples),
        "harness_version": HARNESS_VERSION,
        "source": str(rollouts_jsonl),
        "sealed_excluded": sorted(sealed_ids or []),
    }
    (output_dir / "manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    written: dict[str, Path] = {}
    if "chat" in formats:
        written["chat"] = export_chat_jsonl(examples, output_dir / "train_chat.jsonl")
    if "sharegpt" in formats:
        written["sharegpt"] = export_sharegpt_jsonl(
            examples, output_dir / "train_sharegpt.jsonl"
        )
    if "mlx" in formats:
        written["mlx"] = export_mlx_jsonl(examples, output_dir / "train_mlx.jsonl")
    if "grpo_prompts" in formats:
        written["grpo_prompts"] = export_grpo_prompts(
            examples, output_dir / "grpo_prompts.jsonl"
        )
    return written
