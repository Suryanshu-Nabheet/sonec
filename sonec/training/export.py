"""Convert graded rollouts / trajectories into trainer-ready shards.

Formats:
- chat_messages (OpenAI-style) — TRL / most trainers
- sharegpt — Axolotl conversation format
- mlx_chat — Apple Silicon mlx-lm finetune JSONL

Agent SFT requires real ``tool_calls`` on assistant turns — never
\"Calling tool\" prose or collapsed JSON dumps.
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


def _normalize_tool_calls(raw: object) -> list[dict[str, Any]] | None:
    if not raw or not isinstance(raw, list):
        return None
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        function = item.get("function") if isinstance(item.get("function"), dict) else item
        name = function.get("name") or item.get("name")
        args = function.get("arguments", item.get("arguments", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"_raw": args}
        if not isinstance(args, dict):
            args = {"value": args}
        if not name:
            continue
        out.append(
            {
                "id": str(item.get("id") or f"call_{len(out)}"),
                "type": "function",
                "function": {"name": str(name), "arguments": args},
            }
        )
    return out or None


def is_broken_agent_format(messages: list[dict[str, Any]]) -> bool:
    """Reject text tool dumps / fake 'Calling tool' trajectories."""
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = str(msg.get("content") or "")
        if content.startswith("Calling tool"):
            return True
        if '"tool_calls"' in content and not msg.get("tool_calls"):
            return True
        # XML-only without structured tool_calls is allowed only if no OpenAI tools expected;
        # for agent SFT we require structured tool_calls whenever tools were used.
        if msg.get("tool_calls"):
            continue
        if "<tool_call>" in content:
            return True
    return False


def has_real_tool_use(messages: list[dict[str, Any]]) -> bool:
    return any(m.get("role") == "assistant" and m.get("tool_calls") for m in messages)


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
        if role not in {"system", "user", "assistant", "tool"}:
            continue
        msg: dict[str, Any] = {"role": role, "content": event.get("content") or ""}
        if event.get("name"):
            msg["name"] = event["name"]
        if event.get("tool_call_id"):
            msg["tool_call_id"] = event["tool_call_id"]
        tool_calls = _normalize_tool_calls(event.get("tool_calls"))
        if tool_calls:
            msg["tool_calls"] = tool_calls
            # Prefer empty content when structured tool_calls exist
            if not (msg["content"] or "").strip():
                msg["content"] = ""
        messages.append(msg)
    return messages


def load_successful_rollouts(
    rollouts_jsonl: Path,
    *,
    sealed_ids: set[str] | None = None,
    min_reward: float = 0.0,
    require_tool_calls: bool = True,
) -> list[dict[str, Any]]:
    """Keep grader-passed trajectories only (reward may be shaped < 1.0)."""
    sealed = sealed_ids or set()
    out: list[dict[str, Any]] = []
    for row in _load_jsonl(rollouts_jsonl):
        if row.get("task_id") in sealed:
            continue
        if not row.get("passed"):
            continue
        if float(row.get("reward", 0)) < min_reward:
            continue
        messages = _trajectory_to_messages(str(row.get("trajectory_path") or ""))
        if len(messages) < 2:
            continue
        if is_broken_agent_format(messages):
            continue
        if require_tool_calls and not has_real_tool_use(messages):
            # Allow Q-only restraint tasks (no tools expected)
            if any(m.get("role") == "tool" for m in messages):
                continue
            # pure text answer — keep if no tool role and no broken format
            pass
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
                    conversations.append({"from": "system", "value": msg.get("content") or ""})
                elif role == "user":
                    conversations.append({"from": "human", "value": msg.get("content") or ""})
                elif role == "assistant":
                    value = msg.get("content") or ""
                    if msg.get("tool_calls"):
                        value = json.dumps(
                            {"content": value, "tool_calls": msg["tool_calls"]},
                            ensure_ascii=False,
                        )
                    conversations.append({"from": "gpt", "value": value})
                elif role == "tool":
                    conversations.append({"from": "tool", "value": msg.get("content") or ""})
            handle.write(
                json.dumps(
                    {"id": ex["id"], "conversations": conversations},
                    ensure_ascii=False,
                )
                + "\n"
            )
    return path


def export_mlx_jsonl(examples: list[dict[str, Any]], path: Path) -> Path:
    """mlx-lm chat finetune format with OpenAI-style tool_calls preserved."""
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
                (m.get("content") or "" for m in ex["messages"] if m["role"] == "user"),
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
        "require_openai_tool_calls": True,
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
