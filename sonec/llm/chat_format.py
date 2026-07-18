"""Normalize chat messages for Qwen / mlx_lm chat templates.

Qwen 3.5 templates raise ``No user query found in messages`` when no
non-tool_response user turn exists, and require tool ``arguments`` to be a
mapping (not a JSON string) when rendering history.
"""

from __future__ import annotations

import json
from typing import Any

from sonec.core.types import Message, Role, ToolCall


def ensure_user_present(messages: list[Message], *, fallback_user: str) -> list[Message]:
    """Guarantee at least one real user turn (not only system / tool)."""
    has_user = any(
        m.role == Role.USER
        and (m.content or "").strip()
        and not (
            (m.content or "").strip().startswith("<tool_response>")
            and (m.content or "").strip().endswith("</tool_response>")
        )
        for m in messages
    )
    if has_user:
        return messages
    out = list(messages)
    # Insert after leading system message(s).
    insert_at = 0
    while insert_at < len(out) and out[insert_at].role == Role.SYSTEM:
        insert_at += 1
    out.insert(
        insert_at,
        Message(role=Role.USER, content=fallback_user.strip() or "Continue the task."),
    )
    return out


def message_to_openai(message: Message) -> dict[str, Any]:
    """Wire format that Qwen chat templates accept."""
    payload: dict[str, Any] = {
        "role": message.role.value,
        "content": message.content if message.content is not None else "",
    }
    if message.name:
        payload["name"] = message.name
    if message.tool_call_id:
        payload["tool_call_id"] = message.tool_call_id
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    # Dict, not JSON string — Qwen template uses |items
                    "arguments": call.arguments if isinstance(call.arguments, dict) else {},
                },
            }
            for call in message.tool_calls
        ]
    return payload


def openai_tool_calls_from_dict(raw: list[dict[str, Any]] | None) -> list[ToolCall] | None:
    if not raw:
        return None
    out: list[ToolCall] = []
    for item in raw:
        function = item.get("function") or item
        args = function.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"_raw": args}
        if not isinstance(args, dict):
            args = {"value": args}
        out.append(
            ToolCall(
                id=str(item.get("id") or f"call_{len(out)}"),
                name=str(function.get("name") or ""),
                arguments=args,
            )
        )
    return out or None
