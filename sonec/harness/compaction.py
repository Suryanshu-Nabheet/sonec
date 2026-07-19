"""Context compaction — self-summarization for long-horizon runs."""

from __future__ import annotations

from sonec.core.types import Message, Role


def should_compact(messages: list[Message], *, max_messages: int = 40) -> bool:
    return len(messages) > max_messages


def compact_messages(
    messages: list[Message],
    *,
    keep_recent: int = 16,
    goal: str = "",
) -> list[Message]:
    """Replace middle transcript with a summary, keeping system + original user + recent.

    Qwen chat templates raise if no real user query remains — never drop the goal user turn.
    """
    if len(messages) <= keep_recent + 3:
        return messages

    system = messages[0] if messages and messages[0].role == Role.SYSTEM else None
    # First real user message (the goal)
    user_goal: Message | None = None
    for msg in messages:
        if msg.role == Role.USER and (msg.content or "").strip():
            content = (msg.content or "").strip()
            if content.startswith("<tool_response>") and content.endswith("</tool_response>"):
                continue
            user_goal = msg
            break
    if user_goal is None and goal.strip():
        user_goal = Message(role=Role.USER, content=goal.strip())

    head_start = 1 if system else 0
    # Skip original user in the "middle" slice if it sits right after system
    mid_start = head_start
    if user_goal is not None and mid_start < len(messages) and messages[mid_start] is user_goal:
        mid_start += 1
    elif (
        user_goal is not None
        and mid_start < len(messages)
        and messages[mid_start].role == Role.USER
    ):
        mid_start += 1

    recent = messages[-keep_recent:]
    # Avoid duplicating user_goal / system inside recent
    recent = [m for m in recent if m is not system and m is not user_goal]

    middle = messages[mid_start : len(messages) - keep_recent]
    summary_bits: list[str] = []
    for msg in middle:
        if msg.role == Role.TOOL:
            summary_bits.append(f"tool:{msg.name} → {(msg.content or '')[:160]}")
        elif msg.tool_calls:
            names = ",".join(c.name for c in msg.tool_calls)
            summary_bits.append(f"assistant tools={names}")
        elif msg.content:
            summary_bits.append(f"{msg.role.value}: {msg.content[:160]}")
    summary = Message(
        role=Role.USER,
        content=(
            "CONTEXT COMPACTION (harness self-summary):\n"
            + "\n".join(summary_bits[-40:])
            + "\nContinue from the recent messages below."
        ),
    )
    out: list[Message] = []
    if system:
        out.append(system)
    if user_goal is not None:
        out.append(user_goal)
    out.append(summary)
    out.extend(recent)
    return out
