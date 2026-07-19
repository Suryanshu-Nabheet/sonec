"""Context compaction — self-summarization for long-horizon runs."""

from __future__ import annotations

from sonec.core.types import Message, Role


def should_compact(messages: list[Message], *, max_messages: int = 40) -> bool:
    return len(messages) > max_messages


def _align_recent_start(messages: list[Message], keep_recent: int) -> int:
    """Start of the recent window; never cut between assistant tool_calls and tool results."""
    start = max(0, len(messages) - keep_recent)
    # If we land on a tool result, walk back to include its assistant tool_calls turn.
    while start > 0 and messages[start].role == Role.TOOL:
        start -= 1
    # If the first recent message has tool_calls, keep extending until all tool results follow.
    if start < len(messages) and messages[start].tool_calls:
        needed = {c.id for c in messages[start].tool_calls if c.id}
        seen: set[str] = set()
        i = start + 1
        while i < len(messages) and messages[i].role == Role.TOOL:
            tid = messages[i].tool_call_id or ""
            if tid:
                seen.add(tid)
            i += 1
            if needed and needed <= seen:
                break
    return start


def compact_messages(
    messages: list[Message],
    *,
    keep_recent: int = 16,
    goal: str = "",
) -> list[Message]:
    """Replace middle transcript with a summary, keeping system + original user + recent.

    Qwen chat templates raise if no real user query remains — never drop the goal user turn.
    Never leave orphaned tool_calls without their tool results in the recent window.
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
            if content.startswith("CONTEXT COMPACTION"):
                continue
            if content.startswith("Guidance:"):
                continue
            user_goal = msg
            break
    if user_goal is None and goal.strip():
        user_goal = Message(role=Role.USER, content=goal.strip())

    head_start = 1 if system else 0
    mid_start = head_start
    if user_goal is not None and mid_start < len(messages):
        at = messages[mid_start]
        if at is user_goal or at.role == Role.USER:
            mid_start += 1

    recent_start = _align_recent_start(messages, keep_recent)
    recent = messages[recent_start:]
    # Avoid duplicating user_goal / system inside recent
    recent = [m for m in recent if m is not system and m is not user_goal]

    middle = messages[mid_start:recent_start]
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
