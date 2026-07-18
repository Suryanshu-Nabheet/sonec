"""Context compaction — self-summarization for long-horizon runs."""

from __future__ import annotations

from sonec.core.types import Message, Role


def should_compact(messages: list[Message], *, max_messages: int = 40) -> bool:
    return len(messages) > max_messages


def compact_messages(
    messages: list[Message],
    *,
    keep_recent: int = 16,
) -> list[Message]:
    """Replace middle transcript with a single summary system note.

    Keeps the first system message and the most recent turns intact.
    """
    if len(messages) <= keep_recent + 2:
        return messages
    system = messages[0] if messages and messages[0].role == Role.SYSTEM else None
    head_start = 1 if system else 0
    recent = messages[-keep_recent:]
    middle = messages[head_start : len(messages) - keep_recent]
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
        role=Role.SYSTEM,
        content=(
            "CONTEXT COMPACTION (harness self-summary):\n"
            + "\n".join(summary_bits[-40:])
            + "\nContinue from the recent messages below."
        ),
    )
    out: list[Message] = []
    if system:
        out.append(system)
    out.append(summary)
    out.extend(recent)
    return out
