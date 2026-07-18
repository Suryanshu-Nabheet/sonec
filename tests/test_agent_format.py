"""Chat format + export guards for agent SFT."""

from __future__ import annotations

from sonec.core.types import Message, Role, ToolCall
from sonec.harness.compaction import compact_messages
from sonec.llm.chat_format import ensure_user_present, message_to_openai
from sonec.training.export import is_broken_agent_format


def test_message_to_openai_args_are_dict() -> None:
    msg = Message(
        role=Role.ASSISTANT,
        content="",
        tool_calls=[ToolCall(id="c1", name="fs_write", arguments={"path": "a.txt"})],
    )
    payload = message_to_openai(msg)
    args = payload["tool_calls"][0]["function"]["arguments"]
    assert isinstance(args, dict)
    assert args["path"] == "a.txt"


def test_ensure_user_present_recovers_missing_user() -> None:
    msgs = [
        Message(role=Role.SYSTEM, content="sys"),
        Message(role=Role.ASSISTANT, content="hi"),
    ]
    out = ensure_user_present(msgs, fallback_user="Create notes/a.txt")
    assert any(m.role == Role.USER and "notes/a.txt" in (m.content or "") for m in out)


def test_compact_keeps_user_goal() -> None:
    msgs = [Message(role=Role.SYSTEM, content="sys")]
    msgs.append(Message(role=Role.USER, content="Create notes/hello.txt"))
    for i in range(30):
        msgs.append(Message(role=Role.ASSISTANT, content=f"step {i}"))
        msgs.append(Message(role=Role.TOOL, content=f"ok {i}", name="fs_list", tool_call_id=f"c{i}"))
    out = compact_messages(msgs, keep_recent=8, goal="Create notes/hello.txt")
    assert any(m.role == Role.USER and "notes/hello.txt" in (m.content or "") for m in out)


def test_reject_calling_tool_text() -> None:
    assert is_broken_agent_format(
        [
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": "Calling tool fs_write"},
        ]
    )
    assert not is_broken_agent_format(
        [
            {"role": "user", "content": "x"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "fs_write", "arguments": {"path": "a"}},
                    }
                ],
            },
        ]
    )
