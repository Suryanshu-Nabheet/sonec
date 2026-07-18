"""Tests for frozen Phase-0 harness identity."""

from __future__ import annotations

from pathlib import Path

import pytest

from sonec.app import build_runtime
from sonec.core.config import load_settings
from sonec.core.types import Message, Role, ToolCall
from sonec.harness.compaction import compact_messages
from sonec.harness.context import ContextAssembler, MAX_ALWAYS_ON_CHARS
from sonec.harness.versioning import HARNESS_VERSION, tool_schema_hash
from sonec.llm.provider import MockProvider


def test_thin_prompt_budget() -> None:
    prompt = ContextAssembler().build_system_prompt("fix a bug")
    assert len(prompt) <= MAX_ALWAYS_ON_CHARS
    assert "sonec" in prompt.lower()


def test_compaction() -> None:
    msgs = [Message(role=Role.SYSTEM, content="sys")]
    for i in range(30):
        msgs.append(Message(role=Role.USER, content=f"u{i}"))
        msgs.append(Message(role=Role.ASSISTANT, content=f"a{i}"))
    out = compact_messages(msgs, keep_recent=8)
    assert len(out) < len(msgs)
    assert any("COMPACTION" in (m.content or "") for m in out)


@pytest.mark.asyncio
async def test_cli_eval_same_runtime(tmp_path: Path) -> None:
    settings = load_settings(workspace=tmp_path, provider="mock")
    provider = MockProvider(
        [
            Message(
                role=Role.ASSISTANT,
                content=None,
                tool_calls=[
                    ToolCall(
                        id="1",
                        name="fs_write",
                        arguments={"path": "a.txt", "content": "x"},
                    )
                ],
            ),
            Message(role=Role.ASSISTANT, content="done"),
        ]
    )
    a, *_ = build_runtime(
        settings=settings, provider=provider, persist_memory=False, log_dir=tmp_path / "logs"
    )
    r1 = await a.run("write a.txt")
    assert r1.harness_version == HARNESS_VERSION
    assert r1.tool_schema_hash
    assert r1.completed
    logs = list((tmp_path / "logs").glob("*.jsonl"))
    assert logs
    text = logs[0].read_text(encoding="utf-8")
    assert "harness_version" in text
    assert "tool_schema_hash" in text


def test_tool_schema_hash_stable(tmp_path: Path) -> None:
    settings = load_settings(workspace=tmp_path, provider="mock")
    r1, _, _, reg1 = build_runtime(
        settings=settings, provider=MockProvider(), persist_memory=False, log_dir=tmp_path / "l1"
    )
    r2, _, _, reg2 = build_runtime(
        settings=settings, provider=MockProvider(), persist_memory=False, log_dir=tmp_path / "l2"
    )
    assert r1.tool_hash == r2.tool_hash
    assert tool_schema_hash(reg1.specs()) == tool_schema_hash(reg2.specs())
