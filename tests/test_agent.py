"""Agent runtime integration test."""

from __future__ import annotations

from pathlib import Path

import pytest
from sonec.app import build_agent
from sonec.core.config import load_settings
from sonec.core.types import Message, Role, ToolCall
from sonec.llm.provider import MockProvider


@pytest.mark.asyncio
async def test_agent_tool_loop(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Demo\n", encoding="utf-8")
    provider = MockProvider(
        [
            Message(
                role=Role.ASSISTANT,
                content=None,
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="fs_write",
                        arguments={"path": "out.txt", "content": "done"},
                    )
                ],
            ),
            Message(role=Role.ASSISTANT, content="Wrote out.txt"),
        ]
    )
    settings = load_settings(workspace=tmp_path, provider="mock", max_iterations=5)
    agent, *_ = build_agent(settings=settings, provider=provider, persist_memory=False)
    result = await agent.run("Write out.txt with done")
    assert result.success
    assert (tmp_path / "out.txt").read_text() == "done"
    assert result.iterations == 2
