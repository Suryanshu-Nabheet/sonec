"""LLM provider unit tests."""

from __future__ import annotations

import json

import httpx
import pytest
from sonec.core.types import CompletionRequest, Message, Role, ToolSpec
from sonec.llm.provider import MockProvider, OpenAICompatibleProvider


@pytest.mark.asyncio
async def test_mock_provider_script() -> None:
    provider = MockProvider([Message(role=Role.ASSISTANT, content="hello")])
    response = await provider.complete(CompletionRequest(messages=[]))
    assert response.message.content == "hello"


@pytest.mark.asyncio
async def test_openai_compatible_parses_tool_calls() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "fs_read",
                                    "arguments": json.dumps({"path": "a.py"}),
                                },
                            }
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2},
        }
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    provider = OpenAICompatibleProvider(
        api_key="test",
        base_url="https://example.test/v1",
        model="sonec",
        client=client,
    )
    response = await provider.complete(
        CompletionRequest(
            messages=[Message(role=Role.USER, content="read")],
            tools=[ToolSpec(name="fs_read", description="read")],
        )
    )
    assert response.message.tool_calls
    assert response.message.tool_calls[0].name == "fs_read"
    assert response.message.tool_calls[0].arguments["path"] == "a.py"
    await provider.aclose()
