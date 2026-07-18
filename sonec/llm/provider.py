"""LLM provider abstractions and implementations."""

from __future__ import annotations

import json
from typing import Any, Protocol, runtime_checkable

import httpx

from sonec.core.config import Settings
from sonec.core.errors import LLMError
from sonec.core.types import (
    CompletionRequest,
    CompletionResponse,
    Message,
    Role,
    ToolCall,
)


@runtime_checkable
class LLMProvider(Protocol):
    async def complete(self, request: CompletionRequest) -> CompletionResponse: ...


def _message_to_openai(message: Message) -> dict[str, Any]:
    payload: dict[str, Any] = {"role": message.role.value}
    # mlx_lm / Qwen chat templates require content keys on every turn.
    payload["content"] = message.content if message.content is not None else ""
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
                    "arguments": json.dumps(call.arguments),
                },
            }
            for call in message.tool_calls
        ]
    # Do not forward reasoning into the next request — mlx_lm rejects odd shapes.
    return payload


def _parse_tool_arguments(raw: str) -> dict[str, Any]:
    if not raw or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"_raw": raw}
    if isinstance(parsed, dict):
        return parsed
    return {"value": parsed}


class OpenAICompatibleProvider:
    """HTTP client for OpenAI-compatible chat completions (local runners / vLLM / OpenAI)."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 8192,
        timeout_s: float = 120.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self._client = client
        self._owns_client = client is None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout_s)
        return self._client

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        model = request.model or self.model
        body: dict[str, Any] = {
            "model": model,
            "messages": [_message_to_openai(m) for m in request.messages],
            "temperature": (
                request.temperature if request.temperature is not None else self.temperature
            ),
            "max_tokens": request.max_tokens or self.max_tokens,
        }
        if request.tools:
            body["tools"] = [tool.to_openai() for tool in request.tools]
            body["tool_choice"] = "auto"
        if request.stop:
            body["stop"] = request.stop

        client = await self._get_client()
        last_error: Exception | None = None
        response: httpx.Response | None = None
        for attempt in range(3):
            try:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=body,
                )
            except httpx.HTTPError as exc:
                last_error = exc
                continue
            if response.status_code == 404 and "No user query" in response.text:
                # mlx_lm intermittent template glitch — retry once or twice
                last_error = LLMError(
                    f"LLM HTTP {response.status_code}: {response.text[:500]}",
                    status_code=response.status_code,
                )
                continue
            break
        else:
            if last_error:
                raise LLMError(f"LLM request failed: {last_error}") from last_error
            raise LLMError("LLM request failed after retries")

        assert response is not None
        if response.status_code >= 400:
            raise LLMError(
                f"LLM HTTP {response.status_code}: {response.text[:500]}",
                status_code=response.status_code,
            )

        data = response.json()
        try:
            choice = data["choices"][0]
            raw_message = choice["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected LLM response shape: {data!r}") from exc

        tool_calls: list[ToolCall] | None = None
        if raw_message.get("tool_calls"):
            tool_calls = []
            for item in raw_message["tool_calls"]:
                function = item.get("function", {})
                tool_calls.append(
                    ToolCall(
                        id=item.get("id") or f"call_{len(tool_calls)}",
                        name=function.get("name", ""),
                        arguments=_parse_tool_arguments(function.get("arguments", "{}")),
                    )
                )

        message = Message(
            role=Role.ASSISTANT,
            content=raw_message.get("content"),
            tool_calls=tool_calls,
            reasoning_content=raw_message.get("reasoning_content")
            or raw_message.get("reasoning"),
        )
        usage = data.get("usage") or {}
        return CompletionResponse(
            message=message,
            finish_reason=choice.get("finish_reason") or "stop",
            usage={k: int(v) for k, v in usage.items() if isinstance(v, (int, float))},
            raw=data,
        )


class MockProvider:
    """Deterministic provider for tests and offline demos."""

    def __init__(
        self,
        scripted: list[Message] | None = None,
        *,
        default: Message | None = None,
    ) -> None:
        self._scripted = list(scripted or [])
        self.default = default
        self.calls: list[CompletionRequest] = []

    def enqueue(self, message: Message) -> None:
        self._scripted.append(message)

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.calls.append(request)
        if self._scripted:
            message = self._scripted.pop(0)
        elif self.default is not None:
            message = self.default.model_copy(deep=True)
        else:
            message = Message(
                role=Role.ASSISTANT,
                content="Mock provider has no further scripted responses.",
            )
        return CompletionResponse(message=message, finish_reason="stop")

    @classmethod
    def harness_smoke(cls, goal: str) -> MockProvider:
        """Scripted offline run using training-critical tools only."""
        return cls(
            [
                Message(
                    role=Role.ASSISTANT,
                    content=None,
                    tool_calls=[ToolCall(id="c1", name="index_build", arguments={})],
                ),
                Message(
                    role=Role.ASSISTANT,
                    content=(
                        f"Indexed the repo for goal: {goal}. "
                        "Verification: index_build returned file inventory. Done."
                    ),
                ),
            ],
            default=Message(role=Role.ASSISTANT, content="Mock run complete."),
        )


def create_provider(settings: Settings) -> LLMProvider:
    if settings.provider == "mock":
        return MockProvider()
    return OpenAICompatibleProvider(
        api_key=settings.require_api_key(),
        base_url=settings.resolved_base_url(),
        model=settings.model,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        timeout_s=settings.request_timeout_s,
    )
