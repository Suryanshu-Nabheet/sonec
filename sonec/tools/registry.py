"""Tool protocol and registry."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Protocol

from sonec.core.errors import ToolError, ValidationError
from sonec.core.types import ToolResult, ToolSpec


class Tool(Protocol):
    @property
    def spec(self) -> ToolSpec: ...

    async def run(self, arguments: Mapping[str, Any]) -> ToolResult: ...


ToolHandler = Callable[[Mapping[str, Any]], Awaitable[ToolResult]]


class FunctionTool:
    """Adapter that turns an async function + ToolSpec into a Tool."""

    def __init__(self, spec: ToolSpec, handler: ToolHandler) -> None:
        self._spec = spec
        self._handler = handler

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    async def run(self, arguments: Mapping[str, Any]) -> ToolResult:
        return await self._handler(arguments)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        name = tool.spec.name
        if not name:
            raise ValidationError("Tool name must be non-empty")
        if name in self._tools:
            raise ValidationError(f"Tool already registered: {name}")
        self._tools[name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolError(f"Unknown tool: {name}", tool=name) from exc

    def specs(self) -> list[ToolSpec]:
        return [tool.spec for tool in self._tools.values()]

    def names(self) -> list[str]:
        return sorted(self._tools)

    def restrict(self, allow: set[str]) -> ToolRegistry:
        """Return a new registry with only the named tools (for small-model eval)."""
        narrowed = ToolRegistry()
        for name in sorted(allow):
            if name in self._tools:
                narrowed.register(self._tools[name])
        return narrowed

    async def execute(
        self,
        *,
        tool_call_id: str,
        name: str,
        arguments: Mapping[str, Any],
    ) -> ToolResult:
        tool = self.get(name)
        try:
            result = await tool.run(arguments)
        except Exception as exc:  # noqa: BLE001 — surface tool failures to the model
            return ToolResult(
                tool_call_id=tool_call_id,
                name=name,
                content=f"Tool error: {exc}",
                ok=False,
                data={"error": str(exc), "type": type(exc).__name__},
            )
        return ToolResult(
            tool_call_id=tool_call_id,
            name=name,
            content=result.content,
            ok=result.ok,
            data=result.data,
        )


def json_content(data: object) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False, default=str)
