"""Git integration via the system git binary."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sonec.core.types import ToolResult, ToolSpec
from sonec.core.workspace import Workspace
from sonec.terminal.service import TerminalService
from sonec.tools.registry import FunctionTool, Tool, json_content


class GitService:
    def __init__(self, workspace: Workspace, terminal: TerminalService) -> None:
        self.workspace = workspace
        self.terminal = terminal

    def tools(self) -> list[Tool]:
        return [
            FunctionTool(
                ToolSpec(
                    name="git_status",
                    description="Show git status --short in the workspace.",
                    parameters={"type": "object", "properties": {}, "additionalProperties": False},
                ),
                self.status,
            ),
            FunctionTool(
                ToolSpec(
                    name="git_diff",
                    description="Show git diff (optionally staged).",
                    parameters={
                        "type": "object",
                        "properties": {
                            "staged": {"type": "boolean", "default": False},
                            "path": {"type": "string"},
                        },
                        "additionalProperties": False,
                    },
                ),
                self.diff,
            ),
            FunctionTool(
                ToolSpec(
                    name="git_log",
                    description="Show recent git commits.",
                    parameters={
                        "type": "object",
                        "properties": {"limit": {"type": "integer", "default": 10}},
                        "additionalProperties": False,
                    },
                ),
                self.log,
            ),
            FunctionTool(
                ToolSpec(
                    name="git_branch",
                    description="Show current branch and local branches.",
                    parameters={"type": "object", "properties": {}, "additionalProperties": False},
                ),
                self.branch,
            ),
        ]

    async def _git(self, *args: str) -> dict[str, Any]:
        argv = ["git", *args]
        return await self.terminal.run(" ".join(argv), argv=argv)

    async def status(self, arguments: Mapping[str, Any]) -> ToolResult:
        del arguments
        result = await self._git("status", "--short", "--branch")
        return ToolResult(
            tool_call_id="",
            name="git_status",
            content=json_content(result),
            ok=result.get("exit_code") == 0,
            data=result,
        )

    async def diff(self, arguments: Mapping[str, Any]) -> ToolResult:
        args = ["diff"]
        if arguments.get("staged"):
            args.append("--staged")
        path = arguments.get("path")
        if path:
            # Ensure path stays in workspace
            self.workspace.resolve(str(path))
            args.extend(["--", str(path)])
        result = await self._git(*args)
        return ToolResult(
            tool_call_id="",
            name="git_diff",
            content=json_content(result),
            ok=result.get("exit_code") == 0,
            data=result,
        )

    async def log(self, arguments: Mapping[str, Any]) -> ToolResult:
        limit = int(arguments.get("limit") or 10)
        result = await self._git("log", f"-n{limit}", "--oneline", "--decorate")
        return ToolResult(
            tool_call_id="",
            name="git_log",
            content=json_content(result),
            ok=result.get("exit_code") == 0,
            data=result,
        )

    async def branch(self, arguments: Mapping[str, Any]) -> ToolResult:
        del arguments
        result = await self._git("branch", "-vv")
        return ToolResult(
            tool_call_id="",
            name="git_branch",
            content=json_content(result),
            ok=result.get("exit_code") == 0,
            data=result,
        )
