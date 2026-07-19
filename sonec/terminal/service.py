"""Terminal execution with workspace cwd and basic safety."""

from __future__ import annotations

import asyncio
import os
import shlex
import signal
from collections.abc import Mapping, Sequence
from typing import Any

from sonec.core.errors import SecurityError
from sonec.core.types import ToolResult, ToolSpec
from sonec.core.workspace import Workspace
from sonec.tools.registry import FunctionTool, Tool, json_content

BLOCKED_PATTERNS = (
    "rm -rf /",
    "rm -rf/*",
    "rm -rf ~",
    "rm -rf~",
    "mkfs",
    ":(){",
    "shutdown",
    "reboot",
    "dd if=",
    "mkfifo",
    "> /dev/",
    "chmod 777 /",
)


class TerminalService:
    def __init__(
        self,
        workspace: Workspace,
        *,
        timeout_s: float = 60.0,
        allow_network: bool = False,
    ) -> None:
        self.workspace = workspace
        self.timeout_s = timeout_s
        self.allow_network = allow_network

    def tools(self) -> list[Tool]:
        return [
            FunctionTool(
                ToolSpec(
                    name="terminal_run",
                    description=(
                        "Run a shell command inside the workspace. "
                        "Prefer non-interactive commands. Output is truncated."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "command": {"type": "string"},
                            "timeout_s": {"type": "number"},
                        },
                        "required": ["command"],
                        "additionalProperties": False,
                    },
                ),
                self.run_tool,
            )
        ]

    def _validate_command(self, command: str) -> None:
        lowered = " ".join(command.lower().split())
        for pattern in BLOCKED_PATTERNS:
            if pattern in lowered:
                raise SecurityError(f"Blocked dangerous command pattern: {pattern}")
        if not self.allow_network:
            network_tokens = (
                "curl ",
                "curl\t",
                "wget ",
                "wget\t",
                "nc ",
                "ncat ",
                "ssh ",
                "scp ",
                "sftp ",
                "/usr/bin/curl",
                "/usr/bin/wget",
            )
            for token in network_tokens:
                if token in lowered:
                    raise SecurityError(
                        f"Network command blocked ({token.strip()}). "
                        "Enable allow_network_tools to permit."
                    )

    async def _kill_process_tree(self, process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        try:
            if process.pid:
                os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                process.kill()
            except ProcessLookupError:
                return
        try:
            await process.communicate()
        except Exception:  # noqa: BLE001
            return

    async def run(
        self,
        command: str,
        *,
        timeout_s: float | None = None,
        argv: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        self._validate_command(command if argv is None else " ".join(argv))
        timeout = timeout_s or self.timeout_s
        if argv is None:
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=str(self.workspace.root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        else:
            process = await asyncio.create_subprocess_exec(
                *argv,
                cwd=str(self.workspace.root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except TimeoutError:
            await self._kill_process_tree(process)
            return {
                "command": command,
                "exit_code": None,
                "timed_out": True,
                "stdout": "",
                "stderr": f"Command timed out after {timeout}s",
            }
        stdout = stdout_b.decode("utf-8", errors="replace")[-20_000:]
        stderr = stderr_b.decode("utf-8", errors="replace")[-20_000:]
        return {
            "command": command,
            "exit_code": process.returncode,
            "timed_out": False,
            "stdout": stdout,
            "stderr": stderr,
        }

    async def run_tool(self, arguments: Mapping[str, Any]) -> ToolResult:
        command = str(arguments["command"])
        timeout = arguments.get("timeout_s")
        result = await self.run(command, timeout_s=float(timeout) if timeout else None)
        ok = result["exit_code"] == 0 and not result["timed_out"]
        return ToolResult(
            tool_call_id="",
            name="terminal_run",
            content=json_content(result),
            ok=ok,
            data=result,
        )

    @staticmethod
    def split(command: str) -> list[str]:
        return shlex.split(command)
