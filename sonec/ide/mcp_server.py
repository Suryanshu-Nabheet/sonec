"""MCP stdio server — embed sonec into IDE hosts.

Implements a minimal Model Context Protocol subset over newline-delimited
JSON-RPC without requiring the `mcp` package.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from sonec import __version__
from sonec.app import build_runtime
from sonec.core.config import load_settings
from sonec.harness.versioning import HARNESS_VERSION

TOOLS = [
    {
        "name": "sonec_run",
        "description": (
            "Run a software-engineering goal in a workspace with sonec. "
            "Returns the final message and harness metadata."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "goal": {"type": "string"},
                "workspace": {"type": "string", "description": "Absolute workspace path"},
                "provider": {
                    "type": "string",
                    "description": "local|openai|openai_compatible|mock",
                },
                "model": {"type": "string"},
                "base_url": {
                    "type": "string",
                    "description": "OpenAI-compatible root including /v1",
                },
            },
            "required": ["goal"],
        },
    },
    {
        "name": "sonec_version",
        "description": "Return sonec package + harness versions.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def _reply(msg_id: Any, result: Any) -> None:
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": msg_id, "result": result}) + "\n")
    sys.stdout.flush()


def _error(msg_id: Any, code: int, message: str) -> None:
    sys.stdout.write(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": code, "message": message},
            }
        )
        + "\n"
    )
    sys.stdout.flush()


async def _sonec_run(
    goal: str,
    workspace: str | None,
    *,
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    ws = Path(workspace or Path.cwd()).expanduser().resolve()
    overrides: dict[str, object] = {"workspace": ws}
    if provider:
        overrides["provider"] = provider
    if model:
        overrides["model"] = model
    if base_url:
        overrides["base_url"] = base_url
    settings = load_settings(**overrides)
    runtime, *_ = build_runtime(
        settings=settings,
        persist_memory=False,
        log_dir=ws / ".sonec" / "trajectories",
        goal_for_prompt=goal,
    )
    result = await runtime.run(goal)
    return {
        "final_message": result.final_message,
        "completed": result.completed,
        "success": result.success,
        "iterations": result.iterations,
        "harness_version": result.harness_version,
        "tool_schema_hash": result.tool_schema_hash,
        "model_id": result.model_id,
        "run_id": result.run_id,
    }


def handle_message(msg: dict[str, Any]) -> None:
    method = msg.get("method")
    msg_id = msg.get("id")
    params = msg.get("params") or {}

    if method == "initialize":
        _reply(
            msg_id,
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": "sonec",
                    "version": __version__,
                    "author": "Suryanshu Nabheet",
                },
            },
        )
        return
    if method == "notifications/initialized":
        return
    if method == "ping":
        _reply(msg_id, {})
        return
    if method == "tools/list":
        _reply(msg_id, {"tools": TOOLS})
        return
    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments") or {}
        if name == "sonec_version":
            _reply(
                msg_id,
                {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(
                                {
                                    "version": __version__,
                                    "harness_version": HARNESS_VERSION,
                                    "author": "Suryanshu Nabheet",
                                }
                            ),
                        }
                    ]
                },
            )
            return
        if name == "sonec_run":
            goal = str(args.get("goal") or "")
            if not goal:
                _error(msg_id, -32602, "goal required")
                return
            try:
                result = asyncio.run(
                    _sonec_run(
                        goal,
                        args.get("workspace"),
                        provider=args.get("provider"),
                        model=args.get("model"),
                        base_url=args.get("base_url"),
                    )
                )
                _reply(
                    msg_id,
                    {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]},
                )
            except Exception as exc:  # noqa: BLE001
                _error(msg_id, -32000, str(exc))
            return
        _error(msg_id, -32601, f"unknown tool {name}")
        return
    if msg_id is not None:
        _error(msg_id, -32601, f"unknown method {method}")


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        handle_message(msg)


if __name__ == "__main__":
    main()
