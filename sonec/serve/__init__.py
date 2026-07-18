"""Local OpenAI-compatible agent gateway for IDE / CLI embedding.

Exposes:
  POST /v1/chat/completions  — optional passthrough to base LLM
  POST /v1/agent/run         — run frozen sonec harness on a goal
  GET  /healthz              — readiness

This is the productization surface for Cursor/VS Code/Continue adapters.
"""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from sonec import __version__
from sonec.app import build_runtime
from sonec.core.config import load_settings
from sonec.harness.versioning import HARNESS_VERSION
from sonec.models import DEFAULT_MODEL, DEFAULT_PROVIDER


class _Handler(BaseHTTPRequestHandler):
    settings_overrides: dict[str, Any] = {}
    workspace: Path = Path.cwd()

    def log_message(self, fmt: str, *args: object) -> None:  # noqa: A003
        return

    def _json(self, code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or "0")
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8") or "{}")

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in {"/healthz", "/v1/health"}:
            self._json(
                200,
                {
                    "ok": True,
                    "name": "sonec",
                    "version": __version__,
                    "harness_version": HARNESS_VERSION,
                },
            )
            return
        self._json(404, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        data = self._read_json()
        if path == "/v1/agent/run":
            goal = str(data.get("goal") or data.get("prompt") or "").strip()
            if not goal:
                self._json(400, {"error": "goal required"})
                return
            workspace = Path(data.get("workspace") or self.workspace).expanduser().resolve()
            result = asyncio.run(self._run_agent(goal, workspace))
            self._json(200, result)
            return
        if path == "/v1/chat/completions":
            # Thin agent mode: last user message becomes the goal.
            messages = data.get("messages") or []
            goal = ""
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    goal = str(msg.get("content") or "")
                    break
            if not goal:
                self._json(400, {"error": "no user message"})
                return
            result = asyncio.run(self._run_agent(goal, self.workspace))
            self._json(
                200,
                {
                    "id": result.get("run_id"),
                    "object": "chat.completion",
                    "model": f"sonec-agent/{result.get('model_id')}",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": result.get("final_message"),
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "sonec": {
                        "completed": result.get("completed"),
                        "success": result.get("success"),
                        "harness_version": result.get("harness_version"),
                        "tool_schema_hash": result.get("tool_schema_hash"),
                        "iterations": result.get("iterations"),
                    },
                },
            )
            return
        self._json(404, {"error": "not_found"})

    async def _run_agent(self, goal: str, workspace: Path) -> dict[str, Any]:
        overrides = dict(self.settings_overrides)
        overrides["workspace"] = workspace
        settings = load_settings(**overrides)
        runtime, *_ = build_runtime(
            settings=settings,
            persist_memory=False,
            log_dir=workspace / ".sonec" / "trajectories",
            goal_for_prompt=goal,
        )
        result = await runtime.run(goal)
        return {
            "run_id": result.run_id,
            "goal": result.goal,
            "final_message": result.final_message,
            "completed": result.completed,
            "success": result.success,
            "iterations": result.iterations,
            "harness_version": result.harness_version,
            "tool_schema_hash": result.tool_schema_hash,
            "model_id": result.model_id,
            "usage": result.usage,
        }


def serve(
    *,
    host: str = "127.0.0.1",
    port: int = 8787,
    workspace: Path | None = None,
    provider: str = DEFAULT_PROVIDER,
    model: str = DEFAULT_MODEL,
) -> ThreadingHTTPServer:
    _Handler.workspace = (workspace or Path.cwd()).expanduser().resolve()
    _Handler.settings_overrides = {"provider": provider, "model": model}
    server = ThreadingHTTPServer((host, port), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def serve_blocking(
    *,
    host: str = "127.0.0.1",
    port: int = 8787,
    workspace: Path | None = None,
    provider: str = DEFAULT_PROVIDER,
    model: str = DEFAULT_MODEL,
) -> None:
    _Handler.workspace = (workspace or Path.cwd()).expanduser().resolve()
    _Handler.settings_overrides = {"provider": provider, "model": model}
    server = ThreadingHTTPServer((host, port), _Handler)
    print(
        json.dumps(
            {
                "serving": f"http://{host}:{port}",
                "health": f"http://{host}:{port}/healthz",
                "agent_run": f"http://{host}:{port}/v1/agent/run",
                "provider": provider,
                "model": model,
                "harness_version": HARNESS_VERSION,
            },
            indent=2,
        )
    )
    server.serve_forever()
