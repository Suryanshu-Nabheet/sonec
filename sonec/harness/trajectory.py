"""Trajectory JSONL logging for training / RL / audit."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, TextIO

from sonec.core.types import AgentEvent, AgentRunResult, Message, utc_now
from sonec.harness.versioning import HARNESS_VERSION


class TrajectoryLogger:
    """Append-only JSONL logger for every harness run."""

    def __init__(
        self,
        path: Path,
        *,
        run_id: str,
        goal: str,
        model_id: str,
        tool_schema_hash: str,
        harness_version: str = HARNESS_VERSION,
    ) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.goal = goal
        self.model_id = model_id
        self.tool_schema_hash = tool_schema_hash
        self.harness_version = harness_version
        self.started = time.perf_counter()
        self._handle: TextIO = self.path.open("a", encoding="utf-8")
        self._write(
            {
                "type": "run_start",
                "run_id": run_id,
                "goal": goal,
                "model_id": model_id,
                "harness_version": harness_version,
                "tool_schema_hash": tool_schema_hash,
                "ts": utc_now().isoformat(),
            }
        )

    def _write(self, record: dict[str, Any]) -> None:
        self._handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._handle.flush()

    def log_message(self, message: Message) -> None:
        self._write(
            {
                "type": "message",
                "run_id": self.run_id,
                "role": message.role.value,
                "content": message.content,
                "name": message.name,
                "tool_call_id": message.tool_call_id,
                "tool_calls": [c.model_dump() for c in message.tool_calls]
                if message.tool_calls
                else None,
                "ts": utc_now().isoformat(),
            }
        )

    def log_event(self, event: AgentEvent) -> None:
        self._write(
            {
                "type": "event",
                "run_id": self.run_id,
                "kind": event.kind.value,
                "message": event.message,
                "payload": event.payload,
                "ts": utc_now().isoformat(),
            }
        )

    def log_usage(self, usage: dict[str, int], *, latency_s: float) -> None:
        self._write(
            {
                "type": "usage",
                "run_id": self.run_id,
                "usage": usage,
                "latency_s": latency_s,
                "ts": utc_now().isoformat(),
            }
        )

    def close(self, result: AgentRunResult) -> None:
        self._write(
            {
                "type": "run_end",
                "run_id": self.run_id,
                "success": result.success,
                "final_message": result.final_message,
                "iterations": result.iterations,
                "duration_s": time.perf_counter() - self.started,
                "harness_version": self.harness_version,
                "tool_schema_hash": self.tool_schema_hash,
                "model_id": self.model_id,
                "ts": utc_now().isoformat(),
            }
        )
        self._handle.close()
