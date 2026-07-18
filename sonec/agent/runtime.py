"""Canonical SONEC agent runtime — single production loop for CLI, eval, training."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from sonec.core.events import EventBus
from sonec.core.types import (
    AgentEvent,
    AgentEventKind,
    AgentRunResult,
    CompletionRequest,
    Message,
    Role,
    ToolResult,
    new_id,
)
from sonec.harness.compaction import compact_messages, should_compact
from sonec.harness.trajectory import TrajectoryLogger
from sonec.harness.versioning import (
    HARNESS_VERSION,
    filter_core_specs,
    tool_schema_hash,
)
from sonec.llm.provider import LLMProvider
from sonec.memory.store import InMemoryStore, MemoryStore
from sonec.tools.registry import ToolRegistry

# Optional phase hints — same loop, same tools, same logging. Not a second product.
PHASE_HINTS: list[str] = [
    "Recon first: index/search/read before edits.",
    "Plan briefly with success criteria, then implement with minimal diffs.",
    "Verify with commands/tests before finishing. Evidence required.",
]


class AgentRuntime:
    """Frozen production harness loop (Phase 0).

    Used identically by CLI, eval, and rollout workers.
    `success` on the result is provisional (model finished); graders set
    environment evidence. Prefer `evidence_success` from eval.
    """

    def __init__(
        self,
        provider: LLMProvider,
        tools: ToolRegistry,
        *,
        memory: MemoryStore | None = None,
        events: EventBus | None = None,
        max_iterations: int = 48,
        system_prompt: str = "",
        model_id: str = "unknown",
        log_dir: Path | None = None,
        enable_phase_hints: bool = False,
        compact_after: int = 40,
        on_event: Callable[[AgentEvent], None] | None = None,
    ) -> None:
        self.provider = provider
        self.tools = tools
        self.memory = memory or InMemoryStore()
        self.events = events or EventBus()
        self.max_iterations = max_iterations
        self.system_prompt = system_prompt
        self.model_id = model_id
        self.log_dir = log_dir
        self.enable_phase_hints = enable_phase_hints
        self.compact_after = compact_after
        self.core_specs = filter_core_specs(tools.specs())
        self.tool_hash = tool_schema_hash(self.core_specs)
        if on_event is not None:
            self.events.subscribe(on_event)

    def _emit(self, kind: AgentEventKind, message: str = "", **payload: object) -> None:
        event = AgentEvent(kind=kind, message=message, payload=dict(payload))
        self.events.emit(event)
        return

    async def run(self, goal: str) -> AgentRunResult:
        run_id = new_id("run_")
        goal = goal.strip()
        logger: TrajectoryLogger | None = None
        if self.log_dir is not None:
            logger = TrajectoryLogger(
                self.log_dir / f"{run_id}.jsonl",
                run_id=run_id,
                goal=goal,
                model_id=self.model_id,
                tool_schema_hash=self.tool_hash,
            )
            self.events.subscribe(logger.log_event)

        self._emit(
            AgentEventKind.STARTED,
            "run started",
            run_id=run_id,
            goal=goal,
            harness_version=HARNESS_VERSION,
            tool_schema_hash=self.tool_hash,
            model_id=self.model_id,
        )

        messages: list[Message] = [
            Message(role=Role.SYSTEM, content=self.system_prompt),
            Message(role=Role.USER, content=goal),
        ]
        if self.enable_phase_hints:
            messages.append(
                Message(
                    role=Role.USER,
                    content="Guidance:\n- " + "\n- ".join(PHASE_HINTS),
                )
            )
        for message in messages:
            self.memory.add_message(message)
            if logger:
                logger.log_message(message)

        final_message = ""
        iterations = 0
        total_usage: dict[str, int] = {}

        for iteration in range(1, self.max_iterations + 1):
            iterations = iteration
            if should_compact(messages, max_messages=self.compact_after):
                messages = compact_messages(messages)
                self._emit(AgentEventKind.WARNING, "context compacted", iteration=iteration)

            self._emit(AgentEventKind.STEP, f"iteration {iteration}", iteration=iteration)
            t0 = time.perf_counter()
            response = await self.provider.complete(
                CompletionRequest(messages=messages, tools=self.core_specs)
            )
            latency = time.perf_counter() - t0
            for k, v in response.usage.items():
                total_usage[k] = total_usage.get(k, 0) + int(v)
            if logger:
                logger.log_usage(response.usage, latency_s=latency)

            assistant = response.message
            messages.append(assistant)
            self.memory.add_message(assistant)
            if logger:
                logger.log_message(assistant)
            self._emit(
                AgentEventKind.MESSAGE,
                assistant.content or "",
                finish_reason=response.finish_reason,
            )

            if not assistant.tool_calls:
                final_message = assistant.content or ""
                break

            for call in assistant.tool_calls:
                self._emit(
                    AgentEventKind.TOOL_CALL,
                    call.name,
                    tool=call.name,
                    arguments=call.arguments,
                    tool_call_id=call.id,
                )
                result: ToolResult = await self.tools.execute(
                    tool_call_id=call.id,
                    name=call.name,
                    arguments=call.arguments,
                )
                tool_message = Message(
                    role=Role.TOOL,
                    content=result.content,
                    name=result.name,
                    tool_call_id=result.tool_call_id or call.id,
                )
                messages.append(tool_message)
                self.memory.add_message(tool_message)
                if logger:
                    logger.log_message(tool_message)
                self._emit(
                    AgentEventKind.TOOL_RESULT,
                    result.content[:500],
                    tool=result.name,
                    ok=result.ok,
                    tool_call_id=result.tool_call_id or call.id,
                )
        else:
            final_message = "Stopped: reached max iterations."
            result = AgentRunResult(
                run_id=run_id,
                goal=goal,
                success=False,
                final_message=final_message,
                messages=messages,
                events=list(self.events.history),
                iterations=iterations,
                harness_version=HARNESS_VERSION,
                tool_schema_hash=self.tool_hash,
                model_id=self.model_id,
                usage=total_usage,
                evidence_success=None,
            )
            self._emit(AgentEventKind.FAILED, final_message)
            if logger:
                logger.close(result)
            return result

        # Model finished speaking — NOT environment success. Graders decide.
        result = AgentRunResult(
            run_id=run_id,
            goal=goal,
            success=False,
            final_message=final_message,
            messages=messages,
            events=list(self.events.history),
            iterations=iterations,
            harness_version=HARNESS_VERSION,
            tool_schema_hash=self.tool_hash,
            model_id=self.model_id,
            usage=total_usage,
            evidence_success=None,
            completed=True,
        )
        self._emit(AgentEventKind.COMPLETED, final_message, completed=True)
        if logger:
            logger.close(result)
        return result
