"""Multi-phase agentic orchestrator — the advanced harness around the model."""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum

from sonec.agent.runtime import AgentRuntime
from sonec.core.events import EventBus
from sonec.core.types import (
    AgentEvent,
    AgentEventKind,
    AgentRunResult,
    CompletionRequest,
    Message,
    Plan,
    Role,
    ToolResult,
    new_id,
)
from sonec.harness.context import ContextAssembler
from sonec.harness.critic import Critic
from sonec.llm.provider import LLMProvider, MockProvider
from sonec.memory.store import InMemoryStore, MemoryStore
from sonec.planning.planner import Planner
from sonec.tools.registry import ToolRegistry


class Phase(StrEnum):
    RECON = "recon"
    PLAN = "plan"
    EXECUTE = "execute"
    VERIFY = "verify"
    CRITIQUE = "critique"
    DELIVER = "deliver"


PHASE_INSTRUCTIONS: dict[Phase, str] = {
    Phase.RECON: (
        "PHASE=RECON. Do not edit yet. Build/use the index, search, and read the "
        "minimum files needed to understand the goal. Summarize findings, then stop "
        "with a short recon brief (no more tool calls after the brief)."
    ),
    Phase.PLAN: (
        "PHASE=PLAN. Using recon, produce a concrete 3–8 step plan with success criteria. "
        "Prefer text; tools only if a critical fact is still missing. End with the plan."
    ),
    Phase.EXECUTE: (
        "PHASE=EXECUTE. Implement the plan with surgical tool use. Prefer fs_edit. "
        "Do not claim success yet — leave verification for the next phase. "
        "When implementation is done, summarize files touched and stop."
    ),
    Phase.VERIFY: (
        "PHASE=VERIFY. Run the success criteria (tests/commands). Read the output. "
        "If failing, fix with minimal edits and re-verify. End with evidence."
    ),
    Phase.CRITIQUE: (
        "PHASE=CRITIQUE. Self-check: does evidence satisfy the goal? "
        "If gaps remain, list them; otherwise confirm readiness to deliver."
    ),
    Phase.DELIVER: (
        "PHASE=DELIVER. Final response only: what changed, how verified, residual risks. "
        "No further tool calls unless critically required."
    ),
}


class AgenticOrchestrator:
    """Benchmark-oriented multi-phase agent harness.

    This is the product. The LLM provider is a dependency — SONEC adds
    rules, skills, phased control, budgets, and critique.
    """

    def __init__(
        self,
        provider: LLMProvider,
        tools: ToolRegistry,
        assembler: ContextAssembler,
        *,
        memory: MemoryStore | None = None,
        planner: Planner | None = None,
        critic: Critic | None = None,
        events: EventBus | None = None,
        max_iterations_per_phase: int = 10,
        max_total_iterations: int = 48,
        on_event: Callable[[AgentEvent], None] | None = None,
        phases: list[Phase] | None = None,
    ) -> None:
        self.provider = provider
        self.tools = tools
        self.assembler = assembler
        self.memory = memory or InMemoryStore()
        self.planner = planner or Planner(
            None if isinstance(provider, MockProvider) else provider
        )
        self.critic = critic or Critic(
            None if isinstance(provider, MockProvider) else provider
        )
        self.events = events or EventBus()
        self.max_iterations_per_phase = max_iterations_per_phase
        self.max_total_iterations = max_total_iterations
        self.phases = phases or [
            Phase.RECON,
            Phase.PLAN,
            Phase.EXECUTE,
            Phase.VERIFY,
            Phase.CRITIQUE,
            Phase.DELIVER,
        ]
        if on_event is not None:
            self.events.subscribe(on_event)

    def _emit(self, kind: AgentEventKind, message: str = "", **payload: object) -> None:
        self.events.emit(AgentEvent(kind=kind, message=message, payload=dict(payload)))

    async def run(self, goal: str) -> AgentRunResult:
        run_id = new_id("run_")
        goal = goal.strip()
        self._emit(AgentEventKind.STARTED, "Agentic harness started", run_id=run_id, goal=goal)

        system_prompt = self.assembler.build_system_prompt(goal)
        plan = await self.planner.create_plan(goal)
        self._emit(AgentEventKind.PLAN, "Seed plan", plan=plan.model_dump())

        messages: list[Message] = [
            Message(role=Role.SYSTEM, content=system_prompt),
            Message(
                role=Role.USER,
                content=(
                    f"Goal:\n{goal}\n\n"
                    f"Initial plan rationale: {plan.rationale or '(none)'}\n"
                    + "\n".join(
                        f"{i + 1}. {s.title} — {s.detail}" for i, s in enumerate(plan.steps)
                    )
                ),
            ),
        ]
        for message in messages:
            self.memory.add_message(message)

        total_iterations = 0
        phase_notes: list[str] = []
        final_message = ""

        for phase in self.phases:
            self._emit(AgentEventKind.STEP, f"Enter phase {phase.value}", phase=phase.value)
            phase_user = Message(
                role=Role.USER,
                content=PHASE_INSTRUCTIONS[phase],
            )
            messages.append(phase_user)
            self.memory.add_message(phase_user)

            phase_final = ""
            for iteration in range(1, self.max_iterations_per_phase + 1):
                if total_iterations >= self.max_total_iterations:
                    final_message = "Stopped: total iteration budget exhausted."
                    self._emit(AgentEventKind.FAILED, final_message)
                    return self._result(
                        run_id, goal, False, final_message, messages, plan, total_iterations
                    )
                total_iterations += 1
                self._emit(
                    AgentEventKind.STEP,
                    f"{phase.value} iteration {iteration}",
                    phase=phase.value,
                    iteration=iteration,
                )

                # Critique phase can short-circuit to programmatic critic
                if phase == Phase.CRITIQUE and iteration == 1:
                    summary = self._transcript_summary(messages, phase_notes)
                    critique = await self.critic.review(goal=goal, transcript_summary=summary)
                    critique_msg = Message(
                        role=Role.ASSISTANT,
                        content=(
                            f"Critic verdict: {critique.get('verdict')}\n"
                            f"Reason: {critique.get('reason')}\n"
                            f"Actions: {critique.get('required_actions')}"
                        ),
                    )
                    messages.append(critique_msg)
                    self.memory.add_message(critique_msg)
                    phase_final = critique_msg.content or ""
                    phase_notes.append(f"[{phase.value}] {phase_final}")
                    self._emit(AgentEventKind.MESSAGE, phase_final, phase=phase.value)
                    if critique.get("verdict") == "fail":
                        # Inject a recovery EXECUTE+VERIFY mini-loop instruction once
                        recovery = Message(
                            role=Role.USER,
                            content=(
                                "CRITIC FAILED. Perform the required_actions with tools now, "
                                "then re-verify. When done, summarize evidence."
                            ),
                        )
                        messages.append(recovery)
                        self.memory.add_message(recovery)
                        # Temporarily run a short execute/verify burst inside critique recovery
                        recovered = await self._tool_loop(
                            messages,
                            budget=min(6, self.max_iterations_per_phase),
                            phase="critique_recovery",
                        )
                        total_iterations += recovered
                    break

                allow_tools = phase not in {Phase.DELIVER}
                response = await self.provider.complete(
                    CompletionRequest(
                        messages=messages,
                        tools=self.tools.specs() if allow_tools else [],
                    )
                )
                assistant = response.message
                messages.append(assistant)
                self.memory.add_message(assistant)
                self._emit(
                    AgentEventKind.MESSAGE,
                    assistant.content or "",
                    phase=phase.value,
                    finish_reason=response.finish_reason,
                )

                if not assistant.tool_calls:
                    phase_final = assistant.content or ""
                    break

                if not allow_tools:
                    phase_final = assistant.content or "(tools ignored in deliver)"
                    break

                for call in assistant.tool_calls:
                    self._emit(
                        AgentEventKind.TOOL_CALL,
                        call.name,
                        tool=call.name,
                        arguments=call.arguments,
                        tool_call_id=call.id,
                        phase=phase.value,
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
                    self._emit(
                        AgentEventKind.TOOL_RESULT,
                        result.content[:500],
                        tool=result.name,
                        ok=result.ok,
                        tool_call_id=result.tool_call_id or call.id,
                        phase=phase.value,
                    )
            else:
                phase_final = phase_final or f"Phase {phase.value} hit iteration cap."

            phase_notes.append(f"[{phase.value}] {phase_final[:1000]}")
            if phase == Phase.DELIVER:
                final_message = phase_final

        if not final_message:
            final_message = phase_notes[-1] if phase_notes else "No delivery message."

        success = True
        self._emit(AgentEventKind.COMPLETED, final_message, success=success)
        return self._result(run_id, goal, success, final_message, messages, plan, total_iterations)

    async def _tool_loop(self, messages: list[Message], *, budget: int, phase: str) -> int:
        used = 0
        for _ in range(budget):
            used += 1
            response = await self.provider.complete(
                CompletionRequest(messages=messages, tools=self.tools.specs())
            )
            assistant = response.message
            messages.append(assistant)
            self.memory.add_message(assistant)
            if not assistant.tool_calls:
                break
            for call in assistant.tool_calls:
                result = await self.tools.execute(
                    tool_call_id=call.id, name=call.name, arguments=call.arguments
                )
                tool_message = Message(
                    role=Role.TOOL,
                    content=result.content,
                    name=result.name,
                    tool_call_id=result.tool_call_id or call.id,
                )
                messages.append(tool_message)
                self.memory.add_message(tool_message)
        return used

    def _transcript_summary(self, messages: list[Message], phase_notes: list[str]) -> str:
        parts = ["Phase notes:", *phase_notes, "", "Recent messages:"]
        for msg in messages[-20:]:
            content = (msg.content or "")[:400]
            tool_bits = ""
            if msg.tool_calls:
                tool_bits = " tools=" + ",".join(c.name for c in msg.tool_calls)
            parts.append(f"- {msg.role.value}{tool_bits}: {content}")
        return "\n".join(parts)

    def _result(
        self,
        run_id: str,
        goal: str,
        success: bool,
        final_message: str,
        messages: list[Message],
        plan: Plan,
        iterations: int,
    ) -> AgentRunResult:
        return AgentRunResult(
            run_id=run_id,
            goal=goal,
            success=success,
            final_message=final_message,
            messages=messages,
            plan=plan,
            events=list(self.events.history),
            iterations=iterations,
        )


def as_simple_runtime(orchestrator: AgenticOrchestrator) -> AgentRuntime:
    """Adapter for callers that still expect AgentRuntime.run."""
    return AgentRuntime(
        provider=orchestrator.provider,
        tools=orchestrator.tools,
        memory=orchestrator.memory,
        planner=orchestrator.planner,
        events=orchestrator.events,
        max_iterations=orchestrator.max_total_iterations,
        system_prompt=orchestrator.assembler.build_system_prompt("general engineering"),
    )
