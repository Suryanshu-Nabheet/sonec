"""Planning engine: decompose goals into ordered steps."""

from __future__ import annotations

import json
import re
from typing import Any

from sonec.core.types import Message, Plan, PlanStep, Role
from sonec.llm.provider import LLMProvider

PLAN_SYSTEM_PROMPT = """You are the planning engine for sonec.
Given a goal, produce a concise, actionable plan as JSON only (no markdown fences).

Schema:
{
  "rationale": "string",
  "steps": [
    {"title": "string", "detail": "string", "depends_on": []}
  ]
}

Rules:
- Prefer 3–8 steps.
- Steps must be concrete engineering actions (inspect, edit, test, verify).
- depends_on lists step indices (0-based) or empty.
"""


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def heuristic_plan(goal: str) -> Plan:
    """Fallback planner used when the LLM is unavailable or returns invalid JSON."""
    steps = [
        PlanStep(title="Understand the repository and goal", detail=goal),
        PlanStep(title="Locate relevant code", detail="Search and read the most relevant files."),
        PlanStep(title="Implement the change", detail="Apply minimal, correct edits."),
        PlanStep(title="Verify", detail="Run tests or commands that prove the change works."),
        PlanStep(title="Summarize", detail="Report what changed and how to validate."),
    ]
    for index, step in enumerate(steps):
        if index > 0:
            step.depends_on = [steps[index - 1].id]
    return Plan(goal=goal, steps=steps, rationale="Heuristic default plan")


class Planner:
    def __init__(self, provider: LLMProvider | None = None) -> None:
        self.provider = provider

    async def create_plan(self, goal: str) -> Plan:
        goal = goal.strip()
        if not goal:
            return Plan(goal="", steps=[], rationale="Empty goal")
        if self.provider is None:
            return heuristic_plan(goal)

        request_messages = [
            Message(role=Role.SYSTEM, content=PLAN_SYSTEM_PROMPT),
            Message(role=Role.USER, content=f"Goal:\n{goal}"),
        ]
        from sonec.core.types import CompletionRequest

        response = await self.provider.complete(
            CompletionRequest(messages=request_messages, temperature=0.1, max_tokens=2048)
        )
        content = response.message.content or ""
        try:
            data = _extract_json(content)
            raw_steps = data.get("steps") or []
            steps: list[PlanStep] = []
            for item in raw_steps:
                steps.append(
                    PlanStep(
                        title=str(item.get("title", "Untitled")).strip() or "Untitled",
                        detail=str(item.get("detail", "")).strip(),
                    )
                )
            # Resolve depends_on indices to step ids after all steps exist
            for index, item in enumerate(raw_steps):
                deps = item.get("depends_on") or []
                resolved: list[str] = []
                for dep in deps:
                    if isinstance(dep, int) and 0 <= dep < len(steps):
                        resolved.append(steps[dep].id)
                steps[index].depends_on = resolved
            if not steps:
                return heuristic_plan(goal)
            return Plan(
                goal=goal,
                steps=steps,
                rationale=str(data.get("rationale", "")).strip(),
            )
        except (json.JSONDecodeError, TypeError, AttributeError, KeyError):
            return heuristic_plan(goal)
