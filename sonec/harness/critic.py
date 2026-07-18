"""Critic — forces a verification/reflection pass before delivery."""

from __future__ import annotations

from sonec.core.types import CompletionRequest, Message, Role
from sonec.llm.provider import LLMProvider, MockProvider

CRITIC_PROMPT = """You are the sonec critic.
Evaluate whether the agent run is grounded in tool evidence and ready to complete.

Given the goal and the agent transcript summary, decide if the work is done.

Respond with EXACTLY this JSON (no markdown fences):
{
  "verdict": "pass" | "fail",
  "reason": "short reason",
  "required_actions": ["optional follow-up actions if fail"]
}

Fail if:
- files were claimed changed but never verified
- tests or commands were not run when they should have been
- the goal is only partially addressed
- claims lack tool evidence
"""


class Critic:
    def __init__(self, provider: LLMProvider | None) -> None:
        self.provider = provider

    async def review(self, *, goal: str, transcript_summary: str) -> dict[str, object]:
        if self.provider is None or isinstance(self.provider, MockProvider):
            # Offline heuristic: pass if verification keywords / phase evidence present
            lowered = transcript_summary.lower()
            evidence = any(
                token in lowered
                for token in (
                    "pytest",
                    "passed",
                    "exit_code\": 0",
                    "verified",
                    "verify:",
                    "[verify]",
                    "wrote",
                    "offline mock",
                    "verification evidence",
                )
            )
            return {
                "verdict": "pass" if evidence else "fail",
                "reason": "mock/heuristic critic",
                "required_actions": []
                if evidence
                else ["Run a verification command and report output"],
            }

        messages = [
            Message(role=Role.SYSTEM, content=CRITIC_PROMPT),
            Message(
                role=Role.USER,
                content=f"Goal:\n{goal}\n\nTranscript summary:\n{transcript_summary[:12_000]}",
            ),
        ]
        response = await self.provider.complete(
            CompletionRequest(messages=messages, temperature=0.0, max_tokens=1024)
        )
        content = response.message.content or ""
        import json
        import re

        text = content.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        try:
            data = json.loads(text)
            if not isinstance(data, dict):
                raise ValueError("not a dict")
            return {
                "verdict": str(data.get("verdict", "fail")),
                "reason": str(data.get("reason", "")),
                "required_actions": list(data.get("required_actions") or []),
            }
        except Exception:
            return {
                "verdict": "fail",
                "reason": f"Critic returned unparseable output: {content[:200]}",
                "required_actions": ["Re-run verification with tools"],
            }
