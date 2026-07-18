"""Core domain types shared across SONEC."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(UTC)


def new_id(prefix: str = "") -> str:
    value = uuid4().hex
    return f"{prefix}{value}" if prefix else value


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    """A single chat or tool message in an agent transcript."""

    role: Role
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None
    reasoning_content: str | None = None
    created_at: datetime = Field(default_factory=utc_now)


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    tool_call_id: str
    name: str
    content: str
    ok: bool = True
    data: dict[str, Any] | None = None


class ToolSpec(BaseModel):
    """OpenAI-compatible function tool description."""

    name: str
    description: str
    parameters: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}, "additionalProperties": False}
    )

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class PlanStep(BaseModel):
    id: str = Field(default_factory=lambda: new_id("step_"))
    title: str
    detail: str = ""
    status: str = "pending"
    depends_on: list[str] = Field(default_factory=list)


class Plan(BaseModel):
    goal: str
    steps: list[PlanStep] = Field(default_factory=list)
    rationale: str = ""
    created_at: datetime = Field(default_factory=utc_now)


class AgentEventKind(StrEnum):
    STARTED = "started"
    PLAN = "plan"
    MESSAGE = "message"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    STEP = "step"
    COMPLETED = "completed"
    FAILED = "failed"
    WARNING = "warning"


class AgentEvent(BaseModel):
    kind: AgentEventKind
    message: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class AgentRunResult(BaseModel):
    run_id: str
    goal: str
    success: bool
    final_message: str
    messages: list[Message] = Field(default_factory=list)
    plan: Plan | None = None
    events: list[AgentEvent] = Field(default_factory=list)
    iterations: int = 0
    harness_version: str = ""
    tool_schema_hash: str = ""
    model_id: str = ""
    usage: dict[str, int] = Field(default_factory=dict)
    evidence_success: bool | None = None
    completed: bool = False


class CompletionRequest(BaseModel):
    messages: list[Message]
    tools: list[ToolSpec] = Field(default_factory=list)
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    stop: list[str] | None = None


class CompletionResponse(BaseModel):
    message: Message
    finish_reason: str = "stop"
    usage: dict[str, int] = Field(default_factory=dict)
    raw: dict[str, Any] | None = None


# Resolve forward refs for Message.tool_calls
Message.model_rebuild()
