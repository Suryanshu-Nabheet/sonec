"""Deprecated module path — use sonec.agent.runtime.AgentRuntime."""

from sonec.agent.runtime import PHASE_HINTS, AgentRuntime

AgenticOrchestrator = AgentRuntime
Phase = None

__all__ = ["AgentRuntime", "AgenticOrchestrator", "PHASE_HINTS", "Phase"]
