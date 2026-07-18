"""Factory helpers to assemble the advanced SONEC harness (and simple runtime)."""

from __future__ import annotations

from pathlib import Path

from sonec.agent.runtime import AgentRuntime
from sonec.core.config import Settings, load_settings
from sonec.core.events import EventBus
from sonec.core.workspace import Workspace
from sonec.harness.context import ContextAssembler
from sonec.harness.critic import Critic
from sonec.harness.orchestrator import AgenticOrchestrator
from sonec.indexing.index import RepositoryIndex
from sonec.llm.provider import LLMProvider, MockProvider, create_provider
from sonec.memory.store import FileMemoryStore, InMemoryStore, MemoryStore
from sonec.planning.planner import Planner
from sonec.rules.engine import RulesEngine
from sonec.skills.registry import SkillsRegistry
from sonec.tools.builtin import build_default_registry
from sonec.tools.registry import ToolRegistry


def _assemble_core(
    *,
    workspace: Path | str | None,
    settings: Settings | None,
    provider: LLMProvider | None,
    memory: MemoryStore | None,
    persist_memory: bool,
) -> tuple[Settings, Workspace, MemoryStore, LLMProvider, RepositoryIndex, SkillsRegistry, RulesEngine, ToolRegistry]:
    cfg = settings or load_settings(**({"workspace": workspace} if workspace else {}))
    if workspace is not None:
        cfg.workspace = Path(workspace).expanduser().resolve()
    ws = Workspace(cfg.workspace)
    mem: MemoryStore
    if memory is not None:
        mem = memory
    elif persist_memory:
        mem = FileMemoryStore(cfg.memory_path())
    else:
        mem = InMemoryStore()
    llm = provider or create_provider(cfg)
    skills = SkillsRegistry()
    rules = RulesEngine()
    index = RepositoryIndex(ws, max_file_bytes=cfg.index_max_file_bytes)
    registry = build_default_registry(
        ws, cfg, memory=mem, index=index, skills=skills, rules=rules
    )
    return cfg, ws, mem, llm, index, skills, rules, registry


def build_harness(
    *,
    workspace: Path | str | None = None,
    settings: Settings | None = None,
    provider: LLMProvider | None = None,
    memory: MemoryStore | None = None,
    persist_memory: bool = True,
    events: EventBus | None = None,
) -> tuple[AgenticOrchestrator, Settings, Workspace, ToolRegistry]:
    """Assemble the full multi-phase agentic harness (default product surface)."""
    cfg, ws, mem, llm, index, skills, rules, registry = _assemble_core(
        workspace=workspace,
        settings=settings,
        provider=provider,
        memory=memory,
        persist_memory=persist_memory,
    )
    assembler = ContextAssembler(rules, skills, index=index)
    planner_llm = None if isinstance(llm, MockProvider) else llm
    orchestrator = AgenticOrchestrator(
        provider=llm,
        tools=registry,
        assembler=assembler,
        memory=mem,
        planner=Planner(planner_llm),
        critic=Critic(None if isinstance(llm, MockProvider) else llm),
        events=events or EventBus(),
        max_total_iterations=cfg.max_iterations,
    )
    return orchestrator, cfg, ws, registry


def build_agent(
    *,
    workspace: Path | str | None = None,
    settings: Settings | None = None,
    provider: LLMProvider | None = None,
    memory: MemoryStore | None = None,
    persist_memory: bool = True,
    events: EventBus | None = None,
) -> tuple[AgentRuntime, Settings, Workspace, ToolRegistry]:
    """Assemble a single-loop AgentRuntime (simpler / eval-compatible)."""
    cfg, ws, mem, llm, index, skills, rules, registry = _assemble_core(
        workspace=workspace,
        settings=settings,
        provider=provider,
        memory=memory,
        persist_memory=persist_memory,
    )
    assembler = ContextAssembler(rules, skills, index=index)
    planner_llm = None if isinstance(llm, MockProvider) else llm
    runtime = AgentRuntime(
        provider=llm,
        tools=registry,
        memory=mem,
        planner=Planner(planner_llm),
        events=events or EventBus(),
        max_iterations=cfg.max_iterations,
        system_prompt=assembler.build_system_prompt("general software engineering"),
    )
    return runtime, cfg, ws, registry
