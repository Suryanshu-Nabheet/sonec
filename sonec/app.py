"""Canonical factory — one harness for CLI, eval, and training workers."""

from __future__ import annotations

from pathlib import Path

from sonec.agent.runtime import AgentRuntime
from sonec.core.config import Settings, load_settings
from sonec.core.events import EventBus
from sonec.core.workspace import Workspace
from sonec.harness.context import ContextAssembler
from sonec.indexing.index import RepositoryIndex
from sonec.llm.provider import LLMProvider, MockProvider, create_provider
from sonec.memory.store import FileMemoryStore, InMemoryStore, MemoryStore
from sonec.rules.engine import RulesEngine
from sonec.skills.registry import SkillsRegistry
from sonec.tools.builtin import build_default_registry
from sonec.tools.registry import ToolRegistry


def build_runtime(
    *,
    workspace: Path | str | None = None,
    settings: Settings | None = None,
    provider: LLMProvider | None = None,
    memory: MemoryStore | None = None,
    persist_memory: bool = True,
    events: EventBus | None = None,
    log_dir: Path | None = None,
    enable_phase_hints: bool = False,
    goal_for_prompt: str = "",
) -> tuple[AgentRuntime, Settings, Workspace, ToolRegistry]:
    """Assemble the frozen Phase-0 production runtime."""
    cfg = settings or load_settings(**({"workspace": workspace} if workspace else {}))
    if workspace is not None:
        cfg.workspace = Path(workspace).expanduser().resolve()
    ws = Workspace(cfg.workspace)
    if memory is not None:
        mem = memory
    elif persist_memory:
        mem = FileMemoryStore(cfg.memory_path())
    else:
        mem = InMemoryStore()
    llm = provider or create_provider(cfg)
    skills = SkillsRegistry()
    rules = RulesEngine()  # available for tools; not dumped into always-on prompt
    index = RepositoryIndex(ws, max_file_bytes=cfg.index_max_file_bytes)
    registry = build_default_registry(
        ws, cfg, memory=mem, index=index, skills=skills, rules=rules
    )
    assembler = ContextAssembler(skills, index=index)
    model_id = cfg.model if not isinstance(llm, MockProvider) else "mock"
    default_log = cfg.workspace / ".sonec" / "trajectories" if log_dir is None else log_dir
    runtime = AgentRuntime(
        provider=llm,
        tools=registry,
        memory=mem,
        events=events or EventBus(),
        max_iterations=cfg.max_iterations,
        system_prompt=assembler.build_system_prompt(goal_for_prompt),
        model_id=model_id,
        log_dir=default_log,
        enable_phase_hints=enable_phase_hints,
    )
    return runtime, cfg, ws, registry


# Back-compat aliases — same runtime, one product.
def build_agent(**kwargs: object) -> tuple[AgentRuntime, Settings, Workspace, ToolRegistry]:
    return build_runtime(**kwargs)  # type: ignore[arg-type]


def build_harness(**kwargs: object) -> tuple[AgentRuntime, Settings, Workspace, ToolRegistry]:
    kwargs.setdefault("enable_phase_hints", True)
    return build_runtime(**kwargs)  # type: ignore[arg-type]
