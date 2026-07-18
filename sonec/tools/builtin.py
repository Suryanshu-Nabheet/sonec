"""Builtin tool assembly for a workspace."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sonec.core.config import Settings
from sonec.core.types import ToolResult, ToolSpec
from sonec.core.workspace import Workspace
from sonec.filesystem.service import FilesystemTools
from sonec.git.service import GitService
from sonec.indexing.index import RepositoryIndex
from sonec.memory.store import MemoryStore
from sonec.rules.engine import RulesEngine
from sonec.skills.registry import SkillsRegistry
from sonec.terminal.service import TerminalService
from sonec.tools.registry import FunctionTool, ToolRegistry, json_content


def build_default_registry(
    workspace: Workspace,
    settings: Settings,
    *,
    memory: MemoryStore | None = None,
    index: RepositoryIndex | None = None,
    skills: SkillsRegistry | None = None,
    rules: RulesEngine | None = None,
) -> ToolRegistry:
    registry = ToolRegistry()
    fs = FilesystemTools(workspace, max_read_bytes=settings.index_max_file_bytes)
    terminal = TerminalService(
        workspace,
        timeout_s=settings.terminal_timeout_s,
        allow_network=settings.allow_network_tools,
    )
    git = GitService(workspace, terminal)
    repo_index = index or RepositoryIndex(workspace, max_file_bytes=settings.index_max_file_bytes)
    skills_reg = skills or SkillsRegistry()
    rules_eng = rules or RulesEngine()

    for tool in fs.tools():
        registry.register(tool)
    for tool in terminal.tools():
        registry.register(tool)
    for tool in git.tools():
        registry.register(tool)

    async def index_build(arguments: Mapping[str, Any]) -> ToolResult:
        del arguments
        count = repo_index.build()
        return ToolResult(
            tool_call_id="",
            name="index_build",
            content=json_content({"indexed_files": count, **repo_index.summary()}),
            ok=True,
            data={"indexed_files": count},
        )

    async def index_search(arguments: Mapping[str, Any]) -> ToolResult:
        query = str(arguments["query"])
        limit = int(arguments.get("limit") or 30)
        hits = repo_index.search(query, limit=limit)
        return ToolResult(
            tool_call_id="",
            name="index_search",
            content=json_content({"query": query, "matches": hits}),
            ok=True,
            data={"matches": hits},
        )

    async def index_symbols(arguments: Mapping[str, Any]) -> ToolResult:
        path = arguments.get("path")
        symbols = repo_index.symbols(str(path) if path else None)
        return ToolResult(
            tool_call_id="",
            name="index_symbols",
            content=json_content({"symbols": symbols[:200], "count": len(symbols)}),
            ok=True,
            data={"symbols": symbols},
        )

    registry.register(
        FunctionTool(
            ToolSpec(
                name="index_build",
                description="Build or refresh the repository file index.",
                parameters={"type": "object", "properties": {}, "additionalProperties": False},
            ),
            index_build,
        )
    )
    registry.register(
        FunctionTool(
            ToolSpec(
                name="index_search",
                description="Search the repository index for a query string.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer", "default": 30},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            ),
            index_search,
        )
    )
    registry.register(
        FunctionTool(
            ToolSpec(
                name="index_symbols",
                description="List lightweight code symbols (defs/classes/functions).",
                parameters={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "additionalProperties": False,
                },
            ),
            index_symbols,
        )
    )

    if memory is not None:

        async def memory_note(arguments: Mapping[str, Any]) -> ToolResult:
            text = str(arguments["text"])
            tags = arguments.get("tags") or []
            note_id = memory.add_note(text, tags=[str(t) for t in tags])
            return ToolResult(
                tool_call_id="",
                name="memory_note",
                content=f"Stored note {note_id}",
                ok=True,
                data={"id": note_id},
            )

        async def memory_search(arguments: Mapping[str, Any]) -> ToolResult:
            query = str(arguments.get("query") or "")
            notes = memory.search_notes(query, limit=int(arguments.get("limit") or 10))
            return ToolResult(
                tool_call_id="",
                name="memory_search",
                content=json_content({"notes": notes}),
                ok=True,
                data={"notes": notes},
            )

        registry.register(
            FunctionTool(
                ToolSpec(
                    name="memory_note",
                    description="Persist a durable note in agent memory.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "tags": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["text"],
                        "additionalProperties": False,
                    },
                ),
                memory_note,
            )
        )
        registry.register(
            FunctionTool(
                ToolSpec(
                    name="memory_search",
                    description="Search durable notes.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "limit": {"type": "integer", "default": 10},
                        },
                        "additionalProperties": False,
                    },
                ),
                memory_search,
            )
        )

    async def skills_list(arguments: Mapping[str, Any]) -> ToolResult:
        del arguments
        return ToolResult(
            tool_call_id="",
            name="skills_list",
            content=json_content({"skills": skills_reg.catalog()}),
            ok=True,
            data={"skills": skills_reg.catalog()},
        )

    async def skills_load(arguments: Mapping[str, Any]) -> ToolResult:
        skill_id = str(arguments["id"])
        try:
            skill = skills_reg.get(skill_id)
        except KeyError:
            return ToolResult(
                tool_call_id="",
                name="skills_load",
                content=f"Unknown skill: {skill_id}",
                ok=False,
            )
        return ToolResult(
            tool_call_id="",
            name="skills_load",
            content=f"# {skill.name}\n\n{skill.body}",
            ok=True,
            data={"id": skill.id, "name": skill.name},
        )

    async def rules_list(arguments: Mapping[str, Any]) -> ToolResult:
        del arguments
        return ToolResult(
            tool_call_id="",
            name="rules_list",
            content=json_content({"rules": rules_eng.list_rules()}),
            ok=True,
            data={"rules": rules_eng.list_rules()},
        )

    async def rules_load(arguments: Mapping[str, Any]) -> ToolResult:
        rule_id = str(arguments["id"])
        try:
            body = rules_eng.get_full(rule_id)
        except KeyError:
            return ToolResult(
                tool_call_id="",
                name="rules_load",
                content=f"Unknown rule: {rule_id}",
                ok=False,
            )
        return ToolResult(
            tool_call_id="",
            name="rules_load",
            content=body,
            ok=True,
            data={"id": rule_id, "chars": len(body)},
        )

    registry.register(
        FunctionTool(
            ToolSpec(
                name="skills_list",
                description="List available SONEC skills (progressive expertise packs).",
                parameters={"type": "object", "properties": {}, "additionalProperties": False},
            ),
            skills_list,
        )
    )
    registry.register(
        FunctionTool(
            ToolSpec(
                name="skills_load",
                description="Load the full body of a skill by id.",
                parameters={
                    "type": "object",
                    "properties": {"id": {"type": "string"}},
                    "required": ["id"],
                    "additionalProperties": False,
                },
            ),
            skills_load,
        )
    )
    registry.register(
        FunctionTool(
            ToolSpec(
                name="rules_list",
                description="List operating rules (including prebuilt rules).",
                parameters={"type": "object", "properties": {}, "additionalProperties": False},
            ),
            rules_list,
        )
    )
    registry.register(
        FunctionTool(
            ToolSpec(
                name="rules_load",
                description="Load the full (untruncated) body of a rule by id (e.g. prebuilt/engineering-constitution).",
                parameters={
                    "type": "object",
                    "properties": {"id": {"type": "string"}},
                    "required": ["id"],
                    "additionalProperties": False,
                },
            ),
            rules_load,
        )
    )

    return registry
