"""Context assembler — builds the advanced system prompt from rules + skills + repo."""

from __future__ import annotations

from sonec.indexing.index import RepositoryIndex
from sonec.rules.engine import RulesEngine
from sonec.skills.registry import SkillsRegistry

BASE_IDENTITY = """You are SONEC (Senior Open-source Neural Engineering Companion),
the apex agentic software-engineering system by Suryanshu Nabheet.

You are the operating layer of elite software work: prebuilt rules, progressive
skills, multi-phase orchestration, verification gates, critique, and sandboxed
tools — assembled into one coherent engineering companion.

Default reasoning engine: Kimi K3 (Moonshot). Your mandate is staff-level
execution: localize precisely, patch minimally, verify with evidence, deliver
production-grade outcomes that set the standard for agentic coding.
"""


class ContextAssembler:
    def __init__(
        self,
        rules: RulesEngine,
        skills: SkillsRegistry,
        *,
        index: RepositoryIndex | None = None,
    ) -> None:
        self.rules = rules
        self.skills = skills
        self.index = index

    def build_system_prompt(self, goal: str) -> str:
        sections = [
            BASE_IDENTITY,
            self.rules.render(goal),
            self.skills.render(goal),
            self._repo_brief(),
            self._tooling_brief(),
            self._phase_contract(),
        ]
        return "\n\n---\n\n".join(section.strip() for section in sections if section.strip())

    def _repo_brief(self) -> str:
        if self.index is None:
            return "# Repository\n\nIndex not yet built. Call `index_build` early in recon."
        if not self.index.files:
            self.index.build()
        summary = self.index.summary()
        sample = sorted(self.index.files.keys())[:40]
        lines = [
            "# Repository brief",
            f"- root: `{summary.get('root')}`",
            f"- files indexed: {summary.get('file_count')}",
            f"- languages: {summary.get('languages')}",
            "- sample paths:",
            *[f"  - `{p}`" for p in sample],
        ]
        return "\n".join(lines)

    def _tooling_brief(self) -> str:
        return """# Tools
Filesystem: fs_list, fs_read, fs_write, fs_edit, fs_search
Terminal: terminal_run (network blocked by default)
Git: git_status, git_diff, git_log, git_branch
Index: index_build, index_search, index_symbols
Memory: memory_note, memory_search
Meta: skills_list, skills_load, rules_list, rules_load
Use meta tools to pull full skill/rule bodies when activation was incomplete.
"""

    def _phase_contract(self) -> str:
        return """# Phase contract (orchestrator)
You will be driven through phases: RECON → PLAN → EXECUTE → VERIFY → CRITIQUE → DELIVER.
In each phase, obey the phase instruction in the user message.
Do not skip VERIFY. Do not DELIVER without verification evidence (or an explicit blocker).
"""
