"""Thin production system prompt — skills/rules load on demand."""

from __future__ import annotations

from sonec.harness.versioning import HARNESS_VERSION
from sonec.indexing.index import RepositoryIndex
from sonec.skills.registry import SkillsRegistry

# Hard cap for always-on identity (Phase 0). Skills are progressive, not dumped.
MAX_ALWAYS_ON_CHARS = 3500

THIN_IDENTITY = f"""You are sonec v{HARNESS_VERSION} — a coding-agent model on Qwen 3.5.
Agentic software engineering for IDEs and CLIs. Prefer minimal diffs.

Contract:
1. Inspect with tools before editing (index/search/read).
2. Prefer fs_edit over full rewrites; stay inside the workspace.
3. Verify with terminal commands/tests before finishing — evidence required.
4. Never invent file contents you have not read.
5. When done, summarize paths changed and exact verification evidence.
6. Question-only asks: answer; do not edit files.
7. Be precise and production-grade.

Tool families: filesystem, terminal, git, repository index.
"""


class ContextAssembler:
    def __init__(
        self,
        skills: SkillsRegistry | None = None,
        *,
        index: RepositoryIndex | None = None,
        max_chars: int = MAX_ALWAYS_ON_CHARS,
    ) -> None:
        self.skills = skills or SkillsRegistry()
        self.index = index
        self.max_chars = max_chars

    def build_system_prompt(self, goal: str = "") -> str:
        parts = [THIN_IDENTITY.strip()]
        # Progressive skill titles only (not full bodies) — durable skill → weights later.
        activations = self.skills.activate(goal or "software engineering", limit=3)
        if activations:
            catalog = "; ".join(
                f"{a.skill.id} ({a.skill.description[:60]})" for a in activations
            )
            parts.append(f"Suggested skills (load if needed): {catalog}")
        if self.index is not None:
            if not self.index.files:
                try:
                    self.index.build()
                except Exception:  # noqa: BLE001
                    pass
            if self.index.files:
                summary = self.index.summary()
                parts.append(
                    f"Repo: {summary.get('file_count')} files, langs={summary.get('languages')}"
                )
        text = "\n\n".join(parts)
        if len(text) > self.max_chars:
            return text[: self.max_chars] + "\n[prompt truncated]"
        return text
