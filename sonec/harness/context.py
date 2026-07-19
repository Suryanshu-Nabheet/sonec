"""Thin production system prompt — skills and rules load on demand."""

from __future__ import annotations

from contextlib import suppress

from sonec.harness.versioning import HARNESS_VERSION
from sonec.indexing.index import RepositoryIndex
from sonec.models import PRODUCT_IDENTITY
from sonec.skills.registry import SkillsRegistry

MAX_ALWAYS_ON_CHARS = 3500

THIN_IDENTITY = f"""{PRODUCT_IDENTITY}

Harness v{HARNESS_VERSION}.

Operating rules:
- Prefer minimal, localized diffs inside the workspace.
- Paths must match the request exactly (notes/a.txt is not a.txt).
- When asked to create or write files, call fs_write immediately — do not list empty directories first.
- When asked to fix a bug in an existing file: fs_read it, then fs_write or fs_edit the fixed contents before finishing — never stop after only reading.
- For whole-file content changes (VERSION, single-line configs), prefer fs_write with the full new contents over fs_edit.
- When the goal is a question only, answer in text and do not create or edit files.
- Verify with terminal_run (or tests) before claiming completion when the goal asks to verify.
- Do not invent unread file contents.

Core tools: fs_list, fs_read, fs_write, fs_edit, fs_search, terminal_run, git_*, index_*.
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
        activations = self.skills.activate(goal or "software engineering", limit=3)
        if activations:
            catalog = "; ".join(
                f"{a.skill.id} ({a.skill.description[:60]})" for a in activations
            )
            # Skills are advisory catalog only — meta load tools are outside CORE freeze.
            parts.append(f"Relevant skill themes: {catalog}")
        if self.index is not None:
            if not self.index.files:
                with suppress(Exception):
                    self.index.build()
            if self.index.files:
                summary = self.index.summary()
                parts.append(
                    f"Repo: {summary.get('file_count')} files, langs={summary.get('languages')}"
                )
        text = "\n\n".join(parts)
        if len(text) > self.max_chars:
            return text[: self.max_chars] + "\n[prompt truncated]"
        return text
