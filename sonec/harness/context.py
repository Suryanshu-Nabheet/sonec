"""Thin production system prompt — skills and rules load on demand."""

from __future__ import annotations

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
- Verify before claiming completion.
- Do not invent unread file contents.

Tools: filesystem, terminal, git, index.
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
