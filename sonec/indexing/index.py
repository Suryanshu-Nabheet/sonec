"""Repository indexing: file inventory + lexical search."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from pathspec import PathSpec

from sonec.core.errors import IndexError_ as IndexError
from sonec.core.workspace import Workspace
from sonec.filesystem.service import DEFAULT_IGNORE_DIRS

DEFAULT_GLOBS = [
    "**/*.py",
    "**/*.ts",
    "**/*.tsx",
    "**/*.js",
    "**/*.jsx",
    "**/*.md",
    "**/*.json",
    "**/*.toml",
    "**/*.yaml",
    "**/*.yml",
    "**/*.rs",
    "**/*.go",
    "**/*.java",
]


@dataclass(frozen=True)
class IndexedFile:
    path: str
    size: int
    digest: str
    language: str
    line_count: int


def detect_language(path: Path) -> str:
    mapping = {
        ".py": "python",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".js": "javascript",
        ".jsx": "jsx",
        ".md": "markdown",
        ".json": "json",
        ".toml": "toml",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".rs": "rust",
        ".go": "go",
        ".java": "java",
    }
    return mapping.get(path.suffix.lower(), "text")


class RepositoryIndex:
    """Builds an in-memory index of workspace files for search and navigation."""

    def __init__(
        self,
        workspace: Workspace,
        *,
        max_file_bytes: int = 1_048_576,
        ignore_dirs: set[str] | None = None,
    ) -> None:
        self.workspace = workspace
        self.max_file_bytes = max_file_bytes
        self.ignore_dirs = ignore_dirs or DEFAULT_IGNORE_DIRS
        self.files: dict[str, IndexedFile] = {}
        self._contents: dict[str, str] = {}
        self._gitignore = self._load_gitignore()

    def _load_gitignore(self) -> PathSpec | None:
        gi = self.workspace.root / ".gitignore"
        if not gi.exists():
            return None
        lines = gi.read_text(encoding="utf-8", errors="replace").splitlines()
        return PathSpec.from_lines("gitwildmatch", lines)

    def _ignored(self, rel: str) -> bool:
        parts = Path(rel).parts
        if any(part in self.ignore_dirs for part in parts):
            return True
        return bool(self._gitignore and self._gitignore.match_file(rel))

    def build(self, patterns: list[str] | None = None) -> int:
        patterns = patterns or DEFAULT_GLOBS
        self.files.clear()
        self._contents.clear()
        seen: set[str] = set()
        for pattern in patterns:
            for path in self.workspace.root.glob(pattern):
                if not path.is_file():
                    continue
                rel = self.workspace.relative_to_root(path)
                if rel in seen or self._ignored(rel):
                    continue
                seen.add(rel)
                size = path.stat().st_size
                if size > self.max_file_bytes:
                    continue
                try:
                    text = path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]
                self.files[rel] = IndexedFile(
                    path=rel,
                    size=size,
                    digest=digest,
                    language=detect_language(path),
                    line_count=text.count("\n") + (1 if text else 0),
                )
                self._contents[rel] = text
        return len(self.files)

    def search(self, query: str, *, limit: int = 30) -> list[dict[str, object]]:
        if not self.files:
            self.build()
        needle = query.strip()
        if not needle:
            return []
        regex = re.compile(re.escape(needle), re.IGNORECASE)
        hits: list[dict[str, object]] = []
        for rel, content in self._contents.items():
            for line_no, line in enumerate(content.splitlines(), start=1):
                if regex.search(line):
                    hits.append({"path": rel, "line": line_no, "text": line.strip()[:240]})
                    if len(hits) >= limit:
                        return hits
        return hits

    def symbols(self, path: str | None = None) -> list[dict[str, object]]:
        """Extract a lightweight symbol list via regex (defs/classes/functions)."""
        if not self.files:
            self.build()
        targets = [path] if path else list(self.files)
        pattern = re.compile(
            r"^\s*(?:async\s+)?(?:def|class|function|const|let|var|fn|pub\s+fn|type|interface)\s+"
            r"([A-Za-z_][A-Za-z0-9_]*)"
        )
        found: list[dict[str, object]] = []
        for rel in targets:
            content = self._contents.get(rel)
            if content is None:
                continue
            for line_no, line in enumerate(content.splitlines(), start=1):
                match = pattern.search(line)
                if match:
                    found.append({"path": rel, "line": line_no, "name": match.group(1), "text": line.strip()})
        return found

    def summary(self) -> dict[str, object]:
        if not self.files:
            self.build()
        by_lang: dict[str, int] = {}
        for item in self.files.values():
            by_lang[item.language] = by_lang.get(item.language, 0) + 1
        return {
            "file_count": len(self.files),
            "languages": by_lang,
            "root": str(self.workspace.root),
        }

    def dump(self, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": self.summary(),
            "files": [item.__dict__ for item in self.files.values()],
        }
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def require_built(self) -> None:
        if not self.files:
            raise IndexError("Repository index is empty; call build() first")
