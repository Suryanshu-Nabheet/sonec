"""Safe filesystem tools scoped to a workspace."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from sonec.core.types import ToolResult, ToolSpec
from sonec.core.workspace import Workspace
from sonec.tools.registry import FunctionTool, Tool, json_content

DEFAULT_IGNORE_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".sonec",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
}


class FilesystemTools:
    def __init__(self, workspace: Workspace, *, max_read_bytes: int = 512_000) -> None:
        self.workspace = workspace
        self.max_read_bytes = max_read_bytes

    def tools(self) -> list[Tool]:
        return [
            FunctionTool(
                ToolSpec(
                    name="fs_list",
                    description="List files and directories under a workspace-relative path.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Relative directory path (default '.')",
                            },
                            "recursive": {"type": "boolean", "default": False},
                            "max_entries": {"type": "integer", "default": 200},
                        },
                        "additionalProperties": False,
                    },
                ),
                self.list_dir,
            ),
            FunctionTool(
                ToolSpec(
                    name="fs_read",
                    description="Read a text file from the workspace.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "offset": {
                                "type": "integer",
                                "description": "1-based start line",
                                "default": 1,
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Max lines to return",
                                "default": 400,
                            },
                        },
                        "required": ["path"],
                        "additionalProperties": False,
                    },
                ),
                self.read_file,
            ),
            FunctionTool(
                ToolSpec(
                    name="fs_write",
                    description="Create or overwrite a text file in the workspace.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["path", "content"],
                        "additionalProperties": False,
                    },
                ),
                self.write_file,
            ),
            FunctionTool(
                ToolSpec(
                    name="fs_edit",
                    description=(
                        "Replace an exact substring in a file. "
                        "Fails if old_string is missing or not unique."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "old_string": {"type": "string"},
                            "new_string": {"type": "string"},
                        },
                        "required": ["path", "old_string", "new_string"],
                        "additionalProperties": False,
                    },
                ),
                self.edit_file,
            ),
            FunctionTool(
                ToolSpec(
                    name="fs_search",
                    description="Search for a substring across text files in the workspace.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "path": {"type": "string", "default": "."},
                            "max_matches": {"type": "integer", "default": 50},
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                ),
                self.search,
            ),
        ]

    async def list_dir(self, arguments: Mapping[str, Any]) -> ToolResult:
        rel = str(arguments.get("path") or ".")
        recursive = bool(arguments.get("recursive", False))
        max_entries = int(arguments.get("max_entries") or 200)
        root = self.workspace.resolve(rel)
        if not root.exists():
            return ToolResult(tool_call_id="", name="fs_list", content=f"Not found: {rel}", ok=False)
        if not root.is_dir():
            return ToolResult(
                tool_call_id="", name="fs_list", content=f"Not a directory: {rel}", ok=False
            )

        entries: list[str] = []
        if recursive:
            for path in sorted(root.rglob("*")):
                if any(part in DEFAULT_IGNORE_DIRS for part in path.parts):
                    continue
                entries.append(self.workspace.relative_to_root(path))
                if len(entries) >= max_entries:
                    break
        else:
            for path in sorted(root.iterdir()):
                if path.name in DEFAULT_IGNORE_DIRS:
                    continue
                suffix = "/" if path.is_dir() else ""
                entries.append(self.workspace.relative_to_root(path) + suffix)
                if len(entries) >= max_entries:
                    break

        return ToolResult(
            tool_call_id="",
            name="fs_list",
            content=json_content({"path": rel, "entries": entries, "count": len(entries)}),
            ok=True,
            data={"entries": entries},
        )

    async def read_file(self, arguments: Mapping[str, Any]) -> ToolResult:
        rel = str(arguments["path"])
        offset = max(1, int(arguments.get("offset") or 1))
        limit = max(1, int(arguments.get("limit") or 400))
        path = self.workspace.resolve(rel)
        if not path.exists() or not path.is_file():
            return ToolResult(tool_call_id="", name="fs_read", content=f"Not found: {rel}", ok=False)
        size = path.stat().st_size
        if size > self.max_read_bytes:
            return ToolResult(
                tool_call_id="",
                name="fs_read",
                content=f"File too large ({size} bytes > {self.max_read_bytes})",
                ok=False,
            )
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        sliced = lines[offset - 1 : offset - 1 + limit]
        # Small / full-file reads: raw text so agents can copy into fs_edit/fs_write.
        # Large slices keep line numbers for navigation only (not for edit paste).
        if len(sliced) <= 80 and offset == 1 and limit >= len(lines):
            body = "\n".join(sliced)
            if text.endswith("\n") and body:
                body += "\n"
        else:
            body = "\n".join(f"{i + offset:>6}|{line}" for i, line in enumerate(sliced))
        return ToolResult(
            tool_call_id="",
            name="fs_read",
            content=body or "(empty file)",
            ok=True,
            data={"path": rel, "total_lines": len(lines)},
        )

    async def write_file(self, arguments: Mapping[str, Any]) -> ToolResult:
        rel = str(arguments["path"])
        content = str(arguments["content"])
        path = self.workspace.resolve(rel)
        self.workspace.ensure_parent(path)
        path.write_text(content, encoding="utf-8")
        return ToolResult(
            tool_call_id="",
            name="fs_write",
            content=f"Wrote {len(content)} characters to {rel}",
            ok=True,
            data={"path": rel, "bytes": len(content.encode('utf-8'))},
        )

    async def edit_file(self, arguments: Mapping[str, Any]) -> ToolResult:
        rel = str(arguments["path"])
        old = str(arguments["old_string"])
        new = str(arguments["new_string"])
        path = self.workspace.resolve(rel)
        if not path.exists():
            return ToolResult(tool_call_id="", name="fs_edit", content=f"Not found: {rel}", ok=False)
        text = path.read_text(encoding="utf-8")
        # Models often paste numbered fs_read lines (e.g. "     1|0.0.0") into old_string.
        candidates = [old]
        stripped = "\n".join(
            line.split("|", 1)[1] if "|" in line[:12] and line.lstrip()[:1].isdigit() else line
            for line in old.splitlines()
        )
        if stripped != old:
            candidates.append(stripped)
        match = next((c for c in candidates if text.count(c) == 1), None)
        if match is None:
            if any(text.count(c) == 0 for c in candidates):
                return ToolResult(
                    tool_call_id="",
                    name="fs_edit",
                    content="old_string not found in file",
                    ok=False,
                )
            return ToolResult(
                tool_call_id="",
                name="fs_edit",
                content="old_string matched multiple times; must be unique",
                ok=False,
            )
        path.write_text(text.replace(match, new, 1), encoding="utf-8")
        return ToolResult(
            tool_call_id="",
            name="fs_edit",
            content=f"Edited {rel}",
            ok=True,
            data={"path": rel},
        )

    async def search(self, arguments: Mapping[str, Any]) -> ToolResult:
        query = str(arguments["query"])
        rel = str(arguments.get("path") or ".")
        max_matches = int(arguments.get("max_matches") or 50)
        root = self.workspace.resolve(rel)
        matches: list[dict[str, Any]] = []
        paths: list[Path] = (
            [root] if root.is_file() else [p for p in root.rglob("*") if p.is_file()]
        )

        for path in paths:
            if any(part in DEFAULT_IGNORE_DIRS for part in path.parts):
                continue
            if path.stat().st_size > self.max_read_bytes:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for line_no, line in enumerate(text.splitlines(), start=1):
                if query in line:
                    matches.append(
                        {
                            "path": self.workspace.relative_to_root(path),
                            "line": line_no,
                            "text": line.strip()[:240],
                        }
                    )
                    if len(matches) >= max_matches:
                        return ToolResult(
                            tool_call_id="",
                            name="fs_search",
                            content=json_content({"query": query, "matches": matches}),
                            ok=True,
                            data={"matches": matches},
                        )
        return ToolResult(
            tool_call_id="",
            name="fs_search",
            content=json_content({"query": query, "matches": matches}),
            ok=True,
            data={"matches": matches},
        )
