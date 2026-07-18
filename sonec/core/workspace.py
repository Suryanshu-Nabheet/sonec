"""Workspace boundary enforcement for filesystem and process tools."""

from __future__ import annotations

from pathlib import Path

from sonec.core.errors import SecurityError, WorkspaceError


class Workspace:
    """Restricts tool I/O to a single resolved root directory.

    All paths provided by models or users are resolved against ``root``
    and rejected if they escape via ``..`` or symlinks outside the root.
    """

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        if not self.root.exists():
            raise WorkspaceError(f"Workspace does not exist: {self.root}")
        if not self.root.is_dir():
            raise WorkspaceError(f"Workspace is not a directory: {self.root}")

    def resolve(self, relative_or_absolute: str | Path) -> Path:
        candidate = Path(relative_or_absolute)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        resolved = candidate.resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise SecurityError(
                f"Path escapes workspace root ({self.root}): {relative_or_absolute}"
            ) from exc
        return resolved

    def relative_to_root(self, path: Path) -> str:
        resolved = self.resolve(path)
        return str(resolved.relative_to(self.root))

    def ensure_parent(self, path: Path) -> None:
        parent = path.parent
        if not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)
