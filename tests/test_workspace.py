"""Workspace sandbox tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from sonec.core.errors import SecurityError, WorkspaceError
from sonec.core.workspace import Workspace


def test_resolve_relative(tmp_path: Path) -> None:
    ws = Workspace(tmp_path)
    target = ws.resolve("a/b.txt")
    assert target == (tmp_path / "a" / "b.txt").resolve()


def test_rejects_escape(tmp_path: Path) -> None:
    ws = Workspace(tmp_path)
    with pytest.raises(SecurityError):
        ws.resolve("../outside.txt")


def test_missing_root(tmp_path: Path) -> None:
    with pytest.raises(WorkspaceError):
        Workspace(tmp_path / "missing")
