"""Indexing tests."""

from __future__ import annotations

from pathlib import Path

from sonec.core.workspace import Workspace
from sonec.indexing.index import RepositoryIndex


def test_index_build_and_search(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text("def greet():\n    return 'hi'\n", encoding="utf-8")
    index = RepositoryIndex(Workspace(tmp_path))
    assert index.build() >= 1
    hits = index.search("greet")
    assert hits
    symbols = index.symbols("pkg/mod.py")
    assert any(s["name"] == "greet" for s in symbols)
