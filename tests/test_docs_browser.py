"""Docs generator and browser import tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from sonec.browser.session import BrowserError, BrowserSession
from sonec.core.workspace import Workspace
from sonec.docsgen.generator import DocGenerator


def test_doc_generator(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("x = 1\n", encoding="utf-8")
    path = DocGenerator(Workspace(tmp_path)).write("docs/GENERATED.md")
    text = path.read_text(encoding="utf-8")
    assert "Overview" in text
    assert path.exists()


@pytest.mark.asyncio
async def test_browser_requires_extra() -> None:
    session = BrowserSession()
    # If playwright is installed this may succeed; we only assert error path when missing.
    try:
        import playwright  # noqa: F401
    except ImportError:
        with pytest.raises(BrowserError):
            await session.start()
