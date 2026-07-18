"""Filesystem tool tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from sonec.core.workspace import Workspace
from sonec.filesystem.service import FilesystemTools


@pytest.mark.asyncio
async def test_write_read_edit_search(tmp_path: Path) -> None:
    ws = Workspace(tmp_path)
    fs = FilesystemTools(ws)
    write = await fs.write_file({"path": "src/hello.py", "content": "print('hi')\n"})
    assert write.ok

    read = await fs.read_file({"path": "src/hello.py"})
    assert "print('hi')" in read.content

    edit = await fs.edit_file(
        {"path": "src/hello.py", "old_string": "print('hi')", "new_string": "print('hello')"}
    )
    assert edit.ok
    assert "hello" in (tmp_path / "src" / "hello.py").read_text()

    listed = await fs.list_dir({"path": "src"})
    assert any("hello.py" in e for e in listed.data["entries"])  # type: ignore[index]

    search = await fs.search({"query": "hello"})
    assert search.data["matches"]  # type: ignore[index]
