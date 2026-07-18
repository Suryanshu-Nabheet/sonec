"""Terminal and git tool tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from sonec.core.errors import SecurityError
from sonec.core.workspace import Workspace
from sonec.git.service import GitService
from sonec.terminal.service import TerminalService


@pytest.mark.asyncio
async def test_terminal_echo(tmp_path: Path) -> None:
    terminal = TerminalService(Workspace(tmp_path))
    result = await terminal.run("echo sonec-ok")
    assert result["exit_code"] == 0
    assert "sonec-ok" in result["stdout"]


@pytest.mark.asyncio
async def test_terminal_blocks_network_by_default(tmp_path: Path) -> None:
    terminal = TerminalService(Workspace(tmp_path), allow_network=False)
    with pytest.raises(SecurityError):
        await terminal.run("curl https://example.com")


@pytest.mark.asyncio
async def test_git_status_in_repo(tmp_path: Path) -> None:
    terminal = TerminalService(Workspace(tmp_path))
    await terminal.run("git init")
    await terminal.run("git config user.email test@example.com")
    await terminal.run("git config user.name Test")
    (tmp_path / "a.txt").write_text("a\n", encoding="utf-8")
    git = GitService(Workspace(tmp_path), terminal)
    status = await git.status({})
    assert status.ok
    assert "a.txt" in status.content
