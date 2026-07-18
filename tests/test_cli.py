"""CLI smoke tests."""

from __future__ import annotations

from sonec.cli.main import app
from typer.testing import CliRunner

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "0.2.0" in result.stdout
