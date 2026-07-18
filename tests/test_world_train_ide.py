"""WorldBench, corpora, train export, local provider tests."""

from __future__ import annotations

from pathlib import Path

from sonec.core.config import load_settings
from sonec.eval.corpora import DEFAULT_CORPORA, write_default_yaml
from sonec.eval.worldbench import build_worldbench_tasks, write_worldbench
from sonec.ide.mcp_server import TOOLS
from sonec.training.export import export_from_rollouts
from sonec.training.rollouts import run_rollouts_sync


def test_worldbench_size_and_tags() -> None:
    tasks = build_worldbench_tasks()
    assert len(tasks) >= 30
    tags = {t for task in tasks for t in task.tags}
    assert "vscode" in tags
    assert "bun" in tags
    assert "codex" in tags or "agent" in tags


def test_write_worldbench(tmp_path: Path) -> None:
    path = write_worldbench(tmp_path / "worldbench_v1.json")
    assert path.exists()
    assert "sealed" in path.read_text(encoding="utf-8")


def test_corpora_yaml(tmp_path: Path) -> None:
    path = write_default_yaml(tmp_path / "corpora.yaml")
    assert path.exists()
    assert any(r["id"] == "vscode-extension-samples" for r in DEFAULT_CORPORA)


def test_local_settings_defaults() -> None:
    settings = load_settings()
    assert settings.provider == "local"
    assert settings.model == "sonec"
    assert settings.resolved_base_url().endswith("/v1")
    assert settings.require_api_key() == "local"


def test_train_export_from_rollouts(tmp_path: Path) -> None:
    from sonec.eval.sonecbench import build_sonecbench_tasks

    tasks = build_sonecbench_tasks()[:1]
    run_rollouts_sync(tasks, tmp_path / "rollouts", group_size=1)
    written = export_from_rollouts(
        tmp_path / "rollouts" / "rollouts.jsonl",
        tmp_path / "train",
        sealed_ids=set(),
    )
    assert "chat" in written
    assert written["chat"].exists()
    assert (tmp_path / "train" / "manifest.json").exists()


def test_mcp_tools_registered() -> None:
    names = {t["name"] for t in TOOLS}
    assert "sonec_run" in names
    assert "sonec_version" in names
