"""Harness, skills, and rules tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from sonec.app import build_harness
from sonec.core.config import load_settings
from sonec.llm.provider import MockProvider
from sonec.rules.engine import RulesEngine
from sonec.skills.registry import SkillsRegistry


def test_skills_packaged() -> None:
    reg = SkillsRegistry()
    catalog = reg.catalog()
    ids = {c["id"] for c in catalog}
    assert "software-engineering" in ids
    assert "benchmark-swe" in ids
    assert "design-engineering" in ids
    activated = reg.activate("fix a failing pytest traceback bug", limit=5)
    assert activated
    assert any(a.skill.id in {"debugging", "software-engineering", "testing-tdd"} for a in activated)


def test_rules_include_prebuilt_pack() -> None:
    engine = RulesEngine()
    ids = {r.id for r in engine.rules}
    assert "sonec/core-agentic" in ids
    assert any(i.startswith("prebuilt/") for i in ids)
    assert "prebuilt/engineering-constitution" in ids
    assert "prebuilt/suryanshu-guidelines" in ids
    full = engine.get_full("prebuilt/engineering-constitution")
    assert "Production First" in full or "production" in full.lower()
    rendered = engine.render("refactor the authentication module")
    assert "SONEC Core Agentic Protocol" in rendered


@pytest.mark.asyncio
async def test_harness_smoke(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# x\n", encoding="utf-8")
    settings = load_settings(workspace=tmp_path, provider="mock", max_iterations=40)
    provider = MockProvider.harness_smoke("smoke goal")
    harness, *_ = build_harness(settings=settings, provider=provider, persist_memory=False)
    result = await harness.run("smoke goal")
    assert result.success
    assert result.iterations >= 5
    assert "DELIVER" in result.final_message or "Mock" in result.final_message
