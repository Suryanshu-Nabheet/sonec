"""Memory and planning tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from sonec.core.types import Message, Role
from sonec.llm.provider import MockProvider
from sonec.memory.store import FileMemoryStore
from sonec.planning.planner import Planner, heuristic_plan


def test_memory_notes(tmp_path: Path) -> None:
    store = FileMemoryStore(tmp_path / "mem")
    note_id = store.add_note("prefer pytest", tags=["testing"])
    assert note_id
    hits = store.search_notes("pytest")
    assert hits
    # reload
    store2 = FileMemoryStore(tmp_path / "mem")
    assert store2.search_notes("pytest")


def test_heuristic_plan() -> None:
    plan = heuristic_plan("Add logging")
    assert plan.steps
    assert plan.goal == "Add logging"


@pytest.mark.asyncio
async def test_planner_with_invalid_llm_json() -> None:
    provider = MockProvider([Message(role=Role.ASSISTANT, content="not-json")])
    plan = await Planner(provider).create_plan("Ship feature X")
    assert plan.steps
