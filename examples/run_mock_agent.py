"""Minimal library example: multi-phase harness with a mock provider."""

from __future__ import annotations

import asyncio
from pathlib import Path

from sonec.app import build_harness
from sonec.llm import MockProvider


async def main() -> None:
    root = Path(__file__).resolve().parents[1]
    provider = MockProvider.harness_smoke("Build the index and summarize.")
    harness, _settings, workspace, registry = build_harness(
        workspace=root,
        provider=provider,
        persist_memory=False,
    )
    print(f"workspace={workspace.root}")
    print(f"tools={len(registry.names())}")
    result = await harness.run("Build the index and summarize.")
    print(result.final_message)
    print(f"success={result.success} iterations={result.iterations}")


if __name__ == "__main__":
    asyncio.run(main())
