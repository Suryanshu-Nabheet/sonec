"""CLI helpers for provider selection."""

from __future__ import annotations

from typing import Any

from sonec.models import DEFAULT_MODEL, DEFAULT_PROVIDER


def provider_overrides(
    *,
    mock: bool = False,
    live: bool = False,
    provider: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if mock and not live:
        overrides["provider"] = "mock"
    elif provider:
        overrides["provider"] = "local" if provider == "ollama" else provider
    elif live:
        overrides["provider"] = DEFAULT_PROVIDER
    if model:
        overrides["model"] = model
    elif overrides.get("provider") in {"local", "ollama"} and "model" not in overrides:
        overrides["model"] = DEFAULT_MODEL
    return overrides
