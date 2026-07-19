"""Coerce tool / JSON argument values from string-heavy LLM output."""

from __future__ import annotations

from typing import Any


def coerce_bool(value: Any, *, default: bool = False) -> bool:
    """Parse booleans from bool/int/str (Qwen XML params are often strings)."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"", "0", "false", "no", "off", "none", "null"}:
        return False
    if text in {"1", "true", "yes", "on"}:
        return True
    return default


def coerce_int(value: Any, *, default: int | None = None) -> int | None:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    text = str(value).strip()
    try:
        return int(text, 10)
    except ValueError:
        try:
            return int(float(text))
        except ValueError:
            return default
