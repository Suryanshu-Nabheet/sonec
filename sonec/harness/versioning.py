"""Harness versioning and tool-schema hashing — frozen for trajectory integrity."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from sonec import __version__
from sonec.core.types import ToolSpec

# Bump when the canonical loop / logging / success contract changes.
HARNESS_VERSION = "1.0.0"

# Training-critical tools only (Phase 0 freeze). Meta/analyzers stay out of this set.
CORE_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "fs_list",
        "fs_read",
        "fs_write",
        "fs_edit",
        "fs_search",
        "terminal_run",
        "git_status",
        "git_diff",
        "git_log",
        "git_branch",
        "index_build",
        "index_search",
        "index_symbols",
    }
)


def tool_schema_hash(specs: list[ToolSpec]) -> str:
    """Stable hash of the tool schemas presented to the model."""
    payload: list[dict[str, Any]] = []
    for spec in sorted(specs, key=lambda s: s.name):
        if spec.name not in CORE_TOOL_NAMES:
            continue
        payload.append(spec.to_openai())
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def filter_core_specs(specs: list[ToolSpec]) -> list[ToolSpec]:
    return [s for s in specs if s.name in CORE_TOOL_NAMES]


def run_metadata(*, model_id: str, tool_hash: str) -> dict[str, str]:
    return {
        "harness_version": HARNESS_VERSION,
        "package_version": __version__,
        "tool_schema_hash": tool_hash,
        "model_id": model_id,
    }
