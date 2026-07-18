"""Harness package — versioning, context, compaction, trajectories."""

from sonec.harness.compaction import compact_messages
from sonec.harness.context import ContextAssembler
from sonec.harness.trajectory import TrajectoryLogger
from sonec.harness.versioning import CORE_TOOL_NAMES, HARNESS_VERSION, tool_schema_hash

__all__ = [
    "CORE_TOOL_NAMES",
    "ContextAssembler",
    "HARNESS_VERSION",
    "TrajectoryLogger",
    "compact_messages",
    "tool_schema_hash",
]
