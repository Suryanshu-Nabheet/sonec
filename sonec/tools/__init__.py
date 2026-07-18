"""Tools package."""

from sonec.tools.registry import FunctionTool, Tool, ToolRegistry

__all__ = ["FunctionTool", "Tool", "ToolRegistry", "build_default_registry"]


def __getattr__(name: str):
    if name == "build_default_registry":
        from sonec.tools.builtin import build_default_registry

        return build_default_registry
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
