"""Typed exceptions for SONEC."""

from __future__ import annotations


class SonecError(Exception):
    """Base error for all SONEC failures."""

    def __init__(self, message: str, *, code: str = "sonec_error") -> None:
        super().__init__(message)
        self.message = message
        self.code = code


class ConfigError(SonecError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="config_error")


class ValidationError(SonecError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="validation_error")


class SecurityError(SonecError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="security_error")


class ToolError(SonecError):
    def __init__(self, message: str, *, tool: str | None = None) -> None:
        super().__init__(message, code="tool_error")
        self.tool = tool


class LLMError(SonecError):
    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message, code="llm_error")
        self.status_code = status_code


class WorkspaceError(SonecError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="workspace_error")


class IndexError_(SonecError):
    """Repository indexing failures (named to avoid shadowing builtins)."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="index_error")


class EvalError(SonecError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="eval_error")
