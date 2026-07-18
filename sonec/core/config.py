"""Configuration loaded from environment and optional files."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from sonec.core.errors import ConfigError


class Settings(BaseSettings):
    """Runtime settings for SONEC.

    Secrets are never hardcoded. Prefer environment variables:
    ``MOONSHOT_API_KEY``, ``SONEC_API_KEY``, ``OPENAI_API_KEY``.
    """

    model_config = SettingsConfigDict(
        env_prefix="SONEC_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    workspace: Path = Field(default_factory=lambda: Path.cwd())
    provider: Literal["moonshot", "openai", "openai_compatible", "mock"] = "moonshot"
    model: str = "kimi-k3"
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.2
    max_tokens: int = 8192
    max_iterations: int = 32
    request_timeout_s: float = 120.0
    terminal_timeout_s: float = 60.0
    allow_network_tools: bool = False
    index_max_file_bytes: int = 1_048_576
    memory_dir: Path | None = None
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    moonshot_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("MOONSHOT_API_KEY", "moonshot_api_key"),
    )
    openai_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENAI_API_KEY", "openai_api_key"),
    )

    @field_validator("workspace", mode="before")
    @classmethod
    def _resolve_workspace(cls, value: object) -> Path:
        path = Path(str(value)).expanduser().resolve()
        return path

    def resolved_api_key(self) -> str | None:
        if self.api_key:
            return self.api_key
        if self.provider == "moonshot":
            return self.moonshot_api_key
        if self.provider in {"openai", "openai_compatible"}:
            return self.openai_api_key or self.moonshot_api_key
        return self.moonshot_api_key or self.openai_api_key

    def resolved_base_url(self) -> str:
        if self.base_url:
            return self.base_url.rstrip("/")
        if self.provider == "moonshot":
            return "https://api.moonshot.ai/v1"
        if self.provider == "openai":
            return "https://api.openai.com/v1"
        if self.base_url is None and self.provider == "openai_compatible":
            raise ConfigError(
                "SONEC_BASE_URL is required when provider=openai_compatible"
            )
        return "https://api.moonshot.ai/v1"

    def memory_path(self) -> Path:
        if self.memory_dir is not None:
            return self.memory_dir.expanduser().resolve()
        return (self.workspace / ".sonec" / "memory").resolve()

    def require_api_key(self) -> str:
        key = self.resolved_api_key()
        if not key and self.provider != "mock":
            raise ConfigError(
                "No API key configured. Set SONEC_API_KEY or MOONSHOT_API_KEY "
                "(or OPENAI_API_KEY for OpenAI providers)."
            )
        return key or "mock"


def load_settings(**overrides: object) -> Settings:
    """Load settings, applying optional overrides for tests and CLI."""
    return Settings(**overrides)  # type: ignore[arg-type]
