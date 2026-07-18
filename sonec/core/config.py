"""Runtime settings for sonec."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from sonec.core.errors import ConfigError
from sonec.models import (
    DEFAULT_LOCAL_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
)


class Settings(BaseSettings):
    """Provider-agnostic settings. Default: local OpenAI-compatible inference."""

    model_config = SettingsConfigDict(
        env_prefix="SONEC_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    workspace: Path = Field(default_factory=lambda: Path.cwd())
    # local | openai_compatible | openai | mock  (ollama kept as alias of local)
    provider: Literal["local", "openai", "openai_compatible", "mock", "ollama"] = (
        DEFAULT_PROVIDER
    )
    model: str = DEFAULT_MODEL
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.1
    max_tokens: int = 8192
    max_iterations: int = 32
    request_timeout_s: float = 120.0
    terminal_timeout_s: float = 60.0
    allow_network_tools: bool = False
    index_max_file_bytes: int = 1_048_576
    memory_dir: Path | None = None
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    openai_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENAI_API_KEY", "openai_api_key"),
    )

    @field_validator("workspace", mode="before")
    @classmethod
    def _resolve_workspace(cls, value: object) -> Path:
        return Path(str(value)).expanduser().resolve()

    def _normalized_provider(self) -> str:
        if self.provider == "ollama":
            return "local"
        return self.provider

    def resolved_api_key(self) -> str | None:
        if self.api_key:
            return self.api_key
        if self._normalized_provider() in {"local", "mock"}:
            return self.api_key or "local"
        return self.openai_api_key

    def resolved_base_url(self) -> str:
        if self.base_url:
            return self.base_url.rstrip("/")
        provider = self._normalized_provider()
        if provider == "local":
            return DEFAULT_LOCAL_BASE_URL.rstrip("/")
        if provider == "openai":
            return "https://api.openai.com/v1"
        if provider == "openai_compatible":
            raise ConfigError("SONEC_BASE_URL is required when provider=openai_compatible")
        return DEFAULT_LOCAL_BASE_URL.rstrip("/")

    def memory_path(self) -> Path:
        if self.memory_dir is not None:
            return self.memory_dir.expanduser().resolve()
        return (self.workspace / ".sonec" / "memory").resolve()

    def require_api_key(self) -> str:
        if self._normalized_provider() in {"mock", "local"}:
            return self.resolved_api_key() or "local"
        key = self.resolved_api_key()
        if not key:
            raise ConfigError(
                "No API key configured. Set SONEC_API_KEY or OPENAI_API_KEY. "
                "For local OpenAI-compatible servers: SONEC_PROVIDER=local "
                f"SONEC_BASE_URL={DEFAULT_LOCAL_BASE_URL} SONEC_MODEL=sonec"
            )
        return key


def load_settings(**overrides: object) -> Settings:
    return Settings(**overrides)  # type: ignore[arg-type]
