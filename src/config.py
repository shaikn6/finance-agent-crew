"""Application settings, loaded from environment variables and an optional .env."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigurationError(RuntimeError):
    """Raised when required configuration is missing or inconsistent."""


class Settings(BaseSettings):
    """Runtime configuration.

    Each field is read from the environment variable of the same name
    (case-insensitive), falling back to a ``.env`` file in the working
    directory and then to the default below.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    anthropic_api_key: str = ""
    anthropic_model: str = "claude-haiku-5-5"
    alpha_vantage_api_key: str = "demo"
    sec_user_agent_contact: str = Field(
        "research@example.com",
        description="Contact e-mail sent in the User-Agent of outbound data requests",
    )
    offline_mode: bool = Field(
        False, description="Swap the LLM and data fetchers for deterministic fakes"
    )
    api_keys: str = Field(
        "", description="Comma-separated keys accepted in the X-API-Key header"
    )
    rate_limit_per_minute: int = Field(30, ge=1)

    @property
    def api_key_set(self) -> frozenset[str]:
        """The configured API keys, split on commas and stripped."""
        return frozenset(key.strip() for key in self.api_keys.split(",") if key.strip())

    def validate_for_serving(self) -> None:
        """Fail fast if the HTTP API cannot run safely with this configuration."""
        if self.offline_mode:
            return
        missing = []
        if not self.anthropic_api_key:
            missing.append("ANTHROPIC_API_KEY (the Anthropic key used by the analysts)")
        if not self.api_key_set:
            missing.append("API_KEYS (comma-separated keys clients send as X-API-Key)")
        if missing:
            raise ConfigurationError(
                "Missing required configuration: "
                + "; ".join(missing)
                + ". Set it in the environment or .env, or set OFFLINE_MODE=1 "
                "to run the keyless demo."
            )


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings (cached; read once)."""
    return Settings()
