"""Shared fixtures: isolate every test from the host environment and .env."""

from __future__ import annotations

import pytest

from src.config import Settings, get_settings

_SETTINGS_ENV_VARS = (
    "ANTHROPIC_MODEL",
    "ALPHA_VANTAGE_API_KEY",
    "SEC_USER_AGENT_CONTACT",
    "OFFLINE_MODE",
    "API_KEYS",
    "RATE_LIMIT_PER_MINUTE",
)


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch):
    """Give each test a clean, live-mode configuration with a dummy LLM key."""
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    for name in _SETTINGS_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
