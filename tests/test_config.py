"""Tests for src/config.py and the settings-driven wiring in agents and tools."""

from __future__ import annotations

import anthropic
import pytest

from src.agents.fundamental_analyst import FundamentalAnalystAgent
from src.agents.report_writer import ReportWriterAgent
from src.agents.risk_assessor import RiskAssessorAgent
from src.agents.sentiment_analyst import SentimentAnalystAgent
from src.config import ConfigurationError, Settings, get_settings
from src.llm import get_llm_client
from src.offline import OfflineLLMClient
from src.tools.market_data import MarketDataClient
from src.tools.news_fetcher import NewsFetcher
from src.tools.sec_scraper import SECEdgarScraper

_AGENTS = [
    FundamentalAnalystAgent,
    SentimentAnalystAgent,
    RiskAssessorAgent,
    ReportWriterAgent,
]


class TestSettings:
    def test_defaults(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY")
        settings = Settings()
        assert settings.anthropic_api_key == ""
        assert settings.anthropic_model == "claude-haiku-5-5"
        assert settings.alpha_vantage_api_key == "demo"
        assert settings.offline_mode is False
        assert settings.api_key_set == frozenset()
        assert settings.rate_limit_per_minute == 30

    def test_reads_existing_env_var_names(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-live")
        monkeypatch.setenv("ANTHROPIC_MODEL", "some-model")
        monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "av-key")
        monkeypatch.setenv("OFFLINE_MODE", "1")
        settings = Settings()
        assert settings.anthropic_api_key == "sk-live"
        assert settings.anthropic_model == "some-model"
        assert settings.alpha_vantage_api_key == "av-key"
        assert settings.offline_mode is True

    def test_loads_dotenv_file(self, monkeypatch, tmp_path):
        monkeypatch.delenv("ANTHROPIC_API_KEY")
        env_file = tmp_path / ".env"
        env_file.write_text(
            "ANTHROPIC_API_KEY=from-dotenv\nOFFLINE_MODE=1\nUNRELATED=ignored\n"
        )
        settings = Settings(_env_file=env_file)
        assert settings.anthropic_api_key == "from-dotenv"
        assert settings.offline_mode is True

    def test_api_keys_are_split_and_stripped(self, monkeypatch):
        monkeypatch.setenv("API_KEYS", " alpha , beta,, ")
        assert Settings().api_key_set == frozenset({"alpha", "beta"})

    def test_rate_limit_must_be_positive(self, monkeypatch):
        monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "0")
        with pytest.raises(ValueError):
            Settings()

    def test_get_settings_is_cached(self):
        assert get_settings() is get_settings()


class TestValidateForServing:
    def test_offline_mode_needs_no_keys(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY")
        monkeypatch.setenv("OFFLINE_MODE", "1")
        Settings().validate_for_serving()

    def test_live_mode_requires_anthropic_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY")
        monkeypatch.setenv("API_KEYS", "alpha")
        with pytest.raises(ConfigurationError, match="ANTHROPIC_API_KEY"):
            Settings().validate_for_serving()

    def test_live_mode_requires_api_keys(self):
        with pytest.raises(ConfigurationError, match="API_KEYS"):
            Settings().validate_for_serving()

    def test_live_mode_with_both_is_valid(self, monkeypatch):
        monkeypatch.setenv("API_KEYS", "alpha")
        Settings().validate_for_serving()


class TestGetLLMClient:
    def test_missing_key_fails_with_clear_message(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY")
        with pytest.raises(ConfigurationError) as excinfo:
            get_llm_client()
        assert "ANTHROPIC_API_KEY" in str(excinfo.value)
        assert "OFFLINE_MODE" in str(excinfo.value)

    def test_live_mode_returns_anthropic_client(self):
        assert isinstance(get_llm_client(), anthropic.AsyncAnthropic)

    def test_offline_mode_returns_fake_without_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY")
        monkeypatch.setenv("OFFLINE_MODE", "1")
        assert isinstance(get_llm_client(), OfflineLLMClient)


class TestSettingsWiring:
    @pytest.mark.parametrize("agent_cls", _AGENTS)
    def test_agents_default_to_current_model(self, agent_cls):
        assert agent_cls()._model == "claude-haiku-5-5"

    @pytest.mark.parametrize("agent_cls", _AGENTS)
    def test_agents_use_model_from_settings(self, agent_cls, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_MODEL", "configured-model")
        assert agent_cls()._model == "configured-model"

    @pytest.mark.parametrize("agent_cls", _AGENTS)
    def test_explicit_model_wins(self, agent_cls, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_MODEL", "configured-model")
        assert agent_cls(model="explicit")._model == "explicit"

    async def test_sec_user_agent_contact_is_configurable(self, monkeypatch):
        monkeypatch.setenv("SEC_USER_AGENT_CONTACT", "ops@mycompany.test")
        async with SECEdgarScraper() as scraper:
            assert scraper._client.headers["User-Agent"] == (
                "FinanceAgentCrew ops@mycompany.test"
            )

    async def test_news_user_agent_uses_same_contact(self, monkeypatch):
        monkeypatch.setenv("SEC_USER_AGENT_CONTACT", "ops@mycompany.test")
        async with NewsFetcher() as fetcher:
            assert "ops@mycompany.test" in fetcher._client.headers["User-Agent"]

    async def test_market_client_uses_key_from_settings(self, monkeypatch):
        monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "av-from-settings")
        async with MarketDataClient() as client:
            assert client._api_key == "av-from-settings"
