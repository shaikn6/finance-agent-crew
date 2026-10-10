"""Tests for the offline/demo mode fakes in src/offline.py."""

from __future__ import annotations

import asyncio

import pytest

from src.agents.data_gatherer import DataGathererAgent
from src.agents.fundamental_analyst import FundamentalAnalystAgent
from src.agents.report_writer import ReportWriterAgent
from src.agents.risk_assessor import RiskAssessorAgent
from src.agents.sentiment_analyst import SentimentAnalystAgent
from src.models.schemas import Signal, Tone
from src.offline import OfflineLLMClient


@pytest.fixture
def offline(monkeypatch):
    """Offline mode with no keys at all, and any real HTTP client forbidden."""
    monkeypatch.setenv("OFFLINE_MODE", "1")
    monkeypatch.delenv("ANTHROPIC_API_KEY")

    def _no_network(*args, **kwargs):
        raise AssertionError("offline mode must not create an HTTP client")

    monkeypatch.setattr("httpx.AsyncClient", _no_network)


async def _run_pipeline(ticker: str):
    raw = await DataGathererAgent().gather(ticker)
    fundamental, sentiment, risk = await asyncio.gather(
        FundamentalAnalystAgent().analyze(raw),
        SentimentAnalystAgent().analyze(raw),
        RiskAssessorAgent().assess(raw),
    )
    report = await ReportWriterAgent().write(
        ticker, fundamental=fundamental, sentiment=sentiment, risk=risk, raw_data=raw
    )
    return raw, report


class TestOfflineGatherer:
    async def test_returns_complete_bundle_without_network(self, offline):
        raw = await DataGathererAgent().gather("demo")
        assert raw["ticker"] == "DEMO"
        assert raw["errors"] == []
        assert "offline" in raw["company_name"].lower()
        assert raw["sec_summary"]["financials"]["revenue"] > 0
        assert len(raw["sec_summary"]["revenue_history"]) == 4
        assert raw["market_overview"]["pe_ratio"] is not None
        assert raw["current_quote"]["price"] is not None
        assert len(raw["price_history"]) == 12
        assert raw["news_headlines"]
        assert raw["earnings_snippets"]
        assert raw["recent_filings"]

    async def test_include_flags_are_respected(self, offline):
        raw = await DataGathererAgent(include_news=False, include_sec=False).gather(
            "DEMO"
        )
        assert raw["news_headlines"] == []
        assert raw["sec_summary"] == {}
        assert raw["market_overview"]


class TestOfflinePipeline:
    async def test_full_pipeline_produces_populated_report(self, offline):
        _, report = await _run_pipeline("DEMO")
        assert report.ticker == "DEMO"
        assert "offline" in report.company_name.lower()
        assert report.overall_signal == Signal.HOLD
        assert report.executive_summary
        assert report.investment_thesis
        assert report.bull_case
        assert report.bear_case
        assert report.fundamental_analysis.signal == Signal.HOLD
        assert report.fundamental_analysis.gross_margin == 40.0
        assert report.fundamental_analysis.revenue_cagr_3y is not None
        assert report.sentiment_analysis.management_tone == Tone.POSITIVE
        assert report.sentiment_analysis.headline_count == 3
        assert report.risk_assessment.overall_risk_level == "medium"
        assert len(report.risk_assessment.risk_factors) == 2

    async def test_pipeline_is_deterministic(self, offline):
        timestamps = {
            "report_date": True,
            "fundamental_analysis": {"data_as_of"},
            "sentiment_analysis": {"analyzed_at"},
        }
        _, first = await _run_pipeline("DEMO")
        _, second = await _run_pipeline("DEMO")
        assert first.model_dump(exclude=timestamps) == second.model_dump(
            exclude=timestamps
        )


class TestOfflineLLMClient:
    async def test_unrecognised_prompt_returns_empty_json(self):
        message = await OfflineLLMClient().messages.create(
            model="any", max_tokens=16, messages=[{"role": "user", "content": "hi"}]
        )
        assert message.content[0].text == "{}"
