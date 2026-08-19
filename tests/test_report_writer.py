"""Tests for src/agents/report_writer.py."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.report_writer import ReportWriterAgent, _parse_report_response
from src.models.schemas import (
    FundamentalAnalysis,
    InvestmentReport,
    RiskAssessment,
    SentimentAnalysis,
    Signal,
)


def _make_fundamental() -> FundamentalAnalysis:
    return FundamentalAnalysis(
        ticker="AAPL",
        signal=Signal.BUY,
        pe_ratio=25.0,
        debt_to_equity=0.5,
        reasoning="Strong fundamentals.",
    )


def _make_sentiment() -> SentimentAnalysis:
    return SentimentAnalysis(
        ticker="AAPL",
        news_sentiment_score=0.6,
        management_tone="positive",
        headline_count=10,
    )


def _make_risk() -> RiskAssessment:
    return RiskAssessment(
        ticker="AAPL",
        overall_risk_level="low",
        key_risks_summary="Low risk.",
    )


class TestParseReportResponse:
    def test_valid_json_buy_signal(self):
        raw = json.dumps(
            {
                "overall_signal": "BUY",
                "executive_summary": "Strong buy.",
                "investment_thesis": "Growth thesis.",
                "bull_case": "Upside.",
                "bear_case": "Downside.",
            }
        )
        result = _parse_report_response(
            raw, "AAPL", "Apple", "Tech", "Software", None, None, None
        )
        assert isinstance(result, InvestmentReport)
        assert result.overall_signal == Signal.BUY
        assert result.executive_summary == "Strong buy."
        assert result.investment_thesis == "Growth thesis."
        assert result.bull_case == "Upside."
        assert result.bear_case == "Downside."
        assert result.company_name == "Apple"
        assert result.sector == "Tech"
        assert result.industry == "Software"

    def test_code_block_stripped(self):
        inner = json.dumps({"overall_signal": "HOLD", "executive_summary": "Hold."})
        raw = f"```json\n{inner}\n```"
        result = _parse_report_response(
            raw, "MSFT", "Microsoft", "Tech", "Software", None, None, None
        )
        assert result.overall_signal == Signal.HOLD
        assert result.executive_summary == "Hold."

    def test_code_block_no_closing_fence(self):
        inner = json.dumps({"overall_signal": "SELL"})
        raw = f"```\n{inner}"
        result = _parse_report_response(raw, "X", "X Corp", "", "", None, None, None)
        assert result.overall_signal == Signal.SELL

    def test_invalid_signal_falls_back(self):
        raw = json.dumps({"overall_signal": "STRONG_BUY"})
        result = _parse_report_response(raw, "T", "AT&T", "", "", None, None, None)
        assert result.overall_signal == Signal.INSUFFICIENT_DATA

    def test_invalid_json_returns_default(self):
        result = _parse_report_response(
            "not json", "ERR", "Err Co", "", "", None, None, None
        )
        assert result.overall_signal == Signal.INSUFFICIENT_DATA
        assert result.executive_summary == "Report generation failed."

    def test_passes_through_analyses(self):
        raw = json.dumps({"overall_signal": "BUY"})
        f = _make_fundamental()
        s = _make_sentiment()
        r = _make_risk()
        result = _parse_report_response(raw, "AAPL", "Apple", "Tech", "SW", f, s, r)
        assert result.fundamental_analysis is f
        assert result.sentiment_analysis is s
        assert result.risk_assessment is r

    def test_missing_optional_fields_use_empty(self):
        raw = json.dumps({"overall_signal": "HOLD"})
        result = _parse_report_response(
            raw, "AAPL", "Apple", "Tech", "SW", None, None, None
        )
        assert result.investment_thesis == ""
        assert result.bull_case == ""
        assert result.bear_case == ""


class TestReportWriterAgentInit:
    def test_default_model(self):
        agent = ReportWriterAgent()
        assert isinstance(agent._model, str)

    def test_custom_model(self):
        agent = ReportWriterAgent(model="claude-3-haiku-20240307")
        assert agent._model == "claude-3-haiku-20240307"


@pytest.mark.asyncio
class TestReportWriterAgentWrite:
    async def test_write_with_no_analyses(self):
        agent = ReportWriterAgent()
        mock_msg = MagicMock()
        mock_msg.content = [
            MagicMock(
                text=json.dumps(
                    {
                        "overall_signal": "INSUFFICIENT_DATA",
                        "executive_summary": "No data.",
                    }
                )
            )
        ]
        agent._client = AsyncMock()
        agent._client.messages.create = AsyncMock(return_value=mock_msg)

        result = await agent.write("UNKN")
        assert isinstance(result, InvestmentReport)
        assert result.ticker == "UNKN"
        assert result.overall_signal == Signal.INSUFFICIENT_DATA
        assert result.executive_summary == "No data."

    async def test_write_with_raw_data(self):
        agent = ReportWriterAgent()
        mock_msg = MagicMock()
        mock_msg.content = [
            MagicMock(
                text=json.dumps(
                    {
                        "overall_signal": "BUY",
                        "executive_summary": "Apple looks strong.",
                    }
                )
            )
        ]
        agent._client = AsyncMock()
        agent._client.messages.create = AsyncMock(return_value=mock_msg)

        result = await agent.write(
            "AAPL",
            raw_data={
                "market_overview": {
                    "name": "Apple Inc.",
                    "sector": "Tech",
                    "industry": "SW",
                }
            },
        )
        assert isinstance(result, InvestmentReport)
        assert result.company_name == "Apple Inc."
        assert result.sector == "Tech"
        assert result.industry == "SW"
        assert result.overall_signal == Signal.BUY

    async def test_write_with_fundamental_sentiment_and_risk(self):
        """End-to-end: write() must not crash when given real analyst outputs."""
        agent = ReportWriterAgent()
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text=json.dumps({"overall_signal": "BUY"}))]
        agent._client = AsyncMock()
        agent._client.messages.create = AsyncMock(return_value=mock_msg)

        fundamental = _make_fundamental()
        sentiment = _make_sentiment()
        risk = _make_risk()

        result = await agent.write(
            "AAPL", fundamental=fundamental, sentiment=sentiment, risk=risk
        )
        assert isinstance(result, InvestmentReport)
        assert result.fundamental_analysis is fundamental
        assert result.sentiment_analysis is sentiment
        assert result.risk_assessment is risk

        # The prompt built from these analyses must reference the real
        # schema fields (news_sentiment_score, overall_risk_score) rather
        # than crashing with AttributeError.
        prompt = agent._client.messages.create.await_args.kwargs["messages"][0][
            "content"
        ]
        assert "0.60" in prompt
        assert f"{risk.overall_risk_score:.2f}" in prompt
