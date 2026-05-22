"""Tests for src/agents/risk_assessor.py."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.risk_assessor import RiskAssessorAgent, _parse_risk_response
from src.models.schemas import RiskAssessment


class TestParseRiskResponse:
    def test_returns_risk_assessment(self):
        raw = json.dumps(
            {
                "risk_factors": [],
                "overall_risk_score": 0.7,
                "summary": "High risk.",
            }
        )
        result = _parse_risk_response(raw, "AAPL")
        assert isinstance(result, RiskAssessment)
        assert result.ticker == "AAPL"

    def test_code_block_stripped_and_parsed(self):
        inner = json.dumps(
            {"risk_factors": [], "overall_risk_score": 0.3, "summary": "Low"}
        )
        raw = f"```json\n{inner}\n```"
        result = _parse_risk_response(raw, "TSLA")
        assert isinstance(result, RiskAssessment)
        assert result.ticker == "TSLA"

    def test_code_block_no_closing_fence(self):
        inner = json.dumps(
            {"risk_factors": [], "overall_risk_score": 0.5, "summary": "Mid"}
        )
        raw = f"```\n{inner}"
        result = _parse_risk_response(raw, "MSFT")
        assert isinstance(result, RiskAssessment)

    def test_invalid_json_returns_fallback(self):
        result = _parse_risk_response("not valid json {{{", "GOOG")
        assert isinstance(result, RiskAssessment)
        assert result.ticker == "GOOG"

    def test_empty_json_returns_fallback(self):
        result = _parse_risk_response("{}", "AMZN")
        assert isinstance(result, RiskAssessment)
        assert result.ticker == "AMZN"


class TestRiskAssessorAgentInit:
    def test_default_model(self):
        agent = RiskAssessorAgent()
        assert agent._model is not None
        assert isinstance(agent._model, str)

    def test_custom_model(self):
        agent = RiskAssessorAgent(model="claude-3-haiku-20240307")
        assert agent._model == "claude-3-haiku-20240307"


@pytest.mark.asyncio
class TestRiskAssessorAgentAssess:
    async def test_assess_returns_risk_assessment(self):
        agent = RiskAssessorAgent()
        mock_msg = MagicMock()
        mock_msg.content = [
            MagicMock(
                text=json.dumps(
                    {
                        "risk_factors": [],
                        "overall_risk_score": 0.2,
                        "summary": "Manageable risk.",
                    }
                )
            )
        ]
        agent._client = AsyncMock()
        agent._client.messages.create = AsyncMock(return_value=mock_msg)

        raw_data = {
            "ticker": "NVDA",
            "market_overview": {"sector": "Technology"},
            "sec_summary": {"financials": {"revenue": 100}},
        }
        result = await agent.assess(raw_data)
        assert isinstance(result, RiskAssessment)
        assert result.ticker == "NVDA"

    async def test_assess_unknown_ticker(self):
        agent = RiskAssessorAgent()
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text="{}")]
        agent._client = AsyncMock()
        agent._client.messages.create = AsyncMock(return_value=mock_msg)

        result = await agent.assess({})
        assert isinstance(result, RiskAssessment)
        assert result.ticker == "UNKNOWN"
