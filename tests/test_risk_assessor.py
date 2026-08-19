"""Tests for src/agents/risk_assessor.py."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.risk_assessor import RiskAssessorAgent, _parse_risk_response
from src.models.schemas import RiskAssessment, RiskFactor


class TestParseRiskResponse:
    def test_returns_risk_assessment_with_factors(self):
        raw = json.dumps(
            {
                "risk_factors": [
                    {
                        "category": "market",
                        "description": "High volatility in tech sector",
                        "severity": "high",
                        "probability": "medium",
                    }
                ],
                "overall_risk_level": "high",
                "key_risks_summary": "High risk.",
            }
        )
        result = _parse_risk_response(raw, "AAPL")
        assert isinstance(result, RiskAssessment)
        assert result.ticker == "AAPL"
        assert result.overall_risk_level == "high"
        assert result.overall_risk_score == 0.75
        assert result.key_risks_summary == "High risk."
        assert len(result.risk_factors) == 1
        factor = result.risk_factors[0]
        assert isinstance(factor, RiskFactor)
        assert factor.category == "market"
        assert factor.description == "High volatility in tech sector"
        assert factor.severity == "high"
        assert factor.probability == "medium"

    def test_code_block_stripped_and_parsed(self):
        inner = json.dumps(
            {"risk_factors": [], "overall_risk_level": "low", "key_risks_summary": "Low"}
        )
        raw = f"```json\n{inner}\n```"
        result = _parse_risk_response(raw, "TSLA")
        assert result.ticker == "TSLA"
        assert result.overall_risk_level == "low"
        assert result.overall_risk_score == 0.25
        assert result.key_risks_summary == "Low"

    def test_code_block_no_closing_fence(self):
        inner = json.dumps(
            {"risk_factors": [], "overall_risk_level": "medium", "key_risks_summary": "Mid"}
        )
        raw = f"```\n{inner}"
        result = _parse_risk_response(raw, "MSFT")
        assert result.overall_risk_level == "medium"
        assert result.overall_risk_score == 0.5

    def test_invalid_json_returns_fallback(self):
        result = _parse_risk_response("not valid json {{{", "GOOG")
        assert result.ticker == "GOOG"
        assert result.overall_risk_level == "medium"
        assert result.risk_factors == []
        assert result.key_risks_summary == "Risk assessment unavailable."

    def test_empty_json_returns_default_assessment(self):
        result = _parse_risk_response("{}", "AMZN")
        assert result.ticker == "AMZN"
        assert result.overall_risk_level == "medium"
        assert result.risk_factors == []
        assert result.key_risks_summary == ""


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
    async def test_assess_returns_populated_risk_assessment(self):
        agent = RiskAssessorAgent()
        mock_msg = MagicMock()
        mock_msg.content = [
            MagicMock(
                text=json.dumps(
                    {
                        "risk_factors": [
                            {
                                "category": "competitive",
                                "description": "Intense GPU market competition",
                                "severity": "medium",
                                "probability": "high",
                            }
                        ],
                        "overall_risk_level": "medium",
                        "key_risks_summary": "Manageable risk.",
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
        assert result.overall_risk_level == "medium"
        assert result.overall_risk_score == 0.5
        assert result.key_risks_summary == "Manageable risk."
        assert len(result.risk_factors) == 1
        assert result.risk_factors[0].category == "competitive"

    async def test_assess_unknown_ticker(self):
        agent = RiskAssessorAgent()
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text="{}")]
        agent._client = AsyncMock()
        agent._client.messages.create = AsyncMock(return_value=mock_msg)

        result = await agent.assess({})
        assert isinstance(result, RiskAssessment)
        assert result.ticker == "UNKNOWN"
