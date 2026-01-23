"""Tests for src/agents/fundamental_analyst.py — FundamentalAnalystAgent."""

from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.models.schemas import Signal
from src.agents.fundamental_analyst import (
    FundamentalAnalystAgent,
    _build_fundamental_prompt,
    _parse_signal_response,
    _pct,
    _margin,
    _ratio,
    _cagr,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RAW_DATA_FULL = {
    "ticker": "AAPL",
    "company_name": "Apple Inc.",
    "market_overview": {
        "company_name": "Apple Inc.",
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "market_cap": 2800000000000.0,
        "pe_ratio": 29.50,
        "pb_ratio": 47.50,
        "return_on_equity_ttm": 1.4726,
        "beta": 1.28,
        "quarterly_revenue_growth": -0.0107,
        "quarterly_earnings_growth": 0.1100,
    },
    "sec_summary": {
        "financials": {
            "revenue": 383285000000,
            "net_income": 96995000000,
            "gross_profit": 169148000000,
            "operating_income": 114301000000,
            "total_debt": 95281000000,
            "stockholders_equity": 62146000000,
        },
        "revenue_history": [
            {"period_end": "2023-09-30", "value": 383285000000},
            {"period_end": "2022-09-24", "value": 394328000000},
            {"period_end": "2021-09-25", "value": 365817000000},
            {"period_end": "2020-09-26", "value": 274515000000},
        ],
    },
    "current_quote": {"ticker": "AAPL", "price": 178.50},
}

RAW_DATA_EMPTY = {
    "ticker": "UNKN",
    "market_overview": {},
    "sec_summary": {"financials": {}, "revenue_history": []},
    "current_quote": {},
}


def _make_claude_client(signal="BUY", reasoning="Strong fundamentals."):
    """Create a mock AsyncAnthropic client."""
    client = AsyncMock()
    response = MagicMock()
    response.content = [MagicMock()]
    response.content[0].text = json.dumps({"signal": signal, "reasoning": reasoning})
    client.messages.create = AsyncMock(return_value=response)
    return client


# ---------------------------------------------------------------------------
# FundamentalAnalystAgent.analyze
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestFundamentalAnalystAgentAnalyze:
    async def test_full_analysis_returns_buy_signal(self):
        agent = FundamentalAnalystAgent()
        agent._client = _make_claude_client(
            "BUY", "Apple has strong margins and solid balance sheet."
        )

        analysis = await agent.analyze(RAW_DATA_FULL)
        assert analysis.ticker == "AAPL"
        assert analysis.signal == Signal.BUY
        assert "Apple" in analysis.reasoning

    async def test_ratios_computed_correctly(self):
        agent = FundamentalAnalystAgent()
        agent._client = _make_claude_client("HOLD", "Fairly valued.")

        analysis = await agent.analyze(RAW_DATA_FULL)
        # gross_margin = gross_profit / revenue * 100
        expected_gross = round((169148000000 / 383285000000) * 100, 2)
        assert analysis.gross_margin == pytest.approx(expected_gross, abs=0.01)

        # operating_margin
        expected_op = round((114301000000 / 383285000000) * 100, 2)
        assert analysis.operating_margin == pytest.approx(expected_op, abs=0.01)

        # net_margin
        expected_net = round((96995000000 / 383285000000) * 100, 2)
        assert analysis.net_margin == pytest.approx(expected_net, abs=0.01)

    async def test_debt_to_equity_computed(self):
        agent = FundamentalAnalystAgent()
        agent._client = _make_claude_client()

        analysis = await agent.analyze(RAW_DATA_FULL)
        expected = round(95281000000 / 62146000000, 3)
        assert analysis.debt_to_equity == pytest.approx(expected, abs=0.001)

    async def test_revenue_cagr_computed(self):
        agent = FundamentalAnalystAgent()
        agent._client = _make_claude_client()

        analysis = await agent.analyze(RAW_DATA_FULL)
        # 3-year CAGR from 2020 to 2023: (383285 / 274515)^(1/3) - 1
        expected = round(((383285000000 / 274515000000) ** (1 / 3) - 1) * 100, 2)
        assert analysis.revenue_cagr_3y == pytest.approx(expected, abs=0.1)

    async def test_empty_data_returns_insufficient(self):
        agent = FundamentalAnalystAgent()
        agent._client = _make_claude_client("INSUFFICIENT_DATA", "No data.")

        analysis = await agent.analyze(RAW_DATA_EMPTY)
        assert analysis.signal == Signal.INSUFFICIENT_DATA
        assert analysis.pe_ratio is None

    async def test_claude_api_failure_returns_insufficient(self):
        agent = FundamentalAnalystAgent()
        agent._client = AsyncMock()
        agent._client.messages.create = AsyncMock(side_effect=Exception("API error"))

        analysis = await agent.analyze(RAW_DATA_FULL)
        assert analysis.signal == Signal.INSUFFICIENT_DATA
        assert "unavailable" in analysis.reasoning

    async def test_data_as_of_is_set(self):
        agent = FundamentalAnalystAgent()
        agent._client = _make_claude_client()

        analysis = await agent.analyze(RAW_DATA_FULL)
        assert analysis.data_as_of is not None

    async def test_sell_signal_parsed(self):
        agent = FundamentalAnalystAgent()
        agent._client = _make_claude_client("SELL", "Overvalued and declining.")

        analysis = await agent.analyze(RAW_DATA_FULL)
        assert analysis.signal == Signal.SELL

    async def test_missing_ticker_uses_unknown(self):
        agent = FundamentalAnalystAgent()
        agent._client = _make_claude_client()

        data = {**RAW_DATA_EMPTY}
        del data["ticker"]
        analysis = await agent.analyze(data)
        assert analysis.ticker == "UNKNOWN"

    async def test_custom_model_used(self):
        agent = FundamentalAnalystAgent(model="claude-3-opus-20240229")
        agent._client = _make_claude_client()

        await agent.analyze(RAW_DATA_FULL)
        call_kwargs = agent._client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-3-opus-20240229"

    async def test_roe_from_overview(self):
        agent = FundamentalAnalystAgent()
        agent._client = _make_claude_client()

        analysis = await agent.analyze(RAW_DATA_FULL)
        # ROE from overview = 1.4726 * 100 = 147.26 (since abs > 10 in pct, it rounds raw)
        assert analysis.roe is not None

    async def test_pe_ratio_from_overview(self):
        agent = FundamentalAnalystAgent()
        agent._client = _make_claude_client()

        analysis = await agent.analyze(RAW_DATA_FULL)
        assert analysis.pe_ratio == 29.50

    async def test_pb_ratio_from_overview(self):
        agent = FundamentalAnalystAgent()
        agent._client = _make_claude_client()

        analysis = await agent.analyze(RAW_DATA_FULL)
        assert analysis.pb_ratio == 47.50


# ---------------------------------------------------------------------------
# _build_fundamental_prompt
# ---------------------------------------------------------------------------


class TestBuildFundamentalPrompt:
    def test_contains_ticker(self):
        prompt = _build_fundamental_prompt(
            "AAPL",
            {"pe_ratio": 29.5, "roe": 45.2},
            {
                "sector": "Technology",
                "industry": "Consumer Electronics",
                "market_cap": 2e12,
            },
            {},
        )
        assert "AAPL" in prompt
        assert "Technology" in prompt

    def test_contains_json_format(self):
        prompt = _build_fundamental_prompt("T", {}, {}, {})
        assert '"signal"' in prompt
        assert '"reasoning"' in prompt

    def test_none_ratios_excluded(self):
        ratios = {"pe_ratio": None, "roe": 25.0}
        prompt = _build_fundamental_prompt("T", ratios, {}, {})
        assert "pe_ratio" not in prompt or "null" not in prompt

    def test_market_cap_formatted(self):
        prompt = _build_fundamental_prompt("T", {}, {"market_cap": 1000000000000.0}, {})
        assert "$" in prompt
        assert "B" in prompt

    def test_unknown_market_cap(self):
        prompt = _build_fundamental_prompt("T", {}, {}, {})
        assert "Unknown" in prompt


# ---------------------------------------------------------------------------
# _parse_signal_response
# ---------------------------------------------------------------------------


class TestParseSignalResponse:
    def test_parses_buy_signal(self):
        raw = json.dumps({"signal": "BUY", "reasoning": "Strong growth."})
        signal, reasoning = _parse_signal_response(raw)
        assert signal == Signal.BUY
        assert reasoning == "Strong growth."

    def test_parses_sell_signal(self):
        raw = json.dumps({"signal": "SELL", "reasoning": "Overvalued."})
        signal, reasoning = _parse_signal_response(raw)
        assert signal == Signal.SELL

    def test_parses_hold_signal(self):
        raw = json.dumps({"signal": "HOLD", "reasoning": "Mixed signals."})
        signal, reasoning = _parse_signal_response(raw)
        assert signal == Signal.HOLD

    def test_strips_markdown_code_fences(self):
        raw = (
            "```json\n" + json.dumps({"signal": "BUY", "reasoning": "Good."}) + "\n```"
        )
        signal, reasoning = _parse_signal_response(raw)
        assert signal == Signal.BUY

    def test_invalid_signal_returns_insufficient(self):
        raw = json.dumps({"signal": "UNKNOWN_SIGNAL", "reasoning": "Weird."})
        signal, _ = _parse_signal_response(raw)
        assert signal == Signal.INSUFFICIENT_DATA

    def test_malformed_json_returns_insufficient(self):
        signal, reasoning = _parse_signal_response("not valid JSON {{{")
        assert signal == Signal.INSUFFICIENT_DATA
        # reasoning should be the truncated raw text
        assert "not valid" in reasoning

    def test_empty_signal_key_returns_insufficient(self):
        raw = json.dumps({"reasoning": "No signal provided."})
        signal, _ = _parse_signal_response(raw)
        assert signal == Signal.INSUFFICIENT_DATA

    def test_lowercase_signal_handled(self):
        raw = json.dumps({"signal": "buy", "reasoning": "Bullish."})
        signal, _ = _parse_signal_response(raw)
        assert signal == Signal.BUY


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------


class TestPct:
    def test_small_value_multiplied(self):
        assert _pct(0.25) == 25.0

    def test_large_value_left_as_is(self):
        # abs >= 10, already a percentage
        assert _pct(147.26) == 147.26

    def test_none_returns_none(self):
        assert _pct(None) is None

    def test_zero(self):
        assert _pct(0.0) == 0.0


class TestMargin:
    def test_computes_margin(self):
        assert _margin(100.0, 1000.0) == 10.0

    def test_none_numerator(self):
        assert _margin(None, 1000.0) is None

    def test_none_denominator(self):
        assert _margin(100.0, None) is None

    def test_zero_denominator(self):
        assert _margin(100.0, 0.0) is None


class TestRatio:
    def test_computes_ratio(self):
        assert _ratio(100.0, 50.0) == 2.0

    def test_none_numerator(self):
        assert _ratio(None, 50.0) is None

    def test_none_denominator(self):
        assert _ratio(100.0, None) is None

    def test_zero_denominator(self):
        assert _ratio(100.0, 0.0) is None


class TestCAGR:
    def test_computes_3y_cagr(self):
        history = [
            {"period_end": "2023-09-30", "value": 383285000000},
            {"period_end": "2022-09-24", "value": 394328000000},
            {"period_end": "2021-09-25", "value": 365817000000},
            {"period_end": "2020-09-26", "value": 274515000000},
        ]
        cagr = _cagr(history, years=3)
        assert cagr is not None
        assert cagr > 0  # Apple grew revenue over this period

    def test_single_entry_returns_none(self):
        assert _cagr([{"period_end": "2023-09-30", "value": 100}]) is None

    def test_empty_returns_none(self):
        assert _cagr([]) is None

    def test_oldest_zero_returns_none(self):
        history = [
            {"period_end": "2023-09-30", "value": 100},
            {"period_end": "2020-09-30", "value": 0},
        ]
        assert _cagr(history) is None

    def test_oldest_none_returns_none(self):
        history = [
            {"period_end": "2023-09-30", "value": 100},
            {"period_end": "2020-09-30", "value": None},
        ]
        assert _cagr(history) is None

    def test_uses_sorted_order(self):
        history = [
            {"period_end": "2020-09-30", "value": 100},
            {"period_end": "2023-09-30", "value": 200},
        ]
        cagr = _cagr(history, years=3)
        # CAGR from 100 to 200 over 3 years = (2^(1/3) - 1) * 100 ≈ 26%
        assert cagr == pytest.approx(25.99, abs=0.1)

    def test_years_capped_to_available(self):
        history = [
            {"period_end": "2022-09-30", "value": 100},
            {"period_end": "2023-09-30", "value": 110},
        ]
        # Only 1 year available, years=3 requested
        cagr = _cagr(history, years=3)
        assert cagr is not None
