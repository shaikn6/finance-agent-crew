"""Tests for src/agents/sentiment_analyst.py — SentimentAnalystAgent."""
from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timezone

from src.models.schemas import Tone
from src.agents.sentiment_analyst import (
    SentimentAnalystAgent,
    _build_sentiment_prompt,
    _parse_json_response,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RAW_DATA_FULL = {
    "ticker": "AAPL",
    "news_headlines": [
        {"title": "Apple Reports Record Q4 Earnings", "url": "https://example.com/1"},
        {"title": "AAPL Beats Revenue Estimates by 8%", "url": "https://example.com/2"},
        {"title": "Apple CEO Highlights AI Roadmap", "url": "https://example.com/3"},
        {"title": "iPhone 16 Sales Exceed Forecasts", "url": "https://example.com/4"},
        {"title": "Apple Services Revenue Hits All-Time High", "url": "https://example.com/5"},
    ],
    "earnings_snippets": [
        "We are incredibly pleased with our Q4 results, exceeding expectations on every metric.",
        "Our AI integration across all products positions us well for continued growth.",
    ],
}

RAW_DATA_EMPTY = {
    "ticker": "EMPTY",
    "news_headlines": [],
    "earnings_snippets": [],
}

RAW_DATA_NO_TITLES = {
    "ticker": "NOTIT",
    "news_headlines": [{"url": "https://example.com/1"}, {"url": "https://example.com/2"}],
    "earnings_snippets": [],
}

CLAUDE_POSITIVE_RESPONSE = {
    "management_tone": "positive",
    "news_sentiment_score": 0.75,
    "key_themes": ["AI investment", "earnings beat", "services growth"],
    "bullish_signals": ["Record revenue", "Strong guidance"],
    "bearish_signals": ["Supply chain risk"],
    "reasoning": "News coverage is overwhelmingly positive with strong management tone.",
}

CLAUDE_NEGATIVE_RESPONSE = {
    "management_tone": "negative",
    "news_sentiment_score": -0.60,
    "key_themes": ["margin pressure", "declining sales", "competition"],
    "bullish_signals": [],
    "bearish_signals": ["Revenue miss", "Guidance cut", "Market share loss"],
    "reasoning": "Company faces significant headwinds with bearish news coverage.",
}


def _make_claude_client(response_data=None):
    """Create a mock AsyncAnthropic client."""
    client = AsyncMock()
    msg = MagicMock()
    msg.content = [MagicMock()]
    msg.content[0].text = json.dumps(response_data or CLAUDE_POSITIVE_RESPONSE)
    client.messages.create = AsyncMock(return_value=msg)
    return client


# ---------------------------------------------------------------------------
# SentimentAnalystAgent.analyze
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestSentimentAnalystAgentAnalyze:
    async def test_analyze_full_data(self):
        agent = SentimentAnalystAgent()
        agent._client = _make_claude_client(CLAUDE_POSITIVE_RESPONSE)

        result = await agent.analyze(RAW_DATA_FULL)
        assert result.ticker == "AAPL"
        assert result.management_tone == Tone.POSITIVE
        assert result.news_sentiment_score == 0.75
        assert len(result.key_themes) == 3
        assert len(result.bullish_signals) == 2
        assert len(result.bearish_signals) == 1
        assert result.headline_count == 5

    async def test_analyze_empty_data_returns_default(self):
        agent = SentimentAnalystAgent()
        agent._client = _make_claude_client()

        result = await agent.analyze(RAW_DATA_EMPTY)
        assert result.ticker == "EMPTY"
        assert result.management_tone == Tone.NEUTRAL
        assert result.news_sentiment_score == 0.0
        assert result.headline_count == 0
        assert "Insufficient data" in result.reasoning

    async def test_headlines_without_titles_ignored(self):
        agent = SentimentAnalystAgent()
        agent._client = _make_claude_client()

        result = await agent.analyze(RAW_DATA_NO_TITLES)
        assert result.ticker == "NOTIT"
        # No titles => no text => empty fallback
        assert "Insufficient data" in result.reasoning

    async def test_negative_sentiment_analysis(self):
        agent = SentimentAnalystAgent()
        agent._client = _make_claude_client(CLAUDE_NEGATIVE_RESPONSE)

        result = await agent.analyze(RAW_DATA_FULL)
        assert result.management_tone == Tone.NEGATIVE
        assert result.news_sentiment_score < 0
        assert len(result.bearish_signals) == 3

    async def test_claude_api_failure_returns_neutral_defaults(self):
        agent = SentimentAnalystAgent()
        agent._client = AsyncMock()
        agent._client.messages.create = AsyncMock(side_effect=Exception("API down"))

        result = await agent.analyze(RAW_DATA_FULL)
        assert result.management_tone == Tone.NEUTRAL
        assert result.news_sentiment_score == 0.0
        assert "unavailable" in result.reasoning

    async def test_analyzed_at_is_set(self):
        agent = SentimentAnalystAgent()
        agent._client = _make_claude_client()

        result = await agent.analyze(RAW_DATA_FULL)
        assert result.analyzed_at is not None

    async def test_custom_model_used(self):
        agent = SentimentAnalystAgent(model="claude-3-opus-20240229")
        agent._client = _make_claude_client()

        await agent.analyze(RAW_DATA_FULL)
        call_kwargs = agent._client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-3-opus-20240229"

    async def test_max_headlines_capped_at_15(self):
        """Agent should only send first 15 headlines to Claude."""
        agent = SentimentAnalystAgent()
        agent._client = _make_claude_client()

        data = {
            "ticker": "T",
            "news_headlines": [{"title": f"Headline {i}"} for i in range(25)],
            "earnings_snippets": [],
        }
        await agent.analyze(data)

        # Verify by checking the prompt content passed to Claude
        call_args = agent._client.messages.create.call_args
        prompt = call_args[1]["messages"][0]["content"]
        # Count headlines in prompt (≤15)
        headline_count = prompt.count("- Headline")
        assert headline_count <= 15

    async def test_only_snippets_no_headlines(self):
        """Even without headlines, if snippets exist, Claude is called."""
        agent = SentimentAnalystAgent()
        agent._client = _make_claude_client()

        data = {
            "ticker": "T",
            "news_headlines": [],
            "earnings_snippets": ["Strong earnings growth expected."],
        }
        result = await agent.analyze(data)
        assert result.ticker == "T"
        agent._client.messages.create.assert_called_once()

    async def test_missing_ticker_uses_unknown(self):
        agent = SentimentAnalystAgent()
        agent._client = _make_claude_client()

        data = {
            "news_headlines": [{"title": "Some news"}],
            "earnings_snippets": [],
        }
        result = await agent.analyze(data)
        assert result.ticker == "UNKNOWN"

    async def test_snippet_chars_capped_at_300(self):
        """Each snippet should be truncated to 300 chars when sent to Claude."""
        agent = SentimentAnalystAgent()
        agent._client = _make_claude_client()

        long_snippet = "x" * 600
        data = {
            "ticker": "T",
            "news_headlines": [{"title": "Headline 1"}],
            "earnings_snippets": [long_snippet],
        }
        await agent.analyze(data)

        call_args = agent._client.messages.create.call_args
        prompt = call_args[1]["messages"][0]["content"]
        assert "x" * 301 not in prompt


# ---------------------------------------------------------------------------
# _build_sentiment_prompt
# ---------------------------------------------------------------------------

class TestBuildSentimentPrompt:
    def test_contains_ticker(self):
        prompt = _build_sentiment_prompt("AAPL", ["Headline 1"], ["Snippet 1"])
        assert "AAPL" in prompt

    def test_contains_headlines(self):
        prompt = _build_sentiment_prompt("T", ["Apple beats earnings"], [])
        assert "Apple beats earnings" in prompt

    def test_contains_snippets(self):
        prompt = _build_sentiment_prompt("T", [], ["Strong guidance from management"])
        assert "Strong guidance" in prompt

    def test_empty_headlines_shows_none(self):
        prompt = _build_sentiment_prompt("T", [], [])
        assert "(none)" in prompt

    def test_contains_json_format(self):
        prompt = _build_sentiment_prompt("T", [], [])
        assert '"management_tone"' in prompt
        assert '"news_sentiment_score"' in prompt

    def test_multiple_headlines_all_present(self):
        headlines = ["Headline A", "Headline B", "Headline C"]
        prompt = _build_sentiment_prompt("T", headlines, [])
        for h in headlines:
            assert h in prompt

    def test_snippet_indices_in_prompt(self):
        prompt = _build_sentiment_prompt("T", [], ["S1", "S2"])
        assert "[Snippet 1]" in prompt
        assert "[Snippet 2]" in prompt


# ---------------------------------------------------------------------------
# _parse_json_response
# ---------------------------------------------------------------------------

class TestParseJsonResponse:
    def test_parses_valid_json(self):
        raw = json.dumps(CLAUDE_POSITIVE_RESPONSE)
        result = _parse_json_response(raw)
        assert result["management_tone"] == "positive"
        assert result["news_sentiment_score"] == 0.75

    def test_strips_markdown_code_fences(self):
        raw = "```json
" + json.dumps(CLAUDE_POSITIVE_RESPONSE) + "
```"
        result = _parse_json_response(raw)
        assert result["management_tone"] == "positive"

    def test_strips_plain_code_fences(self):
        raw = "```
" + json.dumps(CLAUDE_POSITIVE_RESPONSE) + "
```"
        result = _parse_json_response(raw)
        assert result["management_tone"] == "positive"

    def test_invalid_json_returns_neutral_defaults(self):
        result = _parse_json_response("not valid json")
        assert result["management_tone"] == "neutral"
        assert result["news_sentiment_score"] == 0.0
        assert result["key_themes"] == []
        assert result["bullish_signals"] == []
        assert result["bearish_signals"] == []

    def test_invalid_json_reasoning_is_raw_text(self):
        result = _parse_json_response("error: something broke")
        assert "error" in result["reasoning"] or result["reasoning"]

    def test_empty_string_returns_neutral_defaults(self):
        result = _parse_json_response("")
        assert result["management_tone"] == "neutral"

    def test_nested_lists_preserved(self):
        data = {**CLAUDE_POSITIVE_RESPONSE, "key_themes": ["AI", "Cloud", "Mobile"]}
        result = _parse_json_response(json.dumps(data))
        assert result["key_themes"] == ["AI", "Cloud", "Mobile"]
