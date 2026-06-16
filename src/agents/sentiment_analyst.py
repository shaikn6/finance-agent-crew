"""SentimentAnalystAgent — analyzes news + earnings call text with Claude."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

import anthropic

from ..models.schemas import SentimentAnalysis, Tone

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")
_MAX_HEADLINES = 15
_MAX_SNIPPET_CHARS = 300


class SentimentAnalystAgent:
    """Scores news sentiment and management tone using Claude.

    Aggregates headlines and earnings snippets, then makes a single
    structured Claude call to extract:
    - management_tone: positive | neutral | negative
    - news_sentiment_score: float [-1, +1]
    - key_themes: list of recurring topics
    - bullish_signals: positive catalysts mentioned
    - bearish_signals: risks or concerns mentioned
    """

    def __init__(self, model: str | None = None) -> None:
        self._model = model or _DEFAULT_MODEL
        self._client = anthropic.AsyncAnthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )

    async def analyze(self, raw_data: dict[str, Any]) -> SentimentAnalysis:
        """Run sentiment analysis on gathered raw data.

        Args:
            raw_data: Output dict from DataGathererAgent.gather().

        Returns:
            SentimentAnalysis with tone, score, themes, and signal lists.
        """
        ticker = raw_data.get("ticker", "UNKNOWN")
        headlines: list[dict[str, Any]] = raw_data.get("news_headlines", [])
        snippets: list[str] = raw_data.get("earnings_snippets", [])

        headline_texts = [
            h.get("title", "") for h in headlines[:_MAX_HEADLINES] if h.get("title")
        ]
        snippet_texts = [s[:_MAX_SNIPPET_CHARS] for s in snippets[:5] if s]

        if not headline_texts and not snippet_texts:
            logger.warning("No text available for sentiment analysis of %s", ticker)
            return SentimentAnalysis(
                ticker=ticker,
                reasoning="Insufficient data: no headlines or earnings snippets available.",
                analyzed_at=datetime.now(tz=timezone.utc),
            )

        result = await self._call_claude(ticker, headline_texts, snippet_texts)

        return SentimentAnalysis(
            ticker=ticker,
            management_tone=Tone(result.get("management_tone", "neutral")),
            news_sentiment_score=float(result.get("news_sentiment_score", 0.0)),
            key_themes=result.get("key_themes", []),
            headline_count=len(headline_texts),
            bullish_signals=result.get("bullish_signals", []),
            bearish_signals=result.get("bearish_signals", []),
            reasoning=result.get("reasoning", ""),
            analyzed_at=datetime.now(tz=timezone.utc),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _call_claude(
        self,
        ticker: str,
        headlines: list[str],
        snippets: list[str],
    ) -> dict[str, Any]:
        prompt = _build_sentiment_prompt(ticker, headlines, snippets)
        try:
            message = await self._client.messages.create(
                model=self._model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = message.content[0].text.strip()
            return _parse_json_response(raw)
        except Exception as exc:
            logger.error("SentimentAnalyst Claude call failed: %s", exc)
            return {
                "management_tone": "neutral",
                "news_sentiment_score": 0.0,
                "key_themes": [],
                "bullish_signals": [],
                "bearish_signals": [],
                "reasoning": f"Sentiment analysis unavailable: {exc}",
            }


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


def _build_sentiment_prompt(
    ticker: str,
    headlines: list[str],
    snippets: list[str],
) -> str:
    headlines_block = "\n".join(f"- {h}" for h in headlines) or "(none)"
    snippets_block = (
        "\n\n".join(f"[Snippet {i + 1}]: {s}" for i, s in enumerate(snippets))
        or "(none)"
    )

    return f"""You are a buy-side analyst specializing in NLP-driven market sentiment.

Analyze the sentiment of news coverage and earnings communications for {ticker}.

## Recent News Headlines
{headlines_block}

## Earnings Call / Report Snippets
{snippets_block}

## Task
Evaluate the overall market sentiment and management communication tone.

Respond in this exact JSON format:
{{
  "management_tone": "positive" | "neutral" | "negative",
  "news_sentiment_score": <float between -1.0 (very bearish) and +1.0 (very bullish)>,
  "key_themes": ["theme1", "theme2", "theme3"],
  "bullish_signals": ["signal1", "signal2"],
  "bearish_signals": ["signal1", "signal2"],
  "reasoning": "2-3 sentence explanation of the overall sentiment picture."
}}

Guidelines:
- management_tone reflects language in earnings snippets (confidence, forward-looking tone).
- news_sentiment_score aggregates headline polarity: +1 = unanimously bullish, -1 = unanimously bearish.
- key_themes: 3-5 recurring topics (e.g. "AI investment", "margin pressure", "supply chain").
- bullish_signals: specific positive catalysts mentioned.
- bearish_signals: specific risks or concerns mentioned.

Only output valid JSON, nothing else."""


def _parse_json_response(raw: str) -> dict[str, Any]:
    try:
        clean = (
            raw.strip()
            .removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        return json.loads(clean)
    except Exception as exc:
        logger.warning("Failed to parse sentiment JSON response: %s", exc)
        return {
            "management_tone": "neutral",
            "news_sentiment_score": 0.0,
            "key_themes": [],
            "bullish_signals": [],
            "bearish_signals": [],
            "reasoning": raw[:500],
        }
