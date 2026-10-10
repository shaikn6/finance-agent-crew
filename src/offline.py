"""Deterministic stand-ins for the LLM and data sources (``OFFLINE_MODE=1``).

Nothing in this module touches the network. The figures are synthetic and
identical for every ticker; they exist so the whole pipeline can be run and
tested without API keys, not to describe any real company.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Self

# ---------------------------------------------------------------------------
# Fake LLM
# ---------------------------------------------------------------------------

# Each agent opens its prompt with a distinct role line; that is what selects
# the canned reply.
_CANNED_REPLIES: dict[str, dict[str, Any]] = {
    "You are a CFA-level equity analyst": {
        "signal": "HOLD",
        "reasoning": (
            "Offline demo: valuation is in line with peers, margins are healthy "
            "and leverage is modest, so there is no clear mispricing."
        ),
    },
    "You are a buy-side analyst": {
        "management_tone": "positive",
        "news_sentiment_score": 0.35,
        "key_themes": ["revenue growth", "margin expansion", "guidance"],
        "bullish_signals": ["Record quarterly revenue", "Full-year guidance raised"],
        "bearish_signals": ["Higher input costs"],
        "reasoning": (
            "Offline demo: coverage is moderately positive, led by revenue "
            "growth and raised guidance."
        ),
    },
    "You are a risk analyst": {
        "risk_factors": [
            {
                "category": "market",
                "description": "Earnings are sensitive to a cyclical slowdown.",
                "severity": "medium",
                "probability": "medium",
            },
            {
                "category": "competitive",
                "description": "Pricing pressure from larger rivals.",
                "severity": "medium",
                "probability": "low",
            },
        ],
        "overall_risk_level": "medium",
        "key_risks_summary": "Offline demo: moderate cyclical and competitive risk.",
    },
    "You are a senior equity analyst": {
        "overall_signal": "HOLD",
        "executive_summary": (
            "Offline demo report built from fixture data: solid fundamentals and "
            "positive sentiment, balanced by moderate risk."
        ),
        "investment_thesis": (
            "Steady double-digit revenue growth and healthy margins are already "
            "reflected in the valuation."
        ),
        "bull_case": "Margin expansion and raised guidance drive upside.",
        "bear_case": "A cyclical slowdown or pricing pressure compresses margins.",
    },
}


class _OfflineMessages:
    async def create(
        self, *, model: str, max_tokens: int, messages: list[dict[str, Any]]
    ) -> SimpleNamespace:
        prompt = messages[0]["content"]
        reply: dict[str, Any] = {}
        for role_line, canned in _CANNED_REPLIES.items():
            if prompt.startswith(role_line):
                reply = canned
                break
        return SimpleNamespace(content=[SimpleNamespace(text=json.dumps(reply))])


class OfflineLLMClient:
    """Mimics the slice of ``anthropic.AsyncAnthropic`` the agents use."""

    def __init__(self) -> None:
        self.messages = _OfflineMessages()


# ---------------------------------------------------------------------------
# Fake data tools (same async interface as the classes in src/tools)
# ---------------------------------------------------------------------------

_REVENUE_HISTORY = [
    {"period_end": "2025-12-31", "value": 1_000_000_000},
    {"period_end": "2024-12-31", "value": 900_000_000},
    {"period_end": "2023-12-31", "value": 810_000_000},
    {"period_end": "2022-12-31", "value": 729_000_000},
]


class _OfflineTool:
    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None


class OfflineSECEdgarScraper(_OfflineTool):
    async def get_financial_summary(self, ticker: str) -> dict[str, Any]:
        return {
            "ticker": ticker,
            "cik": "0000000000",
            "company_name": f"{ticker} (offline demo data)",
            "financials": {
                "revenue": 1_000_000_000,
                "net_income": 150_000_000,
                "gross_profit": 400_000_000,
                "total_assets": 2_000_000_000,
                "total_liabilities": 1_200_000_000,
                "stockholders_equity": 800_000_000,
                "total_debt": 400_000_000,
                "eps_diluted": 3.0,
                "eps_basic": 3.05,
                "operating_income": 200_000_000,
                "cash": 250_000_000,
            },
            "revenue_history": [dict(entry) for entry in _REVENUE_HISTORY],
        }

    async def get_recent_filings(
        self, ticker: str, form_types: list[str] | None = None
    ) -> list[dict[str, Any]]:
        return [
            {
                "title": f"10-K - {ticker} (offline demo data)",
                "filed": "2026-02-15",
                "period_of_report": "2025-12-31",
                "form_type": "10-K",
                "url": "",
            }
        ]


class OfflineMarketDataClient(_OfflineTool):
    def __init__(self, api_key: str | None = None) -> None:
        pass

    async def get_company_overview(self, ticker: str) -> dict[str, Any]:
        return {
            "ticker": ticker,
            "company_name": f"{ticker} (offline demo data)",
            "description": "Synthetic company used by the offline demo.",
            "sector": "TECHNOLOGY",
            "industry": "SOFTWARE",
            "country": "USA",
            "exchange": "DEMO",
            "market_cap": 5_000_000_000.0,
            "pe_ratio": 20.0,
            "pb_ratio": 6.25,
            "ps_ratio": 5.0,
            "beta": 1.1,
            "eps": 3.0,
            "return_on_equity_ttm": 0.1875,
            "quarterly_revenue_growth": 0.11,
            "quarterly_earnings_growth": 0.08,
        }

    async def get_quote(self, ticker: str) -> dict[str, Any]:
        return {
            "ticker": ticker,
            "price": 60.0,
            "change": 0.6,
            "change_percent": "1.0101",
            "volume": 1_000_000,
            "latest_trading_day": "2026-01-30",
            "previous_close": 59.4,
            "open": 59.5,
            "high": 60.4,
            "low": 59.2,
        }

    async def get_monthly_prices(
        self, ticker: str, months: int = 12
    ) -> list[dict[str, Any]]:
        prices = []
        for offset in range(months):
            close = 60.0 - offset
            year, month = divmod(2026 * 12 - offset, 12)
            prices.append(
                {
                    "date": f"{year}-{month + 1:02d}-28",
                    "open": close - 0.5,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "volume": 20_000_000,
                }
            )
        return prices


class OfflineNewsFetcher(_OfflineTool):
    async def fetch_headlines(
        self, ticker: str, company_name: str = "", max_results: int = 20
    ) -> list[dict[str, Any]]:
        titles = [
            f"{ticker} reports record quarterly revenue",
            f"{ticker} raises full-year guidance",
            f"Analysts flag higher input costs at {ticker}",
        ]
        return [
            {
                "title": title,
                "url": f"offline://news/{index}",
                "source": "offline demo data",
                "published_at": None,
                "snippet": "",
            }
            for index, title in enumerate(titles[:max_results], start=1)
        ]

    async def fetch_earnings_snippets(self, ticker: str) -> list[str]:
        snippet = (
            f"{ticker} management said demand remained strong and raised "
            "full-year revenue guidance (offline demo data)."
        )
        return [snippet]
