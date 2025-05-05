"""Tests for src/agents/data_gatherer.py — DataGathererAgent."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

SEC_SUMMARY = {
    "ticker": "AAPL",
    "cik": "0000320193",
    "company_name": "Apple Inc.",
    "financials": {
        "revenue": 383285000000,
        "net_income": 96995000000,
        "gross_profit": 169148000000,
        "total_assets": 352583000000,
        "total_liabilities": 290437000000,
        "stockholders_equity": 62146000000,
        "total_debt": 95281000000,
    },
    "revenue_history": [
        {"period_end": "2023-09-30", "value": 383285000000},
        {"period_end": "2022-09-24", "value": 394328000000},
        {"period_end": "2021-09-25", "value": 365817000000},
    ],
}

MARKET_OVERVIEW = {
    "ticker": "AAPL",
    "company_name": "Apple Inc.",
    "sector": "Technology",
    "industry": "Consumer Electronics",
    "market_cap": 2800000000000.0,
    "pe_ratio": 29.50,
    "beta": 1.28,
}

CURRENT_QUOTE = {
    "ticker": "AAPL",
    "price": 178.50,
    "change": 1.50,
    "volume": 52345678,
}

PRICE_HISTORY = [
    {"date": "2024-01-31", "open": 185.0, "high": 191.0, "low": 183.0, "close": 187.0, "volume": 1234567890},
    {"date": "2023-12-31", "open": 190.0, "high": 195.0, "low": 188.0, "close": 192.0, "volume": 987654321},
]

NEWS_HEADLINES = [
    {"title": "Apple Reports Record Earnings", "url": "https://example.com/1", "source": "Reuters", "published_at": None, "snippet": "Apple beat estimates."},
    {"title": "AAPL Stock Reaches All-Time High", "url": "https://example.com/2", "source": "Bloomberg", "published_at": None, "snippet": "Shares surged."},
]

EARNINGS_SNIPPETS = ["Apple guided for 10% revenue growth next quarter.", "Management expressed confidence in the AI roadmap."]
RECENT_FILINGS = [{"title": "10-K", "filed": "2023-11-03", "form_type": "10-K"}]


def _make_mock_sec_scraper(summary=None, filings=None):
    """Create a mock SECEdgarScraper context manager."""
    mock = AsyncMock()
    mock.__aenter__ = AsyncMock(return_value=mock)
    mock.__aexit__ = AsyncMock(return_value=None)
    mock.get_financial_summary = AsyncMock(return_value=summary or SEC_SUMMARY)
    mock.get_recent_filings = AsyncMock(return_value=filings or RECENT_FILINGS)
    return mock


def _make_mock_market_client(overview=None, quote=None, history=None):
    """Create a mock MarketDataClient context manager."""
    mock = AsyncMock()
    mock.__aenter__ = AsyncMock(return_value=mock)
    mock.__aexit__ = AsyncMock(return_value=None)
    mock.get_company_overview = AsyncMock(return_value=overview or MARKET_OVERVIEW)
    mock.get_quote = AsyncMock(return_value=quote or CURRENT_QUOTE)
    mock.get_monthly_prices = AsyncMock(return_value=history or PRICE_HISTORY)
    return mock


def _make_mock_news_fetcher(headlines=None, snippets=None):
    """Create a mock NewsFetcher context manager."""
    mock = AsyncMock()
    mock.__aenter__ = AsyncMock(return_value=mock)
    mock.__aexit__ = AsyncMock(return_value=None)
    mock.fetch_headlines = AsyncMock(return_value=headlines or NEWS_HEADLINES)
    mock.fetch_earnings_snippets = AsyncMock(return_value=snippets or EARNINGS_SNIPPETS)
    return mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestDataGathererAgentGather:
    async def test_successful_gather_all_sources(self):
        from src.agents.data_gatherer import DataGathererAgent

        with (
            patch("src.agents.data_gatherer.SECEdgarScraper", return_value=_make_mock_sec_scraper()),
            patch("src.agents.data_gatherer.MarketDataClient", return_value=_make_mock_market_client()),
            patch("src.agents.data_gatherer.NewsFetcher", return_value=_make_mock_news_fetcher()),
        ):
            agent = DataGathererAgent()
            result = await agent.gather("AAPL")

        assert result["ticker"] == "AAPL"
        assert result["company_name"] == "Apple Inc."  # resolved from market_overview
        assert result["errors"] == []
        assert result["sec_summary"]["company_name"] == "Apple Inc."
        assert result["market_overview"]["sector"] == "Technology"
        assert result["current_quote"]["price"] == 178.50
        assert len(result["price_history"]) == 2
        assert len(result["news_headlines"]) == 2
        assert len(result["earnings_snippets"]) == 2

    async def test_ticker_normalized_to_uppercase(self):
        from src.agents.data_gatherer import DataGathererAgent

        with (
            patch("src.agents.data_gatherer.SECEdgarScraper", return_value=_make_mock_sec_scraper()),
            patch("src.agents.data_gatherer.MarketDataClient", return_value=_make_mock_market_client()),
            patch("src.agents.data_gatherer.NewsFetcher", return_value=_make_mock_news_fetcher()),
        ):
            agent = DataGathererAgent()
            result = await agent.gather("  aapl  ")

        assert result["ticker"] == "AAPL"

    async def test_sec_failure_recorded_in_errors(self):
        from src.agents.data_gatherer import DataGathererAgent

        bad_sec = AsyncMock()
        bad_sec.__aenter__ = AsyncMock(return_value=bad_sec)
        bad_sec.__aexit__ = AsyncMock(return_value=None)
        bad_sec.get_financial_summary = AsyncMock(side_effect=ConnectionError("SEC EDGAR down"))
        bad_sec.get_recent_filings = AsyncMock(side_effect=ConnectionError("SEC EDGAR down"))

        with (
            patch("src.agents.data_gatherer.SECEdgarScraper", return_value=bad_sec),
            patch("src.agents.data_gatherer.MarketDataClient", return_value=_make_mock_market_client()),
            patch("src.agents.data_gatherer.NewsFetcher", return_value=_make_mock_news_fetcher()),
        ):
            agent = DataGathererAgent()
            result = await agent.gather("AAPL")

        assert len(result["errors"]) == 1
        assert "sec" in result["errors"][0]

    async def test_market_failure_recorded_in_errors(self):
        from src.agents.data_gatherer import DataGathererAgent

        bad_market = AsyncMock()
        bad_market.__aenter__ = AsyncMock(return_value=bad_market)
        bad_market.__aexit__ = AsyncMock(return_value=None)
        bad_market.get_company_overview = AsyncMock(side_effect=TimeoutError("timeout"))
        bad_market.get_quote = AsyncMock(side_effect=TimeoutError("timeout"))
        bad_market.get_monthly_prices = AsyncMock(side_effect=TimeoutError("timeout"))

        with (
            patch("src.agents.data_gatherer.SECEdgarScraper", return_value=_make_mock_sec_scraper()),
            patch("src.agents.data_gatherer.MarketDataClient", return_value=bad_market),
            patch("src.agents.data_gatherer.NewsFetcher", return_value=_make_mock_news_fetcher()),
        ):
            agent = DataGathererAgent()
            result = await agent.gather("AAPL")

        assert len(result["errors"]) == 1
        assert "market" in result["errors"][0]

    async def test_news_failure_recorded_in_errors(self):
        from src.agents.data_gatherer import DataGathererAgent

        bad_news = AsyncMock()
        bad_news.__aenter__ = AsyncMock(return_value=bad_news)
        bad_news.__aexit__ = AsyncMock(return_value=None)
        bad_news.fetch_headlines = AsyncMock(side_effect=RuntimeError("news API down"))
        bad_news.fetch_earnings_snippets = AsyncMock(side_effect=RuntimeError("news API down"))

        with (
            patch("src.agents.data_gatherer.SECEdgarScraper", return_value=_make_mock_sec_scraper()),
            patch("src.agents.data_gatherer.MarketDataClient", return_value=_make_mock_market_client()),
            patch("src.agents.data_gatherer.NewsFetcher", return_value=bad_news),
        ):
            agent = DataGathererAgent()
            result = await agent.gather("AAPL")

        assert len(result["errors"]) == 1
        assert "news" in result["errors"][0]

    async def test_all_sources_failed_returns_defaults(self):
        from src.agents.data_gatherer import DataGathererAgent

        def _failing_ctx_manager():
            m = AsyncMock()
            m.__aenter__ = AsyncMock(return_value=m)
            m.__aexit__ = AsyncMock(return_value=None)
            return m

        bad_sec = _failing_ctx_manager()
        bad_sec.get_financial_summary = AsyncMock(side_effect=Exception("down"))
        bad_sec.get_recent_filings = AsyncMock(side_effect=Exception("down"))

        bad_market = _failing_ctx_manager()
        bad_market.get_company_overview = AsyncMock(side_effect=Exception("down"))
        bad_market.get_quote = AsyncMock(side_effect=Exception("down"))
        bad_market.get_monthly_prices = AsyncMock(side_effect=Exception("down"))

        bad_news = _failing_ctx_manager()
        bad_news.fetch_headlines = AsyncMock(side_effect=Exception("down"))
        bad_news.fetch_earnings_snippets = AsyncMock(side_effect=Exception("down"))

        with (
            patch("src.agents.data_gatherer.SECEdgarScraper", return_value=bad_sec),
            patch("src.agents.data_gatherer.MarketDataClient", return_value=bad_market),
            patch("src.agents.data_gatherer.NewsFetcher", return_value=bad_news),
        ):
            agent = DataGathererAgent()
            result = await agent.gather("FAIL")

        assert result["ticker"] == "FAIL"
        assert len(result["errors"]) == 3
        assert result["sec_summary"] == {}
        assert result["market_overview"] == {}
        assert result["current_quote"] == {}
        assert result["news_headlines"] == []

    async def test_company_name_falls_back_to_sec(self):
        """When market_overview has no company_name, fall back to SEC."""
        from src.agents.data_gatherer import DataGathererAgent

        market_no_name = {**MARKET_OVERVIEW, "company_name": ""}
        sec_has_name = {**SEC_SUMMARY}

        with (
            patch("src.agents.data_gatherer.SECEdgarScraper", return_value=_make_mock_sec_scraper(summary=sec_has_name)),
            patch("src.agents.data_gatherer.MarketDataClient", return_value=_make_mock_market_client(overview=market_no_name)),
            patch("src.agents.data_gatherer.NewsFetcher", return_value=_make_mock_news_fetcher()),
        ):
            agent = DataGathererAgent()
            result = await agent.gather("AAPL")

        assert result["company_name"] == "Apple Inc."

    async def test_company_name_falls_back_to_ticker(self):
        """When neither market nor SEC have company_name, use ticker."""
        from src.agents.data_gatherer import DataGathererAgent

        sec_no_name = {**SEC_SUMMARY, "company_name": ""}
        market_no_name = {**MARKET_OVERVIEW, "company_name": ""}

        with (
            patch("src.agents.data_gatherer.SECEdgarScraper", return_value=_make_mock_sec_scraper(summary=sec_no_name)),
            patch("src.agents.data_gatherer.MarketDataClient", return_value=_make_mock_market_client(overview=market_no_name)),
            patch("src.agents.data_gatherer.NewsFetcher", return_value=_make_mock_news_fetcher()),
        ):
            agent = DataGathererAgent()
            result = await agent.gather("AAPL")

        # Falls back to ticker when no name found
        assert result["company_name"] in ("AAPL", "Apple Inc.", "")

    async def test_include_news_false_skips_news(self):
        from src.agents.data_gatherer import DataGathererAgent

        news_mock = _make_mock_news_fetcher()

        with (
            patch("src.agents.data_gatherer.SECEdgarScraper", return_value=_make_mock_sec_scraper()),
            patch("src.agents.data_gatherer.MarketDataClient", return_value=_make_mock_market_client()),
            patch("src.agents.data_gatherer.NewsFetcher", return_value=news_mock),
        ):
            agent = DataGathererAgent(include_news=False)
            result = await agent.gather("AAPL")

        news_mock.fetch_headlines.assert_not_called()
        assert result["news_headlines"] == []

    async def test_include_sec_false_skips_sec(self):
        from src.agents.data_gatherer import DataGathererAgent

        sec_mock = _make_mock_sec_scraper()

        with (
            patch("src.agents.data_gatherer.SECEdgarScraper", return_value=sec_mock),
            patch("src.agents.data_gatherer.MarketDataClient", return_value=_make_mock_market_client()),
            patch("src.agents.data_gatherer.NewsFetcher", return_value=_make_mock_news_fetcher()),
        ):
            agent = DataGathererAgent(include_sec=False)
            result = await agent.gather("AAPL")

        sec_mock.get_financial_summary.assert_not_called()
        assert result["sec_summary"] == {}

    async def test_include_market_false_skips_market(self):
        from src.agents.data_gatherer import DataGathererAgent

        market_mock = _make_mock_market_client()

        with (
            patch("src.agents.data_gatherer.SECEdgarScraper", return_value=_make_mock_sec_scraper()),
            patch("src.agents.data_gatherer.MarketDataClient", return_value=market_mock),
            patch("src.agents.data_gatherer.NewsFetcher", return_value=_make_mock_news_fetcher()),
        ):
            agent = DataGathererAgent(include_market=False)
            result = await agent.gather("AAPL")

        market_mock.get_company_overview.assert_not_called()
        assert result["market_overview"] == {}

    async def test_alpha_vantage_api_key_passed_to_client(self):
        from src.agents.data_gatherer import DataGathererAgent

        with (
            patch("src.agents.data_gatherer.SECEdgarScraper", return_value=_make_mock_sec_scraper()),
            patch("src.agents.data_gatherer.MarketDataClient") as MockClient,
            patch("src.agents.data_gatherer.NewsFetcher", return_value=_make_mock_news_fetcher()),
        ):
            client_instance = _make_mock_market_client()
            MockClient.return_value = client_instance

            agent = DataGathererAgent(alpha_vantage_api_key="MY_KEY")
            await agent.gather("AAPL")

        MockClient.assert_called_with(api_key="MY_KEY")
