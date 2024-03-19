"""DataGathererAgent — fetches SEC filings, market data, and news."""

from __future__ import annotations

import logging
from typing import Any

from ..tools.market_data import MarketDataClient
from ..tools.news_fetcher import NewsFetcher
from ..tools.sec_scraper import SECEdgarScraper

logger = logging.getLogger(__name__)


class DataGathererAgent:
    """Orchestrates data collection from SEC EDGAR, Alpha Vantage, and news.

    This agent is intentionally I/O-bound: it fetches raw data from external
    sources and returns it in a structured dict for downstream agents.
    No LLM call is made here — the goal is fast, parallel data retrieval.
    """

    def __init__(
        self,
        include_news: bool = True,
        include_sec: bool = True,
        include_market: bool = True,
        alpha_vantage_api_key: str | None = None,
    ) -> None:
        self._include_news = include_news
        self._include_sec = include_sec
        self._include_market = include_market
        self._alpha_vantage_api_key = alpha_vantage_api_key

    async def gather(self, ticker: str) -> dict[str, Any]:
        """Collect all available financial data for a ticker.

        Runs SEC, market, and news fetches concurrently where possible.

        Args:
            ticker: Stock ticker symbol (e.g. 'AAPL').

        Returns:
            Structured dict containing:
              - ticker
              - company_name (resolved from market data or SEC)
              - sec_summary (XBRL financial facts)
              - market_overview (Alpha Vantage overview)
              - current_quote (real-time price)
              - price_history (monthly OHLCV, 12 months)
              - news_headlines (list of headline dicts)
              - earnings_snippets (list of earnings text snippets)
              - errors (list of non-fatal error messages)
        """
        import asyncio

        ticker = ticker.strip().upper()
        logger.info("DataGathererAgent: starting collection for %s", ticker)

        result: dict[str, Any] = {
            "ticker": ticker,
            "company_name": ticker,
            "sec_summary": {},
            "market_overview": {},
            "current_quote": {},
            "price_history": [],
            "news_headlines": [],
            "earnings_snippets": [],
            "errors": [],
        }

        tasks = []
        task_labels = []

        if self._include_sec:
            tasks.append(self._fetch_sec(ticker))
            task_labels.append("sec")

        if self._include_market:
            tasks.append(self._fetch_market(ticker))
            task_labels.append("market")

        if self._include_news:
            tasks.append(self._fetch_news(ticker, result["company_name"]))
            task_labels.append("news")

        gathered = await asyncio.gather(*tasks, return_exceptions=True)

        for label, outcome in zip(task_labels, gathered):
            if isinstance(outcome, Exception):
                msg = f"{label} fetch failed: {outcome}"
                logger.warning(msg)
                result["errors"].append(msg)
            else:
                result.update(outcome)

        # Prefer company name from market overview, fall back to SEC
        if result["market_overview"].get("company_name"):
            result["company_name"] = result["market_overview"]["company_name"]
        elif result["sec_summary"].get("company_name"):
            result["company_name"] = result["sec_summary"]["company_name"]

        logger.info(
            "DataGathererAgent: collection complete for %s (%d errors)",
            ticker,
            len(result["errors"]),
        )
        return result

    # ------------------------------------------------------------------
    # Private fetch methods (each returns a partial result dict)
    # ------------------------------------------------------------------

    async def _fetch_sec(self, ticker: str) -> dict[str, Any]:
        async with SECEdgarScraper() as scraper:
            summary = await scraper.get_financial_summary(ticker)
            filings = await scraper.get_recent_filings(ticker, ["10-K", "10-Q"])
        return {
            "sec_summary": summary,
            "recent_filings": filings,
        }

    async def _fetch_market(self, ticker: str) -> dict[str, Any]:
        async with MarketDataClient(api_key=self._alpha_vantage_api_key) as client:
            overview = await client.get_company_overview(ticker)
            quote = await client.get_quote(ticker)
            history = await client.get_monthly_prices(ticker, months=12)
        return {
            "market_overview": overview,
            "current_quote": quote,
            "price_history": history,
        }

    async def _fetch_news(self, ticker: str, company_name: str) -> dict[str, Any]:
        async with NewsFetcher() as fetcher:
            headlines = await fetcher.fetch_headlines(
                ticker, company_name=company_name, max_results=20
            )
            snippets = await fetcher.fetch_earnings_snippets(ticker)
        return {
            "news_headlines": headlines,
            "earnings_snippets": snippets,
        }
