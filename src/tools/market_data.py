"""Market data client using Alpha Vantage free tier."""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_AV_BASE = "https://www.alphavantage.co/query"
_DEFAULT_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "demo")


class MarketDataClient:
    """Fetches real-time and historical market data from Alpha Vantage.

    Free tier: 25 requests/day, 5 requests/minute.
    Set ALPHA_VANTAGE_API_KEY env var for production use.

    Docs: https://www.alphavantage.co/documentation/
    """

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 20.0,
    ) -> None:
        self._api_key = api_key or _DEFAULT_API_KEY
        self._client = httpx.AsyncClient(timeout=timeout)

    async def __aenter__(self) -> "MarketDataClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def get_quote(self, ticker: str) -> dict[str, Any]:
        """Return the current global quote for a ticker.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            Dict containing: price, change, change_percent, volume,
            latest_trading_day, previous_close, 52_week_high, 52_week_low.
        """
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": ticker,
            "apikey": self._api_key,
        }
        data = await self._get(params)
        raw = data.get("Global Quote", {})
        if not raw:
            logger.warning("No quote data returned for %s", ticker)
            return {"ticker": ticker, "error": "No data available"}

        return {
            "ticker": ticker,
            "price": _safe_float(raw.get("05. price")),
            "change": _safe_float(raw.get("09. change")),
            "change_percent": raw.get("10. change percent", "").replace("%", ""),
            "volume": _safe_int(raw.get("06. volume")),
            "latest_trading_day": raw.get("07. latest trading day"),
            "previous_close": _safe_float(raw.get("08. previous close")),
            "open": _safe_float(raw.get("02. open")),
            "high": _safe_float(raw.get("03. high")),
            "low": _safe_float(raw.get("04. low")),
        }

    async def get_company_overview(self, ticker: str) -> dict[str, Any]:
        """Return company overview including fundamentals from Alpha Vantage.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            Dict with sector, industry, P/E, P/B, market cap, beta, 52-week range, etc.
        """
        params = {
            "function": "OVERVIEW",
            "symbol": ticker,
            "apikey": self._api_key,
        }
        data = await self._get(params)
        if not data or "Symbol" not in data:
            return {"ticker": ticker, "error": "No overview data"}

        return {
            "ticker": ticker,
            "company_name": data.get("Name", ""),
            "description": data.get("Description", ""),
            "sector": data.get("Sector", ""),
            "industry": data.get("Industry", ""),
            "country": data.get("Country", ""),
            "exchange": data.get("Exchange", ""),
            "market_cap": _safe_float(data.get("MarketCapitalization")),
            "pe_ratio": _safe_float(data.get("PERatio")),
            "pe_forward": _safe_float(data.get("ForwardPE")),
            "pb_ratio": _safe_float(data.get("PriceToBookRatio")),
            "ps_ratio": _safe_float(data.get("PriceToSalesRatioTTM")),
            "peg_ratio": _safe_float(data.get("PEGRatio")),
            "beta": _safe_float(data.get("Beta")),
            "eps": _safe_float(data.get("EPS")),
            "dividend_yield": _safe_float(data.get("DividendYield")),
            "profit_margin": _safe_float(data.get("ProfitMargin")),
            "operating_margin_ttm": _safe_float(data.get("OperatingMarginTTM")),
            "return_on_equity_ttm": _safe_float(data.get("ReturnOnEquityTTM")),
            "return_on_assets_ttm": _safe_float(data.get("ReturnOnAssetsTTM")),
            "revenue_ttm": _safe_float(data.get("RevenueTTM")),
            "gross_profit_ttm": _safe_float(data.get("GrossProfitTTM")),
            "revenue_per_share_ttm": _safe_float(data.get("RevenuePerShareTTM")),
            "quarterly_revenue_growth": _safe_float(
                data.get("QuarterlyRevenueGrowthYOY")
            ),
            "quarterly_earnings_growth": _safe_float(
                data.get("QuarterlyEarningsGrowthYOY")
            ),
            "week_52_high": _safe_float(data.get("52WeekHigh")),
            "week_52_low": _safe_float(data.get("52WeekLow")),
            "moving_avg_50d": _safe_float(data.get("50DayMovingAverage")),
            "moving_avg_200d": _safe_float(data.get("200DayMovingAverage")),
            "shares_outstanding": _safe_float(data.get("SharesOutstanding")),
            "analyst_target_price": _safe_float(data.get("AnalystTargetPrice")),
        }

    async def get_monthly_prices(
        self, ticker: str, months: int = 12
    ) -> list[dict[str, Any]]:
        """Return monthly OHLCV data for the past N months.

        Args:
            ticker: Stock ticker symbol.
            months: Number of months of history to return.

        Returns:
            List of monthly OHLCV dicts sorted newest first.
        """
        params = {
            "function": "TIME_SERIES_MONTHLY",
            "symbol": ticker,
            "apikey": self._api_key,
        }
        data = await self._get(params)
        series: dict[str, Any] = data.get("Monthly Time Series", {})
        if not series:
            return []

        results = []
        for date_str, ohlcv in sorted(series.items(), reverse=True)[:months]:
            results.append(
                {
                    "date": date_str,
                    "open": _safe_float(ohlcv.get("1. open")),
                    "high": _safe_float(ohlcv.get("2. high")),
                    "low": _safe_float(ohlcv.get("3. low")),
                    "close": _safe_float(ohlcv.get("4. close")),
                    "volume": _safe_int(ohlcv.get("5. volume")),
                }
            )
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _get(self, params: dict[str, str]) -> dict[str, Any]:
        try:
            resp = await self._client.get(_AV_BASE, params=params)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as exc:
            logger.error("Alpha Vantage HTTP error: %s", exc)
            return {}
        except Exception as exc:
            logger.error("Alpha Vantage request failed: %s", exc)
            return {}


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _safe_float(value: Any) -> float | None:  # noqa: ANN401
    if value is None or value == "None" or value == "-":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:  # noqa: ANN401
    if value is None or value == "None":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
