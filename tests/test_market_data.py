"""Tests for src/tools/market_data.py — MarketDataClient."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

import httpx

from src.tools.market_data import MarketDataClient, _safe_float, _safe_int


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _mock_http_response(json_data=None, status_code=200):
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json = MagicMock(return_value=json_data or {})
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "HTTP error", request=MagicMock(), response=resp
        )
    return resp


GLOBAL_QUOTE_RESPONSE = {
    "Global Quote": {
        "01. symbol": "AAPL",
        "02. open": "175.00",
        "03. high": "180.00",
        "04. low": "174.50",
        "05. price": "178.50",
        "06. volume": "52345678",
        "07. latest trading day": "2024-01-15",
        "08. previous close": "177.00",
        "09. change": "1.50",
        "10. change percent": "0.85%",
    }
}

OVERVIEW_RESPONSE = {
    "Symbol": "AAPL",
    "Name": "Apple Inc.",
    "Description": "Apple Inc. designs and markets consumer electronics.",
    "Sector": "Technology",
    "Industry": "Consumer Electronics",
    "Country": "USA",
    "Exchange": "NASDAQ",
    "MarketCapitalization": "2800000000000",
    "PERatio": "29.50",
    "ForwardPE": "27.00",
    "PriceToBookRatio": "47.50",
    "PriceToSalesRatioTTM": "7.20",
    "PEGRatio": "2.50",
    "Beta": "1.28",
    "EPS": "6.13",
    "DividendYield": "0.0056",
    "ProfitMargin": "0.2531",
    "OperatingMarginTTM": "0.2984",
    "ReturnOnEquityTTM": "1.4726",
    "ReturnOnAssetsTTM": "0.2869",
    "RevenueTTM": "383285000000",
    "GrossProfitTTM": "169148000000",
    "RevenuePerShareTTM": "24.12",
    "QuarterlyRevenueGrowthYOY": "-0.0107",
    "QuarterlyEarningsGrowthYOY": "0.1100",
    "52WeekHigh": "198.23",
    "52WeekLow": "124.17",
    "50DayMovingAverage": "177.22",
    "200DayMovingAverage": "163.45",
    "SharesOutstanding": "15550000000",
    "AnalystTargetPrice": "195.50",
}

MONTHLY_SERIES_RESPONSE = {
    "Monthly Time Series": {
        "2024-01-31": {
            "1. open": "185.00",
            "2. high": "191.00",
            "3. low": "183.00",
            "4. close": "187.00",
            "5. volume": "1234567890",
        },
        "2023-12-31": {
            "1. open": "190.00",
            "2. high": "195.00",
            "3. low": "188.00",
            "4. close": "192.00",
            "5. volume": "987654321",
        },
    }
}


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestMarketDataClientContextManager:
    async def test_aenter_aexit(self):
        async with MarketDataClient(api_key="testkey") as client:
            assert client is not None

    async def test_client_closed_on_exit(self):
        client = MarketDataClient(api_key="testkey")
        client._client = AsyncMock()
        client._client.aclose = AsyncMock()
        async with client:
            pass
        client._client.aclose.assert_called_once()


# ---------------------------------------------------------------------------
# get_quote
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestGetQuote:
    async def _client_with_response(self, json_data, status_code=200):
        client = MarketDataClient(api_key="demo")
        client._client = AsyncMock()
        client._client.aclose = AsyncMock()
        client._client.get = AsyncMock(
            return_value=_mock_http_response(json_data, status_code)
        )
        return client

    async def test_returns_quote_data(self):
        client = await self._client_with_response(GLOBAL_QUOTE_RESPONSE)
        quote = await client.get_quote("AAPL")
        assert quote["ticker"] == "AAPL"
        assert quote["price"] == 178.50
        assert quote["volume"] == 52345678
        assert quote["change"] == 1.50

    async def test_change_percent_no_percent_sign(self):
        client = await self._client_with_response(GLOBAL_QUOTE_RESPONSE)
        quote = await client.get_quote("AAPL")
        assert "%" not in quote["change_percent"]

    async def test_empty_response_returns_error_dict(self):
        client = await self._client_with_response({"Global Quote": {}})
        quote = await client.get_quote("UNKNOWN")
        assert "error" in quote

    async def test_http_error_returns_empty_dict(self):
        client = MarketDataClient(api_key="demo")
        client._client = AsyncMock()
        client._client.aclose = AsyncMock()
        client._client.get = AsyncMock(
            side_effect=httpx.HTTPStatusError("error", request=MagicMock(), response=MagicMock())
        )
        quote = await client.get_quote("FAIL")
        # Should not raise; _get returns {} on error
        assert "ticker" in quote or quote == {}

    async def test_network_error_returns_empty_dict(self):
        client = MarketDataClient(api_key="demo")
        client._client = AsyncMock()
        client._client.aclose = AsyncMock()
        client._client.get = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
        quote = await client.get_quote("FAIL")
        assert "error" in quote or quote == {}


# ---------------------------------------------------------------------------
# get_company_overview
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestGetCompanyOverview:
    async def _client_with_response(self, json_data):
        client = MarketDataClient(api_key="demo")
        client._client = AsyncMock()
        client._client.aclose = AsyncMock()
        client._client.get = AsyncMock(return_value=_mock_http_response(json_data))
        return client

    async def test_returns_overview_data(self):
        client = await self._client_with_response(OVERVIEW_RESPONSE)
        overview = await client.get_company_overview("AAPL")
        assert overview["ticker"] == "AAPL"
        assert overview["company_name"] == "Apple Inc."
        assert overview["sector"] == "Technology"
        assert overview["pe_ratio"] == 29.50
        assert overview["market_cap"] == 2800000000000.0

    async def test_missing_symbol_returns_error(self):
        client = await self._client_with_response({})
        overview = await client.get_company_overview("ZZZZZ")
        assert "error" in overview

    async def test_none_values_handled(self):
        response = {**OVERVIEW_RESPONSE, "PERatio": "None", "Beta": "-"}
        client = await self._client_with_response(response)
        overview = await client.get_company_overview("AAPL")
        assert overview["pe_ratio"] is None
        assert overview["beta"] is None

    async def test_all_numeric_fields_parsed(self):
        client = await self._client_with_response(OVERVIEW_RESPONSE)
        overview = await client.get_company_overview("AAPL")
        assert overview["beta"] == 1.28
        assert overview["return_on_equity_ttm"] == 1.4726
        assert overview["analyst_target_price"] == 195.50


# ---------------------------------------------------------------------------
# get_monthly_prices
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestGetMonthlyPrices:
    async def _client_with_response(self, json_data):
        client = MarketDataClient(api_key="demo")
        client._client = AsyncMock()
        client._client.aclose = AsyncMock()
        client._client.get = AsyncMock(return_value=_mock_http_response(json_data))
        return client

    async def test_returns_sorted_desc(self):
        client = await self._client_with_response(MONTHLY_SERIES_RESPONSE)
        prices = await client.get_monthly_prices("AAPL", months=12)
        assert len(prices) == 2
        assert prices[0]["date"] > prices[1]["date"]

    async def test_ohlcv_parsed(self):
        client = await self._client_with_response(MONTHLY_SERIES_RESPONSE)
        prices = await client.get_monthly_prices("AAPL")
        p = prices[0]
        assert p["open"] is not None
        assert p["high"] is not None
        assert p["low"] is not None
        assert p["close"] is not None
        assert p["volume"] is not None

    async def test_empty_series_returns_empty_list(self):
        client = await self._client_with_response({})
        prices = await client.get_monthly_prices("ZZZZZ")
        assert prices == []

    async def test_months_limit_respected(self):
        # Create 15 months of data
        series = {
            f"2023-{str(m).zfill(2)}-28": {
                "1. open": "100", "2. high": "110", "3. low": "95",
                "4. close": "105", "5. volume": "1000000"
            }
            for m in range(1, 16)
        }
        client = await self._client_with_response({"Monthly Time Series": series})
        prices = await client.get_monthly_prices("AAPL", months=6)
        assert len(prices) <= 6


# ---------------------------------------------------------------------------
# Module-level helpers: _safe_float, _safe_int
# ---------------------------------------------------------------------------

class TestSafeFloat:
    def test_valid_string(self):
        assert _safe_float("29.50") == 29.50

    def test_valid_float(self):
        assert _safe_float(3.14) == 3.14

    def test_none_returns_none(self):
        assert _safe_float(None) is None

    def test_none_string_returns_none(self):
        assert _safe_float("None") is None

    def test_dash_returns_none(self):
        assert _safe_float("-") is None

    def test_invalid_string_returns_none(self):
        assert _safe_float("N/A") is None

    def test_zero_string(self):
        assert _safe_float("0") == 0.0

    def test_negative(self):
        assert _safe_float("-1.5") == -1.5


class TestSafeInt:
    def test_valid_string(self):
        assert _safe_int("52345678") == 52345678

    def test_valid_int(self):
        assert _safe_int(42) == 42

    def test_none_returns_none(self):
        assert _safe_int(None) is None

    def test_none_string_returns_none(self):
        assert _safe_int("None") is None

    def test_invalid_returns_none(self):
        assert _safe_int("abc") is None

    def test_float_string_truncates(self):
        # int("3.5") raises, so should return None
        assert _safe_int("3.5") is None
