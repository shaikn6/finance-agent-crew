"""Tests for src/tools/sec_scraper.py — SECEdgarScraper."""
from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import httpx

from src.tools.sec_scraper import (
    SECEdgarScraper,
    _text,
    _coalesce,
    _find_revenue_concept,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

COMPANY_TICKERS_JSON = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corporation"},
    "2": {"cik_str": 1018724, "ticker": "AMZN", "title": "Amazon.com Inc."},
}

COMPANY_FACTS_JSON = {
    "cik": "0000320193",
    "entityName": "Apple Inc.",
    "facts": {
        "us-gaap": {
            "RevenueFromContractWithCustomerExcludingAssessedTax": {
                "label": "Revenue",
                "description": "Revenue from contracts",
                "units": {
                    "USD": [
                        {"end": "2021-09-25", "val": 365817000000, "form": "10-K", "filed": "2021-10-29"},
                        {"end": "2022-09-24", "val": 394328000000, "form": "10-K", "filed": "2022-10-28"},
                        {"end": "2023-09-30", "val": 383285000000, "form": "10-K", "filed": "2023-11-03"},
                        {"end": "2023-06-30", "val": 81797000000, "form": "10-Q", "filed": "2023-08-04"},
                    ]
                },
            },
            "NetIncomeLoss": {
                "units": {
                    "USD": [
                        {"end": "2023-09-30", "val": 96995000000, "form": "10-K", "filed": "2023-11-03"},
                        {"end": "2022-09-24", "val": 99803000000, "form": "10-K", "filed": "2022-10-28"},
                    ]
                }
            },
            "GrossProfit": {
                "units": {
                    "USD": [
                        {"end": "2023-09-30", "val": 169148000000, "form": "10-K", "filed": "2023-11-03"},
                    ]
                }
            },
            "StockholdersEquity": {
                "units": {
                    "USD": [
                        {"end": "2023-09-30", "val": 62146000000, "form": "10-K", "filed": "2023-11-03"},
                    ]
                }
            },
            "Assets": {
                "units": {
                    "USD": [
                        {"end": "2023-09-30", "val": 352583000000, "form": "10-K", "filed": "2023-11-03"},
                    ]
                }
            },
            "Liabilities": {
                "units": {
                    "USD": [
                        {"end": "2023-09-30", "val": 290437000000, "form": "10-K", "filed": "2023-11-03"},
                    ]
                }
            },
            "LongTermDebt": {
                "units": {
                    "USD": [
                        {"end": "2023-09-30", "val": 95281000000, "form": "10-K", "filed": "2023-11-03"},
                    ]
                }
            },
            "EarningsPerShareDiluted": {
                "units": {
                    "USD/shares": [
                        {"end": "2023-09-30", "val": 6.13, "form": "10-K", "filed": "2023-11-03"},
                    ]
                }
            },
            "OperatingIncomeLoss": {
                "units": {
                    "USD": [
                        {"end": "2023-09-30", "val": 114301000000, "form": "10-K", "filed": "2023-11-03"},
                    ]
                }
            },
        }
    },
}

EDGAR_ATOM_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns:edgar="https://www.sec.gov/Archives/edgar/full-index/">
  <entry>
    <title>10-K (annual report)</title>
    <filed>2023-11-03</filed>
    <period-of-report>2023-09-30</period-of-report>
    <filing-type>10-K</filing-type>
    <filing-href>https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany</filing-href>
  </entry>
  <entry>
    <title>10-Q (quarterly report)</title>
    <filed>2023-08-04</filed>
    <period-of-report>2023-06-30</period-of-report>
    <filing-type>10-Q</filing-type>
    <filing-href>https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany</filing-href>
  </entry>
</feed>"""


def _mock_response(json_data=None, text_data=None, status_code=200):
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    if json_data is not None:
        resp.json.return_value = json_data
    if text_data is not None:
        resp.text = text_data
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=resp
        )
    return resp


# ---------------------------------------------------------------------------
# Tests for SECEdgarScraper
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestSECEdgarScraperGetCikForTicker:
    async def test_resolves_known_ticker(self):
        scraper = SECEdgarScraper()
        mock_resp = _mock_response(json_data=COMPANY_TICKERS_JSON)
        scraper._client = AsyncMock()
        scraper._client.get = AsyncMock(return_value=mock_resp)
        scraper._client.aclose = AsyncMock()

        cik = await scraper.get_cik_for_ticker("AAPL")
        assert cik == "0000320193"

    async def test_resolves_lowercase_ticker(self):
        scraper = SECEdgarScraper()
        mock_resp = _mock_response(json_data=COMPANY_TICKERS_JSON)
        scraper._client = AsyncMock()
        scraper._client.get = AsyncMock(return_value=mock_resp)
        scraper._client.aclose = AsyncMock()

        cik = await scraper.get_cik_for_ticker("aapl")
        assert cik == "0000320193"

    async def test_unknown_ticker_returns_none(self):
        scraper = SECEdgarScraper()
        mock_resp = _mock_response(json_data=COMPANY_TICKERS_JSON)
        scraper._client = AsyncMock()
        scraper._client.get = AsyncMock(return_value=mock_resp)
        scraper._client.aclose = AsyncMock()

        cik = await scraper.get_cik_for_ticker("ZZZZ")
        assert cik is None

    async def test_cik_zero_padded(self):
        scraper = SECEdgarScraper()
        tickers = {"0": {"cik_str": 1234, "ticker": "TEST", "title": "Test Corp"}}
        mock_resp = _mock_response(json_data=tickers)
        scraper._client = AsyncMock()
        scraper._client.get = AsyncMock(return_value=mock_resp)
        scraper._client.aclose = AsyncMock()

        cik = await scraper.get_cik_for_ticker("TEST")
        assert cik == "0000001234"


@pytest.mark.asyncio
class TestSECEdgarScraperGetCompanyFacts:
    async def test_fetches_and_returns_facts(self):
        scraper = SECEdgarScraper()
        mock_resp = _mock_response(json_data=COMPANY_FACTS_JSON)
        scraper._client = AsyncMock()
        scraper._client.get = AsyncMock(return_value=mock_resp)
        scraper._client.aclose = AsyncMock()

        facts = await scraper.get_company_facts("320193")
        assert facts["entityName"] == "Apple Inc."
        scraper._client.get.assert_called_once()
        call_url = scraper._client.get.call_args[0][0]
        assert "CIK0000320193" in call_url

    async def test_raises_on_http_error(self):
        scraper = SECEdgarScraper()
        mock_resp = _mock_response(status_code=404)
        scraper._client = AsyncMock()
        scraper._client.get = AsyncMock(return_value=mock_resp)
        scraper._client.aclose = AsyncMock()

        with pytest.raises(httpx.HTTPStatusError):
            await scraper.get_company_facts("999999")


@pytest.mark.asyncio
class TestSECEdgarScraperGetFinancialSummary:
    async def _make_scraper_with_mocks(self, tickers_data, facts_data):
        scraper = SECEdgarScraper()
        scraper._client = AsyncMock()
        scraper._client.aclose = AsyncMock()

        def get_side_effect(url, **kwargs):
            if "company_tickers" in url:
                return _mock_response(json_data=tickers_data)
            elif "companyfacts" in url:
                return _mock_response(json_data=facts_data)
            return _mock_response(json_data={})

        scraper._client.get = AsyncMock(side_effect=get_side_effect)
        return scraper

    async def test_summary_for_known_ticker(self):
        scraper = await self._make_scraper_with_mocks(
            COMPANY_TICKERS_JSON, COMPANY_FACTS_JSON
        )
        summary = await scraper.get_financial_summary("AAPL")
        assert summary["ticker"] == "AAPL"
        assert summary["company_name"] == "Apple Inc."
        assert summary["financials"]["revenue"] == 383285000000
        assert summary["financials"]["net_income"] == 96995000000
        assert "revenue_history" in summary

    async def test_summary_unknown_ticker(self):
        scraper = SECEdgarScraper()
        mock_resp = _mock_response(json_data=COMPANY_TICKERS_JSON)
        scraper._client = AsyncMock()
        scraper._client.get = AsyncMock(return_value=mock_resp)
        scraper._client.aclose = AsyncMock()

        summary = await scraper.get_financial_summary("ZZZZ")
        assert summary["error"] == "CIK not found"
        assert summary["financials"] == {}

    async def test_summary_total_debt_aggregation(self):
        scraper = await self._make_scraper_with_mocks(
            COMPANY_TICKERS_JSON, COMPANY_FACTS_JSON
        )
        summary = await scraper.get_financial_summary("AAPL")
        # LongTermDebt=95281000000, ShortTermBorrowings not present => 0
        assert summary["financials"]["total_debt"] == 95281000000

    async def test_revenue_history_length(self):
        scraper = await self._make_scraper_with_mocks(
            COMPANY_TICKERS_JSON, COMPANY_FACTS_JSON
        )
        summary = await scraper.get_financial_summary("AAPL")
        # We have 3 annual 10-K entries
        assert len(summary["revenue_history"]) == 3

    async def test_summary_no_revenue_concept(self):
        facts_no_revenue = {
            "entityName": "No Revenue Corp",
            "facts": {
                "us-gaap": {
                    "NetIncomeLoss": {
                        "units": {
                            "USD": [
                                {"end": "2023-09-30", "val": 1000, "form": "10-K", "filed": "2023-11-03"}
                            ]
                        }
                    }
                }
            }
        }
        scraper = await self._make_scraper_with_mocks(COMPANY_TICKERS_JSON, facts_no_revenue)
        summary = await scraper.get_financial_summary("AAPL")
        assert "revenue_history" not in summary
        assert summary["financials"]["revenue"] is None


@pytest.mark.asyncio
class TestSECEdgarScraperGetRecentFilings:
    async def test_parses_atom_feed(self):
        scraper = SECEdgarScraper()
        scraper._client = AsyncMock()
        scraper._client.aclose = AsyncMock()

        def get_side_effect(url, **kwargs):
            if "company_tickers" in url:
                return _mock_response(json_data=COMPANY_TICKERS_JSON)
            return _mock_response(text_data=EDGAR_ATOM_FEED, status_code=200)

        scraper._client.get = AsyncMock(side_effect=get_side_effect)
        filings = await scraper.get_recent_filings("AAPL", ["10-K", "10-Q"])
        assert len(filings) == 2
        assert filings[0]["form_type"] == "10-K"
        assert filings[0]["filed"] == "2023-11-03"

    async def test_unknown_ticker_returns_empty(self):
        scraper = SECEdgarScraper()
        scraper._client = AsyncMock()
        scraper._client.get = AsyncMock(return_value=_mock_response(json_data=COMPANY_TICKERS_JSON))
        scraper._client.aclose = AsyncMock()

        filings = await scraper.get_recent_filings("ZZZZ")
        assert filings == []

    async def test_non_200_response_returns_empty(self):
        scraper = SECEdgarScraper()
        scraper._client = AsyncMock()
        scraper._client.aclose = AsyncMock()

        def get_side_effect(url, **kwargs):
            if "company_tickers" in url:
                return _mock_response(json_data=COMPANY_TICKERS_JSON)
            return _mock_response(status_code=503, text_data="")

        scraper._client.get = AsyncMock(side_effect=get_side_effect)
        filings = await scraper.get_recent_filings("AAPL")
        assert filings == []

    async def test_default_form_types(self):
        scraper = SECEdgarScraper()
        scraper._client = AsyncMock()
        scraper._client.aclose = AsyncMock()

        def get_side_effect(url, **kwargs):
            if "company_tickers" in url:
                return _mock_response(json_data=COMPANY_TICKERS_JSON)
            return _mock_response(text_data=EDGAR_ATOM_FEED)

        scraper._client.get = AsyncMock(side_effect=get_side_effect)
        await scraper.get_recent_filings("AAPL")  # No form_types arg
        # Should not raise; default form_types applied internally


@pytest.mark.asyncio
class TestSECEdgarScraperContextManager:
    async def test_aenter_aexit(self):
        async with SECEdgarScraper() as scraper:
            assert scraper is not None

    async def test_client_closed_on_exit(self):
        scraper = SECEdgarScraper()
        scraper._client = AsyncMock()
        scraper._client.aclose = AsyncMock()
        async with scraper:
            pass
        scraper._client.aclose.assert_called_once()


# ---------------------------------------------------------------------------
# Tests for private static helpers
# ---------------------------------------------------------------------------

class TestLatestAnnualValue:
    def test_returns_latest_10k_value(self):
        concept_data = {
            "units": {
                "USD": [
                    {"end": "2022-09-30", "val": 100, "form": "10-K"},
                    {"end": "2023-09-30", "val": 200, "form": "10-K"},
                    {"end": "2023-06-30", "val": 50, "form": "10-Q"},
                ]
            }
        }
        result = SECEdgarScraper._latest_annual_value(concept_data)
        assert result == 200

    def test_returns_none_when_no_10k(self):
        concept_data = {
            "units": {
                "USD": [
                    {"end": "2023-06-30", "val": 50, "form": "10-Q"},
                ]
            }
        }
        result = SECEdgarScraper._latest_annual_value(concept_data)
        assert result is None

    def test_falls_back_to_usd_per_shares(self):
        concept_data = {
            "units": {
                "USD/shares": [
                    {"end": "2023-09-30", "val": 6.13, "form": "10-K"},
                ]
            }
        }
        result = SECEdgarScraper._latest_annual_value(concept_data)
        assert result == 6.13

    def test_empty_units_returns_none(self):
        result = SECEdgarScraper._latest_annual_value({"units": {}})
        assert result is None

    def test_10k_amendment_included(self):
        concept_data = {
            "units": {
                "USD": [
                    {"end": "2023-09-30", "val": 999, "form": "10-K/A"},
                ]
            }
        }
        result = SECEdgarScraper._latest_annual_value(concept_data)
        assert result == 999


class TestAnnualValuesHistory:
    def test_returns_sorted_desc(self):
        concept_data = {
            "units": {
                "USD": [
                    {"end": "2021-09-30", "val": 100, "form": "10-K"},
                    {"end": "2023-09-30", "val": 300, "form": "10-K"},
                    {"end": "2022-09-30", "val": 200, "form": "10-K"},
                ]
            }
        }
        history = SECEdgarScraper._annual_values_history(concept_data, n=4)
        assert len(history) == 3
        assert history[0]["value"] == 300
        assert history[1]["value"] == 200

    def test_excludes_10q(self):
        concept_data = {
            "units": {
                "USD": [
                    {"end": "2023-09-30", "val": 100, "form": "10-K"},
                    {"end": "2023-06-30", "val": 25, "form": "10-Q"},
                ]
            }
        }
        history = SECEdgarScraper._annual_values_history(concept_data, n=4)
        assert len(history) == 1

    def test_limits_to_n(self):
        concept_data = {
            "units": {
                "USD": [
                    {"end": f"202{i}-09-30", "val": i * 100, "form": "10-K"}
                    for i in range(1, 6)
                ]
            }
        }
        history = SECEdgarScraper._annual_values_history(concept_data, n=3)
        assert len(history) == 3

    def test_no_usd_units_returns_empty(self):
        concept_data = {"units": {"pure": [{"end": "2023-09-30", "val": 1.5, "form": "10-K"}]}}
        history = SECEdgarScraper._annual_values_history(concept_data, n=4)
        assert history == []


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

class TestTextHelper:
    def test_extracts_text(self):
        from bs4 import BeautifulSoup
        soup = BeautifulSoup("<root><title>Hello World</title></root>", "xml")
        assert _text(soup.find("root"), "title") == "Hello World"

    def test_missing_tag_returns_empty(self):
        from bs4 import BeautifulSoup
        soup = BeautifulSoup("<root></root>", "xml")
        assert _text(soup.find("root"), "missing") == ""


class TestCoalesceHelper:
    def test_returns_first_non_none(self):
        d = {"a": None, "b": 42, "c": 99}
        assert _coalesce(d, ["a", "b", "c"]) == 42

    def test_all_none_returns_none(self):
        d = {"a": None, "b": None}
        assert _coalesce(d, ["a", "b"]) is None

    def test_first_found_returned(self):
        d = {"x": 10, "y": 20}
        assert _coalesce(d, ["x", "y"]) == 10


class TestFindRevenueConcept:
    def test_finds_first_matching(self):
        us_gaap = {"RevenueFromContractWithCustomerExcludingAssessedTax": {}, "Revenues": {}}
        assert _find_revenue_concept(us_gaap) == "RevenueFromContractWithCustomerExcludingAssessedTax"

    def test_falls_back_to_revenues(self):
        us_gaap = {"Revenues": {}}
        assert _find_revenue_concept(us_gaap) == "Revenues"

    def test_falls_back_to_sales(self):
        us_gaap = {"SalesRevenueNet": {}}
        assert _find_revenue_concept(us_gaap) == "SalesRevenueNet"

    def test_returns_none_when_no_match(self):
        us_gaap = {"OperatingIncomeLoss": {}}
        assert _find_revenue_concept(us_gaap) is None
