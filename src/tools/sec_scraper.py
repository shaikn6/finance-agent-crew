"""SEC EDGAR scraper using the free public XBRL API and filing search."""

from __future__ import annotations

import logging
from typing import Any

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_EDGAR_BASE = "https://data.sec.gov"
_EDGAR_SEARCH = "https://efts.sec.gov/LATEST/search-index"
_HEADERS = {
    "User-Agent": "FinanceAgentCrew research@example.com",
    "Accept-Encoding": "gzip, deflate",
}

# Common XBRL concept tags for financial data
_INCOME_CONCEPTS = [
    "us-gaap/Revenues",
    "us-gaap/RevenueFromContractWithCustomerExcludingAssessedTax",
    "us-gaap/SalesRevenueNet",
    "us-gaap/GrossProfit",
    "us-gaap/OperatingIncomeLoss",
    "us-gaap/NetIncomeLoss",
    "us-gaap/EarningsPerShareBasic",
    "us-gaap/EarningsPerShareDiluted",
]

_BALANCE_CONCEPTS = [
    "us-gaap/Assets",
    "us-gaap/Liabilities",
    "us-gaap/StockholdersEquity",
    "us-gaap/CashAndCashEquivalentsAtCarryingValue",
    "us-gaap/LongTermDebt",
    "us-gaap/ShortTermBorrowings",
]


class SECEdgarScraper:
    """Fetches structured financial data from SEC EDGAR public APIs.

    All requests use the free, unauthenticated SEC EDGAR data endpoints.
    Rate limit: max 10 requests/second per SEC fair-use policy.
    """

    def __init__(self, timeout: float = 30.0) -> None:
        self._client = httpx.AsyncClient(
            headers=_HEADERS,
            timeout=timeout,
            follow_redirects=True,
        )

    async def __aenter__(self) -> "SECEdgarScraper":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def get_company_facts(self, cik: str) -> dict[str, Any]:
        """Return the full XBRL company facts JSON for a given CIK.

        Args:
            cik: SEC CIK number (will be zero-padded to 10 digits).

        Returns:
            Raw EDGAR company facts dictionary.
        """
        padded = str(cik).zfill(10)
        url = f"{_EDGAR_BASE}/api/xbrl/companyfacts/CIK{padded}.json"
        logger.info("Fetching company facts for CIK %s", padded)
        resp = await self._client.get(url)
        resp.raise_for_status()
        return resp.json()

    async def get_cik_for_ticker(self, ticker: str) -> str | None:
        """Resolve a stock ticker to a SEC CIK number.

        Args:
            ticker: Stock ticker symbol (e.g. 'AAPL').

        Returns:
            10-digit CIK string or None if not found.
        """
        url = f"{_EDGAR_BASE}/files/company_tickers.json"
        logger.info("Resolving CIK for ticker %s", ticker)
        resp = await self._client.get(url)
        resp.raise_for_status()
        data: dict[str, dict[str, Any]] = resp.json()
        ticker_upper = ticker.upper()
        for entry in data.values():
            if entry.get("ticker", "").upper() == ticker_upper:
                return str(entry["cik_str"]).zfill(10)
        return None

    async def get_financial_summary(self, ticker: str) -> dict[str, Any]:
        """Return a structured financial summary for a ticker.

        Resolves ticker → CIK → company facts, then extracts the most
        recent annual values for key income-statement and balance-sheet
        concepts.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            Dictionary containing company metadata and financial metrics.
        """
        cik = await self.get_cik_for_ticker(ticker)
        if cik is None:
            logger.warning("No CIK found for ticker %s", ticker)
            return {"ticker": ticker, "error": "CIK not found", "financials": {}}

        facts = await self.get_company_facts(cik)
        company_name = facts.get("entityName", ticker)
        us_gaap: dict[str, Any] = facts.get("facts", {}).get("us-gaap", {})

        financials: dict[str, Any] = {}
        for concept_path in _INCOME_CONCEPTS + _BALANCE_CONCEPTS:
            _, concept = concept_path.split("/", 1)
            if concept not in us_gaap:
                continue
            value = self._latest_annual_value(us_gaap[concept])
            if value is not None:
                financials[concept] = value

        # Derive readable aliases
        revenue = _coalesce(
            financials,
            [
                "RevenueFromContractWithCustomerExcludingAssessedTax",
                "Revenues",
                "SalesRevenueNet",
            ],
        )
        net_income = financials.get("NetIncomeLoss")
        gross_profit = financials.get("GrossProfit")
        total_assets = financials.get("Assets")
        total_liabilities = financials.get("Liabilities")
        equity = financials.get("StockholdersEquity")
        long_term_debt = financials.get("LongTermDebt", 0)
        short_term_debt = financials.get("ShortTermBorrowings", 0)

        summary: dict[str, Any] = {
            "ticker": ticker,
            "cik": cik,
            "company_name": company_name,
            "financials": {
                "revenue": revenue,
                "net_income": net_income,
                "gross_profit": gross_profit,
                "total_assets": total_assets,
                "total_liabilities": total_liabilities,
                "stockholders_equity": equity,
                "total_debt": (long_term_debt or 0) + (short_term_debt or 0),
                "eps_diluted": financials.get("EarningsPerShareDiluted"),
                "eps_basic": financials.get("EarningsPerShareBasic"),
                "operating_income": financials.get("OperatingIncomeLoss"),
                "cash": financials.get("CashAndCashEquivalentsAtCarryingValue"),
            },
        }

        # Attach revenue history for CAGR calculation (last 4 annual values)
        rev_concept = _find_revenue_concept(us_gaap)
        if rev_concept:
            summary["revenue_history"] = self._annual_values_history(
                us_gaap[rev_concept], n=4
            )

        return summary

    async def get_recent_filings(
        self, ticker: str, form_types: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Return metadata for recent SEC filings.

        Args:
            ticker: Stock ticker symbol.
            form_types: Filing types to filter (e.g. ['10-K', '10-Q']).
                        Defaults to ['10-K', '10-Q', '8-K'].

        Returns:
            List of filing metadata dicts.
        """
        if form_types is None:
            form_types = ["10-K", "10-Q", "8-K"]

        cik = await self.get_cik_for_ticker(ticker)
        if cik is None:
            return []

        url = f"{_EDGAR_BASE}/cgi-bin/browse-edgar"
        params = {
            "action": "getcompany",
            "CIK": cik,
            "type": ",".join(form_types),
            "dateb": "",
            "owner": "include",
            "count": "10",
            "search_text": "",
            "output": "atom",
        }
        resp = await self._client.get(url, params=params)
        if resp.status_code != 200:
            return []

        soup = BeautifulSoup(resp.text, "xml")
        entries = []
        for entry in soup.find_all("entry"):
            entries.append(
                {
                    "title": _text(entry, "title"),
                    "filed": _text(entry, "filed"),
                    "period_of_report": _text(entry, "period-of-report"),
                    "form_type": _text(entry, "filing-type"),
                    "url": _text(entry, "filing-href"),
                }
            )
        return entries

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _latest_annual_value(concept_data: dict[str, Any]) -> float | None:
        """Extract the most recent annual (form 10-K) value for a concept."""
        units: dict[str, list[dict[str, Any]]] = concept_data.get("units", {})
        # Prefer USD values; fall back to pure numbers for ratios/EPS
        for unit_key in ("USD", "USD/shares", "shares", "pure"):
            if unit_key not in units:
                continue
            annual = [
                entry
                for entry in units[unit_key]
                if entry.get("form") in ("10-K", "10-K/A") and "val" in entry
            ]
            if annual:
                # Sort by end date descending, pick the latest
                annual.sort(key=lambda x: x.get("end", ""), reverse=True)
                return annual[0]["val"]
        return None

    @staticmethod
    def _annual_values_history(
        concept_data: dict[str, Any], n: int = 4
    ) -> list[dict[str, Any]]:
        """Return the last n annual filings for a concept (date + value)."""
        units: dict[str, list[dict[str, Any]]] = concept_data.get("units", {})
        for unit_key in ("USD", "USD/shares"):
            if unit_key not in units:
                continue
            annual = [
                {"period_end": e.get("end"), "value": e["val"]}
                for e in units[unit_key]
                if e.get("form") in ("10-K", "10-K/A") and "val" in e
            ]
            annual.sort(key=lambda x: x["period_end"], reverse=True)
            return annual[:n]
        return []


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _text(tag: Any, name: str) -> str:
    el = tag.find(name)
    return el.get_text(strip=True) if el else ""


def _coalesce(d: dict[str, Any], keys: list[str]) -> Any:  # noqa: ANN401
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _find_revenue_concept(us_gaap: dict[str, Any]) -> str | None:
    for concept in [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ]:
        if concept in us_gaap:
            return concept
    return None
