"""Financial news fetcher using DuckDuckGo search and RSS feeds."""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any
from urllib.parse import quote_plus

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; FinanceAgentCrew/1.0; research@example.com)"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# Free financial RSS feeds
_RSS_SOURCES = {
    "yahoo_finance": "https://finance.yahoo.com/rss/headline?s={ticker}",
    "seeking_alpha": "https://seekingalpha.com/symbol/{ticker}/feed.xml",
}

_DDG_SEARCH_URL = "https://html.duckduckgo.com/html/"


class NewsFetcher:
    """Fetches financial news headlines from DuckDuckGo search and RSS feeds.

    No API keys required. All sources are publicly accessible.
    """

    def __init__(self, timeout: float = 20.0) -> None:
        self._client = httpx.AsyncClient(
            headers=_HEADERS,
            timeout=timeout,
            follow_redirects=True,
        )

    async def __aenter__(self) -> "NewsFetcher":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def fetch_headlines(
        self, ticker: str, company_name: str = "", max_results: int = 20
    ) -> list[dict[str, Any]]:
        """Fetch recent news headlines for a ticker.

        Combines results from DuckDuckGo search and Yahoo Finance RSS.

        Args:
            ticker: Stock ticker symbol.
            company_name: Optional full company name for richer search.
            max_results: Maximum number of headlines to return.

        Returns:
            List of headline dicts with keys: title, url, source, published_at, snippet.
        """
        query = f"{ticker} {company_name} stock earnings news".strip()
        headlines: list[dict[str, Any]] = []

        # Try DuckDuckGo search first
        ddg_results = await self._ddg_search(query, max_results=max_results)
        headlines.extend(ddg_results)

        # Supplement with Yahoo Finance RSS
        rss_results = await self._yahoo_rss(ticker, max_results=10)
        # Deduplicate by URL
        seen_urls = {h["url"] for h in headlines}
        for item in rss_results:
            if item["url"] not in seen_urls:
                headlines.append(item)
                seen_urls.add(item["url"])

        return headlines[:max_results]

    async def fetch_earnings_snippets(self, ticker: str) -> list[str]:
        """Fetch recent earnings call / earnings report excerpts.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            List of text snippets mentioning earnings.
        """
        query = f"{ticker} earnings call transcript guidance Q4"
        results = await self._ddg_search(query, max_results=10)
        snippets = [r["snippet"] for r in results if r.get("snippet")]
        return snippets[:10]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _ddg_search(
        self, query: str, max_results: int = 20
    ) -> list[dict[str, Any]]:
        """Scrape DuckDuckGo HTML results (no API key required)."""
        try:
            resp = await self._client.post(
                _DDG_SEARCH_URL,
                data={"q": query, "b": "", "kl": "us-en"},
                headers={
                    **_HEADERS,
                    "Content-Type": "application/x-www-form-urlencoded",
                },
            )
            if resp.status_code != 200:
                logger.warning("DDG returned status %s", resp.status_code)
                return []

            soup = BeautifulSoup(resp.text, "html.parser")
            results: list[dict[str, Any]] = []

            for result in soup.select(".result")[:max_results]:
                title_el = result.select_one(".result__title a")
                snippet_el = result.select_one(".result__snippet")
                if not title_el:
                    continue

                title = title_el.get_text(strip=True)
                href = title_el.get("href", "")
                url = _extract_ddg_url(href)
                snippet = snippet_el.get_text(strip=True) if snippet_el else ""

                if not title or not url:
                    continue

                results.append(
                    {
                        "title": title,
                        "url": url,
                        "source": _extract_domain(url),
                        "published_at": None,
                        "snippet": snippet,
                    }
                )

            return results
        except Exception as exc:
            logger.warning("DuckDuckGo search failed: %s", exc)
            return []

    async def _yahoo_rss(
        self, ticker: str, max_results: int = 10
    ) -> list[dict[str, Any]]:
        """Parse Yahoo Finance RSS feed for a ticker."""
        url = _RSS_SOURCES["yahoo_finance"].format(ticker=quote_plus(ticker))
        try:
            resp = await self._client.get(url)
            if resp.status_code != 200:
                return []

            soup = BeautifulSoup(resp.text, "xml")
            items: list[dict[str, Any]] = []

            for item in soup.find_all("item")[:max_results]:
                title = _tag_text(item, "title")
                link = _tag_text(item, "link")
                pub_date_str = _tag_text(item, "pubDate")
                description = _tag_text(item, "description")

                published_at: datetime | None = None
                if pub_date_str:
                    try:
                        published_at = datetime.strptime(
                            pub_date_str, "%a, %d %b %Y %H:%M:%S %z"
                        )
                    except ValueError:
                        pass

                if title and link:
                    # Strip HTML tags from description
                    clean_desc = re.sub(r"<[^>]+>", "", description)
                    items.append(
                        {
                            "title": title,
                            "url": link,
                            "source": "Yahoo Finance",
                            "published_at": published_at.isoformat()
                            if published_at
                            else None,
                            "snippet": clean_desc[:300],
                        }
                    )

            return items
        except Exception as exc:
            logger.warning("Yahoo RSS fetch failed for %s: %s", ticker, exc)
            return []


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _extract_ddg_url(href: str) -> str:
    """Extract the real URL from DuckDuckGo's redirect href."""
    # DDG hrefs look like: //duckduckgo.com/l/?uddg=https%3A%2F%2F...
    match = re.search(r"uddg=([^&]+)", href)
    if match:
        from urllib.parse import unquote

        return unquote(match.group(1))
    if href.startswith("http"):
        return href
    return ""


def _extract_domain(url: str) -> str:
    match = re.search(r"https?://(?:www\.)?([^/]+)", url)
    return match.group(1) if match else ""


def _tag_text(tag: Any, name: str) -> str:
    el = tag.find(name)
    return el.get_text(strip=True) if el else ""
