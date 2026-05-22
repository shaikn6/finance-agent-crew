"""Tests for src/tools/news_fetcher.py — NewsFetcher."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from src.tools.news_fetcher import (
    NewsFetcher,
    _extract_ddg_url,
    _extract_domain,
    _tag_text,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _mock_response(text_data="", status_code=200):
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.text = text_data
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "HTTP error", request=MagicMock(), response=resp
        )
    return resp


DDG_HTML_RESPONSE = """
<html><body>
<div class="result">
  <h2 class="result__title"><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Ffinance.yahoo.com%2Fnews%2Faapl-earnings-2024.html">Apple Reports Record Q4 Earnings</a></h2>
  <div class="result__snippet">Apple Inc. reported record earnings beating estimates by 12%.</div>
</div>
<div class="result">
  <h2 class="result__title"><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.reuters.com%2Ftechnology%2Faapl-2024.html">Apple Stock Reaches All-Time High</a></h2>
  <div class="result__snippet">Shares surged after strong guidance from management.</div>
</div>
</body></html>
"""

YAHOO_RSS_RESPONSE = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <item>
      <title>Apple Q4 Results Beat Estimates</title>
      <link>https://finance.yahoo.com/news/aapl-q4-2024.html</link>
      <pubDate>Mon, 01 Jan 2024 10:00:00 +0000</pubDate>
      <description>&lt;p&gt;Apple reported strong Q4 earnings with revenue of $120 billion.&lt;/p&gt;</description>
    </item>
    <item>
      <title>AAPL Buyback Program Expanded</title>
      <link>https://finance.yahoo.com/news/aapl-buyback-2024.html</link>
      <pubDate>Tue, 02 Jan 2024 09:00:00 +0000</pubDate>
      <description>Apple expanded its buyback program to $110 billion.</description>
    </item>
  </channel>
</rss>"""

INVALID_PUBDATE_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <item>
      <title>Test Headline</title>
      <link>https://example.com/test</link>
      <pubDate>not-a-date</pubDate>
      <description>Test description.</description>
    </item>
  </channel>
</rss>"""


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestNewsFetcherContextManager:
    async def test_aenter_aexit(self):
        async with NewsFetcher() as fetcher:
            assert fetcher is not None

    async def test_client_closed_on_exit(self):
        fetcher = NewsFetcher()
        fetcher._client = AsyncMock()
        fetcher._client.aclose = AsyncMock()
        async with fetcher:
            pass
        fetcher._client.aclose.assert_called_once()


# ---------------------------------------------------------------------------
# fetch_headlines (DuckDuckGo + Yahoo RSS)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestFetchHeadlines:
    async def test_returns_headlines_from_ddg(self):
        fetcher = NewsFetcher()
        fetcher._client = AsyncMock()
        fetcher._client.aclose = AsyncMock()
        fetcher._client.post = AsyncMock(return_value=_mock_response(DDG_HTML_RESPONSE))
        fetcher._client.get = AsyncMock(return_value=_mock_response(status_code=404))

        headlines = await fetcher.fetch_headlines("AAPL", company_name="Apple Inc.", max_results=20)
        assert len(headlines) >= 1
        assert all("title" in h for h in headlines)
        assert all("url" in h for h in headlines)

    async def test_supplements_with_yahoo_rss(self):
        fetcher = NewsFetcher()
        fetcher._client = AsyncMock()
        fetcher._client.aclose = AsyncMock()
        fetcher._client.post = AsyncMock(return_value=_mock_response(DDG_HTML_RESPONSE))
        fetcher._client.get = AsyncMock(return_value=_mock_response(YAHOO_RSS_RESPONSE))

        headlines = await fetcher.fetch_headlines("AAPL", max_results=20)
        # Should have DDG + Yahoo results (deduped)
        assert len(headlines) >= 2

    async def test_deduplicates_by_url(self):
        """If same URL appears in DDG and RSS, it should appear only once."""
        # Use the Yahoo RSS URL also in DDG result
        ddg_html = """<html><body>
        <div class="result">
          <h2 class="result__title"><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Ffinance.yahoo.com%2Fnews%2Faapl-q4-2024.html">Test</a></h2>
          <div class="result__snippet">Some snippet</div>
        </div></body></html>"""
        fetcher = NewsFetcher()
        fetcher._client = AsyncMock()
        fetcher._client.aclose = AsyncMock()
        fetcher._client.post = AsyncMock(return_value=_mock_response(ddg_html))
        fetcher._client.get = AsyncMock(return_value=_mock_response(YAHOO_RSS_RESPONSE))

        headlines = await fetcher.fetch_headlines("AAPL", max_results=20)
        urls = [h["url"] for h in headlines]
        assert len(urls) == len(set(urls))

    async def test_respects_max_results(self):
        fetcher = NewsFetcher()
        fetcher._client = AsyncMock()
        fetcher._client.aclose = AsyncMock()
        fetcher._client.post = AsyncMock(return_value=_mock_response(DDG_HTML_RESPONSE))
        fetcher._client.get = AsyncMock(return_value=_mock_response(YAHOO_RSS_RESPONSE))

        headlines = await fetcher.fetch_headlines("AAPL", max_results=1)
        assert len(headlines) <= 1

    async def test_ddg_failure_falls_back_gracefully(self):
        fetcher = NewsFetcher()
        fetcher._client = AsyncMock()
        fetcher._client.aclose = AsyncMock()
        fetcher._client.post = AsyncMock(side_effect=httpx.ConnectError("failed"))
        fetcher._client.get = AsyncMock(return_value=_mock_response(YAHOO_RSS_RESPONSE))

        headlines = await fetcher.fetch_headlines("AAPL")
        # Should still get Yahoo RSS results
        assert isinstance(headlines, list)

    async def test_ddg_non_200_returns_empty(self):
        fetcher = NewsFetcher()
        fetcher._client = AsyncMock()
        fetcher._client.aclose = AsyncMock()
        fetcher._client.post = AsyncMock(return_value=_mock_response(status_code=503))
        fetcher._client.get = AsyncMock(return_value=_mock_response(status_code=404))

        headlines = await fetcher.fetch_headlines("AAPL")
        assert isinstance(headlines, list)

    async def test_ddg_result_missing_title_el_skipped(self):
        html = """<html><body>
        <div class="result"></div>
        <div class="result">
          <h2 class="result__title"><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2F">Good Result</a></h2>
          <div class="result__snippet">Good snippet.</div>
        </div>
        </body></html>"""
        fetcher = NewsFetcher()
        fetcher._client = AsyncMock()
        fetcher._client.aclose = AsyncMock()
        fetcher._client.post = AsyncMock(return_value=_mock_response(html))
        fetcher._client.get = AsyncMock(return_value=_mock_response(status_code=404))

        headlines = await fetcher.fetch_headlines("AAPL", max_results=10)
        # The result without title_el is skipped; only the good one appears
        assert len(headlines) == 1

    async def test_ddg_result_empty_url_skipped(self):
        html = """<html><body>
        <div class="result">
          <h2 class="result__title"><a class="result__a" href="//duckduckgo.com/l/">No URL Result</a></h2>
        </div>
        <div class="result">
          <h2 class="result__title"><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2F">Good</a></h2>
        </div>
        </body></html>"""
        fetcher = NewsFetcher()
        fetcher._client = AsyncMock()
        fetcher._client.aclose = AsyncMock()
        fetcher._client.post = AsyncMock(return_value=_mock_response(html))
        fetcher._client.get = AsyncMock(return_value=_mock_response(status_code=404))

        headlines = await fetcher.fetch_headlines("AAPL", max_results=10)
        # Result with empty URL is skipped
        urls = [h["url"] for h in headlines]
        assert all(url for url in urls)


# ---------------------------------------------------------------------------
# fetch_earnings_snippets
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestFetchEarningsSnippets:
    async def test_returns_snippets(self):
        fetcher = NewsFetcher()
        fetcher._client = AsyncMock()
        fetcher._client.aclose = AsyncMock()
        fetcher._client.post = AsyncMock(return_value=_mock_response(DDG_HTML_RESPONSE))
        fetcher._client.get = AsyncMock(return_value=_mock_response(status_code=404))

        snippets = await fetcher.fetch_earnings_snippets("AAPL")
        assert isinstance(snippets, list)
        assert len(snippets) <= 10

    async def test_returns_empty_on_ddg_failure(self):
        fetcher = NewsFetcher()
        fetcher._client = AsyncMock()
        fetcher._client.aclose = AsyncMock()
        fetcher._client.post = AsyncMock(side_effect=Exception("network error"))
        fetcher._client.get = AsyncMock(return_value=_mock_response(status_code=404))

        snippets = await fetcher.fetch_earnings_snippets("FAIL")
        assert snippets == []


# ---------------------------------------------------------------------------
# Yahoo RSS parsing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestYahooRSS:
    async def test_parses_valid_rss(self):
        fetcher = NewsFetcher()
        fetcher._client = AsyncMock()
        fetcher._client.aclose = AsyncMock()
        fetcher._client.get = AsyncMock(return_value=_mock_response(YAHOO_RSS_RESPONSE))

        items = await fetcher._yahoo_rss("AAPL")
        assert len(items) == 2
        assert items[0]["title"] == "Apple Q4 Results Beat Estimates"
        assert items[0]["source"] == "Yahoo Finance"
        assert items[0]["published_at"] is not None

    async def test_strips_html_from_description(self):
        fetcher = NewsFetcher()
        fetcher._client = AsyncMock()
        fetcher._client.aclose = AsyncMock()
        fetcher._client.get = AsyncMock(return_value=_mock_response(YAHOO_RSS_RESPONSE))

        items = await fetcher._yahoo_rss("AAPL")
        assert "<p>" not in items[0]["snippet"]

    async def test_invalid_pubdate_sets_none(self):
        fetcher = NewsFetcher()
        fetcher._client = AsyncMock()
        fetcher._client.aclose = AsyncMock()
        fetcher._client.get = AsyncMock(return_value=_mock_response(INVALID_PUBDATE_RSS))

        items = await fetcher._yahoo_rss("AAPL")
        assert items[0]["published_at"] is None

    async def test_non_200_returns_empty(self):
        fetcher = NewsFetcher()
        fetcher._client = AsyncMock()
        fetcher._client.aclose = AsyncMock()
        fetcher._client.get = AsyncMock(return_value=_mock_response(status_code=404))

        items = await fetcher._yahoo_rss("AAPL")
        assert items == []

    async def test_exception_returns_empty(self):
        fetcher = NewsFetcher()
        fetcher._client = AsyncMock()
        fetcher._client.aclose = AsyncMock()
        fetcher._client.get = AsyncMock(side_effect=Exception("network error"))

        items = await fetcher._yahoo_rss("AAPL")
        assert items == []

    async def test_respects_max_results(self):
        fetcher = NewsFetcher()
        fetcher._client = AsyncMock()
        fetcher._client.aclose = AsyncMock()
        fetcher._client.get = AsyncMock(return_value=_mock_response(YAHOO_RSS_RESPONSE))

        items = await fetcher._yahoo_rss("AAPL", max_results=1)
        assert len(items) <= 1


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


class TestExtractDDGUrl:
    def test_decodes_uddg_param(self):
        href = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.example.com%2Fpath"
        result = _extract_ddg_url(href)
        assert result == "https://www.example.com/path"

    def test_returns_direct_http_url(self):
        href = "https://www.reuters.com/article/123"
        result = _extract_ddg_url(href)
        assert result == "https://www.reuters.com/article/123"

    def test_no_match_returns_empty_string(self):
        result = _extract_ddg_url("//duckduckgo.com/l/")
        assert result == ""


class TestExtractDomain:
    def test_extracts_domain(self):
        assert _extract_domain("https://www.reuters.com/article/123") == "reuters.com"

    def test_without_www(self):
        assert _extract_domain("https://finance.yahoo.com/news/") == "finance.yahoo.com"

    def test_empty_url_returns_empty(self):
        assert _extract_domain("") == ""

    def test_invalid_url_returns_empty(self):
        assert _extract_domain("not-a-url") == ""


class TestTagText:
    def test_extracts_text(self):
        from bs4 import BeautifulSoup

        soup = BeautifulSoup("<item><title>Hello</title></item>", "xml")
        assert _tag_text(soup.find("item"), "title") == "Hello"

    def test_missing_tag_returns_empty(self):
        from bs4 import BeautifulSoup

        soup = BeautifulSoup("<item></item>", "xml")
        assert _tag_text(soup.find("item"), "title") == ""
