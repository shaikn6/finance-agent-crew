"""FundamentalAnalystAgent — computes financial ratios and Claude interpretation."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

import anthropic

from ..models.schemas import FundamentalAnalysis, Signal

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")


class FundamentalAnalystAgent:
    """Calculates key financial ratios and uses Claude to generate a signal.

    Ratios computed:
    - P/E, P/B, P/S (from market overview)
    - ROE, gross/operating/net margins
    - Revenue CAGR (3-year from SEC history)
    - Debt-to-equity, current ratio
    """

    def __init__(self, model: str | None = None) -> None:
        self._model = model or _DEFAULT_MODEL
        self._client = anthropic.AsyncAnthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )

    async def analyze(self, raw_data: dict[str, Any]) -> FundamentalAnalysis:
        """Run fundamental analysis on gathered raw data.

        Args:
            raw_data: Output dict from DataGathererAgent.gather().

        Returns:
            FundamentalAnalysis with computed ratios, signal, and reasoning.
        """
        ticker = raw_data.get("ticker", "UNKNOWN")
        overview = raw_data.get("market_overview", {})
        sec = raw_data.get("sec_summary", {}).get("financials", {})
        quote = raw_data.get("current_quote", {})
        rev_history = raw_data.get("sec_summary", {}).get("revenue_history", [])

        # ------------------------------------------------------------------
        # Compute ratios
        # ------------------------------------------------------------------
        pe_ratio = overview.get("pe_ratio")
        pb_ratio = overview.get("pb_ratio")
        roe = _pct(overview.get("return_on_equity_ttm"))
        gross_margin = _margin(sec.get("gross_profit"), sec.get("revenue"))
        operating_margin = _margin(sec.get("operating_income"), sec.get("revenue"))
        net_margin = _margin(sec.get("net_income"), sec.get("revenue"))
        debt_to_equity = _ratio(sec.get("total_debt"), sec.get("stockholders_equity"))
        revenue_cagr = _cagr(rev_history)

        # Current ratio requires current assets/liabilities — use proxy from SEC
        # (not always available; omit if missing)
        current_ratio: float | None = None

        ratios: dict[str, Any] = {
            "pe_ratio": pe_ratio,
            "pb_ratio": pb_ratio,
            "roe": roe,
            "gross_margin": gross_margin,
            "operating_margin": operating_margin,
            "net_margin": net_margin,
            "debt_to_equity": debt_to_equity,
            "revenue_cagr_3y": revenue_cagr,
            "current_ratio": current_ratio,
            "beta": overview.get("beta"),
            "quarterly_revenue_growth": overview.get("quarterly_revenue_growth"),
            "quarterly_earnings_growth": overview.get("quarterly_earnings_growth"),
        }

        # ------------------------------------------------------------------
        # Claude interpretation
        # ------------------------------------------------------------------
        signal, reasoning = await self._interpret(ticker, ratios, overview, sec)

        return FundamentalAnalysis(
            ticker=ticker,
            pe_ratio=pe_ratio,
            pb_ratio=pb_ratio,
            roe=roe,
            revenue_cagr_3y=revenue_cagr,
            gross_margin=gross_margin,
            operating_margin=operating_margin,
            net_margin=net_margin,
            debt_to_equity=debt_to_equity,
            current_ratio=current_ratio,
            signal=signal,
            reasoning=reasoning,
            data_as_of=datetime.now(tz=timezone.utc),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _interpret(
        self,
        ticker: str,
        ratios: dict[str, Any],
        overview: dict[str, Any],
        sec: dict[str, Any],
    ) -> tuple[Signal, str]:
        """Ask Claude to interpret the ratios and return a signal."""
        prompt = _build_fundamental_prompt(ticker, ratios, overview, sec)
        try:
            message = await self._client.messages.create(
                model=self._model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = message.content[0].text.strip()
            return _parse_signal_response(raw)
        except Exception as exc:
            logger.error("FundamentalAnalyst Claude call failed: %s", exc)
            return Signal.INSUFFICIENT_DATA, f"Analysis unavailable: {exc}"


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


def _build_fundamental_prompt(
    ticker: str,
    ratios: dict[str, Any],
    overview: dict[str, Any],
    sec: dict[str, Any],
) -> str:
    ratios_str = json.dumps(
        {k: v for k, v in ratios.items() if v is not None}, indent=2
    )
    sector = overview.get("sector", "Unknown")
    industry = overview.get("industry", "Unknown")
    market_cap = overview.get("market_cap")
    market_cap_str = f"${market_cap / 1e9:.1f}B" if market_cap else "Unknown"

    return f"""You are a CFA-level equity analyst specializing in fundamental analysis.

Analyze the following financial data for {ticker} ({sector} / {industry}, Market Cap: {market_cap_str}).

## Key Financial Ratios
{ratios_str}

## Instructions
1. Assess valuation (P/E vs sector norms, P/B, P/S).
2. Assess profitability (margins, ROE, CAGR trend).
3. Assess financial health (debt levels, leverage).
4. Provide an overall investment signal: BUY, HOLD, or SELL.
5. If data is too sparse to form a confident view, use INSUFFICIENT_DATA.

Respond in this exact JSON format:
{{
  "signal": "BUY" | "HOLD" | "SELL" | "INSUFFICIENT_DATA",
  "reasoning": "2-4 sentence explanation covering valuation, profitability, and risk."
}}

Only output valid JSON, nothing else."""


def _parse_signal_response(raw: str) -> tuple[Signal, str]:
    try:
        # Strip markdown code fences if present
        clean = (
            raw.strip()
            .removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        data = json.loads(clean)
        signal_str = data.get("signal", "INSUFFICIENT_DATA").upper()
        signal = (
            Signal(signal_str)
            if signal_str in Signal.__members__
            else Signal.INSUFFICIENT_DATA
        )
        reasoning = data.get("reasoning", "")
        return signal, reasoning
    except Exception as exc:
        logger.warning("Failed to parse fundamental signal response: %s", exc)
        return Signal.INSUFFICIENT_DATA, raw[:500]


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------


def _pct(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value * 100, 2) if abs(value) < 10 else round(value, 2)


def _margin(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return round((numerator / denominator) * 100, 2)


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return round(numerator / denominator, 3)


def _cagr(revenue_history: list[dict[str, Any]], years: int = 3) -> float | None:
    """Calculate revenue CAGR from SEC history (most recent first)."""
    if len(revenue_history) < 2:
        return None
    sorted_history = sorted(revenue_history, key=lambda x: x.get("period_end", ""))
    if len(sorted_history) < 2:
        return None
    # Use the last N+1 entries to get N years of growth
    n = min(years, len(sorted_history) - 1)
    latest = sorted_history[-1].get("value")
    oldest = sorted_history[-(n + 1)].get("value")
    if not latest or not oldest or oldest <= 0:
        return None
    cagr = ((latest / oldest) ** (1 / n) - 1) * 100
    return round(cagr, 2)
