"""ReportWriterAgent — synthesizes analyses into a final investment report."""

from __future__ import annotations

import json
import logging
from typing import Any

from ..config import get_settings
from ..llm import LLMError, complete, get_llm_client
from ..models.schemas import (
    FundamentalAnalysis,
    InvestmentReport,
    RiskAssessment,
    SentimentAnalysis,
    Signal,
)

logger = logging.getLogger(__name__)


class ReportWriterAgent:
    """Synthesizes fundamental, sentiment, and risk analyses into an investment brief."""

    def __init__(self, model: str | None = None) -> None:
        self._model = model or get_settings().anthropic_model
        self._client = get_llm_client()

    async def write(
        self,
        ticker: str,
        fundamental: FundamentalAnalysis | None = None,
        sentiment: SentimentAnalysis | None = None,
        risk: RiskAssessment | None = None,
        raw_data: dict[str, Any] | None = None,
    ) -> InvestmentReport:
        """Generate a final investment report by synthesizing all analyses.

        Args:
            ticker:      Equity ticker symbol.
            fundamental: Output from FundamentalAnalystAgent.
            sentiment:   Output from SentimentAnalystAgent.
            risk:        Output from RiskAssessorAgent.
            raw_data:    Original raw data dict (for company metadata).

        Returns:
            InvestmentReport containing the synthesized investment brief.
        """
        raw_data = raw_data or {}
        overview = raw_data.get("market_overview", {})
        company_name = (
            overview.get("name")
            or overview.get("company_name")
            or raw_data.get("company_name")
            or ticker
        )
        sector = overview.get("sector", "")
        industry = overview.get("industry", "")

        prompt_parts = [
            f"You are a senior equity analyst. Write a concise investment report for {ticker} ({company_name}).",
            f"Sector: {sector}, Industry: {industry}",
        ]
        if fundamental:
            prompt_parts.append(
                f"Fundamental signal: {fundamental.signal}, P/E: {fundamental.pe_ratio}"
            )
        if sentiment:
            prompt_parts.append(
                f"Sentiment: {sentiment.news_sentiment_score:.2f}, tone: {sentiment.management_tone}"
            )
        if risk:
            prompt_parts.append(f"Risk score: {risk.overall_risk_score:.2f}")

        prompt_parts.append(
            "\nRespond with JSON:\n"
            '{"overall_signal": "BUY|SELL|HOLD|INSUFFICIENT_DATA", '
            '"executive_summary": "...", "investment_thesis": "...", '
            '"bull_case": "...", "bear_case": "..."}'
        )

        try:
            raw = await complete(
                self._client, model=self._model, prompt="\n".join(prompt_parts)
            )
        except LLMError as exc:
            logger.error("ReportWriter LLM call failed for %s: %s", ticker, exc)
            return _failed_report(
                ticker, company_name, sector, industry, fundamental, sentiment, risk
            )
        return _parse_report_response(
            raw, ticker, company_name, sector, industry, fundamental, sentiment, risk
        )


def _failed_report(
    ticker: str,
    company_name: str,
    sector: str,
    industry: str,
    fundamental: FundamentalAnalysis | None,
    sentiment: SentimentAnalysis | None,
    risk: RiskAssessment | None,
) -> InvestmentReport:
    """Fallback report that keeps the analyses but carries no synthesis."""
    return InvestmentReport(
        ticker=ticker,
        company_name=company_name,
        sector=sector,
        industry=industry,
        fundamental_analysis=fundamental,
        sentiment_analysis=sentiment,
        risk_assessment=risk,
        overall_signal=Signal.INSUFFICIENT_DATA,
        executive_summary="Report generation failed.",
    )


def _parse_report_response(
    raw: str,
    ticker: str,
    company_name: str,
    sector: str,
    industry: str,
    fundamental: FundamentalAnalysis | None,
    sentiment: SentimentAnalysis | None,
    risk: RiskAssessment | None,
) -> InvestmentReport:
    """Parse Claude's JSON response into an InvestmentReport."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        data = json.loads(text)
        signal_str = data.get("overall_signal", "INSUFFICIENT_DATA").upper()
        try:
            overall_signal = Signal(signal_str)
        except ValueError:
            overall_signal = Signal.INSUFFICIENT_DATA

        return InvestmentReport(
            ticker=ticker,
            company_name=company_name,
            sector=sector,
            industry=industry,
            fundamental_analysis=fundamental,
            sentiment_analysis=sentiment,
            risk_assessment=risk,
            overall_signal=overall_signal,
            executive_summary=data.get("executive_summary", ""),
            investment_thesis=data.get("investment_thesis", ""),
            bull_case=data.get("bull_case", ""),
            bear_case=data.get("bear_case", ""),
        )
    except (ValueError, AttributeError, TypeError):
        # ValueError covers both JSONDecodeError and pydantic's ValidationError.
        logger.warning("Failed to parse report response for %s", ticker)
        return _failed_report(
            ticker, company_name, sector, industry, fundamental, sentiment, risk
        )
