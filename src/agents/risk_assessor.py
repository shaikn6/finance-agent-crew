"""RiskAssessorAgent — evaluates investment risks using Claude."""

from __future__ import annotations

import json
import logging
from typing import Any

from ..config import get_settings
from ..llm import LLMError, complete, get_llm_client
from ..models.schemas import RiskAssessment, RiskFactor

logger = logging.getLogger(__name__)


class RiskAssessorAgent:
    """Identifies and scores investment risk factors using Claude."""

    def __init__(self, model: str | None = None) -> None:
        self._model = model or get_settings().anthropic_model
        self._client = get_llm_client()

    async def assess(self, raw_data: dict[str, Any]) -> RiskAssessment:
        """Assess investment risks for a given ticker.

        Args:
            raw_data: Output dict from DataGathererAgent.gather().

        Returns:
            RiskAssessment with identified risk factors and an overall score.
        """
        ticker = raw_data.get("ticker", "UNKNOWN")
        overview = raw_data.get("market_overview", {})
        sec = raw_data.get("sec_summary", {}).get("financials", {})

        prompt = (
            f"You are a risk analyst. Evaluate the investment risks for {ticker}.\n"
            f"Market data: {json.dumps(overview, default=str)}\n"
            f"Financials: {json.dumps(sec, default=str)}\n\n"
            "Respond with a JSON object:\n"
            '{"risk_factors": [{"category": "market|regulatory|competitive|operational|macro", '
            '"description": "...", "severity": "low|medium|high|critical", "probability": "low|medium|high"}], '
            '"overall_risk_level": "low|medium|high|critical", "key_risks_summary": "..."}'
        )

        try:
            raw = await complete(self._client, model=self._model, prompt=prompt)
        except LLMError as exc:
            logger.error("RiskAssessor LLM call failed for %s: %s", ticker, exc)
            return _unavailable(ticker)
        return _parse_risk_response(raw, ticker)


def _unavailable(ticker: str) -> RiskAssessment:
    return RiskAssessment(
        ticker=ticker,
        risk_factors=[],
        overall_risk_level="medium",
        key_risks_summary="Risk assessment unavailable.",
    )


def _parse_risk_response(raw: str, ticker: str) -> RiskAssessment:
    """Parse Claude's JSON response into a RiskAssessment."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        data = json.loads(text)
        factors = [
            RiskFactor(
                category=f.get("category", "market"),
                description=f.get("description", ""),
                severity=f.get("severity", "medium"),
                probability=f.get("probability", "medium"),
            )
            for f in data.get("risk_factors", [])
        ]
        return RiskAssessment(
            ticker=ticker,
            risk_factors=factors,
            overall_risk_level=data.get("overall_risk_level", "medium"),
            key_risks_summary=data.get("key_risks_summary", ""),
        )
    except (ValueError, AttributeError, TypeError):
        # ValueError covers both JSONDecodeError and pydantic's ValidationError.
        logger.warning("Failed to parse risk response for %s", ticker)
        return _unavailable(ticker)
