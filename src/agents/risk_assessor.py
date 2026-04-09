"""RiskAssessorAgent — evaluates investment risks using Claude."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import anthropic

from ..models.schemas import RiskAssessment, RiskFactor

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")


class RiskAssessorAgent:
    """Identifies and scores investment risk factors using Claude."""

    def __init__(self, model: str | None = None) -> None:
        self._model = model or _DEFAULT_MODEL
        self._client = anthropic.AsyncAnthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )

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
            '{"risk_factors": [{"name": "...", "description": "...", "severity": "LOW|MEDIUM|HIGH"}], '
            '"overall_risk_score": 0.0, "summary": "..."}'
        )

        message = await self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text
        return _parse_risk_response(raw, ticker)


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
                name=f.get("name", "Unknown"),
                description=f.get("description", ""),
                severity=f.get("severity", "MEDIUM"),
            )
            for f in data.get("risk_factors", [])
        ]
        return RiskAssessment(
            ticker=ticker,
            risk_factors=factors,
            overall_risk_score=float(data.get("overall_risk_score", 0.5)),
            summary=data.get("summary", ""),
        )
    except (json.JSONDecodeError, KeyError, ValueError):
        logger.warning("Failed to parse risk response for %s", ticker)
        return RiskAssessment(
            ticker=ticker,
            risk_factors=[],
            overall_risk_score=0.5,
            summary="Risk assessment unavailable.",
        )
