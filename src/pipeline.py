"""The multi-agent research pipeline, shared by the HTTP API and the CLI."""

from __future__ import annotations

import asyncio

from .agents.data_gatherer import DataGathererAgent
from .agents.fundamental_analyst import FundamentalAnalystAgent
from .agents.report_writer import ReportWriterAgent
from .agents.risk_assessor import RiskAssessorAgent
from .agents.sentiment_analyst import SentimentAnalystAgent
from .models.schemas import InvestmentReport


async def run_research(
    ticker: str,
    *,
    include_news: bool = True,
    include_sec: bool = True,
) -> InvestmentReport:
    """Run the fan-out/fan-in pipeline for one ticker and return the report.

    Gathers raw data once, runs the three analysts concurrently, then
    synthesises their output into a single :class:`InvestmentReport`.
    """
    raw_data = await DataGathererAgent(
        include_news=include_news, include_sec=include_sec
    ).gather(ticker)

    fundamental, sentiment, risk = await asyncio.gather(
        FundamentalAnalystAgent().analyze(raw_data),
        SentimentAnalystAgent().analyze(raw_data),
        RiskAssessorAgent().assess(raw_data),
    )

    return await ReportWriterAgent().write(
        ticker,
        fundamental=fundamental,
        sentiment=sentiment,
        risk=risk,
        raw_data=raw_data,
    )
