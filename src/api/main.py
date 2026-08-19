"""Minimal FastAPI app exposing the research pipeline over HTTP."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from uuid import uuid4

from fastapi import FastAPI, HTTPException

from ..agents.data_gatherer import DataGathererAgent
from ..agents.fundamental_analyst import FundamentalAnalystAgent
from ..agents.report_writer import ReportWriterAgent
from ..agents.risk_assessor import RiskAssessorAgent
from ..agents.sentiment_analyst import SentimentAnalystAgent
from ..models.schemas import JobStatusEnum, ResearchRequest, ResearchResponse

app = FastAPI(title="Finance Agent Crew", version="0.1.0")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/research", response_model=ResearchResponse)
async def research(request: ResearchRequest) -> ResearchResponse:
    """Run the full fan-out/fan-in research pipeline for a ticker."""
    started = time.monotonic()
    created_at = datetime.utcnow()
    ticker = request.ticker

    try:
        gatherer = DataGathererAgent(
            include_news=request.include_news,
            include_sec=request.include_sec_filings,
        )
        raw_data = await gatherer.gather(ticker)

        fundamental, sentiment, risk = await asyncio.gather(
            FundamentalAnalystAgent().analyze(raw_data),
            SentimentAnalystAgent().analyze(raw_data),
            RiskAssessorAgent().assess(raw_data),
        )

        report = await ReportWriterAgent().write(
            ticker,
            fundamental=fundamental,
            sentiment=sentiment,
            risk=risk,
            raw_data=raw_data,
        )
    except Exception as exc:  # pragma: no cover - defensive top-level guard
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return ResearchResponse(
        job_id=str(uuid4()),
        ticker=ticker,
        status=JobStatusEnum.COMPLETED,
        report=report,
        created_at=created_at,
        completed_at=datetime.utcnow(),
        duration_seconds=round(time.monotonic() - started, 3),
    )
