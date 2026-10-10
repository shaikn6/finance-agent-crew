"""Minimal FastAPI app exposing the research pipeline over HTTP."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException

from .. import __version__
from ..agents.data_gatherer import DataGathererAgent
from ..agents.fundamental_analyst import FundamentalAnalystAgent
from ..agents.report_writer import ReportWriterAgent
from ..agents.risk_assessor import RiskAssessorAgent
from ..agents.sentiment_analyst import SentimentAnalystAgent
from ..config import get_settings
from ..models.schemas import JobStatusEnum, ResearchRequest, ResearchResponse
from .security import RateLimiter, authorize

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Validate configuration once at startup and set up per-process state."""
    settings = get_settings()
    settings.validate_for_serving()
    app.state.settings = settings
    app.state.rate_limiter = RateLimiter(settings.rate_limit_per_minute)
    if settings.offline_mode:
        logger.warning("OFFLINE_MODE is on: serving canned demo data, not live analysis")
    yield


app = FastAPI(title="Finance Agent Crew", version=__version__, lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/research",
    response_model=ResearchResponse,
    dependencies=[Depends(authorize)],
)
async def research(request: ResearchRequest) -> ResearchResponse:
    """Run the full fan-out/fan-in research pipeline for a ticker."""
    started = time.monotonic()
    created_at = datetime.now(tz=UTC)
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
    except Exception as exc:
        # The cause is logged here; the client only ever sees a generic message.
        logger.exception("Research pipeline failed for %s", ticker)
        raise HTTPException(
            status_code=502, detail="Upstream research pipeline failed."
        ) from exc

    return ResearchResponse(
        job_id=str(uuid4()),
        ticker=ticker,
        status=JobStatusEnum.COMPLETED,
        report=report,
        created_at=created_at,
        completed_at=datetime.now(tz=UTC),
        duration_seconds=round(time.monotonic() - started, 3),
    )
