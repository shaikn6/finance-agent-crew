"""Tests for src/models/schemas.py — Pydantic models and validators."""
from __future__ import annotations

import pytest
from datetime import datetime, timezone
from pydantic import ValidationError

from src.models.schemas import (
    Signal,
    Tone,
    JobStatusEnum,
    FundamentalAnalysis,
    SentimentAnalysis,
    RiskFactor,
    RiskAssessment,
    InvestmentReport,
    ResearchRequest,
    ResearchResponse,
    JobStatus,
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TestSignal:
    def test_values(self):
        assert Signal.BUY == "BUY"
        assert Signal.HOLD == "HOLD"
        assert Signal.SELL == "SELL"
        assert Signal.INSUFFICIENT_DATA == "INSUFFICIENT_DATA"

    def test_membership(self):
        assert "BUY" in Signal.__members__
        assert "INSUFFICIENT_DATA" in Signal.__members__

    def test_from_string(self):
        assert Signal("BUY") == Signal.BUY
        assert Signal("SELL") == Signal.SELL

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            Signal("INVALID")


class TestTone:
    def test_values(self):
        assert Tone.POSITIVE == "positive"
        assert Tone.NEUTRAL == "neutral"
        assert Tone.NEGATIVE == "negative"

    def test_from_string(self):
        assert Tone("positive") == Tone.POSITIVE


class TestJobStatusEnum:
    def test_all_values(self):
        assert JobStatusEnum.PENDING == "pending"
        assert JobStatusEnum.RUNNING == "running"
        assert JobStatusEnum.COMPLETED == "completed"
        assert JobStatusEnum.FAILED == "failed"


# ---------------------------------------------------------------------------
# FundamentalAnalysis
# ---------------------------------------------------------------------------

class TestFundamentalAnalysis:
    def test_defaults(self):
        fa = FundamentalAnalysis(ticker="AAPL")
        assert fa.ticker == "AAPL"
        assert fa.pe_ratio is None
        assert fa.pb_ratio is None
        assert fa.signal == Signal.INSUFFICIENT_DATA
        assert fa.reasoning == ""

    def test_full_construction(self):
        fa = FundamentalAnalysis(
            ticker="MSFT",
            pe_ratio=30.5,
            pb_ratio=12.0,
            roe=45.2,
            revenue_cagr_3y=8.5,
            gross_margin=70.0,
            operating_margin=42.0,
            net_margin=36.0,
            debt_to_equity=0.5,
            current_ratio=2.1,
            signal=Signal.BUY,
            reasoning="Strong fundamentals.",
            data_as_of=datetime.now(tz=timezone.utc),
        )
        assert fa.pe_ratio == 30.5
        assert fa.signal == Signal.BUY
        assert fa.gross_margin == 70.0

    def test_optional_fields_none(self):
        fa = FundamentalAnalysis(ticker="XYZ", signal=Signal.SELL)
        assert fa.pe_ratio is None
        assert fa.current_ratio is None

    def test_signal_enum_coercion(self):
        fa = FundamentalAnalysis(ticker="T", signal="HOLD")
        assert fa.signal == Signal.HOLD


# ---------------------------------------------------------------------------
# SentimentAnalysis
# ---------------------------------------------------------------------------

class TestSentimentAnalysis:
    def test_defaults(self):
        sa = SentimentAnalysis(ticker="AAPL")
        assert sa.management_tone == Tone.NEUTRAL
        assert sa.news_sentiment_score == 0.0
        assert sa.key_themes == []
        assert sa.headline_count == 0

    def test_score_bounds(self):
        # Valid boundaries
        sa = SentimentAnalysis(ticker="T", news_sentiment_score=1.0)
        assert sa.news_sentiment_score == 1.0
        sa2 = SentimentAnalysis(ticker="T", news_sentiment_score=-1.0)
        assert sa2.news_sentiment_score == -1.0

    def test_score_out_of_range_raises(self):
        with pytest.raises(ValidationError):
            SentimentAnalysis(ticker="T", news_sentiment_score=1.1)
        with pytest.raises(ValidationError):
            SentimentAnalysis(ticker="T", news_sentiment_score=-1.1)

    def test_full_construction(self):
        sa = SentimentAnalysis(
            ticker="GOOG",
            management_tone=Tone.POSITIVE,
            news_sentiment_score=0.75,
            key_themes=["AI", "Cloud"],
            headline_count=15,
            bullish_signals=["Revenue beat"],
            bearish_signals=["Margin pressure"],
            reasoning="Overall positive.",
            analyzed_at=datetime.now(tz=timezone.utc),
        )
        assert sa.management_tone == Tone.POSITIVE
        assert len(sa.key_themes) == 2
        assert sa.headline_count == 15


# ---------------------------------------------------------------------------
# RiskFactor and RiskAssessment
# ---------------------------------------------------------------------------

class TestRiskFactor:
    def test_construction(self):
        rf = RiskFactor(
            category="market",
            description="Broad market downturn risk",
            severity="high",
            probability="medium",
        )
        assert rf.category == "market"
        assert rf.severity == "high"


class TestRiskAssessment:
    def test_defaults(self):
        ra = RiskAssessment(ticker="META")
        assert ra.overall_risk_level == "medium"
        assert ra.risk_factors == []
        assert ra.beta is None

    def test_with_risk_factors(self):
        ra = RiskAssessment(
            ticker="META",
            overall_risk_level="high",
            risk_factors=[
                RiskFactor(
                    category="regulatory",
                    description="Antitrust investigations",
                    severity="high",
                    probability="medium",
                )
            ],
            beta=1.3,
            volatility_30d=28.5,
            max_drawdown_1y=-45.0,
            key_risks_summary="Regulatory overhang is significant.",
            mitigating_factors=["Strong cash position"],
        )
        assert len(ra.risk_factors) == 1
        assert ra.beta == 1.3
        assert ra.max_drawdown_1y == -45.0


# ---------------------------------------------------------------------------
# InvestmentReport
# ---------------------------------------------------------------------------

class TestInvestmentReport:
    def test_minimal(self):
        report = InvestmentReport(ticker="AAPL")
        assert report.ticker == "AAPL"
        assert report.overall_signal == Signal.INSUFFICIENT_DATA
        assert report.disclaimer != ""

    def test_upside_computed_from_prices(self):
        report = InvestmentReport(
            ticker="NVDA",
            current_price=100.0,
            target_price=150.0,
        )
        assert report.upside_potential == 50.0

    def test_upside_none_when_prices_missing(self):
        report = InvestmentReport(ticker="X")
        assert report.upside_potential is None

    def test_upside_none_when_current_zero(self):
        report = InvestmentReport(ticker="X", current_price=0.0, target_price=50.0)
        assert report.upside_potential is None

    def test_full_report(self):
        fa = FundamentalAnalysis(ticker="AAPL", signal=Signal.BUY, reasoning="Strong")
        sa = SentimentAnalysis(ticker="AAPL", news_sentiment_score=0.5)
        ra = RiskAssessment(ticker="AAPL")
        report = InvestmentReport(
            ticker="AAPL",
            company_name="Apple Inc.",
            sector="Technology",
            industry="Consumer Electronics",
            fundamental_analysis=fa,
            sentiment_analysis=sa,
            risk_assessment=ra,
            overall_signal=Signal.BUY,
            current_price=180.0,
            target_price=220.0,
            executive_summary="Apple remains a compelling hold.",
            investment_thesis="Strong ecosystem lock-in.",
            bull_case="Services growth accelerates.",
            bear_case="Slowing iPhone replacement cycles.",
            key_catalysts=["Vision Pro launch", "India expansion"],
            key_risks=["China sales risk"],
        )
        assert report.company_name == "Apple Inc."
        assert report.upside_potential == pytest.approx(22.22, abs=0.01)
        assert len(report.key_catalysts) == 2

    def test_upside_passthrough_when_provided(self):
        """Validator should not override explicitly provided upside_potential."""
        report = InvestmentReport(
            ticker="AAPL",
            current_price=100.0,
            target_price=150.0,
            upside_potential=99.9,
        )
        assert report.upside_potential == 99.9


# ---------------------------------------------------------------------------
# ResearchRequest
# ---------------------------------------------------------------------------

class TestResearchRequest:
    def test_ticker_uppercased(self):
        req = ResearchRequest(ticker="aapl")
        assert req.ticker == "AAPL"

    def test_ticker_stripped_and_uppercased(self):
        req = ResearchRequest(ticker="  msft  ")
        assert req.ticker == "MSFT"

    def test_defaults(self):
        req = ResearchRequest(ticker="GOOG")
        assert req.include_news is True
        assert req.include_sec_filings is True
        assert req.depth == "standard"

    def test_empty_ticker_raises(self):
        with pytest.raises(ValidationError):
            ResearchRequest(ticker="")

    def test_ticker_too_long_raises(self):
        with pytest.raises(ValidationError):
            ResearchRequest(ticker="A" * 11)

    def test_invalid_depth_raises(self):
        with pytest.raises(ValidationError):
            ResearchRequest(ticker="AAPL", depth="ultra")

    def test_valid_depths(self):
        for depth in ("quick", "standard", "deep"):
            req = ResearchRequest(ticker="T", depth=depth)
            assert req.depth == depth


# ---------------------------------------------------------------------------
# ResearchResponse
# ---------------------------------------------------------------------------

class TestResearchResponse:
    def test_defaults(self):
        resp = ResearchResponse(job_id="abc123", ticker="AAPL")
        assert resp.status == JobStatusEnum.PENDING
        assert resp.report is None
        assert resp.error is None

    def test_with_report(self):
        report = InvestmentReport(ticker="AAPL")
        resp = ResearchResponse(
            job_id="j1",
            ticker="AAPL",
            status=JobStatusEnum.COMPLETED,
            report=report,
            duration_seconds=12.5,
        )
        assert resp.status == JobStatusEnum.COMPLETED
        assert resp.report is not None

    def test_with_error(self):
        resp = ResearchResponse(
            job_id="j2",
            ticker="FAIL",
            status=JobStatusEnum.FAILED,
            error="Timeout exceeded",
        )
        assert resp.error == "Timeout exceeded"


# ---------------------------------------------------------------------------
# JobStatus
# ---------------------------------------------------------------------------

class TestJobStatus:
    def test_construction(self):
        now = datetime.now(tz=timezone.utc)
        js = JobStatus(
            job_id="j1",
            status=JobStatusEnum.RUNNING,
            ticker="AAPL",
            progress=50,
            current_step="fundamental_analysis",
            created_at=now,
        )
        assert js.progress == 50
        assert js.current_step == "fundamental_analysis"

    def test_progress_bounds(self):
        now = datetime.now(tz=timezone.utc)
        with pytest.raises(ValidationError):
            JobStatus(job_id="j", status=JobStatusEnum.PENDING, ticker="T",
                      progress=-1, created_at=now)
        with pytest.raises(ValidationError):
            JobStatus(job_id="j", status=JobStatusEnum.PENDING, ticker="T",
                      progress=101, created_at=now)

    def test_max_progress(self):
        now = datetime.now(tz=timezone.utc)
        js = JobStatus(
            job_id="j", status=JobStatusEnum.COMPLETED, ticker="T",
            progress=100, created_at=now
        )
        assert js.progress == 100
