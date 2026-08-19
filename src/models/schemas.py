"""Pydantic v2 schemas for Finance Agent Crew."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class Signal(str, Enum):
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class Tone(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class JobStatusEnum(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Sub-analysis models
# ---------------------------------------------------------------------------


class FundamentalAnalysis(BaseModel):
    """Output of the FundamentalAnalystAgent."""

    ticker: str
    pe_ratio: float | None = Field(None, description="Price-to-Earnings ratio")
    pb_ratio: float | None = Field(None, description="Price-to-Book ratio")
    roe: float | None = Field(None, description="Return on Equity (%)")
    revenue_cagr_3y: float | None = Field(None, description="3-year revenue CAGR (%)")
    gross_margin: float | None = Field(None, description="Gross margin (%)")
    operating_margin: float | None = Field(None, description="Operating margin (%)")
    net_margin: float | None = Field(None, description="Net margin (%)")
    debt_to_equity: float | None = Field(None, description="Debt-to-Equity ratio")
    current_ratio: float | None = Field(None, description="Current ratio")
    signal: Signal = Signal.INSUFFICIENT_DATA
    reasoning: str = Field(default="", description="Claude's interpretation")
    data_as_of: datetime | None = None


class SentimentAnalysis(BaseModel):
    """Output of the SentimentAnalystAgent."""

    ticker: str
    management_tone: Tone = Tone.NEUTRAL
    news_sentiment_score: float = Field(
        0.0,
        ge=-1.0,
        le=1.0,
        description="Aggregate news sentiment (-1 bearish → +1 bullish)",
    )
    key_themes: list[str] = Field(default_factory=list)
    headline_count: int = 0
    bullish_signals: list[str] = Field(default_factory=list)
    bearish_signals: list[str] = Field(default_factory=list)
    reasoning: str = ""
    analyzed_at: datetime | None = None


class RiskFactor(BaseModel):
    category: str  # market | regulatory | competitive | operational | macro
    description: str
    severity: str  # low | medium | high | critical
    probability: str  # low | medium | high


_RISK_LEVEL_SCORES: dict[str, float] = {
    "low": 0.25,
    "medium": 0.5,
    "high": 0.75,
    "critical": 1.0,
}


class RiskAssessment(BaseModel):
    """Output of the RiskAssessorAgent."""

    ticker: str
    overall_risk_level: str = Field("medium", description="low | medium | high | critical")
    risk_factors: list[RiskFactor] = Field(default_factory=list)
    beta: float | None = None
    volatility_30d: float | None = None
    max_drawdown_1y: float | None = None
    key_risks_summary: str = ""
    mitigating_factors: list[str] = Field(default_factory=list)
    assessed_at: datetime | None = None

    @property
    def overall_risk_score(self) -> float:
        """Numeric risk score in [0, 1] derived from ``overall_risk_level``."""
        return _RISK_LEVEL_SCORES.get(self.overall_risk_level.lower(), 0.5)


class InvestmentReport(BaseModel):
    """Final synthesized investment research brief."""

    ticker: str
    company_name: str = ""
    sector: str = ""
    industry: str = ""
    report_date: datetime = Field(default_factory=datetime.utcnow)

    # Composed analyses
    fundamental_analysis: FundamentalAnalysis | None = None
    sentiment_analysis: SentimentAnalysis | None = None
    risk_assessment: RiskAssessment | None = None

    # Synthesized output
    overall_signal: Signal = Signal.INSUFFICIENT_DATA
    target_price: float | None = None
    current_price: float | None = None
    upside_potential: float | None = Field(None, description="% upside to target price")

    executive_summary: str = ""
    investment_thesis: str = ""
    bull_case: str = ""
    bear_case: str = ""
    key_catalysts: list[str] = Field(default_factory=list)
    key_risks: list[str] = Field(default_factory=list)

    disclaimer: str = (
        "This report is for informational purposes only and does not constitute "
        "investment advice. Past performance is not indicative of future results. "
        "Always consult a qualified financial advisor before making investment decisions."
    )

    @model_validator(mode="after")
    def compute_upside(self) -> InvestmentReport:
        if self.upside_potential is None:
            current = self.current_price
            target = self.target_price
            if current and target and current > 0:
                self.upside_potential = round(((target - current) / current) * 100, 2)
        return self


# ---------------------------------------------------------------------------
# API Request / Response
# ---------------------------------------------------------------------------


class ResearchRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=10, examples=["AAPL", "MSFT"])
    include_news: bool = True
    include_sec_filings: bool = True
    depth: str = Field("standard", description="quick | standard | deep")

    @field_validator("ticker")
    @classmethod
    def uppercase_ticker(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("depth")
    @classmethod
    def validate_depth(cls, v: str) -> str:
        allowed = {"quick", "standard", "deep"}
        if v not in allowed:
            raise ValueError(f"depth must be one of {allowed}")
        return v


class ResearchResponse(BaseModel):
    job_id: str
    ticker: str
    status: JobStatusEnum = JobStatusEnum.PENDING
    report: InvestmentReport | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
    duration_seconds: float | None = None


class JobStatus(BaseModel):
    job_id: str
    status: JobStatusEnum
    ticker: str
    progress: int = Field(0, ge=0, le=100, description="Completion percentage")
    current_step: str = ""
    created_at: datetime
    updated_at: datetime = Field(default_factory=datetime.utcnow)
