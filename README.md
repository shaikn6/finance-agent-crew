<p align="center"><img src=".github/banner.png" alt="finance-agent-crew" width="100%"></p>

<div align="center">

# Finance Agent Crew

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://python.org)
[![Pydantic](https://img.shields.io/badge/Pydantic-v2-e92063)](https://docs.pydantic.dev/)
[![Anthropic](https://img.shields.io/badge/Claude-Haiku-d97757)](https://docs.anthropic.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker)](docker-compose.yml)

**A multi-agent crew that turns a stock ticker into a structured, sourced equity research brief — fan-out data collection (SEC EDGAR, Alpha Vantage, news), three specialist analysts running in parallel, and a synthesizing report writer that emits a BUY / HOLD / SELL signal.**

</div>

## Architecture

The crew is a deterministic fan-out / fan-in pipeline. `DataGathererAgent` is intentionally LLM-free and I/O-bound: it runs the three data tools concurrently with `asyncio.gather` and returns one structured bundle. That bundle feeds three independent analyst agents (each makes a single structured Claude call), and `ReportWriterAgent` fans the three analyses back in to a final `InvestmentReport`. Every external call is non-fatal — failures are captured in an `errors` list rather than aborting the run.

```mermaid
flowchart TD
    Q[Ticker e.g. AAPL] --> DG[DataGathererAgent<br/>no LLM · asyncio.gather]

    subgraph Tools[Data tools — async context managers]
        SEC[SECEdgarScraper<br/>XBRL facts · 10-K / 10-Q]
        MKT[MarketDataClient<br/>Alpha Vantage overview / quote / OHLCV]
        NEWS[NewsFetcher<br/>headlines · earnings snippets]
    end

    DG --> SEC
    DG --> MKT
    DG --> NEWS
    SEC --> BUNDLE[(Raw data bundle<br/>dict + errors[])]
    MKT --> BUNDLE
    NEWS --> BUNDLE

    BUNDLE --> FA[FundamentalAnalystAgent<br/>ratios + Claude signal]
    BUNDLE --> SA[SentimentAnalystAgent<br/>news + tone scoring]
    BUNDLE --> RA[RiskAssessorAgent<br/>risk factors + score]

    FA --> RW[ReportWriterAgent<br/>synthesis]
    SA --> RW
    RA --> RW
    RW --> REP[InvestmentReport<br/>BUY / HOLD / SELL · thesis · bull/bear case]
```

## Agents

| Agent | Role | Inputs / method |
|-------|------|-----------------|
| `DataGathererAgent` | Concurrent data collection, no LLM | Runs SEC / market / news fetches via `asyncio.gather`; non-fatal error collection |
| `FundamentalAnalystAgent` | Valuation, profitability, leverage | Computes P/E, P/B, ROE, gross/operating/net margins, debt-to-equity, **3-yr revenue CAGR derived from real period-end date spans**, then asks Claude for a `Signal` |
| `SentimentAnalystAgent` | News + management tone | Aggregates ≤15 headlines and earnings snippets into one Claude call → tone, `[-1, +1]` sentiment score, themes, bullish/bearish signals |
| `RiskAssessorAgent` | Risk factor identification | Claude scores named risk factors (LOW/MEDIUM/HIGH) + overall `risk_score` |
| `ReportWriterAgent` | Final synthesis | Fans the three analyses into an `InvestmentReport` with executive summary, thesis, and bull/bear cases |

## Data tools

| Tool | Source | Notes |
|------|--------|-------|
| `SECEdgarScraper` | SEC EDGAR | XBRL financial facts + recent 10-K / 10-Q filings; async context manager |
| `MarketDataClient` | Alpha Vantage | Company overview, real-time quote, 12-month monthly OHLCV |
| `NewsFetcher` | News / RSS | Recent headlines + earnings-call snippets |

## How it works

- **Structured-output discipline.** Every analyst prompts Claude for strict JSON and parses it through a code-fence-tolerant decoder. A parse failure degrades to `INSUFFICIENT_DATA` / neutral defaults rather than crashing — no analyst can take down a run.
- **Real CAGR, not list arithmetic.** `_cagr()` sorts SEC revenue history by `period_end`, then derives the exponent from the *actual* elapsed days between the oldest and newest filing (`(latest/oldest) ** (1/actual_years) - 1`), so the growth rate reflects real time, not the number of rows.
- **Typed boundaries.** All agent outputs are Pydantic v2 models (`FundamentalAnalysis`, `SentimentAnalysis`, `RiskAssessment`, `InvestmentReport`) with enums for `Signal` / `Tone`, so downstream consumers get validated, serializable objects.
- **Cost control.** Defaults to `claude-3-5-haiku` (override via `ANTHROPIC_MODEL`) and one bounded (`max_tokens=1024`) call per analyst.

## Tests

The schema and agent contracts are covered by **231 tests** (`pytest tests/ --co -q`) across the five agents, three tools, and the Pydantic schemas — exercising ratio math, JSON-parse fallbacks, and error-collection paths.

## Quickstart

```bash
git clone https://github.com/shaikn6/finance-agent-crew
cd finance-agent-crew
cp .env.example .env          # set ANTHROPIC_API_KEY, ALPHA_VANTAGE_KEY, NEWS_API_KEY
pip install -e ".[dev]"
```

Run the crew directly in Python (the agents are the public surface):

```python
import asyncio
from src.agents.data_gatherer import DataGathererAgent
from src.agents.fundamental_analyst import FundamentalAnalystAgent
from src.agents.sentiment_analyst import SentimentAnalystAgent
from src.agents.risk_assessor import RiskAssessorAgent
from src.agents.report_writer import ReportWriterAgent

async def research(ticker: str):
    raw = await DataGathererAgent().gather(ticker)
    fundamental, sentiment, risk = await asyncio.gather(
        FundamentalAnalystAgent().analyze(raw),
        SentimentAnalystAgent().analyze(raw),
        RiskAssessorAgent().assess(raw),
    )
    return await ReportWriterAgent().write(
        ticker, fundamental=fundamental, sentiment=sentiment, risk=risk, raw_data=raw
    )

report = asyncio.run(research("AAPL"))
print(report.overall_signal, report.executive_summary)
```

Tests and lint:

```bash
pytest tests/ -v --cov=src
ruff check src/ tests/
```

## Tech stack

Python 3.11 · `anthropic` (Claude Haiku) · `langgraph` / `langchain-anthropic` · `pydantic` v2 + `pydantic-settings` · `httpx` (async clients) · `beautifulsoup4` (EDGAR parsing) · `fastapi` / `uvicorn` · Docker. Tested with `pytest` + `pytest-asyncio`, linted with `ruff`.

## License

MIT
