# Changelog

All notable changes to this project are documented here.

## [1.0.0] - 2026-06-16

### Added
- LangGraph multi-agent system ingesting SEC filings, earnings call transcripts, and live financial news
- SEC EDGAR adapter fetching 10-K, 10-Q, and 8-K documents with structured section extraction
- Earnings call parser identifying guidance, risk factors, and management sentiment shifts
- Investment research brief generator producing structured buy/hold/sell analysis with evidence citations
- n8n webhook integration routing completed research briefs to Slack, email, and CRM destinations
- Portfolio-level aggregation view consolidating signals across multiple tickers into a single dashboard

### Changed
- Production-ready CI/CD with 95%+ test coverage enforcement

### Security
- Financial data cached locally; no customer portfolio data or trade instructions are processed by the agent
