# Changelog

All notable changes to this project are documented here.

## [1.0.0] - 2026-06-16

### Added
- Deterministic asyncio fan-out/fan-in pipeline: `DataGathererAgent` runs data tools concurrently, three independent analyst agents each make a single structured Claude call, and `ReportWriterAgent` synthesizes the results into a final `InvestmentReport`
- SEC EDGAR scraper fetching recent filings via the Atom feed
- Yahoo Finance news fetcher parsing RSS headlines and descriptions
- Market data tool for fundamental metrics (P/E, margins, growth, leverage ratios)
- `FundamentalAnalystAgent`, `SentimentAnalystAgent`, and `RiskAssessorAgent` producing structured, schema-validated analyses via Claude
- Non-fatal error handling: failed external calls are captured in an `errors` list rather than aborting the run

### Changed
- Production-ready CI/CD with 95%+ test coverage enforcement

### Security
- No customer portfolio data or trade instructions are processed by the agent; all inputs are public market data, SEC filings, and news
