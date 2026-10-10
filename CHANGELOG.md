# Changelog

All notable changes to this project are documented here.

## [Unreleased]

### Added
- Offline demo mode (`OFFLINE_MODE=1`): the LLM client and the SEC, market-data and news tools are replaced by deterministic fixtures, so the pipeline and `POST /research` run without keys or network access
- Settings loaded through `pydantic-settings` with `.env` support (`src/config.py`); `.env.example` documents every variable
- API-key authentication on `POST /research` (`X-API-Key` header, keys from `API_KEYS`); mandatory outside offline mode, and the server refuses to start in live mode without `ANTHROPIC_API_KEY` and `API_KEYS`
- In-process sliding-window rate limit on `POST /research` (`RATE_LIMIT_PER_MINUTE`, default 30), returning `429` with `Retry-After`
- `SEC_USER_AGENT_CONTACT` to set the contact address sent to SEC EDGAR and news sources
- Docker `HEALTHCHECK` against `/health`
- Tests for the HTTP API, settings, offline mode and LLM failure handling

### Changed
- Default model is now `claude-haiku-5-5`; it was a retired model id. Still overridable with `ANTHROPIC_MODEL`
- `ticker` must match `^[A-Z0-9][A-Z0-9.\-]*$` (max 10 characters) after trimming and upper-casing
- Removed the `depth` request field, which was accepted but never used
- Timestamps in API responses are timezone-aware (UTC)
- Package and API version aligned to `1.0.0`
- `docker compose` publishes the port on `127.0.0.1` only, reads all settings from `.env`, and no longer uses the obsolete `version:` key
- CI: lint, security scan and Docker build now fail the workflow instead of being ignored; dependencies are installed with `pip install -e ".[dev]"`

### Fixed
- `RiskAssessorAgent` and `ReportWriterAgent` crashed the whole run when the LLM call failed; all four LLM-backed agents now degrade to neutral / `INSUFFICIENT_DATA` results on API, network or malformed-output errors
- `SentimentAnalystAgent` raised on an unknown tone or out-of-range score from the model
- `POST /research` leaked internal exception text in its `502` response; it now returns a generic message and logs the detail server-side
- Report `company_name` fell back to the ticker because the writer read a key the data gatherer never set
- All `ruff check src/ tests/` findings; `.coveragerc` no longer references a file that does not exist

## [1.0.0] - 2026-06-16

### Added
- Deterministic asyncio fan-out/fan-in pipeline: `DataGathererAgent` runs data tools concurrently, three independent analyst agents each make a single structured Claude call, and `ReportWriterAgent` synthesizes the results into a final `InvestmentReport`
- SEC EDGAR scraper fetching recent filings via the Atom feed
- Yahoo Finance news fetcher parsing RSS headlines and descriptions
- Market data tool for fundamental metrics (P/E, margins, growth, leverage ratios)
- `FundamentalAnalystAgent`, `SentimentAnalystAgent`, and `RiskAssessorAgent` producing structured, schema-validated analyses via Claude
- Non-fatal error handling: failed external calls are captured in an `errors` list rather than aborting the run

### Changed
- CI workflow enforcing 95%+ test coverage

### Security
- No customer portfolio data or trade instructions are processed by the agent; all inputs are public market data, SEC filings, and news
