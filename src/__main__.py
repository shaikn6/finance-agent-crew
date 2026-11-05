"""Command-line entry point: run the research crew for one ticker.

Usage:
    python -m src TICKER [--json PATH]

Defaults to ``OFFLINE_MODE`` so it runs with no API keys and prints a
deterministic demo report. Export ``OFFLINE_MODE=0`` (with the real keys set)
to run live analysis instead.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from .models.schemas import InvestmentReport, ResearchRequest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src",
        description="Run the multi-agent equity research crew for one ticker.",
    )
    parser.add_argument("ticker", help="Ticker symbol, e.g. AAPL or BRK.B")
    parser.add_argument(
        "--json",
        metavar="PATH",
        type=Path,
        help="Write the full typed report as JSON to this path.",
    )
    return parser


def _format_report(report: InvestmentReport) -> str:
    """Render a short, human-readable summary of the typed report."""
    summary = report.executive_summary or "No executive summary available."
    return "\n".join(
        [
            f"{report.ticker} - {report.company_name}".rstrip(" -"),
            f"Signal: {report.overall_signal.value}",
            "",
            summary,
        ]
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    # Default to the keyless demo unless the operator opted into live mode.
    os.environ.setdefault("OFFLINE_MODE", "1")

    # Validate and normalise the ticker through the same model the API uses.
    try:
        request = ResearchRequest(ticker=args.ticker)
    except ValueError as exc:
        print(f"error: invalid ticker {args.ticker!r}", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 2

    # Import after OFFLINE_MODE is set so settings are read in the right mode.
    from .config import get_settings
    from .pipeline import run_research

    get_settings.cache_clear()
    report = asyncio.run(run_research(request.ticker))

    print(_format_report(report))
    if args.json:
        args.json.write_text(report.model_dump_json(indent=2))
        print(f"\nWrote report JSON to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
