"""Tests for the command-line entry point in src/__main__.py (offline/demo mode)."""

from __future__ import annotations

import pytest

from src.__main__ import main
from src.models.schemas import InvestmentReport, Signal


@pytest.fixture
def offline_env(monkeypatch):
    """Offline mode with no keys, and any real HTTP client forbidden."""
    monkeypatch.setenv("OFFLINE_MODE", "1")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    def _no_network(*args, **kwargs):
        raise AssertionError("offline mode must not create an HTTP client")

    monkeypatch.setattr("httpx.AsyncClient", _no_network)


class TestCli:
    def test_prints_typed_summary_and_exits_zero(self, offline_env, capsys):
        exit_code = main(["DEMO"])
        assert exit_code == 0
        out = capsys.readouterr().out
        assert "DEMO" in out
        assert f"Signal: {Signal.HOLD.value}" in out
        assert "offline demo" in out.lower()

    def test_normalises_lowercase_ticker(self, offline_env, capsys):
        assert main(["demo"]) == 0
        assert "DEMO" in capsys.readouterr().out

    def test_json_flag_writes_parseable_report(self, offline_env, tmp_path, capsys):
        out_path = tmp_path / "report.json"
        exit_code = main(["DEMO", "--json", str(out_path)])
        assert exit_code == 0
        assert out_path.exists()
        report = InvestmentReport.model_validate_json(out_path.read_text())
        assert report.ticker == "DEMO"
        assert report.overall_signal == Signal.HOLD
        assert report.executive_summary
        assert str(out_path) in capsys.readouterr().out

    def test_invalid_ticker_exits_two(self, offline_env, capsys):
        exit_code = main(["AA PL"])
        assert exit_code == 2
        assert "invalid ticker" in capsys.readouterr().err
