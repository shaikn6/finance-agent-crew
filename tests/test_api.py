"""Tests for src/api/main.py — exercised through FastAPI's TestClient."""

from __future__ import annotations

import logging
from datetime import datetime
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from src import __version__
from src.api.main import app
from src.api.security import RateLimiter
from src.config import ConfigurationError


@pytest.fixture
def offline_env(monkeypatch):
    monkeypatch.setenv("OFFLINE_MODE", "1")
    monkeypatch.delenv("ANTHROPIC_API_KEY")


@pytest.fixture
def client(offline_env):
    with TestClient(app) as test_client:
        yield test_client


class TestHealth:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_version_matches_package(self):
        assert app.version == __version__ == "1.0.0"


class TestResearchOffline:
    def test_success_shape(self, client):
        resp = client.post("/research", json={"ticker": "DEMO"})
        assert resp.status_code == 200
        body = resp.json()
        assert set(body) == {
            "job_id",
            "ticker",
            "status",
            "report",
            "error",
            "created_at",
            "completed_at",
            "duration_seconds",
        }
        assert body["ticker"] == "DEMO"
        assert body["status"] == "completed"
        assert body["error"] is None
        assert body["duration_seconds"] >= 0
        report = body["report"]
        assert report["ticker"] == "DEMO"
        assert report["overall_signal"] == "HOLD"
        assert report["executive_summary"]
        assert report["fundamental_analysis"]["signal"] == "HOLD"
        assert report["sentiment_analysis"]["management_tone"] == "positive"
        assert report["risk_assessment"]["overall_risk_level"] == "medium"

    def test_timestamps_are_timezone_aware(self, client):
        body = client.post("/research", json={"ticker": "DEMO"}).json()
        for field in ("created_at", "completed_at"):
            assert datetime.fromisoformat(body[field]).tzinfo is not None
        assert datetime.fromisoformat(body["report"]["report_date"]).tzinfo is not None

    def test_ticker_is_normalised(self, client):
        resp = client.post("/research", json={"ticker": " brk.b "})
        assert resp.status_code == 200
        assert resp.json()["ticker"] == "BRK.B"

    def test_include_flags_reach_the_gatherer(self, client):
        resp = client.post(
            "/research", json={"ticker": "DEMO", "include_news": False}
        )
        assert resp.status_code == 200
        assert resp.json()["report"]["sentiment_analysis"]["headline_count"] == 0


class TestResearchValidation:
    @pytest.mark.parametrize(
        "payload",
        [
            {},
            {"ticker": ""},
            {"ticker": "A" * 11},
            {"ticker": "AAPL$"},
            {"ticker": "AA PL"},
            {"ticker": "../etc"},
            {"ticker": "-AAPL"},
            {"ticker": 123},
        ],
    )
    def test_invalid_payload_returns_422(self, client, payload):
        assert client.post("/research", json=payload).status_code == 422


class TestResearchUpstreamFailure:
    def test_pipeline_failure_returns_generic_502(self, client, monkeypatch, caplog):
        monkeypatch.setattr(
            "src.api.main.DataGathererAgent.gather",
            AsyncMock(side_effect=RuntimeError("secret-internal-detail")),
        )
        with caplog.at_level(logging.ERROR, logger="src.api.main"):
            resp = client.post("/research", json={"ticker": "DEMO"})
        assert resp.status_code == 502
        assert resp.json() == {"detail": "Upstream research pipeline failed."}
        assert "secret-internal-detail" not in resp.text
        assert "secret-internal-detail" in caplog.text

    def test_single_analyst_llm_failure_does_not_fail_the_run(
        self, client, monkeypatch
    ):
        broken = AsyncMock()
        broken.messages.create = AsyncMock(side_effect=RuntimeError("LLM down"))
        monkeypatch.setattr(
            "src.agents.risk_assessor.get_llm_client", lambda: broken
        )
        resp = client.post("/research", json={"ticker": "DEMO"})
        assert resp.status_code == 200
        report = resp.json()["report"]
        assert report["risk_assessment"]["key_risks_summary"] == (
            "Risk assessment unavailable."
        )
        assert report["fundamental_analysis"]["signal"] == "HOLD"
        assert report["overall_signal"] == "HOLD"


class TestApiKeyAuth:
    @pytest.fixture
    def keyed_client(self, offline_env, monkeypatch):
        monkeypatch.setenv("API_KEYS", "alpha,beta")
        with TestClient(app) as test_client:
            yield test_client

    def test_offline_mode_without_keys_is_open(self, client):
        assert client.post("/research", json={"ticker": "DEMO"}).status_code == 200

    def test_missing_key_rejected(self, keyed_client):
        resp = keyed_client.post("/research", json={"ticker": "DEMO"})
        assert resp.status_code == 401
        assert resp.json() == {"detail": "Invalid or missing API key."}

    def test_wrong_key_rejected(self, keyed_client):
        resp = keyed_client.post(
            "/research", json={"ticker": "DEMO"}, headers={"X-API-Key": "nope"}
        )
        assert resp.status_code == 401

    @pytest.mark.parametrize("key", ["alpha", "beta"])
    def test_valid_key_accepted(self, keyed_client, key):
        resp = keyed_client.post(
            "/research", json={"ticker": "DEMO"}, headers={"X-API-Key": key}
        )
        assert resp.status_code == 200

    def test_health_needs_no_key(self, keyed_client):
        assert keyed_client.get("/health").status_code == 200

    def test_live_mode_without_api_keys_refuses_to_start(self):
        with pytest.raises(ConfigurationError, match="API_KEYS"), TestClient(app):
            pass

    def test_live_mode_without_anthropic_key_refuses_to_start(self, monkeypatch):
        monkeypatch.setenv("API_KEYS", "alpha")
        monkeypatch.delenv("ANTHROPIC_API_KEY")
        with (
            pytest.raises(ConfigurationError, match="ANTHROPIC_API_KEY"),
            TestClient(app),
        ):
            pass


class TestRateLimit:
    def test_requests_over_limit_get_429(self, offline_env, monkeypatch):
        monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "2")
        with TestClient(app) as test_client:
            statuses = [
                test_client.post("/research", json={"ticker": "DEMO"})
                for _ in range(3)
            ]
            assert [r.status_code for r in statuses] == [200, 200, 429]
            assert statuses[2].json() == {"detail": "Rate limit exceeded."}
            assert int(statuses[2].headers["Retry-After"]) >= 1
            assert test_client.get("/health").status_code == 200

    def test_limit_is_tracked_per_api_key(self, offline_env, monkeypatch):
        monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "1")
        monkeypatch.setenv("API_KEYS", "alpha,beta")
        with TestClient(app) as test_client:

            def post(key: str) -> int:
                return test_client.post(
                    "/research", json={"ticker": "DEMO"}, headers={"X-API-Key": key}
                ).status_code

            assert post("alpha") == 200
            assert post("alpha") == 429
            assert post("beta") == 200

    def test_rejected_auth_does_not_consume_quota(self, offline_env, monkeypatch):
        monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "1")
        monkeypatch.setenv("API_KEYS", "alpha")
        with TestClient(app) as test_client:
            for _ in range(3):
                assert (
                    test_client.post("/research", json={"ticker": "DEMO"}).status_code
                    == 401
                )


class TestRateLimiter:
    def test_window_slides(self):
        now = [0.0]
        limiter = RateLimiter(limit=2, window_seconds=60, clock=lambda: now[0])
        assert limiter.check("a") is None
        now[0] = 10.0
        assert limiter.check("a") is None
        now[0] = 20.0
        assert limiter.check("a") == pytest.approx(40.0)
        now[0] = 60.5
        assert limiter.check("a") is None

    def test_clients_are_independent(self):
        limiter = RateLimiter(limit=1, window_seconds=60, clock=lambda: 0.0)
        assert limiter.check("a") is None
        assert limiter.check("b") is None
        assert limiter.check("a") is not None

    def test_idle_clients_are_forgotten(self):
        now = [0.0]
        limiter = RateLimiter(limit=1, window_seconds=60, clock=lambda: now[0])
        limiter.check("a")
        now[0] = 61.0
        limiter.check("b")
        assert set(limiter._hits) == {"b"}
