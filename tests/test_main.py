"""Tests for app.main — FastAPI endpoints."""
from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app, jobs_db, _create_job


@pytest.fixture(autouse=True)
def clear_jobs():
    """Clear the in-memory job registry before each test."""
    jobs_db.clear()
    import app.main
    app.main.latest_job_id = None
    app.main._is_optimizing = False
    app.main._last_completed_at = 0.0
    yield
    jobs_db.clear()
    app.main.latest_job_id = None
    app.main._is_optimizing = False
    app.main._last_completed_at = 0.0


@pytest.fixture
def client():
    return TestClient(app)


# ─── GET /healthz ────────────────────────────────────────────────────────────

class TestHealthz:
    def test_returns_ok(self, client):
        resp = client.get("/healthz")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "timestamp" in body


# ─── POST /optimize ──────────────────────────────────────────────────────────

class TestOptimize:
    @patch("app.main._run_optimization_sync")
    def test_returns_202(self, mock_run, client):
        resp = client.post("/optimize", json={
            "trigger_alert": "TestAlert",
            "test_scenarios": ["ordering food"],
        })
        assert resp.status_code == 202
        body = resp.json()
        assert body["status"] == "pending"
        assert "job_id" in body
        assert body["trigger_alert"] == "TestAlert"

    def test_returns_429_when_running(self, client):
        import app.main
        app.main._is_optimizing = True
        resp = client.post("/optimize", json={
            "trigger_alert": "TestAlert",
        })
        assert resp.status_code == 429
        assert "already running" in resp.json()["detail"]


# ─── GET /jobs/{job_id} ─────────────────────────────────────────────────────

class TestGetJob:
    def test_existing_job(self, client):
        job_id = _create_job("TestAlert")
        resp = client.get(f"/jobs/{job_id}")
        assert resp.status_code == 200
        assert resp.json()["job_id"] == job_id

    def test_nonexistent_job(self, client):
        resp = client.get("/jobs/does-not-exist")
        assert resp.status_code == 404


# ─── GET /jobs/latest/info ───────────────────────────────────────────────────

class TestGetLatestJob:
    def test_no_jobs_404(self, client):
        resp = client.get("/jobs/latest/info")
        assert resp.status_code == 404

    def test_returns_latest(self, client):
        _create_job("First")
        _create_job("Second")
        resp = client.get("/jobs/latest/info")
        assert resp.status_code == 200
        assert resp.json()["trigger_alert"] == "Second"


# ─── GET /jobs ───────────────────────────────────────────────────────────────

class TestListJobs:
    def test_empty_list(self, client):
        resp = client.get("/jobs")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_returns_all_jobs(self, client):
        _create_job("A")
        _create_job("B")
        resp = client.get("/jobs")
        assert resp.status_code == 200
        assert len(resp.json()) == 2


# ─── POST /webhook ──────────────────────────────────────────────────────────

class TestWebhook:
    @patch("app.main._run_optimization_sync")
    def test_firing_with_optimize_action(self, mock_run, client):
        payload = {
            "status": "firing",
            "alerts": [
                {
                    "status": "firing",
                    "labels": {
                        "alertname": "QualityDrop",
                        "action": "optimize_prompt",
                    },
                }
            ],
        }
        resp = client.post("/webhook", json=payload)
        assert resp.status_code == 202
        body = resp.json()
        assert body["jobs_started"] == 1
        assert len(body["job_ids"]) == 1

    @patch("app.main._run_optimization_sync")
    def test_skips_non_optimize_action(self, mock_run, client):
        payload = {
            "status": "firing",
            "alerts": [
                {
                    "status": "firing",
                    "labels": {
                        "alertname": "SomeAlert",
                        "action": "page_oncall",
                    },
                }
            ],
        }
        resp = client.post("/webhook", json=payload)
        assert resp.status_code == 202
        body = resp.json()
        assert body["jobs_started"] == 0

    def test_skips_when_already_optimizing(self, client):
        import app.main
        app.main._is_optimizing = True
        payload = {
            "status": "firing",
            "alerts": [
                {
                    "status": "firing",
                    "labels": {
                        "alertname": "QualityDrop",
                        "action": "optimize_prompt",
                    },
                }
            ],
        }
        resp = client.post("/webhook", json=payload)
        assert resp.status_code == 202
        body = resp.json()
        assert body["status"] == "skipped"
        assert "already in progress" in body["message"]

    def test_skips_when_in_cooldown(self, client):
        import app.main
        import time
        app.main._last_completed_at = time.time()
        payload = {
            "status": "firing",
            "alerts": [
                {
                    "status": "firing",
                    "labels": {
                        "alertname": "QualityDrop",
                        "action": "optimize_prompt",
                    },
                }
            ],
        }
        resp = client.post("/webhook", json=payload)
        assert resp.status_code == 202
        body = resp.json()
        assert body["status"] == "skipped"
        assert "cooldown active" in body["message"]

    def test_resolved_returns_no_action(self, client):
        payload = {
            "status": "resolved",
            "alerts": [],
        }
        resp = client.post("/webhook", json=payload)
        assert resp.status_code == 202
        body = resp.json()
        assert body["status"] == "resolved"


# ─── GET /current-prompt ────────────────────────────────────────────────────

class TestGetCurrentPrompt:
    @patch("app.optimizer.CURRENT_PROMPT_FILE")
    def test_prompt_exists(self, mock_path, client):
        mock_path.exists.return_value = True
        mock_path.read_text.return_value = "Test prompt text"
        mock_path.__str__ = lambda s: "/fake/path.txt"

        resp = client.get("/current-prompt")
        assert resp.status_code == 200
        body = resp.json()
        assert body["prompt"] == "Test prompt text"

    @patch("app.optimizer.CURRENT_PROMPT_FILE")
    def test_prompt_missing(self, mock_path, client):
        mock_path.exists.return_value = False
        mock_path.__str__ = lambda s: "/fake/path.txt"

        resp = client.get("/current-prompt")
        assert resp.status_code == 200
        body = resp.json()
        assert body["prompt"] == ""
        assert "warning" in body
