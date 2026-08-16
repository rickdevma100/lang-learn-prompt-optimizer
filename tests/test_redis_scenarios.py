"""Tests for Redis scenario fetching and concurrent evaluation in experiment_runner."""
from __future__ import annotations

from unittest.mock import MagicMock, patch
import sys
# pyrefly: ignore [missing-import]
import pytest

from app.experiment_runner import DEFAULT_SCENARIOS, fetch_recent_redis_scenarios, CANDIDATES, run_experiments


def test_default_scenarios_length():
    """DEFAULT_SCENARIOS bank contains at least 5 curated scenarios."""
    assert len(DEFAULT_SCENARIOS) >= 5


def test_fetch_recent_redis_scenarios_fallback():
    """When Redis fails or import is missing, fetch_recent_redis_scenarios falls back to default scenario bank of size 5."""
    with patch.dict("sys.modules", {"redis": None}):
        scenarios = fetch_recent_redis_scenarios(limit=5)

    assert len(scenarios) == 5
    assert scenarios[0] == DEFAULT_SCENARIOS[0]


def test_fetch_recent_redis_scenarios_success():
    """When Redis has keys, fetch_recent_redis_scenarios returns Redis scenario strings."""
    mock_redis_mod = MagicMock()
    mock_redis = MagicMock()
    mock_redis_mod.Redis.return_value = mock_redis
    mock_redis.scan_iter.return_value = ["dialog:1", "dialog:2", "dialog:3"]
    mock_redis.hget.side_effect = ["redis scenario 1", "redis scenario 2", "redis scenario 3"]

    with patch.dict("sys.modules", {"redis": mock_redis_mod}):
        scenarios = fetch_recent_redis_scenarios(limit=5)

    # 3 from Redis + 2 backfilled from default scenarios = 5 total
    assert len(scenarios) == 5
    assert scenarios[0] == "redis scenario 1"
    assert scenarios[1] == "redis scenario 2"
    assert scenarios[2] == "redis scenario 3"


def test_all_candidates_require_at_least_7_words():
    """Verify every non-baseline candidate prompt suffix requires at least 7 words per turn."""
    for cand in CANDIDATES:
        if cand["suffix"]:
            lower_suffix = cand["suffix"].lower()
            assert "at least 7 words" in lower_suffix, f"Candidate {cand['name']} missing 'at least 7 words' requirement"


def test_concurrent_run_experiments_simulation():
    """Test run_experiments executes concurrently with fallback simulated metrics when inference is offline."""
    with patch("app.experiment_runner.INFERENCE_URL", "http://invalid-localhost:9999/scenario_dialogue"):
        results = run_experiments(base_prompt="Generate a dialogue.", scenarios=["test scenario 1"])

    assert len(results) == len(CANDIDATES)
    assert all("score" in r for r in results)
    # Check candidates are sorted descending by quality score
    for i in range(len(results) - 1):
        assert results[i]["score"] >= results[i+1]["score"]
