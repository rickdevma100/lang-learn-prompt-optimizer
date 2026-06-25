"""Tests for app.experiment_runner — MLflow experiment orchestration and metrics."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.experiment_runner import (
    CANDIDATES,
    _compute_quality_score,
    _simulate_metrics,
)


# ─── CANDIDATES constant ─────────────────────────────────────────────────────

class TestCandidates:
    def test_has_current_candidate(self):
        names = [c["name"] for c in CANDIDATES]
        assert "current" in names

    def test_current_has_no_suffix(self):
        current = next(c for c in CANDIDATES if c["name"] == "current")
        assert current["suffix"] == ""

    def test_all_have_required_keys(self):
        for c in CANDIDATES:
            assert "name" in c
            assert "description" in c
            assert "suffix" in c
            assert "temperature" in c
            assert "max_tokens" in c

    def test_at_least_two_candidates(self):
        assert len(CANDIDATES) >= 2


# ─── _simulate_metrics ──────────────────────────────────────────────────────

class TestSimulateMetrics:
    def test_returns_expected_keys(self):
        m = _simulate_metrics("current")
        expected_keys = {
            "candidate_name", "avg_word_count", "avg_sentence_count",
            "avg_dialogue_turns", "avg_german_ratio", "a1_ratio",
            "b2_ratio", "avg_generation_time_s",
        }
        assert expected_keys.issubset(set(m.keys()))

    def test_deterministic_for_same_name(self):
        m1 = _simulate_metrics("precise")
        m2 = _simulate_metrics("precise")
        assert m1 == m2

    def test_different_for_different_names(self):
        m1 = _simulate_metrics("precise")
        m2 = _simulate_metrics("natural")
        # At least some metric should differ
        assert m1 != m2

    def test_candidate_name_matches(self):
        m = _simulate_metrics("extended")
        assert m["candidate_name"] == "extended"


# ─── _compute_quality_score ──────────────────────────────────────────────────

class TestComputeQualityScore:
    def test_delegates_to_composite_score(self):
        metrics = {
            "a1_ratio": 0.4,
            "avg_german_ratio": 0.75,
            "avg_dialogue_turns": 10,
            "b2_ratio": 0.02,
            "avg_generation_time_s": 5.0,
        }
        from app.evaluate_prompts import composite_score
        expected = composite_score(metrics)
        assert _compute_quality_score(metrics) == expected


# ─── run_experiments (with MLflow mocked) ────────────────────────────────────

class TestRunExperiments:
    @patch("app.experiment_runner._evaluate_candidate")
    def test_returns_ranked_results(self, mock_eval):
        """run_experiments returns candidates sorted by score, best first."""
        # Make _evaluate_candidate return different scores per candidate
        def side_effect(cand, base):
            scores = {"current": 0.5, "precise": 0.8, "natural": 0.6}
            name = cand["name"]
            return {
                "candidate_name": name,
                "quality_score": scores.get(name, 0.4),
                "a1_ratio": 0.35,
                "b2_ratio": 0.02,
                "avg_german_ratio": 0.7,
                "avg_dialogue_turns": 10,
                "avg_generation_time_s": 5.0,
                "avg_word_count": 200,
                "avg_sentence_count": 20,
                "total_generations": 3,
                "scenarios_evaluated": 3,
                "total_time_s": 15.0,
            }

        mock_eval.side_effect = side_effect

        # Mock mlflow so we don't need a tracking server
        mock_mlflow = MagicMock()
        mock_mlflow.start_run = MagicMock(return_value=MagicMock(
            __enter__=MagicMock(return_value=MagicMock()),
            __exit__=MagicMock(return_value=False),
        ))

        with patch.dict("sys.modules", {"mlflow": mock_mlflow}), \
             patch("app.experiment_runner.METRICS_FILE", Path("/tmp/test_metrics.json")):
            from app.experiment_runner import run_experiments
            results = run_experiments("test prompt")

        assert len(results) == len(CANDIDATES)
        # Results should be sorted descending by score
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    @patch("app.experiment_runner._evaluate_candidate")
    def test_falls_back_to_simulation_on_error(self, mock_eval):
        """When evaluation fails, simulated metrics are used."""
        mock_eval.side_effect = ConnectionError("service down")

        with patch.dict("sys.modules", {"mlflow": MagicMock()}), \
             patch("app.experiment_runner.METRICS_FILE", Path("/tmp/test_metrics.json")):
            from app.experiment_runner import run_experiments
            results = run_experiments("test prompt")

        assert all(r["simulated"] for r in results)
        assert len(results) == len(CANDIDATES)
