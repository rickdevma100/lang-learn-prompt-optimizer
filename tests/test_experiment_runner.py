"""Tests for app.experiment_runner — DVC experiment orchestration and metrics."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.experiment_runner import (
    CANDIDATES,
    _compute_quality_score,
    _read_experiment_metrics,
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


# ─── _read_experiment_metrics ────────────────────────────────────────────────

class TestReadExperimentMetrics:
    def test_reads_valid_json(self, tmp_path):
        metrics = {"quality_score": 0.85, "a1_ratio": 0.4}
        metrics_file = tmp_path / "prompt_quality.json"
        metrics_file.write_text(json.dumps(metrics))

        with patch("app.experiment_runner.METRICS_FILE", metrics_file):
            result = _read_experiment_metrics()

        assert result == metrics

    def test_returns_none_for_missing_file(self, tmp_path):
        with patch("app.experiment_runner.METRICS_FILE", tmp_path / "nonexistent.json"):
            result = _read_experiment_metrics()
        assert result is None

    def test_returns_none_for_invalid_json(self, tmp_path):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not valid json {{{")

        with patch("app.experiment_runner.METRICS_FILE", bad_file):
            result = _read_experiment_metrics()
        assert result is None


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
