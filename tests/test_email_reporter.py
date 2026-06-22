"""Tests for app.email_reporter — email report building and sending."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.email_reporter import _build_html_report, _cluster_status_html, send_optimization_report


# ─── Test fixtures ───────────────────────────────────────────────────────────

def _make_candidate(name: str, score: float, latency: float = 5.0) -> dict:
    return {
        "name": name,
        "score": score,
        "latency_s": latency,
        "metrics": {
            "avg_dialogue_turns": 10,
            "a1_ratio": 0.4,
        },
    }


# ─── _cluster_status_html ───────────────────────────────────────────────────

class TestClusterStatusHtml:
    def test_applied(self):
        html = _cluster_status_html(True)
        assert "✅" in html
        assert "auto-applied" in html

    def test_not_applied(self):
        html = _cluster_status_html(False)
        assert "⚠️" in html
        assert "NOT applied" in html


# ─── _build_html_report ─────────────────────────────────────────────────────

class TestBuildHtmlReport:
    def test_contains_alert_name(self):
        winner = _make_candidate("precise", 0.85)
        current = _make_candidate("current", 0.75)
        html = _build_html_report(
            alert_name="QualityDrop",
            old_prompt="old",
            new_prompt="new",
            winner=winner,
            current=current,
            all_candidates=[current, winner],
        )
        assert "QualityDrop" in html

    def test_contains_prompts(self):
        winner = _make_candidate("precise", 0.85)
        current = _make_candidate("current", 0.75)
        html = _build_html_report(
            alert_name="Test",
            old_prompt="OLD_PROMPT_TEXT",
            new_prompt="NEW_PROMPT_TEXT",
            winner=winner,
            current=current,
            all_candidates=[current, winner],
        )
        assert "OLD_PROMPT_TEXT" in html
        assert "NEW_PROMPT_TEXT" in html

    def test_winner_highlighted(self):
        winner = _make_candidate("precise", 0.85)
        current = _make_candidate("current", 0.75)
        html = _build_html_report(
            alert_name="Test",
            old_prompt="old",
            new_prompt="new",
            winner=winner,
            current=current,
            all_candidates=[current, winner],
        )
        assert "background:#d4edda" in html

    def test_cluster_applied_section(self):
        winner = _make_candidate("precise", 0.85)
        current = _make_candidate("current", 0.75)
        html = _build_html_report(
            alert_name="Test",
            old_prompt="old",
            new_prompt="new",
            winner=winner,
            current=current,
            all_candidates=[current, winner],
            cluster_applied=True,
        )
        assert "auto-applied" in html


# ─── send_optimization_report ────────────────────────────────────────────────

class TestSendOptimizationReport:
    @patch("app.email_reporter.SMTP_PASS", "")
    def test_no_password_returns_false(self):
        """Skips sending when SMTP_PASS is empty."""
        winner = _make_candidate("precise", 0.85)
        current = _make_candidate("current", 0.75)
        result = send_optimization_report(
            alert_name="Test",
            old_prompt="old",
            new_prompt="new",
            winner=winner,
            current=current,
            all_candidates=[current, winner],
        )
        assert result is False

    @patch("app.email_reporter.smtplib.SMTP")
    @patch("app.email_reporter.SMTP_PASS", "fake_password")
    def test_sends_email_successfully(self, mock_smtp_class):
        """Returns True when email is sent successfully."""
        mock_server = MagicMock()
        mock_smtp_class.return_value.__enter__ = lambda s: mock_server
        mock_smtp_class.return_value.__exit__ = MagicMock(return_value=False)

        winner = _make_candidate("precise", 0.85)
        current = _make_candidate("current", 0.75)
        result = send_optimization_report(
            alert_name="Test",
            old_prompt="old",
            new_prompt="new",
            winner=winner,
            current=current,
            all_candidates=[current, winner],
        )
        assert result is True
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once()
        mock_server.sendmail.assert_called_once()

    @patch("app.email_reporter.smtplib.SMTP")
    @patch("app.email_reporter.SMTP_PASS", "fake_password")
    def test_smtp_failure_returns_false(self, mock_smtp_class):
        """Returns False when SMTP connection fails."""
        mock_smtp_class.side_effect = Exception("SMTP connection refused")

        winner = _make_candidate("precise", 0.85)
        current = _make_candidate("current", 0.75)
        result = send_optimization_report(
            alert_name="Test",
            old_prompt="old",
            new_prompt="new",
            winner=winner,
            current=current,
            all_candidates=[current, winner],
        )
        assert result is False
