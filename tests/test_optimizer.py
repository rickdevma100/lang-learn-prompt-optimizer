"""Tests for app.optimizer — prompt I/O, build logic, and the main optimization workflow."""
from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.optimizer import (
    MIN_IMPROVEMENT,
    _archive_and_save,
    _build_new_prompt,
    _load_current_prompt,
)


# ─── _load_current_prompt ───────────────────────────────────────────────────

class TestLoadCurrentPrompt:
    def test_loads_from_file(self, tmp_path):
        prompt_file = tmp_path / "scenario_dialogue.txt"
        prompt_file.write_text("  Test prompt content  ")

        with patch("app.optimizer.CURRENT_PROMPT_FILE", prompt_file):
            result = _load_current_prompt()

        assert result == "Test prompt content"

    def test_fallback_when_file_missing(self, tmp_path):
        with patch("app.optimizer.CURRENT_PROMPT_FILE", tmp_path / "nonexistent.txt"):
            result = _load_current_prompt()

        assert "German conversation" in result
        assert "{scenario}" in result


# ─── _archive_and_save ───────────────────────────────────────────────────────

class TestArchiveAndSave:
    def test_archives_old_and_writes_new(self, tmp_path):
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        prompt_file = prompts_dir / "scenario_dialogue.txt"
        prompt_file.write_text("old prompt content")
        archive_dir = prompts_dir / "archive"

        with patch("app.optimizer.CURRENT_PROMPT_FILE", prompt_file), \
             patch("app.optimizer.ARCHIVE_DIR", archive_dir):
            _archive_and_save("new prompt content")

        # New prompt written
        assert prompt_file.read_text() == "new prompt content"

        # Old prompt archived
        assert archive_dir.exists()
        archived = list(archive_dir.glob("scenario_dialogue_*.txt"))
        assert len(archived) == 1
        assert archived[0].read_text() == "old prompt content"

    def test_creates_archive_dir_if_missing(self, tmp_path):
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        prompt_file = prompts_dir / "scenario_dialogue.txt"
        prompt_file.write_text("old")
        archive_dir = prompts_dir / "archive"

        with patch("app.optimizer.CURRENT_PROMPT_FILE", prompt_file), \
             patch("app.optimizer.ARCHIVE_DIR", archive_dir):
            _archive_and_save("new")

        assert archive_dir.is_dir()

    def test_works_when_no_old_prompt(self, tmp_path):
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        prompt_file = prompts_dir / "scenario_dialogue.txt"
        archive_dir = prompts_dir / "archive"

        with patch("app.optimizer.CURRENT_PROMPT_FILE", prompt_file), \
             patch("app.optimizer.ARCHIVE_DIR", archive_dir):
            _archive_and_save("brand new prompt")

        assert prompt_file.read_text() == "brand new prompt"
        # No archive created since there was no old file
        if archive_dir.exists():
            assert len(list(archive_dir.glob("*.txt"))) == 0


# ─── _build_new_prompt ───────────────────────────────────────────────────────

class TestBuildNewPrompt:
    def test_empty_suffix_returns_base(self):
        result = _build_new_prompt("base prompt", {"suffix": ""})
        assert result == "base prompt"

    def test_no_suffix_key_returns_base(self):
        result = _build_new_prompt("base prompt", {})
        assert result == "base prompt"

    @patch("app.utils.post_json", return_value={"prompt": "LLM-rewritten prompt"})
    def test_uses_llm_rewrite_when_available(self, mock_post):
        with patch("app.experiment_runner.INFERENCE_URL", "http://example.com/scenario_dialogue"):
            result = _build_new_prompt("base", {"suffix": "add more detail"})
        assert result == "LLM-rewritten prompt"
        mock_post.assert_called_once()

    @patch("app.utils.post_json", return_value=None)
    def test_falls_back_to_concatenation(self, mock_post):
        with patch("app.experiment_runner.INFERENCE_URL", "http://example.com/scenario_dialogue"):
            result = _build_new_prompt("base prompt", {"suffix": "extra instructions"})
        assert result == "base prompt\nextra instructions"


# ─── MIN_IMPROVEMENT constant ───────────────────────────────────────────────

class TestMinImprovement:
    def test_is_positive(self):
        assert MIN_IMPROVEMENT > 0

    def test_reasonable_threshold(self):
        # Should be a small fraction, not > 1.0
        assert MIN_IMPROVEMENT < 1.0
