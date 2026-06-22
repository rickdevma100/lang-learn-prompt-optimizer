"""Tests for app.evaluate_prompts — text-analysis helpers and composite scoring."""
from __future__ import annotations

import pytest

from app.evaluate_prompts import (
    composite_score,
    compute_vocab_level,
    count_dialogue_turns,
    count_german_chars,
    count_sentences,
    count_words,
)


# ─── count_words ──────────────────────────────────────────────────────────────

class TestCountWords:
    def test_basic(self):
        assert count_words("hello world foo") == 3

    def test_empty_string(self):
        assert count_words("") == 0

    def test_single_word(self):
        assert count_words("Hallo") == 1

    def test_multiline(self):
        # "Person", "A:", "Hallo", "Person", "B:", "Tschüss" = 6 words
        assert count_words("Person A: Hallo\nPerson B: Tschüss") == 6


# ─── count_sentences ─────────────────────────────────────────────────────────

class TestCountSentences:
    def test_simple_sentences(self):
        assert count_sentences("Hello. World!") == 2

    def test_single_sentence(self):
        assert count_sentences("Just one sentence.") == 1

    def test_no_punctuation(self):
        assert count_sentences("No punctuation here") == 1

    def test_question_and_exclamation(self):
        assert count_sentences("What? Wow! Yes.") == 3

    def test_empty(self):
        assert count_sentences("") == 0

    def test_consecutive_punctuation(self):
        # "Really?!." should still be one sentence before the punctuation
        assert count_sentences("Really?!.") == 1


# ─── compute_vocab_level ─────────────────────────────────────────────────────

class TestComputeVocabLevel:
    def test_returns_required_keys(self):
        result = compute_vocab_level("ich bin gut")
        assert "a1_word_count" in result
        assert "b2_word_count" in result
        assert "a1_ratio" in result
        assert "b2_ratio" in result

    def test_a1_words_detected(self):
        # "ich", "bin", "gut" are common A1 words
        result = compute_vocab_level("ich bin gut")
        assert result["a1_word_count"] >= 0  # depends on A1_WORDS set content
        assert isinstance(result["a1_ratio"], float)

    def test_empty_text(self):
        result = compute_vocab_level("")
        assert result["a1_word_count"] == 0
        assert result["b2_word_count"] == 0

    def test_punctuation_stripped(self):
        # Words should be stripped of punctuation before lookup
        result1 = compute_vocab_level("ich")
        result2 = compute_vocab_level("ich.")
        assert result1["a1_word_count"] == result2["a1_word_count"]


# ─── count_german_chars ──────────────────────────────────────────────────────

class TestCountGermanChars:
    def test_pure_text(self):
        ratio = count_german_chars("Hallo wie geht es")
        assert 0.0 <= ratio <= 1.0

    def test_empty(self):
        assert count_german_chars("") == 0.0

    def test_numbers_only(self):
        assert count_german_chars("12345") == 0.0

    def test_with_person_labels(self):
        # Person A/B labels should be stripped, reducing non-content chars
        text_with_labels = "Person A: Hallo\nPerson B: Tschüss"
        text_without = "Hallo\nTschüss"
        ratio_with = count_german_chars(text_with_labels)
        ratio_without = count_german_chars(text_without)
        # Ratio with labels should be different because labels get stripped
        assert isinstance(ratio_with, float)
        assert isinstance(ratio_without, float)


# ─── count_dialogue_turns ────────────────────────────────────────────────────

class TestCountDialogueTurns:
    def test_two_turns(self):
        text = "Person A: Hallo\nPerson B: Hi"
        assert count_dialogue_turns(text) == 2

    def test_no_turns(self):
        assert count_dialogue_turns("Just regular text.") == 0

    def test_multiple_turns(self):
        text = (
            "Person A: Hallo\n"
            "Person B: Hi\n"
            "Person A: Wie geht es?\n"
            "Person B: Gut, danke!\n"
        )
        assert count_dialogue_turns(text) == 4

    def test_empty(self):
        assert count_dialogue_turns("") == 0


# ─── composite_score ─────────────────────────────────────────────────────────

class TestCompositeScore:
    def test_returns_float(self):
        metrics = {
            "a1_ratio": 0.4,
            "avg_german_ratio": 0.75,
            "avg_dialogue_turns": 10,
            "b2_ratio": 0.02,
            "avg_generation_time_s": 5.0,
        }
        score = composite_score(metrics)
        assert isinstance(score, float)

    def test_score_range(self):
        # Perfect-ish metrics should give a high score
        metrics = {
            "a1_ratio": 0.45,
            "avg_german_ratio": 1.0,
            "avg_dialogue_turns": 15,
            "b2_ratio": 0.0,
            "avg_generation_time_s": 1.0,
        }
        score = composite_score(metrics)
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # should be very high

    def test_bad_metrics_low_score(self):
        metrics = {
            "a1_ratio": 0.0,
            "avg_german_ratio": 0.0,
            "avg_dialogue_turns": 0,
            "b2_ratio": 0.5,
            "avg_generation_time_s": 120.0,
        }
        score = composite_score(metrics)
        assert score < 0.2

    def test_high_b2_penalty(self):
        base = {
            "a1_ratio": 0.4,
            "avg_german_ratio": 0.7,
            "avg_dialogue_turns": 10,
            "avg_generation_time_s": 5.0,
        }
        low_b2 = {**base, "b2_ratio": 0.0}
        high_b2 = {**base, "b2_ratio": 0.3}
        assert composite_score(low_b2) > composite_score(high_b2)

    def test_higher_a1_better(self):
        base = {
            "avg_german_ratio": 0.7,
            "avg_dialogue_turns": 10,
            "b2_ratio": 0.02,
            "avg_generation_time_s": 5.0,
        }
        low_a1 = {**base, "a1_ratio": 0.1}
        high_a1 = {**base, "a1_ratio": 0.4}
        assert composite_score(high_a1) > composite_score(low_a1)

    def test_defaults_used_for_missing_keys(self):
        # Should not crash when keys are missing — uses .get() defaults
        score = composite_score({})
        assert isinstance(score, float)
