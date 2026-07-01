"""Prompt quality evaluation — scoring functions and inference client.

Contains the text-analysis helpers and composite scoring formula used
to evaluate prompt candidates. Called directly by experiment_runner.py.

Standalone usage:
    python -m app.evaluate_prompts
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from app.vocab import A1_WORDS, B2_WORDS

# ---------------------------------------------------------------------------
# Paths — relative to the optimizer project root
# ---------------------------------------------------------------------------
APP_ROOT = Path(os.getenv("APP_ROOT", Path(__file__).parent.parent))
PROMPTS_DIR = APP_ROOT / "prompts"
METRICS_DIR = APP_ROOT / "metrics"
CURRENT_PROMPT_FILE = PROMPTS_DIR / "scenario_dialogue.txt"


# ---------------------------------------------------------------------------
# HTTP client — calls the running inference service
# ---------------------------------------------------------------------------

def call_inference_service(
    service_url: str,
    scenario: str,
    max_tokens: int = 512,
    temperature: float = 0.7,
    timeout: int = 300,
    prompt_template: str = "",
) -> str:
    """Call the inference service via HTTP and return the response text.

    If `prompt_template` is provided, the inference service will use it
    instead of its default prompt loaded from the ConfigMap.  The template
    must contain a ``{scenario}`` placeholder.

    Raises ConnectionError if the service is unreachable.
    """
    body: dict = {
        "scenario": scenario,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "bypass_cache": True,
    }
    if prompt_template:
        body["prompt_template"] = prompt_template

    payload = json.dumps(body).encode("utf-8")

    req = urllib.request.Request(
        service_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp_body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise ConnectionError(
            f"Cannot reach inference service at {service_url}. "
            f"Is the service running?\n  Error: {e}"
        ) from e

    if "error" in resp_body:
        raise RuntimeError(f"Inference service error: {resp_body['error']}")

    return resp_body.get("response", "")


# ---------------------------------------------------------------------------
# Text-analysis helpers
# ---------------------------------------------------------------------------

def count_words(text: str) -> int:
    """Count words in generated text."""
    return len(text.split())


def count_sentences(text: str) -> int:
    """Count sentences based on terminal punctuation."""
    sentences = re.split(r"[.!?]+", text)
    return len([s for s in sentences if s.strip()])


def compute_vocab_level(text: str) -> dict:
    """Estimate CEFR vocabulary distribution in the generated text."""
    words = [w.lower().strip(".,!?;:\"'()") for w in text.split()]
    words = [w for w in words if w]

    total = len(words) if words else 1
    a1_count = sum(1 for w in words if w in A1_WORDS)
    b2_count = sum(1 for w in words if w in B2_WORDS)

    return {
        "a1_word_count": a1_count,
        "b2_word_count": b2_count,
        "a1_ratio": round(a1_count / total, 3),
        "b2_ratio": round(b2_count / total, 3),
    }


def count_german_chars(text: str) -> float:
    """Estimate the fraction of text that is German alphabetic content."""
    total_alpha = len(re.findall(r"[a-zA-ZäöüÄÖÜß]", text))
    if total_alpha == 0:
        return 0.0
    # Remove Person A/B labels as non-content overhead
    cleaned = re.sub(r"Person [AB]:", "", text)
    alpha_chars = len(re.findall(r"[a-zA-ZäöüÄÖÜß]", cleaned))
    return round(alpha_chars / max(total_alpha, 1), 3)


def count_dialogue_turns(text: str) -> int:
    """Count dialogue turns (Person A: / Person B: lines)."""
    return len(re.findall(r"Person [AB]:", text))


def composite_score(metrics: dict) -> float:
    """Compute a 0-1 quality score from text metrics.

    IMPORTANT: This formula MUST match compute_cefr_score() in
    inference/src/metrics.py so the optimizer targets the same
    metric that Prometheus monitors.

    Weights (for A1/A2 targets):
      40%  A1 vocabulary ratio (scaled so 45% A1 ratio = perfect)
      20%  B2 penalty (inverted: 1 - b2_ratio)
      20%  German language content ratio
      20%  Dialogue turns (capped at 10)
    """
    a1 = float(metrics.get("a1_ratio", 0.3))
    german = float(metrics.get("avg_german_ratio", 0.5))
    turns = min(float(metrics.get("avg_dialogue_turns", 5)) / 10.0, 1.0)
    b2 = float(metrics.get("b2_ratio", 0.0))

    # Scale A1 ratio so that 45% is a perfect vocabulary score
    a1_scaled = min(a1 / 0.45, 1.0)

    return round(
        0.40 * a1_scaled
        + 0.20 * (1.0 - b2)
        + 0.20 * german
        + 0.20 * turns,
        4,
    )


# ---------------------------------------------------------------------------
# Standalone entry point (for manual testing / debugging)
# ---------------------------------------------------------------------------

def main() -> None:
    """Standalone evaluation entry point for manual testing."""
    from app.experiment_runner import INFERENCE_URL, DEFAULT_SCENARIOS

    print("=" * 60)
    print("Prompt Quality Evaluation")
    print("=" * 60)

    service_url = INFERENCE_URL
    scenarios = DEFAULT_SCENARIOS

    if CURRENT_PROMPT_FILE.exists():
        base_prompt = CURRENT_PROMPT_FILE.read_text(encoding="utf-8").strip()
    else:
        base_prompt = ""

    print(f"Inference service: {service_url}")
    print(f"Scenarios: {len(scenarios)}")
    print()

    all_word_counts: list[int] = []
    all_sentence_counts: list[int] = []
    all_a1_ratios: list[float] = []
    all_b2_ratios: list[float] = []
    all_dialogue_turns: list[int] = []
    all_german_ratios: list[float] = []
    total_time = 0.0

    for scenario in scenarios:
        print(f"  Evaluating: '{scenario}'")

        start = time.time()
        output = call_inference_service(
            service_url=service_url,
            scenario=scenario,
        )
        elapsed = time.time() - start
        total_time += elapsed

        print(f"    Generated {len(output)} chars in {elapsed:.1f}s")

        wc = count_words(output)
        sc = count_sentences(output)
        vocab = compute_vocab_level(output)
        turns = count_dialogue_turns(output)
        german_ratio = count_german_chars(output)

        all_word_counts.append(wc)
        all_sentence_counts.append(sc)
        all_a1_ratios.append(vocab["a1_ratio"])
        all_b2_ratios.append(vocab["b2_ratio"])
        all_dialogue_turns.append(turns)
        all_german_ratios.append(german_ratio)

    n = len(all_word_counts) or 1

    metrics = {
        "candidate_name": "manual-eval",
        "scenarios_evaluated": len(scenarios),
        "total_generations": n,
        "avg_word_count": round(sum(all_word_counts) / n, 1),
        "avg_sentence_count": round(sum(all_sentence_counts) / n, 1),
        "avg_dialogue_turns": round(sum(all_dialogue_turns) / n, 1),
        "avg_german_ratio": round(sum(all_german_ratios) / n, 3),
        "a1_ratio": round(sum(all_a1_ratios) / n, 3),
        "b2_ratio": round(sum(all_b2_ratios) / n, 3),
        "avg_generation_time_s": round(total_time / n, 2),
        "total_time_s": round(total_time, 2),
    }

    metrics["quality_score"] = composite_score(metrics)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_file = METRICS_DIR / "prompt_quality.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"\nMetrics written to: {metrics_file}")
    print(json.dumps(metrics, indent=2))
    print(f"\nQuality Score: {metrics['quality_score']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
