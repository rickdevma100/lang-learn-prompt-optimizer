"""DVC evaluation script for prompt quality — run via `dvc exp run`.

This script is the entry point for the DVC pipeline stage `evaluate_prompt`.
It reads candidate parameters from params.yaml, calls the inference service,
computes quality metrics (reusing vocab.py for CEFR word lists), and writes
the results to metrics/prompt_quality.json.

Usage (via DVC):
    dvc repro evaluate_prompt
    dvc exp run -S candidate.name=precise -S inference.temperature=0.4

Usage (standalone):
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

import yaml

from app.vocab import A1_WORDS, B2_WORDS

# ---------------------------------------------------------------------------
# Paths — relative to the optimizer project root
# ---------------------------------------------------------------------------
APP_ROOT = Path(os.getenv("APP_ROOT", Path(__file__).parent.parent))
PARAMS_FILE = APP_ROOT / "params.yaml"
PROMPTS_DIR = APP_ROOT / "prompts"
METRICS_DIR = APP_ROOT / "metrics"
CURRENT_PROMPT_FILE = PROMPTS_DIR / "scenario_dialogue.txt"


def load_params() -> dict:
    """Load params.yaml from the project root."""
    with open(PARAMS_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# HTTP client — calls the running inference service
# ---------------------------------------------------------------------------

def call_inference_service(
    service_url: str,
    scenario: str,
    max_tokens: int = 512,
    temperature: float = 0.7,
    timeout: int = 300,
) -> str:
    """Call the inference service via HTTP and return the response text.

    Raises ConnectionError if the service is unreachable.
    """
    payload = json.dumps({
        "scenario": scenario,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode("utf-8")

    req = urllib.request.Request(
        service_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise ConnectionError(
            f"Cannot reach inference service at {service_url}. "
            f"Is the service running?\n  Error: {e}"
        ) from e

    if "error" in body:
        raise RuntimeError(f"Inference service error: {body['error']}")

    return body.get("response", "")


# ---------------------------------------------------------------------------
# Text-analysis helpers — consistent with evaluate_prompts.py in lang-learn-mlops
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

    Weights:
      40% A1 vocabulary (scaled so 45% A1 ratio = perfect score)
      25% German language ratio
      15% Dialogue turns (capped at 15)
      10% Latency (faster = better, penalty above 60s)
     -10% B2 vocabulary (penalty — too advanced for A1-A2 learners)
    """
    a1 = float(metrics.get("a1_ratio", 0.3))
    german = float(metrics.get("avg_german_ratio", 0.5))
    turns = min(float(metrics.get("avg_dialogue_turns", 5)) / 15.0, 1.0)
    b2 = float(metrics.get("b2_ratio", 0.0))
    latency = float(metrics.get("avg_generation_time_s", 5.0))
    lat_ok = max(0.0, 1.0 - latency / 60.0)

    # Scale A1 ratio so that 45% is a perfect vocabulary score
    a1_scaled = min(a1 / 0.45, 1.0)

    return round(
        0.40 * a1_scaled
        + 0.25 * german
        + 0.15 * turns
        + 0.10 * lat_ok
        - 0.10 * b2,
        4,
    )


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate() -> dict:
    """Run prompt evaluation via the inference service and return metrics.

    Reads parameters from params.yaml:
      - inference.service_url, inference.max_tokens, inference.temperature
      - evaluate.test_scenarios, evaluate.num_samples
      - candidate.name, candidate.suffix
    """
    params = load_params()
    inference_cfg = params.get("inference", {})
    eval_cfg = params.get("evaluate", {})
    candidate_cfg = params.get("candidate", {})

    service_url = inference_cfg.get(
        "service_url",
        "http://lang-learn-inference-predictor.lang-learn.svc.cluster.local/scenario_dialogue",
    )
    scenarios = eval_cfg.get("test_scenarios", ["ordering food at a restaurant"])
    num_samples = eval_cfg.get("num_samples", 1)
    max_tokens = inference_cfg.get("max_tokens", 512)
    temperature = inference_cfg.get("temperature", 0.7)

    candidate_name = candidate_cfg.get("name", "current")
    candidate_suffix = candidate_cfg.get("suffix", "")

    # Load the base prompt and append the candidate suffix
    if CURRENT_PROMPT_FILE.exists():
        base_prompt = CURRENT_PROMPT_FILE.read_text(encoding="utf-8").strip()
    else:
        base_prompt = ""

    full_prompt = base_prompt
    if candidate_suffix:
        full_prompt = base_prompt.rstrip() + "\n" + candidate_suffix

    print(f"Candidate: {candidate_name}")
    print(f"Inference service: {service_url}")
    print(f"Scenarios: {len(scenarios)}, Samples/scenario: {num_samples}")
    print(f"Temperature: {temperature}, Max tokens: {max_tokens}")
    if candidate_suffix:
        print(f"Suffix: {candidate_suffix[:80]}...")
    print()

    # Collect metrics across all scenarios
    all_word_counts: list[int] = []
    all_sentence_counts: list[int] = []
    all_a1_ratios: list[float] = []
    all_b2_ratios: list[float] = []
    all_dialogue_turns: list[int] = []
    all_german_ratios: list[float] = []
    total_time = 0.0

    for scenario in scenarios:
        for sample_idx in range(num_samples):
            print(f"  Evaluating: '{scenario}' (sample {sample_idx + 1}/{num_samples})")

            start = time.time()
            output = call_inference_service(
                service_url=service_url,
                scenario=scenario,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            elapsed = time.time() - start
            total_time += elapsed

            print(f"    Generated {len(output)} chars in {elapsed:.1f}s")

            # Compute metrics
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
        "candidate_name": candidate_name,
        "scenarios_evaluated": len(scenarios),
        "samples_per_scenario": num_samples,
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

    # Compute composite quality score
    metrics["quality_score"] = composite_score(metrics)

    return metrics


def main() -> None:
    """Entry point for DVC pipeline stage."""
    print("=" * 60)
    print("DVC Stage: evaluate_prompt")
    print("=" * 60)

    metrics = evaluate()

    # Write metrics
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
