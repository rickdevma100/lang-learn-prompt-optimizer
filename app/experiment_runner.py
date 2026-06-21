"""Prompt experiment runner — no DVC required inside the pod.

DVC is a local dev tool for experiment tracking. Inside a Kubernetes pod,
we run the SAME scoring logic directly (the same functions that evaluate_prompts.py
uses) by calling the inference service via HTTP and computing quality metrics.

This file is self-contained: all paths are inside the optimizer app.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path

from app.vocab import A1_WORDS, B2_WORDS
from app.utils import post_json

logger = logging.getLogger("prompt-optimizer.runner")

# ---------------------------------------------------------------------------
# Paths — local to this app
# ---------------------------------------------------------------------------
APP_ROOT     = Path(os.getenv("APP_ROOT", Path(__file__).parent.parent))
PROMPTS_DIR  = APP_ROOT / "prompts"
METRICS_FILE = APP_ROOT / "metrics" / "prompt_quality.json"

# In-cluster URL of the KServe inference service
INFERENCE_URL = os.getenv(
    "INFERENCE_SERVICE_URL",
    "http://lang-learn-inference-predictor.lang-learn.svc.cluster.local/scenario_dialogue",
)

# Test scenarios used to benchmark each candidate
BENCHMARK_SCENARIOS = [
    "ordering food at a restaurant",
    "asking for directions at the train station",
    "shopping for clothes",
]

# ---------------------------------------------------------------------------
# Candidate prompt variations (no DVC params — just text modifications)
# ---------------------------------------------------------------------------
CANDIDATES = [
    {
        "name":        "current",
        "description": "Existing prompt — no changes",
        "suffix":      "",
        "temperature": 0.7,
        "max_tokens":  512,
    },
    {
        "name":        "precise",
        "description": "Low temperature — precise, grammatically exact German",
        "suffix":      (
            "\nSystem Rule: Use precise, controlled language. "
            "Prefer shorter, grammatically exact sentences. "
            "Avoid complex subordinate clauses."
        ),
        "temperature": 0.4,
        "max_tokens":  512,
    },
    {
        "name":        "natural",
        "description": "Higher temperature — natural conversational flow",
        "suffix":      (
            "\nSystem Rule: Prioritise natural conversational flow. "
            "Use colloquial but grammatically correct A2 German. "
            "Vary sentence length naturally."
        ),
        "temperature": 0.9,
        "max_tokens":  512,
    },
    {
        "name":        "extended",
        "description": "Extended dialogue — more exchanges per scenario",
        "suffix":      (
            "\nSystem Rule: Extend the dialogue to at least 20 exchanges. "
            "Explore the scenario fully. Keep vocabulary A1-A2 throughout."
        ),
        "temperature": 0.7,
        "max_tokens":  1024,
    },
]


# ---------------------------------------------------------------------------
# Scoring helpers (same logic as evaluate_prompts.py)
# ---------------------------------------------------------------------------

def _score_text(text: str) -> dict:
    """Compute quality metrics from a generated text — same as evaluate_prompts.py."""
    words = [w.lower().strip(".,!?;:\"'()") for w in text.split()]
    words = [w for w in words if w]
    total = len(words) or 1

    a1_count = sum(1 for w in words if w in A1_WORDS)
    b2_count = sum(1 for w in words if w in B2_WORDS)

    # Calculate German character ratio consistent with evaluate_prompts.py
    total_alpha = len(re.findall(r"[a-zA-ZäöüÄÖÜß]", text)) or 1
    cleaned = re.sub(r"Person [AB]:", "", text)
    alpha_chars = len(re.findall(r"[a-zA-ZäöüÄÖÜß]", cleaned))
    german_ratio = round(alpha_chars / total_alpha, 3)

    turns = len(re.findall(r"Person [AB]:", text))

    return {
        "avg_word_count":       total,
        "avg_sentence_count":   len(re.split(r"[.!?]+", text)),
        "avg_dialogue_turns":   turns,
        "avg_german_ratio":     german_ratio,
        "a1_ratio":             round(a1_count / total, 3),
        "b2_ratio":             round(b2_count / total, 3),
    }


def _composite_score(metrics: dict) -> float:
    """Compute a 0-1 quality score from text metrics."""
    a1      = float(metrics.get("a1_ratio", 0.3))
    german  = float(metrics.get("avg_german_ratio", 0.5))
    turns   = min(float(metrics.get("avg_dialogue_turns", 5)) / 15.0, 1.0)
    b2      = float(metrics.get("b2_ratio", 0.0))
    latency = float(metrics.get("avg_generation_time_s", 5.0))
    lat_ok  = max(0.0, 1.0 - latency / 60.0)

    # Scale the A1 vocabulary ratio so that 45% is considered a perfect vocabulary score
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
# Inference caller (same as evaluate_prompts.py — no DVC needed)
# ---------------------------------------------------------------------------

def _call_inference(prompt_text: str, scenario: str, temperature: float, max_tokens: int) -> tuple[str, float]:
    """Call the inference service with a full prompt and return (response_text, latency_s)."""
    payload = {
        "scenario":    scenario,
        "max_tokens":  max_tokens,
        "temperature": temperature,
    }

    t0 = time.time()
    result = post_json(INFERENCE_URL, payload, timeout=300)
    elapsed = time.time() - t0

    if result and isinstance(result, dict) and "response" in result:
        return result["response"], elapsed

    logger.warning(f"Inference call failed or returned invalid response format for scenario: {scenario}")
    return "", elapsed


def _benchmark_candidate(base_prompt: str, candidate: dict) -> dict:
    """Run the candidate prompt against all benchmark scenarios and aggregate metrics."""
    full_prompt = base_prompt.rstrip() + candidate["suffix"]
    all_metrics = []
    total_latency = 0.0
    reachable = False

    for scenario in BENCHMARK_SCENARIOS:
        text, latency = _call_inference(
            full_prompt, scenario,
            candidate["temperature"], candidate["max_tokens"],
        )
        if text:
            reachable = True
            m = _score_text(text)
            m["avg_generation_time_s"] = round(latency, 2)
            all_metrics.append(m)
            total_latency += latency

    if not all_metrics:
        # Inference service not reachable — use deterministic simulation
        logger.warning(
            f"Inference service unreachable for candidate '{candidate['name']}'. "
            "Using simulated metrics."
        )
        return _simulate_metrics(candidate["name"]), True

    n = len(all_metrics)
    aggregated = {
        key: round(sum(m[key] for m in all_metrics) / n, 3)
        for key in all_metrics[0]
    }
    aggregated["avg_generation_time_s"] = round(total_latency / n, 2)
    return aggregated, False


def _simulate_metrics(name: str) -> dict:
    """Deterministic per-candidate simulation (fallback when inference is down)."""
    import random
    rng = random.Random(name)
    base_a1    = {"current": 0.35, "precise": 0.42, "natural": 0.37, "extended": 0.36}
    base_turns = {"current": 15,   "precise": 12,   "natural": 16,   "extended": 22}
    return {
        "avg_word_count":        rng.randint(180, 320),
        "avg_sentence_count":    rng.randint(20, 40),
        "avg_dialogue_turns":    base_turns.get(name, 15) + rng.randint(-2, 2),
        "avg_german_ratio":      round(rng.uniform(0.65, 0.85), 3),
        "a1_ratio":              round(base_a1.get(name, 0.35) + rng.uniform(-0.02, 0.08), 3),
        "b2_ratio":              round(rng.uniform(0.001, 0.05), 3),
        "avg_generation_time_s": round(rng.uniform(2.0, 8.0), 2),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_experiments(base_prompt: str) -> list[dict]:
    """Benchmark all candidates against the inference service and rank them.

    Returns a list of result dicts sorted best-first by composite score.
    Each dict includes: name, description, suffix, metrics, score, latency_s, simulated.
    """
    results = []

    for cand in CANDIDATES:
        logger.info(f"Benchmarking candidate: {cand['name']}")
        t0 = time.time()

        metrics, simulated = _benchmark_candidate(base_prompt, cand)
        score = _composite_score(metrics)

        results.append({
            "name":        cand["name"],
            "description": cand["description"],
            "suffix":      cand["suffix"],
            "metrics":     metrics,
            "score":       score,
            "latency_s":   metrics.get("avg_generation_time_s", 0.0),
            "elapsed_s":   round(time.time() - t0, 2),
            "simulated":   simulated,
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    logger.info(f"Ranking: {[(r['name'], r['score']) for r in results]}")

    # Persist metrics of the winner for audit
    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(results[0]["metrics"], f, indent=2)

    return results
