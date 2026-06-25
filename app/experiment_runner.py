"""Prompt experiment runner — benchmarks candidates via MLflow.

Runs each candidate prompt variation, calls the inference service to generate
text, computes quality metrics, logs everything to MLflow, and ranks candidates
by quality_score.

All scoring logic lives in app/evaluate_prompts.py.
"""
from __future__ import annotations

import logging
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("prompt-optimizer.runner")

# ---------------------------------------------------------------------------
# Paths — local to this app
# ---------------------------------------------------------------------------
APP_ROOT = Path(os.getenv("APP_ROOT", Path(__file__).parent.parent))
PROMPTS_DIR = APP_ROOT / "prompts"
METRICS_FILE = APP_ROOT / "metrics" / "prompt_quality.json"

# In-cluster URL of the KServe inference service
INFERENCE_URL = os.getenv(
    "INFERENCE_SERVICE_URL",
    "http://lang-learn-inference-predictor.lang-learn.svc.cluster.local/scenario_dialogue",
)

# MLflow tracking URI
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://mlflow-tracking.lang-learn.svc.cluster.local:5000",
)

# Default test scenarios (used when evaluating each candidate)
DEFAULT_SCENARIOS = [
    "ordering food at a restaurant",
    "asking for directions at the train station",
    "shopping for clothes",
]

# ---------------------------------------------------------------------------
# Candidate prompt variations
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
    {
        "name":        "structured",
        "description": "Numbered turns with clear speaker labels",
        "suffix":      (
            "\nSystem Rule: Number each turn sequentially. "
            "Format each turn as '1. Person A:' and '2. Person B:' etc. "
            "Include exactly one sentence per turn to keep the dialogue clear and scannable."
        ),
        "temperature": 0.5,
        "max_tokens":  512,
    },
    {
        "name":        "immersive",
        "description": "German-only output — no English translations",
        "suffix":      (
            "\nSystem Rule: Output ONLY German. Do not include any English text at all. "
            "Do not include 'Translation:' lines. The learner should immerse fully in German. "
            "Use only A1-A2 vocabulary so the dialogue is understandable without translation."
        ),
        "temperature": 0.8,
        "max_tokens":  512,
    },
    {
        "name":        "situational",
        "description": "Rich context — emotions, props, and stage directions",
        "suffix":      (
            "\nSystem Rule: Add brief context notes in parentheses before each turn, "
            "describing the speakers mood or action, e.g. '(lächelt)' or '(zeigt auf die Karte)'. "
            "Keep notes in simple German. This helps learners understand conversational nuance."
        ),
        "temperature": 0.7,
        "max_tokens":  768,
    },
    {
        "name":        "balanced",
        "description": "Mid-range temperature with inline vocabulary notes",
        "suffix":      (
            "\nSystem Rule: After every 4th exchange, insert a short vocabulary note line "
            "starting with 'Vokabeln:' listing 2-3 key words from the preceding turns with "
            "their English meanings. Keep the main dialogue in A1-A2 German with translations."
        ),
        "temperature": 0.6,
        "max_tokens":  768,
    },
]


# ---------------------------------------------------------------------------
# Direct evaluation (replaces DVC subprocess)
# ---------------------------------------------------------------------------

def _evaluate_candidate(candidate: dict, base_prompt: str) -> dict:
    """Evaluate a single candidate by calling the inference service directly.

    Constructs the full prompt (base + suffix), calls the inference service
    for each test scenario, and computes quality metrics.

    Returns a metrics dict compatible with composite_score().
    """
    from app.evaluate_prompts import (
        call_inference_service,
        composite_score,
        compute_vocab_level,
        count_dialogue_turns,
        count_german_chars,
        count_sentences,
        count_words,
    )

    suffix = candidate["suffix"]
    temperature = candidate["temperature"]
    max_tokens = candidate["max_tokens"]

    full_prompt = base_prompt
    if suffix:
        full_prompt = base_prompt.rstrip() + "\n" + suffix.lstrip("\n")

    scenarios = DEFAULT_SCENARIOS

    all_word_counts: list[int] = []
    all_sentence_counts: list[int] = []
    all_a1_ratios: list[float] = []
    all_b2_ratios: list[float] = []
    all_dialogue_turns: list[int] = []
    all_german_ratios: list[float] = []
    total_time = 0.0

    for scenario in scenarios:
        logger.debug(f"  Evaluating: '{scenario}' for candidate '{candidate['name']}'")

        start = time.time()
        output = call_inference_service(
            service_url=INFERENCE_URL,
            scenario=scenario,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        elapsed = time.time() - start
        total_time += elapsed

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
        "candidate_name": candidate["name"],
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
    return metrics


def _simulate_metrics(name: str) -> dict:
    """Deterministic per-candidate simulation (fallback when inference is down)."""
    rng = random.Random(name)
    base_a1 = {"current": 0.35, "precise": 0.42, "natural": 0.37, "extended": 0.36}
    base_turns = {"current": 15, "precise": 12, "natural": 16, "extended": 22}
    return {
        "candidate_name":        name,
        "avg_word_count":        rng.randint(180, 320),
        "avg_sentence_count":    rng.randint(20, 40),
        "avg_dialogue_turns":    base_turns.get(name, 15) + rng.randint(-2, 2),
        "avg_german_ratio":      round(rng.uniform(0.65, 0.85), 3),
        "a1_ratio":              round(base_a1.get(name, 0.35) + rng.uniform(-0.02, 0.08), 3),
        "b2_ratio":              round(rng.uniform(0.001, 0.05), 3),
        "avg_generation_time_s": round(rng.uniform(2.0, 8.0), 2),
    }


def _compute_quality_score(metrics: dict) -> float:
    """Compute composite quality score — delegates to evaluate_prompts."""
    from app.evaluate_prompts import composite_score
    return composite_score(metrics)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_experiments(base_prompt: str) -> list[dict]:
    """Benchmark all candidates and rank them, logging to MLflow.

    For each candidate:
      1. Evaluate directly against the inference service (no subprocess)
      2. Log parameters, metrics, and prompt text to MLflow
      3. Fall back to simulated metrics if inference is unreachable

    Returns a list of result dicts sorted best-first by quality_score.
    Each dict includes: name, description, suffix, metrics, score, latency_s, simulated.
    """
    # Import mlflow lazily so tests can mock or skip it
    try:
        import mlflow
        mlflow_available = True
    except ImportError:
        logger.warning("mlflow package not installed — experiments will not be tracked.")
        mlflow_available = False

    if mlflow_available:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("prompt-optimization")

    results = []
    run_name = f"optimization-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"

    # Context manager for the parent MLflow run (the full optimization cycle)
    parent_ctx = (
        mlflow.start_run(run_name=run_name) if mlflow_available else _noop_context()
    )

    with parent_ctx:
        if mlflow_available:
            mlflow.log_param("num_candidates", len(CANDIDATES))
            mlflow.log_param("inference_url", INFERENCE_URL)
            mlflow.log_param("scenarios", ", ".join(DEFAULT_SCENARIOS))

        for cand in CANDIDATES:
            logger.info(f"Benchmarking candidate: {cand['name']}")
            t0 = time.time()
            simulated = False

            try:
                metrics = _evaluate_candidate(cand, base_prompt)
            except Exception as e:
                logger.warning(
                    f"Evaluation failed for candidate '{cand['name']}': {e}. "
                    "Using simulated metrics."
                )
                metrics = _simulate_metrics(cand["name"])
                simulated = True

            score = metrics.get("quality_score") or _compute_quality_score(metrics)

            # Log to MLflow as a nested child run
            if mlflow_available:
                with mlflow.start_run(run_name=cand["name"], nested=True):
                    mlflow.log_params({
                        "candidate_name": cand["name"],
                        "description": cand["description"],
                        "temperature": cand["temperature"],
                        "max_tokens": cand["max_tokens"],
                        "suffix": cand["suffix"][:250] if cand["suffix"] else "(none)",
                        "simulated": str(simulated),
                    })
                    mlflow.log_metrics({
                        "quality_score": score,
                        "a1_ratio": metrics.get("a1_ratio", 0.0),
                        "b2_ratio": metrics.get("b2_ratio", 0.0),
                        "avg_german_ratio": metrics.get("avg_german_ratio", 0.0),
                        "avg_dialogue_turns": metrics.get("avg_dialogue_turns", 0.0),
                        "avg_generation_time_s": metrics.get("avg_generation_time_s", 0.0),
                        "avg_word_count": metrics.get("avg_word_count", 0.0),
                    })
                    # Log the full prompt as a text artifact
                    prompt_text = base_prompt
                    if cand["suffix"]:
                        prompt_text = base_prompt.rstrip() + "\n" + cand["suffix"]
                    mlflow.log_text(prompt_text, "prompt.txt")

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

    # Persist metrics of the winner for audit (same as before)
    import json
    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(results[0]["metrics"], f, indent=2)

    return results


class _noop_context:
    """No-op context manager used when MLflow is not available."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
