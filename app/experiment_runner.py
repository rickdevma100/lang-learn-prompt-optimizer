"""Prompt experiment runner — benchmarks candidates via MLflow.

Runs each candidate prompt variation, calls the inference service to generate
text, computes quality metrics, logs everything to MLflow, and ranks candidates
by quality_score.

All scoring logic lives in app/evaluate_prompts.py.
"""
from __future__ import annotations

import concurrent.futures
import logging
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("prompt-optimizer.runner")

# ---------------------------------------------------------------------------
# Paths & Environment — local to this app
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

# Redis Configuration for fetching recent user scenarios
REDIS_HOST = os.getenv("REDIS_HOST", "redis-stack.lang-learn.svc.cluster.local")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

# Maximum concurrent evaluation workers (should match inference pod count)
MAX_EVAL_WORKERS = int(os.getenv("MAX_EVAL_WORKERS", "2"))

# Default test scenarios bank (used when Redis scenarios are unavailable or < 5)
DEFAULT_SCENARIOS = [
    "ordering food at a restaurant",
    "asking for directions at the train station",
    "shopping for clothes",
    "booking a hotel room",
    "buying groceries at the supermarket",
]


def fetch_recent_redis_scenarios(limit: int = 5) -> list[str]:
    """Fetch up to `limit` recent unique scenarios from Redis `dialog:*` keys."""
    scenarios: list[str] = []
    try:
        # pyrefly: ignore [missing-import]
        import redis  # lazy import — optional dependency
        r = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            decode_responses=True,
            socket_connect_timeout=2.0,
        )
        for key in r.scan_iter("dialog:*"):
            scen = r.hget(key, "scenario")
            if scen and isinstance(scen, str) and scen.strip() and scen.strip() not in scenarios:
                scenarios.append(scen.strip())
            if len(scenarios) >= limit:
                break
        if scenarios:
            logger.info(f"Fetched {len(scenarios)} recent scenarios from Redis.")
    except Exception as e:
        logger.warning(f"Could not fetch scenarios from Redis ({e}). Using default scenario bank.")

    # Backfill with DEFAULT_SCENARIOS if Redis returns fewer than `limit`
    for s in DEFAULT_SCENARIOS:
        if len(scenarios) >= limit:
            break
        if s not in scenarios:
            scenarios.append(s)

    return scenarios[:limit]


# ---------------------------------------------------------------------------
# Candidate prompt variations (18 diverse candidates)
# ---------------------------------------------------------------------------
CANDIDATES = [
    # -- Baseline (always required) --
    {
        "name":        "current",
        "description": "Existing prompt — no changes",
        "suffix":      "",
        "temperature": 0.7,
        "max_tokens":  512,
    },

    # -- Target: A1 vocabulary ratio (25% of score) --
    {
        "name":        "a1_strict_vocab",
        "description": "Forces strictly A1-level vocabulary",
        "suffix":      (
            "\nSystem Rule: Use ONLY the most basic A1-level German vocabulary. "
            "Stick to the 500 most common German words. "
            "Replace any complex word with a simpler synonym. "
            "Ensure each dialogue turn contains at least 7 words. "
            "Do NOT number the turns. Do NOT put numbers before Person A or Person B."
        ),
        "temperature": 0.5,
        "max_tokens":  512,
    },

    # -- Target: German language ratio (20% of score) --
    {
        "name":        "rich_german",
        "description": "Maximises German-specific characters and vocabulary",
        "suffix":      (
            "\nSystem Rule: The German dialogue lines must be rich in German-specific "
            "characters (ä, ö, ü, ß) and idiomatic expressions. "
            "Use words like 'Straße', 'Größe', 'schön', 'natürlich', 'Gemütlichkeit'. "
            "Ensure each dialogue turn contains at least 7 words. "
            "Always keep the Translation: lines in English after each German sentence. "
            "Do NOT number the turns. Do NOT put any digits before speaker labels."
        ),
        "temperature": 0.7,
        "max_tokens":  512,
    },
    {
        "name":        "german_filler_idioms",
        "description": "Incorporates natural A1 German modal particles and fillers",
        "suffix":      (
            "\nSystem Rule: Include natural German conversational particles such as 'also', "
            "'ja', 'mal', 'denn', 'doch' to make dialogue authentic. "
            "Ensure each dialogue turn contains at least 7 words. "
            "Do NOT number speaker turns."
        ),
        "temperature": 0.65,
        "max_tokens":  512,
    },
    {
        "name":        "umlaut_emphasis",
        "description": "Encourages words with umlauts and sharp s",
        "suffix":      (
            "\nSystem Rule: Include natural German words containing ä, ö, ü, ß where applicable. "
            "Ensure each dialogue turn contains at least 7 words. "
            "Do NOT number turns."
        ),
        "temperature": 0.6,
        "max_tokens":  512,
    },
    # -- Target: Dialogue turns & depth --
    {
        "name":        "detailed_turns",
        "description": "Ensures detailed dialogue turns with at least 7 words each",
        "suffix":      (
            "\nSystem Rule: Each dialogue turn must contain at least 7 words. "
            "Avoid short micro-turns or single-word replies. "
            "Generate detailed, natural conversational responses. "
            "NEVER number the turns. NEVER write '1.', '2.', etc. "
            "Just use 'Person A:' and 'Person B:' labels directly."
        ),
        "temperature": 0.7,
        "max_tokens":  768,
    },

    # -- Target: Reduce B2 penalty (-10% of score) --
    {
        "name":        "b2_eliminator",
        "description": "Explicitly avoids B2-level vocabulary",
        "suffix":      (
            "\nSystem Rule: Avoid all advanced German vocabulary. "
            "Never use words like 'Gelegenheit', 'Voraussetzung', "
            "'beeindruckend', 'Zusammenhang', 'selbstverständlich', "
            "'allerdings', 'grundsätzlich', 'tatsächlich', 'wahrscheinlich'. "
            "Ensure each dialogue turn contains at least 7 words. "
            "Do NOT number the dialogue turns."
        ),
        "temperature": 0.6,
        "max_tokens":  512,
    },

    # -- Combined & Performance --
    {
        "name":        "optimized_blend",
        "description": "Balanced approach targeting all scoring dimensions",
        "suffix":      (
            "\nSystem Rule: Follow these rules strictly: "
            "1) Use only basic A1 vocabulary (der, die, das, haben, sein, gehen, machen). "
            "2) Ensure each dialogue turn contains at least 7 words. "
            "3) Generate at most 13 turns per person (26 total). "
            "4) Use rich German with umlauts (ä, ö, ü) and ß wherever natural. "
            "5) Always include a Translation: line in English after each German sentence. "
            "6) NEVER number the turns. Write 'Person A:' and 'Person B:' without any "
            "preceding numbers, bullets, or sequential markers."
        ),
        "temperature": 0.65,
        "max_tokens":  768,
    },
    {
        "name":        "concise_fast",
        "description": "Optimized for speed with detailed output",
        "suffix":      (
            "\nSystem Rule: Each dialogue turn must contain at least 7 words. "
            "No filler text, stage directions, or explanations outside the dialogue. "
            "No numbering of turns. No digits before speaker names. "
            "Start immediately with Person A:"
        ),
        "temperature": 0.5,
        "max_tokens":  384,
    },
]


# ---------------------------------------------------------------------------
# Direct evaluation (replaces DVC subprocess)
# ---------------------------------------------------------------------------

def _evaluate_candidate(candidate: dict, base_prompt: str, scenarios: list[str] | None = None) -> dict:
    """Evaluate a single candidate by calling the inference service directly.

    Constructs the full prompt (base + suffix), calls the inference service
    for each test scenario in `scenarios`, and computes quality metrics.

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

    test_scenarios = scenarios if scenarios else fetch_recent_redis_scenarios(5)

    all_word_counts: list[int] = []
    all_sentence_counts: list[int] = []
    all_a1_ratios: list[float] = []
    all_b2_ratios: list[float] = []
    all_dialogue_turns: list[int] = []
    all_german_ratios: list[float] = []
    total_time = 0.0

    for scenario in test_scenarios:
        logger.debug(f"  Evaluating: '{scenario}' for candidate '{candidate['name']}'")

        start = time.time()
        output = call_inference_service(
            service_url=INFERENCE_URL,
            scenario=scenario,
            max_tokens=max_tokens,
            temperature=temperature,
            prompt_template=full_prompt,
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
        "scenarios_evaluated": len(test_scenarios),
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

def run_experiments(base_prompt: str, scenarios: list[str] | None = None) -> list[dict]:
    """Benchmark all candidates concurrently and rank them, logging to MLflow.

    For each candidate:
      1. Evaluate directly against the inference service (concurrent execution)
      2. Log parameters, metrics, and prompt text to MLflow
      3. Fall back to simulated metrics if inference is unreachable

    Returns a list of result dicts sorted best-first by quality_score.
    Each dict includes: name, description, suffix, metrics, score, latency_s, simulated.
    """
    # Fetch 5 Redis scenarios if not explicitly passed
    test_scenarios = scenarios if scenarios else fetch_recent_redis_scenarios(5)

    # Import mlflow lazily so tests can mock or skip it
    try:
        # pyrefly: ignore [missing-import]
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
            mlflow.log_param("scenarios", ", ".join(test_scenarios))

        logger.info(
            f"Benchmarking {len(CANDIDATES)} candidates concurrently across {len(test_scenarios)} scenarios "
            f"(workers={MAX_EVAL_WORKERS})…"
        )

        def _eval_worker(cand: dict) -> dict:
            t0 = time.time()
            simulated = False
            try:
                metrics = _evaluate_candidate(cand, base_prompt, test_scenarios)
            except Exception as e:
                logger.warning(
                    f"Evaluation failed for candidate '{cand['name']}': {e}. "
                    "Using simulated metrics."
                )
                metrics = _simulate_metrics(cand["name"])
                simulated = True

            score = metrics.get("quality_score") or _compute_quality_score(metrics)

            return {
                "cand": cand,
                "metrics": metrics,
                "score": score,
                "simulated": simulated,
                "elapsed_s": round(time.time() - t0, 2),
            }

        cand_eval_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_EVAL_WORKERS) as executor:
            future_to_cand = {executor.submit(_eval_worker, cand): cand for cand in CANDIDATES}
            for future in concurrent.futures.as_completed(future_to_cand):
                try:
                    res = future.result()
                    cand_eval_results.append(res)
                except Exception as exc:
                    logger.error(f"Worker exception during evaluation: {exc}")

        for res in cand_eval_results:
            cand = res["cand"]
            metrics = res["metrics"]
            score = res["score"]
            simulated = res["simulated"]

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
                "elapsed_s":   res["elapsed_s"],
                "simulated":   simulated,
            })

    results.sort(key=lambda r: r["score"], reverse=True)
    logger.info(f"Ranking: {[(r['name'], r['score']) for r in results]}")

    # Persist metrics of the winner for audit
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
