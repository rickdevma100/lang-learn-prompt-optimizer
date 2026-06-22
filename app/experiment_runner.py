"""Prompt experiment runner — drives DVC experiments from inside the pod.

Runs `dvc exp run` for each candidate prompt variation, reads the resulting
metrics from metrics/prompt_quality.json, and ranks candidates by quality_score.

All scoring logic lives in app/evaluate_prompts.py (the DVC pipeline stage).
"""
from __future__ import annotations

import json
import logging
import os
import random
import subprocess
import time
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

# ---------------------------------------------------------------------------
# Candidate prompt variations (driven via DVC params)
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
# DVC experiment runner
# ---------------------------------------------------------------------------

def _run_dvc_experiment(candidate: dict) -> dict | None:
    """Run a single DVC experiment for the given candidate.

    Executes:
        dvc exp run -S candidate.name={name}
                    -S candidate.suffix="{suffix}"
                    -S inference.temperature={temp}
                    -S inference.max_tokens={tokens}

    Returns the parsed metrics dict, or None if the experiment failed.
    """
    name = candidate["name"]
    suffix = candidate["suffix"]
    temperature = candidate["temperature"]
    max_tokens = candidate["max_tokens"]

    # DVC uses Hydra override grammar — string values with special chars
    # (periods, commas, etc.) must be single-quoted.  Strip the leading
    # newline that some suffixes contain since it breaks the Hydra lexer.
    escaped_suffix = suffix.lstrip("\n")

    cmd = [
        "dvc", "exp", "run", "--force",
        "-S", f"candidate.name={name}",
        "-S", f"candidate.suffix='{escaped_suffix}'",
        "-S", f"inference.temperature={temperature}",
        "-S", f"inference.max_tokens={max_tokens}",
    ]

    logger.info(f"Running DVC experiment: {name} (temp={temperature}, tokens={max_tokens})")
    logger.debug(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(APP_ROOT),
            capture_output=True,
            text=True,
            timeout=600,  # 10 min max per experiment
        )

        if result.returncode != 0:
            logger.warning(
                f"DVC experiment '{name}' failed (exit={result.returncode}).\n"
                f"  stdout: {result.stdout[-500:] if result.stdout else '(empty)'}\n"
                f"  stderr: {result.stderr[-500:] if result.stderr else '(empty)'}"
            )
            return None

        logger.info(f"DVC experiment '{name}' completed successfully.")

    except subprocess.TimeoutExpired:
        logger.warning(f"DVC experiment '{name}' timed out after 600s.")
        return None
    except FileNotFoundError:
        logger.error("DVC binary not found. Is DVC installed?")
        return None

    return _read_experiment_metrics()


def _read_experiment_metrics() -> dict | None:
    """Read the metrics/prompt_quality.json file written by the DVC stage."""
    if not METRICS_FILE.exists():
        logger.warning(f"Metrics file not found: {METRICS_FILE}")
        return None

    try:
        with open(METRICS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to read metrics file: {e}")
        return None


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
    """Compute composite quality score — same formula as evaluate_prompts.py."""
    from app.evaluate_prompts import composite_score
    return composite_score(metrics)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_experiments(base_prompt: str) -> list[dict]:
    """Benchmark all candidates via DVC experiments and rank them.

    For each candidate:
      1. Run `dvc exp run` with the candidate's parameters
      2. Read the resulting metrics from metrics/prompt_quality.json
      3. Fall back to simulated metrics if the DVC run fails

    Returns a list of result dicts sorted best-first by quality_score.
    Each dict includes: name, description, suffix, metrics, score, latency_s, simulated.
    """
    results = []

    for cand in CANDIDATES:
        logger.info(f"Benchmarking candidate: {cand['name']}")
        t0 = time.time()
        simulated = False

        metrics = _run_dvc_experiment(cand)

        if metrics is None:
            # DVC experiment failed — use deterministic simulation
            logger.warning(
                f"DVC experiment failed for candidate '{cand['name']}'. "
                "Using simulated metrics."
            )
            metrics = _simulate_metrics(cand["name"])
            simulated = True

        score = metrics.get("quality_score") or _compute_quality_score(metrics)

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
