"""Prompt Optimizer — core workflow engine.

All files live inside THIS application (lang-learn-prompt-optimizer):

  lang-learn-prompt-optimizer/
  ├── prompts/
  │   ├── scenario_dialogue.txt   ← current live prompt
  │   └── archive/                ← old prompts (timestamped)
  ├── metrics/
  │   └── prompt_quality.json     ← latest experiment metrics
  └── app/
      └── optimizer.py            ← this file
"""
from __future__ import annotations

import logging
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from app.experiment_runner import APP_ROOT, PROMPTS_DIR, run_experiments
from app.email_reporter import send_optimization_report

logger = logging.getLogger("prompt-optimizer")

# ---------------------------------------------------------------------------
# Paths — local to this application
# ---------------------------------------------------------------------------
CURRENT_PROMPT_FILE = PROMPTS_DIR / "scenario_dialogue.txt"
ARCHIVE_DIR         = PROMPTS_DIR / "archive"

# Only update the prompt if the winner beats the current by this margin
MIN_IMPROVEMENT = float(os.getenv("MIN_SCORE_IMPROVEMENT", "0.005"))


# ---------------------------------------------------------------------------
# Prompt I/O
# ---------------------------------------------------------------------------

def _load_current_prompt() -> str:
    """Read the live prompt from prompts/scenario_dialogue.txt."""
    if CURRENT_PROMPT_FILE.exists():
        text = CURRENT_PROMPT_FILE.read_text(encoding="utf-8").strip()
        logger.info(f"Loaded current prompt ({len(text)} chars) from {CURRENT_PROMPT_FILE}")
        return text

    logger.warning("prompts/scenario_dialogue.txt not found — using built-in default.")
    return (
        "Generate a natural German conversation between exactly two people, including their English translations.\n\n"
        "Requirements:\n"
        "- Alternate clearly between Person A and Person B.\n"
        "- For each turn, write the German sentence first, followed immediately by its English translation on the next line starting with \"Translation: \".\n"
        "- The conversation must contain at least 20 sentences total.\n"
        "- Use simple and natural German suitable for A1-A2 level learners.\n\n"
        "Specific Scenario for this conversation:\n{scenario}\n\n"
        "Start the conversation now:\nPerson A:"
    )


def _archive_and_save(new_prompt: str) -> None:
    """Archive the current prompt then write the new one atomically."""
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    if CURRENT_PROMPT_FILE.exists():
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        archive_path = ARCHIVE_DIR / f"scenario_dialogue_{ts}.txt"
        try:
            shutil.copy(CURRENT_PROMPT_FILE, archive_path)
            logger.info(f"Archived old prompt → {archive_path.name}")
        except Exception as e:
            logger.warning(f"Failed to archive old prompt: {e}")

    tmp_path = CURRENT_PROMPT_FILE.with_suffix(".txt.tmp")
    try:
        tmp_path.write_text(new_prompt, encoding="utf-8")
        os.replace(tmp_path, CURRENT_PROMPT_FILE)
        logger.info(f"Updated prompt written to {CURRENT_PROMPT_FILE}")
    except Exception as e:
        logger.error(f"Failed to write updated prompt atomically: {e}")
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
        raise


def _build_new_prompt(base: str, winner: dict) -> str:
    """Rewrite prompt by calling the BentoML rewrite_prompt API, with fallback to simple append."""
    suffix = winner.get("suffix", "").strip()
    if not suffix:
        return base

    from app.experiment_runner import INFERENCE_URL
    from app.utils import post_json

    if "/scenario_dialogue" in INFERENCE_URL:
        rewrite_url = INFERENCE_URL.replace("/scenario_dialogue", "/rewrite_prompt")
    else:
        rewrite_url = INFERENCE_URL.rstrip("/") + "/rewrite_prompt"

    logger.info(f"Attempting to rewrite prompt using LLM endpoint: {rewrite_url}")
    payload = {
        "base_prompt": base,
        "suffix": suffix,
        "temperature": 0.3,
        "max_tokens": 1024,
    }

    result = post_json(rewrite_url, payload, timeout=120)
    if result and isinstance(result, dict) and "prompt" in result:
        new_prompt = result["prompt"]
        logger.info("Successfully generated cohesive prompt via BentoML rewrite endpoint.")
        return new_prompt

    logger.info("Falling back to simple concatenation for prompt update.")
    return base.rstrip() + "\n" + suffix





# ---------------------------------------------------------------------------
# Main optimization entry point
# ---------------------------------------------------------------------------

async def run_prompt_optimization(
    job_id: str,
    trigger_alert: str,
    base_prompt: str = "",
    scenarios: list[str] | None = None,
    service_url: str | None = None,
) -> dict:
    """Run the full optimization loop.

    Steps:
      1. Load current prompt from prompts/scenario_dialogue.txt
      2. Benchmark candidates via inference service (experiment_runner)
      3. Select the best candidate
      4. If improved by ≥ MIN_IMPROVEMENT → archive old + write new prompt
      5. Send HTML email report to admin

    Returns a result dict consumed by the jobs registry in main.py.
    """
    logger.info(f"=== Optimization START  job_id={job_id}  alert={trigger_alert} ===")
    start = time.time()

    result: dict = {
        "job_id":            job_id,
        "trigger_alert":     trigger_alert,
        "status":            "running",
        "started_at":        start,
        "base_prompt":       "",
        "optimized_prompt":  "",
        "metrics_before":    {},
        "metrics_after":     {},
        "winner":            None,
        "all_candidates":    [],
        "prompt_updated":    False,
        "cluster_applied":   False,
        "email_sent":        False,
    }

    try:
        # Step 1 — load current prompt
        current_prompt = base_prompt.strip() if base_prompt.strip() else _load_current_prompt()
        result["base_prompt"] = current_prompt

        # Step 2 — run MLflow experiments against inference service
        logger.info("Benchmarking candidate prompts via MLflow experiments…")
        ranked = run_experiments(current_prompt)
        result["all_candidates"] = ranked

        # Current baseline is always in the list (name="current")
        current_cand = next((r for r in ranked if r["name"] == "current"), ranked[-1])
        winner       = ranked[0]

        result["metrics_before"] = current_cand["metrics"]
        result["metrics_after"]  = winner["metrics"]

        logger.info(
            f"Winner: {winner['name']} (score={winner['score']:.4f})  "
            f"Current: {current_cand['score']:.4f}  "
            f"Δ={winner['score'] - current_cand['score']:.4f}"
        )

        # Step 3 — decide whether to update
        improvement = winner["score"] - current_cand["score"]
        new_prompt = _build_new_prompt(current_prompt, winner)

        if winner["name"] == "current" or improvement < MIN_IMPROVEMENT:
            logger.info(
                f"No meaningful improvement (Δ={improvement:.4f} < {MIN_IMPROVEMENT}). "
                "Keeping current prompt."
            )
            result["optimized_prompt"] = current_prompt
            result["winner"]           = current_cand
        else:
            # Step 4a — archive old, write winner locally
            _archive_and_save(new_prompt)
            result["optimized_prompt"] = new_prompt
            result["winner"]           = winner
            result["prompt_updated"]   = True

            # Step 4b — apply to cluster (patch ConfigMap + restart inference pods)
            from app.cluster_apply import apply_prompt_to_cluster
            apply_ok = apply_prompt_to_cluster(new_prompt)
            result["cluster_applied"] = apply_ok

        # Step 5 — email report
        email_ok = send_optimization_report(
            alert_name      = trigger_alert,
            old_prompt      = current_prompt,
            new_prompt      = result["optimized_prompt"],
            winner          = result["winner"],
            current         = current_cand,
            all_candidates  = ranked,
            cluster_applied = result["cluster_applied"],
        )
        result["email_sent"] = email_ok

        result["status"]       = "completed"
        result["completed_at"] = time.time()
        result["duration_s"]   = round(result["completed_at"] - start, 2)

        logger.info(
            f"=== Optimization DONE  job_id={job_id}  "
            f"duration={result['duration_s']}s  "
            f"updated={result['prompt_updated']}  "
            f"cluster={result['cluster_applied']}  "
            f"email={result['email_sent']} ==="
        )

    except Exception as exc:
        logger.exception(f"Optimization FAILED  job_id={job_id}")
        result.update({
            "status":       "failed",
            "error":        str(exc),
            "completed_at": time.time(),
            "duration_s":   round(time.time() - start, 2),
        })

    return result
