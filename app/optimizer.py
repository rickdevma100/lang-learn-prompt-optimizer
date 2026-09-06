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

def _sanitize_prompt(text: str) -> str:
    """Remove contradictory instructions that would strip English translations.

    Previous optimization runs sometimes bake in rules like
    'Do not include any English translation' or 'Output ONLY the German conversation'
    which directly conflict with the requirement to include Translation: lines.

    Also strips accumulated System Rule lines beyond the first — these pile up
    when multiple optimization runs concatenate winner suffixes.
    """
    import re

    lines = text.split("\n")
    cleaned: list[str] = []

    # Lines that explicitly strip English translations — remove them
    poison_patterns = [
        re.compile(r".*do\s+not\s+include\s+any\s+english\s+translation.*", re.IGNORECASE),
        re.compile(r".*output\s+only\s+the\s+german\s+conversation.*", re.IGNORECASE),
        re.compile(r".*100%\s+german\s+with\s+(zero|no)\s+english.*", re.IGNORECASE),
        re.compile(r".*do\s+not\s+include\s+.*english\s+text.*", re.IGNORECASE),
        # Anti-numbering: remove any instruction to number turns sequentially
        re.compile(r".*number\s+each\s+turn.*", re.IGNORECASE),
        re.compile(r".*format\s+each\s+turn\s+as\s+['\"]?\d+\.", re.IGNORECASE),
        re.compile(r".*\d+\.\s*Person\s+[AB].*format.*", re.IGNORECASE),
    ]

    # Only keep the FIRST "System Rule:" block — strip duplicates
    seen_system_rule = False

    for line in lines:
        stripped = line.strip()

        # Skip poison lines
        if any(p.match(stripped) for p in poison_patterns):
            logger.info(f"Sanitize: removing contradictory line: '{stripped[:80]}'")
            continue

        # Collapse duplicate System Rule blocks
        if stripped.startswith("System Rule:"):
            if seen_system_rule:
                logger.info(f"Sanitize: removing duplicate System Rule: '{stripped[:80]}'")
                continue
            seen_system_rule = True

        cleaned.append(line)

    result = "\n".join(cleaned).strip()

    # Ensure Translation: instruction exists
    if "translation" not in result.lower():
        logger.warning("Sanitize: re-injecting Translation: instruction")
        # Insert after the first "Requirements:" line
        idx = result.lower().find("requirements:")
        if idx != -1:
            newline_after = result.find("\n", idx)
            if newline_after != -1:
                result = (
                    result[:newline_after + 1]
                    + '- For each turn, write the German sentence first, followed immediately '
                    'by its English translation on the next line starting with "Translation: ".\n'
                    + result[newline_after + 1:]
                )

    return result


def _load_current_prompt() -> str:
    """Read the live prompt from prompts/scenario_dialogue.txt."""
    if CURRENT_PROMPT_FILE.exists():
        text = CURRENT_PROMPT_FILE.read_text(encoding="utf-8").strip()
        logger.info(f"Loaded current prompt ({len(text)} chars) from {CURRENT_PROMPT_FILE}")
        sanitized = _sanitize_prompt(text)
        if sanitized != text:
            logger.warning("Prompt was sanitized to fix contradictory instructions.")
        return sanitized

    logger.warning("prompts/scenario_dialogue.txt not found — using built-in default.")
    return (
        "Generate a natural German conversation between exactly two people, including their English translations.\n\n"
        "Requirements:\n"
        "- Alternate clearly between Person A and Person B.\n"
        '- For each turn, write the German sentence first, followed immediately by its English translation on the next line starting with "Translation: ".\n'
        "- Each dialogue turn must contain at least 7 words.\n"
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


def _validate_prompt(prompt: str) -> tuple[bool, str]:
    """Validate that a rewritten prompt preserves required format constraints.

    Returns (is_valid, reason). A prompt is invalid if:
    - It no longer instructs the LLM to include English translations
    - It affirmatively instructs the LLM to number turns (causes '1. Person A:' output)
      Note: "Do NOT number the turns" is fine — we only flag affirmative numbering instructions.
    """
    import re as _re
    lower = prompt.lower()

    # Must still require English translations
    if "translation" not in lower and "english translation" not in lower:
        return False, "rewritten prompt lost 'Translation:' instruction — would remove English from output"

    # Check for explicit numbered-format instructions like "1. Person A:" / "2. Person B:"
    if _re.search(r"\b1\.\s+person\s+a\b", lower) or _re.search(r"\b2\.\s+person\s+b\b", lower):
        return False, "rewritten prompt contains '1. Person A:' / '2. Person B:' turn numbering"

    if _re.search(r"format\s+(?:each\s+)?turn\s+as\s+['\"]?1\.", lower):
        return False, "rewritten prompt instructs numbering turns with '1. Person A:' format"

    # Check "number each/the turns" — only flag if NOT preceded by a negation word
    # We do this by checking every occurrence of "number" and seeing if it is negated
    for m in _re.finditer(r"\bnumber\s+(?:each\s+)?turns?\b", lower):
        # Look at the 20 chars before the match for negation words
        start = max(0, m.start() - 20)
        context = lower[start:m.start()]
        if not _re.search(r"\b(?:not|don't|never|no)\b", context):
            return False, "rewritten prompt affirmatively instructs numbering of turns"

    # Check "sequentially" — only flag if describing turn format, not negated
    for m in _re.finditer(r"\bsequentially\b", lower):
        start = max(0, m.start() - 20)
        context = lower[start:m.start()]
        if not _re.search(r"\b(?:not|don't|never|no)\b", context):
            return False, "rewritten prompt instructs sequential turn numbering"

    return True, "ok"


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
        "max_tokens": 512,
    }

    result = post_json(rewrite_url, payload, timeout=120)
    if result and isinstance(result, dict) and "prompt" in result:
        new_prompt = result["prompt"]

        # Validate the LLM-rewritten prompt preserves required format
        is_valid, reason = _validate_prompt(new_prompt)
        if is_valid:
            logger.info("Successfully generated cohesive prompt via BentoML rewrite endpoint.")
            return new_prompt
        else:
            logger.warning(
                f"LLM-rewritten prompt failed validation ({reason}). "
                "Falling back to safe suffix append."
            )

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
        ranked = run_experiments(current_prompt, scenarios=scenarios)
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
