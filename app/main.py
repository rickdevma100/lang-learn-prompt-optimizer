"""FastAPI Prompt Optimizer Webhook Service.

Endpoints:
  POST /webhook          — Alertmanager webhook receiver (202, async)
  POST /optimize         — Manual trigger with custom parameters (202, async)
  GET  /jobs/{job_id}    — Poll job status / results
  GET  /jobs/latest/info — Most recent job result
  POST /feedback         — Record user feedback (up/down) forwarded to metrics
  GET  /healthz          — Kubernetes liveness & readiness probe
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

# pyrefly: ignore [missing-import]
from fastapi import BackgroundTasks, FastAPI, HTTPException, status
# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field

from app.config import settings
from app.optimizer import run_prompt_optimization

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
)
logger = logging.getLogger("prompt-optimizer")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Lang-Learn Prompt Optimizer",
    description=(
        "Scalable, alert-driven prompt optimization service. "
        "Receives Alertmanager webhooks, benchmarks prompt candidates "
        "via the inference service, selects the best prompt, and emails a report."
    ),

    version="1.0.0",
)

# ---------------------------------------------------------------------------
# In-memory job registry
# ---------------------------------------------------------------------------
jobs_db: Dict[str, Dict[str, Any]] = {}
latest_job_id: Optional[str] = None


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class OptimizeRequest(BaseModel):
    trigger_alert: str = Field(default="ManualTrigger")
    base_prompt: str = Field(default="")
    test_scenarios: List[str] = Field(
        default_factory=lambda: [
            "ordering food at a restaurant in Berlin",
            "booking a hotel room in Munich",
        ]
    )
    service_url: Optional[str] = None


class JobResponse(BaseModel):
    job_id: str
    status: str
    trigger_alert: str
    created_at: str
    duration_s: Optional[float] = None
    error: Optional[str] = None
    prompt_updated: Optional[bool] = None
    cluster_applied: Optional[bool] = None
    email_sent: Optional[bool] = None
    optimized_prompt: Optional[str] = None
    winner: Optional[Dict[str, Any]] = None
    all_candidates: Optional[List[Dict[str, Any]]] = None


# Alertmanager standard payload schema
class AlertLabel(BaseModel):
    alertname: Optional[str] = None
    severity: Optional[str] = None
    action: Optional[str] = None
    language: Optional[str] = "German"
    level: Optional[str] = "A2"

    class Config:
        extra = "allow"   # accept any additional labels


class AlertAnnotation(BaseModel):
    summary: Optional[str] = None
    description: Optional[str] = None

    class Config:
        extra = "allow"


class AlertmanagerAlert(BaseModel):
    status: str
    labels: AlertLabel = Field(default_factory=AlertLabel)
    annotations: AlertAnnotation = Field(default_factory=AlertAnnotation)
    startsAt: Optional[str] = None
    endsAt: Optional[str] = None
    fingerprint: Optional[str] = None


class AlertmanagerWebhookPayload(BaseModel):
    receiver: str = "prompt-optimizer"
    status: str
    alerts: List[AlertmanagerAlert] = Field(default_factory=list)
    groupLabels: Dict[str, str] = Field(default_factory=dict)
    commonLabels: Dict[str, str] = Field(default_factory=dict)
    commonAnnotations: Dict[str, str] = Field(default_factory=dict)
    externalURL: Optional[str] = None
    version: Optional[str] = None
    groupKey: Optional[str] = None


# ---------------------------------------------------------------------------
# Background orchestrator
# ---------------------------------------------------------------------------

def _run_optimization_sync(
    job_id: str,
    trigger_alert: str,
    base_prompt: str,
    scenarios: List[str],
    service_url: Optional[str],
):
    """Wrapper to run the async optimizer inside a background thread."""
    jobs_db[job_id]["status"] = JobStatus.RUNNING.value
    try:
        result = asyncio.run(
            run_prompt_optimization(
                job_id=job_id,
                trigger_alert=trigger_alert,
                base_prompt=base_prompt,
                scenarios=scenarios,
                service_url=service_url,
            )
        )
        jobs_db[job_id].update({
            "status": result["status"],
            "duration_s": result.get("duration_s"),
            "error": result.get("error"),
            "prompt_updated": result.get("prompt_updated", False),
            "cluster_applied": result.get("cluster_applied", False),
            "email_sent": result.get("email_sent", False),
            "optimized_prompt": result.get("optimized_prompt", ""),
            "winner": result.get("winner"),
            "all_candidates": result.get("all_candidates", []),
        })
    except Exception as exc:
        logger.exception(f"Background crash job_id={job_id}")
        jobs_db[job_id].update({
            "status": JobStatus.FAILED.value,
            "error": str(exc),
        })


def _create_job(trigger_alert: str) -> str:
    global latest_job_id
    job_id = str(uuid.uuid4())
    jobs_db[job_id] = {
        "job_id": job_id,
        "status": JobStatus.PENDING.value,
        "trigger_alert": trigger_alert,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "duration_s": None,
        "error": None,
        "prompt_updated": None,
        "cluster_applied": None,
        "email_sent": None,
        "optimized_prompt": None,
        "winner": None,
        "all_candidates": None,
    }
    latest_job_id = job_id
    return job_id


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/healthz", status_code=status.HTTP_200_OK)
async def healthz():
    """Kubernetes liveness & readiness probe."""
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.post("/webhook", status_code=status.HTTP_202_ACCEPTED)
async def alertmanager_webhook(
    payload: AlertmanagerWebhookPayload,
    background_tasks: BackgroundTasks,
):
    """Receive Alertmanager alert payloads and trigger optimization jobs.

    Only processes alerts with status=firing AND action=optimize_prompt label.
    Returns 202 immediately; optimization runs asynchronously.
    """
    logger.info(
        f"Received Alertmanager webhook: status={payload.status} "
        f"alerts={len(payload.alerts)}"
    )

    if payload.status == "resolved":
        logger.info("All alerts resolved — no optimization action required.")
        return {"status": "resolved", "message": "No action taken on resolved alerts."}

    triggered = []
    for alert in payload.alerts:
        if alert.status != "firing":
            continue

        # Only trigger when the alert carries action=optimize_prompt
        action = alert.labels.action or payload.commonLabels.get("action", "")
        if action != "optimize_prompt":
            logger.info(
                f"Skipping alert={alert.labels.alertname} (action={action!r} ≠ optimize_prompt)"
            )
            continue

        alert_name = alert.labels.alertname or "UnknownAlert"
        language = alert.labels.language or payload.commonLabels.get("language", "German")
        level = alert.labels.level or payload.commonLabels.get("level", "A2")

        logger.info(f"Triggering optimization for alert={alert_name} language={language} level={level}")

        job_id = _create_job(alert_name)
        background_tasks.add_task(
            _run_optimization_sync,
            job_id=job_id,
            trigger_alert=alert_name,
            base_prompt="",          # optimizer loads from file
            scenarios=[
                "ordering food at a restaurant",
                "asking for directions at the station",
                "shopping for clothes",
            ],
            service_url=None,
        )
        triggered.append(job_id)

    return {
        "status": "Accepted",
        "jobs_started": len(triggered),
        "job_ids": triggered,
    }


@app.post(
    "/optimize",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=JobResponse,
)
async def trigger_manual_optimization(
    request: OptimizeRequest,
    background_tasks: BackgroundTasks,
):
    """Manually trigger a prompt optimization run.

    Returns 202 immediately with job_id; use GET /jobs/{job_id} to poll.
    """
    job_id = _create_job(request.trigger_alert)
    background_tasks.add_task(
        _run_optimization_sync,
        job_id=job_id,
        trigger_alert=request.trigger_alert,
        base_prompt=request.base_prompt,
        scenarios=request.test_scenarios,
        service_url=request.service_url,
    )
    logger.info(f"Manual optimization queued job_id={job_id}")
    return jobs_db[job_id]


@app.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str):
    """Retrieve the status and results of an optimization job."""
    job = jobs_db.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return job


@app.get("/jobs/latest/info", response_model=JobResponse)
async def get_latest_job():
    """Retrieve the status and results of the most recent optimization job."""
    if not latest_job_id:
        raise HTTPException(status_code=404, detail="No jobs have been executed yet.")
    return jobs_db[latest_job_id]


@app.get("/jobs", response_model=List[JobResponse])
async def list_jobs():
    """List all known optimization jobs (newest first)."""
    return list(reversed(list(jobs_db.values())))


@app.get("/current-prompt")
async def get_current_prompt():
    """Return the current live prompt text.

    Any service (e.g. the inference pod) can GET this endpoint to fetch the
    latest winning prompt without needing a shared filesystem.

    Example:
        curl http://lang-learn-prompt-optimizer.lang-learn.svc.cluster.local:8000/current-prompt
    """
    from app.optimizer import CURRENT_PROMPT_FILE
    if CURRENT_PROMPT_FILE.exists():
        return {
            "prompt": CURRENT_PROMPT_FILE.read_text(encoding="utf-8"),
            "file": str(CURRENT_PROMPT_FILE),
        }
    return {"prompt": "", "file": str(CURRENT_PROMPT_FILE), "warning": "Prompt file not found."}
