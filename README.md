# Lang-Learn Prompt Optimizer

An alert-driven prompt optimization service for the **Lang-Learn** German language learning platform. It benchmarks prompt variations against the inference service, tracks experiments in MLflow, selects the best-performing prompt, applies it to the cluster, and emails a report — all triggered automatically by Prometheus/Alertmanager alerts or manual API calls.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [How It Works](#how-it-works)
- [API Endpoints](#api-endpoints)
- [Candidate Strategies](#candidate-strategies)
- [Scoring Formula](#scoring-formula)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Deployment](#deployment)
  - [Step 1 — Deploy MLflow Infrastructure](#step-1--deploy-mlflow-infrastructure)
  - [Step 2 — Verify MLflow UI](#step-2--verify-mlflow-ui)
  - [Step 3 — Build & Push Prompt Optimizer Image](#step-3--build--push-prompt-optimizer-image)
  - [Step 4 — Deploy Prompt Optimizer](#step-4--deploy-prompt-optimizer)
  - [Step 5 — Verify Deployment](#step-5--verify-deployment)
- [Upgrading](#upgrading)
- [Configuration](#configuration)
- [Local Development](#local-development)
- [Troubleshooting](#troubleshooting)

---

## Architecture Overview

```
┌──────────────────┐        ┌─────────────────────────────┐
│   Alertmanager   │──POST──▶  Prompt Optimizer (FastAPI)  │
│   (webhook)      │  /webhook│                             │
└──────────────────┘        │  1. Load base prompt          │
                            │  2. Evaluate 9 candidates     │
                            │  3. Log to MLflow             │
                            │  4. Patch K8s ConfigMap       │
                            │  5. Email report              │
                            └──────────┬──────────┬─────────┘
                                       │          │
                            ┌──────────▼──┐  ┌────▼──────────┐
                            │  Inference  │  │  MLflow        │
                            │  Service    │  │  Tracking      │
                            │  (BentoML)  │  │  Server        │
                            └─────────────┘  │  + PostgreSQL  │
                                             └────────────────┘
```

## How It Works

1. **Trigger** — Alertmanager fires a webhook with `action: optimize_prompt`, or a user calls `POST /optimize`.
2. **Load** — The optimizer reads the current base prompt from the Kubernetes ConfigMap (or local file).
3. **Benchmark** — Each candidate prompt variation is sent to the inference service via the `prompt_template` parameter. The inference LLM generates dialogue using that exact prompt.
4. **Score** — Responses are evaluated on A1 vocabulary ratio, German language purity, dialogue turn density, generation latency, and B2-vocabulary penalty.
5. **Select** — Candidates are ranked by composite quality score. The winner must beat the current prompt by ≥ `MIN_IMPROVEMENT` (default `0.02`).
6. **Apply** — If a new winner is found, the optimizer patches the `lang-learn-prompts` ConfigMap and triggers an inference pod restart.
7. **Log** — All metrics and prompt artifacts are logged to MLflow for experiment tracking.
8. **Report** — An HTML email report is sent to the admin with a full breakdown of results.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/webhook` | Alertmanager webhook receiver. Filters for `action=optimize_prompt` alerts. Returns `202`. |
| `POST` | `/optimize` | Manual optimization trigger with custom parameters. Returns `202`. |
| `GET` | `/jobs/{job_id}` | Poll a specific job's status and results. |
| `GET` | `/jobs/latest/info` | Get the most recent job's result. |
| `GET` | `/jobs` | List all known optimization jobs (newest first). |
| `GET` | `/current-prompt` | Return the current live prompt text. |
| `GET` | `/healthz` | Kubernetes liveness & readiness probe. |

### Manual Trigger Example

```bash
curl -X POST http://localhost:8000/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "trigger_alert": "ManualTrigger",
    "test_scenarios": [
      "ordering food at a restaurant in Berlin",
      "booking a hotel room in Munich"
    ]
  }'
```

---

## Candidate Strategies

Each candidate targets specific dimensions of the scoring formula:

| Candidate | Target Dimension | Score Weight | Strategy |
|---|---|---|---|
| `current` | Baseline | — | Existing prompt, no modifications |
| `a1_strict_vocab` | A1 vocabulary | 40% | Forces top-500 most common German words |
| `a1_word_repetition` | A1 vocabulary | 40% | Natural repetition of high-frequency A1 words |
| `pure_german` | German ratio | 25% | Eliminates all English content, even filler |
| `high_turn_density` | Dialogue turns | 15% | 20+ turns per person, short Q&A exchanges |
| `micro_turns` | Dialogue turns | 15% | Ultra-short one-sentence turns, 30+ total |
| `b2_eliminator` | B2 penalty | -10% | Explicitly blacklists advanced vocabulary |
| `optimized_blend` | All dimensions | Combined | Balances A1 vocab + turns + German purity |
| `concise_fast` | Latency | 10% | Minimal output, lower max_tokens for speed |

---

## Scoring Formula

The composite quality score (0–1) is computed as:

```
Score = 0.40 × A1_vocab_scaled
      + 0.25 × German_ratio
      + 0.15 × Dialogue_turns_scaled
      + 0.10 × Latency_score
      - 0.10 × B2_ratio
```

Where:
- **A1 vocabulary** is scaled so 45% A1 ratio = perfect score
- **Dialogue turns** are capped at 15
- **Latency** penalises responses over 60 seconds

---

## Project Structure

```
lang-learn-prompt-optimizer/
├── app/
│   ├── main.py               # FastAPI application with all endpoints
│   ├── optimizer.py           # Top-level optimization orchestrator
│   ├── experiment_runner.py   # MLflow experiment runner & candidate definitions
│   ├── evaluate_prompts.py    # Scoring functions & inference service client
│   ├── cluster_apply.py       # K8s ConfigMap patching & pod restart
│   ├── email_reporter.py      # HTML email report generation
│   ├── config.py              # App configuration (env vars)
│   ├── utils.py               # HTTP helpers
│   └── vocab.py               # A1/A2/B2 German vocabulary lists
├── prompts/                   # Default prompt templates
├── metrics/                   # Metrics output directory
├── helm/                      # Helm chart for the prompt optimizer
│   ├── Chart.yaml
│   ├── values.yaml
│   └── templates/
│       ├── deployment.yaml
│       ├── service.yaml
│       ├── pvc.yaml
│       ├── rbac.yaml          # RBAC for ConfigMap patching
│       ├── hpa.yaml           # Horizontal Pod Autoscaler
│       └── serviceaccount.yaml
├── helm-mlflow/               # Helm chart for MLflow + PostgreSQL
│   ├── Chart.yaml
│   ├── values.yaml
│   └── templates/
│       ├── mlflow-deployment.yaml
│       ├── mlflow-service.yaml
│       ├── mlflow-pvc.yaml
│       ├── postgres-deployment.yaml
│       ├── postgres-service.yaml
│       ├── postgres-pvc.yaml
│       └── postgres-secret.yaml
├── tests/                     # Unit tests
├── Dockerfile
├── entrypoint.sh
├── requirements.txt
└── README.md
```

---

## Prerequisites

- **Kubernetes cluster** with `kubectl` configured (e.g. MicroK8s)
- **Helm 3** installed
- **Docker** for building and pushing images
- **Docker Hub** account (or other container registry)
- **Namespace** `lang-learn` created:
  ```bash
  kubectl create namespace lang-learn
  ```
- **SMTP secret** for email reports (Gmail app password):
  ```bash
  kubectl create secret generic alertmanager-smtp-secret \
    --from-literal=smtp-password='YOUR_APP_PASSWORD' \
    -n lang-learn
  ```

---

## Deployment

### Step 1 — Deploy MLflow Infrastructure

This installs PostgreSQL (backend store) + MLflow Tracking Server:

```bash
helm install mlflow-tracking helm-mlflow/ -n lang-learn
```

Wait for both pods to become ready:

```bash
kubectl get pods -n lang-learn -l app=mlflow-tracking
kubectl get pods -n lang-learn -l app=mlflow-postgres
```

Expected output (after ~60s for pip install inside the MLflow container):

```
NAME                                READY   STATUS    RESTARTS
mlflow-postgres-xxxxxxxxxx-xxxxx    1/1     Running   0
mlflow-tracking-xxxxxxxxxx-xxxxx    1/1     Running   0
```

### Step 2 — Verify MLflow UI

> **Note:** On macOS Monterey+, port 5000 is occupied by AirPlay Receiver. Use port 5002 instead.

```bash
kubectl port-forward svc/mlflow-tracking 5002:5000 -n lang-learn
```

Open [http://localhost:5002](http://localhost:5002) in your browser to confirm the MLflow UI loads.

### Step 3 — Build & Push Prompt Optimizer Image

```bash
docker build -t rickdevma100/lang-learn-prompt-optimizer:latest .
docker push rickdevma100/lang-learn-prompt-optimizer:latest
```

### Step 4 — Deploy Prompt Optimizer

**First-time install:**

```bash
helm install prompt-optimizer helm/ -n lang-learn
```

Verify the pod is running:

```bash
kubectl get pods -n lang-learn -l app=lang-learn-prompt-optimizer
```

### Step 5 — Verify Deployment

Check all services are up:

```bash
kubectl get pods -n lang-learn
```

Port-forward the optimizer to test locally:

```bash
kubectl port-forward svc/lang-learn-prompt-optimizer 8000:8000 -n lang-learn
```

Health check:

```bash
curl http://localhost:8000/healthz
# {"status":"ok","timestamp":"..."}
```

Trigger a manual optimization run:

```bash
curl -X POST http://localhost:8000/optimize \
  -H "Content-Type: application/json" \
  -d '{"trigger_alert": "ManualTest"}'
```

---

## Upgrading

After making code changes, rebuild and redeploy:

```bash
# Rebuild & push
docker build -t rickdevma100/lang-learn-prompt-optimizer:latest .
docker push rickdevma100/lang-learn-prompt-optimizer:latest

# Upgrade via Helm
helm upgrade prompt-optimizer helm/ -n lang-learn
```

To upgrade the MLflow infrastructure:

```bash
helm upgrade mlflow-tracking helm-mlflow/ -n lang-learn
```

---

## Configuration

All configuration is passed via environment variables (set in `helm/values.yaml`):

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_ROOT` | `/app` | Application root directory |
| `MIN_SCORE_IMPROVEMENT` | `0.02` | Minimum score delta to accept a new prompt |
| `INFERENCE_SERVICE_URL` | `http://lang-learn-inference-predictor...` | In-cluster inference service URL |
| `MLFLOW_TRACKING_URI` | `http://mlflow-tracking...` | MLflow tracking server URL |
| `LOG_LEVEL` | `INFO` | Logging level |
| `ADMIN_EMAIL` | `rickdev.ma100@gmail.com` | Email recipient for optimization reports |
| `SMTP_HOST` | `smtp.gmail.com` | SMTP server hostname |
| `SMTP_PORT` | `587` | SMTP server port |
| `SMTP_USER` | `rickdev.ma100@gmail.com` | SMTP login username |

---

## Local Development

```bash
# Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Start the server locally
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Troubleshooting

### MLflow pod is OOMKilled

The MLflow container runs `pip install` on startup. If memory is too low, it crashes with exit code `137`. Increase `mlflow.resources.limits.memory` in `helm-mlflow/values.yaml` (currently set to `1.5Gi`).

### Port 5000 returns HTTP 403 on macOS

macOS AirPlay Receiver occupies port 5000. Use a different local port:

```bash
kubectl port-forward svc/mlflow-tracking 5002:5000 -n lang-learn
```

### Prompt is not changing after optimization runs

Ensure both the **inference service** and **prompt optimizer** images are rebuilt and deployed. The inference service must support the `prompt_template` parameter so each candidate is evaluated with its own prompt variation (not the fixed ConfigMap template).

### Pods stuck in ImagePullBackOff

Verify the image was pushed to the registry:

```bash
docker push rickdevma100/lang-learn-prompt-optimizer:latest
```

If using a private registry, ensure an `imagePullSecret` is configured.