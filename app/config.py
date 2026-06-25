"""Configuration module for the Prompt Optimizer service.

Loads configuration from environment variables with sensible defaults for in-cluster
and local development environments.
"""

from __future__ import annotations

import os


class Config:
    """Application configuration options."""

    # Server binding options
    PORT: int = int(os.getenv("PORT", "8000"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # In-cluster BentoML/KServe inference service endpoint.
    # By default, uses the Kubernetes service name within the same namespace.
    INFERENCE_SERVICE_URL: str = os.getenv(
        "INFERENCE_SERVICE_URL",
        "http://lang-learn-inference-predictor.lang-learn.svc.cluster.local:80",
    )

    # MLflow Tracking Server URI for experiment logging.
    MLFLOW_TRACKING_URI: str = os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://mlflow-tracking.lang-learn.svc.cluster.local:5000",
    )


# Global config instance
settings = Config()
