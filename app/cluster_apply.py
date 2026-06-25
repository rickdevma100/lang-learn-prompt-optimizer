"""Cluster prompt applicator — pushes the winning prompt to the live inference service.

After the optimizer selects a winning prompt, this module:
  1. Patches the Kubernetes ConfigMap "prompts" (scenario_dialogue.txt key)
  2. Triggers a rolling restart of the inference deployment
  3. Returns success/failure status

Requires:
  - A ServiceAccount with RBAC permissions to patch ConfigMaps and Deployments
  - The `kubernetes` Python package
  - Running inside a Kubernetes pod (uses in-cluster config)
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("prompt-optimizer.cluster")

# Namespace — auto-detected from service account mount, fallback to env/default
_SA_NAMESPACE_FILE = Path("/var/run/secrets/kubernetes.io/serviceaccount/namespace")
NAMESPACE = os.getenv("NAMESPACE", "")
if not NAMESPACE and _SA_NAMESPACE_FILE.exists():
    try:
        NAMESPACE = _SA_NAMESPACE_FILE.read_text().strip()
    except Exception:
        pass
NAMESPACE = NAMESPACE or "lang-learn"

# ConfigMap name that holds the prompt templates
CONFIGMAP_NAME = os.getenv("PROMPT_CONFIGMAP_NAME", "prompts")

# Deployment name for the inference service (KServe raw deployment mode)
INFERENCE_DEPLOYMENT_NAME = os.getenv(
    "INFERENCE_DEPLOYMENT_NAME",
    "lang-learn-inference-predictor",
)


def _load_k8s_config():
    """Load Kubernetes client config (in-cluster or local kubeconfig)."""
    try:
        # pyrefly: ignore [missing-import]
        from kubernetes import config
        try:
            config.load_incluster_config()
            logger.info("Loaded in-cluster Kubernetes config.")
        except config.ConfigException:
            config.load_kube_config()
            logger.info("Loaded local kubeconfig (not running in-cluster).")
    except ImportError:
        logger.error(
            "kubernetes package not installed. "
            "Install with: pip install kubernetes"
        )
        raise


def patch_prompt_configmap(new_prompt: str) -> bool:
    """Patch the 'prompts' ConfigMap to update scenario_dialogue.txt.

    Only the scenario_dialogue.txt key is updated; all other keys
    (image_describe.txt, explain_word.txt) are preserved.

    Returns True on success, False on failure.
    """
    try:
        # pyrefly: ignore [missing-import]
        from kubernetes import client

        _load_k8s_config()
        v1 = client.CoreV1Api()

        # Read current ConfigMap to preserve other keys
        cm = v1.read_namespaced_config_map(CONFIGMAP_NAME, NAMESPACE)

        if cm.data is None:
            cm.data = {}

        old_prompt = cm.data.get("scenario_dialogue.txt", "")
        cm.data["scenario_dialogue.txt"] = new_prompt

        v1.patch_namespaced_config_map(CONFIGMAP_NAME, NAMESPACE, cm)

        logger.info(
            f"Patched ConfigMap '{CONFIGMAP_NAME}' in namespace '{NAMESPACE}'. "
            f"scenario_dialogue.txt updated ({len(old_prompt)} → {len(new_prompt)} chars)."
        )
        return True

    except ImportError:
        logger.error("kubernetes package not installed — cannot patch ConfigMap.")
        return False
    except Exception as e:
        logger.error(f"Failed to patch ConfigMap '{CONFIGMAP_NAME}': {e}")
        return False


def restart_inference_pods() -> bool:
    """Trigger a rolling restart of the inference deployment.

    Works by patching the pod template with a `kubectl.kubernetes.io/restartedAt`
    annotation, which forces Kubernetes to perform a rolling update — identical
    to running `kubectl rollout restart deployment/<name>`.

    Returns True on success, False on failure.
    """
    try:
        # pyrefly: ignore [missing-import]
        from kubernetes import client

        _load_k8s_config()
        apps_v1 = client.AppsV1Api()

        now = datetime.now(timezone.utc).isoformat()
        patch = {
            "spec": {
                "template": {
                    "metadata": {
                        "annotations": {
                            "kubectl.kubernetes.io/restartedAt": now,
                        }
                    }
                }
            }
        }

        apps_v1.patch_namespaced_deployment(
            INFERENCE_DEPLOYMENT_NAME, NAMESPACE, patch
        )

        logger.info(
            f"Triggered rolling restart of deployment '{INFERENCE_DEPLOYMENT_NAME}' "
            f"in namespace '{NAMESPACE}' at {now}."
        )
        return True

    except ImportError:
        logger.error("kubernetes package not installed — cannot restart deployment.")
        return False
    except Exception as e:
        logger.error(
            f"Failed to restart deployment '{INFERENCE_DEPLOYMENT_NAME}': {e}"
        )
        return False


def apply_prompt_to_cluster(new_prompt: str) -> bool:
    """Apply the optimized prompt to the live inference service.

    Orchestrates:
      1. Patch the 'prompts' ConfigMap with the new prompt text
      2. Trigger a rolling restart of the inference pods

    Returns True if BOTH steps succeed, False if either fails.
    A failure here does NOT fail the optimization job — the prompt is still
    saved locally and reported via email.
    """
    logger.info("Applying optimized prompt to cluster…")

    # Step 1: Patch ConfigMap
    cm_ok = patch_prompt_configmap(new_prompt)
    if not cm_ok:
        logger.warning(
            "ConfigMap patch failed — prompt NOT applied to cluster. "
            "Manual intervention required."
        )
        return False

    # Step 2: Restart inference pods to pick up the new ConfigMap
    restart_ok = restart_inference_pods()
    if not restart_ok:
        logger.warning(
            "ConfigMap was patched but pod restart failed. "
            "Pods will pick up the new prompt on next restart, "
            "or you can run: kubectl rollout restart deployment/"
            f"{INFERENCE_DEPLOYMENT_NAME} -n {NAMESPACE}"
        )
        return False

    logger.info("✅ Prompt successfully applied to cluster.")
    return True
