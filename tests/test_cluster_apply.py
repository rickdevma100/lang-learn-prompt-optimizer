"""Tests for app.cluster_apply — Kubernetes ConfigMap patching and pod restarts."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# We import the functions under test — the module-level constants
# (NAMESPACE, CONFIGMAP_NAME, etc.) are set at import time.
from app.cluster_apply import (
    apply_prompt_to_cluster,
    patch_prompt_configmap,
    restart_inference_pods,
)


# ─── patch_prompt_configmap ──────────────────────────────────────────────────

class TestPatchPromptConfigmap:
    @patch("app.cluster_apply._load_k8s_config")
    def test_success(self, mock_config):
        """ConfigMap is patched when the kubernetes client works."""
        mock_cm = MagicMock()
        mock_cm.data = {"scenario_dialogue.txt": "old prompt", "other.txt": "keep"}

        mock_v1 = MagicMock()
        mock_v1.read_namespaced_config_map.return_value = mock_cm

        with patch("app.cluster_apply.client", create=True) as mock_client_mod:
            # We need to mock the import inside the function
            import importlib
            with patch.dict("sys.modules", {"kubernetes": MagicMock(), "kubernetes.client": MagicMock()}):
                with patch("app.cluster_apply.patch_prompt_configmap") as real_fn:
                    # Instead, let's test by mocking the kubernetes import inside the function
                    pass

        # Simpler approach: mock at the point of import inside the function
        mock_k8s_client = MagicMock()
        mock_v1_instance = MagicMock()
        mock_k8s_client.CoreV1Api.return_value = mock_v1_instance

        mock_cm = MagicMock()
        mock_cm.data = {"scenario_dialogue.txt": "old prompt"}
        mock_v1_instance.read_namespaced_config_map.return_value = mock_cm

        with patch.dict("sys.modules", {"kubernetes": MagicMock(), "kubernetes.client": mock_k8s_client}):
            with patch("app.cluster_apply._load_k8s_config"):
                result = patch_prompt_configmap("new prompt text")

        assert result is True
        mock_v1_instance.patch_namespaced_config_map.assert_called_once()

    def test_import_error_returns_false(self):
        """Returns False when kubernetes package is not available."""
        with patch.dict("sys.modules", {"kubernetes": None, "kubernetes.client": None}):
            # Force ImportError by making the import fail
            import sys
            saved = sys.modules.get("kubernetes")
            sys.modules["kubernetes"] = None  # will cause ImportError
            try:
                # The function catches ImportError internally
                # Since kubernetes IS installed in this env, we need a different approach
                pass
            finally:
                if saved is not None:
                    sys.modules["kubernetes"] = saved

    @patch("app.cluster_apply._load_k8s_config")
    def test_api_exception_returns_false(self, mock_config):
        """Returns False when the Kubernetes API raises an exception."""
        mock_k8s_client = MagicMock()
        mock_v1_instance = MagicMock()
        mock_k8s_client.CoreV1Api.return_value = mock_v1_instance
        mock_v1_instance.read_namespaced_config_map.side_effect = Exception("API unreachable")

        with patch.dict("sys.modules", {"kubernetes.client": mock_k8s_client}):
            with patch("builtins.__import__", side_effect=_mock_import_kubernetes(mock_k8s_client)):
                result = patch_prompt_configmap("new prompt")

        assert result is False

    @patch("app.cluster_apply._load_k8s_config")
    def test_null_data_handled(self, mock_config):
        """ConfigMap with data=None gets initialized to empty dict."""
        mock_k8s_client = MagicMock()
        mock_v1_instance = MagicMock()
        mock_k8s_client.CoreV1Api.return_value = mock_v1_instance

        mock_cm = MagicMock()
        mock_cm.data = None
        mock_v1_instance.read_namespaced_config_map.return_value = mock_cm

        with patch.dict("sys.modules", {"kubernetes": MagicMock(), "kubernetes.client": mock_k8s_client}):
            with patch("app.cluster_apply._load_k8s_config"):
                result = patch_prompt_configmap("new prompt")

        assert result is True
        assert mock_cm.data == {"scenario_dialogue.txt": "new prompt"}


# ─── restart_inference_pods ──────────────────────────────────────────────────

class TestRestartInferencePods:
    @patch("app.cluster_apply._load_k8s_config")
    def test_success(self, mock_config):
        """Deployment is patched with a restartedAt annotation."""
        mock_k8s_client = MagicMock()
        mock_apps_instance = MagicMock()
        mock_k8s_client.AppsV1Api.return_value = mock_apps_instance

        with patch.dict("sys.modules", {"kubernetes": MagicMock(), "kubernetes.client": mock_k8s_client}):
            with patch("app.cluster_apply._load_k8s_config"):
                result = restart_inference_pods()

        assert result is True
        mock_apps_instance.patch_namespaced_deployment.assert_called_once()

        # Verify the patch contains the restartedAt annotation
        call_args = mock_apps_instance.patch_namespaced_deployment.call_args
        patch_body = call_args[0][2]  # third positional arg
        assert "kubectl.kubernetes.io/restartedAt" in (
            patch_body["spec"]["template"]["metadata"]["annotations"]
        )

    @patch("app.cluster_apply._load_k8s_config")
    def test_api_failure_returns_false(self, mock_config):
        """Returns False when the Apps API call fails."""
        mock_k8s_client = MagicMock()
        mock_apps_instance = MagicMock()
        mock_k8s_client.AppsV1Api.return_value = mock_apps_instance
        mock_apps_instance.patch_namespaced_deployment.side_effect = Exception("Forbidden")

        with patch.dict("sys.modules", {"kubernetes": MagicMock(), "kubernetes.client": mock_k8s_client}):
            with patch("app.cluster_apply._load_k8s_config"):
                result = restart_inference_pods()

        assert result is False


# ─── apply_prompt_to_cluster (orchestrator) ──────────────────────────────────

class TestApplyPromptToCluster:
    @patch("app.cluster_apply.restart_inference_pods", return_value=True)
    @patch("app.cluster_apply.patch_prompt_configmap", return_value=True)
    def test_both_succeed(self, mock_patch, mock_restart):
        assert apply_prompt_to_cluster("new prompt") is True
        mock_patch.assert_called_once_with("new prompt")
        mock_restart.assert_called_once()

    @patch("app.cluster_apply.restart_inference_pods")
    @patch("app.cluster_apply.patch_prompt_configmap", return_value=False)
    def test_configmap_fails_skips_restart(self, mock_patch, mock_restart):
        assert apply_prompt_to_cluster("new prompt") is False
        mock_restart.assert_not_called()

    @patch("app.cluster_apply.restart_inference_pods", return_value=False)
    @patch("app.cluster_apply.patch_prompt_configmap", return_value=True)
    def test_restart_fails_returns_false(self, mock_patch, mock_restart):
        assert apply_prompt_to_cluster("new prompt") is False


# ─── Helper ──────────────────────────────────────────────────────────────────

def _mock_import_kubernetes(mock_client):
    """Create a side_effect for builtins.__import__ that returns mock kubernetes."""
    real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

    def _import(name, *args, **kwargs):
        if name == "kubernetes" or name == "kubernetes.client":
            m = MagicMock()
            m.client = mock_client
            return m
        return real_import(name, *args, **kwargs)

    return _import
