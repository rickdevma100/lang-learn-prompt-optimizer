"""Tests for app.utils — HTTP utility functions."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from app.utils import post_json


class TestPostJson:
    @patch("app.utils.urllib.request.urlopen")
    def test_success(self, mock_urlopen):
        """Returns parsed JSON on a successful POST."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"result": "ok"}).encode("utf-8")
        mock_response.__enter__ = lambda s: mock_response
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        result = post_json("http://example.com/api", {"key": "value"})

        assert result == {"result": "ok"}
        mock_urlopen.assert_called_once()

    @patch("app.utils.urllib.request.urlopen")
    def test_http_error_returns_none(self, mock_urlopen):
        """Returns None on HTTP error."""
        import urllib.error
        mock_urlopen.side_effect = urllib.error.HTTPError(
            "http://example.com", 500, "Server Error", {}, None
        )
        result = post_json("http://example.com/api", {"key": "value"})
        assert result is None

    @patch("app.utils.urllib.request.urlopen")
    def test_url_error_returns_none(self, mock_urlopen):
        """Returns None on connection error."""
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")
        result = post_json("http://unreachable.local/api", {})
        assert result is None

    @patch("app.utils.urllib.request.urlopen")
    def test_timeout_returns_none(self, mock_urlopen):
        """Returns None on timeout."""
        mock_urlopen.side_effect = TimeoutError("timed out")
        result = post_json("http://slow.local/api", {})
        assert result is None

    @patch("app.utils.urllib.request.urlopen")
    def test_invalid_json_returns_none(self, mock_urlopen):
        """Returns None when response body is not valid JSON."""
        mock_response = MagicMock()
        mock_response.read.return_value = b"not json at all"
        mock_response.__enter__ = lambda s: mock_response
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        result = post_json("http://example.com/api", {"key": "value"})
        assert result is None

    @patch("app.utils.urllib.request.urlopen")
    def test_unexpected_error_returns_none(self, mock_urlopen):
        """Returns None on any unexpected exception."""
        mock_urlopen.side_effect = RuntimeError("something went wrong")
        result = post_json("http://example.com/api", {})
        assert result is None
