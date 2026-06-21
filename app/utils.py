"""Utility functions for the prompt optimizer."""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

logger = logging.getLogger("prompt-optimizer.utils")


def post_json(url: str, payload: dict, timeout: int = 120) -> dict | None:
    """Send a POST request with JSON payload and return parsed JSON response.

    Returns None if any connection error, HTTP error, or parsing error occurs.
    """
    try:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw_data = resp.read().decode("utf-8")
            return json.loads(raw_data)
    except urllib.error.HTTPError as e:
        logger.warning("HTTP error calling %s: %s (code=%s)", url, e.reason, e.code)
    except urllib.error.URLError as e:
        logger.warning("URL/connection error calling %s: %s", url, e.reason)
    except TimeoutError as e:
        logger.warning("Timeout calling %s: %s", url, e)
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse JSON response from %s: %s", url, e)
    except Exception as e:
        logger.warning("Unexpected error calling %s: %s", url, e)
    return None
