"""Tiny JSON-over-HTTP client with retries (stdlib only)."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request

USER_AGENT = "nba-prop-lakehouse/0.1 (portfolio data project)"


def get_json(
    url: str,
    params: dict | None = None,
    *,
    retries: int = 3,
    backoff: float = 2.0,
    timeout: float = 30.0,
) -> tuple[object, dict[str, str]]:
    """GET a URL and return (parsed JSON body, lower-cased response headers).

    Retries on network errors, 429 and 5xx. Other 4xx errors raise immediately.
    Error messages never include the URL, so API keys in query params don't leak
    into logs.
    """
    full_url = f"{url}?{urllib.parse.urlencode(params)}" if params else url
    request = urllib.request.Request(
        full_url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as resp:
                headers = {k.lower(): v for k, v in resp.headers.items()}
                return json.load(resp), headers
        except urllib.error.HTTPError as exc:
            retryable = exc.code == 429 or exc.code >= 500
            if not retryable or attempt == retries - 1:
                raise RuntimeError(f"HTTP {exc.code} from {urllib.parse.urlsplit(url).netloc}") from None
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == retries - 1:
                raise RuntimeError(f"Network error calling {urllib.parse.urlsplit(url).netloc}: {exc}") from None
        time.sleep(backoff * 2**attempt)
    raise AssertionError("unreachable")
