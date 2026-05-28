from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Any

import requests
from dotenv import load_dotenv
from tavily import TavilyClient
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)


FETCH_TIMEOUT_SECONDS = 10
FETCH_MAX_REDIRECTS = 3
FETCH_MAX_CHARS = 50_000


def _is_retryable_error(exc: BaseException) -> bool:
    if isinstance(exc, requests.RequestException):
        return True

    status_code = getattr(exc, "status_code", None)
    if status_code in {408, 409, 425, 429, 500, 502, 503, 504}:
        return True

    message = str(exc).lower()
    retryable_terms = (
        "rate limit",
        "ratelimit",
        "too many requests",
        "timeout",
        "timed out",
        "connection",
        "network",
        "temporarily unavailable",
        "service unavailable",
    )
    return any(term in message for term in retryable_terms)


def _log_tool_call(
    tool: str,
    key_arg: str,
    result_size: int,
    latency_ms: int,
) -> None:
    print(
        f"tool={tool} key_arg={key_arg!r} "
        f"result_size={result_size} latency_ms={latency_ms}"
    )


def _result_size_chars(payload: Any) -> int:
    return len(json.dumps(payload, ensure_ascii=True, default=str))


@retry(
    retry=retry_if_exception(_is_retryable_error),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)
def _tavily_search(query: str, max_results: int) -> dict[str, Any]:
    load_dotenv()
    api_key = os.environ["TAVILY_API_KEY"]
    client = TavilyClient(api_key=api_key)
    return client.search(query=query, max_results=max_results)


def web_search(query: str, max_results: int = 5) -> list[dict[str, str]]:
    start = time.perf_counter()
    results: list[dict[str, str]] = []

    try:
        response = _tavily_search(query=query, max_results=max_results)
        for item in response.get("results", []):
            results.append(
                {
                    "title": str(item.get("title", "")),
                    "url": str(item.get("url", "")),
                    "snippet": str(item.get("content", "")),
                }
            )
        return results
    finally:
        latency_ms = int((time.perf_counter() - start) * 1000)
        _log_tool_call(
            tool="web_search",
            key_arg=query,
            result_size=_result_size_chars(results),
            latency_ms=latency_ms,
        )


def fetch_url(url: str) -> dict[str, Any]:
    start = time.perf_counter()
    result: dict[str, Any]

    try:
        session = requests.Session()
        session.max_redirects = FETCH_MAX_REDIRECTS
        response = session.get(
            url,
            timeout=FETCH_TIMEOUT_SECONDS,
            allow_redirects=True,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 trigger-event-verification/1.0 "
                    "(compatible; research script)"
                )
            },
        )
        content = response.text[:FETCH_MAX_CHARS]
        result = {
            "url": response.url,
            "status": response.status_code,
            "content": content,
            "content_length": len(content),
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
        return result
    except Exception as exc:
        result = {
            "url": url,
            "status": "error",
            "error": str(exc),
            "content": "",
        }
        return result
    finally:
        latency_ms = int((time.perf_counter() - start) * 1000)
        result_size = len(result.get("content", "")) if "result" in locals() else 0
        _log_tool_call(
            tool="fetch_url",
            key_arg=url,
            result_size=result_size,
            latency_ms=latency_ms,
        )
