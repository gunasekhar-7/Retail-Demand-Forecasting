"""Request guards for the forecasting API.

The service is read-only and serves public retail data, so authentication is
optional. When `DEMAND_API_KEY` is set, model and forecast routes require a
matching `X-API-Key`; health and metrics stay open for orchestrators. Rate
limiting is a per-process fixed window: it protects a single worker from a
runaway client, not a fleet. Put a gateway in front for shared limits.
"""

from __future__ import annotations

import os
import secrets
import threading
import time
import uuid

from fastapi import Header, HTTPException, status

MAX_REQUEST_ID_LENGTH = 128
DEFAULT_MAX_BODY_BYTES = 65_536
DEFAULT_RATE_LIMIT = 0


def expected_api_key() -> str | None:
    """Return the configured shared key, or None when auth is disabled."""
    return os.environ.get("DEMAND_API_KEY") or None


def verify_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    """FastAPI dependency: enforce the shared key only when one is configured."""
    expected = expected_api_key()
    if expected is None:
        return
    if x_api_key is None or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )


def verify_api_keys(x_api_keys: str | None = Header(default=None, alias="X-API-Keys")) -> None:
    """FastAPI dependency: enforce one of multiple configured keys.

    Accepts a comma-separated list of keys in the header. When multiple keys are
    configured (via DEMAND_API_KEYS), any one of them grants access. This enables
    key rotation without downtime and per-tenant key policies.
    """
    expected_str = os.environ.get("DEMAND_API_KEYS", "")
    if not expected_str:
        return  # no key auth configured
    expected_keys = [k.strip() for k in expected_str.split(",") if k.strip()]
    if not expected_keys:
        return
    if x_api_keys is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    provided = [k.strip() for k in x_api_keys.split(",") if k.strip()]
    if provided not in expected_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )


def resolve_request_id(value: str | None) -> str:
    """Accept a caller request id when it is short and printable, else generate one."""
    if value is not None:
        candidate = value.strip()
        if 0 < len(candidate) <= MAX_REQUEST_ID_LENGTH and all(
            32 <= ord(char) < 127 for char in candidate
        ):
            return candidate
    return uuid.uuid4().hex


def max_body_bytes() -> int:
    raw = os.environ.get("DEMAND_MAX_BODY_BYTES", str(DEFAULT_MAX_BODY_BYTES))
    try:
        return max(0, int(raw))
    except ValueError:
        return DEFAULT_MAX_BODY_BYTES


class RateLimiter:
    """Fixed-window counter keyed by client address.

    Deliberately process-local and dependency-free: the API is normally behind a
    reverse proxy that owns shared rate limiting. This is a safety net for a
    single worker, and the README says so.
    """

    def __init__(self, limit_per_minute: int) -> None:
        self.limit_per_minute = max(0, limit_per_minute)
        self._hits: dict[str, tuple[int, int]] = {}
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls) -> RateLimiter:
        raw = os.environ.get("DEMAND_RATE_LIMIT_PER_MINUTE", str(DEFAULT_RATE_LIMIT))
        try:
            limit = max(0, int(raw))
        except ValueError:
            limit = DEFAULT_RATE_LIMIT
        return cls(limit)

    def allow(self, key: str, now: float | None = None) -> bool:
        if self.limit_per_minute <= 0:
            return True
        window = int((time.monotonic() if now is None else now) // 60)
        with self._lock:
            current_window, count = self._hits.get(key, (window, 0))
            count = count + 1 if current_window == window else 1
            self._hits[key] = (window, count)
            if len(self._hits) > 4_096:
                self._hits = {
                    address: slot for address, slot in self._hits.items() if slot[0] == window
                }
            return count <= self.limit_per_minute
