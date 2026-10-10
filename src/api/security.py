"""API-key authentication and in-process rate limiting for the HTTP API."""

from __future__ import annotations

import math
import secrets
import time
from collections import deque
from collections.abc import Callable

from fastapi import Header, HTTPException, Request


class RateLimiter:
    """Sliding-window request counter, one window per client.

    State lives in this process only: it is not shared between workers and
    is reset on restart.
    """

    def __init__(
        self,
        limit: int,
        window_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._limit = limit
        self._window = window_seconds
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}

    def check(self, client_id: str) -> float | None:
        """Record a request for ``client_id``.

        Returns ``None`` if it is allowed, otherwise the number of seconds
        until the client may retry (the request is not recorded).
        """
        now = self._clock()
        cutoff = now - self._window
        for key in [k for k, hits in self._hits.items() if hits[-1] <= cutoff]:
            del self._hits[key]

        hits = self._hits.setdefault(client_id, deque())
        while hits and hits[0] <= cutoff:
            hits.popleft()
        if len(hits) >= self._limit:
            return hits[0] + self._window - now
        hits.append(now)
        return None


async def authorize(
    request: Request,
    x_api_key: str | None = Header(default=None),
) -> None:
    """Check the ``X-API-Key`` header, then apply the caller's rate limit.

    Keys are enforced whenever ``API_KEYS`` is configured. Startup validation
    guarantees that is always the case outside offline mode.
    """
    valid_keys = request.app.state.settings.api_key_set
    if valid_keys:
        supplied = (x_api_key or "").encode()
        # Compare against every key so timing does not reveal which one matched.
        matches = [secrets.compare_digest(supplied, key.encode()) for key in valid_keys]
        if not any(matches):
            raise HTTPException(status_code=401, detail="Invalid or missing API key.")
        client_id = f"key:{x_api_key}"
    else:
        client_id = f"ip:{request.client.host if request.client else 'unknown'}"

    retry_after = request.app.state.rate_limiter.check(client_id)
    if retry_after is not None:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded.",
            headers={"Retry-After": str(max(1, math.ceil(retry_after)))},
        )
