from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Deque

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse, Response

from wafer_vision_api.redis_client import get_redis_client
from wafer_vision_api.settings import Settings


class ApiKeyAndRateLimitMiddleware(BaseHTTPMiddleware):
    """Optional API key validation plus Redis-backed rate limiting.

    Redis is the default production limiter so multiple API replicas share one
    bucket. The in-memory fallback remains available for tests and laptop demos.
    """

    def __init__(self, app, settings: Settings):  # type: ignore[no-untyped-def]
        super().__init__(app)
        self.settings = settings
        self._events: dict[str, Deque[float]] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        path = request.url.path
        if not path.startswith(self.settings.api_prefix):
            return await call_next(request)

        public_paths = {f"{self.settings.api_prefix}/health", f"{self.settings.api_prefix}/model"}
        if self.settings.api_key and path not in public_paths:
            provided = request.headers.get("x-api-key") or request.headers.get("authorization", "").removeprefix("Bearer ").strip()
            if provided != self.settings.api_key:
                return JSONResponse(status_code=401, content={"detail": "Missing or invalid API key."})

        if self.settings.rate_limit_enabled:
            limited = self._rate_limit_key(request)
            try:
                allowed, retry_after = self._allow(limited, self._limit_for_path(path))
            except Exception:
                if self.settings.rate_limit_fail_open:
                    allowed, retry_after = self._allow_memory(limited, self._limit_for_path(path))
                else:
                    return JSONResponse(status_code=503, content={"detail": "Rate limiter unavailable."})
            if not allowed:
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded.", "retry_after_seconds": retry_after},
                    headers={"Retry-After": str(retry_after)},
                )

        return await call_next(request)

    def _rate_limit_key(self, request: Request) -> str:
        client = request.client.host if request.client else "unknown"
        api_key = request.headers.get("x-api-key") or "anonymous"
        path = request.url.path
        if "/simulator" in path:
            bucket = "simulator"
        elif "/predict" in path:
            bucket = "predict"
        else:
            bucket = "general"
        return f"{self.settings.redis_key_prefix}:rate:{bucket}:{api_key}:{client}"

    def _limit_for_path(self, path: str) -> int:
        if "/simulator" in path:
            return max(1, int(self.settings.simulator_rate_limit_requests))
        return max(1, int(self.settings.rate_limit_requests))

    def _allow(self, key: str, limit: int) -> tuple[bool, int]:
        if self.settings.normalized_rate_limit_backend == "redis":
            return self._allow_redis(key, limit)
        return self._allow_memory(key, limit)

    def _allow_redis(self, key: str, limit: int) -> tuple[bool, int]:
        now_ms = int(time.time() * 1000)
        window_ms = max(1, int(self.settings.rate_limit_window_seconds)) * 1000
        redis = get_redis_client(self.settings, decode_responses=True)
        member = f"{now_ms}-{id(self)}"
        pipe = redis.pipeline()
        pipe.zremrangebyscore(key, 0, now_ms - window_ms)
        pipe.zcard(key)
        pipe.zadd(key, {member: now_ms})
        pipe.expire(key, int(self.settings.rate_limit_window_seconds) + 2)
        _, count, _, _ = pipe.execute()
        if int(count) >= limit:
            redis.zrem(key, member)
            oldest = redis.zrange(key, 0, 0, withscores=True)
            retry_after = 1
            if oldest:
                retry_after = max(1, int(((oldest[0][1] + window_ms) - now_ms) / 1000))
            return False, retry_after
        return True, 0

    def _allow_memory(self, key: str, limit: int) -> tuple[bool, int]:
        now = time.monotonic()
        window = max(1, int(self.settings.rate_limit_window_seconds))
        queue = self._events[key]
        while queue and now - queue[0] > window:
            queue.popleft()
        if len(queue) >= limit:
            retry_after = max(1, int(window - (now - queue[0]))) if queue else window
            return False, retry_after
        queue.append(now)
        return True, 0
