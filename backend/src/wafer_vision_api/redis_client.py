from __future__ import annotations

from functools import lru_cache
from typing import Any

from wafer_vision_api.settings import Settings, get_settings


class RedisUnavailable(RuntimeError):
    pass


@lru_cache(maxsize=16)
def _redis_from_url(url: str, decode_responses: bool, socket_timeout: float, health_check_interval: int):
    try:
        import redis
    except ImportError as exc:  # pragma: no cover
        raise RedisUnavailable("Install redis to use Redis-backed jobs or rate limiting.") from exc
    return redis.Redis.from_url(
        url,
        decode_responses=decode_responses,
        socket_timeout=socket_timeout,
        socket_connect_timeout=socket_timeout,
        health_check_interval=health_check_interval,
    )


def get_redis_client(settings: Settings | None = None, *, decode_responses: bool = True):
    settings = settings or get_settings()
    return _redis_from_url(
        settings.redis_url,
        bool(decode_responses),
        float(settings.redis_socket_timeout_seconds),
        int(settings.redis_health_check_interval_seconds),
    )


def get_rq_redis_connection(settings: Settings | None = None):
    return get_redis_client(settings, decode_responses=False)


def ping_redis(settings: Settings | None = None) -> bool:
    try:
        return bool(get_redis_client(settings, decode_responses=True).ping())
    except Exception:
        return False


def reset_redis_client_for_tests() -> None:
    _redis_from_url.cache_clear()
