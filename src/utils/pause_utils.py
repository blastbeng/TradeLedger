"""Centralized helpers for trading pause key management in Redis."""
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

PAUSE_KEYS = [
    "trading:paused",
    "trading:pause_source",
    "trading:pause_start",
    "trading:pause_duration",
    "trading:pause_reason",
    "trading:llm_pause_time",
]


def clear_trading_pause_keys(redis_client) -> None:
    """Delete all trading pause-related Redis keys."""
    for key in PAUSE_KEYS:
        redis_client.delete(key)


def set_trading_pause(
    redis_client,
    source: str,
    reason: Optional[str] = None,
    pause_duration: Optional[int] = None,
    set_pause_start: bool = True,
    set_llm_pause_time: bool = False,
    ttl: Optional[int] = None,
) -> None:
    """Set trading pause keys in Redis.

    Clears any existing pause keys first, then sets only the specified ones.

    Args:
        redis_client: Redis client instance.
        source: The pause source (e.g., "llm", "manual", "market_closed").
        reason: Optional human-readable reason for the pause.
        pause_duration: Optional pause duration in seconds.
        set_pause_start: If True (default), sets trading:pause_start to current time.
        set_llm_pause_time: If True, sets trading:llm_pause_time to current time.
        ttl: Optional TTL in seconds for all set keys. If None, keys persist indefinitely
             (except trading:pause_duration which always gets a 7-day TTL).
    """
    clear_trading_pause_keys(redis_client)

    now = time.time()

    def _set(key: str, value: str):
        if ttl is not None:
            redis_client.setex(key, ttl, value)
        else:
            redis_client.set(key, value)

    _set("trading:paused", "1")
    _set("trading:pause_source", source)

    if set_pause_start:
        _set("trading:pause_start", str(now))

    if pause_duration is not None:
        redis_client.setex("trading:pause_duration", 7 * 24 * 3600, str(int(pause_duration)))

    if reason:
        _set("trading:pause_reason", reason)

    if set_llm_pause_time:
        _set("trading:llm_pause_time", str(now))
