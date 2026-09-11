"""Centralized helpers for trading pause key management in Redis."""
import logging
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

# In-process fallback pause, used when Redis is unavailable so that the
# Redis-backed pause keys can be neither read nor written. The drawdown
# circuit breaker (and other fail-safe paths) set this to block new BUYs
# conservatively until Redis connectivity is restored.
_local_pause_reason: Optional[str] = None
_local_pause_lock = threading.Lock()

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


def set_local_pause(reason: str) -> bool:
    """Activate the in-process fallback pause (fail-closed when Redis is down).

    Returns True if this call transitioned the local pause from inactive to
    active (useful to send a one-shot notification without spamming).
    """
    global _local_pause_reason
    with _local_pause_lock:
        if _local_pause_reason is not None:
            return False
        _local_pause_reason = reason or "unknown"
    logger.critical(
        "Local fail-safe pause activated (reason=%s): new BUY decisions blocked "
        "until Redis connectivity is restored.",
        _local_pause_reason,
    )
    return True


def clear_local_pause() -> None:
    """Deactivate the in-process fallback pause."""
    global _local_pause_reason
    with _local_pause_lock:
        _local_pause_reason = None


def is_locally_paused() -> bool:
    """Return True if the in-process fallback pause is active."""
    return _local_pause_reason is not None


def get_local_pause_reason() -> Optional[str]:
    return _local_pause_reason
