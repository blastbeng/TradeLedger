import asyncio
import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


class UnifiedConfigService:
    """Unified configuration service for LLM-decided parameters in Redis."""

    def __init__(self, redis_client):
        self.redis = redis_client
        self._prefix = "trading:"
        self._cache: dict = {}  # key -> (value, timestamp)
        self._cache_ttl: float = 60.0  # 60 seconds

    async def get_config(self, key: str, default: Any = None) -> Any:
        """Get a config value from Redis, with a 60-second in-memory cache."""
        now = time.time()
        cached = self._cache.get(key)
        if cached is not None:
            val, ts = cached
            if now - ts < self._cache_ttl:
                return val

        redis_key = f"{self._prefix}{key}"
        try:
            val = await asyncio.to_thread(self.redis.get, redis_key)
            if val is not None:
                if isinstance(val, bytes):
                    val = val.decode('utf-8')
                self._cache[key] = (val, now)
                return val
        except Exception as e:
            logger.debug(f"UnifiedConfigService.get_config: failed to read '{key}' from Redis: {type(e).__name__}: {e}")
        return default

    async def set_llm_config(self, key: str, value: Any, ttl: int = 7 * 24 * 3600) -> None:
        """Store an LLM-decided parameter in Redis."""
        redis_key = f"{self._prefix}{key}"
        try:
            await asyncio.to_thread(self.redis.setex, redis_key, ttl, str(value))
            self._cache.pop(key, None)  # invalidate cache
        except Exception as e:
            logger.warning(f"Failed to set LLM config {key}: {type(e).__name__}: {e}")

    async def clear_llm_config(self, key: str) -> None:
        """Remove an LLM-decided override from Redis."""
        redis_key = f"{self._prefix}{key}"
        try:
            await asyncio.to_thread(self.redis.delete, redis_key)
            self._cache.pop(key, None)  # invalidate cache
        except Exception as e:
            logger.warning(f"Failed to clear LLM config {key}: {type(e).__name__}: {e}")
