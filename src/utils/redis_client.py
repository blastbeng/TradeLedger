import redis
import logging
from src.config.settings import settings

logger = logging.getLogger(__name__)

_redis_client: redis.Redis = None
_redis_available = True

class DummyRedis:
    """A no-op Redis client used when Redis is unavailable."""
    def __getattr__(self, name):
        def method(*args, **kwargs):
            return None
        return method

def is_redis_available() -> bool:
    return _redis_available

def set_redis_available(status: bool):
    global _redis_available
    if not status and _redis_available:
        logger.critical("Redis connection lost. Degrading to no-cache mode.")
    _redis_available = status

def get_redis_client() -> redis.Redis:
    """Return a singleton Redis client configured from settings.

    A single shared connection pool (max 50 connections) is reused across
    all callers to avoid exhausting file descriptors and Redis's max client
    connections under heavy logging or concurrent API calls.
    """
    global _redis_client
    if not _redis_available:
        return DummyRedis()
    if _redis_client is None:
        _redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            decode_responses=True,
            socket_timeout=5,           # seconds – max time for any Redis command
            socket_connect_timeout=5,   # seconds – max time to establish connection
            max_connections=50,
        )
    return _redis_client

def check_redis_connection() -> bool:
    """Test if Redis is reachable. Returns True if successful, False otherwise."""
    global _redis_client, _redis_available
    try:
        if _redis_client is None:
            _redis_client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                decode_responses=True,
                socket_timeout=5,           # seconds – max time for any Redis command
                socket_connect_timeout=5,   # seconds – max time to establish connection
                max_connections=50,
            )
        _redis_client.ping()
        set_redis_available(True)
        return True
    except (redis.ConnectionError, redis.TimeoutError) as e:
        logger.critical("Redis connection failed: %s", e)
        set_redis_available(False)
        return False
