import redis
import logging
from src.config.settings import settings

logger = logging.getLogger(__name__)

_redis_client: redis.Redis = None

def get_redis_client() -> redis.Redis:
    """Return a singleton Redis client configured from settings.

    A single shared connection pool (max 50 connections) is reused across
    all callers to avoid exhausting file descriptors and Redis's max client
    connections under heavy logging or concurrent API calls.
    """
    global _redis_client
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
    try:
        r = get_redis_client()
        r.ping()
        return True
    except redis.ConnectionError as e:
        logger.warning("Redis connection failed: %s", e)
        return False
