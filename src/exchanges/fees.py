import logging
from typing import Optional
import redis

from src.config.settings import settings

logger = logging.getLogger(__name__)

def get_fee_rate(
    exchange,  # ignored (kept for signature compatibility)
    symbol: str,
    redis_client: Optional[redis.Redis] = None,
    default: float = 0.0,
) -> float:
    """Return the taker fee rate.
    
    In paper mode, returns the configured PAPER_TRADING_FEE_PCT.
    In notify mode, returns 0.0 (no simulated fees — user tracks real fees manually).
    """
    if settings.TRADING_MODE == "paper":
        return settings.PAPER_TRADING_FEE_PCT
    return 0.0
