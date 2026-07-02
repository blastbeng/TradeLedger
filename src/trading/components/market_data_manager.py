"""Market data management component for the TradingEngine.

Handles OHLCV downloads, gap filling, and indicator computation.
Extracted from TradingEngine to reduce class size and improve maintainability.
"""
import asyncio
import logging
from typing import List

from src.config.settings import settings
from src.database import get_ohlcv, save_indicators
from src.indicators import compute_all_indicators

logger = logging.getLogger(__name__)


class MarketDataManager:
    """Handles market data downloads and indicator computation for the TradingEngine."""

    def __init__(self, engine):
        self.engine = engine

    async def compute_and_store_indicators(self, symbol: str, timeframe: str, candles: List[List]):
        """Compute indicators for a symbol/timeframe using TA-Lib and store in DB."""
        engine = self.engine
        if not candles or len(candles) < 2:
            return
        try:
            async with engine._indicator_semaphore:
                loop = asyncio.get_running_loop()
                ind = await loop.run_in_executor(engine._download_executor, compute_all_indicators, candles)
            if ind:
                latest_ts = candles[-1][0]
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(engine._db_executor, save_indicators, symbol, timeframe, latest_ts, ind)
                logger.debug(f"Indicators computed and stored for {symbol} {timeframe}")
        except Exception as e:
            logger.warning(f"Failed to compute/store indicators for {symbol} {timeframe}: {e}")
