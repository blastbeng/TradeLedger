"""Market data management component for the TradingEngine.

Handles OHLCV downloads, gap filling, and indicator computation.
Extracted from TradingEngine to reduce class size and improve maintainability.
"""
import asyncio
import logging
import re
from typing import List

from src.config.settings import settings
from src.database import get_ohlcv, save_indicators, get_symbol_name_from_db, save_discovered_symbol
from src.exchanges.market_data import _check_yf_circuit, _get_yf_session
from src.indicators import compute_all_indicators

logger = logging.getLogger(__name__)


class MarketDataManager:
    """Handles market data downloads and indicator computation for the TradingEngine."""

    def __init__(self, engine):
        self.engine = engine

    async def get_stock_name(self, symbol: str) -> str:
        """Return the human-readable company name for a symbol, cached in Redis.

        Uses yfinance to fetch the name.
        """
        engine = self.engine
        base = symbol.split("/")[0] if "/" in symbol else symbol

        if re.match(r'^IT[A-Z0-9]{10}$', base):
            # It's a BTP bond, try to get the name from the BTP cache (includes DB-merged BTPs)
            try:
                btp_bonds = await engine._get_btp_bonds()
                for b in btp_bonds:
                    if b["isin"] == base:
                        name = b.get("name") or base
                        if name and name != b["isin"]:
                            return name
            except Exception:
                pass
            # Fallback: try DB directly
            try:
                db_name = await asyncio.to_thread(get_symbol_name_from_db, base)
                if db_name:
                    return db_name
            except Exception:
                pass

            # If we got a name from the BTP cache, save it to DB for future lookups
            try:
                btp_bonds = await engine._get_btp_bonds()
                for b in btp_bonds:
                    if b["isin"] == base:
                        name = b.get("name") or base
                        if name and name != b["isin"]:
                            await asyncio.to_thread(
                                save_discovered_symbol, base, base, "btp", name,
                                country="italy"
                            )
                            return name
            except Exception:
                pass
            return base

        # Check Redis cache first
        cache_key = f"stock_name:{base}"
        try:
            cached = await asyncio.to_thread(engine.redis.get, cache_key)
            if cached:
                return cached.decode() if isinstance(cached, bytes) else cached
        except Exception:
            pass

        # Check discovered_symbols table (works even when yf circuit is open)
        try:
            db_name = await asyncio.to_thread(get_symbol_name_from_db, base)
            if db_name:
                try:
                    await asyncio.to_thread(engine.redis.setex, cache_key, 7 * 24 * 3600, db_name)
                except Exception:
                    pass
                return db_name
        except Exception:
            pass

        if _check_yf_circuit():
            return base

        try:
            def _fetch_yf_name():
                import yfinance as yf
                ticker = yf.Ticker(base, session=_get_yf_session())
                info = ticker.info
                return info.get("longName") or info.get("shortName") or base
            name = await asyncio.to_thread(_fetch_yf_name)
        except Exception:
            name = base

        # If yfinance returned a name, save it to the DB for future use
        if name and name != base:
            try:
                db_base = base
                suffix = settings.TICKER_SUFFIX
                if suffix and db_base.endswith(suffix):
                    db_base = db_base[:-len(suffix)]
                save_discovered_symbol(db_base, None, None, name, country=None)
            except Exception:
                pass

        # Cache for 7 days (names rarely change)
        try:
            await asyncio.to_thread(engine.redis.setex, cache_key, 7 * 24 * 3600, name)
        except Exception:
            pass
        return name

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
