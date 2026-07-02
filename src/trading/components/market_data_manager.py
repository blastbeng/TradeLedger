"""Market data management component for the TradingEngine.

Handles OHLCV downloads, gap filling, and indicator computation.
Extracted from TradingEngine to reduce class size and improve maintainability.
"""
import asyncio
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.config.settings import settings
from src.database import get_ohlcv, save_indicators, get_symbol_name_from_db, save_discovered_symbol
from src.exchanges.market_data import get_tradable_assets, discover_btp_bonds, discover_italian_ucits_etfs, _check_yf_circuit, _get_yf_session
from src.indicators import compute_all_indicators

logger = logging.getLogger(__name__)


@dataclass
class AssetInfo:
    """Asset info for yfinance-based trading (min order size, fractionability, etc.)."""
    name: str = ""
    min_order_size: Optional[float] = 0.0
    fractionable: bool = True


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

    async def get_tradable_assets(self) -> List[str]:
        """Return tradable assets, cached for 5 minutes to reduce API calls."""
        engine = self.engine
        now = time.time()
        if engine._tradable_assets_cache and (now - engine._tradable_assets_cache_time) < 300:
            return engine._tradable_assets_cache
        async with engine._tradable_assets_lock:
            # Double-check cache after acquiring lock (another task may have populated it)
            now = time.time()
            if engine._tradable_assets_cache and (now - engine._tradable_assets_cache_time) < 300:
                return engine._tradable_assets_cache
            assets = await asyncio.to_thread(get_tradable_assets)
            engine._tradable_assets_cache = assets
            engine._tradable_assets_cache_time = now
            return assets

    async def get_btp_bonds(self) -> List[Dict[str, Any]]:
        """Return BTP bonds, cached for 30 minutes to reduce scraping calls."""
        engine = self.engine
        now = time.time()
        if engine._btp_bonds_cache and (now - engine._btp_bonds_cache_time) < 1800:
            return engine._btp_bonds_cache
        async with engine._tradable_assets_lock:
            # Double-check after lock
            now = time.time()
            if engine._btp_bonds_cache and (now - engine._btp_bonds_cache_time) < 1800:
                return engine._btp_bonds_cache
            bonds = await asyncio.to_thread(discover_btp_bonds)
            # Merge with DB-saved BTPs so nothing is lost between runs
            try:
                from src.database import get_all_discovered_symbols
                db_symbols = await asyncio.to_thread(get_all_discovered_symbols)
                existing_isins = {b["isin"] for b in bonds}
                for db_entry in db_symbols:
                    if db_entry.get("asset_type") == "btp" and db_entry["symbol"] not in existing_isins:
                        bonds.append({
                            "isin": db_entry["symbol"],
                            "name": db_entry.get("name") or db_entry["symbol"],
                            "last_price": None,
                            "change_pct": 0.0,
                            "coupon": db_entry.get("coupon"),
                            "maturity": db_entry.get("maturity"),
                        })
                        existing_isins.add(db_entry["symbol"])
            except Exception as e:
                logger.warning(f"Failed to merge BTPs from DB: {e}")
            engine._btp_bonds_cache = bonds
            engine._btp_bonds_cache_time = now
            return bonds

    async def get_etf_symbols(self) -> List[str]:
        """Return Italian UCITS ETF symbols, cached for 1 hour."""
        engine = self.engine
        now = time.time()
        if engine._etf_symbols_cache and (now - engine._etf_symbols_cache_time) < 3600:
            return engine._etf_symbols_cache
        async with engine._tradable_assets_lock:
            # Double-check after lock
            now = time.time()
            if engine._etf_symbols_cache and (now - engine._etf_symbols_cache_time) < 3600:
                return engine._etf_symbols_cache
            symbols = await asyncio.to_thread(discover_italian_ucits_etfs)
            engine._etf_symbols_cache = symbols
            engine._etf_symbols_cache_time = now
            return symbols

    async def get_asset_info(self, symbol: str) -> Any:
        """Return asset info (min order size, name, etc.), cached for 1 hour.

        Fetches from yfinance (subject to circuit breaker) with database
        fallback for the name. Returns permissive defaults only when no
        data is available.
        """
        engine = self.engine
        base = symbol.split("/")[0] if "/" in symbol else symbol
        now = time.time()
        if base in engine._asset_cache and (now - engine._asset_cache_time.get(base, 0)) < 3600:
            return engine._asset_cache[base]

        name = base
        min_order_size: Optional[float] = None
        fractionable = True

        # Try yfinance first (subject to circuit breaker)
        if not _check_yf_circuit():
            try:
                def _fetch_yf_info():
                    import yfinance as yf
                    ticker = yf.Ticker(base, session=_get_yf_session())
                    return ticker.info
                info = await asyncio.to_thread(_fetch_yf_info)
                if info:
                    name = info.get("longName") or info.get("shortName") or base
                    raw_min = info.get("minimumOrderSize")
                    if raw_min is not None:
                        try:
                            min_order_size = float(raw_min)
                        except (TypeError, ValueError):
                            pass
                    raw_frac = info.get("fractionalTrading")
                    if raw_frac is not None:
                        fractionable = bool(raw_frac)
            except Exception as e:
                logger.debug(f"yfinance asset info fetch failed for {base}: {e}")

        # Database fallback for name
        if name == base:
            try:
                db_name = await asyncio.to_thread(get_symbol_name_from_db, base)
                if db_name:
                    name = db_name
            except Exception:
                pass

        # Default to permissive 0.0 when no minimum was found
        if min_order_size is None:
            min_order_size = 0.0

        asset = AssetInfo(name=name, min_order_size=min_order_size, fractionable=fractionable)
        engine._asset_cache[base] = asset
        engine._asset_cache_time[base] = now
        return asset

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
