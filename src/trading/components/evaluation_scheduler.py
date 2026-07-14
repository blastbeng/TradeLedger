import asyncio
import json
import logging
from typing import Dict, List

from src.config.settings import settings

logger = logging.getLogger(__name__)

class EvaluationScheduler:
    """Handles dynamic evaluation interval calculation and symbol selection."""

    def __init__(self, engine):
        self.engine = engine

    async def get_symbols_to_process(self, now: float) -> List[Dict[str, str]]:
        """Determine which symbols need evaluation this cycle based on market conditions and intervals."""
        engine = self.engine

        # Compute active period status once per loop iteration
        clock = await engine._market_data_manager.get_clock()
        is_active_period = False
        if clock and clock.is_open:
            now_rome = clock.timestamp
            market_open_dt = now_rome.replace(hour=settings.MARKET_OPEN_HOUR, minute=settings.MARKET_OPEN_MINUTE, second=0, microsecond=0)
            minutes_since_open = (now_rome - market_open_dt).total_seconds() / 60
            if 0 <= minutes_since_open < settings.MARKET_OPEN_ACTIVE_MINUTES:
                is_active_period = True
            if not is_active_period:
                market_close_dt = now_rome.replace(hour=settings.MARKET_CLOSE_HOUR, minute=settings.MARKET_CLOSE_MINUTE, second=0, microsecond=0)
                minutes_to_close = (market_close_dt - now_rome).total_seconds() / 60
                if 0 < minutes_to_close < settings.MARKET_CLOSE_ACTIVE_MINUTES:
                    is_active_period = True

        # Fetch full market breadth once per loop iteration
        full_market_breadth = None
        try:
            full_breadth_raw = await asyncio.to_thread(engine.redis.get, "market:breadth:full")
            if full_breadth_raw:
                full_market_breadth = json.loads(full_breadth_raw)
        except (ValueError, TypeError, ConnectionError, TimeoutError, OSError, json.JSONDecodeError):
            pass

        is_highly_active = False
        if full_market_breadth:
            pos_pct = full_market_breadth.get("positive_pct", 50)
            if pos_pct > 80 or pos_pct < 20:
                is_highly_active = True

        # Snapshot current_symbols to avoid race condition
        current_symbols_snapshot = list(engine.shared_state.current_symbols)

        # Check for significant news sentiment shifts for all tracked symbols
        symbol_has_significant_news: Dict[str, bool] = {}
        if settings.NEWS_ENABLED:
            for entry in current_symbols_snapshot:
                symbol = entry["symbol"]
                try:
                    agg = await engine._get_cached_sentiment(symbol)
                    if agg:
                        base_symbol = symbol.split("/")[0] if "/" in symbol else symbol
                        prev_key = f"sentiment:reeval_baseline:{base_symbol}"
                        prev_raw = await asyncio.to_thread(engine.redis.get, prev_key)
                        current_compound = agg.get("avg_compound", 0)
                        if prev_raw:
                            prev_compound = float(prev_raw)
                            if abs(current_compound - prev_compound) > 0.3:
                                symbol_has_significant_news[symbol] = True
                except (ValueError, TypeError, KeyError, ConnectionError, TimeoutError, OSError):
                    continue

        # Collect symbols that need evaluation this cycle
        symbols_to_process = []
        for symbol_entry in current_symbols_snapshot:
            symbol = symbol_entry["symbol"]
            tf = symbol_entry.get("timeframe", "1d")

            # Base interval proportional to timeframe
            if tf in ("1h",):
                tf_base_interval = settings.EVAL_INTERVAL_1H
            elif tf in ("1d",):
                tf_base_interval = settings.EVAL_INTERVAL_1D
            elif tf in ("1w",):
                tf_base_interval = settings.EVAL_INTERVAL_1W
            elif tf in ("1M",):
                tf_base_interval = settings.EVAL_INTERVAL_1M
            elif tf in ("3M",):
                tf_base_interval = settings.EVAL_INTERVAL_3M
            elif tf in ("6M", "1Y"):
                tf_base_interval = settings.EVAL_INTERVAL_6M_1Y
            else:
                tf_base_interval = settings.EVAL_INTERVAL_DEFAULT

            # Adjust based on market conditions
            if is_active_period or is_highly_active:
                # Active market: evaluate more frequently (halve the interval, min 15m)
                tf_base_interval = max(900, tf_base_interval // 2)

            if symbol_has_significant_news.get(symbol, False):
                # Significant news for this specific ticker: evaluate quickly (min 15m)
                tf_base_interval = max(900, min(tf_base_interval, 1800))

            if full_market_breadth and 40 <= full_market_breadth.get("positive_pct", 50) <= 60 and not is_active_period:
                # Quiet market: evaluate less frequently (double the interval, max 8h)
                tf_base_interval = max(tf_base_interval, min(tf_base_interval * 2, 28800))

            # Use the dynamically computed tf_base_interval, but allow LLM to override per-symbol
            default_interval = tf_base_interval
            async with engine._eval_state_lock:
                interval = engine._strategy_intervals.get(symbol, default_interval)
                last_eval = engine._last_strategy_eval.get(symbol, 0)
            if now - last_eval >= interval:
                symbols_to_process.append(symbol_entry)

        return symbols_to_process
