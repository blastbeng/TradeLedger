"""Handles post-selection cleanup, backfill, and stale state pruning."""
import asyncio
import logging
import time
from typing import Any, Dict, List

from src.config.settings import settings

logger = logging.getLogger(__name__)


class ReevalPostSelectionManager:
    """Handles post-selection cleanup, backfill, and stale state pruning."""

    def __init__(self, engine, event_bus):
        self.engine = engine
        self.shared_state = engine.shared_state
        self.event_bus = event_bus

    def _create_background_task(self, coro_factory, task_name: str, max_retries: int = 3):
        """Wraps a coroutine factory in a task with error logging and retry."""
        async def _safe_run():
            for attempt in range(1, max_retries + 1):
                try:
                    await coro_factory()
                    return
                except Exception as e:
                    logger.exception(f"Error in background task {task_name} (attempt {attempt}/{max_retries}): {e}")
                    if attempt < max_retries:
                        await asyncio.sleep(2 ** (attempt - 1))
        return asyncio.create_task(_safe_run())

    async def post_selection_cleanup_and_backfill(
        self,
        old_symbols: List[Dict[str, str]],
        deduped: List[Dict[str, str]],
        force: bool,
    ) -> None:
        """Ensure open positions remain tracked, update tenure, and trigger backfill/news."""
        engine = self.engine

        # Ensure all open positions remain in current_symbols so they continue to be managed by the LLM strategy
        for symbol, pos in self.shared_state.positions.items():
            if not any(entry["symbol"] == symbol for entry in self.shared_state.current_symbols):
                tf = pos.get("timeframe") or (settings.OHLCV_TIMEFRAMES[0] if settings.OHLCV_TIMEFRAMES else "1h")
                async with self.shared_state._current_symbols_lock:
                    self.shared_state.current_symbols.append({"symbol": symbol, "timeframe": tf})
                logger.info(f"Keeping {symbol} in current_symbols due to open position (timeframe={tf})")

        # If trading is paused, we still keep all symbols so the LLM can generate signals
        # (which will be notified but not executed in paper mode).
        # The LLM may have just set pause_trading = true, so re-read Redis.
        paused_now = await asyncio.to_thread(engine.redis.get, "trading:paused")
        if paused_now and paused_now == "1" and not force:
            logger.info("Trading is paused. Keeping all symbols for signal generation.")

        # Update symbol tenure tracking
        now_ts = time.time()
        new_symbol_set = {entry["symbol"] for entry in self.shared_state.current_symbols}
        for sym in new_symbol_set:
            if sym not in self.shared_state._symbol_first_seen:
                self.shared_state._symbol_first_seen[sym] = now_ts
        for sym in list(self.shared_state._symbol_first_seen.keys()):
            if sym not in new_symbol_set:
                del self.shared_state._symbol_first_seen[sym]

        # Trigger immediate backfill for newly selected symbols
        old_symbol_set = {entry["symbol"] for entry in old_symbols}
        for entry in self.shared_state.current_symbols:
            if entry["symbol"] not in old_symbol_set:
                sym = entry["symbol"]
                tf = entry["timeframe"]
                logger.info(f"Triggering immediate backfill for newly selected symbol {sym} ({tf})")
                self._create_background_task(
                    lambda sym=sym, tf=tf: self.event_bus.publish("backfill_new_symbol", sym, tf),
                    f"backfill_new_symbol:{sym}"
                )

        # Also trigger immediate news fetch for newly selected symbols
        if settings.NEWS_ENABLED:
            for entry in deduped:
                sym = entry["symbol"]
                logger.info(f"Triggering immediate news fetch for newly selected symbol {sym}")
                self._create_background_task(
                    lambda sym=sym: engine._fetch_and_store_news_for_symbol(sym),
                    f"fetch_news:{sym}"
                )

    async def cleanup_stale_state_entries(self):
        """Remove stale entries from engine state dicts and base-symbol caches.

        Called at the end of each re-evaluation cycle to prune entries for
        symbols that are no longer tracked and have no open position.
        """
        engine = self.engine
        active_symbols = {entry["symbol"] for entry in self.shared_state.current_symbols}
        active_symbols.update(self.shared_state.positions.keys())
        async with self.shared_state._eval_state_lock:
            for state_dict in (
                self.shared_state._force_eval,
                self.shared_state._last_decisions,
                self.shared_state._entry_signal_state,
                self.shared_state._force_eval_time,
                self.shared_state._last_strategy_eval,
                self.shared_state._strategy_intervals,
                self.shared_state._last_eval_snapshot,
                self.shared_state.last_loss_time,
                self.shared_state.cooldown_durations,
            ):
                stale_keys = [s for s in state_dict if s not in active_symbols]
                for s in stale_keys:
                    state_dict.pop(s, None)
                if stale_keys:
                    logger.debug(f"Cleaned {len(stale_keys)} stale entries from engine state dicts")

        async with self.shared_state._pending_entries_lock:
            state_dict = self.shared_state._pending_entries
            stale_keys = [s for s in state_dict if s not in active_symbols]
            for s in stale_keys:
                state_dict.pop(s, None)
            if stale_keys:
                logger.debug(f"Cleaned {len(stale_keys)} stale entries from engine state dicts")

        active_bases = {s.split("/")[0] for s in active_symbols}
        for cache_dict in (
            self.shared_state._sentiment_cache,
            engine._asset_cache,
            engine._asset_cache_time,
        ):
            stale_keys = [k for k in cache_dict if k not in active_bases]
            for k in stale_keys:
                cache_dict.pop(k, None)
            if stale_keys:
                logger.debug(f"Cleaned {len(stale_keys)} stale entries from base-symbol caches")
