import asyncio
import json
import logging
import random
import time
from typing import Optional, Tuple, List

from src.config.settings import settings
from src.database import get_latest_close_prices, store_news_articles, cleanup_old_news, get_latest_ohlcv_timestamps_batch, cleanup_old_position_pnl, cleanup_old_backtest_results, insert_dividend, cleanup_old_dividends, get_pending_dividends_for_symbol, mark_dividend_reinvested, get_pending_llm_decisions, update_llm_decision_outcome, get_llm_decision_quality_metrics, cleanup_old_llm_decisions
from src.exchanges.market_data import get_quotes_cached
from src.exchanges.yahoo_finance import get_yahoo_dividends
from src.utils.symbol_utils import is_btp_isin
from src.llm.cache import _should_use_primary_model
from src.utils.health_metrics import health_metrics
from src.utils.redis_client import is_redis_available, check_redis_connection
from src.trading.engine_utils import timeframe_to_seconds, timeframe_to_ms, get_effective_refresh_interval

try:
    from src.news.fetcher import fetch_news_for_symbol, discover_trending_stocks, discover_tickers_from_news
except ImportError:
    fetch_news_for_symbol = None
    discover_trending_stocks = None
    discover_tickers_from_news = None

logger = logging.getLogger(__name__)


class BackgroundTaskManager:
    def __init__(self, engine, event_bus):
        self.engine = engine
        self.event_bus = event_bus

    async def _periodic_reconcile(self):
        """Run position reconciliation every 5 minutes (medium/long-term)."""
        while self.engine._running:
            if self.engine._reconcile_running:
                logger.warning("Reconcile still running; skipping this cycle.")
                await asyncio.sleep(60)
                continue
            self.engine._reconcile_running = True
            try:
                await self.event_bus.request("reconcile_positions")
            except asyncio.CancelledError:
                raise
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.warning(f"Reconcile network/IO error: {type(e).__name__}: {e}")
            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, json.JSONDecodeError) as e:
                logger.error(f"Reconcile data/logic error: {type(e).__name__}: {e}", exc_info=True)
                await self.engine._record_unexpected_exception("reconcile", e)
            finally:
                self.engine._reconcile_running = False
            await asyncio.sleep(300)

    async def _periodic_reevaluate(self):
        """Re-evaluate stock selection periodically."""
        # Initial delay to allow WebSocket and Telegram bot to initialize
        logger.info(
            f"Waiting {settings.INITIAL_EVALUATION_DELAY_SECONDS}s before initial symbol evaluation..."
        )
        await asyncio.sleep(settings.INITIAL_EVALUATION_DELAY_SECONDS)
        while self.engine._running:
            if self.engine._reevaluate_running:
                # Wait briefly for the current re-evaluation to finish.
                # Use a short sleep so queued triggers are picked up quickly.
                await asyncio.sleep(settings.SYMBOL_EVALUATION_DELAY_SECONDS)
                continue

            # Check if market is open or in pre-market (1 hour before open)
            is_open = await self.engine._is_market_open()
            is_premarket = not is_open and _should_use_primary_model()
            is_forced = self.engine._force_reeval or self.engine._reeval_pending_force

            # Disable automatic re-evaluation when market is closed (outside pre-market)
            if not is_open and not is_premarket and not is_forced:
                logger.info("Market is closed; skipping automatic symbol re-evaluation.")
                await asyncio.sleep(300)  # Wait 5 minutes before checking again
                continue

            # Clear the trigger before starting re-evaluation so that any
            # trigger set DURING re-evaluation is caught in the next wait.
            self.engine._reeval_trigger.clear()
            self.engine._reevaluate_running = True
            try:
                # Always run re-evaluation, even if paused, to keep generating signals
                reeval_start_time = time.time()
                logger.info("Starting symbol re-evaluation...")
                # Force re-evaluation during pre-market to use main models
                is_forced = self.engine._force_reeval or self.engine._reeval_pending_force or is_premarket
                self.engine._force_reeval = False
                self.engine._reeval_pending_force = False
                async with self.engine._symbol_reeval_lock:
                    start_time = time.time()
                    await self.event_bus.request("reevaluate_symbols_impl", force=is_forced)
                    health_metrics.record_loop_latency("reevaluate", time.time() - start_time)
                elapsed = time.time() - reeval_start_time
                logger.info(f"Symbol re-evaluation complete (took {elapsed:.1f}s).")
            except asyncio.CancelledError:
                raise
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.warning(f"Stock re-evaluation network/IO error: {type(e).__name__}: {e}")
            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, json.JSONDecodeError) as e:
                logger.error(f"Stock re-evaluation data/logic error: {type(e).__name__}: {e}", exc_info=True)
                await self.engine._record_unexpected_exception("reevaluate", e)
                if self.engine.notifier:
                    await self.engine.notifier.send_notification(
                        f"❌ Stock re-evaluation failed: {str(e)[:200]}",
                        summary={
                            "action": "ERROR",
                            "reason": f"Re-evaluation error: {str(e)[:200]}",
                        }
                    )
            finally:
                self.engine._reevaluate_running = False
            self.engine._settings_reload_event.clear()
            wait_task = asyncio.create_task(self.engine._reeval_trigger.wait())
            reload_task = asyncio.create_task(self.engine._settings_reload_event.wait())
            await asyncio.wait(
                [wait_task, reload_task],
                timeout=settings.SYMBOL_REEVALUATION_INTERVAL,
                return_when=asyncio.FIRST_COMPLETED
            )
            for task in (wait_task, reload_task):
                if not task.done():
                    try:
                        task.cancel()
                    except asyncio.CancelledError:
                        pass
            # Don't clear _reeval_trigger here — it will be cleared at the
            # start of the next iteration, before re-evaluation begins.
            # This prevents losing a trigger that was set during the wait.

    async def _clear_pause_and_resume(self, reason: str, notification_msg: str, notification_summary: dict) -> None:
        """Helper to clear pause keys, set resume cooldown, and notify."""
        from src.utils.pause_utils import clear_trading_pause_keys
        await asyncio.to_thread(clear_trading_pause_keys, self.engine.redis)
        self.engine._reeval_trigger.set()
        await asyncio.to_thread(self.engine.redis.set, "trading:last_auto_resume", str(time.time()))
        await asyncio.to_thread(self.engine.redis.setex, "trading:auto_resume_cooldown", 600, "1")
        if self.engine.notifier:
            await self.engine.notifier.send_notification(notification_msg, summary=notification_summary)

    async def _handle_missing_pause_duration(self, pause_start_raw: Optional[bytes]) -> Tuple[bool, bool]:
        """Handle fallback when no pause_duration was set.
        
        Returns a tuple (skip_normal_logic, resumed):
        - skip_normal_logic: True if the caller should skip normal duration logic.
        - resumed: True if trading was actually resumed.
        """
        default_max_pause = settings.MIN_LLM_PAUSE_DURATION
        try:
            raw = await self.engine.config_service.get_config("min_llm_pause_duration")
            if raw:
                default_max_pause = int(raw)
        except (ValueError, TypeError, ConnectionError, TimeoutError, OSError):
            pass

        if pause_start_raw is None:
            logger.warning("Pause has no duration and no start time; forcing auto-resume immediately.")
            await self._clear_pause_and_resume(
                "Fallback: no pause start time",
                "⏰ Trading auto-resumed (pause had no duration and no start time).",
                {"action": "RESUME", "reason": "Fallback: no pause start time"}
            )
            return True, True

        try:
            elapsed = time.time() - float(pause_start_raw)
            if elapsed >= default_max_pause:
                logger.warning(f"Pause has no duration; forcing auto-resume after default fallback ({default_max_pause // 60} minutes).")
                await self._clear_pause_and_resume(
                    "Fallback pause timeout",
                    "⏰ Trading auto‑resumed after maximum pause duration (no LLM‑set duration).",
                    {"action": "RESUME", "reason": "Fallback pause timeout"}
                )
                return True, True
        except (ValueError, TypeError):
            pass
        
        # Did not resume, but we still need to skip normal duration logic
        return True, False

    async def _handle_pause_duration_elapsed(self, pause_start_raw: bytes, pause_duration_raw: bytes) -> None:
        """Check if the pause duration has elapsed and resume if so."""
        try:
            pause_start = float(pause_start_raw)
            pause_duration = int(pause_duration_raw)
            if time.time() - pause_start >= pause_duration:
                logger.info("Pause duration elapsed – auto-resuming trading.")
                stored_reason_raw = await asyncio.to_thread(self.engine.redis.get, "trading:pause_reason")
                stored_reason = stored_reason_raw.decode() if isinstance(stored_reason_raw, bytes) else (stored_reason_raw or "")
                reason_text = f" (was paused: {stored_reason})" if stored_reason else ""
                await self._clear_pause_and_resume(
                    f"Pause duration elapsed{reason_text}",
                    f"▶️ Trading auto-resumed after pause duration elapsed.{reason_text}",
                    {"action": "RESUME", "reason": f"Pause duration elapsed{reason_text}"}
                )
        except (ValueError, TypeError):
            pass  # ignore malformed values

    async def _periodic_pause_check(self):
        """Check and handle auto-resume from pause (only for LLM-initiated pauses)."""
        while self.engine._running:
            if self.engine._pause_check_running:
                logger.warning("Pause check still running; skipping this cycle.")
                await asyncio.sleep(30)
                continue
            self.engine._pause_check_running = True
            try:
                paused = await asyncio.to_thread(self.engine.redis.get, "trading:paused")
                if paused:
                    # Only auto-resume if the pause was initiated by the LLM
                    source = await asyncio.to_thread(self.engine.redis.get, "trading:pause_source")
                    if source and (source.decode() if isinstance(source, bytes) else source) == "llm":
                        pause_start_raw = await asyncio.to_thread(self.engine.redis.get, "trading:pause_start")
                        pause_duration_raw = await asyncio.to_thread(self.engine.redis.get, "trading:pause_duration")

                        # --- Fallback if no pause_duration was set ---
                        if not pause_duration_raw:
                            skip_normal, _resumed = await self._handle_missing_pause_duration(pause_start_raw)
                            if skip_normal:
                                await asyncio.sleep(30)
                                continue   # skip the original duration logic, proceed to next loop iteration

                        if pause_start_raw and pause_duration_raw:
                            await self._handle_pause_duration_elapsed(pause_start_raw, pause_duration_raw)
            except asyncio.CancelledError:
                raise
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.warning(f"Pause check network/IO error: {type(e).__name__}: {e}")
            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, json.JSONDecodeError) as e:
                logger.error(f"Pause check data/logic error: {type(e).__name__}: {e}", exc_info=True)
                await self.engine._record_unexpected_exception("pause_check", e)
            finally:
                self.engine._pause_check_running = False
            await asyncio.sleep(30)

    async def _periodic_pause_resume_check(self):
        """Periodically ask the LLM whether to resume trading when paused."""
        await asyncio.sleep(60)  # initial delay
        while self.engine._running:
            try:
                # Skip LLM pause/resume check if auto-resume cooldown is active
                cooldown_active = await asyncio.to_thread(self.engine.redis.get, "trading:auto_resume_cooldown")
                if cooldown_active:
                    logger.debug("Skipping LLM pause/resume check – auto-resume cooldown is active.")
                    await asyncio.sleep(1800)
                    continue

                # Skip LLM pause/resume check if duration-based auto-resume is imminent
                pause_duration_raw = await asyncio.to_thread(self.engine.redis.get, "trading:pause_duration")
                pause_start_raw = await asyncio.to_thread(self.engine.redis.get, "trading:pause_start")
                if pause_duration_raw and pause_start_raw:
                    try:
                        pause_start = float(pause_start_raw)
                        pause_duration = int(pause_duration_raw)
                        remaining = (pause_start + pause_duration) - time.time()
                        if remaining < 120:  # less than 2 minutes until auto-resume
                            logger.debug("Skipping LLM pause/resume check – duration-based auto-resume imminent.")
                            await asyncio.sleep(1800)
                            continue
                    except (ValueError, TypeError):
                        pass

                if await self.engine._is_market_open():
                    await self.engine.event_bus.request("check_pause_resume_decision")
            except asyncio.CancelledError:
                raise
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.warning(f"Pause/resume check network/IO error: {type(e).__name__}: {e}")
            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, json.JSONDecodeError) as e:
                logger.error(f"Pause/resume check data/logic error: {type(e).__name__}: {e}", exc_info=True)
                await self.engine._record_unexpected_exception("pause_resume_check", e)
            await asyncio.sleep(1800)  # every 30 minutes

    async def _periodic_full_market_breadth(self):
        """Periodically compute market breadth over all available pairs.

        Uses cached quotes (Redis/DB only, no network calls) to avoid
        thread pool exhaustion. Falls back to DB close prices for symbols
        without cached quotes. Uses a random sample of up to 500 symbols
        when the universe is larger, ensuring a representative sample.
        """
        await asyncio.sleep(60)  # initial delay
        while self.engine._running:
            if self.engine._full_breadth_running:
                logger.warning("Full market breadth computation still running; skipping this cycle.")
                await asyncio.sleep(300)
                continue
            self.engine._full_breadth_running = True
            try:
                # Fetch all asset types for stratified sampling
                stock_assets = await self.engine.event_bus.request("get_tradable_assets")
                stock_pairs = [f"{sym}/{self.engine.base_currency}" for sym in stock_assets]
                etf_symbols = await self.engine.event_bus.request("get_etf_symbols")
                etf_pairs = [f"{sym}/{self.engine.base_currency}" for sym in etf_symbols]
                btp_bonds = await self.engine.event_bus.request("get_btp_bonds")
                btp_pairs = [f"{b['isin']}/{self.engine.base_currency}" for b in btp_bonds]

                # Build strata: (pairs, label) for each asset type
                strata = [
                    (stock_pairs, "stocks"),
                    (etf_pairs, "etfs"),
                    (btp_pairs, "btps"),
                ]
                # Filter out empty strata
                strata = [(pairs, label) for pairs, label in strata if pairs]

                available_pairs = stock_pairs + etf_pairs + btp_pairs
                if available_pairs:
                    MAX_BREADTH_SAMPLE = 200
                    if len(available_pairs) <= MAX_BREADTH_SAMPLE:
                        # Universe is small enough — use everything
                        breadth_pairs = available_pairs
                    else:
                        # Proportional stratified sampling across asset types
                        total_universe = len(available_pairs)
                        breadth_pairs = []
                        for pairs, label in strata:
                            # Proportional allocation: stratum_size / total * MAX_SAMPLE
                            stratum_sample_size = max(1, round(len(pairs) / total_universe * MAX_BREADTH_SAMPLE))
                            # Cap at the stratum's actual size
                            stratum_sample_size = min(stratum_sample_size, len(pairs))
                            sampled = random.sample(pairs, stratum_sample_size)
                            breadth_pairs.extend(sampled)
                            logger.debug(
                                f"Breadth stratum '{label}': {len(pairs)} total, "
                                f"sampled {len(sampled)}"
                            )
                        # If rounding caused us to exceed the cap, trim randomly
                        if len(breadth_pairs) > MAX_BREADTH_SAMPLE:
                            breadth_pairs = random.sample(breadth_pairs, MAX_BREADTH_SAMPLE)
                    plain_breadth = [s.split("/")[0] for s in breadth_pairs]

                    # Use cached quotes (Redis/DB only, no network calls)
                    loop = asyncio.get_running_loop()
                    raw_breadth = await loop.run_in_executor(self.engine._quote_executor, get_quotes_cached, plain_breadth)
                    breadth_tickers = {pair: raw_breadth.get(pair.split("/")[0], {}) for pair in breadth_pairs}

                    # Fall back to DB close prices for symbols without cached quotes
                    missing_breadth = [
                        s.split("/")[0] for s in breadth_pairs
                        if breadth_tickers.get(s, {}).get('percentage') is None
                    ]
                    if missing_breadth:
                        try:
                            db_candles = await loop.run_in_executor(self.engine._db_executor, get_latest_close_prices, missing_breadth)
                            for pair in breadth_pairs:
                                base = pair.split("/")[0]
                                if base in db_candles and db_candles[base].get("last", 0) > 0:
                                    last = db_candles[base]["last"]
                                    prev_close = db_candles[base].get("prev_close")
                                    if prev_close and prev_close > 0:
                                        pct = ((last - prev_close) / prev_close) * 100
                                        breadth_tickers[pair] = {
                                            "last": last,
                                            "percentage": round(pct, 4),
                                        }
                        except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, json.JSONDecodeError, ConnectionError, TimeoutError, OSError) as e:
                            logger.warning(f"DB close price fallback for breadth failed: {e}")

                    positive_count = sum(
                        1 for sym in breadth_pairs
                        if (breadth_tickers.get(sym, {}).get('percentage') or 0) > 0
                    )
                    total_count = len(breadth_pairs)
                    full_market_breadth = {
                        "positive_pct": round(positive_count / total_count * 100, 1) if total_count > 0 else 0.0,
                        "positive_count": positive_count,
                        "total_count": total_count,
                        "universe_size": len(available_pairs),
                    }
                    await asyncio.to_thread(
                        self.engine.redis.setex, "market:breadth:full", 600, json.dumps(full_market_breadth)
                    )
                    logger.info(f"Full market breadth updated: {full_market_breadth} (sampled from {len(available_pairs)} symbols)")
            except asyncio.CancelledError:
                raise
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.warning(f"Full market breadth computation network/IO error: {type(e).__name__}: {e}")
            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, json.JSONDecodeError) as e:
                logger.error(f"Full market breadth computation data/logic error: {type(e).__name__}: {e}", exc_info=True)
                await self.engine._record_unexpected_exception("full_market_breadth", e)
            finally:
                self.engine._full_breadth_running = False
            await asyncio.sleep(1800)  # every 30 minutes (medium/long-term)

    async def _periodic_market_condition_check(self):
        """Check for market conditions that warrant more frequent symbol re-evaluation.

        Triggers re-evaluation when:
        - Significant news sentiment shifts are detected on tracked symbols
        - Unusually active market (many stocks with large daily price movements)
        - Extreme indicator values or Bollinger Band squeeze breakouts on tracked symbols
        """
        await asyncio.sleep(120)  # initial delay
        while self.engine._running:
            try:
                await self.engine.event_bus.request("check_market_conditions")
            except asyncio.CancelledError:
                raise
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.warning(f"Market condition check network/IO error: {type(e).__name__}: {e}")
            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, json.JSONDecodeError) as e:
                logger.error(f"Market condition check data/logic error: {type(e).__name__}: {e}", exc_info=True)
                await self.engine._record_unexpected_exception("market_condition_check", e)
            await asyncio.sleep(1800)  # check every 30 minutes (medium/long-term)

    async def _periodic_portfolio_rebalance(self):
        """Periodically trigger portfolio rebalance for long-term trading."""
        if not settings.PORTFOLIO_REBALANCE_ENABLED:
            logger.info("Portfolio rebalance is disabled (PORTFOLIO_REBALANCE_ENABLED=False). Task sleeping.")
            while self.engine._running:
                await self.engine._interruptible_sleep(3600)
            return
        await asyncio.sleep(3600)  # initial delay
        while self.engine._running:
            try:
                logger.info("Periodic portfolio rebalance triggered.")
                self.engine.trigger_portfolio_rebalance()
            except asyncio.CancelledError:
                raise
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.warning(f"Periodic portfolio rebalance network/IO error: {type(e).__name__}: {e}")
            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, json.JSONDecodeError) as e:
                logger.error(f"Periodic portfolio rebalance data/logic error: {type(e).__name__}: {e}", exc_info=True)
                await self.engine._record_unexpected_exception("periodic_portfolio_rebalance", e)
            await self.engine._interruptible_sleep(settings.PORTFOLIO_REBALANCE_INTERVAL_SECONDS)

    async def _risk_management_loop(self):
        """Check stop-loss, take-profit, and other risk rules on every ticker update."""
        await asyncio.sleep(5)  # initial delay

        while self.engine._running:
            try:
                now = time.time()
                symbols_to_check = []
                min_interval = settings.RISK_CHECK_INTERVAL_SECONDS
                async with self.engine._last_risk_check_lock:
                    for symbol, pos in list(self.engine.shared_state.positions.items()):
                        pos_tf = pos.get("timeframe")
                        if not pos_tf:
                            pos_tf_secs = settings.RISK_CHECK_INTERVAL_SECONDS
                        else:
                            pos_tf_secs = timeframe_to_seconds(pos_tf)

                        if pos_tf_secs >= 31_536_000:  # >= 1 year
                            pos_interval = settings.RISK_CHECK_INTERVAL_VERY_LONG_TF_SECONDS
                        elif pos_tf_secs >= settings.LONG_TERM_TF_SECONDS:  # >= 1 month
                            pos_interval = max(3600, min(3600, int(pos_tf_secs * 0.01)))
                        else:
                            pos_interval = settings.RISK_CHECK_INTERVAL_SECONDS

                        if pos_interval < min_interval:
                            min_interval = pos_interval

                        last_check = self.engine._last_risk_check.get(symbol, 0)
                        if now - last_check >= pos_interval:
                            symbols_to_check.append(symbol)
                            self.engine._last_risk_check[symbol] = now

                    # Clean up last_risk_check for closed positions
                    closed_symbols = [s for s in self.engine._last_risk_check if s not in self.engine.shared_state.positions]
                    for s in closed_symbols:
                        del self.engine._last_risk_check[s]

                if symbols_to_check:
                    await self.engine.event_bus.request("check_risk_management", symbols_to_check)
                    await self.engine._state_persistence.save_state()
                    self.engine.shared_state._state_dirty = True

                # Dynamically compute sleep interval based on the shortest timeframe
                # among current positions. This ensures the interval is updated immediately
                # when positions are closed and the shortest timeframe changes.
                await self.engine._interruptible_sleep(min_interval)
            except asyncio.CancelledError:
                raise
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.warning(f"Risk management loop network/IO error: {type(e).__name__}: {e}")
                await self.engine._interruptible_sleep(settings.RISK_CHECK_INTERVAL_SECONDS)
            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, json.JSONDecodeError) as e:
                logger.error(f"Risk management loop data/logic error: {type(e).__name__}: {e}", exc_info=True)
                await self.engine._record_unexpected_exception("risk_management_loop", e)
                await self.engine._interruptible_sleep(settings.RISK_CHECK_INTERVAL_SECONDS)

    async def _refresh_current_symbols_news_fast(self):
        """Fast news refresh loop – only for the symbols currently tracked by the engine."""
        if not settings.NEWS_ENABLED:
            logger.info("News is disabled (NEWS_ENABLED=False). Fast news refresh task sleeping.")
            while self.engine._running:
                await self.engine._interruptible_sleep(3600)
            return
        # Fetch immediately on startup, then periodically
        while self.engine._running:
            if self.engine._news_fast_running:
                logger.warning("Fast news refresh still running; skipping this cycle.")
                await self.engine._interruptible_sleep(settings.NEWS_FAST_UPDATE_INTERVAL_MINUTES * 60)
                continue
            self.engine._news_fast_running = True
            try:
                symbols = [entry["symbol"] for entry in self.engine.shared_state.current_symbols]
                if symbols:
                    logger.info(f"Fast news refresh for {len(symbols)} current symbols")
                    async def _fetch_news_with_limit(sym):
                        async with self.engine._news_semaphore:
                            await self.engine._fetch_and_store_news_for_symbol(sym)
                    await asyncio.gather(
                        *[_fetch_news_with_limit(sym) for sym in symbols]
                    )
            except asyncio.CancelledError:
                raise
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.warning(f"Fast news refresh network/IO error: {type(e).__name__}: {e}")
            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, json.JSONDecodeError) as e:
                logger.error(f"Fast news refresh data/logic error: {type(e).__name__}: {e}", exc_info=True)
                await self.engine._record_unexpected_exception("fast_news_refresh", e)
            finally:
                self.engine._news_fast_running = False
            await self.engine._interruptible_sleep(settings.NEWS_FAST_UPDATE_INTERVAL_MINUTES * 60)

    async def _refresh_news_cache(self):
        """Periodically fetch news for tracked stocks/ETFs and top-volume stocks to keep cache warm."""
        if not settings.NEWS_ENABLED:
            logger.info("News is disabled (NEWS_ENABLED=False). News cache refresh task sleeping.")
            while self.engine._running:
                await self.engine._interruptible_sleep(3600)
            return
        if fetch_news_for_symbol is None:
            logger.warning("News module not available; skipping background news refresh.")
            return

        while self.engine._running:
            if self.engine._news_cache_running:
                logger.warning("News cache refresh still running; skipping this cycle.")
                await self.engine._interruptible_sleep(settings.NEWS_UPDATE_INTERVAL_MINUTES * 60)
                continue
            self.engine._news_cache_running = True
            try:
                cycle_start = time.time()
                # Slow refresh: all available pairs EXCEPT the stocks already handled by the fast loop
                current_symbols = {entry["symbol"] for entry in self.engine.shared_state.current_symbols}
                symbols_to_refresh = set()
                try:
                    plain_assets = await self.engine.event_bus.request("get_tradable_assets")
                    available_pairs = [f"{sym}/{self.engine.base_currency}" for sym in plain_assets]
                    # Fetch tickers for a subset to determine top volume symbols
                    # (limit to 200 to avoid excessive API calls)
                    sample_for_vol = available_pairs[:200]
                    plain_sample = [s.split("/")[0] for s in sample_for_vol]
                    raw_quotes = await self.engine.event_bus.request("get_quotes_batched", plain_sample, timeout_per_chunk=45.0)
                    tickers = {pair: raw_quotes.get(pair.split("/")[0], {}) for pair in sample_for_vol}
                    def _vol(sym):
                        t = tickers.get(sym, {})
                        return t.get('quoteVolume', 0) or 0
                    symbols_to_refresh = set(sample_for_vol) - current_symbols
                except (ValueError, TypeError, KeyError, ConnectionError, TimeoutError, OSError) as e:
                    logger.warning(f"Could not get available pairs for news refresh: {e}")

                for sym in symbols_to_refresh:
                    try:
                        async with self.engine._news_semaphore:
                            stock_name = await self.engine.event_bus.request("get_stock_name", sym)
                            articles = await fetch_news_for_symbol(sym, stock_name)
                            if articles:
                                base_symbol = sym.split("/")[0] if "/" in sym else sym
                                await asyncio.to_thread(store_news_articles, base_symbol, articles)
                    except (ValueError, TypeError, KeyError, ConnectionError, TimeoutError, OSError) as e:
                        logger.info(f"News refresh failed for {sym}: {e}")
                    await asyncio.sleep(0.2)

                logger.info(f"News cache refreshed for {len(symbols_to_refresh)} symbols in {time.time() - cycle_start:.2f}s")
            except asyncio.CancelledError:
                raise
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.warning(f"Background news refresh network/IO error: {type(e).__name__}: {e}")
            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, json.JSONDecodeError) as e:
                logger.error(f"Background news refresh data/logic error: {type(e).__name__}: {e}", exc_info=True)
                await self.engine._record_unexpected_exception("news_cache_refresh", e)
            finally:
                self.engine._news_cache_running = False

            # Clean up old news articles
            try:
                await asyncio.to_thread(cleanup_old_news, settings.NEWS_RETENTION_SECONDS)
            except (ValueError, TypeError, ConnectionError, TimeoutError, OSError) as e:
                logger.warning(f"News cleanup failed: {e}")

            await self.engine._interruptible_sleep(settings.NEWS_UPDATE_INTERVAL_MINUTES * 60)

    async def _download_market_data_loop(self):
        """Periodically download and store OHLCV data for tracked stocks, with gap detection."""
        # Initial delay to let the engine settle
        await asyncio.sleep(30)
        while self.engine._running:
            if self.engine._market_data_running:
                logger.warning("Market data download still running; skipping this cycle.")
                await self.engine._interruptible_sleep(get_effective_refresh_interval(settings.MARKET_DATA_REFRESH_SECONDS, self.engine.shared_state.current_symbols, "data"))
                continue
            self.engine._market_data_running = True
            try:
                if not self.engine.shared_state.current_symbols:
                    logger.info("No symbols tracked; skipping market data download.")
                else:
                    logger.info("Starting market data download cycle...")
                    now_ms = int(time.time() * 1000)
                    start_ms = now_ms - settings.OHLCV_RETENTION_DAYS * 24 * 60 * 60 * 1000

                    async def _download_symbol_data(symbol_entry):
                        symbol = symbol_entry["symbol"]
                        tf = symbol_entry["timeframe"]
                        logger.debug(f"Downloading market data for {symbol} ({tf})")
                        await self.engine.event_bus.request("download_symbol_ohlcv", symbol, tf, start_ms, now_ms)

                    shuffled_symbols = list(self.engine.shared_state.current_symbols)
                    random.shuffle(shuffled_symbols)
                    download_tasks = [_download_symbol_data(entry) for entry in shuffled_symbols]
                    await asyncio.gather(*download_tasks)
                    logger.info("Market data download cycle complete.")
            except asyncio.CancelledError:
                raise
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.warning(f"Market data download loop network/IO error: {type(e).__name__}: {e}")
            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, json.JSONDecodeError) as e:
                logger.error(f"Market data download loop data/logic error: {type(e).__name__}: {e}", exc_info=True)
                await self.engine._record_unexpected_exception("market_data_download_loop", e)
            finally:
                self.engine._market_data_running = False

            await self.engine._interruptible_sleep(get_effective_refresh_interval(settings.MARKET_DATA_REFRESH_SECONDS, self.engine.shared_state.current_symbols, "data"))

    async def _download_all_assets_data_loop(self):
        """Periodically download OHLCV for ALL tradable assets (stocks, ETFs, BTPs)."""
        await asyncio.sleep(120)  # initial delay to let the engine settle
        while self.engine._running:
            if self.engine._full_download_running:
                logger.info("Full download already running (likely force download); skipping this cycle.")
                await self.engine._interruptible_sleep(get_effective_refresh_interval(settings.FULL_ASSET_OHLCV_DOWNLOAD_INTERVAL_SECONDS, self.engine.shared_state.current_symbols, "data"))
                continue
            self.engine._full_download_running = True
            try:
                logger.info("Starting full asset OHLCV download cycle...")
                # 1. Get all stock + ETF symbols
                plain_assets = await self.engine.event_bus.request("get_tradable_assets")
                stock_pairs = [f"{sym}/{self.engine.base_currency}" for sym in plain_assets]
                etf_symbols = await self.engine.event_bus.request("get_etf_symbols")
                etf_pairs = [f"{sym}/{self.engine.base_currency}" for sym in etf_symbols]

                # 2. Get all BTP symbols
                btp_bonds = await self.engine.event_bus.request("get_btp_bonds")
                btp_pairs = [f"{b['isin']}/{self.engine.base_currency}" for b in btp_bonds]

                all_pairs = stock_pairs + etf_pairs + btp_pairs
                if not all_pairs:
                    logger.info("No tradable assets found; skipping full download.")
                    await self.engine._interruptible_sleep(get_effective_refresh_interval(settings.FULL_ASSET_OHLCV_DOWNLOAD_INTERVAL_SECONDS, self.engine.shared_state.current_symbols, "data"))
                    continue

                now_ms = int(time.time() * 1000)
                start_ms = now_ms - settings.OHLCV_RETENTION_DAYS * 24 * 60 * 60 * 1000

                # Prioritize symbols with missing or stale data for configured timeframes
                loop = asyncio.get_running_loop()
                latest_timestamps = await loop.run_in_executor(
                    self.engine._db_executor,
                    get_latest_ohlcv_timestamps_batch,
                    all_pairs,
                    settings.OHLCV_TIMEFRAMES
                )

                pairs_with_stale_data = []
                pairs_complete = []
                now_ms = int(time.time() * 1000)

                for pair in all_pairs:
                    stale_tfs = []
                    for tf in settings.OHLCV_TIMEFRAMES:
                        latest_ts = latest_timestamps.get(pair, {}).get(tf)
                        if latest_ts is None:
                            stale_tfs.append(tf)
                        else:
                            interval_ms = timeframe_to_ms(tf)
                            if latest_ts < now_ms - interval_ms:
                                stale_tfs.append(tf)
                    if stale_tfs:
                        pairs_with_stale_data.append((pair, stale_tfs))
                    else:
                        pairs_complete.append(pair)

                random.shuffle(pairs_with_stale_data)
                if pairs_with_stale_data:
                    logger.info(f"Prioritizing {len(pairs_with_stale_data)} symbols with stale/missing OHLCV data out of {len(all_pairs)} total.")

                async def _download_symbol_data(pair: str, tfs: List[str]):
                    for tf in tfs:
                        await self.engine.event_bus.request("download_symbol_ohlcv", pair, tf, start_ms, now_ms, quiet=True)

                # Limit concurrent symbol downloads to 2 to avoid exhausting the
                # _download_executor thread pool, leaving threads available for
                # tracked tickers.
                download_concurrency = asyncio.Semaphore(settings.FULL_DOWNLOAD_CONCURRENCY)
                async def _limited_download(pair: str, tfs: List[str]):
                    async with download_concurrency:
                        await _download_symbol_data(pair, tfs)
                download_tasks = [_limited_download(pair, tfs) for pair, tfs in pairs_with_stale_data]
                await asyncio.gather(*download_tasks)

                # Clean up old data
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(self.engine._db_executor, cleanup_old_position_pnl, 90)
                await loop.run_in_executor(self.engine._db_executor, cleanup_old_backtest_results, 90)
                logger.info("Full asset OHLCV download cycle complete.")
            except asyncio.CancelledError:
                raise
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.warning(f"Full asset download loop network/IO error: {type(e).__name__}: {e}")
            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, json.JSONDecodeError) as e:
                logger.error(f"Full asset download loop data/logic error: {type(e).__name__}: {e}", exc_info=True)
                await self.engine._record_unexpected_exception("full_asset_download_loop", e)
            finally:
                self.engine._full_download_running = False

            # Wait before next full download
            await self.engine._interruptible_sleep(get_effective_refresh_interval(settings.FULL_ASSET_OHLCV_DOWNLOAD_INTERVAL_SECONDS, self.engine.shared_state.current_symbols, "data"))

    async def _download_all_news_loop(self):
        """Periodically pre‑fetch news for ALL tradable assets (stocks, ETFs, BTPs)."""
        if not settings.NEWS_ENABLED:
            logger.info("News is disabled (NEWS_ENABLED=False). Full news download task sleeping.")
            while self.engine._running:
                await self.engine._interruptible_sleep(3600)
            return
        await asyncio.sleep(180)  # initial delay to let the engine settle
        while self.engine._running:
            try:
                logger.info("Starting full asset news download cycle...")
                # 1. Get all stock + ETF symbols
                plain_assets = await self.engine.event_bus.request("get_tradable_assets")
                stock_pairs = [f"{sym}/{self.engine.base_currency}" for sym in plain_assets]

                # 2. Get all BTP symbols
                btp_bonds = await self.engine.event_bus.request("get_btp_bonds")
                btp_pairs = [f"{b['isin']}/{self.engine.base_currency}" for b in btp_bonds]

                all_pairs = stock_pairs + btp_pairs
                if not all_pairs:
                    logger.info("No tradable assets found; skipping full news download.")
                    await self.engine._interruptible_sleep(get_effective_refresh_interval(settings.FULL_ASSET_NEWS_DOWNLOAD_INTERVAL_SECONDS, self.engine.shared_state.current_symbols, "news"))
                    continue

                # Prioritize currently tracked symbols first, then the rest.
                current_symbol_set = {entry["symbol"] for entry in self.engine.shared_state.current_symbols}
                priority_pairs = [p for p in all_pairs if p in current_symbol_set]
                other_pairs = [p for p in all_pairs if p not in current_symbol_set]
                ordered_pairs = priority_pairs + other_pairs

                # Download concurrently, respecting rate limits via _news_semaphore
                async def _download_news_for_symbol(pair: str):
                    try:
                        async with self.engine._news_semaphore:
                            await self.engine._fetch_and_store_news_for_symbol(pair)
                    except (ValueError, TypeError, KeyError, ConnectionError, TimeoutError, OSError) as e:
                        logger.warning(f"Full news download failed for {pair}: {e}")

                news_tasks = [_download_news_for_symbol(pair) for pair in ordered_pairs]
                await asyncio.gather(*news_tasks)

                logger.info("Full asset news download cycle complete.")
            except asyncio.CancelledError:
                raise
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.warning(f"Full asset news download loop network/IO error: {type(e).__name__}: {e}")
            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, json.JSONDecodeError) as e:
                logger.error(f"Full asset news download loop data/logic error: {type(e).__name__}: {e}", exc_info=True)
                await self.engine._record_unexpected_exception("full_asset_news_download_loop", e)

            await self.engine._interruptible_sleep(get_effective_refresh_interval(settings.FULL_ASSET_NEWS_DOWNLOAD_INTERVAL_SECONDS, self.engine.shared_state.current_symbols, "news"))

    async def _refresh_all_quotes_loop(self):
        """Periodically fetch quotes for all tradable assets and cache them in Redis."""
        await asyncio.sleep(60)  # initial delay
        while self.engine._running:
            if self.engine._quotes_fetch_running:
                logger.info("Quotes fetch already running (likely re-evaluation or breadth); skipping this cycle.")
                await self.engine._interruptible_sleep(get_effective_refresh_interval(settings.QUOTE_REFRESH_INTERVAL_SECONDS, self.engine.shared_state.current_symbols, "quotes"))
                continue
            self.engine._quotes_fetch_running = True
            try:
                # Do NOT skip when the circuit breaker is open — get_quotes
                # internally checks the circuit breaker and falls back to DB
                # close prices (from market_data candles).  Skipping here
                # prevents those fallback prices from being saved to the
                # quotes table, leaving it stale when yfinance is down.
                plain_assets = await self.engine.event_bus.request("get_tradable_assets")
                etf_symbols = await self.engine.event_bus.request("get_etf_symbols")
                btp_bonds = await self.engine.event_bus.request("get_btp_bonds")
                btp_isins = [b["isin"] for b in btp_bonds if b.get("isin")]

                all_quote_symbols = plain_assets + etf_symbols + btp_isins
                if all_quote_symbols:
                    # Fetch quotes in batches to avoid yfinance timeouts on large symbol lists
                    await self.engine.event_bus.request("get_quotes_batched", all_quote_symbols, timeout_per_chunk=180.0)
            except asyncio.CancelledError:
                raise
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.warning(f"Background quote refresh network/IO error: {type(e).__name__}: {e}")
            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, json.JSONDecodeError) as e:
                logger.error(f"Background quote refresh data/logic error: {type(e).__name__}: {e}", exc_info=True)
                await self.engine._record_unexpected_exception("quote_refresh_loop", e)
            finally:
                self.engine._quotes_fetch_running = False
            await self.engine._interruptible_sleep(get_effective_refresh_interval(settings.QUOTE_REFRESH_INTERVAL_SECONDS, self.engine.shared_state.current_symbols, "quotes"))

    async def _refresh_ticker_discovery_loop(self):
        """Periodically discover tickers from news RSS feeds and trending stocks.
        Caches results in Redis so re-evaluation never blocks on slow HTTP calls."""
        await asyncio.sleep(120)  # initial delay
        while self.engine._running:
            try:
                plain_assets = await self.engine.event_bus.request("get_tradable_assets")
                available_pairs = [f"{sym}/{self.engine.base_currency}" for sym in plain_assets]

                if settings.NEWS_ENABLED and settings.NEWS_TICKER_DISCOVERY_ENABLED and discover_tickers_from_news is not None:
                    logger.info("Background: refreshing RSS ticker discovery...")
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(
                        self.engine._download_executor,
                        lambda: discover_tickers_from_news(
                            existing_pairs=available_pairs,
                            cache_only=False,
                        )
                    )

                if settings.NEWS_ENABLED and settings.NEWS_SYMBOL_DISCOVERY_ENABLED and discover_trending_stocks is not None:
                    logger.info("Background: refreshing trending stock discovery...")
                    await loop.run_in_executor(
                        self.engine._download_executor,
                        lambda: discover_trending_stocks(
                            self.engine.base_currency,
                            available_pairs,
                            max_symbols=settings.NEWS_SYMBOL_DISCOVERY_MAX_SYMBOLS,
                            min_sentiment=settings.NEWS_SYMBOL_DISCOVERY_MIN_SENTIMENT,
                            min_articles=settings.NEWS_SYMBOL_DISCOVERY_MIN_ARTICLES,
                            cache_only=False,
                        )
                    )
            except asyncio.CancelledError:
                raise
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.warning(f"Ticker discovery refresh network/IO error: {type(e).__name__}: {e}")
            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, json.JSONDecodeError) as e:
                logger.error(f"Ticker discovery refresh data/logic error: {type(e).__name__}: {e}", exc_info=True)
                await self.engine._record_unexpected_exception("ticker_discovery_loop", e)
            await asyncio.sleep(3600)  # every 60 minutes (medium/long-term)

    async def _fetch_dividends_loop(self):
        """Periodically fetch and store dividends for tracked symbols."""
        await self.engine._interruptible_sleep(300)  # initial delay 5 minutes
        while self.engine._running:
            try:
                symbols = [entry["symbol"] for entry in self.engine.shared_state.current_symbols]
                if not symbols:
                    await self.engine._interruptible_sleep(3600)
                    continue
                # Only fetch for non-BTP symbols (BTPs use coupons, not dividends)
                stock_symbols = [s for s in symbols if not is_btp_isin(s.split("/")[0])]
                if stock_symbols:
                    async def _fetch_dividends(symbol: str):
                        try:
                            divs = await asyncio.to_thread(get_yahoo_dividends, symbol)
                            for d in divs:
                                await asyncio.to_thread(insert_dividend, symbol, d["date"], d["amount"])
                        except (ValueError, TypeError, KeyError, ConnectionError, TimeoutError, OSError) as e:
                            logger.debug(f"Dividend fetch failed for {symbol}: {e}")
                    await asyncio.gather(*[_fetch_dividends(s) for s in stock_symbols])
                # Cleanup old dividends
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(self.engine._db_executor, cleanup_old_dividends, 365)
            except asyncio.CancelledError:
                raise
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.warning(f"Dividend fetch loop network/IO error: {type(e).__name__}: {e}")
            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, json.JSONDecodeError) as e:
                logger.error(f"Dividend fetch loop data/logic error: {type(e).__name__}: {e}", exc_info=True)
                await self.engine._record_unexpected_exception("dividend_fetch_loop", e)
            await self.engine._interruptible_sleep(86400)  # daily

    async def _reinvest_dividends_loop(self):
        """Periodically check for pending dividends and reinvest them if enabled."""
        if not settings.REINVEST_DIVIDENDS:
            logger.info("Dividend reinvestment is disabled (REINVEST_DIVIDENDS=False). Task sleeping.")
            while self.engine._running:
                await self.engine._interruptible_sleep(86400)
            return

        await self.engine._interruptible_sleep(3600)  # initial delay 1 hour
        while self.engine._running:
            try:
                if not await self.engine._is_market_open():
                    logger.info("Skipping dividend reinvestment: market is closed.")
                else:
                    for symbol, pos in list(self.engine.shared_state.positions.items()):
                        if is_btp_isin(symbol.split("/")[0]):
                            continue  # BTPs use coupons, not dividends

                        pending_divs = await asyncio.to_thread(get_pending_dividends_for_symbol, symbol)
                        if not pending_divs:
                            continue

                        # Get current price
                        base = symbol.split("/")[0]
                        quotes = await asyncio.to_thread(get_quotes_cached, [base])
                        price = quotes.get(base, {}).get("last")
                        if not price or price <= 0:
                            logger.warning(f"Cannot reinvest dividends for {symbol}: no valid price.")
                            continue

                        for div in pending_divs:
                            div_per_share = div["amount"]
                            current_shares = pos.get("amount", 0.0)
                            total_div_value = div_per_share * current_shares
                            
                            if total_div_value <= 0:
                                # Mark as reinvested to skip it in the future if we no longer hold shares
                                await asyncio.to_thread(mark_dividend_reinvested, div["id"])
                                continue

                            # Execute buy order for the reinvested amount (amount is in quote currency)
                            order = await asyncio.to_thread(
                                self.engine.trader.create_market_buy_order, symbol, total_div_value
                            )

                            if order.get("status") == "filled":
                                filled_qty = order["amount"]
                                filled_price = order["price"]
                                
                                # Update position
                                async with self.engine.shared_state._positions_lock:
                                    if symbol in self.engine.shared_state.positions:
                                        pos = self.engine.shared_state.positions[symbol]
                                        old_amount = pos.get("amount", 0.0)
                                        old_cost = pos.get("cost_basis", 0.0)
                                        new_amount = old_amount + filled_qty
                                        new_cost = old_cost + (filled_qty * filled_price)
                                        pos["amount"] = new_amount
                                        pos["cost_basis"] = new_cost
                                        self.engine.shared_state._state_dirty = True

                                # Mark dividend as reinvested
                                await asyncio.to_thread(mark_dividend_reinvested, div["id"])

                                if self.engine.notifier:
                                    await self.engine.notifier.send_notification(
                                        f"💰 Dividend Reinvested: {total_div_value:.2f} {self.engine.base_currency} for {symbol} "
                                        f"bought {filled_qty:.6f} shares @ {filled_price:.2f}",
                                        summary={"action": "DIVIDEND_REINVEST", "symbol": symbol, "amount": total_div_value}
                                    )
                                logger.info(f"Reinvested dividend of {total_div_value:.2f} for {symbol}: bought {filled_qty:.6f} shares @ {filled_price:.2f}")

                await self.engine._state_persistence.save_state()
            except asyncio.CancelledError:
                raise
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.warning(f"Dividend reinvestment loop network/IO error: {type(e).__name__}: {e}")
            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, json.JSONDecodeError) as e:
                logger.error(f"Dividend reinvestment loop data/logic error: {type(e).__name__}: {e}", exc_info=True)
                await self.engine._record_unexpected_exception("reinvest_dividends_loop", e)

            await self.engine._interruptible_sleep(3600)  # check every hour

    async def _evaluate_llm_decisions_loop(self):
        """Periodically evaluate past LLM decisions against actual market outcomes."""
        await asyncio.sleep(300)  # initial delay 5 minutes
        while self.engine._running:
            try:
                # Evaluate decisions older than 1 hour (3600 seconds)
                pending_decisions = await asyncio.to_thread(get_pending_llm_decisions, 3600)
                if not pending_decisions:
                    await self.engine._interruptible_sleep(600)
                    continue

                # Fetch latest close prices for all symbols in pending decisions
                symbols = [d["symbol"] for d in pending_decisions]
                # get_latest_close_prices expects base symbols without /currency
                base_symbols = [s.split("/")[0] if "/" in s else s for s in symbols]
                prices = await asyncio.to_thread(get_latest_close_prices, base_symbols)

                for decision in pending_decisions:
                    symbol = decision["symbol"]
                    base_symbol = symbol.split("/")[0] if "/" in symbol else symbol
                    price_data = prices.get(base_symbol)
                    if not price_data or price_data.get("last") is None:
                        continue

                    # Skip decisions that haven't reached their timeframe-scaled
                    # evaluation window yet. This prevents judging a 1Y HOLD
                    # after just 1 hour.
                    tf_seconds = timeframe_to_seconds(decision.get("timeframe") or "1d")
                    eval_window = max(3600, min(tf_seconds * 0.1, 604800))
                    decision_age = (time.time() * 1000 - decision["timestamp"]) / 1000
                    if decision_age < eval_window:
                        continue

                    outcome_price = price_data["last"]
                    entry_price = decision["entry_price"]
                    action = decision["action"]

                    if entry_price is None or entry_price <= 0:
                        continue

                    # Determine if the decision was profitable using
                    # timeframe-aware thresholds to filter out noise.
                    tf_seconds = timeframe_to_seconds(decision.get("timeframe") or "1d")
                    # Minimum meaningful price movement (1% or 0.5% for long TFs)
                    min_move_pct = 0.005 if tf_seconds >= 2592000 else 0.01

                    price_change_pct = ((outcome_price - entry_price) / entry_price) if entry_price else 0.0

                    if action == "BUY":
                        # BUY is profitable if price went up by at least min_move_pct
                        profitable = price_change_pct >= min_move_pct
                    elif action == "SELL":
                        # SELL is profitable if price went down by at least min_move_pct
                        profitable = price_change_pct <= -min_move_pct
                    elif action == "HOLD":
                        # HOLD is profitable if price did NOT rise significantly
                        # (avoided a bad buy) OR if it fell (avoided a loss)
                        profitable = price_change_pct < min_move_pct
                    else:
                        profitable = False

                    # Generate a heuristic analysis of why the decision was right or wrong
                    price_change_display = abs(price_change_pct * 100)
                    analysis_parts = []
                    if profitable:
                        analysis_parts.append(f"Price moved {price_change_display:.2f}% in favor of the {action} decision (threshold: {min_move_pct*100:.1f}%).")
                    else:
                        analysis_parts.append(f"Price moved {price_change_display:.2f}% against the {action} decision (threshold: {min_move_pct*100:.1f}%).")
                    
                    ctx = decision.get("market_context")
                    if isinstance(ctx, dict):
                        if ctx.get("market_regime"):
                            analysis_parts.append(f"Market regime was '{ctx['market_regime']}'.")
                        if ctx.get("sentiment") is not None:
                            analysis_parts.append(f"News sentiment was {ctx['sentiment']}.")
                        if ctx.get("atr") is not None:
                            analysis_parts.append(f"ATR was {ctx['atr']:.4f}.")
                    
                    outcome_analysis = " ".join(analysis_parts)

                    await asyncio.to_thread(
                        update_llm_decision_outcome,
                        decision["id"],
                        outcome_price,
                        profitable,
                        outcome_analysis
                    )
                
                logger.info(f"Evaluated {len(pending_decisions)} LLM decisions for quality tracking.")

                # --- Check for accuracy degradation and alert ---
                metrics = await asyncio.to_thread(get_llm_decision_quality_metrics, 7)
                if metrics["total_evaluated"] >= 10 and metrics["accuracy"] < 40.0:
                    # Check cooldown to avoid spamming
                    alert_key = "llm:accuracy_alert_cooldown"
                    cooldown = await asyncio.to_thread(self.engine.redis.get, alert_key)
                    if not cooldown:
                        if self.engine.notifier:
                            await self.engine.notifier.send_notification(
                                f"⚠️ LLM Decision Quality Alert: Accuracy has dropped to {metrics['accuracy']:.1f}% "
                                f"over the last 7 days ({metrics['total_evaluated']} evaluated decisions).",
                                summary={
                                    "action": "ALERT",
                                    "reason": "LLM accuracy degradation",
                                    "accuracy": metrics["accuracy"],
                                    "total_evaluated": metrics["total_evaluated"]
                                }
                            )
                        # Set cooldown for 24 hours
                        await asyncio.to_thread(self.engine.redis.setex, alert_key, 86400, "1")

                # Clean up old evaluated decisions to prevent database bloat
                await asyncio.to_thread(cleanup_old_llm_decisions, 30)
            except asyncio.CancelledError:
                raise
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.warning(f"LLM decision evaluation loop network/IO error: {type(e).__name__}: {e}")
            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, json.JSONDecodeError) as e:
                logger.error(f"LLM decision evaluation loop data/logic error: {type(e).__name__}: {e}", exc_info=True)
                await self.engine._record_unexpected_exception("llm_decision_eval_loop", e)
            
            await self.engine._interruptible_sleep(600)  # check every 10 minutes

    async def _redis_health_check_loop(self):
        """Periodically check Redis connection and alert on state changes."""
        await asyncio.sleep(30)
        last_degraded_notify_time = 0.0
        while self.engine._running:
            try:
                was_available = is_redis_available()
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(self.engine._db_executor, check_redis_connection)
                is_available = is_redis_available()
                if was_available and not is_available:
                    logger.critical("Redis connection lost. Degrading to no-cache mode.")
                    if self.engine.notifier:
                        await self.engine.notifier.send_notification(
                            "⚠️ Redis connection lost. The bot is now running in degraded mode (caching disabled).",
                            summary={"action": "ERROR", "reason": "Redis connection lost"}
                        )
                    last_degraded_notify_time = time.time()
                elif not was_available and is_available:
                    logger.info("Redis connection restored. Caching resumed.")
                    if self.engine.notifier:
                        await self.engine.notifier.send_notification(
                            "✅ Redis connection restored. Caching resumed.",
                            summary={"action": "INFO", "reason": "Redis connection restored"}
                        )
                elif not is_available:
                    # Periodic reminder every 6 hours if still in degraded mode
                    if time.time() - last_degraded_notify_time >= 21600:
                        if self.engine.notifier:
                            await self.engine.notifier.send_notification(
                                "⚠️ Redis is still unavailable. The bot remains in degraded mode (caching disabled).",
                                summary={"action": "ERROR", "reason": "Redis still unavailable"}
                            )
                        last_degraded_notify_time = time.time()
            except asyncio.CancelledError:
                raise
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.warning(f"Redis health check loop network/IO error: {type(e).__name__}: {e}")
            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, json.JSONDecodeError) as e:
                logger.error(f"Redis health check loop data/logic error: {type(e).__name__}: {e}", exc_info=True)
                await self.engine._record_unexpected_exception("redis_health_check_loop", e)
            await asyncio.sleep(60)

    async def _health_check_loop(self):
        """Periodically check the health of supervised background tasks and alert on failures."""
        await asyncio.sleep(300)  # initial delay 5 minutes
        alerted_supervisors = set()
        while self.engine._running:
            try:
                for sup in self.engine._supervisors:
                    health = sup.get_health()
                    if not health["is_healthy"] or not health["running"]:
                        if sup.name not in alerted_supervisors:
                            alerted_supervisors.add(sup.name)
                            if self.engine.notifier:
                                await self.engine.notifier.send_notification(
                                    f"🚨 Critical background task '{health['name']}' is not healthy or has stopped running. "
                                    f"Last error: {health['last_exception']}",
                                    summary={"action": "CRITICAL", "reason": f"Task {health['name']} unhealthy"}
                                )
                    else:
                        # If the task somehow becomes healthy/running again, clear the alert flag
                        alerted_supervisors.discard(sup.name)
            except asyncio.CancelledError:
                raise
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.warning(f"Health check loop network/IO error: {type(e).__name__}: {e}")
            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, json.JSONDecodeError) as e:
                logger.error(f"Health check loop data/logic error: {type(e).__name__}: {e}", exc_info=True)
                await self.engine._record_unexpected_exception("health_check_loop", e)
            await asyncio.sleep(300)  # check every 5 minutes

    async def _monitor_entry_signals_loop(self):
        """Periodically check tracked symbols for favourable entry conditions.
        When a condition is met, force an immediate LLM evaluation."""
        await asyncio.sleep(10)  # initial delay
        while self.engine._running:
            try:
                for entry in self.engine.shared_state.current_symbols:
                    symbol = entry["symbol"]
                    tf = entry["timeframe"]
                    # Skip entry signal monitoring for very long timeframes (>= 1 month)
                    # where short-term crossovers are irrelevant.
                    tf_seconds = timeframe_to_seconds(tf)
                    # Avoid re‑triggering too often – enforce a cooldown of at least
                    # the normal strategy interval.
                    # Use a short, dedicated cooldown so the bot reacts quickly to new signals
                    cooldown = settings.ENTRY_SIGNAL_COOLDOWN_SECONDS
                    async with self.engine.shared_state._eval_state_lock:
                        last_forced = self.engine.shared_state._force_eval_time.get(symbol, 0)
                    if time.time() - last_forced < cooldown:
                        continue

                    if await self.engine.event_bus.request("detect_entry_signal", symbol, tf):
                        logger.info(f"Entry signal detected for {symbol}, forcing LLM evaluation.")
                        async with self.engine.shared_state._eval_state_lock:
                            self.engine.shared_state._force_eval[symbol] = True
                            self.engine.shared_state._force_eval_time[symbol] = time.time()
                            # Clear last evaluation timestamp so the main loop picks it up immediately
                            self.engine.shared_state._last_strategy_eval.pop(symbol, None)
            except asyncio.CancelledError:
                raise
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.warning(f"Entry signal monitor network/IO error: {type(e).__name__}: {e}")
            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, json.JSONDecodeError) as e:
                logger.error(f"Entry signal monitor data/logic error: {type(e).__name__}: {e}", exc_info=True)
                await self.engine._record_unexpected_exception("entry_signal_monitor", e)
            await self.engine._interruptible_sleep(settings.ENTRY_SIGNAL_CHECK_INTERVAL_SECONDS)

    async def _check_pending_entries(self):
        """Periodically check pending entry conditions and execute if met."""
        await asyncio.sleep(10)  # short initial delay
        while self.engine._running:
            try:
                now = time.time()
                for symbol in list(self.engine.shared_state._pending_entries.keys()):
                    await self.engine.event_bus.request("process_pending_entry", symbol, now)
            except asyncio.CancelledError:
                raise
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.warning(f"Pending entries check network/IO error: {type(e).__name__}: {e}")
            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, json.JSONDecodeError) as e:
                logger.error(f"Error checking pending entries data/logic: {type(e).__name__}: {e}", exc_info=True)
                await self.engine._record_unexpected_exception("check_pending_entries", e)
            await asyncio.sleep(60)  # check every 60 seconds (medium/long-term)

    async def _process_queued_orders(self):
        """Periodically check queued limit orders in the simulator and process fills,
        including partial fills.

        Instead of re-executing the signal (which would place a duplicate order
        while the original is still open), we poll the actual simulator order
        status. When the order fills (fully or partially), we record the trade
        and update positions exactly as a normal fill would. We never place a
        new order here.
        """
        await asyncio.sleep(10)
        # --- Notify mode: no queued order processing ---
        if settings.TRADING_MODE == "notify":
            logger.info("Notify mode: skipping queued order processing.")
            return
        while self.engine._running:
            try:
                for queued in list(self.engine.shared_state.queued_orders):
                    await self.engine.event_bus.request("process_single_queued_order", queued)
            except asyncio.CancelledError:
                raise
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.warning(f"Queued orders processing network/IO error: {type(e).__name__}: {e}")
            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, json.JSONDecodeError) as e:
                logger.error(f"Error processing queued orders data/logic: {type(e).__name__}: {e}", exc_info=True)
                await self.engine._record_unexpected_exception("process_queued_orders", e)
            await asyncio.sleep(15)  # check every 15 seconds for faster fill detection

    async def _cleanup_orphaned_orders(self):
        """Periodically cancel any open orders that are older than 10 minutes,
        but never cancel orders that are still being tracked as queued."""
        await asyncio.sleep(120)  # initial delay
        # --- Notify mode: no orphaned order cleanup ---
        if settings.TRADING_MODE == "notify":
            return
        while self.engine._running:
            try:
                await self.engine.event_bus.request("cancel_orphaned_orders")
            except asyncio.CancelledError:
                raise
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.warning(f"Orphaned order cleanup network/IO error: {type(e).__name__}: {e}")
            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, json.JSONDecodeError) as e:
                logger.error(f"Orphaned order cleanup data/logic error: {type(e).__name__}: {e}", exc_info=True)
                await self.engine._record_unexpected_exception("cleanup_orphaned_orders", e)
            await asyncio.sleep(900)  # every 15 minutes
