import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Dict
from zoneinfo import ZoneInfo

from src.config.settings import settings
from src.trading.components.evaluation_scheduler import EvaluationScheduler
from src.utils.task_supervisor import TaskSupervisor

logger = logging.getLogger(__name__)

class EngineOrchestrator:
    """Handles background task orchestration for the TradingEngine."""

    def __init__(self, engine):
        self.engine = engine
        self._evaluation_scheduler = EvaluationScheduler(engine)

    async def start_background_tasks(self):
        """Initialize and start all supervised background tasks."""
        self.engine._background_tasks.clear()
        self.engine._supervisors.clear()
        
        background_factories = [
            self.engine._refresh_news_cache,
            self.engine._refresh_current_symbols_news_fast,
            self.engine._download_market_data_loop,
            self.engine._download_all_assets_data_loop,
            self.engine._download_all_news_loop,
            self.engine._risk_management_loop,
            self.engine._periodic_reconcile,
            self.engine._periodic_reevaluate,
            self.engine._periodic_pause_check,
            self.engine._periodic_pause_resume_check,
            self.engine._periodic_full_market_breadth,
            self.engine._periodic_market_condition_check,
            self.engine._periodic_portfolio_rebalance,
            self.engine._check_pending_entries,
            self.engine._cleanup_orphaned_orders,
            self.engine._process_queued_orders,
            self.engine._monitor_entry_signals_loop,
            self._market_clock_monitor,
            self.engine._refresh_all_quotes_loop,
            self.engine._refresh_ticker_discovery_loop,
            self.engine._fetch_dividends_loop,
            self.engine._reinvest_dividends_loop,
            self.engine._redis_health_check_loop,
            self.engine._health_check_loop,
            self.engine._evaluate_llm_decisions_loop,
        ]
        
        for factory in background_factories:
            sup = TaskSupervisor(factory, name=factory.__qualname__)
            sup.set_notifier(self.engine.notifier)
            task = asyncio.create_task(sup.run(), name=f"supervisor:{factory.__qualname__}")
            task.add_done_callback(self.engine._log_task_exception)
            self.engine._background_tasks.append(task)
            self.engine._supervisors.append(sup)

    async def run(self):
        """Main event-driven loop using WebSocket ticker updates."""
        engine = self.engine
        logger.info("Trading engine initializing...")
        await engine._initialize_clients()
        logger.info("Trading engine started.")
        # Start background tasks
        await self.start_background_tasks()

        while engine._running:
            try:
                if getattr(settings, 'PAPER_BALANCE_CHANGED', False):
                    settings.PAPER_BALANCE_CHANGED = False
                    await engine.reset_paper_trading_state()
                    continue

                await engine._interruptible_sleep(settings.ENGINE_LOOP_INTERVAL_SECONDS)

                # Process any symbol whose evaluation interval has elapsed
                now = time.time()

                # If re-evaluation is running, skip symbol processing to avoid
                # race conditions with _cycle_spent reset and concurrent budget
                # calculations, but still allow state saving to run.
                if not engine._reevaluate_running:
                    symbols_to_process = await self._evaluation_scheduler.get_symbols_to_process(now)

                    if symbols_to_process:
                        # Check if trading is paused (skip BUY signals)
                        paused = await asyncio.to_thread(engine.redis.get, "trading:paused")
                        trading_paused = paused is not None and paused == "1"

                        async def _process_symbol_task(entry: Dict[str, str]):
                            async with engine._symbol_processing_semaphore:
                                sym = entry["symbol"]
                                try:
                                    await asyncio.wait_for(
                                        engine.event_bus.request("process_symbol", entry, trading_paused=trading_paused),
                                        timeout=settings.LLM_TIMEOUT + 10
                                    )
                                    async with engine._eval_state_lock:
                                        engine._last_strategy_eval[sym] = now
                                except asyncio.TimeoutError:
                                    logger.error(f"Timeout processing symbol {sym} – skipping. Will retry on next loop iteration.")
                                    if engine.notifier:
                                        await engine.notifier.send_notification(
                                            f"⏱️ Processing timeout for {sym} – skipping this cycle.",
                                            summary={"symbol": sym, "action": "SKIP", "reason": "Processing timeout"}
                                        )
                                except asyncio.CancelledError:
                                    raise
                                except (ConnectionError, TimeoutError, OSError) as e:
                                    logger.warning(f"Network/IO error processing symbol {sym}: {type(e).__name__}: {e}")
                                except Exception as e:
                                    logger.error(f"Error processing symbol {sym}: {e}", exc_info=True)
                                    await engine._record_unexpected_exception("process_symbol_task", e)

                        await asyncio.gather(*[_process_symbol_task(entry) for entry in symbols_to_process])

                # Save state periodically (every 5 minutes) when dirty
                if now - engine._last_state_save > 300 and engine.shared_state._state_dirty:
                    await engine._state_persistence.save_state()
                    engine._last_state_save = now
                    engine.shared_state._state_dirty = False

            except asyncio.CancelledError:
                raise
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.warning(f"Engine loop network/IO error: {type(e).__name__}: {e}")
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Engine loop error: {type(e).__name__}: {e}", exc_info=True)
                await engine._record_unexpected_exception("engine_loop", e)
                await asyncio.sleep(5)

    async def _market_clock_monitor(self):
        """Periodically check market clock and pause/resume trading based on market open/close."""
        engine = self.engine
        await asyncio.sleep(5)  # initial delay
        while engine._running:
            try:
                clock = await engine._market_data_manager.get_clock()
                if clock is None:
                    await asyncio.sleep(30)
                    continue

                is_open = clock.is_open
                paused = await asyncio.to_thread(engine.redis.get, "trading:paused")

                if not is_open:
                    # Market closed – always pause trading immediately
                    now_rome = clock.timestamp.astimezone(ZoneInfo(settings.MARKET_TIMEZONE))
                    next_open_rome = clock.next_open.astimezone(ZoneInfo(settings.MARKET_TIMEZONE))
                    remaining_seconds = (clock.next_open - clock.timestamp).total_seconds()
                    if remaining_seconds > 3600:
                        hours = int(remaining_seconds // 3600)
                        minutes = int((remaining_seconds % 3600) // 60)
                        countdown_str = f"{hours}h {minutes}m"
                    elif remaining_seconds > 60:
                        minutes = int(remaining_seconds // 60)
                        seconds = int(remaining_seconds % 60)
                        countdown_str = f"{minutes}m {seconds}s"
                    else:
                        countdown_str = f"{int(remaining_seconds)}s"
                    reason = "Market closed"
                    # Check if we already sent the market closed notification
                    market_closed_raw = await asyncio.to_thread(engine.redis.get, "trading:market_closed")
                    already_market_closed = market_closed_raw is not None

                    # Set market closed flag and next open time
                    await asyncio.to_thread(engine.redis.set, "trading:market_closed", "1")
                    await asyncio.to_thread(engine.redis.set, "trading:market_next_open", clock.next_open.isoformat())

                    source_raw = await asyncio.to_thread(engine.redis.get, "trading:pause_source")
                    source = source_raw.decode() if isinstance(source_raw, bytes) else (source_raw or "")

                    if source != "llm":
                        from src.utils.pause_utils import set_trading_pause
                        await asyncio.to_thread(
                            set_trading_pause,
                            engine.redis,
                            "market_closed",
                            reason=reason,
                            set_pause_start=False,
                        )
                        logger.debug(f"Market closed, pausing trading. Reason: {reason}")
                        if engine.notifier and not already_market_closed:
                            await engine.notifier.send_notification(
                                f"⏸️ {reason}",
                                summary={"action": "PAUSE", "reason": reason}
                            )
                    elif not already_market_closed:
                        # Market closed but LLM already paused. Just notify.
                        if engine.notifier:
                            await engine.notifier.send_notification(
                                f"⏸️ Market closed (trading already paused by LLM).",
                                summary={"action": "PAUSE", "reason": "Market closed (LLM pause active)"}
                            )
                    # Invalidate clock cache so the next monitor cycle sees the updated state
                    engine._market_data_manager.invalidate_clock_cache()

                # --- Periodic countdown updates while market is closed ---
                # Only send updates if the market is currently closed
                market_closed_raw = await asyncio.to_thread(engine.redis.get, "trading:market_closed")
                is_market_closed_flag = market_closed_raw is not None
                if is_market_closed_flag and not is_open:
                    now_ts = time.time()
                    # Fetch pause source to customize the "opening soon" message
                    source_raw = await asyncio.to_thread(engine.redis.get, "trading:pause_source")
                    source = source_raw.decode() if isinstance(source_raw, bytes) else (source_raw or "")

                    # Recompute remaining seconds from the live clock (or fallback)
                    if clock is not None:
                        remaining_seconds = (clock.next_open - datetime.now(timezone.utc)).total_seconds()
                    else:
                        # Fallback to stored next_open
                        next_open_raw = await asyncio.to_thread(engine.redis.get, "trading:market_next_open")
                        if next_open_raw:
                            next_open_str = next_open_raw.decode() if isinstance(next_open_raw, bytes) else next_open_raw
                            next_open_dt = datetime.fromisoformat(next_open_str)
                            remaining_seconds = (next_open_dt - datetime.now(timezone.utc)).total_seconds()
                        else:
                            remaining_seconds = 0

                    # Periodic update every 30 minutes on weekdays, 2 hours on weekends
                    now_rome_check = clock.timestamp.astimezone(ZoneInfo(settings.MARKET_TIMEZONE)) if clock else None
                    is_weekend = now_rome_check.weekday() >= 5 if now_rome_check else False
                    notify_interval = 7200 if is_weekend else 1800
                    if now_ts - engine._last_market_closed_notify_time >= notify_interval:
                        if remaining_seconds > 0:
                            hours = int(remaining_seconds // 3600)
                            minutes = int((remaining_seconds % 3600) // 60)
                            if hours > 0:
                                countdown_str = f"{hours}h {minutes}m"
                            elif minutes > 0:
                                seconds = int(remaining_seconds % 60)
                                countdown_str = f"{minutes}m {seconds}s"
                            else:
                                countdown_str = f"{int(remaining_seconds)}s"
                            next_open_rome = clock.next_open.astimezone(ZoneInfo(settings.MARKET_TIMEZONE)) if clock else None
                            next_open_str = next_open_rome.strftime('%H:%M %d/%m/%Y') if next_open_rome else "?"
                            update_msg = (
                                f"⏸️ Market still closed. Reopens in {countdown_str} "
                                f"(at {next_open_str})"
                            )
                            if engine.notifier:
                                await engine.notifier.send_notification(
                                    update_msg,
                                    summary={"action": "PAUSE", "reason": update_msg}
                                )
                        engine._last_market_closed_notify_time = now_ts

                    # "Opening soon" alert when less than 5 minutes remain
                    if 0 < remaining_seconds <= 900 and not engine._market_opening_soon_notified:
                        minutes_left = int(remaining_seconds // 60)
                        if source == "llm":
                            soon_msg = f"⏰ Market opens in ~{minutes_left} minute(s) – trading will remain paused (LLM pause active)."
                        else:
                            soon_msg = f"⏰ Market opens in ~{minutes_left} minute(s) – trading will resume automatically."
                        if engine.notifier:
                            await engine.notifier.send_notification(
                                soon_msg,
                                summary={"action": "INFO", "reason": "Market opening soon"}
                            )
                        engine._market_opening_soon_notified = True
                        # Invalidate correlation matrix cache before pre-market re-evaluation
                        await asyncio.to_thread(engine.redis.delete, "reeval:correlation_matrix")
                        # Trigger pre-market re-evaluation so we're prepared with fresh signals
                        engine._force_reeval = True
                        engine._pre_market_reeval = True
                        engine._reeval_trigger.set()
                else:
                    # Market open – resume trading only if paused due to market closure.
                    # Respect LLM-initiated and manual pauses while the market is open.
                    # Always clear market-closed specific keys
                    await asyncio.to_thread(engine.redis.delete, "trading:market_closed")
                    await asyncio.to_thread(engine.redis.delete, "trading:market_next_open")

                    paused = await asyncio.to_thread(engine.redis.get, "trading:paused")
                    if paused:
                        source_raw = await asyncio.to_thread(engine.redis.get, "trading:pause_source")
                        source = source_raw.decode() if isinstance(source_raw, bytes) else (source_raw or "")
                        if source == "market_closed":
                            from src.utils.pause_utils import clear_trading_pause_keys
                            await asyncio.to_thread(clear_trading_pause_keys, engine.redis)
                            logger.info("Market opened, clearing market-closed pause (trading resumed).")
                            if engine.notifier:
                                await engine.notifier.send_notification(
                                    "▶️ Market opened, trading resumed.",
                                    summary={"action": "RESUME", "reason": "Market opened"}
                                )
                            # Invalidate correlation matrix cache on market open
                            await asyncio.to_thread(engine.redis.delete, "reeval:correlation_matrix")
                            # Only trigger re-evaluation when we actually resumed from a pause
                            engine._reeval_trigger.set()
                            # Invalidate clock cache so subsequent calls get fresh data
                            engine._market_data_manager.invalidate_clock_cache()
                        else:
                            logger.debug("Market open, but trading paused by '%s' – not clearing.", source)
                    else:
                        logger.debug("Market open, trading already active.")
                        # Do NOT trigger re-evaluation when already active — let the normal
                        # periodic interval handle it to avoid spamming re-evaluations every 60s.
                    # Reset the "opening soon" notification flag
                    engine._market_opening_soon_notified = False
                    # Reset the periodic countdown timer so the first update
                    # after the next market close is not skipped due to a stale timestamp.
                    engine._last_market_closed_notify_time = 0.0
            except asyncio.CancelledError:
                raise
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.warning(f"Market clock monitor network/IO error: {type(e).__name__}: {e}")
            except Exception as e:
                logger.error(f"Market clock monitor error: {type(e).__name__}: {e}", exc_info=True)
                await engine._record_unexpected_exception("market_clock_monitor", e)
            await asyncio.sleep(30)  # check every 30 seconds
