import asyncio
import json
import logging
import time
from typing import Optional, Tuple

from src.config.settings import settings
from src.llm.cache import _should_use_primary_model
from src.utils.health_metrics import health_metrics

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
