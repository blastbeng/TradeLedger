"""Handles LLM pause/resume decisions and global risk multiplier settings."""
import asyncio
import logging
import time
from typing import Any, Dict, Optional, Tuple

from src.config.settings import settings

logger = logging.getLogger(__name__)


class ReevalPauseResumeManager:
    """Manages LLM pause/resume decisions and global risk multiplier."""

    def __init__(self, engine, event_bus):
        self.engine = engine
        self.event_bus = event_bus

    async def handle_pause_resume_and_risk_multiplier(
        self,
        parsed: Dict[str, Any],
        pause_trading: Optional[bool],
        trading_paused_bool: bool,
    ) -> Tuple[Optional[bool], str, Optional[Any]]:
        """Handle LLM pause/resume decision and global risk multiplier setting.

        Returns (pause_trading, pause_reason, pause_duration) — pause_trading
        may be modified to None if an auto-resume cooldown is active.
        """
        engine = self.engine
        pause_reason = parsed.get("pause_reason", "")
        pause_duration = parsed.get("pause_duration_seconds")

        # --- Auto-resume cooldown: ignore pause requests shortly after an auto-resume ---
        cooldown_active = await asyncio.to_thread(engine.redis.get, "trading:auto_resume_cooldown")
        if cooldown_active and pause_trading is True:
            logger.info(
                "Ignoring LLM pause request because auto‑resume cooldown is active "
                "(trading was recently auto‑resumed)."
            )
            pause_trading = None  # treat as "no decision"

        skip_resume = False
        if pause_trading is not None:
            if isinstance(pause_trading, bool):
                if pause_trading:
                    # Only pause if not already manually paused
                    current_source = await asyncio.to_thread(engine.redis.get, "trading:pause_source")
                    if current_source and current_source == "manual":
                        logger.info("LLM pause request ignored because trading is manually paused.")
                    else:
                        await asyncio.to_thread(engine.redis.set, "trading:paused", "1")
                        await asyncio.to_thread(engine.redis.set, "trading:pause_source", "llm")
                        await asyncio.to_thread(engine.redis.set, "trading:pause_start", str(time.time()))
                        await asyncio.to_thread(engine.redis.set, "trading:llm_pause_time", str(time.time()))
                        # Fallback if LLM did not provide pause_duration_seconds
                        if pause_duration is None:
                            _min_pause = settings.MIN_LLM_PAUSE_DURATION
                            try:
                                raw = await engine.config_service.get_config("min_llm_pause_duration")
                                if raw:
                                    _min_pause = int(raw)
                            except (ValueError, TypeError, ConnectionError, TimeoutError, OSError):
                                pass
                            pause_duration = _min_pause
                            await asyncio.to_thread(
                                engine.redis.setex, "trading:pause_duration", 7 * 24 * 3600, str(int(pause_duration))
                            )
                        if pause_reason:
                            await asyncio.to_thread(engine.redis.set, "trading:pause_reason", pause_reason)
                        logger.info("LLM requested to pause trading.")
                else:
                    # LLM requests resume – only allowed if the pause was LLM-initiated
                    current_source = await asyncio.to_thread(engine.redis.get, "trading:pause_source")
                    if current_source and current_source != "llm":
                        logger.info("LLM resume request ignored because pause was not initiated by LLM.")
                    else:
                        if trading_paused_bool:
                            # Determine the required pause duration:
                            # - the LLM-set pause_duration_seconds (if any) stored in Redis
                            # - but never less than MIN_LLM_PAUSE_DURATION
                            pause_start_raw = await asyncio.to_thread(engine.redis.get, "trading:pause_start")
                            pause_duration_raw = await asyncio.to_thread(engine.redis.get, "trading:pause_duration")
                            required_pause = settings.MIN_LLM_PAUSE_DURATION
                            try:
                                raw = await engine.config_service.get_config("min_llm_pause_duration")
                                if raw:
                                    required_pause = int(raw)
                            except (ValueError, TypeError, ConnectionError, TimeoutError, OSError):
                                pass
                            if pause_duration_raw:
                                try:
                                    llm_set_duration = int(pause_duration_raw)
                                    required_pause = max(settings.MIN_LLM_PAUSE_DURATION, llm_set_duration)
                                except (ValueError, TypeError):
                                    pass

                            if pause_start_raw:
                                try:
                                    pause_start = float(pause_start_raw)
                                    elapsed = time.time() - pause_start
                                    if elapsed < required_pause:
                                        remaining = required_pause - elapsed
                                        logger.info(
                                            f"Ignoring LLM resume request: required pause duration "
                                            f"({required_pause}s) not yet elapsed ({remaining:.0f}s remaining)."
                                        )
                                        skip_resume = True
                                except (ValueError, TypeError):
                                    pass
                            if not skip_resume:
                                # Delete all pause keys
                                pause_keys = [
                                    "trading:paused",
                                    "trading:pause_source",
                                    "trading:pause_start",
                                    "trading:pause_duration",
                                    "trading:pause_reason",
                                    "trading:llm_pause_time",
                                ]
                                for key in pause_keys:
                                    await asyncio.to_thread(engine.redis.delete, key)
                                logger.info("LLM requested to resume trading.")
                                engine._reeval_trigger.set()
                        else:
                            # Trading is already active – LLM confirms to keep it active
                            logger.info("LLM decided to keep trading active (already active).")
            else:
                logger.warning(f"Invalid pause_trading value: {pause_trading}")

        # Store LLM-provided pause duration in Redis (if not already stored by pause logic)
        if pause_duration is not None and isinstance(pause_duration, (int, float)) and pause_duration > 0:
            await asyncio.to_thread(
                engine.redis.setex, "trading:pause_duration", 7 * 24 * 3600, str(int(pause_duration))
            )
            logger.info(f"LLM set pause duration: {pause_duration}s")
        elif pause_duration is not None:
            logger.warning(f"Invalid pause_duration_seconds: {pause_duration}")

        # Optional: LLM can set a global risk multiplier to scale all position sizes
        global_risk_mult = parsed.get("global_risk_multiplier")
        if global_risk_mult is not None:
            if isinstance(global_risk_mult, (int, float)) and 0.0 <= global_risk_mult <= 1.0:
                await engine._set_global_risk_multiplier(global_risk_mult)
                logger.info(f"LLM set global risk multiplier: {global_risk_mult}")
            else:
                logger.warning(f"Invalid global_risk_multiplier: {global_risk_mult}")

        return pause_trading, pause_reason, pause_duration
