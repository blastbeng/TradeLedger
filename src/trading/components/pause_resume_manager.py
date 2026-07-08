"""Pause and resume trading decision management."""
import asyncio
import json
import logging
import time

from src.config.settings import settings
from src.llm.cache import get_cached_llm_response, compute_market_hash
from src.llm.prompts import compact_prompt, build_system_prompt

logger = logging.getLogger(__name__)


class PauseResumeManager:
    """Handles the LLM decision to pause or resume trading."""

    def __init__(self, signal_processor):
        self.sp = signal_processor
        self.engine = signal_processor.engine
        self.event_bus = signal_processor.event_bus

    async def check_pause_resume_decision(self) -> None:
        """When trading is paused, ask the LLM whether to resume (lightweight)."""
        engine = self.engine
        async with engine._symbol_reeval_lock:
            # Only run if actually paused
            paused_raw = await asyncio.to_thread(engine.redis.get, "trading:paused")
            if not paused_raw or paused_raw != "1":
                return

            # Only handle LLM-initiated pauses. Manual pauses are not subject to auto-resume logic.
            source_raw = await asyncio.to_thread(engine.redis.get, "trading:pause_source")
            source = source_raw.decode() if isinstance(source_raw, bytes) else (source_raw or "")
            if source != "llm":
                logger.info("Pause/resume check skipped: pause was not initiated by LLM (source=%s).", source or "unknown")
                return

            # Read LLM-decided pause recovery settings from Redis
            max_keep = settings.PAUSE_MAX_CONSECUTIVE_KEEP
            force_resume_mult = settings.PAUSE_FORCE_RESUME_RISK_MULTIPLIER
            try:
                raw = await engine.config_service.get_config("pause_max_consecutive_keep")
                if raw:
                    max_keep = int(raw)
                raw = await engine.config_service.get_config("pause_force_resume_risk_multiplier")
                if raw:
                    force_resume_mult = float(raw)
            except (ValueError, TypeError, ConnectionError, TimeoutError, OSError):
                pass

            # Gather minimal market context
            benchmark_price = None
            try:
                tickers_map = await engine._market_data_manager._get_quotes_async([settings.BENCHMARK_SYMBOL], timeout=45.0)
                benchmark_ticker = tickers_map.get(settings.BENCHMARK_SYMBOL)
                benchmark_price = benchmark_ticker.get("last") if benchmark_ticker else None
            except (ConnectionError, TimeoutError, OSError, ValueError, TypeError, KeyError):
                pass

            # Market breadth from Redis (already computed by background task)
            full_market_breadth = None
            try:
                raw = await asyncio.to_thread(engine.redis.get, "market:breadth:full")
                if raw:
                    full_market_breadth = json.loads(raw)
            except (ValueError, TypeError, ConnectionError, TimeoutError, OSError, json.JSONDecodeError):
                pass
            market_breadth = getattr(engine, '_market_breadth', None)

            # Current pause reason
            reason_raw = await asyncio.to_thread(engine.redis.get, "trading:pause_reason")
            pause_reason = reason_raw.decode() if isinstance(reason_raw, bytes) else (reason_raw or "")

            # --- Consecutive "keep paused" counter ---
            keep_key = "trading:pause:keep_count"
            keep_count_raw = await asyncio.to_thread(engine.redis.get, keep_key)
            try:
                keep_count = int(keep_count_raw) if keep_count_raw else 0
            except (ValueError, TypeError):
                keep_count = 0

            # Build a richer prompt with performance context
            perf = await engine.event_bus.request("compute_performance_metrics")
            daily_pnl = perf["equity_curve"].get("daily_pnl", 0.0)
            total_pnl = perf["equity_curve"].get("total_pnl", 0.0)
            consecutive_losses = perf["equity_curve"].get("consecutive_losses", 0)
            drawdown_pct = perf["equity_curve"].get("drawdown_pct", 0.0)

            # Compute total unrealized P&L of open positions
            total_unrealized_pnl = 0.0
            if engine.positions:
                open_symbols = list(engine.positions.keys())
                base_symbols = [s.split("/")[0] for s in open_symbols]
                try:
                    tickers_map = await engine._market_data_manager._get_quotes_async(base_symbols, timeout=45.0)
                    for sym, pos in engine.positions.items():
                        base_sym = sym.split("/")[0]
                        ticker = tickers_map.get(base_sym)
                        if ticker and ticker.get("last"):
                            current_price = ticker["last"]
                            entry_price = pos.get("price", 0.0)
                            amount = pos.get("amount", 0.0)
                            total_unrealized_pnl += (current_price - entry_price) * amount
                except (ConnectionError, TimeoutError, OSError, ValueError, TypeError, KeyError):
                    pass

            prompt_parts = [
                "Trading is currently paused.",
            ]
            if pause_reason:
                prompt_parts.append(f"Pause reason: {pause_reason}")
            prompt_parts.append(f"Account P&L: daily={daily_pnl:.4f}, total={total_pnl:.4f}, drawdown={drawdown_pct:.2f}%")
            if engine.positions:
                prompt_parts.append(f"Unrealized P&L of open positions: {total_unrealized_pnl:.4f}")
            if consecutive_losses > 0:
                prompt_parts.append(f"Consecutive losing trades: {consecutive_losses}")
            if benchmark_price is not None:
                prompt_parts.append(f"Benchmark ({settings.BENCHMARK_SYMBOL}) price: {benchmark_price}")
            if market_breadth:
                prompt_parts.append(f"Market breadth (top stocks): {market_breadth['positive_pct']}% positive")
            if full_market_breadth:
                prompt_parts.append(f"Full market breadth: {full_market_breadth['positive_pct']}% positive")

            # Check if this is a recent auto-resume situation
            last_auto_resume_raw = await asyncio.to_thread(engine.redis.get, "trading:last_auto_resume")
            if last_auto_resume_raw:
                try:
                    last_auto_resume_ts = float(last_auto_resume_raw)
                    seconds_since = time.time() - last_auto_resume_ts
                    if seconds_since < 3600:  # within the last hour
                        minutes_since = seconds_since / 60
                        prompt_parts.append(
                            f"Trading was auto‑resumed {minutes_since:.1f} minutes ago. "
                            "Market conditions may not have changed significantly. "
                            "Only resume if there is clear, concrete improvement in the data above."
                        )
                except (ValueError, TypeError):
                    pass

            # --- Consecutive keep warning and recovery nudge ---
            if keep_count > 0:
                prompt_parts.append(
                    f"You have chosen to keep trading paused {keep_count} time(s) in a row. "
                    f"If you keep it paused {max_keep} times consecutively, the engine will "
                    f"force‑resume trading with a reduced global risk multiplier of "
                    f"{force_resume_mult} to attempt recovery."
                )

            prompt_parts.append(
                "If the account is in drawdown or has consecutive losses, consider resuming "
                "with a **reduced global risk multiplier** (e.g., 0.3–0.5) instead of staying "
                "paused indefinitely. This allows the bot to cautiously seek small profitable "
                "trades to recover, while limiting downside. You can provide an optional "
                "`global_risk_multiplier` field in your JSON response (0.0–1.0) to set the "
                "risk level upon resume. If you omit it, the current multiplier (or 1.0) will be used. "
                "If you are even slightly uncertain, **resume trading** with a very small "
                "`global_risk_multiplier` (e.g., 0.15–0.25). This lets the bot probe for "
                "opportunities with minimal risk. Staying paused forever guarantees zero recovery."
            )

            prompt = (
                "\n".join(prompt_parts)
                + "\n\nShould we resume trading now? Reply with a JSON object: "
                '{"resume_trading": true/false, "reason": "short explanation", '
                '"global_risk_multiplier": 0.0-1.0 (optional)}'
                + "\n\n**Important:** Only resume if you see specific, high‑confidence opportunities. "
                "If conditions are still poor, you may keep trading paused, but remember that "
                "staying paused forever prevents any recovery. A cautious resume with a low risk "
                "multiplier is often better than doing nothing."
            )

            pause_resume_complexity = self.sp.model_tier_manager.compute_prompt_complexity(
                num_candidates=0,
                market_breadth=market_breadth,
                fear_greed=None,
                volatility_percentile=None,
                sentiment_trend_magnitude=None,
                conflicting_signals=False,
                is_critical=False,
            )
            effective_temp = self.sp.model_tier_manager._get_effective_temperature("actuator", pause_resume_complexity)

            try:
                pause_result = await asyncio.wait_for(
                    asyncio.to_thread(
                        get_cached_llm_response, compact_prompt(prompt), compact_prompt(build_system_prompt()), 120,
                        model_type="actuator",
                        temperature=effective_temp,
                        market_hash=compute_market_hash({"pause_resume_prompt": prompt}),
                    ),
                    timeout=settings.LLM_TIMEOUT
                )
                response = pause_result["response"]
                llm_provider = pause_result["provider"]
                llm_model = pause_result["model"]
                decision = json.loads(response)
            except (asyncio.TimeoutError, ConnectionError, TimeoutError, OSError, ValueError, TypeError, RuntimeError, json.JSONDecodeError) as e:
                logger.warning(f"Pause/resume LLM call failed: {e}")
                # Track consecutive failures in Redis
                fail_key = "trading:pause:llm_fail_count"
                current_fails = await asyncio.to_thread(engine.redis.incr, fail_key)
                await asyncio.to_thread(engine.redis.expire, fail_key, 3600)
                _min_pause = settings.MIN_LLM_PAUSE_DURATION
                try:
                    raw = await engine.config_service.get_config("min_llm_pause_duration")
                    if raw:
                        _min_pause = int(raw)
                except (ValueError, TypeError, ConnectionError, TimeoutError, OSError):
                    pass
                if engine.notifier:
                    await engine.notifier.send_notification(
                        f"⚠️ Could not reach LLM to decide pause/resume (failure #{current_fails}). "
                        f"Auto‑resume will be attempted after {_min_pause}s if LLM stays silent.",
                        summary={"action": "INFO", "reason": "LLM pause-resume call failed"}
                    )
                # If we failed 3 times in a row, force‑resume (optional but safe)
                if current_fails >= 3:
                    # Double-check source before force-resuming
                    fail_source = await asyncio.to_thread(engine.redis.get, "trading:pause_source")
                    if fail_source and (fail_source.decode() if isinstance(fail_source, bytes) else fail_source) != "llm":
                        logger.warning("Force-resume on LLM failure skipped: pause source is not LLM.")
                        return
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
                    await asyncio.to_thread(engine.redis.delete, fail_key)
                    # --- Also reset keep counter and set force‑resume risk multiplier ---
                    await asyncio.to_thread(engine.redis.delete, keep_key)
                    await engine._set_global_risk_multiplier(force_resume_mult)
                    engine._reeval_trigger.set()
                    if engine.notifier:
                        await engine.notifier.send_notification(
                            "▶️ Trading auto‑resumed because LLM could not be reached for pause decision. "
                            f"Global risk multiplier set to {force_resume_mult}.",
                            summary={"action": "RESUME", "reason": "LLM pause-resume failures exceeded limit"}
                        )
                return

            resume_trading = decision.get("resume_trading")
            reason = decision.get("reason", "")

            if resume_trading is True:
                # Source is already verified as "llm" by the early check at the top of this method.

                # Check minimum LLM pause duration
                llm_pause_time_raw = await asyncio.to_thread(engine.redis.get, "trading:llm_pause_time")
                if llm_pause_time_raw:
                    try:
                        llm_pause_time = float(llm_pause_time_raw)
                        _min_pause = settings.MIN_LLM_PAUSE_DURATION
                        try:
                            raw = await engine.config_service.get_config("min_llm_pause_duration")
                            if raw:
                                _min_pause = int(raw)
                        except (ValueError, TypeError, ConnectionError, TimeoutError, OSError):
                            pass
                        if time.time() - llm_pause_time < _min_pause:
                            remaining = _min_pause - (time.time() - llm_pause_time)
                            logger.info(f"Ignoring LLM resume request: minimum pause duration not elapsed ({remaining:.0f}s remaining).")
                            if engine.notifier:
                                await engine.notifier.send_notification(
                                    f"⏸️ LLM resume request ignored: minimum pause duration "
                                    f"({_min_pause}s) not yet elapsed ({remaining:.0f}s remaining).",
                                    summary={"action": "RESUME", "reason": f"LLM resume blocked by minimum pause duration ({_min_pause}s)", "model_type": "actuator"}
                                )
                            return
                    except (ValueError, TypeError):
                        pass

                # --- Apply optional global_risk_multiplier from LLM ---
                global_mult_raw = decision.get("global_risk_multiplier")
                applied_mult = None
                if global_mult_raw is not None:
                    try:
                        mult_val = float(global_mult_raw)
                        if 0.0 <= mult_val <= 1.0:
                            await engine._set_global_risk_multiplier(mult_val)
                            logger.info(f"LLM set global risk multiplier on resume: {mult_val}")
                            applied_mult = mult_val
                        else:
                            logger.warning(f"Invalid global_risk_multiplier in resume decision: {global_mult_raw}")
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid global_risk_multiplier value: {global_mult_raw}")

                # Resume trading
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
                # Reset the keep counter
                await asyncio.to_thread(engine.redis.delete, keep_key)
                logger.info("LLM decided to resume trading.")
                engine._reeval_trigger.set()
                if engine.notifier:
                    reason_text = f" – {reason}" if reason else ""
                    mult_text = f" (risk multiplier: {applied_mult})" if applied_mult is not None else ""
                    await engine.notifier.send_notification(
                        f"▶️ Trading resumed by LLM decision{reason_text}{mult_text}",
                        summary={"action": "RESUME", "reason": f"LLM resume request: {reason}" if reason else "LLM resume request", "model_type": "actuator", "llm_provider": llm_provider, "llm_model": llm_model}
                    )
            elif resume_trading is False:
                # LLM wants to stay paused – optionally update reason
                if reason:
                    await asyncio.to_thread(engine.redis.set, "trading:pause_reason", reason)

                # Increment consecutive keep counter
                new_keep_count = await asyncio.to_thread(engine.redis.incr, keep_key)
                # Set a TTL so it doesn't persist forever (e.g., 24h)
                await asyncio.to_thread(engine.redis.expire, keep_key, 86400)

                if new_keep_count >= max_keep:
                    # Double-check that the pause is still LLM-initiated (should always be true here)
                    current_source = await asyncio.to_thread(engine.redis.get, "trading:pause_source")
                    if current_source and (current_source.decode() if isinstance(current_source, bytes) else current_source) != "llm":
                        logger.warning("Force-resume skipped: pause source changed to non-LLM.")
                        return

                    # --- Drawdown circuit breaker: do not force-resume in significant drawdown ---
                    max_drawdown = settings.PAUSE_FORCE_RESUME_MAX_DRAWDOWN_PCT
                    if drawdown_pct >= max_drawdown:
                        logger.warning(
                            f"Force-resume blocked: account drawdown {drawdown_pct:.2f}% "
                            f"exceeds circuit breaker threshold {max_drawdown:.2f}%. "
                            f"Keeping trading paused for safety."
                        )
                        if engine.notifier:
                            await engine.notifier.send_notification(
                                f"⛔ Force-resume blocked: drawdown {drawdown_pct:.2f}% exceeds "
                                f"circuit breaker threshold {max_drawdown:.2f}%. "
                                f"Trading stays paused to protect capital. "
                                f"LLM has kept it paused {new_keep_count} time(s).",
                                summary={
                                    "action": "PAUSE",
                                    "reason": f"Force-resume blocked by drawdown circuit breaker ({drawdown_pct:.2f}% >= {max_drawdown:.2f}%)",
                                    "model_type": "actuator",
                                    "llm_provider": llm_provider,
                                    "llm_model": llm_model,
                                }
                            )
                        # Do NOT force-resume; let the LLM continue deciding
                        return

                    logger.warning(
                        f"LLM kept trading paused {new_keep_count} times consecutively – "
                        f"forcing resume with risk multiplier {force_resume_mult}."
                    )
                    # Force resume
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
                    await asyncio.to_thread(engine.redis.delete, keep_key)
                    await engine._set_global_risk_multiplier(force_resume_mult)
                    engine._reeval_trigger.set()
                    if engine.notifier:
                        await engine.notifier.send_notification(
                            f"▶️ Trading force‑resumed after {new_keep_count} consecutive pauses. "
                            f"Global risk multiplier set to {force_resume_mult}.",
                            summary={
                                "action": "RESUME",
                                "reason": f"Force resume after {new_keep_count} consecutive keep-paused decisions",
                                "model_type": "actuator",
                            }
                        )
                else:
                    logger.info(f"LLM decided to keep trading paused. Reason: {reason} (keep count: {new_keep_count}/{max_keep})")
                    if engine.notifier:
                        reason_text = f" – {reason}" if reason else ""
                        await engine.notifier.send_notification(
                            f"⏸️ LLM decided to keep trading paused{reason_text} "
                            f"({new_keep_count}/{max_keep} consecutive keeps)",
                            summary={"action": "PAUSE", "reason": f"LLM keep paused: {reason}" if reason else "LLM keep paused", "model_type": "actuator", "llm_provider": llm_provider, "llm_model": llm_model}
                        )
            else:
                logger.warning(f"Invalid resume_trading value in LLM response: {resume_trading}")
