"""Monitors market conditions to trigger immediate symbol re-evaluation."""
import asyncio
import logging
import time

from src.config.settings import settings
from src.database import get_indicators, get_ohlcv

logger = logging.getLogger(__name__)


class ReevalMarketConditionMonitor:
    """Checks for market conditions that warrant more frequent symbol re-evaluation.

    Triggers re-evaluation when:
    - Significant news sentiment shifts are detected on tracked symbols
    - Unusually active market (many stocks with large daily price movements)
    - Extreme indicator values or Bollinger Band squeeze breakouts on tracked symbols
    """

    def __init__(self, engine, event_bus):
        self.engine = engine
        self.shared_state = engine.shared_state
        self.event_bus = event_bus
        self.event_bus.subscribe("check_market_conditions", self.check_market_conditions)

    async def check_market_conditions(self) -> None:
        """Check for market conditions that warrant more frequent symbol re-evaluation.

        Triggers re-evaluation when:
        - Significant news sentiment shifts are detected on tracked symbols
        - Unusually active market (many stocks with large daily price movements)
        - Extreme indicator values or Bollinger Band squeeze breakouts on tracked symbols
        """
        engine = self.engine
        # Respect a cooldown so we don't re-evaluate too frequently
        last_triggered_key = "trading:last_triggered_reeval"
        last_triggered = await asyncio.to_thread(engine.redis.get, last_triggered_key)
        if last_triggered:
            elapsed = time.time() - float(last_triggered)
            if elapsed < settings.TRIGGERED_REEVALUATION_COOLDOWN:
                return

        should_trigger = False

        # 1. Significant news sentiment shift on tracked symbols
        if settings.NEWS_ENABLED and self.shared_state.current_symbols:
            for entry in self.shared_state.current_symbols:
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
                                logger.info(f"Significant sentiment shift for {symbol}, triggering re-evaluation")
                                should_trigger = True
                                # Update the baseline only when a trigger fires
                                if current_compound is not None:
                                    await asyncio.to_thread(
                                        engine.redis.setex, prev_key, 3600, str(current_compound)
                                    )
                                break
                        else:
                            # No previous baseline — initialize it
                            if current_compound is not None:
                                await asyncio.to_thread(
                                    engine.redis.setex, prev_key, 3600, str(current_compound)
                                )
                except Exception:
                    continue

        # 2. Unusually active market (many stocks with >5% daily change)
        if not should_trigger:
            try:
                plain_assets = await engine._market_data_manager.get_tradable_assets()
                sample_pairs = [f"{sym}/{engine.base_currency}" for sym in plain_assets[:50]]
                plain_sample = [s.split("/")[0] for s in sample_pairs]
                quotes = await engine._market_data_manager._get_quotes_batched(plain_sample, timeout_per_chunk=45.0)
                large_movers = sum(
                    1 for q in quotes.values()
                    if abs(q.get("percentage") or 0) > 5.0
                )
                if large_movers >= 5:
                    logger.info(f"Unusually active market: {large_movers} stocks with >5% daily change, triggering re-evaluation")
                    should_trigger = True
            except Exception:
                pass

        # 3. Extreme indicator values or BB squeeze breakout on tracked symbols
        if not should_trigger:
            for entry in self.shared_state.current_symbols:
                symbol = entry["symbol"]
                tf = entry["timeframe"]
                try:
                    # Fetch pre-computed indicators from DB
                    ind = await asyncio.to_thread(get_indicators, symbol, tf)
                    if not ind:
                        continue

                    # Extreme RSI — use LLM-configured thresholds (fallback to 20/80)
                    rsi_oversold = 20.0
                    rsi_overbought = 80.0
                    try:
                        raw = await engine.config_service.get_config("skip_eval_rsi_oversold")
                        if raw:
                            rsi_oversold = float(raw)
                        raw = await engine.config_service.get_config("skip_eval_rsi_overbought")
                        if raw:
                            rsi_overbought = float(raw)
                    except (ValueError, TypeError, ConnectionError, TimeoutError, OSError):
                        pass
                    rsi = ind.get("rsi")
                    if rsi is not None and (rsi < rsi_oversold or rsi > rsi_overbought):
                        logger.info(f"Extreme RSI ({rsi:.1f}) for {symbol}, triggering re-evaluation")
                        should_trigger = True
                        break

                    # Bollinger Band squeeze breakout
                    bb_upper = ind.get("bb_upper")
                    bb_lower = ind.get("bb_lower")
                    bb_middle = ind.get("bb_middle")
                    if bb_upper and bb_lower and bb_middle and bb_middle > 0:
                        bb_width = (bb_upper - bb_lower) / bb_middle
                        bb_squeeze_width = 0.02
                        try:
                            raw = await engine.config_service.get_config("regime_bb_squeeze_width")
                            if raw:
                                bb_squeeze_width = float(raw)
                        except (ValueError, TypeError, ConnectionError, TimeoutError, OSError):
                            pass
                        if bb_width < bb_squeeze_width:
                            db_candles = await asyncio.to_thread(get_ohlcv, symbol, tf, limit=1)
                            if db_candles:
                                current_close = db_candles[-1]["close"]
                                if current_close > bb_upper or current_close < bb_lower:
                                    logger.info(f"Bollinger Band squeeze breakout for {symbol}, triggering re-evaluation")
                                    should_trigger = True
                                    break
                except Exception as e:
                    logger.debug(f"check_market_conditions: indicator check failed for {symbol}: {type(e).__name__}: {e}")
                    continue

        if should_trigger:
            logger.info("Market condition trigger fired – forcing symbol re-evaluation")
            # Invalidate correlation matrix cache due to significant market changes
            await asyncio.to_thread(engine.redis.delete, "reeval:correlation_matrix")
            if engine.notifier:
                await engine.notifier.send_notification(
                    "🔄 Market conditions changed – triggering immediate symbol re-evaluation.",
                    summary={"action": "INFO", "reason": "Market condition triggered re-evaluation"}
                )
            engine._force_reeval = True
            engine._reeval_trigger.set()
