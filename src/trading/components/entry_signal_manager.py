"""Entry signal detection and pending entry management."""
import asyncio
import logging
import time
from typing import Any, Dict, Optional

from src.config.settings import settings
from src.database import get_indicators, get_ohlcv
from src.indicators import compute_all_indicators, compute_ema

logger = logging.getLogger(__name__)


class EntrySignalManager:
    """Handles entry signal detection and pending entry processing."""

    def __init__(self, engine, event_bus):
        self.engine = engine
        self.shared_state = engine.shared_state
        self.event_bus = event_bus

    async def detect_entry_signal(self, symbol: str, timeframe: str) -> bool:
        """Return True if a favourable entry condition is detected for the symbol.
        Uses recent OHLCV data from the database and compares with previous state."""
        engine = self.engine
        
        # Clean up stale entry signal state for symbols no longer tracked
        async with self.shared_state._eval_state_lock:
            active_symbols = {entry["symbol"] for entry in self.shared_state.current_symbols}
            active_symbols.update(self.shared_state.positions.keys())
            stale_keys = [s for s in self.shared_state._entry_signal_state if s not in active_symbols]
            for s in stale_keys:
                self.shared_state._entry_signal_state.pop(s, None)

        # Fetch pre-computed indicators from DB
        ind = await asyncio.to_thread(get_indicators, symbol, timeframe)

        # Still need candles for volume EMA computation and fallback
        # indicator computation
        db_candles = await asyncio.to_thread(
            get_ohlcv, symbol, timeframe, limit=50
        )
        tf_seconds = engine._timeframe_to_seconds(timeframe)
        min_candles = 5 if tf_seconds >= 2_592_000 else 26  # Long timeframes need fewer candles
        if len(db_candles) < min_candles:
            return False

        # If DB indicators are missing (common for long timeframes like
        # 1Y where indicators may not be stored), compute them
        # on-the-fly from the OHLCV candles so entry signal detection
        # still works.
        if not ind:
            raw_candles = [
                [c["timestamp"], c["open"], c["high"], c["low"], c["close"], c["volume"]]
                for c in db_candles
            ]
            try:
                ind = await asyncio.to_thread(compute_all_indicators, raw_candles)
            except (ValueError, TypeError, IndexError, ZeroDivisionError) as e:
                logger.debug(
                    f"Failed to compute indicators on-the-fly for {symbol} {timeframe}: {type(e).__name__}: {e}"
                )
                return False
            if not ind:
                return False

        closes = [c["close"] for c in db_candles]
        volumes = [c["volume"] for c in db_candles]

        # Retrieve previous state
        prev = self.shared_state._entry_signal_state.get(symbol, {})

        # Current values
        rsi = ind.get("rsi")
        macd_hist = ind.get("macd_hist")
        macd_val = ind.get("macd")
        macd_signal = ind.get("macd_signal")
        stoch_k = ind.get("stochastic_k")
        adx = ind.get("adx")
        plus_di = ind.get("plus_di")
        minus_di = ind.get("minus_di")
        bb_upper = ind.get("bb_upper")
        bb_lower = ind.get("bb_lower")
        bb_middle = ind.get("bb_middle")
        ema_9 = ind.get("ema_9")
        ema_21 = ind.get("ema_21")
        parabolic_sar = ind.get("parabolic_sar")
        ichimoku = ind.get("ichimoku")
        current_close = closes[-1] if closes else None

        # Volume EMA for spike detection (using talib via compute_ema).
        # Exclude the latest candle (which may be incomplete for intraday
        # timeframes) by using the second-to-last EMA value.
        volume_ema_list = compute_ema(volumes, 20)
        _raw_ema = volume_ema_list[-2] if len(volume_ema_list) >= 2 else None
        volume_ema = _raw_ema if _raw_ema is not None else 0.0

        # Store current state for next cycle
        new_state = {
            "rsi": rsi,
            "macd_hist": macd_hist,
            "macd_val": macd_val,
            "macd_signal": macd_signal,
            "stoch_k": stoch_k,
            "adx": adx,
            "plus_di": plus_di,
            "minus_di": minus_di,
            "bb_upper": bb_upper,
            "bb_lower": bb_lower,
            "bb_middle": bb_middle,
            "ema_9": ema_9,
            "ema_21": ema_21,
            "parabolic_sar": parabolic_sar,
            "ichimoku_cloud_top": ichimoku["cloud_top"] if ichimoku else None,
            "ichimoku_cloud_bottom": ichimoku["cloud_bottom"] if ichimoku else None,
            "close": current_close,
            "volume_ema": volume_ema,
        }
        self.shared_state._entry_signal_state[symbol] = new_state

        # --- Read LLM-defined thresholds from Redis (fallback to defaults) ---
        rsi_oversold = 30.0
        rsi_overbought = 70.0
        adx_moderate = 25.0
        bb_squeeze_width = 0.02
        try:
            raw = await engine.config_service.get_config("skip_eval_rsi_oversold")
            if raw:
                rsi_oversold = float(raw)
            raw = await engine.config_service.get_config("skip_eval_rsi_overbought")
            if raw:
                rsi_overbought = float(raw)
            raw = await engine.config_service.get_config("regime_adx_moderate")
            if raw:
                adx_moderate = float(raw)
            raw = await engine.config_service.get_config("regime_bb_squeeze_width")
            if raw:
                bb_squeeze_width = float(raw)
        except (ValueError, TypeError, ConnectionError, TimeoutError, OSError):
            pass

        prev_close = prev.get("close")

        # --- Long-term timeframe detection (>= 1 month) ---
        # Use trend reversal and regime shift logic instead of short-term
        # crossovers, which fire on every candle for long timeframes because
        # indicators change dramatically between candles (e.g., RSI 20→80).
        tf_seconds = engine._timeframe_to_seconds(timeframe)
        if tf_seconds >= 2_592_000:  # >= 1 month (1M, 3M, 6M, 1Y)
            # Volume confirmation: require the last complete candle's volume
            # to be above the 20-period EMA to validate long-term breakouts.
            volume_confirmed = (
                len(volumes) >= 2 and volume_ema > 0 and volumes[-2] > volume_ema
            )

            # 1. Trend direction reversal: +DI crosses above -DI
            prev_plus_di = prev.get("plus_di")
            prev_minus_di = prev.get("minus_di")
            if (prev_plus_di is not None and prev_minus_di is not None
                    and plus_di is not None and minus_di is not None
                    and prev_plus_di <= prev_minus_di and plus_di > minus_di):
                return True

            # 2. Trend initiation: ADX crosses above moderate threshold
            prev_adx = prev.get("adx")
            if (prev_adx is not None and adx is not None
                    and prev_adx <= adx_moderate and adx > adx_moderate
                    and plus_di is not None and minus_di is not None
                    and plus_di > minus_di):
                return True

            # 3. Major breakout: price breaks above Donchian upper channel
            donchian = ind.get("donchian_channels")
            if (donchian is not None and prev_close is not None
                    and current_close is not None and volume_confirmed):
                dc_upper = donchian.get("upper")
                if dc_upper is not None and prev_close <= dc_upper and current_close > dc_upper:
                    return True

            # 4. Ichimoku cloud breakout: price crosses above cloud top
            prev_cloud_top = prev.get("ichimoku_cloud_top")
            if (prev_cloud_top is not None and ichimoku is not None
                    and prev_close is not None and current_close is not None
                    and volume_confirmed):
                cloud_top = ichimoku.get("cloud_top")
                if cloud_top is not None and prev_close <= cloud_top and current_close > cloud_top:
                    return True

            # No long-term entry signal detected
            return False

        # --- Trend strength filter for medium/long-term timeframes ---
        # For timeframes >= 1 day, require sufficient overall trend strength
        # rather than relying solely on individual indicator thresholds.
        if tf_seconds >= 86_400:  # >= 1 day
            trend_score = 0
            if adx is not None and adx > adx_moderate:
                trend_score += 1
            if macd_hist is not None and macd_hist > 0:
                trend_score += 1
            if plus_di is not None and minus_di is not None and plus_di > minus_di:
                trend_score += 1
            if ema_9 is not None and ema_21 is not None and ema_9 > ema_21:
                trend_score += 1
            if current_close is not None and ema_21 is not None and current_close > ema_21:
                trend_score += 1
            
            # Require at least 3 out of 5 trend strength conditions
            if trend_score < 3:
                return False

        # --- Condition checks ---
        # 1. RSI oversold
        if rsi is not None and rsi < rsi_oversold:
            return True

        # 2. MACD histogram bullish crossover (was negative, now positive)
        prev_macd_hist = prev.get("macd_hist")
        if (prev_macd_hist is not None and macd_hist is not None
                and prev_macd_hist <= 0 and macd_hist > 0):
            return True

        # 3. RSI leaving oversold (momentum shift)
        prev_rsi = prev.get("rsi")
        if (prev_rsi is not None and rsi is not None
                and prev_rsi < rsi_oversold and rsi >= rsi_oversold):
            return True

        # 4. MACD line crossing above signal line (bullish crossover)
        prev_macd_val = prev.get("macd_val")
        prev_macd_signal = prev.get("macd_signal")
        if (prev_macd_val is not None and prev_macd_signal is not None
                and macd_val is not None and macd_signal is not None
                and prev_macd_val <= prev_macd_signal and macd_val > macd_signal):
            return True

        # 6. ADX rising above moderate threshold and +DI > -DI (trend start)
        prev_adx = prev.get("adx")
        if (adx is not None and plus_di is not None and minus_di is not None
                and plus_di > minus_di
                and prev_adx is not None and prev_adx <= adx_moderate and adx > adx_moderate):
            return True

        # 7. Bollinger Band squeeze breakout
        prev_bb_upper = prev.get("bb_upper")
        prev_bb_lower = prev.get("bb_lower")
        prev_bb_middle = prev.get("bb_middle")
        if (prev_bb_upper is not None and prev_bb_lower is not None and prev_bb_middle is not None
                and bb_upper is not None and bb_lower is not None and bb_middle is not None
                and prev_bb_middle > 0 and bb_middle > 0):
            prev_width = (prev_bb_upper - prev_bb_lower) / prev_bb_middle
            curr_width = (bb_upper - bb_lower) / bb_middle
            if prev_width < bb_squeeze_width and current_close is not None and current_close > bb_upper:
                return True

        # 8. Volume spike (last COMPLETE candle volume > 3 * EMA of volume)
        # Use the second-to-last candle to avoid false signals from the
        # latest candle which may still be forming (incomplete volume).
        if len(volumes) >= 2 and volume_ema > 0 and volumes[-2] > 3.0 * volume_ema:
            return True

        # 9. EMA9 crossing above EMA21 (golden cross)
        prev_ema_9 = prev.get("ema_9")
        prev_ema_21 = prev.get("ema_21")
        if (prev_ema_9 is not None and prev_ema_21 is not None
                and ema_9 is not None and ema_21 is not None
                and prev_ema_9 <= prev_ema_21 and ema_9 > ema_21):
            return True

        # 11. Parabolic SAR flip (from above price to below price → uptrend)
        prev_sar = prev.get("parabolic_sar")
        if (prev_sar is not None and parabolic_sar is not None
                and prev_close is not None and current_close is not None
                and prev_sar > prev_close and parabolic_sar < current_close):
            return True

        # 12. Ichimoku: price crossing above cloud
        prev_cloud_top = prev.get("ichimoku_cloud_top")
        prev_cloud_bottom = prev.get("ichimoku_cloud_bottom")
        if (prev_cloud_top is not None and prev_cloud_bottom is not None
                and ichimoku is not None
                and prev_close is not None and current_close is not None):
            cloud_top = ichimoku["cloud_top"]
            cloud_bottom = ichimoku["cloud_bottom"]
            # Previous close was below or inside cloud, current close above cloud top
            if prev_close <= cloud_top and current_close > cloud_top:
                return True

        return False

    async def check_entry_condition_once(
        self, symbol: str, condition: Dict[str, Any], timeframe: str
    ) -> bool:
        """Check a single entry condition immediately. Return True if met."""
        engine = self.engine
        etype = condition.get("type")
        if etype == "limit_price":
            target_price = condition["price"]
            try:
                tickers_map = await engine._market_data_manager._get_quotes_async([symbol.split("/")[0]], timeout=45.0)
                ticker = tickers_map.get(symbol.split("/")[0])
            except (ConnectionError, TimeoutError, OSError, ValueError, TypeError, KeyError):
                return False
            current_price = ticker.get("last", 0) if ticker else 0
            return current_price > 0 and current_price <= target_price

        elif etype == "rsi_threshold":
            target_rsi = condition["rsi_below"]
            ind = await asyncio.to_thread(get_indicators, symbol, timeframe)
            if ind:
                rsi = ind.get("rsi")
                return rsi is not None and rsi <= target_rsi
            return False

        elif etype == "delay":
            # Delay conditions are handled by _execute_delayed_entry, not the
            # pending-entries system. If we somehow reach here, treat as not met
            # so the deadline handler can deal with it.
            return False

        elif etype == "indicator_combo":
            conditions = condition["conditions"]
            ind = await asyncio.to_thread(get_indicators, symbol, timeframe)
            if not ind:
                return False
            # Mapping of indicator names the LLM can use to DB keys.
            # All scalar indicators stored in the indicators table are supported.
            _INDICATOR_KEYS = {
                "rsi": "rsi",
                "macd": "macd",
                "macd_signal": "macd_signal",
                "macd_hist": "macd_hist",
                "bb_upper": "bb_upper",
                "bb_middle": "bb_middle",
                "bb_lower": "bb_lower",
                "ema_9": "ema_9",
                "ema_21": "ema_21",
                "stochastic_k": "stochastic_k",
                "stochastic_d": "stochastic_d",
                "adx": "adx",
                "plus_di": "plus_di",
                "minus_di": "minus_di",
                "obv": "obv",
                "mfi": "mfi",
                "cci": "cci",
                "williams_r": "williams_r",
                "parabolic_sar": "parabolic_sar",
                "atr": "atr",
            }
            for cond in conditions:
                indicator_name = cond["indicator"]
                thresh = cond["threshold"]
                direction = cond["direction"]
                db_key = _INDICATOR_KEYS.get(indicator_name)
                if db_key is None:
                    logger.warning(
                        f"Unsupported indicator '{indicator_name}' in indicator_combo "
                        f"entry condition for {symbol}"
                    )
                    return False
                val = ind.get(db_key)
                if val is None:
                    return False
                if direction == "below" and val > thresh:
                    return False
                if direction == "above" and val < thresh:
                    return False
            return True

        return False

    async def process_pending_entry(self, symbol: str, now: float) -> None:
        """Process a single pending entry: check timeout and condition, execute if met."""
        engine = self.engine
        async with self.shared_state._pending_entries_lock:
            entry = self.shared_state._pending_entries.get(symbol)
            if entry is None:
                return
            entry_tf = entry.get("timeframe")
            stock_name = await engine._market_data_manager.get_stock_name(symbol)
            display_symbol = engine._format_symbol_display(symbol, stock_name, entry_tf)
            if now >= entry["deadline"]:
                # Timeout – clear and notify
                logger.info(f"Entry condition timeout for {symbol}")
                del self.shared_state._pending_entries[symbol]
                self.shared_state._state_dirty = True
                _timed_out = True
                _signal = None
            else:
                # Check the condition (non‑blocking)
                condition_met = await self.check_entry_condition_once(
                    symbol, entry["condition"], entry["timeframe"]
                )
                if condition_met:
                    logger.info(f"Entry condition met for {symbol}, executing BUY")
                    # Remove from pending before executing to avoid re‑trigger
                    _signal = entry["signal"]
                    del self.shared_state._pending_entries[symbol]
                    self.shared_state._state_dirty = True
                    _timed_out = False
                else:
                    return  # condition not met, keep pending

        if _timed_out:
            if engine.notifier:
                await engine.notifier.send_notification(
                    f"⏭️ Entry condition timeout for {display_symbol} – skipping BUY.",
                    summary={
                        "symbol": symbol,
                        "action": "SKIP",
                        "reason": "Entry condition timeout",
                    }
                )
            return

        # condition_met is True here
        # Check trading pause again (may have changed)
        paused = await asyncio.to_thread(engine.redis.get, "trading:paused")
        if paused:
            logger.info(f"Ignoring queued BUY {symbol}: trading is now paused.")
            if engine.notifier:
                await engine.notifier.send_notification(
                    f"⏸️ Queued BUY for {display_symbol} skipped – trading paused.",
                    summary={"symbol": symbol, "action": "SKIP", "reason": "Trading paused"}
                )
        else:
            await self.event_bus.publish(
                "execute_signal",
                symbol,
                _signal,
                timeframe=entry_tf,
                atr=None,
            )
