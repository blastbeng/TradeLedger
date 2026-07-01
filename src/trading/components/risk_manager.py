"""Risk management component for the TradingEngine.

Handles stop-loss, take-profit, trailing stop, partial TP, dust sweep,
and other risk rule checks on open positions.
Extracted from TradingEngine to reduce class size and improve maintainability.
"""
import asyncio
import logging
import time
from typing import Any, Dict

from src.config.settings import settings
from src.database import insert_position_pnl_snapshot, get_indicators, get_latest_ohlcv_timestamp, get_ohlcv
from src.strategies.base import Signal
from src.utils.symbol_utils import is_btp_isin

logger = logging.getLogger(__name__)


class RiskManager:
    """Handles risk management checks for open positions."""

    def __init__(self, engine):
        self.engine = engine

    async def record_position_pnl_snapshots(self):
        """Record P&L snapshots for all open positions to the database."""
        engine = self.engine
        if not engine.positions:
            return
        pos_tickers = await engine._get_all_position_tickers_sync()
        now_ms = int(time.time() * 1000)
        for symbol, pos in engine.positions.items():
            try:
                t = pos_tickers.get(symbol)
                current_price = t['last'] if t and t.get('last') else pos.get('price', 0.0)
                amount = pos.get('amount', 0.0)
                entry_price = pos.get('price', 0.0)
                cost_basis = pos.get('cost_basis', amount * entry_price)
                position_value = amount * current_price
                unrealized_pnl = (current_price - entry_price) * amount
                pnl_pct = (unrealized_pnl / cost_basis) if cost_basis > 0 else 0.0
                # Realized P&L: sum of all closed sell trades for this symbol
                realized_pnl = sum(
                    t.get("realized_pnl", 0.0)
                    for t in engine.trade_history
                    if t.get("symbol") == symbol and t.get("side") == "sell"
                )
                await asyncio.to_thread(
                    insert_position_pnl_snapshot,
                    symbol=symbol,
                    timestamp=now_ms,
                    unrealized_pnl=round(unrealized_pnl, 6),
                    realized_pnl=round(realized_pnl, 6),
                    position_value=round(position_value, 6),
                    cost_basis=round(cost_basis, 6),
                    amount=amount,
                    current_price=current_price,
                    pnl_pct=round(pnl_pct, 6),
                )
            except Exception as e:
                logger.debug(f"Failed to record P&L snapshot for {symbol}: {e}")

    async def read_review_limits(self) -> Dict[str, int]:
        """Read LLM-decided review limits from Redis, falling back to settings defaults."""
        engine = self.engine
        max_sl_reviews = settings.MAX_STOP_LOSS_REVIEWS
        max_tp_reviews = settings.MAX_TAKE_PROFIT_REVIEWS
        max_partial_tp_reviews = settings.MAX_PARTIAL_TP_REVIEWS
        max_dust_sweep_reviews = settings.MAX_DUST_SWEEP_REVIEWS
        try:
            raw = await asyncio.to_thread(engine.redis.get, "trading:max_stop_loss_reviews")
            if raw:
                max_sl_reviews = int(raw)
            raw = await asyncio.to_thread(engine.redis.get, "trading:max_take_profit_reviews")
            if raw:
                max_tp_reviews = int(raw)
            raw = await asyncio.to_thread(engine.redis.get, "trading:max_partial_tp_reviews")
            if raw:
                max_partial_tp_reviews = int(raw)
            raw = await asyncio.to_thread(engine.redis.get, "trading:max_dust_sweep_reviews")
            if raw:
                max_dust_sweep_reviews = int(raw)
        except Exception:
            pass
        return {
            "max_sl_reviews": max_sl_reviews,
            "max_tp_reviews": max_tp_reviews,
            "max_partial_tp_reviews": max_partial_tp_reviews,
            "max_dust_sweep_reviews": max_dust_sweep_reviews,
        }

    async def check_hard_stop(
        self,
        symbol: str,
        pos: Dict[str, Any],
        current_price: float,
        display_symbol: str,
    ) -> bool:
        """Check if the position has exceeded the hard maximum loss threshold.

        Returns True if the hard stop was triggered (caller should skip to
        the next position), False otherwise.
        """
        engine = self.engine
        entry_price = pos["price"]
        if entry_price <= 0:
            return False
        unrealized_loss_pct = (entry_price - current_price) / entry_price
        _is_btp = is_btp_isin(symbol)
        _hard_max_loss = settings.BTP_HARD_MAX_LOSS_PCT if _is_btp else settings.HARD_MAX_LOSS_PCT
        if unrealized_loss_pct >= _hard_max_loss:
            logger.warning(
                f"Hard max loss threshold reached for {symbol}: "
                f"unrealized loss {unrealized_loss_pct:.2%} >= {_hard_max_loss:.2%}. Forcing SELL."
            )
            if engine.notifier:
                await engine.notifier.send_notification(
                    f"⛔ Hard stop for {display_symbol}: unrealized loss {unrealized_loss_pct:.2%} "
                    f"exceeds maximum {_hard_max_loss:.2%} – force selling.",
                    summary={
                        "symbol": symbol,
                        "action": "SELL",
                        "reason": "Hard maximum loss threshold",
                        "price": current_price,
                        "unrealized_loss_pct": round(unrealized_loss_pct, 4),
                        "exit_reason": "hard_max_loss",
                    }
                )
            await engine._execute_signal(
                symbol,
                Signal(action="SELL", confidence=1.0, reasoning="Hard maximum loss threshold exceeded"),
                exit_reason="hard_max_loss"
            )
            return True
        return False

    async def check_soft_stop(
        self,
        symbol: str,
        pos: Dict[str, Any],
        current_price: float,
        display_symbol: str,
    ) -> bool:
        """Check if the position has exceeded the maximum unrealized loss threshold.

        Returns True if the soft stop was triggered (caller should skip to
        the next position), False otherwise.
        """
        engine = self.engine
        max_ul_pct = pos.get("max_unrealized_loss_pct")
        if max_ul_pct is not None and max_ul_pct > 0:
            entry_price = pos["price"]
            if current_price <= entry_price * (1 - max_ul_pct):
                logger.info(f"Max unrealized loss reached for {symbol} ({max_ul_pct:.2%}). Closing position.")
                if engine.notifier:
                    await engine.notifier.send_notification(
                        f"📉 Soft stop triggered for {display_symbol} at {current_price:.4f} (max loss {max_ul_pct:.2%})",
                        summary={
                            "symbol": symbol,
                            "action": "SELL",
                            "reason": "Max unrealized loss",
                            "price": current_price,
                            "exit_reason": "max_unrealized_loss",
                        }
                    )
                await engine._execute_signal(
                    symbol,
                    Signal(action="SELL", confidence=1.0, reasoning="Max unrealized loss"),
                    exit_reason="max_unrealized_loss"
                )
                return True
        return False

    async def check_news_sentiment_exit(
        self,
        symbol: str,
        pos: Dict[str, Any],
        display_symbol: str,
    ) -> bool:
        """Check if negative news sentiment should trigger an exit.

        Returns True if the sentiment exit was triggered (caller should skip to
        the next position), False otherwise.
        """
        engine = self.engine
        news_threshold = pos.get("news_sentiment_exit_threshold")
        if news_threshold is not None and settings.NEWS_ENABLED:
            pos_tf = pos.get("timeframe")
            if pos_tf and engine._timeframe_to_seconds(pos_tf) >= 604_800:
                logger.debug(
                    f"Skipping news sentiment exit for {symbol}: "
                    f"long-term timeframe ({pos_tf}) ignores short-term sentiment."
                )
            else:
                # Clamp to non-positive: a positive threshold would trigger
                # an exit even when sentiment is mildly positive, which is
                # almost certainly not the LLM's intent.  Only negative
                # compound scores should trigger a sentiment-based exit.
                effective_threshold = min(float(news_threshold), 0.0)
                try:
                    agg = await engine._get_cached_sentiment(symbol)
                    if agg and agg["avg_compound"] < effective_threshold:
                        logger.info(
                            f"News sentiment exit for {symbol}: compound {agg['avg_compound']:.2f} < threshold {effective_threshold}"
                        )
                        if engine.notifier:
                            await engine.notifier.send_notification(
                                f"📰 Negative news exit for {display_symbol} (sentiment {agg['avg_compound']:.2f})",
                                summary={
                                    "symbol": symbol,
                                    "action": "SELL",
                                    "reason": "News sentiment exit",
                                    "sentiment": agg,
                                    "exit_reason": "news_sentiment_exit",
                                }
                            )
                        await engine._execute_signal(
                            symbol,
                            Signal(action="SELL", confidence=1.0, reasoning="News sentiment exit"),
                            exit_reason="news_sentiment_exit"
                        )
                        return True
                except Exception as e:
                    logger.info(f"News sentiment check failed for {symbol}: {e}")
        return False

    async def update_trailing_stop(
        self,
        symbol: str,
        pos: Dict[str, Any],
        current_price: float,
        display_symbol: str,
    ) -> None:
        """Update the trailing stop for a position if enabled and activated.

        Handles BTP warnings, activation threshold, highest-price tracking
        (including OHLCV candle highs), ATR-based (Chandelier Exit) and
        fixed-percentage trailing stops, and the stop-loss price update.
        """
        engine = self.engine
        _ts_is_btp = is_btp_isin(symbol)

        # Warn if a BTP position has trailing_stop enabled (not supported by Intesa Sanpaolo Investo)
        if _ts_is_btp and pos.get("trailing_stop") and not pos.get("_ts_btp_warned"):
            logger.warning(
                f"Trailing stop is enabled for BTP {symbol} but is not supported by Intesa Sanpaolo Investo. "
                f"It will be ignored. The position is protected only by the fixed stop-loss."
            )
            if engine.notifier:
                await engine.notifier.send_notification(
                    f"⚠️ Trailing stop ignored for BTP {display_symbol}: not supported by Intesa Sanpaolo Investo. "
                    f"Position is protected by fixed stop-loss only.",
                    summary={
                        "symbol": symbol,
                        "action": "INFO",
                        "reason": "Trailing stop not supported for BTPs",
                    }
                )
            async with engine._positions_lock:
                pos["_ts_btp_warned"] = True

        if (
            pos.get("trailing_stop")
            and pos.get("stop_loss_order_type") != "trailing_stop"
            and not _ts_is_btp
        ):
            # Check activation threshold
            activation_pct = pos.get("trailing_stop_activation_pct")
            activated = True
            if activation_pct is not None:
                entry_price = pos["price"]
                profit_pct = (current_price - entry_price) / entry_price
                if profit_pct < activation_pct:
                    activated = False

            if activated:
                # Track highest price since activation.
                # Use both the current ticker price AND the highest high
                # from recent OHLCV candles to capture intra-check price
                # spikes (the risk check only runs every
                # RISK_CHECK_INTERVAL_SECONDS, so the ticker price alone
                # may miss brief highs between checks).
                candidate_prices = [current_price]
                tf = pos.get("timeframe")
                if not tf:
                    # Fallback to the assigned timeframe from current_symbols
                    for entry in engine.current_symbols:
                        if entry["symbol"] == symbol:
                            tf = entry.get("timeframe")
                            break
                if tf:
                    try:
                        last_check_ts = pos.get("_last_trailing_check_ts", 0)
                        tf_secs = engine._timeframe_to_seconds(tf)
                        now_ts = time.time()
                        # Skip OHLCV fetch for very long timeframes (>= 1 month).
                        # OHLCV data is too sparse (2-10 candles) to provide meaningful
                        # intra-check price spikes. The ticker price alone is sufficient.
                        if tf_secs >= 2_592_000:
                            if last_check_ts == 0:
                                async with engine._positions_lock:
                                    pos["_last_trailing_check_ts"] = now_ts
                        else:
                            # Throttle OHLCV fetches: only fetch every ~10% of the
                            # timeframe interval, clamped between 5 min and 1 hour.
                            fetch_interval = max(300, min(3600, int(tf_secs * 0.1)))
                            # On first check (last_check_ts == 0), initialize
                            # timestamp but don't fetch (avoids using pre-entry
                            # candles, matching the original _load_state behavior).
                            if last_check_ts == 0:
                                async with engine._positions_lock:
                                    pos["_last_trailing_check_ts"] = now_ts
                            elif (now_ts - last_check_ts) >= fetch_interval:
                                since_ms = int(last_check_ts * 1000)
                                db_candles = await asyncio.to_thread(get_ohlcv, symbol, tf, since_ms=since_ms, limit=200)
                                if db_candles:
                                    candle_high = max(c["high"] for c in db_candles)
                                    candidate_prices.append(candle_high)
                                async with engine._positions_lock:
                                    pos["_last_trailing_check_ts"] = now_ts
                    except Exception as e:
                        logger.debug(f"Failed to fetch OHLCV for trailing stop on {symbol}: {e}")

                best_high = max(candidate_prices)
                async with engine._positions_lock:
                    if "_highest_price" not in pos or best_high > pos["_highest_price"]:
                        pos["_highest_price"] = best_high

                highest_price = pos["_highest_price"]
                new_stop = None

                # ATR-based trailing stop (Chandelier Exit)
                atr_mult = pos.get("trailing_stop_atr_multiple")
                if atr_mult is not None and atr_mult > 0:
                    # Determine the position timeframe for ATR reliability check
                    tf_for_atr = pos.get("timeframe")
                    if not tf_for_atr:
                        for entry in engine.current_symbols:
                            if entry["symbol"] == symbol:
                                tf_for_atr = entry.get("timeframe")
                                break
                    tf_secs_atr = engine._timeframe_to_seconds(tf_for_atr) if tf_for_atr else 0
                    # For very long timeframes (>= 1 month), ATR is computed from
                    # too few candles (2-10) to be statistically reliable.
                    # Skip ATR fetch and fall back to fixed percentage trailing stop.
                    skip_atr = tf_secs_atr >= 2_592_000

                    if not skip_atr:
                        # Fetch ATR from DB if we don't have it in this loop
                        if "_current_atr" not in pos or time.time() - pos.get("_atr_fetched_at", 0) > 300:
                            tf = pos.get("timeframe")
                            if tf:
                                try:
                                    ind = await asyncio.to_thread(get_indicators, symbol, tf)
                                    if ind and ind.get("atr") and ind["atr"] > 0:
                                        # Check indicator staleness: if the latest candle
                                        # used to compute ATR is older than 2× the timeframe
                                        # interval, the ATR may not reflect current volatility.
                                        ind_ts = ind.get("_indicator_timestamp")
                                        atr_is_stale = False
                                        if ind_ts is not None:
                                            # Compare against the latest candle timestamp
                                            # from the database instead of wall-clock time
                                            latest_candle_ts = await asyncio.to_thread(
                                                get_latest_ohlcv_timestamp, symbol, tf
                                            )
                                            if latest_candle_ts is not None:
                                                tf_ms = engine._timeframe_to_ms(tf)
                                                if (latest_candle_ts - ind_ts) > 2 * tf_ms:
                                                    logger.info(
                                                        f"ATR for {symbol} {tf} is stale "
                                                        f"(indicator ts={ind_ts}, latest candle ts={latest_candle_ts}, "
                                                        f"gap={latest_candle_ts - ind_ts}ms > {2 * tf_ms}ms). "
                                                        f"Falling back to fixed-percentage trailing stop."
                                                    )
                                                    atr_is_stale = True
                                            else:
                                                # Fallback to wall-clock check if no candles available
                                                tf_secs = engine._timeframe_to_seconds(tf)
                                                max_age_secs = min(tf_secs * 2, 86400)
                                                age_secs = (time.time() * 1000 - ind_ts) / 1000
                                                effective_age = max(0, age_secs - tf_secs)
                                                if effective_age > max_age_secs:
                                                    atr_is_stale = True
                                        async with engine._positions_lock:
                                            if not atr_is_stale:
                                                pos["_current_atr"] = ind["atr"]
                                            else:
                                                pos["_current_atr"] = None
                                            pos["_atr_fetched_at"] = time.time()
                                except Exception as e:
                                    logger.warning(f"Failed to fetch ATR for trailing stop on {symbol}: {e}")

                    current_atr = pos.get("_current_atr") if not skip_atr else None
                    if current_atr is not None and current_atr > 0:
                        new_stop = highest_price - (current_atr * atr_mult)
                    else:
                        # Fallback to fixed percentage if ATR fetch failed, is stale,
                        # or was skipped due to very long timeframe
                        distance = pos.get("trailing_stop_distance_pct")
                        if distance is not None:
                            new_stop = highest_price * (1 - distance)
                else:
                    # Fixed percentage trailing stop
                    distance = pos.get("trailing_stop_distance_pct")
                    if distance is not None:
                        new_stop = highest_price * (1 - distance)

                if new_stop is not None:
                    async with engine._positions_lock:
                        if new_stop > pos["stop_loss"]:
                            # Only update trailing stop if the improvement is at least 0.1%
                            # to avoid over-tightening on micro-movements (medium/long-term)
                            min_improvement = pos["stop_loss"] * 0.001
                            if new_stop - pos["stop_loss"] >= min_improvement:
                                pos["stop_loss"] = new_stop
                                logger.info(f"Trailing stop updated for {symbol}: new stop {new_stop:.4f}")
                    engine._portfolio_exposure_cache = None

    async def update_native_stop_order(
        self,
        symbol: str,
        pos: Dict[str, Any],
    ) -> None:
        """Update the native stop order if the stop-loss price has changed.

        Compares the current stop_loss with the order's original stop price
        and replaces the native stop order if the price has moved by more
        than half a tick.
        """
        engine = self.engine
        if (pos.get("stop_loss_order_id")
                and pos.get("stop_loss_order_type") in ("stop", "stop_limit")):
            # Compare current stop_loss with the order's original stop price
            original_stop = pos.get("_native_stop_price")
            if original_stop is None:
                # First time – store the current stop_loss as the baseline
                async with engine._positions_lock:
                    pos["_native_stop_price"] = pos["stop_loss"]
            else:
                # Check if stop_loss has moved by more than a tick
                tick = 0.01 if pos["stop_loss"] >= 1.0 else 0.0001
                if abs(pos["stop_loss"] - original_stop) > tick * 0.5:
                    logger.info(
                        f"Stop price changed for {symbol}: {original_stop:.4f} -> {pos['stop_loss']:.4f}. "
                        f"Replacing native stop order."
                    )
                    await engine._replace_native_stop_order(
                        symbol, pos, original_stop, pos["stop_loss"]
                    )
                    # Update the stored baseline
                    async with engine._positions_lock:
                        pos["_native_stop_price"] = pos["stop_loss"]

    async def check_partial_take_profit(
        self,
        symbol: str,
        pos: Dict[str, Any],
        current_price: float,
        display_symbol: str,
        max_partial_tp_reviews: int,
        ticker: Dict[str, Any],
    ) -> None:
        """Check and trigger partial take-profit levels (multi-level and single).

        Instead of executing immediately, sets a trigger flag for LLM review
        (unless max reviews have been reached, in which case it force-executes).
        """
        engine = self.engine
        partial_levels = pos.get("partial_take_profit_levels")
        if partial_levels:
            # Multiple levels
            triggered = pos.get("partial_tp_levels_triggered", [])
            for i, level in enumerate(partial_levels):
                if i in triggered:
                    continue
                if i in pos.get("_partial_tp_triggered_levels", []):
                    continue
                lvl_pct = level["take_profit_pct"]
                entry_price = pos["price"]
                # Time‑based cancellation
                max_time = level.get("max_time_seconds")
                if max_time is not None:
                    entry_ts = pos.get("timestamp", 0) / 1000.0
                    if time.time() - entry_ts > max_time:
                        logger.info(f"Partial TP level {i} for {symbol} expired (max {max_time}s). Cancelling.")
                        triggered.append(i)
                        async with engine._positions_lock:
                            pos["partial_tp_levels_triggered"] = triggered
                        continue
                if current_price >= entry_price * (1 + lvl_pct):
                    # --- Instead of executing immediately, set a trigger flag for LLM review ---
                    # Check if we are already waiting for LLM on this level
                    async with engine._positions_lock:
                        triggered_levels = pos.setdefault("_partial_tp_triggered_levels", [])
                        already_pending = i in triggered_levels
                        review_count = pos.get("_partial_tp_review_count", 0) + 1
                    if already_pending:
                        continue  # already pending

                    if review_count > max_partial_tp_reviews:
                        # Force execute
                        logger.info(f"Partial TP level {i} for {symbol}: max reviews reached, executing.")
                        await engine._execute_partial_tp_level(symbol, i, current_price, None, ticker)
                        # After execution, the level is marked triggered; clear the review flags for this level
                        async with engine._positions_lock:
                            pos.pop("_partial_tp_triggered", None)
                            pos.pop("_partial_tp_review_count", None)
                            pos["_partial_tp_triggered_levels"] = [x for x in pos.get("_partial_tp_triggered_levels", []) if x != i]
                        continue

                    # Set trigger and ask LLM
                    async with engine._positions_lock:
                        pos["_partial_tp_triggered"] = True
                        pos["_partial_tp_review_count"] = review_count
                        triggered_levels.append(i)
                    engine._last_strategy_eval.pop(symbol, None)  # force immediate re‑eval
                    logger.info(f"Partial TP level {i} triggered for {symbol} – asking LLM (review {review_count})")
                    if engine.notifier:
                        await engine.notifier.send_notification(
                            f"🔸 Partial TP level {i} triggered for {display_symbol} – consulting LLM...",
                            summary={"symbol": symbol, "action": "HOLD", "reason": f"Partial TP level {i} triggered – awaiting LLM"}
                        )
                    break  # only handle one new trigger per cycle; others will be picked up after LLM responds
        else:
            # Single partial TP – trigger LLM review instead of immediate execution
            partial_tp_pct = pos.get("partial_take_profit_pct")
            partial_tp_fraction = pos.get("partial_take_profit_fraction")
            if (
                partial_tp_pct is not None
                and partial_tp_fraction is not None
                and not pos.get("partial_tp_triggered", False)
                and not pos.get("_partial_tp_triggered_single")
            ):
                entry_price = pos["price"]
                if current_price >= entry_price * (1 + partial_tp_pct):
                    review_count = pos.get("_partial_tp_single_review_count", 0) + 1
                    if review_count > max_partial_tp_reviews:
                        logger.info(f"Single partial TP for {symbol}: max reviews reached, executing.")
                        await engine._execute_partial_tp_single(symbol, current_price, None, ticker)
                        async with engine._positions_lock:
                            pos.pop("_partial_tp_triggered_single", None)
                            pos.pop("_partial_tp_single_review_count", None)
                    else:
                        async with engine._positions_lock:
                            pos["_partial_tp_triggered_single"] = True
                            pos["_partial_tp_single_review_count"] = review_count
                        engine._last_strategy_eval.pop(symbol, None)
                        logger.info(f"Single partial TP triggered for {symbol} – asking LLM (review {review_count})")
                        if engine.notifier:
                            await engine.notifier.send_notification(
                                f"🔸 Partial TP triggered for {display_symbol} – consulting LLM...",
                                summary={"symbol": symbol, "action": "HOLD", "reason": "Partial TP triggered – awaiting LLM"}
                            )

    async def check_dust_sweep(
        self,
        symbol: str,
        pos: Dict[str, Any],
        display_symbol: str,
        max_dust_sweep_reviews: int,
    ) -> bool:
        """Check if a position has become dust and should be swept.

        Returns True if the dust sweep was triggered or executed (caller
        should skip to the next position), False otherwise.
        """
        engine = self.engine
        base = symbol.split("/")[0]
        amount = pos["amount"]

        # Fetch min amount
        try:
            asset = await engine._get_asset_info(symbol)
            min_amount = float(asset.min_order_size) if asset.min_order_size else None
        except Exception:
            min_amount = None

        is_dust = min_amount is not None and amount < min_amount

        if not pos.get("_dust_sweep_triggered"):
            if is_dust:
                # Check if dust has been kept past the timeout
                dust_keep_since = pos.get("_dust_keep_since")
                if dust_keep_since is not None and (time.time() - dust_keep_since) > settings.DUST_KEEP_TIMEOUT_SECONDS:
                    logger.info(
                        f"Dust keep timeout reached for {symbol} "
                        f"(kept for {(time.time() - dust_keep_since) / 3600:.1f}h), force-selling."
                    )
                    if engine.notifier:
                        await engine.notifier.send_notification(
                            f"🧹 Dust keep timeout for {display_symbol} – auto-selling "
                            f"after {settings.DUST_KEEP_TIMEOUT_SECONDS // 3600:.0f}h.",
                            summary={
                                "symbol": symbol,
                                "action": "SELL",
                                "reason": "Dust keep timeout",
                                "exit_reason": "dust_keep_timeout",
                            }
                        )
                    await engine._sweep_dust(symbol)
                    return True
                review_count = pos.get("_dust_sweep_review_count", 0) + 1
                if review_count > max_dust_sweep_reviews:
                    logger.info(f"Dust sweep max reviews reached for {symbol}, force sweeping.")
                    await engine._sweep_dust(symbol)
                    return True
                else:
                    async with engine._positions_lock:
                        pos["_dust_sweep_triggered"] = True
                        pos["_dust_sweep_review_count"] = review_count
                    engine._last_strategy_eval.pop(symbol, None)
                    logger.info(f"Dust condition triggered for {symbol} – asking LLM (review {review_count})")
                    if engine.notifier:
                        await engine.notifier.send_notification(
                            f"🧹 Dust sweep triggered for {display_symbol} – consulting LLM...",
                            summary={"symbol": symbol, "action": "HOLD", "reason": "Dust sweep triggered – awaiting LLM"}
                        )
        else:
            # If dust was previously triggered but condition no longer holds, clear it
            if not is_dust:
                async with engine._positions_lock:
                    pos.pop("_dust_sweep_triggered", None)
                    pos.pop("_dust_sweep_review_count", None)
                    pos.pop("_dust_keep_since", None)
                logger.info(f"Dust condition cleared for {symbol}")

        return False

    async def check_max_hold_expired(
        self,
        symbol: str,
        pos: Dict[str, Any],
        display_symbol: str,
    ) -> bool:
        """Check if the position has exceeded its max hold time.

        Sets a flag so the LLM is asked whether to sell or extend on the
        next evaluation cycle. Returns True if max hold expired (caller
        should skip to the next position), False otherwise.
        """
        engine = self.engine
        max_hold = pos.get("max_hold_time_seconds")
        if max_hold is not None and max_hold > 0:
            entry_ts = pos.get("timestamp", 0) / 1000.0  # convert ms to seconds
            if time.time() - entry_ts > max_hold:
                # Already waiting for LLM – do not re‑trigger
                if pos.get("_max_hold_expired"):
                    return True
                # First expiry – ask LLM
                expired_count = pos.get("_max_hold_expired_count", 0) + 1
                async with engine._positions_lock:
                    pos["_max_hold_expired"] = True
                    pos["_max_hold_expired_count"] = expired_count

                # Force re‑evaluation on the next main loop tick
                engine._last_strategy_eval.pop(symbol, None)

                logger.info(
                    f"Max hold time expired for {symbol} (attempt {expired_count}) – asking LLM to decide."
                )
                if engine.notifier:
                    await engine.notifier.send_notification(
                        f"⏰ Max hold time expired for {display_symbol} – asking LLM whether to sell or extend.",
                        summary={
                            "symbol": symbol,
                            "action": "HOLD",
                            "reason": "Max hold time expired – awaiting LLM decision",
                        }
                    )
                return True
        return False

    async def check_native_exit_triggers(
        self,
        symbol: str,
        pos: Dict[str, Any],
        current_price: float,
        display_symbol: str,
    ) -> bool:
        """Check if native exit orders (stop-loss/take-profit) have been triggered.

        When native exit orders are active, proactively cancels the OCO pair
        when the trigger price is reached (instead of waiting for the queued-
        order polling loop). Includes race condition guards to prevent
        double-sells.

        Returns True if native exit orders are active (caller should skip
        manual stop/tp checks and continue to the next position).
        Returns False if no native exit orders are active.
        """
        engine = self.engine
        if not (pos.get("stop_loss_order_id") or pos.get("take_profit_order_id")):
            return False

        sl_order_id = pos.get("stop_loss_order_id")
        tp_order_id = pos.get("take_profit_order_id")
        sl_order_type = pos.get("stop_loss_order_type", "stop")

        # Stop price reached → cancel take-profit OCO pair
        if (sl_order_id and tp_order_id
                and sl_order_type in ("stop", "stop_limit")
                and pos.get("stop_loss") is not None
                and current_price <= pos["stop_loss"]):
            # --- Race condition guard: check if the OCO pair
            # (take-profit) has already filled before cancelling. ---
            tp_already_filled = False
            try:
                tp_order_obj = await asyncio.to_thread(engine.trader.get_order, tp_order_id)
                if tp_order_obj is not None and tp_order_obj.status == "filled":
                    tp_already_filled = True
            except Exception:
                pass

            if tp_already_filled:
                logger.info(
                    f"OCO take-profit {tp_order_id} already filled for {symbol}; "
                    f"skipping cancel to avoid double-sell."
                )
                async with engine._positions_lock:
                    pos.pop("take_profit_order_id", None)
            else:
                try:
                    await asyncio.to_thread(engine.trader.cancel_order, tp_order_id)
                    logger.info(
                        f"Risk check: stop price reached for {symbol}, "
                        f"cancelled OCO take-profit {tp_order_id}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to cancel OCO TP {tp_order_id} for {symbol}: {e}")
                async with engine._queued_orders_lock:
                    engine.queued_orders = [
                        q for q in engine.queued_orders
                        if q.get("order_id") != tp_order_id
                    ]
                    for q in engine.queued_orders:
                        if q.get("order_id") == sl_order_id:
                            q["oco_pair"] = None
                            break
                async with engine._positions_lock:
                    pos.pop("take_profit_order_id", None)
                if engine.notifier:
                    await engine.notifier.send_notification(
                        f"🛑 Stop triggered for {display_symbol} at {current_price:.4f}, "
                        f"take‑profit order cancelled.",
                        summary={
                            "symbol": symbol,
                            "action": "CANCEL",
                            "reason": "Stop triggered, OCO pair cancelled (risk check)",
                        }
                    )
                # Process the stop-loss order: check if it has already filled
                # (race condition prevention — the order may have filled between
                # cancelling the TP and now).  Calling get_order will also trigger
                # the fill if the stop price has been reached, which is the
                # desired behaviour: we process the native fill instead of
                # executing a duplicate manual sell.
                sl_filled = False
                sl_order_obj = None
                try:
                    sl_order_obj = await asyncio.to_thread(engine.trader.get_order, sl_order_id)
                    if sl_order_obj is not None and sl_order_obj.status == "filled":
                        sl_filled = True
                except Exception:
                    pass

                if sl_filled:
                    # The native stop-loss order filled — process the fill to
                    # update positions and trade history, avoiding a double sell.
                    logger.info(f"Stop-loss order {sl_order_id} filled for {symbol}, processing native fill.")
                    await engine._process_native_exit_fill(symbol, sl_order_id, sl_order_obj, pos, "stop_loss")
                else:
                    # Stop-loss not yet filled — cancel it and execute manual sell
                    try:
                        await asyncio.to_thread(engine.trader.cancel_order, sl_order_id)
                    except Exception:
                        pass
                    async with engine._queued_orders_lock:
                        engine.queued_orders = [
                            q for q in engine.queued_orders
                            if q.get("order_id") != sl_order_id
                        ]
                    async with engine._positions_lock:
                        pos.pop("stop_loss_order_id", None)
                        pos.pop("stop_loss_order_type", None)
                        pos.pop("_native_stop_price", None)
                    await engine._execute_signal(
                        symbol,
                        Signal(action="SELL", confidence=1.0, reasoning="Stop-loss triggered (risk check)"),
                        exit_reason="stop_loss"
                    )
            return True  # position has been closed, move to next

        # Take-profit price reached → cancel stop OCO pair
        if (sl_order_id and tp_order_id
                and pos.get("take_profit") is not None
                and current_price >= pos["take_profit"]):
            # --- Race condition guard: check if the OCO pair
            # (stop-loss) has already filled before cancelling. ---
            sl_already_filled = False
            try:
                sl_order_obj_check = await asyncio.to_thread(engine.trader.get_order, sl_order_id)
                if sl_order_obj_check is not None and sl_order_obj_check.status == "filled":
                    sl_already_filled = True
            except Exception:
                pass

            if sl_already_filled:
                logger.info(
                    f"OCO stop-loss {sl_order_id} already filled for {symbol}; "
                    f"skipping cancel to avoid double-sell."
                )
                async with engine._positions_lock:
                    pos.pop("stop_loss_order_id", None)
                    pos.pop("stop_loss_order_type", None)
                    pos.pop("_native_stop_price", None)
            else:
                try:
                    await asyncio.to_thread(engine.trader.cancel_order, sl_order_id)
                    logger.info(
                        f"Risk check: take-profit price reached for {symbol}, "
                        f"cancelled OCO stop {sl_order_id}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to cancel OCO stop {sl_order_id} for {symbol}: {e}")
                async with engine._queued_orders_lock:
                    engine.queued_orders = [
                        q for q in engine.queued_orders
                        if q.get("order_id") != sl_order_id
                    ]
                    for q in engine.queued_orders:
                        if q.get("order_id") == tp_order_id:
                            q["oco_pair"] = None
                            break
                async with engine._positions_lock:
                    pos.pop("stop_loss_order_id", None)
                    pos.pop("stop_loss_order_type", None)
                    pos.pop("_native_stop_price", None)
                if engine.notifier:
                    await engine.notifier.send_notification(
                        f"🎯 Take‑profit reached for {display_symbol} at {current_price:.4f}, "
                        f"stop order cancelled.",
                        summary={
                            "symbol": symbol,
                            "action": "CANCEL",
                            "reason": "Take-profit reached, OCO pair cancelled (risk check)",
                        }
                    )
                # Process the take-profit order: check if it has already
                # filled (race condition prevention — the order may have
                # filled between cancelling the SL and now).  Calling
                # get_order will also trigger the fill if the TP price has
                # been reached.
                tp_filled = False
                tp_order_obj = None
                try:
                    tp_order_obj = await asyncio.to_thread(engine.trader.get_order, tp_order_id)
                    if tp_order_obj is not None and tp_order_obj.status == "filled":
                        tp_filled = True
                except Exception:
                    pass

                if tp_filled:
                    logger.info(f"Take-profit order {tp_order_id} filled for {symbol}, processing native fill.")
                    await engine._process_native_exit_fill(symbol, tp_order_id, tp_order_obj, pos, "take_profit")
                else:
                    # TP not yet filled — cancel it and execute manual sell
                    try:
                        await asyncio.to_thread(engine.trader.cancel_order, tp_order_id)
                    except Exception:
                        pass
                    async with engine._queued_orders_lock:
                        engine.queued_orders = [
                            q for q in engine.queued_orders
                            if q.get("order_id") != tp_order_id
                        ]
                    async with engine._positions_lock:
                        pos.pop("take_profit_order_id", None)
                    await engine._execute_signal(
                        symbol,
                        Signal(action="SELL", confidence=1.0, reasoning="Take-profit triggered (risk check)"),
                        exit_reason="take_profit"
                    )
            return True  # position has been closed, move to next

        # Native exit orders are active but neither trigger price reached.
        # Skip manual stop/tp checks — native orders handle it.
        return True

    async def check_breakeven_stop(
        self,
        symbol: str,
        pos: Dict[str, Any],
        current_price: float,
    ) -> None:
        """Activate breakeven stop if the position has gained enough profit.

        Moves the stop-loss to the entry price once the current price
        exceeds the entry price by the configured activation percentage.
        """
        engine = self.engine
        breakeven_activation = pos.get("breakeven_activation_pct")
        if breakeven_activation is not None and breakeven_activation > 0:
            entry_price = pos["price"]
            if current_price >= entry_price * (1 + breakeven_activation):
                # Compute exact break-even price that covers exit fee
                breakeven_price = entry_price
                async with engine._positions_lock:
                    if breakeven_price > pos["stop_loss"]:
                        pos["stop_loss"] = breakeven_price
                        logger.info(f"Breakeven stop activated for {symbol}: new stop {breakeven_price:.4f}")
                engine._portfolio_exposure_cache = None

    async def update_trailing_take_profit(
        self,
        symbol: str,
        pos: Dict[str, Any],
        current_price: float,
    ) -> None:
        """Update the trailing take-profit for a position if enabled.

        Moves the take-profit price up as the current price rises,
        maintaining a fixed distance below the current price.
        """
        engine = self.engine
        if pos.get("trailing_take_profit") and pos.get("trailing_take_profit_distance_pct"):
            ttp_dist = pos["trailing_take_profit_distance_pct"]
            new_tp = current_price * (1 + ttp_dist)
            async with engine._positions_lock:
                if new_tp > pos["take_profit"]:
                    pos["take_profit"] = new_tp
                    logger.info(f"Trailing take-profit updated for {symbol}: new TP {new_tp:.4f}")
        engine._portfolio_exposure_cache = None
