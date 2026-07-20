"""Risk management component for the TradingEngine.

Handles stop-loss, take-profit, trailing stop, partial TP, dust sweep,
and other risk rule checks on open positions.
Extracted from TradingEngine to reduce class size and improve maintainability.
"""
import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from src.config.settings import settings
from src.database import insert_position_pnl_snapshot, get_indicators, get_latest_ohlcv_timestamp, get_ohlcv, get_peak_total_equity, save_peak_total_equity
from src.strategies.base import Signal
from src.utils.btp_policy import BTPPolicy
from src.utils.redis_client import is_redis_available
from src.exchanges.fees import calculate_transaction_costs

logger = logging.getLogger(__name__)


class RiskManager:
    """Handles risk management checks for open positions."""

    def __init__(self, engine, event_bus):
        self.engine = engine
        self.shared_state = engine.shared_state
        self.event_bus = event_bus
        self.event_bus.subscribe("check_risk_management", self.check_risk_management)
        self.event_bus.subscribe("record_position_pnl_snapshots", self.record_position_pnl_snapshots)
        self.event_bus.subscribe("update_native_stop_order", self._handle_update_native_stop_order)

    async def _handle_update_native_stop_order(self, symbol: str) -> None:
        """Event handler to immediately update native stop order for a symbol."""
        engine = self.engine
        if symbol in self.shared_state.positions:
            await self.update_native_stop_order(symbol, self.shared_state.positions[symbol])

    async def record_position_pnl_snapshots(self, symbols_to_check: Optional[List[str]] = None):
        """Record P&L snapshots for all open positions to the database."""
        engine = self.engine
        if not self.shared_state.positions:
            return
        pos_tickers = await asyncio.to_thread(engine._market_data_manager._get_all_position_tickers_sync)
        now_ms = int(time.time() * 1000)

        target_symbols = symbols_to_check if symbols_to_check is not None else list(self.shared_state.positions.keys())

        for symbol in target_symbols:
            pos = self.shared_state.positions.get(symbol)
            if not pos:
                continue
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
                    for t in self.shared_state.trade_history
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
            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, json.JSONDecodeError, ConnectionError, TimeoutError, OSError) as e:
                logger.debug(f"Failed to record P&L snapshot for {symbol}: {type(e).__name__}: {e}")
            except Exception as e:
                logger.debug(f"Failed to record P&L snapshot for {symbol}: {type(e).__name__}: {e}")

    async def check_risk_management(self, symbols_to_check: Optional[List[str]] = None):
        """Check open positions and close if stop-loss, take-profit, or trailing stop is hit."""
        engine = self.engine
        # --- Notify mode: no automated risk management ---
        if settings.TRADING_MODE == "notify":
            return

        # --- Portfolio-level drawdown circuit breaker ---
        await self._check_portfolio_drawdown_circuit_breaker()

        # --- Portfolio-level loss cooldown ---
        await self._check_portfolio_loss_cooldown()

        # --- Daily loss limit ---
        await self._check_daily_loss_limit()

        # Read LLM-decided review limits from Redis once (before the per-position loop)
        _review_limits = await self.read_review_limits()
        max_sl_reviews = _review_limits["max_sl_reviews"]
        max_tp_reviews = _review_limits["max_tp_reviews"]
        max_partial_tp_reviews = _review_limits["max_partial_tp_reviews"]
        max_dust_sweep_reviews = _review_limits["max_dust_sweep_reviews"]

        # Batch-fetch missing tickers once before the per-position loop
        risk_tickers = await self._fetch_risk_tickers(symbols_to_check)

        for symbol, pos in list(self.shared_state.positions.items()):
            if symbols_to_check is not None and symbol not in symbols_to_check:
                continue
            await self._check_position_risk(
                symbol, pos, risk_tickers, max_sl_reviews, max_tp_reviews,
                max_partial_tp_reviews, max_dust_sweep_reviews,
            )

        # Record position-level P&L snapshots for all open positions
        await self.record_position_pnl_snapshots(symbols_to_check)

    async def _check_portfolio_drawdown_circuit_breaker(self) -> None:
        """Check portfolio-level drawdown and pause/resume trading via a circuit breaker."""
        engine = self.engine
        try:
            with self.shared_state._trade_history_lock:
                trades_snapshot = list(self.shared_state.trade_history)
            # Compute drawdown based on realized equity (initial balance + realized P&L)
            # plus current unrealized P&L to catch deep drawdowns from open positions.
            initial_balance = engine.initial_balance
            cumulative_pnl = self.shared_state._realized_pnl_offset
            for trade in sorted(trades_snapshot, key=lambda x: x.get("timestamp", 0)):
                if trade.get("side") == "sell":
                    cumulative_pnl += trade.get("realized_pnl", 0.0)

            # Include unrealized P&L from open positions in the current equity
            unrealized_pnl = 0.0
            if self.shared_state.positions:
                pos_tickers = await asyncio.to_thread(engine._market_data_manager._get_all_position_tickers_sync)
                for symbol, pos in self.shared_state.positions.items():
                    t = pos_tickers.get(symbol)
                    current_price = t['last'] if t and t.get('last') else pos.get('price', 0.0)
                    amount = pos.get('amount', 0.0)
                    entry_price = pos.get('price', 0.0)
                    unrealized_pnl += (current_price - entry_price) * amount

            current_equity = initial_balance + cumulative_pnl + unrealized_pnl

            # Fetch or initialize peak total equity from Redis to persist
            # high-water marks driven by unrealized P&L across calls.
            peak_equity = None
            if is_redis_available():
                peak_equity_raw = await asyncio.to_thread(engine.redis.get, "trading:peak_total_equity")
                if peak_equity_raw:
                    try:
                        peak_equity = float(peak_equity_raw)
                    except (ValueError, TypeError):
                        peak_equity = None
            
            if peak_equity is None:
                # Fallback to database if Redis is unavailable or key is missing
                try:
                    peak_equity = await asyncio.to_thread(get_peak_total_equity)
                except Exception:
                    peak_equity = None
                
                if peak_equity is None:
                    peak_equity = initial_balance
                elif is_redis_available():
                    # Restore to Redis if it was missing
                    await asyncio.to_thread(engine.redis.set, "trading:peak_total_equity", str(peak_equity))

            # Update peak equity if current total equity is higher
            if current_equity > peak_equity:
                peak_equity = current_equity
                if is_redis_available():
                    await asyncio.to_thread(engine.redis.set, "trading:peak_total_equity", str(peak_equity))
                # Persist to database to survive Redis restarts
                try:
                    await asyncio.to_thread(save_peak_total_equity, peak_equity)
                except Exception as e:
                    logger.error(f"Failed to persist peak total equity to database: {type(e).__name__}: {e}")

            drawdown_pct = 0.0
            if peak_equity > 0:
                drawdown_pct = (peak_equity - current_equity) / peak_equity

            paused = await asyncio.to_thread(engine.redis.get, "trading:paused")
            source_raw = await asyncio.to_thread(engine.redis.get, "trading:pause_source")
            source = source_raw.decode() if isinstance(source_raw, bytes) else (source_raw or "")

            if drawdown_pct * 100 >= settings.PAUSE_FORCE_RESUME_MAX_DRAWDOWN_PCT:
                if not paused or source != "portfolio_drawdown":
                    logger.warning(
                        f"Portfolio drawdown circuit breaker triggered: "
                        f"{drawdown_pct * 100:.2f}% >= {settings.PAUSE_FORCE_RESUME_MAX_DRAWDOWN_PCT:.2f}%. Pausing trading."
                    )
                    from src.utils.pause_utils import set_trading_pause
                    await asyncio.to_thread(
                        set_trading_pause,
                        engine.redis,
                        "portfolio_drawdown",
                        reason=f"Portfolio drawdown {drawdown_pct * 100:.2f}% exceeded circuit breaker threshold",
                        set_pause_start=False,
                    )
                    if engine.notifier:
                        await engine.notifier.send_notification(
                            f"🛑 Portfolio drawdown circuit breaker triggered ({drawdown_pct * 100:.2f}%). Trading paused.",
                            summary={"action": "PAUSE", "reason": "Portfolio drawdown circuit breaker"}
                        )
            elif source == "portfolio_drawdown" and paused:
                logger.info(f"Portfolio drawdown recovered to {drawdown_pct * 100:.2f}%. Resuming trading.")
                from src.utils.pause_utils import clear_trading_pause_keys
                await asyncio.to_thread(clear_trading_pause_keys, engine.redis)
                if engine.notifier:
                    await engine.notifier.send_notification(
                        "▶️ Portfolio drawdown recovered, trading resumed.",
                        summary={"action": "RESUME", "reason": "Portfolio drawdown recovered"}
                    )
        except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, json.JSONDecodeError, ConnectionError, TimeoutError, OSError) as e:
            logger.error(f"Failed to compute portfolio drawdown for circuit breaker: {type(e).__name__}: {e}")
        except Exception as e:
            logger.error(f"Failed to compute portfolio drawdown for circuit breaker: {type(e).__name__}: {e}")

    async def _check_portfolio_loss_cooldown(self) -> None:
        """Check for consecutive losses and trigger a portfolio-level cooldown."""
        engine = self.engine
        if not is_redis_available():
            return

        # Check if currently in a portfolio cooldown
        source_raw = await asyncio.to_thread(engine.redis.get, "trading:pause_source")
        source = source_raw.decode() if isinstance(source_raw, bytes) else (source_raw or "")
        paused = await asyncio.to_thread(engine.redis.get, "trading:paused")

        if source == "portfolio_cooldown" and paused:
            # Cooldown is active, it will expire automatically due to TTL
            return

        max_consec_losses = settings.PORTFOLIO_COOLDOWN_MAX_CONSEC_LOSSES
        cooldown_seconds = settings.PORTFOLIO_COOLDOWN_SECONDS

        with self.shared_state._trade_history_lock:
            trades_snapshot = list(self.shared_state.trade_history)

        consec_losses = 0
        for trade in reversed(trades_snapshot):
            if trade.get("side") == "sell":
                if trade.get("realized_pnl", 0.0) < 0:
                    consec_losses += 1
                else:
                    break

        if consec_losses >= max_consec_losses:
            logger.warning(
                f"Portfolio loss cooldown triggered: {consec_losses} consecutive losses. "
                f"Pausing trading for {cooldown_seconds} seconds."
            )
            from src.utils.pause_utils import set_trading_pause
            await asyncio.to_thread(
                set_trading_pause,
                engine.redis,
                "portfolio_cooldown",
                reason=f"Portfolio cooldown after {consec_losses} consecutive losses",
                set_pause_start=False,
                ttl=cooldown_seconds,
            )
            
            if engine.notifier:
                await engine.notifier.send_notification(
                    f"🧊 Portfolio cooldown triggered after {consec_losses} consecutive losses. Trading paused for {cooldown_seconds // 60} minutes.",
                    summary={"action": "PAUSE", "reason": "Portfolio loss cooldown"}
                )

    async def _check_daily_loss_limit(self) -> None:
        """Check if daily realized losses exceed the maximum daily loss limit.

        When daily realized P&L falls below -MAX_DAILY_LOSS_PCT * initial_balance,
        trading is paused until the next calendar day (auto-resume at midnight market timezone).

        Note: This check only considers realized P&L from closed sell trades.
        Unrealized losses from open positions are intentionally excluded.
        For medium/long-term trading, intraday drawdowns on open positions
        are expected and should not trigger a daily trading halt.
        """
        engine = self.engine
        if not is_redis_available():
            return

        # Check if already paused by daily loss limit
        source_raw = await asyncio.to_thread(engine.redis.get, "trading:pause_source")
        source = source_raw.decode() if isinstance(source_raw, bytes) else (source_raw or "")
        paused = await asyncio.to_thread(engine.redis.get, "trading:paused")

        if source == "daily_loss_limit" and paused:
            # Check if it's a new day — auto-resume
            pause_start_raw = await asyncio.to_thread(engine.redis.get, "trading:pause_start")
            if pause_start_raw:
                try:
                    pause_start_ts = float(pause_start_raw)
                    tz = ZoneInfo(settings.MARKET_TIMEZONE)
                    pause_date = datetime.fromtimestamp(pause_start_ts, tz=tz).date()
                    today = datetime.now(tz).date()
                    if today > pause_date:
                        logger.info("Auto-resuming from daily loss limit: new day started.")
                        from src.utils.pause_utils import clear_trading_pause_keys
                        await asyncio.to_thread(clear_trading_pause_keys, engine.redis)
                        if engine.notifier:
                            await engine.notifier.send_notification(
                                "▶️ Auto-resumed from daily loss limit (new day started).",
                                summary={"action": "RESUME", "reason": "Daily loss limit auto-resume (new day)"}
                            )
                        engine._reeval_trigger.set()
                except (ValueError, TypeError):
                    pass
            return

        # Skip if trading is already paused by another source
        if paused:
            return

        daily_pnl = engine._daily_realized_pnl()
        daily_buy_fees = engine._daily_buy_fees()
        max_daily_loss = settings.MAX_DAILY_LOSS_PCT * engine.initial_balance

        # Reduce the threshold by buy-side fees from today's trades that are
        # not yet reflected in realized_pnl (positions still open). This ensures
        # the total daily loss (including fees) doesn't exceed the threshold.
        adjusted_max_daily_loss = max(0.0, max_daily_loss - daily_buy_fees)

        if daily_pnl < -adjusted_max_daily_loss:
            logger.warning(
                f"Daily loss limit reached: daily P&L={daily_pnl:.2f}, "
                f"max loss={adjusted_max_daily_loss:.2f} ({settings.MAX_DAILY_LOSS_PCT:.2%} of initial balance"
                f" - {daily_buy_fees:.2f} buy fees). "
                f"Pausing trading until tomorrow."
            )
            from src.utils.pause_utils import set_trading_pause
            await asyncio.to_thread(
                set_trading_pause,
                engine.redis,
                "daily_loss_limit",
                reason=f"Daily loss limit reached ({daily_pnl:.2f})",
            )

            if engine.notifier:
                await engine.notifier.send_notification(
                    f"🛑 Daily loss limit reached: {daily_pnl:.2f} {engine.base_currency} "
                    f"(max: -{adjusted_max_daily_loss:.2f}, incl. {daily_buy_fees:.2f} fees). Trading paused until tomorrow.",
                    summary={"action": "PAUSE", "reason": "Daily loss limit reached"}
                )

    async def _fetch_risk_tickers(self, symbols_to_check: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """Batch-fetch tickers for all open positions for risk checks."""
        engine = self.engine
        risk_tickers: Dict[str, Dict[str, Any]] = {}
        missing_risk: List[str] = []

        target_symbols = symbols_to_check if symbols_to_check is not None else list(self.shared_state.positions.keys())

        for sym in target_symbols:
            missing_risk.append(sym.split("/")[0])
        if missing_risk:
            try:
                raw = await engine._market_data_manager._get_quotes_batched(missing_risk, timeout_per_chunk=45.0)
                self.shared_state._portfolio_exposure_cache = None
                for sym in target_symbols:
                    base = sym.split("/")[0]
                    if base in raw:
                        risk_tickers[sym] = raw[base]
            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, json.JSONDecodeError, ConnectionError, TimeoutError, OSError) as e:
                logger.warning(f"Batch quote fetch failed in risk management: {type(e).__name__}: {e}")
            except Exception as e:
                logger.warning(f"Batch quote fetch failed in risk management: {type(e).__name__}: {e}")
        return risk_tickers

    async def _check_position_risk(
        self,
        symbol: str,
        pos: Dict[str, Any],
        risk_tickers: Dict[str, Dict[str, Any]],
        max_sl_reviews: int,
        max_tp_reviews: int,
        max_partial_tp_reviews: int,
        max_dust_sweep_reviews: int,
    ) -> None:
        """Run all risk checks for a single open position."""
        engine = self.engine
        try:
            # Skip if there is already a queued non-exit BUY order for this symbol.
            # Exit orders (SELL) should not block risk checks.
            async with self.shared_state._queued_orders_lock:
                has_queued_buy = any(
                    q['symbol'] == symbol and (q.get('side') or q.get('action') or '').lower() == 'buy'
                    for q in self.shared_state.queued_orders
                )
            if has_queued_buy:
                return

            # --- Retry deferred dust sweep if market is now open ---
            if pos.get("_dust_sweep_pending") and await engine._is_market_open():
                logger.info(f"Retrying deferred dust sweep for {symbol} (market is now open).")
                async with self.shared_state._positions_lock:
                    pos.pop("_dust_sweep_pending", None)
                await self.event_bus.publish("sweep_dust", symbol)
                return

            ticker = risk_tickers.get(symbol)
            if ticker is None:
                return  # no real-time data yet, skip this check
            current_price = ticker['last']

            # --- Staleness guard: skip risk checks if the quote is too stale ---
            pos_tf = pos.get("timeframe")
            if not pos_tf:
                for entry in self.shared_state.current_symbols:
                    if entry["symbol"] == symbol:
                        pos_tf = entry.get("timeframe")
                        break
            if pos_tf and await engine._is_quote_too_stale(ticker, pos_tf):
                logger.warning(
                    f"Skipping risk management for {symbol}: quote data is too stale "
                    f"for timeframe {pos_tf}."
                )
                return

            # --- Format symbol for notifications ---
            stock_name = await engine._market_data_manager.get_stock_name(symbol)
            display_symbol = engine._format_symbol_display(symbol, stock_name, pos.get("timeframe"))

            # --- Hard stop: maximum total loss regardless of LLM decisions ---
            if await self.check_hard_stop(symbol, pos, current_price, display_symbol):
                return

            # Skip positions that don't have LLM-defined risk parameters yet
            if pos.get("stop_loss") is None or pos.get("take_profit") is None:
                return

            # --- Maximum position age safeguard ---
            if await self.check_max_position_age(symbol, pos, display_symbol, current_price):
                return

            # --- Trailing stop update ---
            await self.update_trailing_stop(symbol, pos, current_price, display_symbol)

            # --- Trailing take-profit ---
            await self.update_trailing_take_profit(symbol, pos, current_price)

            # --- Breakeven stop ---
            await self.check_breakeven_stop(symbol, pos, current_price)

            # --- Update native stop order if stop price changed ---
            await self.update_native_stop_order(symbol, pos)

            # --- Partial take-profit ---
            await self.check_partial_take_profit(
                symbol, pos, current_price, display_symbol, max_partial_tp_reviews, ticker
            )

            # --- Dust sweep check ---
            if await self.check_dust_sweep(symbol, pos, display_symbol, max_dust_sweep_reviews):
                return

            # --- News sentiment exit ---
            if await self.check_news_sentiment_exit(symbol, pos, display_symbol):
                return

            # --- Soft stop: max unrealized loss ---
            if await self.check_soft_stop(symbol, pos, current_price, display_symbol):
                return

            # --- Max hold time expired → ask LLM instead of auto‑closing ---
            if await self.check_max_hold_expired(symbol, pos, display_symbol):
                return

            # --- Native exit order triggers (OCO handling) ---
            if await self.check_native_exit_triggers(
                symbol, pos, current_price, display_symbol
            ):
                return

            # --- Manual stop-loss / take-profit triggers (no native orders) ---
            if current_price <= pos["stop_loss"]:
                await self.check_manual_stop_loss(
                    symbol, pos, current_price, display_symbol, max_sl_reviews
                )
            elif current_price >= pos["take_profit"]:
                if await self.check_manual_take_profit(
                    symbol, pos, current_price, display_symbol, max_tp_reviews
                ):
                    return
        except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, json.JSONDecodeError, ConnectionError, TimeoutError, OSError) as e:
            logger.error(f"Risk check failed for {symbol}: {type(e).__name__}: {e}")
        except Exception as e:
            logger.error(f"Risk check failed for {symbol}: {type(e).__name__}: {e}")

    async def _is_llm_circuit_breaker_active(self) -> bool:
        """Check if the LLM circuit breaker is currently active.

        The circuit breaker short-circuit is only active during pre-market and
        market open hours (when primary models are in use). During market closed
        hours with fallback models, the short-circuit is disabled to allow the
        fallback model to handle decisions — it may be temporarily rate-limited
        but should not trigger graceful degradation without consulting the LLM.
        """
        engine = self.engine
        try:
            cb_raw = await asyncio.to_thread(engine.redis.get, "llm:circuit_breaker")
            if cb_raw:
                cb_data = json.loads(cb_raw)
                if time.time() < cb_data.get("active_until", 0):
                    # Circuit breaker is active — but only short-circuit if
                    # primary models are in use (pre-market or market open).
                    # During market closed hours with fallback models, let the
                    # normal LLM flow proceed (fallback model may recover).
                    from src.llm.cache import _should_use_primary_model
                    use_primary = await asyncio.to_thread(_should_use_primary_model)
                    if not use_primary:
                        return False
                    return True
        except (ValueError, TypeError, ConnectionError, TimeoutError, OSError, json.JSONDecodeError):
            pass
        return False

    async def read_review_limits(self) -> Dict[str, int]:
        """Read LLM-decided review limits from Redis, falling back to settings defaults."""
        engine = self.engine
        max_sl_reviews = settings.MAX_STOP_LOSS_REVIEWS
        max_tp_reviews = settings.MAX_TAKE_PROFIT_REVIEWS
        max_partial_tp_reviews = settings.MAX_PARTIAL_TP_REVIEWS
        max_dust_sweep_reviews = settings.MAX_DUST_SWEEP_REVIEWS
        try:
            raw = await engine.config_service.get_config("max_stop_loss_reviews")
            if raw:
                max_sl_reviews = int(raw)
            raw = await engine.config_service.get_config("max_take_profit_reviews")
            if raw:
                max_tp_reviews = int(raw)
            raw = await engine.config_service.get_config("max_partial_tp_reviews")
            if raw:
                max_partial_tp_reviews = int(raw)
            raw = await engine.config_service.get_config("max_dust_sweep_reviews")
            if raw:
                max_dust_sweep_reviews = int(raw)
        except (ValueError, TypeError, ConnectionError, TimeoutError, OSError):
            pass
        return {
            "max_sl_reviews": max_sl_reviews,
            "max_tp_reviews": max_tp_reviews,
            "max_partial_tp_reviews": max_partial_tp_reviews,
            "max_dust_sweep_reviews": max_dust_sweep_reviews,
        }

    def _get_hard_max_loss_pct(self, symbol: str, pos: Dict[str, Any]) -> float:
        """Determine the hard max loss percentage based on asset type and timeframe.

        Longer timeframes inherently exhibit higher volatility, so a uniform
        hard stop percentage is inappropriate. This method selects a
        timeframe-specific loss threshold from settings (e.g., 1h vs 5Y)
        to ensure the hard stop is proportional to the expected volatility
        of the position's holding period. BTPs use their own dedicated
        policy thresholds.
        """
        _is_btp = BTPPolicy.is_btp(symbol)
        default_loss = settings.BTP_HARD_MAX_LOSS_PCT if _is_btp else settings.HARD_MAX_LOSS_PCT

        pos_tf = pos.get("timeframe")
        if not pos_tf:
            for entry in self.shared_state.current_symbols:
                if entry["symbol"] == symbol:
                    pos_tf = entry.get("timeframe")
                    break

        if _is_btp:
            return BTPPolicy.get_hard_max_loss_pct(symbol, pos_tf)

        tf_loss = 0.0
        if pos_tf == "1h":
            tf_loss = settings.HARD_MAX_LOSS_PCT_1H
        elif pos_tf == "1d":
            tf_loss = settings.HARD_MAX_LOSS_PCT_1D
        elif pos_tf == "1w":
            tf_loss = settings.HARD_MAX_LOSS_PCT_1W
        elif pos_tf == "1M":
            tf_loss = settings.HARD_MAX_LOSS_PCT_1M
        elif pos_tf == "3M":
            tf_loss = settings.HARD_MAX_LOSS_PCT_3M
        elif pos_tf in ("6M", "1Y", "3Y", "5Y"):
            tf_loss = settings.HARD_MAX_LOSS_PCT_6M_1Y

        return tf_loss if tf_loss > 0 else default_loss

    async def check_hard_stop(
        self,
        symbol: str,
        pos: Dict[str, Any],
        current_price: float,
        display_symbol: str,
    ) -> bool:
        """Check if the position has exceeded the hard maximum loss threshold.

        The maximum loss threshold is dynamically determined based on the
        position's assigned timeframe and asset type (via
        `_get_hard_max_loss_pct`). This prevents uniform hard stops from
        prematurely liquidating long-term positions (e.g., 5Y) that
        naturally experience wider price swings than short-term ones (e.g., 1h).

        Returns True if the hard stop was triggered (caller should skip to
        the next position), False otherwise.
        """
        engine = self.engine
        entry_price = pos["price"]
        if entry_price <= 0:
            return False
        
        _is_btp = BTPPolicy.is_btp(symbol)
        if _is_btp:
            # Use duration/convexity based risk model for BTPs
            est_price_drop_pct = BTPPolicy.compute_btp_price_change(
                symbol, entry_price, BTPPolicy.BTP_MAX_YIELD_SHIFT_BPS
            )
            if est_price_drop_pct is not None:
                _hard_max_loss = abs(est_price_drop_pct)
            else:
                _hard_max_loss = self._get_hard_max_loss_pct(symbol, pos)
        else:
            _hard_max_loss = self._get_hard_max_loss_pct(symbol, pos)
            
        unrealized_loss_pct = (entry_price - current_price) / entry_price
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
            await self.event_bus.publish(
                "execute_signal",
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
                await self.event_bus.publish(
                    "execute_signal",
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
            tf_seconds = engine._timeframe_to_seconds(pos_tf) if pos_tf else 0
            
            if tf_seconds >= settings.NEWS_SENTIMENT_EXIT_TF_SECONDS_MEDIUM:
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
                
                # For medium-term timeframes (1 week to 30 days), apply a stricter
                # (more negative) threshold to avoid exiting on mild sentiment shifts.
                if tf_seconds >= settings.NEWS_SENTIMENT_EXIT_TF_SECONDS:
                    effective_threshold = min(
                        float(news_threshold) * settings.NEWS_SENTIMENT_EXIT_MEDIUM_THRESHOLD_MULTIPLIER,
                        0.0
                    )
                    logger.debug(
                        f"Applying stricter sentiment exit threshold for {symbol} "
                        f"(medium-term timeframe {pos_tf}): {effective_threshold:.2f}"
                    )
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
                        await self.event_bus.publish(
                            "execute_signal",
                            symbol,
                            Signal(action="SELL", confidence=1.0, reasoning="News sentiment exit"),
                            exit_reason="news_sentiment_exit"
                        )
                        return True
                except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, json.JSONDecodeError, ConnectionError, TimeoutError, OSError) as e:
                    logger.warning(f"News sentiment check failed for {symbol}: {type(e).__name__}: {e}")
                except Exception as e:
                    logger.warning(f"News sentiment check failed for {symbol}: {type(e).__name__}: {e}")
        return False

    async def _get_current_atr(self, symbol: str, pos: Dict[str, Any], tf: Optional[str]) -> Optional[float]:
        """Fetch and validate ATR for trailing stop calculation.
        
        Returns None if the timeframe is too long, the ATR is stale, or fetch fails.
        """
        engine = self.engine
        if not tf:
            return None
            
        tf_secs_atr = engine._timeframe_to_seconds(tf)
        # For very long timeframes (>= 1 month), ATR is computed from
        # too few candles (2-10) to be statistically reliable.
        if tf_secs_atr >= settings.LONG_TERM_TF_SECONDS:
            return None

        if "_current_atr" not in pos or time.time() - pos.get("_atr_fetched_at", 0) > settings.ATR_STALENESS_CHECK_SECONDS:
            try:
                ind = await asyncio.to_thread(get_indicators, symbol, tf)
                if ind and ind.get("atr") and ind["atr"] > 0:
                    ind_ts = ind.get("_indicator_timestamp")
                    atr_is_stale = False
                    if ind_ts is not None:
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
                    async with self.shared_state._positions_lock:
                        pos["_current_atr"] = ind["atr"] if not atr_is_stale else None
                        pos["_atr_fetched_at"] = time.time()
            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, json.JSONDecodeError, ConnectionError, TimeoutError, OSError) as e:
                logger.warning(f"Failed to fetch ATR for trailing stop on {symbol}: {type(e).__name__}: {e}")
            except Exception as e:
                logger.warning(f"Failed to fetch ATR for trailing stop on {symbol}: {e}")

        return pos.get("_current_atr")

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
        _ts_is_btp = BTPPolicy.is_btp(symbol)

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
            async with self.shared_state._positions_lock:
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
                    for entry in self.shared_state.current_symbols:
                        if entry["symbol"] == symbol:
                            tf = entry.get("timeframe")
                            break
                if tf:
                    try:
                        last_check_ts = pos.get("_last_trailing_check_ts", 0)
                        tf_secs = engine._timeframe_to_seconds(tf)
                        now_ts = time.time()
                        # For very long timeframes (>= 1 month), the assigned timeframe's
                        # OHLCV is too sparse (2-10 candles).  Instead, fetch daily candles
                        # which have enough data points to capture intra-check price spikes
                        # that the ticker alone would miss (risk checks run every ~4h).
                        ohlcv_tf = "1d" if tf_secs >= settings.LONG_TERM_TF_SECONDS else tf
                        # Throttle OHLCV fetches: only fetch every ~10% of the
                        # timeframe interval, clamped between 5 min and 1 hour.
                        # For very long timeframes (>= 1 month), fetch daily
                        # candles frequently (e.g., every hour) to capture
                        # intraday highs that occur between risk checks,
                        # preventing a trailing stop that is too loose.
                        if tf_secs >= settings.LONG_TERM_TF_SECONDS:
                            fetch_interval = settings.TRAILING_STOP_LONG_TF_FETCH_INTERVAL_SECONDS
                        else:
                            fetch_interval = max(settings.TRAILING_STOP_FETCH_INTERVAL_MIN_SECONDS, min(settings.TRAILING_STOP_FETCH_INTERVAL_MAX_SECONDS, int(tf_secs * settings.TRAILING_STOP_FETCH_INTERVAL_FRACTION)))
                        # On first check (last_check_ts == 0), initialize
                        # timestamp but don't fetch (avoids using pre-entry
                        # candles, matching the original _load_state behavior).
                        if last_check_ts == 0:
                            async with self.shared_state._positions_lock:
                                pos["_last_trailing_check_ts"] = now_ts
                        elif (now_ts - last_check_ts) >= fetch_interval:
                            since_ms = int(last_check_ts * 1000)
                            db_candles = await asyncio.to_thread(get_ohlcv, symbol, ohlcv_tf, since_ms=since_ms, limit=200)
                            if db_candles:
                                candle_high = max(c["high"] for c in db_candles)
                                candidate_prices.append(candle_high)
                            async with self.shared_state._positions_lock:
                                pos["_last_trailing_check_ts"] = now_ts
                    except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, json.JSONDecodeError, ConnectionError, TimeoutError, OSError) as e:
                        logger.warning(f"Failed to fetch OHLCV for trailing stop on {symbol}: {type(e).__name__}: {e}")
                    except Exception as e:
                        logger.warning(f"Failed to fetch OHLCV for trailing stop on {symbol}: {type(e).__name__}: {e}")

                best_high = max(candidate_prices)
                async with self.shared_state._positions_lock:
                    if "_highest_price" not in pos or best_high > pos["_highest_price"]:
                        pos["_highest_price"] = best_high

                highest_price = pos["_highest_price"]
                new_stop = None

                # ATR-based trailing stop (Chandelier Exit)
                atr_mult = pos.get("trailing_stop_atr_multiple")
                if atr_mult is not None and atr_mult > 0:
                    current_atr = await self._get_current_atr(symbol, pos, tf)
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
                    async with self.shared_state._positions_lock:
                        if new_stop > pos["stop_loss"]:
                            # Only update trailing stop if the improvement is at least 0.1%
                            # to avoid over-tightening on micro-movements (medium/long-term)
                            min_improvement = pos["stop_loss"] * settings.TRAILING_STOP_MIN_IMPROVEMENT_PCT
                            if new_stop - pos["stop_loss"] >= min_improvement:
                                pos["stop_loss"] = new_stop
                                logger.info(f"Trailing stop updated for {symbol}: new stop {new_stop:.4f}")
                    self.shared_state._portfolio_exposure_cache = None

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
                async with self.shared_state._positions_lock:
                    pos["_native_stop_price"] = pos["stop_loss"]
            else:
                # Check if stop_loss has moved by more than a tick
                tick = settings.TICK_SIZE_LARGE if pos["stop_loss"] >= 1.0 else settings.TICK_SIZE_SMALL
                if abs(pos["stop_loss"] - original_stop) > tick * settings.NATIVE_STOP_TICK_THRESHOLD:
                    logger.info(
                        f"Stop price changed for {symbol}: {original_stop:.4f} -> {pos['stop_loss']:.4f}. "
                        f"Replacing native stop order."
                    )
                    await self.event_bus.publish(
                        "replace_native_stop_order", symbol, pos, original_stop, pos["stop_loss"]
                    )
                    # Update the stored baseline and clear the trigger timestamp
                    # so the timeout doesn't fire prematurely for the new stop price.
                    async with self.shared_state._positions_lock:
                        pos["_native_stop_price"] = pos["stop_loss"]
                        pos.pop("_native_stop_trigger_ts", None)

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

        # --- Circuit breaker: execute partial TP immediately without LLM review ---
        if await self._is_llm_circuit_breaker_active():
            partial_levels = pos.get("partial_take_profit_levels")
            if partial_levels:
                triggered = pos.get("partial_tp_levels_triggered", [])
                for i, level in enumerate(partial_levels):
                    if i in triggered:
                        continue
                    if i in pos.get("_partial_tp_triggered_levels", []):
                        continue
                    lvl_pct = level["take_profit_pct"]
                    entry_price = pos["price"]
                    if current_price >= entry_price * (1 + lvl_pct):
                        logger.info(
                            f"LLM circuit breaker active — executing partial TP level {i} "
                            f"for {symbol} without LLM review."
                        )
                        await self.event_bus.publish("execute_partial_tp_level", symbol, i, current_price, None, ticker)
                        async with self.shared_state._positions_lock:
                            pos.pop("_partial_tp_triggered", None)
                            pos.pop("_partial_tp_review_count", None)
                            triggered_levels = pos.get("_partial_tp_triggered_levels", [])
                            pos["_partial_tp_triggered_levels"] = [x for x in triggered_levels if x != i]
            else:
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
                        logger.info(
                            f"LLM circuit breaker active — executing single partial TP "
                            f"for {symbol} without LLM review."
                        )
                        await self.event_bus.publish("execute_partial_tp_single", symbol, current_price, None, ticker)
                        async with self.shared_state._positions_lock:
                            pos.pop("_partial_tp_triggered_single", None)
                            pos.pop("_partial_tp_single_review_count", None)
            return
        # --- End circuit breaker short-circuit ---

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
                        async with self.shared_state._positions_lock:
                            pos["partial_tp_levels_triggered"] = triggered
                        continue
                if current_price >= entry_price * (1 + lvl_pct):
                    # --- Instead of executing immediately, set a trigger flag for LLM review ---
                    # Check if we are already waiting for LLM on this level
                    async with self.shared_state._positions_lock:
                        triggered_levels = pos.setdefault("_partial_tp_triggered_levels", [])
                        already_pending = i in triggered_levels
                        review_count = pos.get("_partial_tp_review_count", 0) + 1
                    if already_pending:
                        continue  # already pending

                    if review_count > max_partial_tp_reviews:
                        # Force execute
                        logger.info(f"Partial TP level {i} for {symbol}: max reviews reached, executing.")
                        await self.event_bus.publish("execute_partial_tp_level", symbol, i, current_price, None, ticker)
                        # After execution, the level is marked triggered; clear the review flags for this level
                        async with self.shared_state._positions_lock:
                            pos.pop("_partial_tp_triggered", None)
                            pos.pop("_partial_tp_review_count", None)
                            pos["_partial_tp_triggered_levels"] = [x for x in pos.get("_partial_tp_triggered_levels", []) if x != i]
                        continue

                    # Set trigger and ask LLM
                    async with self.shared_state._positions_lock:
                        pos["_partial_tp_triggered"] = True
                        pos["_partial_tp_review_count"] = review_count
                        triggered_levels.append(i)
                    self.shared_state._last_strategy_eval.pop(symbol, None)  # force immediate re‑eval
                    logger.info(f"Partial TP level {i} triggered for {symbol} – asking LLM (review {review_count})")
                    if engine.notifier:
                        await engine.notifier.send_notification(
                            f"🔸 Partial TP level {i} triggered for {display_symbol} – consulting LLM...",
                            summary={"symbol": symbol, "action": "HOLD", "reason": f"Partial TP level {i} triggered – awaiting LLM"},
                            disable_notification=False
                        )
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
                        await self.event_bus.publish("execute_partial_tp_single", symbol, current_price, None, ticker)
                        async with self.shared_state._positions_lock:
                            pos.pop("_partial_tp_triggered_single", None)
                            pos.pop("_partial_tp_single_review_count", None)
                    else:
                        async with self.shared_state._positions_lock:
                            pos["_partial_tp_triggered_single"] = True
                            pos["_partial_tp_single_review_count"] = review_count
                        self.shared_state._last_strategy_eval.pop(symbol, None)
                        logger.info(f"Single partial TP triggered for {symbol} – asking LLM (review {review_count})")
                        if engine.notifier:
                            await engine.notifier.send_notification(
                                f"🔸 Partial TP triggered for {display_symbol} – consulting LLM...",
                                summary={"symbol": symbol, "action": "HOLD", "reason": "Partial TP triggered – awaiting LLM"},
                                disable_notification=False
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

        # --- Circuit breaker: sweep dust immediately without LLM review ---
        if await self._is_llm_circuit_breaker_active():
            base = symbol.split("/")[0]
            amount = pos["amount"]
            try:
                asset = await engine._market_data_manager.get_asset_info(symbol)
                min_amount = float(asset.min_order_size) if asset.min_order_size else None
            except Exception:
                min_amount = None
            is_dust = min_amount is not None and amount < min_amount
            if is_dust:
                logger.info(
                    f"LLM circuit breaker active — sweeping dust for {symbol} "
                    f"without LLM review."
                )
                if engine.notifier:
                    await engine.notifier.send_notification(
                        f"🧹 Dust sweep for {display_symbol} – "
                        f"LLM unavailable (circuit breaker), selling.",
                        summary={
                            "symbol": symbol,
                            "action": "SELL",
                            "reason": "Dust sweep (circuit breaker active)",
                            "exit_reason": "dust_sweep_circuit_breaker",
                        }
                    )
                await self.event_bus.publish("sweep_dust", symbol)
                return True
            return False
        # --- End circuit breaker short-circuit ---

        base = symbol.split("/")[0]
        amount = pos["amount"]

        # Fetch min amount
        try:
            asset = await engine._market_data_manager.get_asset_info(symbol)
            min_amount = float(asset.min_order_size) if asset.min_order_size else None
        except (ValueError, TypeError, AttributeError, ConnectionError, TimeoutError, OSError):
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
                    await self.event_bus.publish("sweep_dust", symbol)
                    return True
                review_count = pos.get("_dust_sweep_review_count", 0) + 1
                if review_count > max_dust_sweep_reviews:
                    logger.info(f"Dust sweep max reviews reached for {symbol}, force sweeping.")
                    await self.event_bus.publish("sweep_dust", symbol)
                    return True
                else:
                    async with self.shared_state._positions_lock:
                        pos["_dust_sweep_triggered"] = True
                        pos["_dust_sweep_review_count"] = review_count
                    self.shared_state._last_strategy_eval.pop(symbol, None)
                    logger.info(f"Dust condition triggered for {symbol} – asking LLM (review {review_count})")
                    if engine.notifier:
                        await engine.notifier.send_notification(
                            f"🧹 Dust sweep triggered for {display_symbol} – consulting LLM...",
                            summary={"symbol": symbol, "action": "HOLD", "reason": "Dust sweep triggered – awaiting LLM"},
                            disable_notification=False
                        )
        else:
            # If dust was previously triggered but condition no longer holds, clear it
            if not is_dust:
                async with self.shared_state._positions_lock:
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

        # --- Circuit breaker: force-sell immediately without LLM review ---
        if await self._is_llm_circuit_breaker_active():
            max_hold = pos.get("max_hold_time_seconds")
            if max_hold is not None and max_hold > 0:
                entry_ts = pos.get("timestamp", 0) / 1000.0
                if time.time() - entry_ts > max_hold:
                    logger.warning(
                        f"LLM circuit breaker active — force-selling {symbol} "
                        f"at max hold expiry without LLM review."
                    )
                    if engine.notifier:
                        await engine.notifier.send_notification(
                            f"⏰ Max hold expired for {display_symbol} – "
                            f"LLM unavailable (circuit breaker), selling.",
                            summary={
                                "symbol": symbol,
                                "action": "SELL",
                                "reason": "Max hold expired (circuit breaker active)",
                                "exit_reason": "max_hold_circuit_breaker",
                            }
                        )
                    await self.event_bus.publish(
                        "execute_signal",
                        symbol,
                        Signal(action="SELL", confidence=1.0, reasoning="Max hold expired (circuit breaker active)"),
                        exit_reason="max_hold_circuit_breaker"
                    )
                    return True
        # --- End circuit breaker short-circuit ---

        max_hold = pos.get("max_hold_time_seconds")
        if max_hold is not None and max_hold > 0:
            entry_ts = pos.get("timestamp", 0) / 1000.0  # convert ms to seconds
            if time.time() - entry_ts > max_hold:
                # Already waiting for LLM – do not re‑trigger
                if pos.get("_max_hold_expired"):
                    return True
                # First expiry – ask LLM
                expired_count = pos.get("_max_hold_expired_count", 0) + 1
                async with self.shared_state._positions_lock:
                    pos["_max_hold_expired"] = True
                    pos["_max_hold_expired_count"] = expired_count

                # Force re‑evaluation on the next main loop tick
                self.shared_state._last_strategy_eval.pop(symbol, None)

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
                        },
                        disable_notification=False
                    )
                return True
        return False

    async def check_max_position_age(
        self,
        symbol: str,
        pos: Dict[str, Any],
        display_symbol: str,
        current_price: float,
    ) -> bool:
        """Check if a position has exceeded its maximum age safeguard.

        If the LLM keeps extending max_hold_time_seconds, this safeguard
        force-closes the position once its age exceeds
        MAX_POSITION_AGE_MULTIPLIER × the original max hold time.

        Returns True if the position was force-closed (caller should skip
        to the next position), False otherwise.
        """
        engine = self.engine
        multiplier = settings.MAX_POSITION_AGE_MULTIPLIER
        if multiplier <= 0:
            return False

        original_max_hold = pos.get("_original_max_hold_time_seconds")
        if original_max_hold is None or original_max_hold <= 0:
            return False

        entry_ts = pos.get("timestamp", 0) / 1000.0
        position_age = time.time() - entry_ts
        max_age = multiplier * original_max_hold

        if position_age > max_age:
            # Skip force-close if the position is currently profitable
            entry_price = pos.get("price", 0.0)
            if current_price > entry_price:
                logger.info(
                    f"Max position age reached for {symbol} but position is profitable "
                    f"(current: {current_price:.4f}, entry: {entry_price:.4f}). "
                    f"Skipping force-close to allow LLM to manage the trade."
                )
                return False

            logger.warning(
                f"Maximum position age reached for {symbol}: "
                f"age {position_age / 86400:.1f} days > limit {max_age / 86400:.1f} days "
                f"({multiplier}× original max hold {original_max_hold / 86400:.1f} days). Forcing SELL."
            )
            if engine.notifier:
                await engine.notifier.send_notification(
                    f"⏳ Max position age reached for {display_symbol} – "
                    f"held {position_age / 86400:.1f} days (limit {max_age / 86400:.1f} days). Force selling.",
                    summary={
                        "symbol": symbol,
                        "action": "SELL",
                        "reason": "Maximum position age safeguard",
                        "exit_reason": "max_position_age",
                    }
                )
            await self.event_bus.publish(
                "execute_signal",
                symbol,
                Signal(action="SELL", confidence=1.0, reasoning="Maximum position age safeguard"),
                exit_reason="max_position_age"
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

        # Stop price reached → cancel take-profit OCO pair (if still present) and handle SL
        if (sl_order_id
                and sl_order_type in ("stop", "stop_limit")
                and pos.get("stop_loss") is not None
                and current_price <= pos["stop_loss"]):
            return await self._handle_stop_trigger(symbol, pos, current_price, display_symbol, sl_order_id, tp_order_id)

        # Take-profit price reached → cancel stop OCO pair
        if (sl_order_id and tp_order_id
                and pos.get("take_profit") is not None
                and current_price >= pos["take_profit"]):
            return await self._handle_take_profit_trigger(symbol, pos, current_price, display_symbol, sl_order_id, tp_order_id)

        # Native exit orders are active but neither trigger price reached.
        # Skip manual stop/tp checks — native orders handle it.
        # If the price has recovered above the stop-loss, clear the trigger timestamp.
        if pos.get("_native_stop_trigger_ts") is not None and current_price > pos.get("stop_loss", float('inf')):
            async with self.shared_state._positions_lock:
                pos.pop("_native_stop_trigger_ts", None)
        return True

    async def _handle_stop_trigger(
        self,
        symbol: str,
        pos: Dict[str, Any],
        current_price: float,
        display_symbol: str,
        sl_order_id: str,
        tp_order_id: Optional[str],
    ) -> bool:
        """Handle stop-loss trigger: cancel TP, check SL fill, or fallback to manual sell."""
        engine = self.engine
        
        if tp_order_id:
            tp_already_filled = False
            try:
                tp_order_obj = await asyncio.to_thread(engine.trader.get_order, tp_order_id)
                if tp_order_obj is not None and tp_order_obj.status == "filled":
                    tp_already_filled = True
            except Exception as e:
                logger.debug(f"check_native_exit_triggers: failed to check TP fill status for {symbol}: {type(e).__name__}: {e}")

            async with self.shared_state._positions_lock:
                if not pos.get("take_profit_order_id"):
                    return True
                if tp_already_filled:
                    pos.pop("take_profit_order_id", None)
                    pos.pop("stop_loss_order_id", None)
                    pos.pop("stop_loss_order_type", None)
                    pos.pop("_native_stop_price", None)
                    pos.pop("_native_stop_trigger_ts", None)
                else:
                    pos.pop("take_profit_order_id", None)

            if tp_already_filled:
                async with self.shared_state._queued_orders_lock:
                    self.shared_state.queued_orders = [
                        q for q in self.shared_state.queued_orders
                        if q.get("order_id") != sl_order_id and q.get("order_id") != tp_order_id
                    ]
                return True

            try:
                await asyncio.to_thread(engine.trader.cancel_order, tp_order_id)
                logger.info(f"Risk check: stop price reached for {symbol}, cancelled OCO take-profit {tp_order_id}")
            except Exception as e:
                logger.warning(f"Failed to cancel OCO TP {tp_order_id} for {symbol}: {type(e).__name__}: {e}")

            async with self.shared_state._queued_orders_lock:
                self.shared_state.queued_orders = [
                    q for q in self.shared_state.queued_orders
                    if q.get("order_id") != tp_order_id
                ]
                for q in self.shared_state.queued_orders:
                    if q.get("order_id") == sl_order_id:
                        q["oco_pair"] = None
                        break

            if engine.notifier:
                await engine.notifier.send_notification(
                    f"🛑 Stop triggered for {display_symbol} at {current_price:.4f}, "
                    f"take‑profit order cancelled.",
                    summary={
                        "symbol": symbol,
                        "action": "CANCEL",
                        "reason": "Stop triggered, OCO pair cancelled (risk check)",
                    },
                    disable_notification=False
                )

        # Process the stop-loss order: check if it has already filled
        sl_filled = False
        sl_order_obj = None
        manual_sell = False
        
        try:
            sl_order_obj = await asyncio.to_thread(engine.trader.get_order, sl_order_id)
            if sl_order_obj is not None and sl_order_obj.status == "filled":
                sl_filled = True
        except Exception:
            pass

        if sl_filled:
            async with self.shared_state._positions_lock:
                if not pos.get("stop_loss_order_id"):
                    return True
                pos.pop("stop_loss_order_id", None)
                pos.pop("stop_loss_order_type", None)
                pos.pop("_native_stop_price", None)
                pos.pop("_native_stop_trigger_ts", None)
            async with self.shared_state._queued_orders_lock:
                self.shared_state.queued_orders = [
                    q for q in self.shared_state.queued_orders
                    if q.get("order_id") != sl_order_id
                ]
            await self.event_bus.publish("process_native_exit_fill", symbol, sl_order_id, sl_order_obj, pos, "stop_loss")
            return True
        else:
            async with self.shared_state._positions_lock:
                if not pos.get("stop_loss_order_id"):
                    return True
                now_ts = time.time()
                trigger_ts = pos.get("_native_stop_trigger_ts")
                if trigger_ts is None:
                    pos["_native_stop_trigger_ts"] = now_ts
                    trigger_ts = now_ts
                elapsed = now_ts - trigger_ts
                if elapsed >= settings.NATIVE_STOP_FILL_TIMEOUT_SECONDS:
                    logger.warning(
                        f"Native stop-loss order {sl_order_id} for {symbol} "
                        f"not filled after {elapsed:.0f}s, falling back to manual market sell."
                    )
                    pos.pop("stop_loss_order_id", None)
                    pos.pop("stop_loss_order_type", None)
                    pos.pop("_native_stop_price", None)
                    pos.pop("_native_stop_trigger_ts", None)
                    manual_sell = True
                else:
                    logger.debug(
                        f"Stop price reached for {symbol}, waiting for native "
                        f"stop-loss order {sl_order_id} to fill "
                        f"({elapsed:.0f}s / {settings.NATIVE_STOP_FILL_TIMEOUT_SECONDS}s)."
                    )
                    return True

        if manual_sell:
            try:
                await asyncio.to_thread(engine.trader.cancel_order, sl_order_id)
            except Exception as e:
                logger.warning(f"Failed to cancel native stop {sl_order_id} for {symbol}: {type(e).__name__}: {e}")
        
            async with self.shared_state._queued_orders_lock:
                self.shared_state.queued_orders = [
                    q for q in self.shared_state.queued_orders
                    if q.get("order_id") != sl_order_id
                ]
            await self.event_bus.publish(
                "execute_signal",
                symbol,
                Signal(action="SELL", confidence=1.0, reasoning="Stop-loss native fill timeout"),
                exit_reason="stop_loss"
            )
        return True

    async def _handle_take_profit_trigger(
        self,
        symbol: str,
        pos: Dict[str, Any],
        current_price: float,
        display_symbol: str,
        sl_order_id: str,
        tp_order_id: str,
    ) -> bool:
        """Handle take-profit trigger: cancel SL, check TP fill, or fallback to manual sell."""
        engine = self.engine
        
        async with self.shared_state._positions_lock:
            if not pos.get("stop_loss_order_id"):
                return True
            pos.pop("stop_loss_order_id", None)
            pos.pop("stop_loss_order_type", None)
            pos.pop("_native_stop_price", None)

        sl_filled = False
        sl_order_obj = None
        try:
            sl_order_obj = await asyncio.to_thread(engine.trader.get_order, sl_order_id)
            if sl_order_obj is not None and sl_order_obj.status == "filled":
                sl_filled = True
        except Exception as e:
            logger.debug(f"check_native_exit_triggers: failed to check SL fill status for {symbol}: {type(e).__name__}: {e}")

        if sl_filled:
            async with self.shared_state._positions_lock:
                pos.pop("take_profit_order_id", None)
                pos.pop("_native_stop_trigger_ts", None)
            async with self.shared_state._queued_orders_lock:
                self.shared_state.queued_orders = [
                    q for q in self.shared_state.queued_orders
                    if q.get("order_id") != sl_order_id and q.get("order_id") != tp_order_id
                ]
            await self.event_bus.publish("process_native_exit_fill", symbol, sl_order_id, sl_order_obj, pos, "stop_loss")
            return True

        try:
            await asyncio.to_thread(engine.trader.cancel_order, sl_order_id)
            logger.info(f"Risk check: take-profit price reached for {symbol}, cancelled OCO stop {sl_order_id}")
        except Exception as e:
            logger.warning(f"Failed to cancel OCO stop {sl_order_id} for {symbol}: {type(e).__name__}: {e}")
            try:
                sl_order_obj = await asyncio.to_thread(engine.trader.get_order, sl_order_id)
                if sl_order_obj is not None and sl_order_obj.status == "filled":
                    sl_filled = True
            except Exception:
                pass
            
            if sl_filled:
                async with self.shared_state._positions_lock:
                    pos.pop("take_profit_order_id", None)
                    pos.pop("_native_stop_trigger_ts", None)
                async with self.shared_state._queued_orders_lock:
                    self.shared_state.queued_orders = [
                        q for q in self.shared_state.queued_orders
                        if q.get("order_id") != sl_order_id and q.get("order_id") != tp_order_id
                    ]
                await self.event_bus.publish("process_native_exit_fill", symbol, sl_order_id, sl_order_obj, pos, "stop_loss")
                return True

        async with self.shared_state._queued_orders_lock:
            self.shared_state.queued_orders = [
                q for q in self.shared_state.queued_orders
                if q.get("order_id") != sl_order_id
            ]
            for q in self.shared_state.queued_orders:
                if q.get("order_id") == tp_order_id:
                    q["oco_pair"] = None
                    break

        if engine.notifier:
            await engine.notifier.send_notification(
                f"🎯 Take‑profit reached for {display_symbol} at {current_price:.4f}, "
                f"stop order cancelled.",
                summary={
                    "symbol": symbol,
                    "action": "CANCEL",
                    "reason": "Take-profit reached, OCO pair cancelled (risk check)",
                },
                disable_notification=False
            )

        tp_filled = False
        tp_order_obj = None
        manual_sell = False

        try:
            tp_order_obj = await asyncio.to_thread(engine.trader.get_order, tp_order_id)
            if tp_order_obj is not None and tp_order_obj.status == "filled":
                tp_filled = True
        except Exception as e:
            logger.debug(f"check_native_exit_triggers: failed to check TP fill for {symbol}: {type(e).__name__}: {e}")

        if tp_filled:
            async with self.shared_state._positions_lock:
                if not pos.get("take_profit_order_id"):
                    return True
                pos.pop("take_profit_order_id", None)
            async with self.shared_state._queued_orders_lock:
                self.shared_state.queued_orders = [
                    q for q in self.shared_state.queued_orders
                    if q.get("order_id") != tp_order_id
                ]
            await self.event_bus.publish("process_native_exit_fill", symbol, tp_order_id, tp_order_obj, pos, "take_profit")
            return True
        else:
            async with self.shared_state._positions_lock:
                if not pos.get("take_profit_order_id"):
                    return True
                pos.pop("take_profit_order_id", None)
                manual_sell = True

        if manual_sell:
            try:
                await asyncio.to_thread(engine.trader.cancel_order, tp_order_id)
            except Exception as e:
                logger.debug(f"check_native_exit_triggers: failed to cancel TP {tp_order_id} for {symbol}: {type(e).__name__}: {e}")

            async with self.shared_state._queued_orders_lock:
                self.shared_state.queued_orders = [
                    q for q in self.shared_state.queued_orders
                    if q.get("order_id") != tp_order_id
                ]
            await self.event_bus.publish(
                "execute_signal",
                symbol,
                Signal(action="SELL", confidence=1.0, reasoning="Take-profit triggered (risk check)"),
                exit_reason="take_profit"
            )
        return True

    async def check_manual_stop_loss(
        self,
        symbol: str,
        pos: Dict[str, Any],
        current_price: float,
        display_symbol: str,
        max_sl_reviews: int,
    ) -> None:
        """Handle a manual stop-loss trigger (no native orders).

        Instead of immediately selling, asks the LLM whether to sell or
        adjust the stop. Scales max reviews based on position timeframe.
        Force-sells after max reviews are reached.
        """
        engine = self.engine

        # --- Circuit breaker: sell immediately without LLM review ---
        if await self._is_llm_circuit_breaker_active():
            logger.warning(
                f"LLM circuit breaker active — selling {symbol} at stop-loss "
                f"{current_price:.4f} without LLM review."
            )
            if engine.notifier:
                await engine.notifier.send_notification(
                    f"⛔ Stop‑loss triggered for {display_symbol} at {current_price:.4f} – "
                    f"LLM unavailable (circuit breaker), selling.",
                    summary={
                        "symbol": symbol,
                        "action": "SELL",
                        "reason": "Stop-loss (circuit breaker active)",
                        "price": current_price,
                        "exit_reason": "stop_loss_circuit_breaker",
                    }
                )
            await self.event_bus.publish(
                "execute_signal",
                symbol,
                Signal(action="SELL", confidence=1.0, reasoning="Stop-loss (circuit breaker active)"),
                exit_reason="stop_loss_circuit_breaker"
            )
            return

        effective_max_sl_reviews = max_sl_reviews
        pos_tf = pos.get("timeframe")
        if pos_tf:
            pos_tf_secs = engine._timeframe_to_seconds(pos_tf)
            if pos_tf_secs >= settings.LONG_TERM_TF_SECONDS:  # >= 1 month
                effective_max_sl_reviews = min(effective_max_sl_reviews, settings.LONG_TERM_MAX_STOP_LOSS_REVIEWS)
            elif pos_tf_secs >= 604_800:  # >= 1 week
                effective_max_sl_reviews = min(effective_max_sl_reviews, settings.WEEKLY_MAX_STOP_LOSS_REVIEWS)
        review_count = pos.get("_stop_loss_review_count", 0)
        if review_count >= effective_max_sl_reviews:
            # Fallback: force-sell after too many reviews
            logger.warning(
                f"Stop-loss triggered for {symbol} at {current_price} – "
                f"review count {review_count} >= {effective_max_sl_reviews}, forcing SELL."
            )
            if engine.notifier:
                await engine.notifier.send_notification(
                    f"⛔ Stop‑loss triggered for {display_symbol} at {current_price:.4f} – "
                    f"max reviews reached ({effective_max_sl_reviews}), selling.",
                    summary={
                        "symbol": symbol,
                        "action": "SELL",
                        "reason": "Stop-loss (max reviews)",
                        "price": current_price,
                        "exit_reason": "stop_loss_max_reviews",
                    }
                )
            await self.event_bus.publish(
                "execute_signal",
                symbol,
                Signal(action="SELL", confidence=1.0, reasoning="Stop-loss (max reviews)"),
                exit_reason="stop_loss_max_reviews"
            )
        else:
            # First or repeated trigger: set flag and ask LLM
            if not pos.get("_stop_loss_triggered"):
                async with self.shared_state._positions_lock:
                    pos["_stop_loss_triggered"] = True
                    pos["_stop_loss_review_count"] = review_count + 1
                # Force immediate strategy re-evaluation for this symbol
                self.shared_state._last_strategy_eval.pop(symbol, None)
                logger.info(
                    f"Stop-loss triggered for {symbol} at {current_price} – "
                    f"asking LLM (review {pos['_stop_loss_review_count']}/{effective_max_sl_reviews})."
                )
                if engine.notifier:
                    await engine.notifier.send_notification(
                        f"⛔ Stop‑loss hit for {display_symbol} at {current_price:.4f} – consulting LLM...",
                        summary={
                            "symbol": symbol,
                            "action": "HOLD",
                            "reason": "Stop-loss triggered – awaiting LLM decision",
                            "price": current_price,
                        },
                        disable_notification=False
                    )
            else:
                # Already waiting for LLM; do nothing (avoid re-triggering)
                logger.debug(
                    f"Stop-loss still triggered for {symbol}, waiting for LLM response "
                    f"(review {review_count}/{effective_max_sl_reviews})."
                )

    async def check_manual_take_profit(
        self,
        symbol: str,
        pos: Dict[str, Any],
        current_price: float,
        display_symbol: str,
        max_tp_reviews: int,
    ) -> bool:
        """Handle a manual take-profit trigger (no native orders).

        Always asks the LLM whether to sell or adjust the take-profit,
        but caps reviews. Force-sells after max reviews are reached.

        Returns True if the position was force-sold (caller should continue
        to the next position), False otherwise.
        """
        engine = self.engine

        # --- Circuit breaker: sell immediately without LLM review ---
        if await self._is_llm_circuit_breaker_active():
            logger.warning(
                f"LLM circuit breaker active — selling {symbol} at take-profit "
                f"{current_price:.4f} without LLM review."
            )
            if engine.notifier:
                await engine.notifier.send_notification(
                    f"🎯 Take‑profit triggered for {display_symbol} at {current_price:.4f} – "
                    f"LLM unavailable (circuit breaker), selling.",
                    summary={
                        "symbol": symbol,
                        "action": "SELL",
                        "reason": "Take-profit (circuit breaker active)",
                        "price": current_price,
                        "exit_reason": "take_profit_circuit_breaker",
                    }
                )
            await self.event_bus.publish(
                "execute_signal",
                symbol,
                Signal(action="SELL", confidence=1.0, reasoning="Take-profit (circuit breaker active)"),
                exit_reason="take_profit_circuit_breaker"
            )
            return True

        review_count = pos.get("_take_profit_review_count", 0)
        if review_count >= max_tp_reviews:
            logger.warning(
                f"Take-profit triggered for {symbol} at {current_price} – "
                f"review count {review_count} >= {max_tp_reviews}, forcing SELL."
            )
            if engine.notifier:
                await engine.notifier.send_notification(
                    f"🎯 Take‑profit triggered for {display_symbol} at {current_price:.4f} – "
                    f"max reviews reached, selling.",
                    summary={
                        "symbol": symbol,
                        "action": "SELL",
                        "reason": "Take-profit (max reviews)",
                        "price": current_price,
                        "exit_reason": "take_profit_max_reviews",
                    }
                )
            await self.event_bus.publish(
                "execute_signal",
                symbol,
                Signal(action="SELL", confidence=1.0, reasoning="Take-profit (max reviews)"),
                exit_reason="take_profit_max_reviews"
            )
            return True
        # First or repeated trigger: set flag and ask LLM
        if not pos.get("_take_profit_triggered"):
            async with self.shared_state._positions_lock:
                pos["_take_profit_triggered"] = True
                pos["_take_profit_review_count"] = review_count + 1
            # Force immediate strategy re-evaluation for this symbol
            self.shared_state._last_strategy_eval.pop(symbol, None)
            logger.info(
                f"Take-profit triggered for {symbol} at {current_price} – "
                f"asking LLM (review {pos['_take_profit_review_count']}/{max_tp_reviews})."
            )
            if engine.notifier:
                await engine.notifier.send_notification(
                    f"🎯 Take‑profit hit for {display_symbol} at {current_price:.4f} – consulting LLM...",
                    summary={
                        "symbol": symbol,
                        "action": "HOLD",
                        "reason": "Take-profit triggered – awaiting LLM decision",
                        "price": current_price,
                    },
                    disable_notification=False
                )
        else:
            # Already waiting for LLM; do nothing
            logger.debug(
                f"Take-profit still triggered for {symbol}, waiting for LLM response "
                f"(review {review_count}/{max_tp_reviews})."
            )
        return False

    async def check_breakeven_stop(
        self,
        symbol: str,
        pos: Dict[str, Any],
        current_price: float,
    ) -> None:
        """Activate breakeven stop if the position has gained enough profit.

        Moves the stop-loss to the entry price plus exit fees once the current
        price exceeds the entry price by the configured activation percentage.
        """
        engine = self.engine
        breakeven_activation = pos.get("breakeven_activation_pct")
        if breakeven_activation is not None and breakeven_activation > 0:
            entry_price = pos["price"]
            if current_price >= entry_price * (1 + breakeven_activation):
                # Compute exact break-even price that covers actual exit fees
                # BTPs have different fee structures, so we only apply the buffer
                # to non-BTP assets.
                if BTPPolicy.is_btp(symbol):
                    breakeven_price = BTPPolicy.compute_breakeven_price(symbol, entry_price, pos.get("amount", 0.0))
                else:
                    amount = pos.get("amount", 0.0)
                    if amount > 0:
                        costs = calculate_transaction_costs("SELL", entry_price, amount, symbol)
                        exit_fee_per_share = costs["bank_fee"] / amount
                        breakeven_price = entry_price + exit_fee_per_share
                    else:
                        # Fallback if amount is missing or zero
                        breakeven_price = entry_price * (1 + settings.BREAKEVEN_FALLBACK_BUFFER_PCT)
                async with self.shared_state._positions_lock:
                    if breakeven_price > pos["stop_loss"]:
                        pos["stop_loss"] = breakeven_price
                        logger.info(f"Breakeven stop activated for {symbol}: new stop {breakeven_price:.4f}")
                self.shared_state._portfolio_exposure_cache = None

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
            async with self.shared_state._positions_lock:
                if new_tp > pos["take_profit"]:
                    pos["take_profit"] = new_tp
                    logger.info(f"Trailing take-profit updated for {symbol}: new TP {new_tp:.4f}")
        self.shared_state._portfolio_exposure_cache = None
