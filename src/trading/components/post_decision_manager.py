from __future__ import annotations

"""Post-LLM decision processing, validation, and execution."""
import asyncio
import json
import logging
import time
from typing import Any, Dict, Optional, TYPE_CHECKING

from src.config.settings import settings
from src.database import get_aggregate_sentiment_from_db, insert_signal, insert_llm_decision
from src.llm.prompts import get_cached_news_summary
from src.strategies.base import Signal
from src.strategies.validator import validate_signal

if TYPE_CHECKING:
    from src.trading.components.signal_processor import DecisionContext

logger = logging.getLogger(__name__)


class PostDecisionManager:
    """Handles post-LLM decision validation, logging, and execution."""

    def __init__(self, engine, event_bus):
        self.engine = engine
        self.shared_state = engine.shared_state
        self.event_bus = event_bus
        self.event_bus.subscribe("process_post_llm_decision", self.process_post_llm_decision)

    async def _get_sentiment_str(self, symbol: str) -> str:
        """Get a short news sentiment string for notifications, including an LLM summary."""
        if not settings.NEWS_ENABLED:
            return ""
        try:
            base_symbol = symbol.split("/")[0] if "/" in symbol else symbol
            agg_sent = await asyncio.to_thread(get_aggregate_sentiment_from_db, base_symbol, max_age_seconds=settings.NEWS_CACHE_TTL_SECONDS)
            if not agg_sent:
                return ""

            compound = agg_sent["avg_compound"]
            sentiment_label = "positive" if compound > 0.05 else "negative" if compound < -0.05 else "neutral"
            total = agg_sent["total_articles"]

            # Try to get an LLM-generated summary of the news
            summary = ""
            try:
                summary_raw = await asyncio.to_thread(get_cached_news_summary, symbol)
                if isinstance(summary_raw, dict):
                    summary = summary_raw.get("summary", "")
                else:
                    summary = summary_raw
                if summary in ("No recent news.", "Could not generate summary."):
                    summary = ""
            except (ValueError, TypeError, KeyError, ConnectionError, TimeoutError, OSError, json.JSONDecodeError) as e:
                logger.debug(f"_get_sentiment_str: failed to get LLM summary for {symbol}: {type(e).__name__}: {e}", extra={"event": "llm_summary_fetch_failed", "symbol": symbol, "error_type": type(e).__name__})

            base = f"📰 (sentiment: {compound:+.2f}[{sentiment_label}], {total} articles)"
            if summary:
                return f"{base} – {summary}"
            return base
        except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, json.JSONDecodeError, ConnectionError, TimeoutError, OSError) as e:
            logger.warning(f"_get_sentiment_str: failed to get sentiment for {symbol}: {type(e).__name__}: {e}", extra={"event": "sentiment_fetch_failed", "symbol": symbol, "error_type": type(e).__name__})
        except Exception as e:
            logger.warning(f"_get_sentiment_str: failed to get sentiment for {symbol}: {type(e).__name__}: {e}", extra={"event": "sentiment_fetch_failed", "symbol": symbol, "error_type": type(e).__name__})
        return ""

    async def log_and_notify_decision(
        self,
        data: DecisionContext,
        validated: Signal,
        backtest_stats: Optional[Dict[str, Any]],
    ) -> None:
        """Log the decision, record it in recent_signals, and send notification."""
        engine = self.engine

        logger.info(
            f"Decision for {data.symbol}: {validated.action} (confidence: {validated.confidence:.2f})",
            extra={
                "event": "trading_decision",
                "symbol": data.symbol,
                "action": validated.action,
                "confidence": validated.confidence,
                "reasoning": validated.reasoning,
                "strategy_type": data.signal.strategy_type,
                "model_type": getattr(validated, 'model_type', None),
                "llm_provider": data.llm_provider,
                "llm_model": data.llm_model,
            }
        )

        # Store the last decision for the next prompt cycle
        params = data.signal.strategy_params
        self.shared_state._last_decisions[data.symbol] = {
            "action": validated.action,
            "confidence": validated.confidence,
            "reasoning": validated.reasoning[:300],
            "strategy_type": data.signal.strategy_type,
            "timestamp": time.time(),
            "stop_loss_pct": params.get("stop_loss_pct") if params else None,
            "take_profit_pct": params.get("take_profit_pct") if params else None,
            "position_size_fraction": params.get("position_size_fraction") if params else None,
            "stop_loss_method": params.get("stop_loss_method") if params else None,
        }
        self.shared_state._state_dirty = True

        # Compute trade amount for display in the signals card
        _params = data.signal.strategy_params or {}
        _psf = _params.get("position_size_fraction")
        if validated.action == "BUY" and _psf is not None:
            _trade_amount = data.base_balance * float(_psf)
        elif validated.action == "SELL" and data.symbol in self.shared_state.positions:
            _pos = self.shared_state.positions[data.symbol]
            _trade_amount = _pos.get("amount", 0) * data.current_price
        else:
            _trade_amount = 0.0

        # Extract strategy parameters for the signal detail modal
        _sig_params = data.signal.strategy_params or {}
        _entry_cond_str = None
        if validated.entry_condition:
            _ec = validated.entry_condition
            _etype = _ec.get("type", "")
            if _etype == "limit_price":
                _entry_cond_str = f"Wait for price to drop to {_ec.get('price', '?')} (timeout: {_ec.get('timeout_seconds', '?')}s)"
            elif _etype == "rsi_threshold":
                _entry_cond_str = f"Wait for RSI(14) to fall below {_ec.get('rsi_below', '?')} (timeout: {_ec.get('timeout_seconds', '?')}s)"
            elif _etype == "delay":
                _entry_cond_str = f"Wait {_ec.get('delay_seconds', '?')}s before executing"
            elif _etype == "indicator_combo":
                _conds = _ec.get("conditions", [])
                _cond_strs = []
                for c in _conds:
                    _cond_strs.append(f"{c.get('indicator','?')} {c.get('direction','?')} {c.get('threshold','?')}")
                _entry_cond_str = f"Wait for ALL: {', '.join(_cond_strs)} (timeout: {_ec.get('timeout_seconds', '?')}s)"
        _sl_method = _sig_params.get("stop_loss_method", "fixed")
        _sl_str = ""
        if _sl_method == "atr_multiple":
            _sl_str = f"ATR × {_sig_params.get('stop_loss_atr_multiple', '?')} (fallback: {_sig_params.get('stop_loss_pct', '?')})"
        else:
            _sl_str = f"{_sig_params.get('stop_loss_pct', '?')}"
        _tp_str = ""
        if _sig_params.get("take_profit_atr_multiple"):
            _tp_str = f"ATR × {_sig_params.get('take_profit_atr_multiple', '?')} (fallback: {_sig_params.get('take_profit_pct', '?')})"
        else:
            _tp_str = f"{_sig_params.get('take_profit_pct', '?')}"

        # Record signal for the web dashboard
        signal_record = {
            "symbol": data.symbol,
            "display_symbol": data.display_symbol,
            "stock_name": data.stock_name,
            "timeframe": data.assigned_tf,
            "action": validated.action,
            "confidence": validated.confidence,
            "reasoning": validated.reasoning or "",
            "strategy_type": data.signal.strategy_type,
            "model_type": getattr(validated, 'model_type', None),
            "llm_provider": data.llm_provider,
            "llm_model": data.llm_model,
            "trade_amount": round(_trade_amount, 2),
            "base_currency": engine.base_currency,
            "timestamp": time.time(),
            "entry_condition": _entry_cond_str,
            "stop_loss": _sl_str,
            "take_profit": _tp_str,
            "position_size_fraction": _sig_params.get("position_size_fraction"),
            "trailing_stop": _sig_params.get("trailing_stop"),
            "trailing_stop_distance_pct": _sig_params.get("trailing_stop_distance_pct"),
            "max_hold_time_seconds": _sig_params.get("max_hold_time_seconds"),
            "cooldown_after_loss_seconds": _sig_params.get("cooldown_after_loss_seconds"),
            "order_type": data.signal.order_type,
            "limit_price": _sig_params.get("limit_price"),
        }
        await asyncio.to_thread(insert_signal, signal_record)

        # Record decision for quality tracking
        try:
            await asyncio.to_thread(
                insert_llm_decision,
                data.symbol,
                validated.action,
                data.current_price,
                data.assigned_tf,
                is_fallback=data.is_fallback
            )
        except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, json.JSONDecodeError, ConnectionError, TimeoutError, OSError) as e:
            logger.warning(f"Failed to insert LLM decision for quality tracking: {type(e).__name__}: {e}", extra={"event": "insert_llm_decision_failed", "symbol": data.symbol, "error_type": type(e).__name__})
        except Exception as e:
            logger.warning(f"Failed to insert LLM decision for quality tracking: {type(e).__name__}: {e}", extra={"event": "insert_llm_decision_failed", "symbol": data.symbol, "error_type": type(e).__name__})

        if engine.notifier:
            emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⏸️"}.get(validated.action, "❓")
            paused_tag = " (PAUSED)" if data.trading_paused and validated.action == "BUY" else ""
            sentiment_str = await self._get_sentiment_str(data.symbol)
            reasoning_str = f" – {validated.reasoning}" if validated.reasoning else ""
            msg = f"{emoji} {data.display_symbol}: {validated.action} (confidence: {validated.confidence:.2f}){reasoning_str}{paused_tag}"
            if sentiment_str:
                msg += f"\n{sentiment_str}"
            if getattr(validated, 'backtest_summary', None):
                msg += f"\n📈 Backtest: {validated.backtest_summary}"
            # Build summary dict for logging
            decision_summary = {
                "symbol": data.symbol,
                "action": validated.action,
                "confidence": validated.confidence,
                "reason": validated.reasoning[:200],
                "sentiment": data.aggregate_sentiment,
                "backtest": backtest_stats,
                "strategy_type": data.signal.strategy_type,
                "market_regime": data.market_regime,
                "model_type": getattr(validated, 'model_type', None),
                "llm_provider": data.llm_provider,
                "llm_model": data.llm_model,
            }
            await engine.notifier.send_notification(msg, summary=decision_summary)

    async def handle_triggered_flags(
        self,
        symbol: str,
        display_symbol: str,
        signal: Signal,
        validated: Signal,
        assigned_tf: str,
        current_price: float,
        atr: Optional[float],
        ticker: Dict[str, Any],
        max_hold_expired: bool,
        stop_loss_triggered: bool,
        take_profit_triggered: bool,
        partial_tp_triggered: bool,
        dust_sweep_triggered: bool,
        strategy_model_type: str,
        llm_provider: Optional[str],
        llm_model: Optional[str],
    ) -> bool:
        """Handle triggered position flags (max hold, stop loss, take profit, partial TP, dust sweep).

        Returns True if the caller should return immediately (flag was handled).
        Returns False if the caller should continue with normal execution.
        """
        engine = self.engine
        params = signal.strategy_params or {}

        # --- Handle max‑hold‑expired LLM decision ---
        if max_hold_expired and signal.action == "HOLD":
            new_max_hold = params.get("max_hold_time_seconds") if params else None
            if new_max_hold is not None and new_max_hold > 0:
                logger.info(f"LLM extended max hold time for {symbol} to {new_max_hold}s", extra={"event": "max_hold_extended", "symbol": symbol, "new_max_hold_seconds": new_max_hold})
                if symbol in self.shared_state.positions:
                    async with self.shared_state._positions_lock:
                        self.shared_state.positions[symbol]["max_hold_time_seconds"] = new_max_hold
                        self.shared_state.positions[symbol]["timestamp"] = int(time.time() * 1000)
                        self.shared_state.positions[symbol].pop("_max_hold_expired", None)
                        self.shared_state.positions[symbol].pop("_max_hold_expired_count", None)
                for symbol_entry in self.shared_state.current_symbols:
                    if symbol_entry["symbol"] == symbol:
                        symbol_entry["entry_time"] = time.time()
                        break
                if engine.notifier:
                    await engine.notifier.send_notification(
                        f"⏰ Max hold time for {display_symbol} extended to {new_max_hold}s by LLM.\n"
                        f"Reasoning: {validated.reasoning}",
                        summary={
                            "symbol": symbol,
                            "action": "HOLD",
                            "reason": validated.reasoning,
                            "new_max_hold_seconds": new_max_hold,
                            "model_type": strategy_model_type,
                            "llm_provider": llm_provider,
                            "llm_model": llm_model,
                        }
                    )
                await self.event_bus.publish(
                    "update_position_params",
                    symbol, params, signal.indicator_config, assigned_tf, current_price, atr,
                )
                self.shared_state._state_dirty = True
            else:
                logger.warning(
                    f"LLM returned HOLD without new max_hold_time_seconds for {symbol} "
                    f"after max hold expiry – forcing SELL.",
                    extra={"event": "max_hold_expired_force_sell", "symbol": symbol}
                )
                if engine.notifier:
                    await engine.notifier.send_notification(
                        f"⏰ LLM did not extend hold time for {display_symbol} – closing position.",
                        summary={
                            "symbol": symbol, "action": "SELL",
                            "reason": "Max hold expired, LLM did not extend",
                            "exit_reason": "max_hold_time_llm_no_extend",
                            "model_type": strategy_model_type, "llm_provider": llm_provider, "llm_model": llm_model,
                        }
                    )
                await self.event_bus.publish(
                    "execute_signal",
                    symbol, Signal(action="SELL", confidence=1.0, reasoning="Max hold expired, LLM did not extend"),
                    exit_reason="max_hold_time_llm_no_extend"
                )
            return True

        # --- Handle stop-loss-triggered LLM decision ---
        if stop_loss_triggered and signal.action == "HOLD":
            new_params = signal.strategy_params or {}
            new_stop_method = new_params.get("stop_loss_method", "fixed")
            new_stop_pct = None
            if new_stop_method == "atr_multiple" and atr is not None and atr > 0:
                atr_mult = new_params.get("stop_loss_atr_multiple")
                if atr_mult is not None:
                    new_stop_pct = (atr_mult * atr) / current_price
            else:
                new_stop_pct = new_params.get("stop_loss_pct")

            if new_stop_pct is not None and new_stop_pct > 0:
                logger.info(
                    f"LLM decided to hold {symbol} after stop-loss trigger, "
                    f"new stop_loss_pct={new_stop_pct:.4%}",
                    extra={"event": "stop_loss_adjusted_hold", "symbol": symbol, "new_stop_loss_pct": new_stop_pct}
                )
                if symbol in self.shared_state.positions:
                    async with self.shared_state._positions_lock:
                        self.shared_state.positions[symbol]["stop_loss"] = current_price * (1 - new_stop_pct)
                        self.shared_state.positions[symbol].pop("_stop_loss_triggered", None)
                        self.shared_state.positions[symbol].pop("_stop_loss_review_count", None)
                    await self.event_bus.publish(
                        "update_position_params",
                        symbol, new_params, signal.indicator_config, assigned_tf, current_price, atr,
                    )
                    await self.event_bus.publish("update_native_stop_order", symbol)
                    self.shared_state._state_dirty = True
                if engine.notifier:
                    await engine.notifier.send_notification(
                        f"🔄 {display_symbol}: LLM adjusted stop-loss to {new_stop_pct:.4%} – holding.\n"
                        f"Reasoning: {validated.reasoning}",
                        summary={
                            "symbol": symbol, "action": "HOLD", "reason": validated.reasoning,
                            "new_stop_loss_pct": new_stop_pct, "model_type": strategy_model_type,
                            "llm_provider": llm_provider, "llm_model": llm_model,
                        }
                    )
                return True
            else:
                logger.warning(
                    f"LLM returned HOLD for {symbol} after stop-loss trigger but did not provide "
                    f"a new stop-loss. Forcing SELL.",
                    extra={"event": "stop_loss_force_sell", "symbol": symbol}
                )
                if engine.notifier:
                    await engine.notifier.send_notification(
                        f"⛔ {display_symbol}: LLM did not provide new stop-loss – selling.",
                        summary={
                            "symbol": symbol, "action": "SELL",
                            "reason": "Stop-loss triggered, LLM did not provide new stop",
                            "exit_reason": "stop_loss_llm_no_action",
                            "model_type": strategy_model_type, "llm_provider": llm_provider, "llm_model": llm_model,
                        }
                    )
                await self.event_bus.publish(
                    "execute_signal",
                    symbol, Signal(action="SELL", confidence=1.0, reasoning="Stop-loss triggered, LLM did not provide new stop"),
                    exit_reason="stop_loss_llm_no_action"
                )
                return True

        elif stop_loss_triggered and signal.action == "SELL":
            if symbol in self.shared_state.positions:
                async with self.shared_state._positions_lock:
                    self.shared_state.positions[symbol].pop("_stop_loss_triggered", None)
                    self.shared_state.positions[symbol].pop("_stop_loss_review_count", None)
            # Continue to normal SELL execution

        # --- Handle take-profit-triggered LLM decision ---
        if take_profit_triggered and signal.action == "HOLD":
            new_params = signal.strategy_params or {}
            new_tp_pct = new_params.get("take_profit_pct")
            if new_tp_pct is not None and new_tp_pct > 0:
                logger.info(
                    f"LLM decided to hold {symbol} after take-profit trigger, "
                    f"new take_profit_pct={new_tp_pct:.4%}",
                    extra={"event": "take_profit_adjusted_hold", "symbol": symbol, "new_take_profit_pct": new_tp_pct}
                )
                if symbol in self.shared_state.positions:
                    async with self.shared_state._positions_lock:
                        self.shared_state.positions[symbol]["take_profit"] = current_price * (1 + new_tp_pct)
                        self.shared_state.positions[symbol].pop("_take_profit_triggered", None)
                        self.shared_state.positions[symbol].pop("_take_profit_review_count", None)
                    await self.event_bus.publish(
                        "update_position_params",
                        symbol, new_params, signal.indicator_config, assigned_tf, current_price, atr,
                    )
                    self.shared_state._state_dirty = True
                if engine.notifier:
                    await engine.notifier.send_notification(
                        f"🔄 {display_symbol}: LLM adjusted take-profit to {new_tp_pct:.4%} – holding.\n"
                        f"Reasoning: {validated.reasoning}",
                        summary={
                            "symbol": symbol, "action": "HOLD", "reason": validated.reasoning,
                            "new_take_profit_pct": new_tp_pct, "model_type": strategy_model_type,
                            "llm_provider": llm_provider, "llm_model": llm_model,
                        }
                    )
                return True
            else:
                logger.warning(
                    f"LLM returned HOLD for {symbol} after take-profit trigger but did not provide "
                    f"a new take-profit. Forcing SELL.",
                    extra={"event": "take_profit_force_sell", "symbol": symbol}
                )
                if engine.notifier:
                    await engine.notifier.send_notification(
                        f"🎯 {display_symbol}: LLM did not provide new take-profit – selling.",
                        summary={
                            "symbol": symbol, "action": "SELL",
                            "reason": "Take-profit triggered, LLM did not provide new take-profit",
                            "exit_reason": "take_profit_llm_no_action",
                            "model_type": strategy_model_type, "llm_provider": llm_provider, "llm_model": llm_model,
                        }
                    )
                await self.event_bus.publish(
                    "execute_signal",
                    symbol, Signal(action="SELL", confidence=1.0, reasoning="Take-profit triggered, LLM did not provide new take-profit"),
                    exit_reason="take_profit_llm_no_action"
                )
                return True

        elif take_profit_triggered and signal.action == "SELL":
            if symbol in self.shared_state.positions:
                async with self.shared_state._positions_lock:
                    self.shared_state.positions[symbol].pop("_take_profit_triggered", None)
                    self.shared_state.positions[symbol].pop("_take_profit_review_count", None)
            # Continue to normal SELL execution

        # --- Handle partial TP triggered ---
        if partial_tp_triggered and signal.action == "HOLD":
            new_levels = params.get("partial_take_profit_levels") if params else None
            if new_levels is not None:
                async with self.shared_state._positions_lock:
                    self.shared_state.positions[symbol]["partial_take_profit_levels"] = new_levels
                    self.shared_state.positions[symbol].pop("_partial_tp_triggered", None)
                    self.shared_state.positions[symbol].pop("_partial_tp_triggered_single", None)
                    self.shared_state.positions[symbol].pop("_partial_tp_review_count", None)
                    self.shared_state.positions[symbol].pop("_partial_tp_single_review_count", None)
                    self.shared_state.positions[symbol].pop("_partial_tp_triggered_levels", None)
                    self.shared_state.positions[symbol]["partial_tp_levels_triggered"] = []
                    self.shared_state.positions[symbol]["partial_tp_depth_wait_start"] = {}
                logger.info(f"LLM updated partial TP levels for {symbol}", extra={"event": "partial_tp_updated", "symbol": symbol})
                await self.event_bus.publish(
                    "update_position_params",
                    symbol, params, signal.indicator_config, assigned_tf, current_price, atr,
                )
                self.shared_state._state_dirty = True
                if engine.notifier:
                    await engine.notifier.send_notification(
                        f"🔄 {display_symbol}: LLM adjusted partial TP levels – holding.",
                        summary={"symbol": symbol, "action": "HOLD", "reason": "Partial TP levels adjusted by LLM", "model_type": strategy_model_type, "llm_provider": llm_provider, "llm_model": llm_model}
                    )
                return True
            else:
                logger.info(f"LLM did not update partial TP levels for {symbol}, executing triggered level(s)", extra={"event": "partial_tp_executed", "symbol": symbol})
                if self.shared_state.positions[symbol].get("_partial_tp_triggered_single"):
                    await self.event_bus.publish("execute_partial_tp_single", symbol, current_price, None, ticker)
                    async with self.shared_state._positions_lock:
                        self.shared_state.positions[symbol].pop("_partial_tp_triggered_single", None)
                        self.shared_state.positions[symbol].pop("_partial_tp_single_review_count", None)
                if self.shared_state.positions[symbol].get("_partial_tp_triggered"):
                    for lvl in self.shared_state.positions[symbol].get("_partial_tp_triggered_levels", []):
                        await self.event_bus.publish("execute_partial_tp_level", symbol, lvl, current_price, None, ticker)
                    async with self.shared_state._positions_lock:
                        self.shared_state.positions[symbol].pop("_partial_tp_triggered", None)
                        self.shared_state.positions[symbol].pop("_partial_tp_review_count", None)
                        self.shared_state.positions[symbol].pop("_partial_tp_triggered_levels", None)
                return True

        elif partial_tp_triggered and signal.action == "SELL":
            async with self.shared_state._positions_lock:
                self.shared_state.positions[symbol].pop("_partial_tp_triggered", None)
                self.shared_state.positions[symbol].pop("_partial_tp_triggered_single", None)
                self.shared_state.positions[symbol].pop("_partial_tp_review_count", None)
                self.shared_state.positions[symbol].pop("_partial_tp_single_review_count", None)
                self.shared_state.positions[symbol].pop("_partial_tp_triggered_levels", None)
            # Continue to normal SELL execution

        # --- Handle dust sweep triggered ---
        if dust_sweep_triggered and signal.action == "HOLD":
            async with self.shared_state._positions_lock:
                self.shared_state.positions[symbol].pop("_dust_sweep_triggered", None)
                if self.shared_state.positions[symbol].get("_dust_keep_since") is None:
                    self.shared_state.positions[symbol]["_dust_keep_since"] = time.time()
            self.shared_state._state_dirty = True
            logger.info(f"LLM decided to hold dust for {symbol}", extra={"event": "dust_hold", "symbol": symbol})
            if engine.notifier:
                await engine.notifier.send_notification(
                    f"🧹 {display_symbol}: LLM decided to keep dust – holding.",
                    summary={"symbol": symbol, "action": "HOLD", "reason": "Dust kept by LLM"}
                )
            return True
        elif dust_sweep_triggered and signal.action == "SELL":
            async with self.shared_state._positions_lock:
                self.shared_state.positions[symbol].pop("_dust_sweep_triggered", None)
                self.shared_state.positions[symbol].pop("_dust_sweep_review_count", None)
            logger.info(f"LLM decided to sell dust for {symbol}", extra={"event": "dust_sell", "symbol": symbol})
            await self.event_bus.publish("sweep_dust", symbol)
            return True

        return False

    async def handle_entry_condition(
        self,
        symbol: str,
        display_symbol: str,
        validated: Signal,
        assigned_tf: str,
        tf_seconds: int,
        trading_paused: bool,
    ) -> bool:
        """Handle entry condition for a BUY signal.

        Returns True if the entry was deferred (caller should return),
        False if no entry condition is present (caller should continue to execute).
        """
        engine = self.engine

        if validated.action != "BUY" or validated.entry_condition is None or trading_paused:
            return False

        etype = validated.entry_condition.get("type")
        if etype == "delay":
            # Delay entries are simple time-based waits – schedule directly
            delay_sec = validated.entry_condition.get("delay_seconds", 0)
            logger.info(f"Scheduling delayed BUY for {symbol} in {delay_sec}s", extra={"event": "delayed_entry_scheduled", "symbol": symbol, "delay_seconds": delay_sec})
            task = asyncio.create_task(
                engine._execute_delayed_entry(symbol, validated, assigned_tf, delay_sec)
            )
            self.shared_state._delayed_entry_tasks.add(task)
            task.add_done_callback(self.shared_state._delayed_entry_tasks.discard)
            if engine.notifier:
                await engine.notifier.send_notification(
                    f"⏳ Delayed entry for {display_symbol} – executing in {delay_sec}s.",
                    summary={
                        "symbol": symbol,
                        "action": "WAIT",
                        "reason": "Delay entry scheduled",
                        "delay_seconds": delay_sec,
                    }
                )
            return True

        timeout = validated.entry_condition.get("timeout_seconds", 600)
        # Enforce a minimum based on the candle timeframe
        min_timeout = max(300, int(settings.ENTRY_CONDITION_MIN_TIMEOUT_MULT * tf_seconds))
        # Cap the minimum timeout to avoid absurd values for very long timeframes
        min_timeout = min(min_timeout, 15_552_000)  # 180 days
        if timeout < min_timeout:
            logger.info(
                f"Entry condition timeout for {symbol} too short ({timeout}s), "
                f"clamping to minimum {min_timeout}s (timeframe={assigned_tf})",
                extra={"event": "entry_timeout_clamped", "symbol": symbol, "original_timeout": timeout, "min_timeout": min_timeout, "timeframe": assigned_tf}
            )
            timeout = min_timeout
        deadline = time.time() + timeout
        # Store for background checking – do NOT block the main loop
        async with self.shared_state._pending_entries_lock:
            self.shared_state._pending_entries[symbol] = {
                "signal": validated,
                "deadline": deadline,
                "timeframe": assigned_tf,
                "condition": validated.entry_condition,
            }
        logger.info(
            f"Queued entry condition for {symbol} (type={etype}, deadline in {timeout}s). "
            f"Will monitor in background.",
            extra={"event": "entry_condition_queued", "symbol": symbol, "condition_type": etype, "timeout_seconds": timeout}
        )
        if engine.notifier:
            await engine.notifier.send_notification(
                f"⏳ Waiting for entry condition on {display_symbol} "
                f"(type={etype}, timeout {timeout}s).",
                summary={
                    "symbol": symbol,
                    "action": "WAIT",
                    "reason": "Entry condition pending",
                }
            )
        return True

    async def check_trade_filters(
        self,
        symbol: str,
        display_symbol: str,
        validated: Signal,
        params: Dict[str, Any],
    ) -> bool:
        """Check trade filters (confidence thresholds, SELL without position).

        Returns True if the trade should be skipped (caller should return),
        False if processing should continue.
        """
        engine = self.engine

        # --- Global confidence rejection threshold (set during stock selection) ---
        if validated.action == "BUY":
            conf_rejection_raw = await engine.config_service.get_config("confidence_rejection_threshold")
            if conf_rejection_raw:
                try:
                    conf_threshold = float(conf_rejection_raw)
                    if conf_threshold > 0 and validated.confidence < conf_threshold:
                        logger.info(f"Skipping {symbol}: confidence {validated.confidence:.2f} below global rejection threshold {conf_threshold:.2f}", extra={"event": "skip_confidence_threshold", "symbol": symbol, "confidence": validated.confidence, "threshold": conf_threshold})
                        if engine.notifier:
                            await engine.notifier.send_notification(
                                f"⚠️ Skipping {display_symbol}: confidence {validated.confidence:.2f} below threshold {conf_threshold:.2f}",
                                summary={
                                    "symbol": symbol,
                                    "action": "SKIP",
                                    "reason": "Confidence below rejection threshold",
                                    "confidence": validated.confidence,
                                    "threshold": conf_threshold,
                                }
                            )
                        return True
                except (ValueError, TypeError):
                    pass

        min_conf = params.get("min_confidence")
        if min_conf is not None and validated.confidence < min_conf:
            logger.info(f"Skipping {symbol}: confidence {validated.confidence:.2f} below LLM min {min_conf:.2f}", extra={"event": "skip_confidence_min", "symbol": symbol, "confidence": validated.confidence, "min_confidence": min_conf})
            if engine.notifier:
                await engine.notifier.send_notification(
                    f"⚠️ Skipping {display_symbol}: confidence too low ({validated.confidence:.2f})",
                    summary={
                        "symbol": symbol,
                        "action": "SKIP",
                        "reason": "Confidence too low",
                        "confidence": validated.confidence,
                        "min_confidence": min_conf,
                    }
                )
            return True

        # Prevent SELL without an open position (no shorting)
        if validated.action == "SELL" and symbol not in self.shared_state.positions:
            logger.info(f"Skipping SELL for {symbol}: no open position.", extra={"event": "skip_sell_no_position", "symbol": symbol})
            if engine.notifier:
                await engine.notifier.send_notification(
                    f"⚠️ Skipping SELL for {display_symbol}: no open position.",
                    summary={
                        "symbol": symbol,
                        "action": "SKIP",
                        "reason": "No open position",
                    }
                )
            return True

        return False

    async def check_sector_concentration(
        self,
        symbol: str,
        display_symbol: str,
        assigned_tf: str,
    ) -> bool:
        """Check if buying this symbol would exceed the sector concentration limit.

        Returns True if the trade should be skipped (caller should return),
        False if processing should continue.
        """
        engine = self.engine

        current_sector = None
        for entry in self.shared_state.current_symbols:
            if entry["symbol"] == symbol:
                current_sector = entry.get("sector")
                break

        if not current_sector:
            return False

        max_positions_per_sector_raw = await engine.config_service.get_config("max_positions_per_sector")
        if max_positions_per_sector_raw:
            try:
                max_positions_per_sector = int(max_positions_per_sector_raw)
            except ValueError:
                max_positions_per_sector = None
        else:
            max_positions_per_sector = None

        if max_positions_per_sector is None or max_positions_per_sector <= 0:
            return False

        sector_count = 0
        for pos_sym in self.shared_state.positions.keys():
            for entry in self.shared_state.current_symbols:
                if entry["symbol"] == pos_sym and entry.get("sector") == current_sector:
                    sector_count += 1
                    break

        if sector_count >= max_positions_per_sector:
            logger.info(
                f"Skipping BUY {symbol}: sector '{current_sector}' already has "
                f"{sector_count} open positions (max {max_positions_per_sector})",
                extra={"event": "skip_sector_concentration", "symbol": symbol, "sector": current_sector, "sector_count": sector_count, "max_positions_per_sector": max_positions_per_sector}
            )
            if engine.notifier:
                stock_name = await engine._market_data_manager.get_stock_name(symbol)
                display = engine._format_symbol_display(symbol, stock_name, assigned_tf)
                await engine.notifier.send_notification(
                    f"⚠️ Skipping BUY {display}: sector '{current_sector}' concentration limit reached ({sector_count}/{max_positions_per_sector})",
                    summary={
                        "symbol": symbol,
                        "action": "SKIP",
                        "reason": "Sector concentration limit",
                        "sector": current_sector,
                        "sector_count": sector_count,
                        "max_positions_per_sector": max_positions_per_sector,
                    }
                )
            return True

        return False

    async def process_post_llm_decision(
        self,
        data: DecisionContext,
    ) -> None:
        """Validate the LLM signal, log/notify, and execute if all checks pass."""
        engine = self.engine

        # Ensure llm_provider and llm_model are never None for notifications/signals
        llm_provider = data.llm_provider or "fallback"
        llm_model = data.llm_model or "default_hold"

        validated = validate_signal(
            data.signal,
            atr=data.atr,
            price=data.current_price,
            timeframe_seconds=data.tf_seconds,
            min_stop_atr_mult=data.min_stop_atr_mult,
            min_hold_time_mult=data.min_hold_time_mult,
            global_min_risk_reward_ratio=data.global_min_rr,
            symbol=data.symbol,
        )
        validated.model_type = getattr(data.signal, 'model_type', None)
        validated.backtest_summary = getattr(data.signal, 'backtest_summary', None)
        validated.backtest_stats = getattr(data.signal, 'backtest_stats', None)

        # Clear _needs_risk_params flag if the LLM has now provided risk parameters
        if data.symbol in self.shared_state.positions:
            _pos = self.shared_state.positions[data.symbol]
            if _pos.get("_needs_risk_params"):
                if _pos.get("stop_loss") is not None and _pos.get("take_profit") is not None:
                    _pos.pop("_needs_risk_params", None)
                    _pos.pop("_needs_risk_params_attempts", None)
                    logger.info(f"Risk parameters obtained for {data.symbol}; cleared _needs_risk_params flag.", extra={"event": "risk_params_obtained", "symbol": data.symbol})

        await self.log_and_notify_decision(
            data=data,
            validated=validated,
            backtest_stats=getattr(validated, 'backtest_stats', None),
        )

        params = data.signal.strategy_params or {}

        # --- Handle triggered position flags (max hold, stop loss, take profit, partial TP, dust sweep) ---
        if await self.handle_triggered_flags(
            symbol=data.symbol,
            display_symbol=data.display_symbol,
            signal=data.signal,
            validated=validated,
            assigned_tf=data.assigned_tf,
            current_price=data.current_price,
            atr=data.atr,
            ticker=data.ticker,
            max_hold_expired=data.max_hold_expired,
            stop_loss_triggered=data.stop_loss_triggered,
            take_profit_triggered=data.take_profit_triggered,
            partial_tp_triggered=data.partial_tp_triggered,
            dust_sweep_triggered=data.dust_sweep_triggered,
            strategy_model_type=data.strategy_model_type,
            llm_provider=llm_provider,
            llm_model=llm_model,
        ):
            return

        # --- LLM‑controlled trade filters ---
        if await self.check_trade_filters(
            data.symbol, data.display_symbol, validated, params
        ):
            return

        # Apply any updated risk parameters from the LLM to the open position
        if data.symbol in self.shared_state.positions and data.signal.strategy_params:
            await self.event_bus.publish(
                "update_position_params",
                data.symbol,
                data.signal.strategy_params,
                data.signal.indicator_config,
                data.assigned_tf,
                data.current_price,
                data.atr,
            )

        if validated.action != "HOLD":
            # --- Sector concentration limit check (only for BUY) ---
            if validated.action == "BUY":
                if await self.check_sector_concentration(
                    data.symbol, data.display_symbol, data.assigned_tf
                ):
                    return

            if await self.handle_entry_condition(
                symbol=data.symbol,
                display_symbol=data.display_symbol,
                validated=validated,
                assigned_tf=data.assigned_tf,
                tf_seconds=data.tf_seconds,
                trading_paused=data.trading_paused,
            ):
                return

            await self.event_bus.publish("execute_signal", data.symbol, validated, timeframe=data.assigned_tf, atr=data.atr)
