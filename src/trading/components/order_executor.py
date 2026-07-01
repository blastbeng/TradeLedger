"""Order execution component for the TradingEngine.

Handles order creation, fill processing, and exit order management.
Extracted from TradingEngine to reduce class size and improve maintainability.
"""
import asyncio
import logging
import time
from dataclasses import asdict
from typing import Any, Dict, Optional

from src.config.settings import settings
from src.database import insert_trade
from src.strategies.base import Signal
from src.utils.symbol_utils import is_btp_isin

logger = logging.getLogger(__name__)


class OrderExecutor:
    """Handles order execution and fill processing for the TradingEngine."""

    def __init__(self, engine):
        self.engine = engine

    def compute_exit_order_prices(
        self,
        entry_price: float,
        signal: Signal,
        atr: Optional[float] = None,
    ) -> Dict[str, Optional[float]]:
        """
        Return a dict with keys:
          - stop_loss_price: the trigger/limit price for the stop-loss order
          - take_profit_price: the limit price for the take-profit order
        Uses the LLM's exit order type fields; falls back to the standard
        stop_loss_pct / take_profit_pct if exit order types are not provided.
        """
        params = signal.strategy_params or {}
        stop_loss_pct = params.get("stop_loss_pct")
        take_profit_pct = params.get("take_profit_pct")

        # --- Stop-loss price ---
        sl_ot = signal.stop_loss_order_type
        if sl_ot == "stop":
            sl_price = signal.stop_loss_stop_price
            if sl_price is None and stop_loss_pct is not None:
                sl_price = entry_price * (1 - stop_loss_pct)
        elif sl_ot == "stop_limit":
            sl_price = signal.stop_loss_stop_price
            if sl_price is None and stop_loss_pct is not None:
                sl_price = entry_price * (1 - stop_loss_pct)
        elif sl_ot == "trailing_stop":
            sl_price = None  # not a fixed price
        else:
            sl_price = None

        # --- Take-profit price ---
        tp_ot = signal.take_profit_order_type
        if tp_ot == "limit":
            tp_price = signal.take_profit_limit_price
            if tp_price is None and take_profit_pct is not None:
                tp_price = entry_price * (1 + take_profit_pct)
        elif tp_ot == "market":
            tp_price = None  # will be handled by risk loop later
        else:
            tp_price = None

        return {
            "stop_loss_price": sl_price,
            "take_profit_price": tp_price,
        }

    async def cancel_exit_orders(self, symbol: str):
        """Cancel any native stop-loss and take-profit orders for a symbol."""
        engine = self.engine
        pos = engine.positions.get(symbol)
        if not pos:
            return
        for order_id_key in ("stop_loss_order_id", "take_profit_order_id"):
            order_id = pos.pop(order_id_key, None)
            if order_id:
                try:
                    await asyncio.to_thread(engine.trader.cancel_order, order_id)
                    logger.info(f"Cancelled exit order {order_id} for {symbol}")
                except Exception as e:
                    logger.warning(f"Failed to cancel exit order {order_id}: {e}")
                async with engine._queued_orders_lock:
                    engine.queued_orders = [
                        q for q in engine.queued_orders
                        if q.get("order_id") != order_id
                    ]
        pos.pop("stop_loss_order_type", None)
        pos.pop("_native_stop_price", None)

    async def place_exit_orders(
        self,
        symbol: str,
        signal: Signal,
        exit_prices: Dict[str, Optional[float]],
        timeframe: Optional[str] = None,
    ):
        """Place native stop-loss and take-profit orders for a position."""
        engine = self.engine
        pos = engine.positions.get(symbol)
        if not pos:
            return

        # --- Cancel any existing exit orders for this position ---
        old_sl_id = pos.get("stop_loss_order_id")
        old_tp_id = pos.get("take_profit_order_id")
        for old_id in (old_sl_id, old_tp_id):
            if old_id:
                try:
                    await asyncio.to_thread(engine.trader.cancel_order, old_id)
                    logger.info(f"Cancelled old exit order {old_id} for {symbol}")
                except Exception as e:
                    logger.warning(f"Failed to cancel old exit order {old_id}: {e}")
                # Remove from queued_orders
                async with engine._queued_orders_lock:
                    engine.queued_orders = [
                        q for q in engine.queued_orders
                        if q.get("order_id") != old_id
                    ]
        # Clear the stored IDs so they are not reused
        pos.pop("stop_loss_order_id", None)
        pos.pop("take_profit_order_id", None)
        pos.pop("stop_loss_order_type", None)
        pos.pop("_native_stop_price", None)

        base, quote = symbol.split("/")
        qty = pos["amount"]  # base quantity to sell
        if qty <= 0:
            return

        # --- Stop-loss order ---
        sl_ot = signal.stop_loss_order_type
        sl_price = exit_prices.get("stop_loss_price")
        sl_order_id = None
        actual_sl_ot = None
        if sl_ot in ("stop", "stop_limit") and sl_price is not None:
            actual_sl_ot = sl_ot
            try:
                if sl_ot == "stop" or (sl_ot == "stop_limit" and signal.stop_loss_limit_price is None):
                    if sl_ot == "stop_limit":
                        actual_sl_ot = "stop"
                    order = await asyncio.to_thread(
                        engine.trader.create_stop_sell_order,
                        symbol, qty, sl_price,
                        time_in_force="gtc", timeout=60.0
                    )
                else:
                    limit_price = signal.stop_loss_limit_price
                    order = await asyncio.to_thread(
                        engine.trader.create_stop_limit_sell_order,
                        symbol, qty, sl_price, limit_price,
                        time_in_force="gtc", timeout=60.0
                    )
                sl_order_id = order["id"]
                _sl_queued = {
                    "symbol": symbol,
                    "side": "sell",
                    "amount": qty,
                    "original_amount": qty,
                    "limit_price": order.get("limit_price"),
                    "stop_price": order.get("stop_price"),
                    "trail_offset": order.get("trail_offset"),
                    "order_type": actual_sl_ot,
                    "time_in_force": "gtc",
                    "signal": asdict(signal),
                    "timeframe": timeframe,
                    "atr": None,
                    "exit_reason": "stop_loss",
                    "order_id": sl_order_id,
                    "queued_at": time.time(),
                    "filled_qty": 0,
                    "filled_cost": 0.0,
                    "is_exit_order": True,
                    "oco_pair": None,
                }
                async with engine._queued_orders_lock:
                    engine.queued_orders.append(_sl_queued)
            except Exception as e:
                logger.error(f"Failed to place stop-loss order for {symbol}: {e}")

        elif sl_ot == "trailing_stop":
            if is_btp_isin(symbol):
                logger.warning(
                    f"Skipping trailing-stop order for BTP {symbol}: "
                    f"trailing stops are not supported for BTPs on Intesa Sanpaolo Investo."
                )
            else:
                trail_offset = signal.stop_loss_trail_offset
                if trail_offset is not None and trail_offset > 0:
                    try:
                        order = await asyncio.to_thread(
                            engine.trader.create_trailing_stop_sell_order,
                            symbol, qty, trail_offset,
                            time_in_force="gtc", timeout=60.0
                        )
                        sl_order_id = order["id"]
                        _trail_queued = {
                            "symbol": symbol,
                            "side": "sell",
                            "amount": qty,
                            "original_amount": qty,
                            "limit_price": None,
                            "stop_price": None,
                            "trail_offset": trail_offset,
                            "order_type": "trailing_stop",
                            "time_in_force": "gtc",
                            "signal": asdict(signal),
                            "timeframe": timeframe,
                            "atr": None,
                            "exit_reason": "stop_loss",
                            "order_id": sl_order_id,
                            "queued_at": time.time(),
                            "filled_qty": 0,
                            "filled_cost": 0.0,
                            "is_exit_order": True,
                            "oco_pair": None,
                        }
                        async with engine._queued_orders_lock:
                            engine.queued_orders.append(_trail_queued)
                    except Exception as e:
                        logger.error(f"Failed to place trailing-stop order for {symbol}: {e}")

        # --- Take-profit order ---
        params = signal.strategy_params or {}
        trailing_tp = params.get("trailing_take_profit", False)
        partial_tp_levels = params.get("partial_take_profit_levels")
        partial_tp_pct = params.get("partial_take_profit_pct")

        tp_ot = signal.take_profit_order_type
        tp_price = exit_prices.get("take_profit_price")
        tp_order_id = None
        if (tp_ot == "limit" and tp_price is not None
                and not trailing_tp
                and not partial_tp_levels
                and not partial_tp_pct):
            try:
                order = await asyncio.to_thread(
                    engine.trader.create_market_sell_order,
                    symbol, qty, 60.0,
                    limit_price=tp_price, time_in_force="gtc"
                )
                tp_order_id = order["id"]
                _tp_queued = {
                    "symbol": symbol,
                    "side": "sell",
                    "amount": qty,
                    "original_amount": qty,
                    "limit_price": tp_price,
                    "stop_price": None,
                    "trail_offset": None,
                    "order_type": "limit",
                    "time_in_force": "gtc",
                    "signal": asdict(signal),
                    "timeframe": timeframe,
                    "atr": None,
                    "exit_reason": "take_profit",
                    "order_id": tp_order_id,
                    "queued_at": time.time(),
                    "filled_qty": 0,
                    "filled_cost": 0.0,
                    "is_exit_order": True,
                    "oco_pair": None,
                }
                async with engine._queued_orders_lock:
                    engine.queued_orders.append(_tp_queued)
            except Exception as e:
                logger.error(f"Failed to place take-profit order for {symbol}: {e}")

        # --- Link OCO pair ---
        if sl_order_id and tp_order_id:
            for q in engine.queued_orders:
                if q.get("order_id") == sl_order_id:
                    q["oco_pair"] = tp_order_id
                elif q.get("order_id") == tp_order_id:
                    q["oco_pair"] = sl_order_id

        # Store order IDs in position for risk management
        pos["stop_loss_order_id"] = sl_order_id
        if actual_sl_ot:
            pos["stop_loss_order_type"] = actual_sl_ot
        pos["take_profit_order_id"] = tp_order_id

        # Notify user
        if engine.notifier:
            stock_name = await engine._get_stock_name(symbol)
            display_symbol = engine._format_symbol_display(symbol, stock_name, pos.get("timeframe"))
            msg = f"🛡️ Exit orders placed for {display_symbol}:\n"
            if sl_order_id:
                sl_type = actual_sl_ot or "stop"
                if actual_sl_ot == "trailing_stop":
                    msg += f"  🛑 Trailing stop: offset ${trail_offset:.2f}\n"
                elif actual_sl_ot == "stop_limit":
                    msg += f"  🛑 Stop-limit: stop ${sl_price:.2f}, limit ${signal.stop_loss_limit_price:.2f}\n"
                else:
                    msg += f"  🛑 Stop: ${sl_price:.2f}\n"
            if tp_order_id:
                msg += f"  🎯 Take-profit: limit ${tp_price:.2f}\n"
            if sl_order_id and tp_order_id:
                msg += "  (OCO – one cancels the other)"
            await engine.notifier.send_notification(
                msg,
                summary={
                    "symbol": symbol,
                    "action": "INFO",
                    "reason": "Exit orders placed",
                    "stop_loss_order_id": sl_order_id,
                    "take_profit_order_id": tp_order_id,
                }
            )

    async def cleanup_orphaned_orders(self):
        """Periodically cancel any open orders that are older than 10 minutes,
        but never cancel orders that are still being tracked as queued."""
        engine = self.engine
        open_orders = await asyncio.to_thread(engine.trader.get_open_orders)
        now = time.time()
        # Build a set of order IDs that are currently queued (waiting for fill)
        queued_ids = {q.get('order_id') for q in engine.queued_orders if q.get('order_id')}
        for order in open_orders:
            order_id = order.get('id')
            if order_id in queued_ids:
                continue   # this order is being monitored by _process_queued_orders
            created_at = order.get('timestamp', 0) / 1000.0  # ms to seconds
            if now - created_at > 600:  # 10 minutes
                logger.warning(
                    f"Cancelling orphaned order {order_id} for {order['symbol']} "
                    f"(open for {now - created_at:.0f}s)."
                )
                await asyncio.to_thread(engine.trader.cancel_order, order_id)

    async def handle_queued_buy_fill(self, trade_dict: Dict[str, Any], queued: Dict[str, Any]):
        """Process a queued BUY limit order that has filled in the simulator."""
        engine = self.engine
        symbol = trade_dict['symbol']
        parts = symbol.split("/")
        if len(parts) != 2:
            logger.error(f"Invalid symbol format in queued buy fill: {symbol}")
            return
        base, quote = parts
        fee = trade_dict.get('fee', {})
        fee_cost = float(fee.get('cost', 0.0) or 0.0)
        fee_currency = fee.get('currency', '')
        cost_basis = trade_dict['cost'] + (fee_cost if fee_currency == quote else 0.0)
        net_base = trade_dict['amount'] - (fee_cost if fee_currency == base else 0.0)

        signal_dict = queued.get('signal', {}) or {}
        params = signal_dict.get('strategy_params', {}) or {}
        timeframe = queued.get('timeframe')
        atr = queued.get('atr')
        fill_price = trade_dict['price']

        # Determine stop-loss percentage based on method
        stop_method = params.get("stop_loss_method", "fixed")
        if stop_method == "atr_multiple" and atr is not None and atr > 0 and fill_price > 0:
            atr_mult = params.get("stop_loss_atr_multiple")
            if atr_mult is not None:
                sl_pct = (atr_mult * atr) / fill_price
            else:
                sl_pct = params.get("stop_loss_pct")
        else:
            sl_pct = params.get("stop_loss_pct")

        # Determine take-profit percentage based on method
        if "take_profit_atr_multiple" in params and atr is not None and atr > 0 and fill_price > 0:
            tp_atr_mult = params["take_profit_atr_multiple"]
            tp_pct = (tp_atr_mult * atr) / fill_price
        else:
            tp_pct = params.get("take_profit_pct")
        # --- BTP take-profit cap: enforce smaller targets for bonds ---
        if is_btp_isin(symbol) and tp_pct is not None and tp_pct > 0:
            if tp_pct > settings.BTP_MAX_TAKE_PROFIT_PCT:
                logger.info(
                    f"BTP take-profit capped for {symbol} (queued fill): {tp_pct:.4%} -> "
                    f"{settings.BTP_MAX_TAKE_PROFIT_PCT:.4%}"
                )
                tp_pct = settings.BTP_MAX_TAKE_PROFIT_PCT
        trailing_stop = params.get("trailing_stop", False)
        _qbuy_is_btp = is_btp_isin(symbol)
        if _qbuy_is_btp and trailing_stop:
            logger.warning(
                f"LLM set trailing_stop=true for BTP {symbol} (queued fill), but trailing stops are not supported "
                f"for BTPs on Intesa Sanpaolo Investo. Forcing trailing_stop=false."
            )
            trailing_stop = False
        trailing_stop_distance_pct = params.get("trailing_stop_distance_pct")
        indicator_config = signal_dict.get('indicator_config')

        if symbol in engine.positions:
            old_cost_basis = engine.positions[symbol].get("cost_basis", engine.positions[symbol]["amount"] * engine.positions[symbol]["price"])
            old_net_base = engine.positions[symbol].get("net_base", engine.positions[symbol]["amount"])
            new_cost_basis = old_cost_basis + cost_basis
            new_net_base = old_net_base + net_base
            new_price = new_cost_basis / new_net_base if new_net_base > 0 else 0.0
            engine.positions[symbol]["amount"] = new_net_base
            engine.positions[symbol]["price"] = new_price
            engine.positions[symbol]["cost_basis"] = new_cost_basis
            engine.positions[symbol]["net_base"] = new_net_base
            # Preserve existing absolute SL/TP prices when scaling in.
            # Recalculating based on the new weighted average would shift
            # them from where the LLM originally intended.
            engine.positions[symbol]["take_profit_atr_multiple"] = params.get("take_profit_atr_multiple")
            engine.positions[symbol]["trailing_stop"] = trailing_stop
            engine.positions[symbol]["trailing_stop_distance_pct"] = trailing_stop_distance_pct
            engine.positions[symbol]["max_hold_time_seconds"] = params.get("max_hold_time_seconds")
            engine.positions[symbol]["trailing_stop_activation_pct"] = params.get("trailing_stop_activation_pct")
            engine.positions[symbol]["trailing_take_profit"] = params.get("trailing_take_profit", False)
            engine.positions[symbol]["trailing_take_profit_distance_pct"] = params.get("trailing_take_profit_distance_pct")
            engine.positions[symbol]["breakeven_activation_pct"] = params.get("breakeven_activation_pct")
            partial_levels = params.get("partial_take_profit_levels")
            if partial_levels:
                engine.positions[symbol]["partial_take_profit_levels"] = partial_levels
                engine.positions[symbol]["partial_tp_levels_triggered"] = []
                engine.positions[symbol]["partial_tp_depth_wait_start"] = {}
                engine.positions[symbol]["partial_take_profit_pct"] = None
                engine.positions[symbol]["partial_take_profit_fraction"] = None
                engine.positions[symbol]["partial_tp_triggered"] = None
            else:
                engine.positions[symbol]["partial_take_profit_pct"] = params.get("partial_take_profit_pct")
                engine.positions[symbol]["partial_take_profit_fraction"] = params.get("partial_take_profit_fraction")
                engine.positions[symbol]["partial_tp_triggered"] = False
            engine.positions[symbol]["cooldown_after_loss_seconds"] = params.get("cooldown_after_loss_seconds", 0)
            engine.positions[symbol]["news_sentiment_exit_threshold"] = params.get("news_sentiment_exit_threshold")
            engine.positions[symbol]["max_unrealized_loss_pct"] = params.get("max_unrealized_loss_pct")
            engine.positions[symbol]["timeframe"] = timeframe
            engine.positions[symbol]["indicator_config"] = indicator_config
            engine.positions[symbol]["entry_order_type"] = queued.get('order_type', 'market')
            engine.positions[symbol]["buy_confidence"] = signal_dict.get('confidence', 0.0)
            engine.positions[symbol]["buy_reasoning"] = (signal_dict.get('reasoning', '') or '')[:200]
        else:
            entry_price = cost_basis / net_base if net_base > 0 else trade_dict["price"]
            engine.positions[symbol] = {
                "symbol": symbol,
                "side": "buy",
                "amount": net_base,
                "price": entry_price,
                "timestamp": trade_dict["timestamp"],
                "stop_loss": entry_price * (1 - sl_pct) if sl_pct else None,
                "take_profit": entry_price * (1 + tp_pct) if tp_pct else None,
                "take_profit_atr_multiple": params.get("take_profit_atr_multiple"),
                "cost_basis": cost_basis,
                "net_base": net_base,
                "buy_confidence": signal_dict.get('confidence', 0.0),
                "buy_reasoning": (signal_dict.get('reasoning', '') or '')[:200],
                "trailing_stop": trailing_stop,
                "trailing_stop_distance_pct": trailing_stop_distance_pct,
                "max_hold_time_seconds": params.get("max_hold_time_seconds"),
                "trailing_stop_activation_pct": params.get("trailing_stop_activation_pct"),
                "trailing_take_profit": params.get("trailing_take_profit", False),
                "trailing_take_profit_distance_pct": params.get("trailing_take_profit_distance_pct"),
                "breakeven_activation_pct": params.get("breakeven_activation_pct"),
                "partial_take_profit_levels": params.get("partial_take_profit_levels"),
                "partial_tp_levels_triggered": [],
                "partial_tp_depth_wait_start": {},
                "original_amount": net_base,
                "partial_take_profit_pct": params.get("partial_take_profit_pct") if not params.get("partial_take_profit_levels") else None,
                "partial_take_profit_fraction": params.get("partial_take_profit_fraction") if not params.get("partial_take_profit_levels") else None,
                "partial_tp_triggered": False if not params.get("partial_take_profit_levels") else None,
                "cooldown_after_loss_seconds": params.get("cooldown_after_loss_seconds", 0),
                "news_sentiment_exit_threshold": params.get("news_sentiment_exit_threshold"),
                "max_unrealized_loss_pct": params.get("max_unrealized_loss_pct"),
                "timeframe": timeframe,
                "indicator_config": indicator_config,
                "entry_order_type": queued.get('order_type', 'market'),
            }

        custom_interval = params.get("strategy_interval_seconds")
        if custom_interval is not None:
            engine._strategy_intervals[symbol] = custom_interval

        trade_dict['strategy_type'] = signal_dict.get('strategy_type')
        trade_dict['timeframe'] = timeframe
        trade_dict['buy_confidence'] = signal_dict.get('confidence', 0.0)
        trade_dict['buy_reasoning'] = (signal_dict.get('reasoning', '') or '')[:200]
        engine._append_trade(trade_dict)
        engine._balance_cache = None
        # Note: _cycle_spent was already updated when the order was queued
        # in _execute_signal, so we do NOT add to it here to avoid double-counting.
        await asyncio.to_thread(insert_trade, trade_dict)
        await engine._save_state(force=True)
        engine._portfolio_exposure_cache = None
        if engine.notifier:
            stock_name = await engine._get_stock_name(symbol)
            display_symbol = engine._format_symbol_display(symbol, stock_name, timeframe)
            buy_msg = f"🟢 BUY {display_symbol}: {trade_dict['amount']:.6f} @ {trade_dict['price']:.4f}"
            buy_summary = {
                "symbol": symbol,
                "action": "BUY",
                "price": trade_dict["price"],
                "amount": trade_dict["amount"],
                "confidence": signal_dict.get('confidence', 0.0),
                "reason": (signal_dict.get('reasoning', '') or '')[:200],
                "strategy_type": signal_dict.get('strategy_type'),
                "indicators": {"atr": atr},
            }
            if signal_dict.get('model_type'):
                buy_summary["model_type"] = signal_dict.get('model_type')
            if signal_dict.get('llm_provider'):
                buy_summary["llm_provider"] = signal_dict.get('llm_provider')
            if signal_dict.get('llm_model'):
                buy_summary["llm_model"] = signal_dict.get('llm_model')
            await engine.notifier.send_notification(buy_msg, summary=buy_summary)

        # Place native exit orders for the new/updated position
        signal_dict = queued.get('signal', {}) or {}
        if signal_dict:
            try:
                # Reconstruct a Signal from the stored dict, filtering to only
                # valid Signal fields and providing fallbacks for required fields.
                import dataclasses as _dc
                valid_keys = {f.name for f in _dc.fields(Signal)}
                filtered = {k: v for k, v in signal_dict.items() if k in valid_keys}
                # Ensure required fields have fallbacks
                if "action" not in filtered:
                    filtered["action"] = "BUY"
                if "confidence" not in filtered:
                    filtered["confidence"] = 0.0
                if "reasoning" not in filtered:
                    filtered["reasoning"] = ""
                reconstructed_signal = Signal(**filtered)
                exit_prices = self.compute_exit_order_prices(
                    entry_price=engine.positions[symbol]["price"],
                    signal=reconstructed_signal,
                    atr=queued.get('atr'),
                )
                await self.place_exit_orders(symbol, reconstructed_signal, exit_prices, queued.get('timeframe'))
            except Exception as e:
                logger.error(f"Failed to place exit orders after queued buy fill for {symbol}: {e}")
                if engine.notifier:
                    stock_name = await engine._get_stock_name(symbol)
                    display_symbol = engine._format_symbol_display(symbol, stock_name, queued.get('timeframe'))
                    await engine.notifier.send_notification(
                        f"⚠️ Exit order placement failed for {display_symbol} after queued fill: {e}",
                        summary={
                            "symbol": symbol,
                            "action": "ERROR",
                            "reason": f"Exit order placement failed after queued fill: {str(e)[:200]}",
                        }
                    )

    async def handle_queued_sell_fill(self, trade_dict: Dict[str, Any], queued: Dict[str, Any], partial: bool = False):
        """Process a queued SELL limit order that has filled in the simulator.

        When *partial* is True, only a portion of the order has filled; the
        position is prorated and updated rather than removed.
        """
        engine = self.engine
        symbol = trade_dict['symbol']
        parts = symbol.split("/")
        if len(parts) != 2:
            logger.error(f"Invalid symbol format in queued sell fill: {symbol}")
            return
        base, quote = parts
        pos = engine.positions.get(symbol)
        fee = trade_dict.get('fee', {})
        fee_cost = float(fee.get('cost', 0.0) or 0.0)
        fee_currency = fee.get('currency', '')
        net_quote = trade_dict['cost'] - (fee_cost if fee_currency == quote else 0.0)
        exit_reason = queued.get('exit_reason', 'limit_order')
        trade_dict['exit_reason'] = exit_reason
        signal_dict = queued.get('signal', {}) or {}
        trade_dict['strategy_type'] = signal_dict.get('strategy_type')
        trade_dict['timeframe'] = queued.get('timeframe')
        if pos:
            trade_dict['buy_confidence'] = pos.get("buy_confidence", 0.0)
            trade_dict['buy_reasoning'] = pos.get("buy_reasoning", "")
        if pos and "timestamp" in pos:
            trade_dict['hold_time_seconds'] = (trade_dict['timestamp'] - pos["timestamp"]) / 1000.0
        else:
            trade_dict['hold_time_seconds'] = None

        if partial and pos:
            # Prorated cost basis for the sold portion
            cost_basis = pos.get("cost_basis", pos["amount"] * pos["price"])
            net_base = pos.get("net_base", pos["amount"])
            prorated_cost_basis = cost_basis * (trade_dict['amount'] / net_base) if net_base > 0 else 0.0
            realized_pnl = net_quote - prorated_cost_basis
            trade_dict['realized_pnl'] = realized_pnl
            trade_dict['cost_basis'] = prorated_cost_basis

            # Update position
            remaining_amount = pos["amount"] - trade_dict['amount']
            remaining_cost_basis = cost_basis - prorated_cost_basis
            remaining_net_base = net_base - trade_dict['amount']

            # Cancel old exit orders because quantity changed
            await engine._cancel_exit_orders(symbol)

            if remaining_amount <= 0 or remaining_net_base <= 0:
                # Position fully closed via partial fills
                if realized_pnl < 0:
                    engine.last_loss_time[symbol] = time.time()
                    engine.cooldown_durations[symbol] = pos.get("cooldown_after_loss_seconds", 0)
                pos.pop("_stop_loss_triggered", None)
                pos.pop("_stop_loss_review_count", None)
                async with engine._positions_lock:
                    engine.positions.pop(symbol, None)
                engine._strategy_intervals.pop(symbol, None)
                engine._last_strategy_eval.pop(symbol, None)
                engine._last_decisions.pop(symbol, None)
                engine._pending_entries.pop(symbol, None)
                await engine._remove_symbol_if_paused(symbol)
            else:
                async with engine._positions_lock:
                    engine.positions[symbol]["amount"] = remaining_amount
                    engine.positions[symbol]["cost_basis"] = remaining_cost_basis
                    engine.positions[symbol]["net_base"] = remaining_net_base
                    engine.positions[symbol]["price"] = remaining_cost_basis / remaining_net_base if remaining_net_base > 0 else 0.0

                # Replace exit orders for the remaining amount
                from src.strategies.base import Signal
                async with engine._positions_lock:
                    dummy_params = {
                        "trailing_take_profit": engine.positions[symbol].get("trailing_take_profit", False),
                        "partial_take_profit_levels": engine.positions[symbol].get("partial_take_profit_levels"),
                        "partial_take_profit_pct": engine.positions[symbol].get("partial_take_profit_pct"),
                    }
                    dummy_signal = Signal(
                        action="BUY",
                        confidence=1.0,
                        reasoning="Replacing exit orders after partial sell fill",
                        stop_loss_order_type=engine.positions[symbol].get("stop_loss_order_type"),
                        stop_loss_stop_price=engine.positions[symbol].get("stop_loss"),
                        stop_loss_limit_price=None,
                        take_profit_order_type=engine.positions[symbol].get("take_profit_order_type"),
                        take_profit_limit_price=engine.positions[symbol].get("take_profit"),
                        strategy_params=dummy_params,
                    )
                    exit_prices = {
                        "stop_loss_price": engine.positions[symbol].get("stop_loss"),
                        "take_profit_price": engine.positions[symbol].get("take_profit"),
                    }
                await engine._place_exit_orders(symbol, dummy_signal, exit_prices, engine.positions[symbol].get("timeframe"))
        else:
            # Full fill (non-partial) – original logic
            # Cancel any remaining exit orders before removing the position
            await engine._cancel_exit_orders(symbol)
            if pos:
                cost_basis = pos.get("cost_basis", pos["amount"] * pos["price"])
                realized_pnl = net_quote - cost_basis
            else:
                realized_pnl = 0.0
            trade_dict['realized_pnl'] = realized_pnl
            trade_dict['cost_basis'] = pos.get("cost_basis", 0.0) if pos else 0.0
            if realized_pnl < 0:
                engine.last_loss_time[symbol] = time.time()
                engine.cooldown_durations[symbol] = pos.get("cooldown_after_loss_seconds", 0) if pos else 0
            if pos:
                pos.pop("_stop_loss_triggered", None)
                pos.pop("_stop_loss_review_count", None)
            async with engine._positions_lock:
                engine.positions.pop(symbol, None)
            engine._strategy_intervals.pop(symbol, None)
            engine._last_strategy_eval.pop(symbol, None)
            engine._last_decisions.pop(symbol, None)
            engine._pending_entries.pop(symbol, None)
            await engine._remove_symbol_if_paused(symbol)

        engine._append_trade(trade_dict)
        engine._balance_cache = None
        await asyncio.to_thread(insert_trade, trade_dict)
        await engine._save_state(force=True)
        engine._portfolio_exposure_cache = None
        if engine.notifier:
            reason_labels = {
                "manual_sell": "🖐️ Manual",
                "manual_sell_all": "🖐️ Manual (Sell All)",
                "stop_loss": "⛔ Stop-Loss",
                "take_profit": "✅ Take-Profit",
                "max_hold_time": "⏰ Max Hold Time",
                "news_sentiment_exit": "📰 News Sentiment",
                "force_close": "🔻 Force Close",
                "external_sell": "🔄 External Sell",
                "delisted": "🗑️ Delisted",
            }
            reason_label = reason_labels.get(exit_reason, exit_reason) if exit_reason else None
            reason_str = f" [{reason_label}]" if reason_label else ""
            stock_name = await engine._get_stock_name(symbol)
            tf = queued.get('timeframe') or (pos.get("timeframe") if pos else None)
            display_symbol = engine._format_symbol_display(symbol, stock_name, tf)
            partial_str = " (partial)" if partial else ""
            sell_msg = f"🔴 SELL{reason_str}{partial_str} {display_symbol}: {trade_dict['amount']:.6f} @ {trade_dict['price']:.4f}"
            if pos:
                cb = trade_dict.get('cost_basis', 0.0) or (pos.get("cost_basis", pos["amount"] * pos["price"]) if pos else 0.0)
                pnl_pct = (realized_pnl / cb * 100) if cb > 0 else 0.0
                sell_msg += f" | P&L: {realized_pnl:+.4f} ({pnl_pct:+.2f}%)"
            sell_summary = {
                "symbol": symbol,
                "action": "SELL",
                "price": trade_dict["price"],
                "amount": trade_dict["amount"],
                "confidence": signal_dict.get('confidence', 0.0),
                "reason": (signal_dict.get('reasoning', '') or '')[:200],
                "exit_reason": exit_reason,
                "realized_pnl": realized_pnl,
                "strategy_type": signal_dict.get('strategy_type'),
                "indicators": {"atr": queued.get('atr')},
            }
            if signal_dict.get('model_type'):
                sell_summary["model_type"] = signal_dict.get('model_type')
            if signal_dict.get('llm_provider'):
                sell_summary["llm_provider"] = signal_dict.get('llm_provider')
            if signal_dict.get('llm_model'):
                sell_summary["llm_model"] = signal_dict.get('llm_model')
            await engine.notifier.send_notification(sell_msg, summary=sell_summary)

    async def sweep_dust(self, symbol: str):
        """Sell any remaining dust balance of a symbol after a partial sell."""
        engine = self.engine
        base = symbol.split("/")[0]
        try:
            balance = await asyncio.to_thread(engine.trader.get_balance, base)
        except Exception as e:
            logger.warning(f"Dust sweep: could not fetch balance for {base}: {e}")
            return
        if balance <= 0:
            return

        stock_name = await engine._get_stock_name(symbol)
        tf = engine.positions.get(symbol, {}).get("timeframe") if symbol in engine.positions else None
        display_symbol = engine._format_symbol_display(symbol, stock_name, tf)

        try:
            base = symbol.split("/")[0]
            quotes = await engine._get_quotes_async([base], timeout=45.0)
            ticker = quotes.get(base)
            price = ticker["last"]
        except Exception as e:
            logger.warning(f"Dust sweep: could not fetch price for {symbol}: {e}")
            return

        # Fetch minimum order size from asset info
        try:
            asset = await engine._get_asset_info(symbol)
            min_amount = float(asset.min_order_size) if asset.min_order_size else None
            if not asset.fractionable and (min_amount is None or min_amount < 1.0):
                min_amount = 1.0
        except Exception:
            min_amount = None
        if min_amount is not None and balance < float(min_amount):
            logger.info(f"Dust sweep: {balance} {base} below min amount {min_amount}, cannot sell.")
            return

        if not await engine._is_market_open():
            logger.info(f"Dust sweep for {symbol} deferred: market closed. Will retry on next market open.")
            if symbol in engine.positions:
                async with engine._positions_lock:
                    engine.positions[symbol]["_dust_sweep_pending"] = True
            return

        need_limit = not engine._is_regular_hours()
        limit_price = None
        time_in_force = "day"
        if need_limit:
            limit_price = engine._default_limit_price(symbol, "SELL", ticker, atr=None)
            if limit_price is None:
                logger.error(f"Cannot place limit order for dust sweep on {symbol}: no limit price.")
                return

        if limit_price is not None and limit_price <= 0:
            logger.error(f"Invalid limit_price for dust sweep on {symbol}, skipping.")
            return

        try:
            order = await asyncio.to_thread(
                engine.trader.create_market_sell_order, symbol, balance,
                settings.ORDER_FILL_TIMEOUT_SECONDS, limit_price, time_in_force
            )
            logger.info(f"Dust sweep: sold {balance} {base} from {symbol} – order {order.get('id')}")

            # Record the dust sale in trade history for consistency
            fee = order.get('fee', {})
            fee_cost = float(fee.get('cost', 0.0) or 0.0)
            fee_currency = fee.get('currency', '')
            pos = engine.positions.get(symbol)
            if pos:
                cost_basis = pos.get("cost_basis", pos["amount"] * pos["price"])
                net_quote = order['cost'] - (fee_cost if fee_currency == symbol.split('/')[1] else 0.0)
                realized_pnl = net_quote - cost_basis
                order["realized_pnl"] = realized_pnl
                order["cost_basis"] = cost_basis
                order["exit_reason"] = "dust_sweep"
                order["strategy_type"] = pos.get("strategy_type", "unknown")
                order["timeframe"] = pos.get("timeframe")
                if "timestamp" in pos:
                    order["hold_time_seconds"] = (order["timestamp"] - pos["timestamp"]) / 1000.0
                engine._append_trade(order)
                await asyncio.to_thread(insert_trade, order)
                await engine._save_state(force=True)
                engine._portfolio_exposure_cache = None

            # Cancel any remaining exit orders before removing the position
            await engine._cancel_exit_orders(symbol)

            # Remove the now-empty position
            async with engine._positions_lock:
                engine.positions.pop(symbol, None)
            engine._strategy_intervals.pop(symbol, None)
            engine._last_strategy_eval.pop(symbol, None)
            await engine._remove_symbol_if_paused(symbol)

            if engine.notifier:
                await engine.notifier.send_notification(
                    f"🧹 Dust sweep: sold remaining {balance} {base} from {display_symbol}",
                    summary={
                        "symbol": symbol,
                        "action": "SELL",
                        "reason": "Dust sweep",
                        "amount": balance,
                        "exit_reason": "dust_sweep",
                    }
                )
        except Exception as e:
            logger.error(f"Dust sweep failed for {symbol}: {e}")

    async def execute_partial_sell(
        self,
        symbol: str,
        sell_amount: float,
        level_label: str,
        exit_reason: str,
        ticker: Optional[Dict[str, Any]] = None,
        atr: Optional[float] = None,
        current_price: float = 0.0,
        cleanup_callback=None,
        extra_summary: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Execute a partial sell (used by partial take-profit single and multi-level).

        Handles order creation, cost-basis proration, position update, exit-order
        replacement, dust sweep, trade recording, and notification.

        cleanup_callback(symbol, position_dict) is called inside the positions lock
        after the position amount/cost is updated (only when the position survives).
        """
        engine = self.engine
        pos = engine.positions.get(symbol)
        if not pos:
            logger.warning(f"Cannot execute partial sell for {symbol}: no position.")
            return False

        stock_name = await engine._get_stock_name(symbol)
        tf = pos.get("timeframe")
        display_symbol = engine._format_symbol_display(symbol, stock_name, tf)
        base, quote = symbol.split("/")

        # Check minimum sell size
        try:
            asset = await engine._get_asset_info(symbol)
            min_amount = float(asset.min_order_size) if asset.min_order_size else None
            if not asset.fractionable and (min_amount is None or min_amount < 1.0):
                min_amount = 1.0
        except Exception:
            min_amount = None
        if min_amount is not None and sell_amount < float(min_amount):
            logger.info(f"{level_label} sell amount {sell_amount:.6f} below min {min_amount} for {symbol}, skipping.")
            return False

        if not await engine._is_market_open():
            logger.info(f"{level_label} for {symbol} skipped: market closed.")
            return False

        need_limit = not engine._is_regular_hours()
        limit_price = None
        time_in_force = "day"
        if need_limit:
            limit_price = engine._default_limit_price(symbol, "SELL", ticker, atr=atr)
            if limit_price is None:
                logger.error(f"Cannot place limit order for {level_label} on {symbol}: no limit price.")
                return False
            if limit_price <= 0:
                logger.error(f"Invalid limit_price for {level_label} on {symbol}, skipping.")
                return False

        try:
            order = await asyncio.to_thread(
                engine.trader.create_market_sell_order, symbol, sell_amount,
                settings.ORDER_FILL_TIMEOUT_SECONDS, limit_price, time_in_force
            )
            filled_amount = order.get("amount", sell_amount)
            fill_price = order.get("price", current_price)
            logger.info(f"{level_label} SELL {symbol}: {filled_amount:.6f} @ {fill_price:.4f}")

            fee = order.get("fee", {})
            fee_cost = float(fee.get("cost", 0.0) or 0.0)
            fee_currency = fee.get("currency", "")
            cost_basis = pos.get("cost_basis", pos["amount"] * pos["price"])
            net_base = pos.get("net_base", pos["amount"])
            prorated_cost_basis = cost_basis * (filled_amount / net_base) if net_base > 0 else 0.0
            net_quote = order["cost"] - (fee_cost if fee_currency == quote else 0.0)
            realized_pnl = net_quote - prorated_cost_basis

            order["realized_pnl"] = realized_pnl
            order["cost_basis"] = prorated_cost_basis
            order["exit_reason"] = exit_reason
            order["strategy_type"] = pos.get("strategy_type", "unknown")
            order["timeframe"] = pos.get("timeframe")
            if "timestamp" in pos:
                order["hold_time_seconds"] = (order["timestamp"] - pos["timestamp"]) / 1000.0

            remaining_amount = pos["amount"] - filled_amount
            remaining_cost_basis = cost_basis - prorated_cost_basis
            remaining_net_base = net_base - filled_amount

            await engine._cancel_exit_orders(symbol)

            if remaining_amount <= 0 or remaining_net_base <= 0:
                async with engine._positions_lock:
                    engine.positions.pop(symbol, None)
                engine._strategy_intervals.pop(symbol, None)
                engine._last_strategy_eval.pop(symbol, None)
                engine._pending_entries.pop(symbol, None)
                await engine._remove_symbol_if_paused(symbol)
            else:
                async with engine._positions_lock:
                    engine.positions[symbol]["amount"] = remaining_amount
                    engine.positions[symbol]["cost_basis"] = remaining_cost_basis
                    engine.positions[symbol]["net_base"] = remaining_net_base
                    engine.positions[symbol]["price"] = remaining_cost_basis / remaining_net_base if remaining_net_base > 0 else 0.0
                    if cleanup_callback:
                        cleanup_callback(symbol, engine.positions[symbol])

                is_dust = min_amount is not None and remaining_amount < float(min_amount)
                if is_dust:
                    logger.info(f"Remaining {remaining_amount:.6f} {base} is dust after {level_label} for {symbol}, sweeping.")
                    await engine._sweep_dust(symbol)
                else:
                    from src.strategies.base import Signal
                    dummy_params = {
                        "trailing_take_profit": engine.positions[symbol].get("trailing_take_profit", False),
                        "partial_take_profit_levels": engine.positions[symbol].get("partial_take_profit_levels"),
                        "partial_take_profit_pct": engine.positions[symbol].get("partial_take_profit_pct"),
                    }
                    dummy_signal = Signal(
                        action="BUY",
                        confidence=1.0,
                        reasoning=f"Replacing exit orders after {level_label}",
                        stop_loss_order_type=engine.positions[symbol].get("stop_loss_order_type"),
                        stop_loss_stop_price=engine.positions[symbol].get("stop_loss"),
                        stop_loss_limit_price=None,
                        take_profit_order_type=engine.positions[symbol].get("take_profit_order_type"),
                        take_profit_limit_price=engine.positions[symbol].get("take_profit"),
                        strategy_params=dummy_params,
                    )
                    exit_prices = {
                        "stop_loss_price": engine.positions[symbol].get("stop_loss"),
                        "take_profit_price": engine.positions[symbol].get("take_profit"),
                    }
                    await engine._place_exit_orders(symbol, dummy_signal, exit_prices, engine.positions[symbol].get("timeframe"))

            engine._append_trade(order)
            await asyncio.to_thread(insert_trade, order)
            await engine._save_state(force=True)
            engine._portfolio_exposure_cache = None

            if engine.notifier:
                pnl_pct = (realized_pnl / prorated_cost_basis * 100) if prorated_cost_basis > 0 else 0.0
                summary = {
                    "symbol": symbol,
                    "action": "SELL",
                    "reason": level_label,
                    "amount": filled_amount,
                    "price": fill_price,
                    "realized_pnl": realized_pnl,
                    "exit_reason": exit_reason,
                }
                if extra_summary:
                    summary.update(extra_summary)
                await engine.notifier.send_notification(
                    f"🔸 {level_label} SELL {display_symbol}: {filled_amount:.6f} @ {fill_price:.4f} "
                    f"| P&L: {realized_pnl:+.4f} ({pnl_pct:+.2f}%)",
                    summary=summary,
                )
            return True
        except Exception as e:
            logger.error(f"{level_label} sell failed for {symbol}: {e}")
            if engine.notifier:
                await engine.notifier.send_notification(
                    f"❌ {level_label} sell failed for {display_symbol}: {e}",
                    summary={"symbol": symbol, "action": "ERROR", "reason": f"{level_label} sell failed: {e}"[:200]}
                )
            return False

    async def execute_partial_tp_single(
        self, symbol: str, current_price: float, atr: Optional[float], ticker: Dict[str, Any]
    ) -> None:
        """Execute a single partial take-profit sell for a position."""
        engine = self.engine
        pos = engine.positions.get(symbol)
        if not pos:
            logger.warning(f"Cannot execute partial TP for {symbol}: no position.")
            return

        fraction = pos.get("partial_take_profit_fraction")
        if fraction is None or fraction <= 0 or fraction >= 1:
            logger.warning(f"Invalid partial_take_profit_fraction for {symbol}: {fraction}")
            return

        sell_amount = pos["amount"] * fraction

        def _cleanup(sym, position):
            position.pop("partial_tp_triggered", None)
            position.pop("_partial_tp_triggered_single", None)
            position.pop("_partial_tp_single_review_count", None)

        await self.execute_partial_sell(
            symbol=symbol,
            sell_amount=sell_amount,
            level_label="Partial TP",
            exit_reason="partial_take_profit",
            ticker=ticker,
            atr=atr,
            current_price=current_price,
            cleanup_callback=_cleanup,
        )

    async def execute_partial_tp_level(
        self, symbol: str, level_index: int, current_price: float, atr: Optional[float], ticker: Dict[str, Any]
    ) -> None:
        """Execute a partial take-profit sell for a specific level."""
        engine = self.engine
        pos = engine.positions.get(symbol)
        if not pos:
            logger.warning(f"Cannot execute partial TP level for {symbol}: no position.")
            return

        levels = pos.get("partial_take_profit_levels")
        if not levels or level_index >= len(levels):
            logger.warning(f"Invalid partial TP level index {level_index} for {symbol}")
            return

        level = levels[level_index]
        fraction = level.get("fraction")
        if fraction is None or fraction <= 0 or fraction >= 1:
            logger.warning(f"Invalid fraction for partial TP level {level_index} of {symbol}: {fraction}")
            return

        sell_amount = pos["amount"] * fraction

        # Mark this level as triggered before the sell
        if symbol in engine.positions:
            async with engine._positions_lock:
                triggered = engine.positions[symbol].get("partial_tp_levels_triggered", [])
                if level_index not in triggered:
                    triggered.append(level_index)
                    engine.positions[symbol]["partial_tp_levels_triggered"] = triggered
                if "partial_tp_depth_wait_start" in engine.positions[symbol]:
                    engine.positions[symbol]["partial_tp_depth_wait_start"].pop(level_index, None)

        def _cleanup(sym, position):
            position.pop("_partial_tp_triggered", None)
            position.pop("_partial_tp_review_count", None)
            triggered_levels = position.get("_partial_tp_triggered_levels", [])
            position["_partial_tp_triggered_levels"] = [
                x for x in triggered_levels if x != level_index
            ]

        await self.execute_partial_sell(
            symbol=symbol,
            sell_amount=sell_amount,
            level_label=f"Partial TP level {level_index}",
            exit_reason=f"partial_take_profit_level_{level_index}",
            ticker=ticker,
            atr=atr,
            current_price=current_price,
            cleanup_callback=_cleanup,
            extra_summary={"level_index": level_index},
        )

    async def replace_native_stop_order(
        self,
        symbol: str,
        pos: Dict[str, Any],
        old_stop_price: float,
        new_stop_price: float,
    ):
        """Cancel the existing native stop order and place a new one with the updated stop price."""
        engine = self.engine
        old_order_id = pos.get("stop_loss_order_id")
        if not old_order_id:
            return

        # Capture the old queued entry's limit price (read-only, do NOT remove yet)
        async with engine._queued_orders_lock:
            old_queued = next(
                (q for q in engine.queued_orders if q.get("order_id") == old_order_id),
                None
            )
            old_limit_price = old_queued.get("limit_price") if old_queued else None

        # Place a new stop order
        qty = pos["amount"]
        sl_ot = pos.get("stop_loss_order_type", "stop")
        new_order_id = None
        try:
            if sl_ot == "stop":
                order = await asyncio.to_thread(
                    engine.trader.create_stop_sell_order,
                    symbol, qty, new_stop_price,
                    time_in_force="gtc", timeout=60.0
                )
            else:  # stop_limit
                # For stop_limit, use the original limit price from the old
                # queued entry, or fall back to the new stop price.
                limit_price = old_limit_price if old_limit_price is not None else new_stop_price
                order = await asyncio.to_thread(
                    engine.trader.create_stop_limit_sell_order,
                    symbol, qty, new_stop_price, limit_price,
                    time_in_force="gtc", timeout=60.0
                )
            new_order_id = order["id"]
            _replace_queued = {
                "symbol": symbol,
                "side": "sell",
                "amount": qty,
                "original_amount": qty,
                "limit_price": order.get("limit_price"),
                "stop_price": order.get("stop_price"),
                "trail_offset": order.get("trail_offset"),
                "order_type": sl_ot,
                "time_in_force": "gtc",
                "signal": {},  # no original signal for replacement
                "timeframe": pos.get("timeframe"),
                "atr": None,
                "exit_reason": "stop_loss",
                "order_id": new_order_id,
                "queued_at": time.time(),
                "filled_qty": 0,
                "filled_cost": 0.0,
                "is_exit_order": True,
                "oco_pair": pos.get("take_profit_order_id"),  # maintain OCO link
            }
            async with engine._queued_orders_lock:
                engine.queued_orders.append(_replace_queued)
                # Update OCO link on the take-profit order if it exists
                tp_order_id = pos.get("take_profit_order_id")
                if tp_order_id:
                    for q in engine.queued_orders:
                        if q.get("order_id") == tp_order_id:
                            q["oco_pair"] = new_order_id
                            break
            # Update position
            pos["stop_loss_order_id"] = new_order_id
            logger.info(f"Placed new stop order {new_order_id} for {symbol} at {new_stop_price:.4f}")

            # New order placed successfully — now safe to cancel the old one
            try:
                await asyncio.to_thread(engine.trader.cancel_order, old_order_id)
                logger.info(f"Cancelled old stop order {old_order_id} for {symbol} (replaced by {new_order_id})")
            except Exception as e:
                logger.warning(f"Failed to cancel old stop order {old_order_id} (new order {new_order_id} already placed): {e}")

            # Remove the old queued entry now that the new one is active
            async with engine._queued_orders_lock:
                engine.queued_orders = [
                    q for q in engine.queued_orders
                    if q.get("order_id") != old_order_id
                ]

            # Notify user
            if engine.notifier:
                stock_name = await engine._get_stock_name(symbol)
                display_symbol = engine._format_symbol_display(symbol, stock_name, pos.get("timeframe"))
                msg = f"🔄 Stop order updated for {display_symbol}: {old_stop_price:.4f} → {new_stop_price:.4f}"
                await engine.notifier.send_notification(
                    msg,
                    summary={
                        "symbol": symbol,
                        "action": "INFO",
                        "reason": "Stop order replaced",
                        "old_stop_price": old_stop_price,
                        "new_stop_price": new_stop_price,
                    }
                )
        except Exception as e:
            logger.error(
                f"Failed to place replacement stop order for {symbol}: {e}. "
                f"Old stop order {old_order_id} remains active at the previous price."
            )
            if engine.notifier:
                stock_name = await engine._get_stock_name(symbol)
                display_symbol = engine._format_symbol_display(symbol, stock_name, pos.get("timeframe"))
                await engine.notifier.send_notification(
                    f"⚠️ Stop order replacement failed for {display_symbol}: old stop order kept active.",
                    summary={
                        "symbol": symbol,
                        "action": "ERROR",
                        "reason": f"Stop order replacement failed, old order kept: {str(e)[:200]}",
                    }
                )
            return

    async def process_native_exit_fill(
        self,
        symbol: str,
        order_id: str,
        order_obj: Any,
        pos: Dict[str, Any],
        exit_reason: str,
    ):
        """Process a filled native exit order (stop-loss or take-profit) inline.

        This avoids the race condition where a native exit order fills between
        the OCO cancellation and a manual _execute_signal call, which would
        result in a double sell.
        """
        engine = self.engine
        # Find and remove the queued entry under the lock, but do NOT call
        # _handle_queued_sell_fill inside the lock — it internally acquires
        # _queued_orders_lock via _cancel_exit_orders, which would deadlock.
        async with engine._queued_orders_lock:
            queued = next((q for q in engine.queued_orders if q.get("order_id") == order_id), None)
            if queued:
                engine.queued_orders = [q for q in engine.queued_orders if q.get("order_id") != order_id]

        if queued:
            filled_qty = float(order_obj.filled_qty) if order_obj.filled_qty else 0.0
            filled_avg_price = float(order_obj.filled_avg_price) if order_obj.filled_avg_price else 0.0
            if filled_qty > 0:
                delta_cost = filled_qty * filled_avg_price
                from src.exchanges.fees import calculate_transaction_costs
                _quote_ccy = symbol.split("/")[1] if "/" in symbol else engine.base_currency
                _fee_costs = calculate_transaction_costs("SELL", filled_avg_price, filled_qty, symbol=symbol)
                trade_dict = {
                    'id': str(order_obj.id),
                    'symbol': symbol,
                    'side': 'sell',
                    'amount': filled_qty,
                    'price': filled_avg_price,
                    'cost': delta_cost,
                    'fee': {'cost': _fee_costs["total_costs"], 'currency': _quote_ccy},
                    'status': 'closed',
                    'timestamp': int(time.time() * 1000),
                }
                await engine._handle_queued_sell_fill(trade_dict, queued, partial=False)

        # Cancel the OCO pair if it still exists
        oco_pair_id = queued.get("oco_pair") if queued else None
        if oco_pair_id:
            try:
                await asyncio.to_thread(engine.trader.cancel_order, oco_pair_id)
            except Exception:
                pass
            async with engine._queued_orders_lock:
                engine.queued_orders = [q for q in engine.queued_orders if q.get("order_id") != oco_pair_id]
        pos.pop("stop_loss_order_id", None)
        pos.pop("take_profit_order_id", None)
        pos.pop("stop_loss_order_type", None)
        pos.pop("_native_stop_price", None)
