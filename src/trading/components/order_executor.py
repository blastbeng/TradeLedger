"""Order execution component for the TradingEngine.

Handles order creation, fill processing, and exit order management.
Extracted from TradingEngine to reduce class size and improve maintainability.
"""
import asyncio
import logging
import time
from dataclasses import asdict
from typing import Any, Dict, Optional

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
                    # Fall back to a regular stop order when no explicit limit
                    # price is provided for stop_limit. Defaulting the limit to
                    # the stop price defeats the purpose of a stop-limit order.
                    if sl_ot == "stop_limit":
                        actual_sl_ot = "stop"
                    order = await asyncio.to_thread(
                        engine.trader.create_stop_sell_order,
                        symbol, qty, sl_price,
                        time_in_force="gtc", timeout=60.0
                    )
                else:  # stop_limit with explicit limit price
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
        # If trailing_take_profit or partial take-profit is enabled, do not place a
        # native limit order because the take-profit price will move or only a
        # fraction of the position should be sold. The risk management loop will
        # handle these cases instead.
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
        # Store order type for risk management decisions
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
