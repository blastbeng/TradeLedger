"""Exit order management component for the TradingEngine.

Handles native stop-loss, take-profit, and OCO order placement, cancellation,
replacement, and fill processing.
Extracted from OrderExecutor to reduce class size and improve maintainability.
"""
import asyncio
import logging
import time
from dataclasses import asdict
from typing import Any, Dict, Optional

from src.strategies.base import Signal
from src.utils.symbol_utils import is_btp_isin
from src.trading.engine_utils import format_symbol_display

logger = logging.getLogger(__name__)


class ExitOrderManager:
    """Handles native exit order management for the TradingEngine."""

    def __init__(self, engine, event_bus):
        self.engine = engine
        self.shared_state = engine.shared_state
        self.event_bus = event_bus
        self.event_bus.subscribe("cancel_exit_orders", self.cancel_exit_orders)
        self.event_bus.subscribe("place_exit_orders", self.place_exit_orders)
        self.event_bus.subscribe("replace_native_stop_order", self.replace_native_stop_order)
        self.event_bus.subscribe("process_native_exit_fill", self.process_native_exit_fill)
        self.event_bus.subscribe("check_and_cancel_oco_on_stop_trigger", self.check_and_cancel_oco_on_stop_trigger)

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
        if sl_ot in ("stop", "stop_limit"):
            sl_price = signal.stop_loss_stop_price
            if sl_price is None and stop_loss_pct is not None:
                sl_price = entry_price * (1 - stop_loss_pct)
        elif sl_ot == "trailing_stop":
            sl_price = None  # not a fixed price
        else:
            # Fallback to standard stop loss if order type not specified
            sl_price = entry_price * (1 - stop_loss_pct) if stop_loss_pct is not None else None

        # --- Take-profit price ---
        tp_ot = signal.take_profit_order_type
        if tp_ot == "limit":
            tp_price = signal.take_profit_limit_price
            if tp_price is None and take_profit_pct is not None:
                tp_price = entry_price * (1 + take_profit_pct)
        elif tp_ot == "market":
            tp_price = None  # will be handled by risk loop later
        else:
            # Fallback to standard take profit if order type not specified
            tp_price = entry_price * (1 + take_profit_pct) if take_profit_pct is not None else None

        return {
            "stop_loss_price": sl_price,
            "take_profit_price": tp_price,
        }

    async def cancel_exit_orders(self, symbol: str):
        """Cancel any native stop-loss and take-profit orders for a symbol."""
        engine = self.engine
        pos = self.shared_state.positions.get(symbol)
        if not pos:
            return
        for order_id_key in ("stop_loss_order_id", "take_profit_order_id"):
            order_id = pos.pop(order_id_key, None)
            if order_id:
                try:
                    await asyncio.to_thread(engine.trader.cancel_order, order_id)
                    logger.info(f"Cancelled exit order {order_id} for {symbol}")
                except (RuntimeError, ValueError, ConnectionError) as e:
                    logger.warning(f"Failed to cancel exit order {order_id}: {type(e).__name__}: {e}")
                async with self.shared_state._queued_orders_lock:
                    self.shared_state.queued_orders = [
                        q for q in self.shared_state.queued_orders
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
        pos = self.shared_state.positions.get(symbol)
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
                except (RuntimeError, ValueError, ConnectionError) as e:
                    logger.error(
                        f"Failed to cancel old exit order {old_id} for {symbol}: "
                        f"{type(e).__name__}: {e}. Aborting new exit order placement to avoid duplicates."
                    )
                    return
                # Remove from queued_orders
                async with self.shared_state._queued_orders_lock:
                    self.shared_state.queued_orders = [
                        q for q in self.shared_state.queued_orders
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
                async with self.shared_state._queued_orders_lock:
                    self.shared_state.queued_orders.append(_sl_queued)
            except (RuntimeError, ValueError, ConnectionError) as e:
                logger.error(f"Failed to place stop-loss order for {symbol}: {type(e).__name__}: {e}")

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
                        async with self.shared_state._queued_orders_lock:
                            self.shared_state.queued_orders.append(_trail_queued)
                    except (RuntimeError, ValueError, ConnectionError) as e:
                        logger.error(f"Failed to place trailing-stop order for {symbol}: {type(e).__name__}: {e}")

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
                async with self.shared_state._queued_orders_lock:
                    self.shared_state.queued_orders.append(_tp_queued)
            except (RuntimeError, ValueError, ConnectionError) as e:
                logger.error(f"Failed to place take-profit order for {symbol}: {type(e).__name__}: {e}")

        # --- Link OCO pair ---
        if sl_order_id and tp_order_id:
            for q in self.shared_state.queued_orders:
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
            stock_name = await engine._market_data_manager.get_stock_name(symbol)
            display_symbol = format_symbol_display(symbol, stock_name, pos.get("timeframe"))
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

    async def check_and_cancel_oco_on_stop_trigger(
        self,
        queued: Dict[str, Any],
    ) -> bool:
        """Check if a native stop/stop_limit exit order's stop price has been reached.

        If so, cancels the OCO take-profit pair immediately (instead of waiting
        for the queued-order polling loop). Includes a race-condition guard to
        prevent double-sells when the OCO pair has already filled.

        Returns True if the OCO pair was cancelled or already filled (caller
        should continue to the next order), False if no action was taken.
        """
        engine = self.engine

        if not (queued.get("is_exit_order")
                and queued.get("order_type") in ("stop", "stop_limit")
                and queued.get("side") == "sell"
                and queued.get("oco_pair") is not None):
            return False

        stop_price = queued.get("stop_price")
        if stop_price is None:
            return False

        # Fetch current price
        try:
            base = queued["symbol"].split("/")[0]
            quotes = await engine._market_data_manager._get_quotes_async([base], timeout=45.0)
            ticker = quotes.get(base)
        except (KeyError, RuntimeError, ConnectionError, ValueError):
            return False
        if not ticker or ticker.get("last") is None:
            return False

        current_price = ticker["last"]
        if current_price > stop_price:
            return False  # stop price not yet reached

        # Stop triggered – cancel OCO pair immediately
        oco_pair_id = queued["oco_pair"]

        # --- Race condition guard: check if the OCO pair (take-profit) has
        # already filled before cancelling. If it has, we must NOT cancel it;
        # the fill-detection code below will process the fill. ---
        oco_already_filled = False
        try:
            oco_order_obj = await asyncio.to_thread(
                engine.trader.get_order, oco_pair_id
            )
            if oco_order_obj is not None and oco_order_obj.status == "filled":
                oco_already_filled = True
        except (RuntimeError, ValueError, AttributeError):
            pass

        if oco_already_filled:
            logger.info(
                f"OCO pair {oco_pair_id} already filled for "
                f"{queued['symbol']}; skipping cancel to avoid double-sell."
            )
            queued["oco_pair"] = None
            pos = self.shared_state.positions.get(queued["symbol"])
            if pos:
                pos.pop("take_profit_order_id", None)
        else:
            try:
                await asyncio.to_thread(engine.trader.cancel_order, oco_pair_id)
                logger.info(
                    f"Stop triggered for {queued['symbol']} at {current_price:.4f}, "
                    f"cancelled OCO pair {oco_pair_id}"
                )
            except (RuntimeError, ValueError, ConnectionError) as e:
                logger.warning(f"Failed to cancel OCO order {oco_pair_id}: {type(e).__name__}: {e}")
            # Remove the cancelled take-profit from queued_orders (with lock)
            async with self.shared_state._queued_orders_lock:
                self.shared_state.queued_orders = [
                    q for q in self.shared_state.queued_orders
                    if q.get("order_id") != oco_pair_id
                ]
            # Clear OCO reference so we don't try again
            queued["oco_pair"] = None
            # Clear take-profit order ID from position
            pos = self.shared_state.positions.get(queued["symbol"])
            if pos:
                pos.pop("take_profit_order_id", None)
            # Notify user
            stock_name = await engine._market_data_manager.get_stock_name(queued["symbol"])
            if engine.notifier:
                display_symbol = format_symbol_display(
                    queued["symbol"], stock_name, queued.get("timeframe")
                )
                await engine.notifier.send_notification(
                    f"🛑 Stop triggered for {display_symbol} at {current_price:.4f}, "
                    f"take‑profit order cancelled.",
                    summary={
                        "symbol": queued["symbol"],
                        "action": "CANCEL",
                        "reason": "Stop triggered, OCO pair cancelled",
                    }
                )
        self.shared_state._state_dirty = True
        return True

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
        async with self.shared_state._queued_orders_lock:
            old_queued = next(
                (q for q in self.shared_state.queued_orders if q.get("order_id") == old_order_id),
                None
            )
            old_limit_price = old_queued.get("limit_price") if old_queued else None

        # Place a new stop order
        qty = pos["amount"]
        sl_ot = pos.get("stop_loss_order_type", "stop")
        new_order_id = None
        max_retries = 3
        for attempt in range(1, max_retries + 1):
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
                break
            except (RuntimeError, ValueError, ConnectionError, KeyError) as e:
                logger.error(
                    f"Failed to place replacement stop order for {symbol} (attempt {attempt}/{max_retries}): {type(e).__name__}: {e}."
                )
                if attempt == max_retries:
                    if engine.notifier:
                        stock_name = await engine._market_data_manager.get_stock_name(symbol)
                        display_symbol = format_symbol_display(symbol, stock_name, pos.get("timeframe"))
                        await engine.notifier.send_notification(
                            f"🚨 CRITICAL: Stop order replacement failed for {display_symbol} after {max_retries} attempts. Old stop order kept active at {old_stop_price:.4f}.",
                            summary={
                                "symbol": symbol,
                                "action": "ERROR",
                                "reason": f"Stop order replacement failed after retries, old order kept: {str(e)[:200]}",
                            }
                        )
                    return
                await asyncio.sleep(1.0 * attempt)

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
        async with self.shared_state._queued_orders_lock:
            self.shared_state.queued_orders.append(_replace_queued)
            # Update OCO link on the take-profit order if it exists
            tp_order_id = pos.get("take_profit_order_id")
            if tp_order_id:
                for q in self.shared_state.queued_orders:
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
        except (RuntimeError, ValueError, ConnectionError) as e:
            logger.warning(f"Failed to cancel old stop order {old_order_id} (new order {new_order_id} already placed): {type(e).__name__}: {e}")

        # Remove the old queued entry now that the new one is active
        async with self.shared_state._queued_orders_lock:
            self.shared_state.queued_orders = [
                q for q in self.shared_state.queued_orders
                if q.get("order_id") != old_order_id
            ]

        # Notify user
        if engine.notifier:
            stock_name = await engine._market_data_manager.get_stock_name(symbol)
            display_symbol = format_symbol_display(symbol, stock_name, pos.get("timeframe"))
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
        async with self.shared_state._queued_orders_lock:
            queued = next((q for q in self.shared_state.queued_orders if q.get("order_id") == order_id), None)
            if queued:
                self.shared_state.queued_orders = [q for q in self.shared_state.queued_orders if q.get("order_id") != order_id]

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
                await self.event_bus.request("handle_queued_sell_fill", trade_dict, queued, partial=False)

        # Cancel the OCO pair if it still exists
        oco_pair_id = queued.get("oco_pair") if queued else None
        if oco_pair_id:
            try:
                await asyncio.to_thread(engine.trader.cancel_order, oco_pair_id)
            except (RuntimeError, ValueError, ConnectionError) as e:
                logger.debug(f"process_native_exit_fill: failed to cancel OCO pair {oco_pair_id} for {symbol}: {type(e).__name__}: {e}")
            async with self.shared_state._queued_orders_lock:
                self.shared_state.queued_orders = [q for q in self.shared_state.queued_orders if q.get("order_id") != oco_pair_id]
        pos.pop("stop_loss_order_id", None)
        pos.pop("take_profit_order_id", None)
        pos.pop("stop_loss_order_type", None)
        pos.pop("_native_stop_price", None)
