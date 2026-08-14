"""Order execution component for the TradingEngine.

Handles order creation, fill processing, and exit order management.
Extracted from TradingEngine to reduce class size and improve maintainability.
"""
import asyncio
import logging
import time
from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple

from src.config.settings import settings
from src.database import insert_trade
from src.strategies.base import Signal
from src.utils.symbol_utils import is_btp_isin
from src.trading.engine_utils import format_symbol_display, timeframe_to_seconds
from src.trading.components.order_executor_base import OrderExecutorBase

logger = logging.getLogger(__name__)


class OrderExecutor(OrderExecutorBase):
    """Handles order execution and fill processing for the TradingEngine."""

    def __init__(self, engine, event_bus):
        super().__init__(engine, event_bus)
        self._exit_order_manager = None
        self._buy_executor = None
        from src.trading.components.sell_executor import SellExecutor
        self._sell_executor = SellExecutor(self.engine, self.event_bus, self)
        self.event_bus.subscribe("execute_signal", self.execute_signal)
        self.event_bus.subscribe("sweep_dust", self._sell_executor.sweep_dust)
        self.event_bus.subscribe("execute_partial_tp_single", self._sell_executor.execute_partial_tp_single)
        self.event_bus.subscribe("execute_partial_tp_level", self._sell_executor.execute_partial_tp_level)
        self.event_bus.subscribe("sell_all_positions", self.sell_all_positions)
        self.event_bus.subscribe("sell_position", self.sell_position)
        self.event_bus.subscribe("process_single_queued_order", self.process_single_queued_order)
        self.event_bus.subscribe("cancel_orphaned_orders", self.cancel_orphaned_orders)
        self.event_bus.subscribe("place_replacement_exit_orders_with_retry", self._place_replacement_exit_orders_with_retry)

    async def execute_signal(
        self,
        symbol: str,
        signal,
        timeframe: str = None,
        exit_reason: str = None,
        atr: Optional[float] = None,
    ):
        """Execute a BUY or SELL signal."""
        engine = self.engine
        # --- Format symbol for notifications ---
        stock_name = await engine._market_data_manager.get_stock_name(symbol)
        tf = timeframe or (self.shared_state.positions.get(symbol, {}).get("timeframe") if symbol in self.shared_state.positions else None)
        display_symbol = format_symbol_display(symbol, stock_name, tf)

        # --- Notify mode: do not execute any orders, only send notifications ---
        if settings.TRADING_MODE == "notify":
            logger.info(f"Notify mode: skipping order execution for {signal.action} {symbol}.")
            return

        # --- Paper mode + Paused: do not execute automated BUY orders, only send notifications ---
        # Manual overrides (exit_reason starts with "manual") are still allowed.
        # Automated SELL orders are allowed if the market is open (to manage open positions).
        paused = await asyncio.to_thread(engine.redis.get, "trading:paused")
        if settings.TRADING_MODE == "paper" and paused and not (exit_reason and exit_reason.startswith("manual")):
            is_market_open = await engine._is_market_open()
            if signal.action == "SELL" and is_market_open:
                logger.info(f"Paper mode + Paused: allowing automated SELL for risk management {symbol}.")
                # Fall through to execute the SELL order
            else:
                logger.info(f"Paper mode + Paused: skipping automated order execution for {signal.action} {symbol}.")
                return

        # Prevent executing new signals if an order is already queued for this symbol
        # (unless it's a manual override or an automated SELL from risk management)
        queued_to_cancel = []
        async with self.shared_state._queued_orders_lock:
            has_queued = any(q['symbol'] == symbol for q in self.shared_state.queued_orders)
            if has_queued:
                is_manual = bool(exit_reason and exit_reason.startswith("manual"))
                is_risk_sell = signal.action == "SELL" and exit_reason is not None
                
                if not (is_manual or is_risk_sell):
                    logger.info(f"Skipping {signal.action} for {symbol}: order already queued.")
                    return
                
                # If proceeding with a SELL, cancel any pending BUY order for this symbol
                if signal.action == "SELL":
                    queued_to_cancel = [q for q in self.shared_state.queued_orders if q['symbol'] == symbol and q['side'] == 'buy']
                    # If this is a manual sell, also cancel any queued SELL order for this symbol to avoid duplicate sells
                    if is_manual:
                        queued_to_cancel.extend([q for q in self.shared_state.queued_orders if q['symbol'] == symbol and q['side'] == 'sell'])
                    
                    # Remove the orders to cancel from the queue immediately to prevent race conditions
                    for q in queued_to_cancel:
                        if q in self.shared_state.queued_orders:
                            self.shared_state.queued_orders.remove(q)
                    self.shared_state._state_dirty = True

        if queued_to_cancel:
            logger.info(f"Cancelling queued orders for {symbol} to execute SELL ({exit_reason}).")
            for q in queued_to_cancel:
                order_id = q.get('order_id')
                try:
                    await asyncio.to_thread(engine.trader.cancel_order, order_id)
                except (RuntimeError, ValueError, ConnectionError, KeyError, TypeError, AttributeError) as e:
                    logger.warning(f"Failed to cancel queued order {order_id} for {symbol}: {type(e).__name__}: {e}")
                # Refund remaining reserved capital for buy orders
                if q['side'] == 'buy':
                    async with self.shared_state._cycle_spent_lock:
                        self.shared_state._cycle_spent = max(0.0, self.shared_state._cycle_spent - q.get('amount', 0.0))

        if signal.action == "SELL":
            async with self.shared_state._positions_lock:
                pos = self.shared_state.positions.get(symbol)
                if pos and pos.get("_selling"):
                    logger.info(f"Skipping SELL for {symbol}: already selling.")
                    return
                if pos:
                    pos["_selling"] = True

        # In live mode, only execute during regular market hours (manual overrides are allowed anytime)
        if not await engine._is_market_open() and not (exit_reason and exit_reason.startswith("manual")):
            logger.info(f"Skipping {signal.action} for {symbol}: market closed (live mode).")
            if engine.notifier:
                await engine.notifier.send_notification(
                    f"⏸️ Skipping {signal.action} for {display_symbol}: market closed.",
                    summary={"symbol": symbol, "action": "SKIP", "reason": "Market closed"}
                )
            return

        parts = symbol.split("/")
        if len(parts) != 2:
            logger.error(f"Invalid symbol format: {symbol}")
            return
        base, quote = parts
        _exec_base = symbol.split("/")[0]
        _exec_is_btp = is_btp_isin(symbol)
        balance = await engine._get_cached_balance()

        if signal.action == "BUY":
            await self.event_bus.request(
                "execute_buy",
                symbol=symbol,
                display_symbol=display_symbol,
                signal=signal,
                timeframe=timeframe,
                exit_reason=exit_reason,
                atr=atr,
                balance=balance,
            )
        elif signal.action == "SELL":
            await self.event_bus.request(
                "execute_sell",
                symbol=symbol,
                display_symbol=display_symbol,
                signal=signal,
                timeframe=timeframe,
                exit_reason=exit_reason,
                atr=atr,
                balance=balance,
            )

    async def sell_all_positions(self):
        """Sell all open positions at market price."""
        engine = self.engine
        if not await engine._is_market_open():
            logger.warning("Sell all positions skipped: market is closed.")
            if engine.notifier:
                await engine.notifier.send_notification(
                    "⏸️ Sell all skipped: market is currently closed.",
                    summary={"action": "SKIP", "reason": "Market closed"}
                )
            return
        for symbol in list(self.shared_state.positions.keys()):
            await self.execute_signal(
                symbol,
                Signal(action="SELL", confidence=1.0, reasoning="Manual sell all"),
                exit_reason="manual_sell_all"
            )

    async def sell_position(self, symbol: str):
        """Sell a specific open position at market price."""
        engine = self.engine
        if not await engine._is_market_open():
            logger.warning(f"Sell position {symbol} skipped: market is closed.")
            if engine.notifier:
                await engine.notifier.send_notification(
                    f"⏸️ Sell {symbol} skipped: market is currently closed.",
                    summary={"symbol": symbol, "action": "SKIP", "reason": "Market closed"}
                )
            return
        if symbol in self.shared_state.positions:
            await self.execute_signal(
                symbol,
                Signal(action="SELL", confidence=1.0, reasoning="Manual sell"),
                exit_reason="manual_sell"
            )
        else:
            logger.warning(f"No open position for {symbol}")

    async def _place_replacement_exit_orders_with_retry(
        self, symbol: str, signal: Signal, exit_prices: Dict[str, Optional[float]], 
        timeframe: Optional[str], max_retries: int = 3, delay: float = 1.0
    ) -> None:
        """Place replacement exit orders with a retry mechanism to avoid leaving positions unprotected."""
        for attempt in range(max_retries):
            try:
                await self._exit_order_manager.place_exit_orders(symbol, signal, exit_prices, timeframe)
                return
            except (RuntimeError, ValueError, ConnectionError, KeyError, TypeError, AttributeError) as e:
                logger.warning(
                    f"Failed to place replacement exit orders for {symbol} "
                    f"(attempt {attempt + 1}/{max_retries}): {type(e).__name__}: {e}"
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(delay)

        logger.error(
            f"Failed to place replacement exit orders for {symbol} after {max_retries} attempts. "
            f"Position remains unprotected until next risk cycle."
        )
        if self.engine.notifier:
            stock_name = await self.engine._market_data_manager.get_stock_name(symbol)
            display_symbol = format_symbol_display(symbol, stock_name, timeframe)
            await self.engine.notifier.send_notification(
                f"⚠️ Failed to place replacement exit orders for {display_symbol} after {max_retries} attempts. Position unprotected!",
                summary={"symbol": symbol, "action": "ERROR", "reason": "Replacement exit orders failed"}
            )

    async def process_queued_order_fill(
        self,
        queued: Dict[str, Any],
        paper_order: Any,
        order_id: str,
        filled_qty: float,
        filled_avg_price: float,
    ) -> None:
        """Process a new fill (partial or final) for a queued order."""
        engine = self.engine
        delta_qty = filled_qty - queued.get('filled_qty', 0.0)
        if delta_qty <= 0:
            return

        delta_cost = delta_qty * filled_avg_price
        from src.exchanges.fees import calculate_transaction_costs
        _quote_ccy = queued['symbol'].split("/")[1] if "/" in queued['symbol'] else engine.base_currency
        _fee_costs = calculate_transaction_costs(
            queued['side'].upper(), filled_avg_price, delta_qty, symbol=queued['symbol']
        )
        trade_dict = {
            'id': str(paper_order.id),
            'symbol': queued['symbol'],
            'side': queued['side'],
            'amount': delta_qty,
            'price': filled_avg_price,
            'cost': delta_cost,
            'fee': {'cost': _fee_costs["total_costs"], 'currency': _quote_ccy},
            'status': 'closed',
            'timestamp': int(time.time() * 1000),
        }
        # Update tracking fields
        queued['filled_qty'] = filled_qty
        queued['filled_cost'] = queued.get('filled_cost', 0.0) + delta_cost

        if queued['side'] == 'buy':
            # Update remaining quote amount
            original_amount = queued.get('original_amount', queued['amount'])
            queued['amount'] = original_amount - queued['filled_cost']
            await self.handle_queued_buy_fill(trade_dict, queued)
        else:
            # Update remaining base amount
            original_amount = queued.get('original_amount', queued['amount'])
            queued['amount'] = original_amount - filled_qty
            await self.event_bus.request("handle_queued_sell_fill", trade_dict, queued, partial=True)

        # --- OCO handling for exit orders ---
        if queued.get("is_exit_order"):
            oco_pair_id = queued.get("oco_pair")
            if oco_pair_id:
                try:
                    await asyncio.to_thread(engine.trader.cancel_order, oco_pair_id)
                    logger.info(f"Cancelled OCO pair {oco_pair_id} for {queued['symbol']}")
                except (RuntimeError, ValueError, ConnectionError, KeyError, TypeError, AttributeError) as e:
                    logger.warning(f"Failed to cancel OCO order {oco_pair_id}: {type(e).__name__}: {e}")
                async with self.shared_state._queued_orders_lock:
                    self.shared_state.queued_orders = [
                        q for q in self.shared_state.queued_orders
                        if q.get("order_id") != oco_pair_id
                    ]

            # If the fill was partial, cancel the remaining part of this exit order
            # to avoid leaving a dangling order that is no longer linked to the position.
            # The risk management loop will handle the remaining position.
            if filled_qty < queued.get('original_amount', queued['amount']):
                try:
                    await asyncio.to_thread(engine.trader.cancel_order, order_id)
                    logger.info(f"Cancelled remaining part of partially filled exit order {order_id} for {queued['symbol']}")
                except (RuntimeError, ValueError, ConnectionError, KeyError, TypeError, AttributeError) as e:
                    logger.warning(f"Failed to cancel remaining part of exit order {order_id}: {e}")
                async with self.shared_state._queued_orders_lock:
                    self.shared_state.queued_orders = [
                        q for q in self.shared_state.queued_orders
                        if q.get("order_id") != order_id
                    ]

            if queued["symbol"] in self.shared_state.positions:
                pos = self.shared_state.positions[queued["symbol"]]
                pos.pop("stop_loss_order_id", None)
                pos.pop("take_profit_order_id", None)
                # Place replacement exit orders for the remaining position to avoid
                # a protection gap until the next risk-management loop tick.
                if pos.get("amount", 0) > 0 and pos.get("stop_loss") is not None:
                    from src.strategies.base import Signal as _Signal
                    _dummy_params = {
                        "trailing_take_profit": pos.get("trailing_take_profit", False),
                        "partial_take_profit_levels": pos.get("partial_take_profit_levels"),
                        "partial_take_profit_pct": pos.get("partial_take_profit_pct"),
                    }
                    _dummy_signal = _Signal(
                        action="BUY",
                        confidence=1.0,
                        reasoning="Replacing exit orders after partial exit-order fill",
                        stop_loss_order_type=pos.get("stop_loss_order_type"),
                        stop_loss_stop_price=pos.get("stop_loss"),
                        stop_loss_limit_price=None,
                        take_profit_order_type=pos.get("take_profit_order_type"),
                        take_profit_limit_price=pos.get("take_profit"),
                        strategy_params=_dummy_params,
                    )
                    _exit_prices = {
                        "stop_loss_price": pos.get("stop_loss"),
                        "take_profit_price": pos.get("take_profit"),
                    }
                    # Re-verify the position still exists and has amount to avoid
                    # orphaned orders if a concurrent risk check removed it.
                    if queued["symbol"] not in self.shared_state.positions or self.shared_state.positions[queued["symbol"]].get("amount", 0) <= 0:
                        logger.info(f"Position for {queued['symbol']} removed concurrently, skipping replacement exit orders.")
                    else:
                        await self.event_bus.request(
                            "place_replacement_exit_orders_with_retry",
                            queued["symbol"], _dummy_signal, _exit_prices, pos.get("timeframe")
                        )
            # Notify user
            if engine.notifier:
                stock_name = await self.engine._market_data_manager.get_stock_name(queued["symbol"])
                display_symbol = format_symbol_display(queued["symbol"], stock_name, queued.get("timeframe"))
                await engine.notifier.send_notification(
                    f"🔗 OCO pair {oco_pair_id} cancelled for {display_symbol} (other order filled).",
                    summary={
                        "symbol": queued["symbol"],
                        "action": "CANCEL",
                        "reason": "OCO pair cancelled",
                    }
                )

        # Update open positions flag in Redis for LLM model selection
        try:
            if self.shared_state.positions:
                await asyncio.to_thread(self.engine.redis.set, "trading:has_open_positions", "1")
            else:
                await asyncio.to_thread(self.engine.redis.delete, "trading:has_open_positions")
        except Exception:
            pass

    async def process_single_queued_order(self, queued: Dict[str, Any]) -> None:
        """Process a single queued order: check timeouts, fetch status, handle fills/cancellations."""
        engine = self.engine
        order_id = queued.get('order_id')
        if not order_id:
            logger.warning(f"Queued order for {queued['symbol']} missing order_id, removing.")
            async with self.shared_state._queued_orders_lock:
                if queued in self.shared_state.queued_orders:
                    self.shared_state.queued_orders.remove(queued)
            return

        if await self.check_and_cancel_timed_out_order(queued):
            return

        paper_order = await asyncio.to_thread(engine.trader.get_order, order_id)
        if paper_order is None:
            await self.handle_missing_order(queued)
            return

        status = paper_order.status
        if isinstance(status, str):
            status = status.lower()

        if await self.event_bus.request("check_and_cancel_oco_on_stop_trigger", queued):
            pass  # OCO handled, continue processing this order for fill detection

        filled_qty = float(paper_order.filled_qty) if paper_order.filled_qty else 0.0
        filled_avg_price = float(paper_order.filled_avg_price) if paper_order.filled_avg_price else 0.0
        last_filled_qty = queued.get('filled_qty', 0.0)
        delta_qty = filled_qty - last_filled_qty

        if delta_qty > 0:
            await self.process_queued_order_fill(
                queued=queued,
                paper_order=paper_order,
                order_id=order_id,
                filled_qty=filled_qty,
                filled_avg_price=filled_avg_price,
            )

        if status == 'filled':
            logger.info(f"Queued limit order {order_id} for {queued['symbol']} completely filled.")
            # Safety: refund any remaining amount for buy orders (handles rounding edge cases
            # where filled_cost doesn't exactly equal original_amount due to slippage/fees)
            if queued['side'] == 'buy' and queued.get('amount', 0) > 0:
                async with self.shared_state._cycle_spent_lock:
                    self.shared_state._cycle_spent = max(0.0, self.shared_state._cycle_spent - queued['amount'])
                logger.debug(f"Refunded remaining {queued['amount']:.2f} to _cycle_spent for filled buy order {order_id}")
            async with self.shared_state._queued_orders_lock:
                if queued in self.shared_state.queued_orders:
                    self.shared_state.queued_orders.remove(queued)
            self.shared_state._state_dirty = True
        elif status in ('rejected', 'canceled', 'cancelled', 'expired'):
            await self.handle_canceled_or_rejected_order(queued, status)

    async def cancel_orphaned_orders(self):
        """Periodically cancel any open orders that are older than 10 minutes,
        but never cancel orders that are still being tracked as queued."""
        engine = self.engine
        open_orders = await asyncio.to_thread(engine.trader.get_open_orders)
        now = time.time()
        # Build a set of order IDs that are currently queued (waiting for fill)
        queued_ids = {q.get('order_id') for q in self.shared_state.queued_orders if q.get('order_id')}
        for order in open_orders:
            order_id = order.get('id')
            if order_id in queued_ids:
                continue   # this order is being monitored by _process_queued_orders
            created_at = order.get('timestamp', 0) / 1000.0  # ms to seconds
            if now - created_at > settings.ORPHANED_ORDER_TIMEOUT_SECONDS:
                logger.warning(
                    f"Cancelling orphaned order {order_id} for {order['symbol']} "
                    f"(open for {now - created_at:.0f}s)."
                )
                await asyncio.to_thread(engine.trader.cancel_order, order_id)

    async def check_and_cancel_timed_out_order(
        self,
        queued: Dict[str, Any],
    ) -> bool:
        """Check if a queued order has timed out and cancel it if so.

        Returns True if the order was timed out and cancelled (caller should
        continue to the next order), False if the order is still within its
        timeout window or is an exit order (exempt from timeout).
        """
        engine = self.engine
        if queued.get("is_exit_order"):
            return False

        queued_at = queued.get('queued_at', 0)
        queued_tf = queued.get('timeframe')
        base_timeout = settings.QUEUED_ORDER_TIMEOUT_SECONDS
        if queued_tf:
            tf_secs = timeframe_to_seconds(queued_tf)
            # Scale timeout by timeframe, but cap at 30 days to avoid excessively long waits
            scaled_timeout = min(max(base_timeout, int(tf_secs * 0.5)), 30 * 24 * 3600)
        else:
            scaled_timeout = base_timeout

        if time.time() - queued_at <= scaled_timeout:
            return False

        order_id = queued.get('order_id')
        logger.warning(
            f"Queued order {order_id} for {queued['symbol']} timed out "
            f"after {scaled_timeout}s. Cancelling."
        )
        try:
            await asyncio.to_thread(engine.trader.cancel_order, order_id)
        except (RuntimeError, ValueError, ConnectionError, KeyError, TypeError, AttributeError) as e:
            logger.error(f"Failed to cancel timed-out order {order_id}: {type(e).__name__}: {e}")

        # Refund remaining reserved capital for buy orders
        if queued['side'] == 'buy':
            async with self.shared_state._cycle_spent_lock:
                self.shared_state._cycle_spent = max(0.0, self.shared_state._cycle_spent - queued.get('amount', 0.0))
        else:
            async with self.shared_state._positions_lock:
                pos = self.shared_state.positions.get(queued["symbol"])
                if pos:
                    pos.pop("_selling", None)

        # Remove from queue regardless of cancel success
        async with self.shared_state._queued_orders_lock:
            if queued in self.shared_state.queued_orders:
                self.shared_state.queued_orders.remove(queued)
        self.shared_state._state_dirty = True

        if engine.notifier:
            stock_name = await engine._market_data_manager.get_stock_name(queued['symbol'])
            tf = queued.get('timeframe')
            display = format_symbol_display(queued['symbol'], stock_name, tf)
            await engine.notifier.send_notification(
                f"⏰ Queued {queued['side']} order for {display} timed out and was cancelled.",
                summary={
                    "symbol": queued['symbol'],
                    "action": "CANCEL",
                    "reason": "Queued order timeout",
                }
            )
        return True

    async def handle_missing_order(self, queued: Dict[str, Any]) -> None:
        """Handle a queued order that is no longer found in the simulator."""
        engine = self.engine
        order_id = queued.get('order_id')
        logger.warning(f"Order {order_id} not found for {queued['symbol']}, removing from queue.")
        # Refund remaining reserved capital for buy orders
        if queued['side'] == 'buy':
            async with self.shared_state._cycle_spent_lock:
                self.shared_state._cycle_spent = max(0.0, self.shared_state._cycle_spent - queued.get('amount', 0.0))
        else:
            async with self.shared_state._positions_lock:
                pos = self.shared_state.positions.get(queued["symbol"])
                if pos:
                    pos.pop("_selling", None)
        async with self.shared_state._queued_orders_lock:
            if queued in self.shared_state.queued_orders:
                self.shared_state.queued_orders.remove(queued)
        self.shared_state._state_dirty = True

    async def handle_canceled_or_rejected_order(
        self,
        queued: Dict[str, Any],
        status: str,
    ) -> None:
        """Handle cleanup for a queued order that ended as rejected, canceled, or expired."""
        engine = self.engine
        order_id = queued.get('order_id')
        logger.warning(
            f"Queued order {order_id} for {queued['symbol']} ended as {status}, removing."
        )
        # Refund remaining reserved capital for buy orders
        if queued['side'] == 'buy':
            async with self.shared_state._cycle_spent_lock:
                self.shared_state._cycle_spent = max(0.0, self.shared_state._cycle_spent - queued.get('amount', 0.0))
        else:
            async with self.shared_state._positions_lock:
                pos = self.shared_state.positions.get(queued["symbol"])
                if pos:
                    pos.pop("_selling", None)
        if engine.notifier:
            stock_name = await engine._market_data_manager.get_stock_name(queued['symbol'])
            tf = queued.get('timeframe')
            display = format_symbol_display(queued['symbol'], stock_name, tf)
            await engine.notifier.send_notification(
                f"❌ Queued {queued['side']} order for {display} {status}.",
                summary={
                    "symbol": queued['symbol'],
                    "action": "INFO",
                    "reason": f"Order {status}",
                }
            )
        async with self.shared_state._queued_orders_lock:
            if queued in self.shared_state.queued_orders:
                self.shared_state.queued_orders.remove(queued)
        self.shared_state._state_dirty = True
        if queued.get("is_exit_order"):
            oco_pair_id = queued.get("oco_pair")
            if oco_pair_id:
                try:
                    await asyncio.to_thread(engine.trader.cancel_order, oco_pair_id)
                    logger.info(f"Cancelled OCO pair {oco_pair_id} for {status} exit order {order_id}")
                except (RuntimeError, ValueError, ConnectionError, KeyError, TypeError, AttributeError) as e:
                    logger.warning(f"Failed to cancel OCO order {oco_pair_id}: {type(e).__name__}: {e}")
                async with self.shared_state._queued_orders_lock:
                    self.shared_state.queued_orders = [
                        q for q in self.shared_state.queued_orders
                        if q.get("order_id") != oco_pair_id
                    ]
            pos = self.shared_state.positions.get(queued["symbol"])
            if pos:
                pos.pop("stop_loss_order_id", None)
                pos.pop("take_profit_order_id", None)
            if engine.notifier:
                display_symbol = format_symbol_display(queued["symbol"], stock_name, queued.get("timeframe"))
                await engine.notifier.send_notification(
                    f"🔗 OCO pair {oco_pair_id} cancelled for {display_symbol} (main order {status}).",
                    summary={"symbol": queued["symbol"], "action": "CANCEL", "reason": f"OCO pair cancelled due to main order {status}"}
                )

    async def handle_queued_buy_fill(self, trade_dict: Dict[str, Any], queued: Dict[str, Any]):
        """Process a queued BUY limit order that has filled in the simulator."""
        engine = self.engine
        symbol = trade_dict['symbol']
        parts = symbol.split("/")
        if len(parts) != 2:
            logger.error(f"Invalid symbol format in queued buy fill: {symbol}")
            return
        base, quote = parts
        fee_cost, fee_currency = self._extract_fee(trade_dict)
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

        await self._buy_executor._apply_buy_to_position(
            symbol=symbol,
            cost_basis=cost_basis,
            net_base=net_base,
            timestamp=trade_dict["timestamp"],
            sl_pct=sl_pct,
            tp_pct=tp_pct,
            trailing_stop=trailing_stop,
            trailing_stop_distance_pct=trailing_stop_distance_pct,
            params=params,
            signal_confidence=signal_dict.get('confidence', 0.0),
            signal_reasoning=signal_dict.get('reasoning', ''),
            signal_strategy_type=signal_dict.get('strategy_type'),
            indicator_config=indicator_config,
            order_type=queued.get('order_type', 'market'),
            timeframe=timeframe,
        )

        trade_dict['strategy_type'] = signal_dict.get('strategy_type')
        trade_dict['timeframe'] = timeframe
        trade_dict['buy_confidence'] = signal_dict.get('confidence', 0.0)
        trade_dict['buy_reasoning'] = (signal_dict.get('reasoning', '') or '')[:200]
        self.shared_state.append_trade(trade_dict, settings.MAX_TRADES_IN_MEMORY)
        self.shared_state._balance_cache = None
        # Note: _cycle_spent was already updated when the order was queued
        # in _execute_signal, so we do NOT add to it here to avoid double-counting.
        await asyncio.to_thread(insert_trade, trade_dict)
        await self.event_bus.publish("save_state", force=True)
        self.shared_state._portfolio_exposure_cache = None
        if engine.notifier:
            stock_name = await engine._market_data_manager.get_stock_name(symbol)
            display_symbol = format_symbol_display(symbol, stock_name, timeframe)
            signal_obj = Signal.from_dict(signal_dict)
            await self._send_buy_notification(symbol, display_symbol, trade_dict, signal_obj, atr)

        # Place native exit orders for the new/updated position
        signal_dict = queued.get('signal', {}) or {}
        if signal_dict:
            try:
                # Reconstruct a Signal from the stored dict using the dedicated method
                reconstructed_signal = Signal.from_dict(signal_dict)
                exit_prices = self._exit_order_manager.compute_exit_order_prices(
                    entry_price=self.shared_state.positions[symbol]["price"],
                    signal=reconstructed_signal,
                    atr=queued.get('atr'),
                )
                await self.event_bus.request(
                    "place_replacement_exit_orders_with_retry",
                    symbol, reconstructed_signal, exit_prices, queued.get('timeframe')
                )
            except (RuntimeError, ValueError, ConnectionError, KeyError, TypeError, AttributeError) as e:
                logger.error(f"Failed to setup exit orders after queued buy fill for {symbol}: {type(e).__name__}: {e}")
                if engine.notifier:
                    stock_name = await engine._market_data_manager.get_stock_name(symbol)
                    display_symbol = format_symbol_display(symbol, stock_name, queued.get('timeframe'))
                    await engine.notifier.send_notification(
                        f"⚠️ Exit order setup failed for {display_symbol} after queued fill: {e}",
                        summary={
                            "symbol": symbol,
                            "action": "ERROR",
                            "reason": f"Exit order setup failed after queued fill: {str(e)[:200]}",
                        }
                    )



