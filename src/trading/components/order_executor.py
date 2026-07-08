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

logger = logging.getLogger(__name__)


class OrderExecutor:
    """Handles order execution and fill processing for the TradingEngine."""

    def __init__(self, engine, event_bus):
        self.engine = engine
        self.event_bus = event_bus
        self._exit_order_manager = None
        self._buy_executor = None
        self.event_bus.subscribe("execute_signal", self.execute_signal)
        self.event_bus.subscribe("sweep_dust", self.sweep_dust)
        self.event_bus.subscribe("execute_partial_tp_single", self.execute_partial_tp_single)
        self.event_bus.subscribe("execute_partial_tp_level", self.execute_partial_tp_level)
        self.event_bus.subscribe("sell_all_positions", self.sell_all_positions)
        self.event_bus.subscribe("sell_position", self.sell_position)

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
        tf = timeframe or (engine.positions.get(symbol, {}).get("timeframe") if symbol in engine.positions else None)
        display_symbol = engine._format_symbol_display(symbol, stock_name, tf)

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
        # (unless it's a manual override)
        async with engine._queued_orders_lock:
            has_queued = any(q['symbol'] == symbol for q in engine.queued_orders)
        if has_queued and not (exit_reason and exit_reason.startswith("manual")):
            logger.info(f"Skipping {signal.action} for {symbol}: order already queued.")
            return

        # If this is a manual sell, cancel any queued SELL order for this symbol to avoid duplicate sells
        if exit_reason and exit_reason.startswith("manual") and signal.action == "SELL":
            async with engine._queued_orders_lock:
                engine.queued_orders = [q for q in engine.queued_orders if not (q['symbol'] == symbol and q['side'] == 'sell')]

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
            await self._buy_executor.execute_buy(
                symbol=symbol,
                display_symbol=display_symbol,
                signal=signal,
                timeframe=timeframe,
                exit_reason=exit_reason,
                atr=atr,
                balance=balance,
            )
        elif signal.action == "SELL":
            await self.execute_sell(
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
        for symbol in list(engine.positions.keys()):
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
        if symbol in engine.positions:
            await self.execute_signal(
                symbol,
                Signal(action="SELL", confidence=1.0, reasoning="Manual sell"),
                exit_reason="manual_sell"
            )
        else:
            logger.warning(f"No open position for {symbol}")

    @staticmethod
    def _default_limit_price(
        symbol: str, action: str, ticker: Dict[str, Any], atr: Optional[float] = None
    ) -> Optional[float]:
        """Compute a default aggressive limit price for extended‑hours trading.

        The buffer is scaled by ATR when available: buffer_pct = atr / price,
        clamped to [0.001, 0.02] (0.1%–2%). Falls back to 0.2% when ATR is
        unavailable.
        """
        last = ticker.get('last')
        if not last or last <= 0:
            return None

        # Compute buffer percentage from ATR, clamped to [0.1%, 2%]
        if atr is not None and atr > 0:
            buffer_pct = max(0.001, min(atr / last, 0.02))
        else:
            buffer_pct = 0.002  # fallback 0.2%

        if action == "BUY":
            limit = last * (1 + buffer_pct)
        elif action == "SELL":
            limit = last * (1 - buffer_pct)
        else:
            return None

        if last >= 1.0:
            limit = round(limit, 2)
        else:
            limit = round(limit, 4)
        return limit

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
                    f"(attempt {attempt + 1}/{max_retries}): {e}"
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(delay)

        logger.error(
            f"Failed to place replacement exit orders for {symbol} after {max_retries} attempts. "
            f"Position remains unprotected until next risk cycle."
        )
        if self.engine.notifier:
            stock_name = await self.engine._market_data_manager.get_stock_name(symbol)
            display_symbol = self.engine._format_symbol_display(symbol, stock_name, timeframe)
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
            await self.handle_queued_sell_fill(trade_dict, queued, partial=True)

        # --- OCO handling for exit orders ---
        if queued.get("is_exit_order"):
            oco_pair_id = queued.get("oco_pair")
            if oco_pair_id:
                try:
                    await asyncio.to_thread(engine.trader.cancel_order, oco_pair_id)
                    logger.info(f"Cancelled OCO pair {oco_pair_id} for {queued['symbol']}")
                except (RuntimeError, ValueError, ConnectionError) as e:
                    logger.warning(f"Failed to cancel OCO order {oco_pair_id}: {e}")
                async with engine._queued_orders_lock:
                    engine.queued_orders = [
                        q for q in engine.queued_orders
                        if q.get("order_id") != oco_pair_id
                    ]

            # If the fill was partial, cancel the remaining part of this exit order
            # to avoid leaving a dangling order that is no longer linked to the position.
            # The risk management loop will handle the remaining position.
            if filled_qty < queued.get('original_amount', queued['amount']):
                try:
                    await asyncio.to_thread(engine.trader.cancel_order, order_id)
                    logger.info(f"Cancelled remaining part of partially filled exit order {order_id} for {queued['symbol']}")
                except (RuntimeError, ValueError, ConnectionError) as e:
                    logger.warning(f"Failed to cancel remaining part of exit order {order_id}: {e}")
                async with engine._queued_orders_lock:
                    engine.queued_orders = [
                        q for q in engine.queued_orders
                        if q.get("order_id") != order_id
                    ]

            pos = engine.positions.get(queued["symbol"])
            if pos:
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
                    await self._place_replacement_exit_orders_with_retry(
                        queued["symbol"], _dummy_signal, _exit_prices, pos.get("timeframe")
                    )
            # Notify user
            if engine.notifier:
                stock_name = await self.engine._market_data_manager.get_stock_name(queued["symbol"])
                display_symbol = engine._format_symbol_display(queued["symbol"], stock_name, queued.get("timeframe"))
                await engine.notifier.send_notification(
                    f"🔗 OCO pair {oco_pair_id} cancelled for {display_symbol} (other order filled).",
                    summary={
                        "symbol": queued["symbol"],
                        "action": "CANCEL",
                        "reason": "OCO pair cancelled",
                    }
                )

    async def process_single_queued_order(self, queued: Dict[str, Any]) -> None:
        """Process a single queued order: check timeouts, fetch status, handle fills/cancellations."""
        engine = self.engine
        order_id = queued.get('order_id')
        if not order_id:
            logger.warning(f"Queued order for {queued['symbol']} missing order_id, removing.")
            async with engine._queued_orders_lock:
                if queued in engine.queued_orders:
                    engine.queued_orders.remove(queued)
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

        if await self._exit_order_manager.check_and_cancel_oco_on_stop_trigger(queued):
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
            async with engine._queued_orders_lock:
                if queued in engine.queued_orders:
                    engine.queued_orders.remove(queued)
            engine._state_dirty = True
        elif status in ('rejected', 'canceled', 'cancelled', 'expired'):
            await self.handle_canceled_or_rejected_order(queued, status)

        self,
        symbol: str,
        display_symbol: str,
        signal: Signal,
        timeframe: Optional[str],
        exit_reason: Optional[str],
        atr: Optional[float],
        balance: Dict[str, float],
    ) -> None:
        """Execute a SELL signal."""
        engine = self.engine
        base, quote = symbol.split("/")
        pos = engine.positions.get(symbol)

        # Cancel any native exit orders before selling
        if pos:
            await self._exit_order_manager.cancel_exit_orders(symbol)

        params = signal.strategy_params or {}
        fill_timeout = params.get("order_fill_timeout_seconds", settings.ORDER_FILL_TIMEOUT_SECONDS)

        # Determine the amount of base currency to sell
        if pos:
            gross_amount = pos["amount"]
        else:
            gross_amount = balance.get(base, 0.0)

        # Guard against overselling: cap sell amount to actual balance
        actual_base_balance = balance.get(base, 0.0)
        if pos and gross_amount > actual_base_balance:
            logger.warning(
                f"Tracked position amount {gross_amount} exceeds actual balance "
                f"{actual_base_balance} for {symbol}. Capping sell amount to actual balance."
            )
            gross_amount = actual_base_balance

        if gross_amount <= 0:
            logger.info(f"No {base} to sell for {symbol}")
            if engine.notifier:
                await engine.notifier.send_notification(
                    f"⚠️ No {base} to sell for {display_symbol}",
                    summary={
                        "symbol": symbol,
                        "action": "SKIP",
                        "reason": "No base balance to sell",
                    }
                )
            return

        # Check minimum sell size
        ticker = None
        try:
            quotes = await engine._market_data_manager._get_quotes_async([base], timeout=45.0)
            ticker = quotes.get(base)
            price = ticker['last']
            # --- Stale quote guard: skip SELL if the price is too old ---
            tf = timeframe or (pos.get("timeframe") if pos else None)
            if tf and await engine._is_quote_too_stale(ticker, tf):
                age_seconds = (time.time() * 1000 - ticker.get("last_update", 0)) / 1000
                logger.warning(
                    f"Skipping SELL {symbol}: quote is {age_seconds:.0f}s old "
                    f"(threshold scaled for timeframe {tf}). "
                    f"Stale prices lead to incorrect realized P&L and suboptimal exit prices."
                )
                if engine.notifier:
                    await engine.notifier.send_notification(
                        f"⚠️ Skipping SELL {display_symbol}: quote data is {age_seconds / 60:.0f} min old. "
                        f"Waiting for fresher data.",
                        summary={
                            "symbol": symbol,
                            "action": "SKIP",
                            "reason": "Stale quote data",
                            "age_seconds": round(age_seconds, 1),
                        }
                    )
                return
            # Fetch minimum order size from asset info
            try:
                asset = await engine._market_data_manager.get_asset_info(symbol)
                min_amount_limit = float(asset.min_order_size) if asset.min_order_size else None
                if not asset.fractionable and (min_amount_limit is None or min_amount_limit < 1.0):
                    min_amount_limit = 1.0
            except (AttributeError, TypeError, ValueError):
                min_amount_limit = None
            if min_amount_limit is not None and price:
                min_cost_limit = min_amount_limit * price
            else:
                min_cost_limit = None
            if min_amount_limit is not None and gross_amount < float(min_amount_limit):
                logger.info(f"SELL amount {gross_amount:.6f} {base} below min amount {min_amount_limit} for {symbol}, skipping")
                if engine.notifier:
                    await engine.notifier.send_notification(
                        f"⚠️ SELL skipped for {display_symbol}: amount too small",
                        summary={
                            "symbol": symbol,
                            "action": "SKIP",
                            "reason": "Sell amount below minimum",
                        }
                    )
                return
            if min_cost_limit is not None and gross_amount * price < float(min_cost_limit):
                logger.info(f"SELL cost {gross_amount * price:.2f} {quote} below min cost {min_cost_limit} for {symbol}, skipping")
                if engine.notifier:
                    await engine.notifier.send_notification(
                        f"⚠️ SELL skipped for {display_symbol}: cost too small",
                        summary={
                            "symbol": symbol,
                            "action": "SKIP",
                            "reason": "Sell cost below minimum",
                        }
                    )
                return
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
            logger.warning(f"Could not verify min sell size for {symbol}: {e}")

        need_limit = not await engine._is_market_open()
        limit_price = None
        time_in_force = "day"
        # If LLM provided a limit_price, use it even during regular hours
        llm_limit_price = params.get("limit_price")
        if llm_limit_price is not None and llm_limit_price > 0:
            limit_price = llm_limit_price
            time_in_force = params.get("time_in_force", "day")
            need_limit = True  # force limit order path
        elif need_limit:
            limit_price = self._default_limit_price(symbol, "SELL", ticker, atr=atr)
            time_in_force = params.get("time_in_force", "day")
            if limit_price is None:
                logger.error(f"Cannot place limit order for {symbol}: no limit price available.")
                return

        if limit_price is not None:
            # Round to valid tick size ($0.01 for >=$1, $0.0001 for <$1)
            if limit_price >= 1.0:
                limit_price = round(limit_price, 2)
            else:
                limit_price = round(limit_price, 4)

        if limit_price is not None and limit_price <= 0:
            logger.error(f"Invalid limit_price {limit_price} for {symbol}, skipping.")
            if engine.notifier:
                await engine.notifier.send_notification(
                    f"❌ Invalid limit price for {display_symbol}, skipping.",
                    summary={"symbol": symbol, "action": "SKIP", "reason": "Invalid limit price"}
                )
            return

        if limit_price is not None:
            # Read LLM-controlled limit price max distance (fallback to static setting)
            max_distance = settings.LIMIT_PRICE_MAX_DISTANCE_PCT
            try:
                raw = await engine.config_service.get_config("limit_price_max_distance_pct")
                if raw:
                    max_distance = float(raw)
            except (ValueError, TypeError, ConnectionError, TimeoutError, OSError):
                pass
            # For a sell, the limit must not be too far above the bid
            if max_distance > 0 and ticker and ticker.get('bid'):
                bid = ticker['bid']
                if limit_price > bid * (1 + max_distance):
                    logger.warning(
                        f"LLM limit_price {limit_price} for SELL {symbol} is >{max_distance*100:.0f}% above bid {bid}. "
                        f"Rejecting SELL to avoid indefinite queuing."
                    )
                    if engine.notifier:
                        await engine.notifier.send_notification(
                            f"⚠️ Skipping SELL {display_symbol}: limit price {limit_price} too far above bid {bid}.",
                            summary={"symbol": symbol, "action": "SKIP", "reason": "Limit price too far from market"}
                        )
                    return

        # --- Determine order type for SELL ---
        order_type = signal.order_type
        if order_type not in ("market", "limit", "stop", "stop_limit", "trailing_stop"):
            # Fallback: limit if limit_price provided, else market
            if limit_price is not None:
                order_type = "limit"
            else:
                order_type = "market"

        try:
            if order_type == "market":
                order = await asyncio.to_thread(
                    engine.trader.create_market_sell_order, symbol, gross_amount, fill_timeout,
                    limit_price=None, time_in_force='day'
                )
            elif order_type == "limit":
                order = await asyncio.to_thread(
                    engine.trader.create_market_sell_order, symbol, gross_amount, fill_timeout,
                    limit_price=limit_price, time_in_force=time_in_force
                )
            elif order_type == "stop":
                stop_price = signal.stop_price
                if stop_price is None or stop_price <= 0:
                    raise ValueError("Missing or invalid stop_price for stop order")
                order = await asyncio.to_thread(
                    engine.trader.create_stop_sell_order, symbol, gross_amount, stop_price,
                    time_in_force=time_in_force, timeout=fill_timeout
                )
            elif order_type == "stop_limit":
                stop_price = signal.stop_price
                limit_price_sl = limit_price  # LLM-provided limit_price for stop_limit
                if stop_price is None or stop_price <= 0:
                    raise ValueError("Missing or invalid stop_price for stop_limit order")
                if limit_price_sl is None or limit_price_sl <= 0:
                    raise ValueError("Missing or invalid limit_price for stop_limit order")
                order = await asyncio.to_thread(
                    engine.trader.create_stop_limit_sell_order, symbol, gross_amount,
                    stop_price, limit_price_sl,
                    time_in_force=time_in_force, timeout=fill_timeout
                )
            elif order_type == "trailing_stop":
                trail_offset = signal.trail_offset
                if trail_offset is None or trail_offset <= 0:
                    raise ValueError("Missing or invalid trail_offset for trailing_stop order")
                order = await asyncio.to_thread(
                    engine.trader.create_trailing_stop_sell_order, symbol, gross_amount,
                    trail_offset,
                    time_in_force=time_in_force, timeout=fill_timeout
                )
            else:
                raise ValueError(f"Unknown order_type: {order_type}")
            if order.get('status') == 'open':
                order_type_str = "limit" if limit_price is not None else "market"
                # Override with actual order_type if explicitly set
                if signal.order_type in ("market", "limit", "stop", "stop_limit", "trailing_stop"):
                    order_type_str = signal.order_type
                price_str = f" at {limit_price}" if limit_price is not None else ""
                logger.info(f"SELL {order_type_str} order for {symbol} queued{price_str}")
                _sell_queued_entry = {
                    'symbol': symbol,
                    'side': 'sell',
                    'amount': gross_amount,
                    'original_amount': gross_amount,
                    'limit_price': limit_price if order_type in ("limit", "stop_limit") else None,
                    'stop_price': signal.stop_price if order_type in ("stop", "stop_limit") else None,
                    'trail_offset': signal.trail_offset if order_type == "trailing_stop" else None,
                    'order_type': order_type_str,
                    'time_in_force': time_in_force,
                    'signal': asdict(signal),
                    'timeframe': timeframe,
                    'atr': atr,
                    'exit_reason': exit_reason,
                    'order_id': order['id'],
                    'queued_at': time.time(),
                    'filled_qty': 0,
                    'filled_cost': 0.0,
                }
                async with engine._queued_orders_lock:
                    engine.queued_orders.append(_sell_queued_entry)
                if engine.notifier:
                    await engine.notifier.send_notification(
                        f"⏳ SELL {order_type_str} order for {display_symbol} queued{price_str}",
                        summary={"symbol": symbol, "action": "QUEUE", "reason": "Order not yet filled"}
                    )
                return
            if order.get('status') == 'rejected':
                logger.warning(f"SELL order rejected for {symbol}")
                if engine.notifier:
                    await engine.notifier.send_notification(
                        f"❌ SELL order rejected for {display_symbol}",
                        summary={"symbol": symbol, "action": "REJECT", "reason": "Order rejected by simulator"}
                    )
                return
            logger.info(f"SELL {symbol}: {order}")
            # Queue remaining partial market order for polling
            if order.get("remaining_order_id"):
                _sell_queued_entry = {
                    'symbol': symbol,
                    'side': 'sell',
                    'amount': gross_amount - order['amount'],
                    'original_amount': gross_amount - order['amount'],
                    'limit_price': order['price'],
                    'stop_price': None,
                    'trail_offset': None,
                    'order_type': 'limit',
                    'time_in_force': 'day',
                    'signal': asdict(signal),
                    'timeframe': timeframe,
                    'atr': atr,
                    'exit_reason': exit_reason,
                    'order_id': order['remaining_order_id'],
                    'queued_at': time.time(),
                    'filled_qty': 0,
                    'filled_cost': 0.0,
                }
                async with engine._queued_orders_lock:
                    engine.queued_orders.append(_sell_queued_entry)
            # Compute realized P&L
            fee = order.get('fee', {})
            fee_cost = float(fee.get('cost', 0.0) or 0.0)
            fee_currency = fee.get('currency', '')

            net_quote = order['cost'] - (fee_cost if fee_currency == quote else 0.0)
            is_partial_sell = order.get("remaining_order_id") is not None
            if pos:
                cost_basis = pos.get("cost_basis", pos["amount"] * pos["price"])
                net_base = pos.get("net_base", pos["amount"])
                if is_partial_sell and net_base > 0:
                    # Prorate cost basis for the sold portion
                    prorated_cost_basis = cost_basis * (order['amount'] / net_base)
                    realized_pnl = net_quote - prorated_cost_basis
                    order["cost_basis"] = prorated_cost_basis
                else:
                    realized_pnl = net_quote - cost_basis
                    order["cost_basis"] = cost_basis
            else:
                realized_pnl = 0.0
                order["cost_basis"] = 0.0
            order["realized_pnl"] = realized_pnl
            # Track loss timestamps for cooldown
            if realized_pnl < 0:
                engine.last_loss_time[symbol] = time.time()
                cd = pos.get("cooldown_after_loss_seconds", 0) if pos else 0
                engine.cooldown_durations[symbol] = cd
            tf = timeframe or (pos.get("timeframe") if pos else None)
            order["timeframe"] = tf
            order["strategy_type"] = signal.strategy_type
            if pos:
                order["buy_confidence"] = pos.get("buy_confidence", 0.0)
                order["buy_reasoning"] = pos.get("buy_reasoning", "")
            order["exit_reason"] = exit_reason
            order["exit_price"] = order["price"]
            if pos and "timestamp" in pos:
                hold_time = (order["timestamp"] - pos["timestamp"]) / 1000.0
                order["hold_time_seconds"] = hold_time
            else:
                order["hold_time_seconds"] = None
            # Clear any stop-loss review flags
            if pos:
                pos.pop("_stop_loss_triggered", None)
                pos.pop("_stop_loss_review_count", None)

            if is_partial_sell and pos:
                # Partial sell: reduce position instead of removing it
                remaining_amount = pos["amount"] - order['amount']
                remaining_cost_basis = cost_basis - order["cost_basis"]
                remaining_net_base = net_base - order['amount']
                if remaining_amount <= 0 or remaining_net_base <= 0:
                    async with engine._positions_lock:
                        engine.positions.pop(symbol, None)
                    engine._strategy_intervals.pop(symbol, None)
                    engine._last_strategy_eval.pop(symbol, None)
                    engine._last_decisions.pop(symbol, None)
                    engine._pending_entries.pop(symbol, None)
                    await self.event_bus.publish("remove_symbol_if_paused", symbol)
                else:
                    async with engine._positions_lock:
                        engine.positions[symbol]["amount"] = remaining_amount
                        engine.positions[symbol]["cost_basis"] = remaining_cost_basis
                        engine.positions[symbol]["net_base"] = remaining_net_base
                        engine.positions[symbol]["price"] = remaining_cost_basis / remaining_net_base if remaining_net_base > 0 else 0.0
                    # Place replacement exit orders for the remaining position
                    from src.strategies.base import Signal as _Signal
                    _dummy_params = {
                        "trailing_take_profit": engine.positions[symbol].get("trailing_take_profit", False),
                        "partial_take_profit_levels": engine.positions[symbol].get("partial_take_profit_levels"),
                        "partial_take_profit_pct": engine.positions[symbol].get("partial_take_profit_pct"),
                    }
                    _dummy_signal = _Signal(
                        action="BUY",
                        confidence=1.0,
                        reasoning="Replacing exit orders after partial sell",
                        stop_loss_order_type=engine.positions[symbol].get("stop_loss_order_type"),
                        stop_loss_stop_price=engine.positions[symbol].get("stop_loss"),
                        stop_loss_limit_price=None,
                        take_profit_order_type=engine.positions[symbol].get("take_profit_order_type"),
                        take_profit_limit_price=engine.positions[symbol].get("take_profit"),
                        strategy_params=_dummy_params,
                    )
                    _exit_prices = {
                        "stop_loss_price": engine.positions[symbol].get("stop_loss"),
                        "take_profit_price": engine.positions[symbol].get("take_profit"),
                    }
                    await self._place_replacement_exit_orders_with_retry(
                        symbol, _dummy_signal, _exit_prices, engine.positions[symbol].get("timeframe")
                    )
            else:
                # Full sell: remove position
                async with engine._positions_lock:
                    engine.positions.pop(symbol, None)
                engine._strategy_intervals.pop(symbol, None)
                engine._last_strategy_eval.pop(symbol, None)
                engine._last_decisions.pop(symbol, None)
                engine._pending_entries.pop(symbol, None)
                await self.event_bus.publish("remove_symbol_if_paused", symbol)
            engine._append_trade(order)
            engine._balance_cache = None
            await asyncio.to_thread(insert_trade, order)
            await self.event_bus.publish("save_state", force=True)
            engine._portfolio_exposure_cache = None
            if engine.notifier:
                # Human-readable labels for common exit reasons
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
                # --- Format symbol for notification ---
                stock_name = await engine._market_data_manager.get_stock_name(symbol)
                # Use the timeframe from the position or the passed parameter
                tf = timeframe or (pos.get("timeframe") if pos else None)
                display_symbol = engine._format_symbol_display(symbol, stock_name, tf)
                partial_str = " (partial)" if is_partial_sell else ""
                sell_msg = f"🔴 SELL{reason_str}{partial_str} {display_symbol}: {order['amount']:.6f} @ {order['price']:.4f}"
                # Add profit/loss info
                if pos:
                    pnl_pct = (realized_pnl / order["cost_basis"] * 100) if order["cost_basis"] > 0 else 0.0
                    sell_msg += f" | P&L: {realized_pnl:+.4f} ({pnl_pct:+.2f}%)"
                sell_summary = {
                    "symbol": symbol,
                    "action": "SELL",
                    "price": order["price"],
                    "amount": order["amount"],
                    "confidence": signal.confidence,
                    "reason": signal.reasoning[:200],
                    "exit_reason": exit_reason,
                    "realized_pnl": realized_pnl,
                    "strategy_type": signal.strategy_type,
                    "indicators": {
                        "atr": atr,
                    },
                }
                if signal.model_type:
                    sell_summary["model_type"] = signal.model_type
                if signal.llm_provider:
                    sell_summary["llm_provider"] = signal.llm_provider
                if signal.llm_model:
                    sell_summary["llm_model"] = signal.llm_model
                await engine.notifier.send_notification(
                    sell_msg,
                    summary=sell_summary,
                )
        except (RuntimeError, ValueError, ConnectionError, KeyError) as e:
            logger.error(f"Sell order failed for {symbol}: {e}")
            if engine.notifier:
                await engine.notifier.send_notification(
                    f"❌ Sell order failed for {display_symbol}: {e}",
                    summary={
                        "symbol": symbol,
                        "action": "ERROR",
                        "reason": f"Sell order failed: {e}"[:200],
                    }
                )

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
            tf_secs = engine._timeframe_to_seconds(queued_tf)
            scaled_timeout = min(max(base_timeout, int(tf_secs * 0.5)), 604_800)
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
        except (RuntimeError, ValueError, ConnectionError) as e:
            logger.error(f"Failed to cancel timed-out order {order_id}: {e}")

        # Refund remaining reserved capital for buy orders
        if queued['side'] == 'buy':
            async with engine._cycle_spent_lock:
                engine._cycle_spent = max(0.0, engine._cycle_spent - queued.get('amount', 0.0))

        # Remove from queue regardless of cancel success
        async with engine._queued_orders_lock:
            if queued in engine.queued_orders:
                engine.queued_orders.remove(queued)
        engine._state_dirty = True

        if engine.notifier:
            stock_name = await engine._market_data_manager.get_stock_name(queued['symbol'])
            tf = queued.get('timeframe')
            display = engine._format_symbol_display(queued['symbol'], stock_name, tf)
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
            async with engine._cycle_spent_lock:
                engine._cycle_spent = max(0.0, engine._cycle_spent - queued.get('amount', 0.0))
        async with engine._queued_orders_lock:
            if queued in engine.queued_orders:
                engine.queued_orders.remove(queued)
        engine._state_dirty = True

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
            async with engine._cycle_spent_lock:
                engine._cycle_spent = max(0.0, engine._cycle_spent - queued.get('amount', 0.0))
        if engine.notifier:
            stock_name = await engine._market_data_manager.get_stock_name(queued['symbol'])
            tf = queued.get('timeframe')
            display = engine._format_symbol_display(queued['symbol'], stock_name, tf)
            await engine.notifier.send_notification(
                f"❌ Queued {queued['side']} order for {display} {status}.",
                summary={
                    "symbol": queued['symbol'],
                    "action": "INFO",
                    "reason": f"Order {status}",
                }
            )
        async with engine._queued_orders_lock:
            if queued in engine.queued_orders:
                engine.queued_orders.remove(queued)
        engine._state_dirty = True
        if queued.get("is_exit_order"):
            oco_pair_id = queued.get("oco_pair")
            if oco_pair_id:
                try:
                    await asyncio.to_thread(engine.trader.cancel_order, oco_pair_id)
                    logger.info(f"Cancelled OCO pair {oco_pair_id} for {status} exit order {order_id}")
                except (RuntimeError, ValueError, ConnectionError) as e:
                    logger.warning(f"Failed to cancel OCO order {oco_pair_id}: {e}")
                async with engine._queued_orders_lock:
                    engine.queued_orders = [
                        q for q in engine.queued_orders
                        if q.get("order_id") != oco_pair_id
                    ]
            pos = engine.positions.get(queued["symbol"])
            if pos:
                pos.pop("stop_loss_order_id", None)
                pos.pop("take_profit_order_id", None)
            if engine.notifier:
                display_symbol = engine._format_symbol_display(queued["symbol"], stock_name, queued.get("timeframe"))
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

        self._buy_executor._apply_buy_to_position(
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
        engine._append_trade(trade_dict)
        engine._balance_cache = None
        # Note: _cycle_spent was already updated when the order was queued
        # in _execute_signal, so we do NOT add to it here to avoid double-counting.
        await asyncio.to_thread(insert_trade, trade_dict)
        await self.event_bus.publish("save_state", force=True)
        engine._portfolio_exposure_cache = None
        if engine.notifier:
            stock_name = await engine._market_data_manager.get_stock_name(symbol)
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
                exit_prices = self._exit_order_manager.compute_exit_order_prices(
                    entry_price=engine.positions[symbol]["price"],
                    signal=reconstructed_signal,
                    atr=queued.get('atr'),
                )
                await self._place_replacement_exit_orders_with_retry(
                    symbol, reconstructed_signal, exit_prices, queued.get('timeframe')
                )
            except (TypeError, ValueError, RuntimeError, AttributeError) as e:
                logger.error(f"Failed to setup exit orders after queued buy fill for {symbol}: {e}")
                if engine.notifier:
                    stock_name = await engine._market_data_manager.get_stock_name(symbol)
                    display_symbol = engine._format_symbol_display(symbol, stock_name, queued.get('timeframe'))
                    await engine.notifier.send_notification(
                        f"⚠️ Exit order setup failed for {display_symbol} after queued fill: {e}",
                        summary={
                            "symbol": symbol,
                            "action": "ERROR",
                            "reason": f"Exit order setup failed after queued fill: {str(e)[:200]}",
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
            await self._exit_order_manager.cancel_exit_orders(symbol)

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
                await self.event_bus.publish("remove_symbol_if_paused", symbol)
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
                await self._place_replacement_exit_orders_with_retry(
                    symbol, dummy_signal, exit_prices, engine.positions[symbol].get("timeframe")
                )
        else:
            # Full fill (non-partial) – original logic
            # Cancel any remaining exit orders before removing the position
            await self._exit_order_manager.cancel_exit_orders(symbol)
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
            await self.event_bus.publish("remove_symbol_if_paused", symbol)

        engine._append_trade(trade_dict)
        engine._balance_cache = None
        await asyncio.to_thread(insert_trade, trade_dict)
        await self.event_bus.publish("save_state", force=True)
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
            stock_name = await engine._market_data_manager.get_stock_name(symbol)
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
        except (RuntimeError, ValueError, ConnectionError) as e:
            logger.warning(f"Dust sweep: could not fetch balance for {base}: {e}")
            return
        if balance <= 0:
            return

        stock_name = await engine._market_data_manager.get_stock_name(symbol)
        tf = engine.positions.get(symbol, {}).get("timeframe") if symbol in engine.positions else None
        display_symbol = engine._format_symbol_display(symbol, stock_name, tf)

        try:
            base = symbol.split("/")[0]
            quotes = await engine._market_data_manager._get_quotes_async([base], timeout=45.0)
            ticker = quotes.get(base)
            price = ticker["last"]
        except (KeyError, RuntimeError, ConnectionError, ValueError) as e:
            logger.warning(f"Dust sweep: could not fetch price for {symbol}: {e}")
            return

        # Fetch minimum order size from asset info
        try:
            asset = await engine._market_data_manager.get_asset_info(symbol)
            min_amount = float(asset.min_order_size) if asset.min_order_size else None
            if not asset.fractionable and (min_amount is None or min_amount < 1.0):
                min_amount = 1.0
        except (AttributeError, TypeError, ValueError):
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

        need_limit = not await engine._is_market_open()
        limit_price = None
        time_in_force = "day"
        if need_limit:
            limit_price = self._default_limit_price(symbol, "SELL", ticker, atr=None)
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
                await self.event_bus.publish("save_state", force=True)
                engine._portfolio_exposure_cache = None

            # Cancel any remaining exit orders before removing the position
            await self._exit_order_manager.cancel_exit_orders(symbol)

            # Remove the now-empty position
            async with engine._positions_lock:
                engine.positions.pop(symbol, None)
            engine._strategy_intervals.pop(symbol, None)
            engine._last_strategy_eval.pop(symbol, None)
            await self.event_bus.publish("remove_symbol_if_paused", symbol)

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
        except (RuntimeError, ValueError, ConnectionError, KeyError) as e:
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

        stock_name = await engine._market_data_manager.get_stock_name(symbol)
        tf = pos.get("timeframe")
        display_symbol = engine._format_symbol_display(symbol, stock_name, tf)
        base, quote = symbol.split("/")

        # Check minimum sell size
        try:
            asset = await engine._market_data_manager.get_asset_info(symbol)
            min_amount = float(asset.min_order_size) if asset.min_order_size else None
            if not asset.fractionable and (min_amount is None or min_amount < 1.0):
                min_amount = 1.0
        except (AttributeError, TypeError, ValueError):
            min_amount = None
        if min_amount is not None and sell_amount < float(min_amount):
            logger.info(f"{level_label} sell amount {sell_amount:.6f} below min {min_amount} for {symbol}, skipping.")
            return False

        if not await engine._is_market_open():
            logger.info(f"{level_label} for {symbol} skipped: market closed.")
            return False

        need_limit = not await engine._is_market_open()
        limit_price = None
        time_in_force = "day"
        if need_limit:
            limit_price = self._default_limit_price(symbol, "SELL", ticker, atr=atr)
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

            await self._exit_order_manager.cancel_exit_orders(symbol)

            if remaining_amount <= 0 or remaining_net_base <= 0:
                async with engine._positions_lock:
                    engine.positions.pop(symbol, None)
                engine._strategy_intervals.pop(symbol, None)
                engine._last_strategy_eval.pop(symbol, None)
                engine._pending_entries.pop(symbol, None)
                await self.event_bus.publish("remove_symbol_if_paused", symbol)
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
                    await self.sweep_dust(symbol)
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
                    await self._place_replacement_exit_orders_with_retry(
                        symbol, dummy_signal, exit_prices, engine.positions[symbol].get("timeframe")
                    )

            engine._append_trade(order)
            await asyncio.to_thread(insert_trade, order)
            await self.event_bus.publish("save_state", force=True)
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
        except (RuntimeError, ValueError, ConnectionError, KeyError) as e:
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

