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

    def __init__(self, engine):
        self.engine = engine

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
        stock_name = await engine._get_stock_name(symbol)
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
            await self.execute_buy(
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
            await engine._execute_signal(
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
            await engine._execute_signal(
                symbol,
                Signal(action="SELL", confidence=1.0, reasoning="Manual sell"),
                exit_reason="manual_sell"
            )
        else:
            logger.warning(f"No open position for {symbol}")

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
                except (RuntimeError, ValueError, ConnectionError) as e:
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
                except (RuntimeError, ValueError, ConnectionError) as e:
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
            except (RuntimeError, ValueError, ConnectionError) as e:
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
                    except (RuntimeError, ValueError, ConnectionError) as e:
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
            except (RuntimeError, ValueError, ConnectionError) as e:
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

    async def log_manual_trade(self, ticker: str, side: str, quantity: float, money_spent: float, fee: float) -> dict:
        """Log a manually executed trade in notify mode. Persists to DB and updates positions."""
        engine = self.engine
        symbol = f"{ticker}/{engine.base_currency}"
        base = ticker
        quote = engine.base_currency
        price = money_spent / quantity if quantity > 0 else 0.0
        cost = money_spent
        timestamp = int(time.time() * 1000)

        # If fee is not provided (0.0), calculate it using the Intesa Sanpaolo Investo logic
        if fee == 0.0:
            from src.exchanges.fees import calculate_transaction_costs
            costs = calculate_transaction_costs(side.upper(), price, quantity, symbol=ticker)
            fee = costs["total_costs"]

        trade = {
            "id": f"manual_{timestamp}",
            "symbol": symbol,
            "side": side,
            "amount": quantity,
            "price": price,
            "cost": cost,
            "fee": {"cost": fee, "currency": quote},
            "timestamp": timestamp,
            "note": "manual",
            "status": "closed",
            "strategy_type": "manual",
        }

        if side == "buy":
            cost_basis = cost + fee
            net_base = quantity
            if symbol in engine.positions:
                old_pos = engine.positions[symbol]
                old_cost_basis = old_pos.get("cost_basis", old_pos["amount"] * old_pos["price"])
                old_net_base = old_pos.get("net_base", old_pos["amount"])
                new_cost_basis = old_cost_basis + cost_basis
                new_net_base = old_net_base + net_base
                new_price = new_cost_basis / new_net_base if new_net_base > 0 else 0.0
                engine.positions[symbol]["amount"] = new_net_base
                engine.positions[symbol]["price"] = new_price
                engine.positions[symbol]["cost_basis"] = new_cost_basis
                engine.positions[symbol]["net_base"] = new_net_base
            else:
                entry_price = cost_basis / net_base if net_base > 0 else price
                engine.positions[symbol] = {
                    "symbol": symbol,
                    "side": "buy",
                    "amount": net_base,
                    "price": entry_price,
                    "timestamp": timestamp,
                    "stop_loss": None,
                    "take_profit": None,
                    "cost_basis": cost_basis,
                    "net_base": net_base,
                    "timeframe": None,
                    "entry_order_type": "manual",
                    "buy_confidence": 1.0,
                    "buy_reasoning": "Manual trade",
                }
            engine._balance_cache = None

            # Update virtual cash balance
            engine.trader._balances[quote] = engine.trader._balances.get(quote, 0.0) - cost_basis
            engine.trader._balances[base] = engine.trader._balances.get(base, 0.0) + net_base
            engine.trader._balances_dirty = True
            await asyncio.to_thread(engine.trader._save_balances)
        elif side == "sell":
            pos = engine.positions.get(symbol)
            if pos:
                cost_basis = pos.get("cost_basis", pos["amount"] * pos["price"])
                net_quote = cost - fee
                realized_pnl = net_quote - cost_basis
                trade["realized_pnl"] = realized_pnl
                trade["cost_basis"] = cost_basis
                trade["exit_reason"] = "manual_sell"
                if "timestamp" in pos:
                    trade["hold_time_seconds"] = (timestamp - pos["timestamp"]) / 1000.0
                engine.positions.pop(symbol, None)
                engine._balance_cache = None

                # Update virtual cash balance
                engine.trader._balances[base] = engine.trader._balances.get(base, 0.0) - quantity
                engine.trader._balances[quote] = engine.trader._balances.get(quote, 0.0) + net_quote
                engine.trader._balances_dirty = True
                await asyncio.to_thread(engine.trader._save_balances)
            else:
                # Check if the user actually holds enough of the base asset
                current_base_balance = engine.trader._balances.get(base, 0.0)
                if current_base_balance < quantity:
                    logger.warning(
                        f"Manual sell rejected for {symbol}: insufficient {base} balance "
                        f"(have {current_base_balance}, need {quantity})"
                    )
                    return {
                        "status": "error",
                        "error": f"Insufficient {base} balance: have {current_base_balance}, need {quantity}",
                    }

                trade["realized_pnl"] = 0.0
                trade["cost_basis"] = 0.0
                trade["exit_reason"] = "manual_sell"

                # Update virtual cash balance even if position wasn't tracked
                engine.trader._balances[base] = current_base_balance - quantity
                engine.trader._balances[quote] = engine.trader._balances.get(quote, 0.0) + (cost - fee)
                engine.trader._balances_dirty = True
                await asyncio.to_thread(engine.trader._save_balances)

        engine._append_trade(trade)
        await asyncio.to_thread(insert_trade, trade)
        await engine._save_state(force=True)
        engine._portfolio_exposure_cache = None
        logger.info(f"Manual trade logged: {side} {quantity} {symbol} @ {price:.4f}")
        return {"status": "ok", "trade": trade}

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
            await engine._handle_queued_buy_fill(trade_dict, queued)
        else:
            # Update remaining base amount
            original_amount = queued.get('original_amount', queued['amount'])
            queued['amount'] = original_amount - filled_qty
            await engine._handle_queued_sell_fill(trade_dict, queued, partial=True)

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
                    try:
                        await engine._place_exit_orders(
                            queued["symbol"], _dummy_signal, _exit_prices, pos.get("timeframe")
                        )
                    except (RuntimeError, ValueError, ConnectionError) as _e:
                        logger.warning(
                            f"Failed to place replacement exit orders after partial "
                            f"fill for {queued['symbol']}: {_e}"
                        )
            # Notify user
            if engine.notifier:
                stock_name = await engine._get_stock_name(queued["symbol"])
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

        if await self.check_and_cancel_oco_on_stop_trigger(queued):
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

    async def compute_position_size(
        self,
        symbol: str,
        display_symbol: str,
        quote_balance: float,
        desired_amount: float,
        params: Dict[str, Any],
        sl_pct: float,
    ) -> Optional[Tuple[float, float, float]]:
        """Compute the final position size applying all risk caps.

        Applies global risk multiplier, per-symbol multiplier, and a single
        hard ceiling from all risk caps (max_risk_per_trade, max_portfolio_risk,
        max_portfolio_exposure, max_portfolio_stop_risk, and remaining cycle budget).

        Returns (amount, desired_amount, available) or None if the position
        should be skipped (amount <= 0).
        """
        engine = self.engine

        # --- Consolidated position sizing: single hard ceiling from all caps ---
        pos_tickers = await engine._get_all_position_tickers()

        # Compute current portfolio state once
        total_value = quote_balance
        total_open_exposure = 0.0
        total_open_stop_risk = 0.0
        for sym, pos in engine.positions.items():
            try:
                t = pos_tickers.get(sym)
                price = t['last'] if t and t.get('last') else 0.0
                pos_value = pos['amount'] * price
                total_open_exposure += pos_value
                total_value += pos_value
                stop_loss = pos.get('stop_loss')
                if stop_loss is not None and price > 0:
                    loss_if_stop = pos_value * (price - stop_loss) / price
                    total_open_stop_risk += max(0, loss_if_stop)
            except (KeyError, TypeError, ValueError):
                pass

        # Apply global risk multiplier to desired amount (scales all positions)
        global_mult = await engine._get_global_risk_multiplier()
        if global_mult is not None and 0.0 <= global_mult <= 1.0:
            desired_amount *= global_mult

        # Apply per-symbol position size multiplier to desired amount
        per_symbol_mult = params.get("position_size_multiplier")
        if per_symbol_mult is not None:
            try:
                per_symbol_mult = float(per_symbol_mult)
                if 0.0 <= per_symbol_mult <= 1.0:
                    desired_amount *= per_symbol_mult
            except (ValueError, TypeError):
                pass

        # --- Compute individual caps ---
        caps = []

        # Cap 1: max_risk_per_trade_pct (per-trade risk from LLM strategy params)
        max_risk_pct = params.get("max_risk_per_trade_pct")
        if max_risk_pct is not None and sl_pct > 0:
            caps.append(((total_value * max_risk_pct) / sl_pct, f"max_risk_per_trade={max_risk_pct:.2%}"))

        # Cap 2: max_portfolio_risk_pct (portfolio risk from LLM strategy params)
        max_portfolio_risk_pct = params.get("max_portfolio_risk_pct")
        if max_portfolio_risk_pct is not None and sl_pct > 0:
            available_risk_budget = max(0.0, (total_value * max_portfolio_risk_pct) - total_open_stop_risk)
            caps.append((available_risk_budget / sl_pct, f"max_portfolio_risk={max_portfolio_risk_pct:.2%}"))

        # Cap 3: max_portfolio_exposure_pct (global LLM setting from stock selection)
        max_port_exp_raw = await asyncio.to_thread(engine.redis.get, "trading:max_portfolio_exposure_pct")
        max_port_exp = float(max_port_exp_raw) if max_port_exp_raw else None
        if max_port_exp is not None and total_value > 0:
            available_exposure = max(0.0, (max_port_exp * total_value) - total_open_exposure)
            caps.append((available_exposure, f"max_exposure={max_port_exp:.2%}"))

        # Cap 4: max_portfolio_stop_risk_pct (global LLM setting from stock selection)
        max_port_risk_raw = await asyncio.to_thread(engine.redis.get, "trading:max_portfolio_stop_risk_pct")
        max_port_risk = float(max_port_risk_raw) if max_port_risk_raw else None
        if max_port_risk is not None and sl_pct > 0 and total_value > 0:
            available_stop_risk_budget = max(0.0, (total_value * max_port_risk) - total_open_stop_risk)
            caps.append((available_stop_risk_budget / sl_pct, f"max_stop_risk={max_port_risk:.2%}"))

        # Cap 5: remaining cycle budget
        async with engine._cycle_spent_lock:
            available = max(0.0, quote_balance - engine._cycle_spent)
        caps.append((available, "cycle_budget"))

        # --- Determine the binding cap ---
        hard_max = float('inf')
        binding_reason = None
        for cap_value, cap_reason in caps:
            if cap_value < hard_max:
                hard_max = cap_value
                binding_reason = cap_reason

        if hard_max == float('inf'):
            hard_max = 0.0

        # Final amount: min of LLM's desired amount and the single hard ceiling
        amount = min(desired_amount, hard_max)

        if amount <= 0:
            logger.info(f"Skipping BUY {symbol}: position size reduced to 0 by portfolio constraints")
            if engine.notifier:
                await engine.notifier.send_notification(
                    f"⚠️ Skipping BUY {display_symbol}: portfolio constraints leave no room for new position",
                    summary={
                        "symbol": symbol,
                        "action": "SKIP",
                        "reason": "Portfolio constraints exhausted",
                        "desired_amount": desired_amount,
                        "hard_max": 0.0,
                    }
                )
            return None

        if amount < desired_amount:
            # Single consolidated notification about which cap was binding
            cap_reasons = []
            if binding_reason:
                cap_reasons.append(binding_reason)
            if global_mult is not None and global_mult < 1.0:
                cap_reasons.append(f"global_risk_mult={global_mult:.2f}")
            if per_symbol_mult is not None and per_symbol_mult < 1.0:
                cap_reasons.append(f"position_size_mult={per_symbol_mult:.2f}")
            reason_str = ", ".join(cap_reasons) if cap_reasons else "portfolio constraints"
            logger.info(
                f"Position size capped for {symbol}: {desired_amount:.2f} -> {amount:.2f} "
                f"({reason_str})"
            )
            if engine.notifier:
                await engine.notifier.send_notification(
                    f"⚠️ {display_symbol}: position capped {desired_amount:.2f} → {amount:.2f} ({reason_str})",
                    summary={
                        "symbol": symbol,
                        "action": "INFO",
                        "reason": f"Position size capped: {reason_str}",
                        "desired_amount": desired_amount,
                        "capped_amount": amount,
                    }
                )

        return amount, desired_amount, available

    async def compute_sl_tp_params(
        self,
        symbol: str,
        display_symbol: str,
        params: Dict[str, Any],
        atr: Optional[float],
        current_price: float,
        is_btp: bool,
    ) -> Optional[Tuple[float, float, bool, Optional[float]]]:
        """Compute stop-loss and take-profit percentages from LLM params and ATR.

        Returns (sl_pct, tp_pct, trailing_stop, trailing_stop_distance_pct)
        or None if the position should be skipped (missing required params).
        """
        engine = self.engine

        # Determine take-profit percentage based on method
        if "take_profit_atr_multiple" in params and atr is not None and atr > 0 and current_price > 0:
            tp_atr_mult = params["take_profit_atr_multiple"]
            tp_pct = (tp_atr_mult * atr) / current_price
            logger.info(f"ATR-based take-profit: ATR={atr}, multiplier={tp_atr_mult}, take_profit_pct={tp_pct:.4%}")
        else:
            if "take_profit_atr_multiple" in params:
                logger.warning(f"ATR unavailable for {symbol}, falling back to fixed take_profit_pct from LLM params.")
            tp_pct = params.get("take_profit_pct")
            if tp_pct is None or tp_pct <= 0:
                logger.warning(f"Cannot execute BUY for {symbol}: take_profit_pct missing/invalid and ATR unavailable.")
                if engine.notifier:
                    await engine.notifier.send_notification(
                        f"⚠️ Skipping BUY {display_symbol}: missing take_profit_pct and ATR unavailable.",
                        summary={"symbol": symbol, "action": "SKIP", "reason": "Missing take_profit_pct and ATR unavailable"}
                    )
                return None

        # --- BTP take-profit cap: enforce smaller targets for bonds ---
        if is_btp and tp_pct is not None and tp_pct > 0:
            if tp_pct > settings.BTP_MAX_TAKE_PROFIT_PCT:
                logger.info(
                    f"BTP take-profit capped for {symbol}: {tp_pct:.4%} -> "
                    f"{settings.BTP_MAX_TAKE_PROFIT_PCT:.4%}"
                )
                tp_pct = settings.BTP_MAX_TAKE_PROFIT_PCT

        trailing_stop = params["trailing_stop"]
        # Force trailing_stop off for BTPs — not supported by Intesa Sanpaolo Investo
        if is_btp and trailing_stop:
            logger.warning(
                f"LLM set trailing_stop=true for BTP {symbol}, but trailing stops are not supported "
                f"for BTPs on Intesa Sanpaolo Investo. Forcing trailing_stop=false."
            )
            trailing_stop = False
        trailing_stop_distance_pct = params.get("trailing_stop_distance_pct")

        # Determine stop-loss percentage based on method
        stop_method = params.get("stop_loss_method", "fixed")
        if stop_method == "atr_multiple" and atr is not None and atr > 0:
            atr_mult = params["stop_loss_atr_multiple"]
            sl_pct = (atr_mult * atr) / current_price
            logger.info(f"ATR-based stop: ATR={atr}, multiplier={atr_mult}, stop_loss_pct={sl_pct:.4%}")
        else:
            if stop_method == "atr_multiple":
                logger.warning(f"ATR unavailable for {symbol}, falling back to fixed stop_loss_pct from LLM params.")
            sl_pct = params.get("stop_loss_pct")
            if sl_pct is None:
                logger.warning(f"Cannot execute BUY for {symbol}: stop_loss_pct missing and ATR method not applicable/available.")
                if engine.notifier:
                    await engine.notifier.send_notification(
                        f"⚠️ Skipping BUY {display_symbol}: missing stop_loss_pct and ATR unavailable.",
                        summary={"symbol": symbol, "action": "SKIP", "reason": "Missing stop_loss_pct and ATR unavailable"}
                    )
                return None

        return sl_pct, tp_pct, trailing_stop, trailing_stop_distance_pct

    async def check_min_profit_and_order_size(
        self,
        symbol: str,
        display_symbol: str,
        quote: str,
        params: Dict[str, Any],
        amount: float,
        desired_amount: float,
        available: float,
        tp_pct: float,
        current_price: float,
    ) -> Optional[float]:
        """Check minimum profit and adjust order size to meet exchange minimums.

        Returns the final amount (possibly adjusted upward), or None if the
        order should be skipped.
        """
        engine = self.engine

        # --- Minimum absolute profit check (LLM‑defined) ---
        if settings.ENFORCE_MIN_PROFIT_PER_TRADE:
            min_profit = params.get("min_profit_per_trade")
            if min_profit is not None and min_profit > 0:
                expected_gross_profit = amount * tp_pct
                if expected_gross_profit < min_profit:
                    logger.info(
                        f"Skipping BUY {symbol}: expected gross profit {expected_gross_profit:.4f} {quote} "
                        f"below LLM minimum {min_profit:.4f}"
                    )
                    if engine.notifier:
                        await engine.notifier.send_notification(
                            f"⚠️ Skipping BUY {display_symbol}: profit too small ({expected_gross_profit:.4f} {quote})",
                            summary={
                                "symbol": symbol,
                                "action": "SKIP",
                                "reason": "Expected profit below minimum",
                                "expected_profit": expected_gross_profit,
                                "min_profit": min_profit,
                            }
                        )
                    return None

        # Check minimum order size and adjust upward if needed
        try:
            price = current_price
            # Fetch minimum order size from asset info
            try:
                asset = await engine._get_asset_info(symbol)
                min_amount_limit = float(asset.min_order_size) if asset.min_order_size else None
                if not asset.fractionable and (min_amount_limit is None or min_amount_limit < 1.0):
                    min_amount_limit = 1.0
            except (AttributeError, TypeError, ValueError):
                min_amount_limit = None
            # Compute min cost from min amount and current price
            if min_amount_limit is not None and price:
                min_cost_limit = min_amount_limit * price
            else:
                min_cost_limit = None

            # Determine the required minimum quote amount
            required_quote = amount
            if min_amount_limit is not None:
                min_base = float(min_amount_limit)
                required_quote = max(required_quote, min_base * price)
            if min_cost_limit is not None:
                required_quote = max(required_quote, float(min_cost_limit))

            if required_quote > amount:
                # If the required minimum exceeds the risk-limited desired_amount, skip
                if required_quote > desired_amount:
                    logger.info(
                        f"Skipping BUY {symbol}: exchange minimum {required_quote:.2f} "
                        f"exceeds risk-limited amount {desired_amount:.2f}"
                    )
                    if engine.notifier:
                        await engine.notifier.send_notification(
                            f"⚠️ Skipping BUY {display_symbol}: exchange minimum exceeds risk limit",
                            summary={
                                "symbol": symbol,
                                "action": "SKIP",
                                "reason": "Exchange minimum exceeds risk limit",
                                "required_quote": required_quote,
                                "desired_amount": desired_amount,
                            }
                        )
                    return None
                # Adjust amount upward to meet the minimum
                old_amount = amount
                amount = required_quote
                # Check if the adjusted amount exceeds remaining cycle budget
                if amount > available:
                    logger.info(
                        f"BUY amount adjusted from {old_amount:.2f} to {amount:.2f} {quote} "
                        f"to meet minimum, but exceeds remaining cycle budget ({available:.2f}). Skipping."
                    )
                    if engine.notifier:
                        await engine.notifier.send_notification(
                            f"⚠️ BUY skipped for {display_symbol}: amount adjusted to {amount:.2f} but insufficient remaining budget",
                            summary={
                                "symbol": symbol,
                                "action": "SKIP",
                                "reason": "Adjusted amount exceeds remaining budget",
                                "adjusted_amount": amount,
                            }
                        )
                    return None
                logger.info(
                    f"BUY amount adjusted from {old_amount:.2f} to {amount:.2f} {quote} "
                    f"to meet exchange minimum"
                )
                if engine.notifier:
                    await engine.notifier.send_notification(
                        f"ℹ️ {display_symbol}: buy amount adjusted to {amount:.2f} {quote} to meet minimum",
                        summary={
                            "symbol": symbol,
                            "action": "INFO",
                            "reason": "Buy amount adjusted to meet minimum",
                            "adjusted_amount": amount,
                        }
                    )
        except (AttributeError, TypeError, ValueError, RuntimeError) as e:
            logger.warning(f"Could not verify/adjust min order size for {symbol}: {e}")

        return amount

    async def execute_buy(
        self,
        symbol: str,
        display_symbol: str,
        signal: Signal,
        timeframe: Optional[str],
        exit_reason: Optional[str],
        atr: Optional[float],
        balance: Dict[str, float],
    ) -> None:
        """Execute a BUY signal."""
        engine = self.engine
        parts = symbol.split("/")
        if len(parts) != 2:
            logger.error(f"Invalid symbol format: {symbol}")
            return
        base, quote = parts
        _exec_is_btp = is_btp_isin(symbol)

        # Safety: never buy when trading is paused
        paused = await asyncio.to_thread(engine.redis.get, "trading:paused")
        if paused:
            logger.info(f"Ignoring BUY {symbol}: trading is paused (safety check).")
            return
        # Extract known parameters from the LLM's strategy_params (if any)
        params = signal.strategy_params or {}
        fill_timeout = params.get("order_fill_timeout_seconds", settings.ORDER_FILL_TIMEOUT_SECONDS)

        # Fetch current price early for position sizing and stop calculations
        base = symbol.split("/")[0]
        quotes = await engine._get_quotes_async([base], timeout=45.0)
        ticker = quotes.get(base)
        current_price = ticker['last'] if ticker else None
        if current_price is None or current_price <= 0:
            logger.warning(f"Cannot execute BUY for {symbol}: no valid current price.")
            return

        # --- Stale quote guard: skip BUY if the price is too old ---
        tf = timeframe or (engine.positions.get(symbol, {}).get("timeframe") if symbol in engine.positions else None)
        if tf and await engine._is_quote_too_stale(ticker, tf):
            age_seconds = (time.time() * 1000 - ticker.get("last_update", 0)) / 1000
            logger.warning(
                f"Skipping BUY {symbol}: quote is {age_seconds:.0f}s old "
                f"(threshold scaled for timeframe {tf}). "
                f"Stale prices lead to incorrect position sizing and stop-loss calculations."
            )
            if engine.notifier:
                await engine.notifier.send_notification(
                    f"⚠️ Skipping BUY {display_symbol}: quote data is {age_seconds / 60:.0f} min old. "
                    f"Waiting for fresher data.",
                    summary={
                        "symbol": symbol,
                        "action": "SKIP",
                        "reason": "Stale quote data",
                        "age_seconds": round(age_seconds, 1),
                    }
                )
            return

        # --- Compute stop-loss and take-profit parameters from LLM params ---
        _sl_tp_result = await self.compute_sl_tp_params(
            symbol=symbol,
            display_symbol=display_symbol,
            params=params,
            atr=atr,
            current_price=current_price,
            is_btp=_exec_is_btp,
        )
        if _sl_tp_result is None:
            return
        sl_pct, tp_pct, trailing_stop, trailing_stop_distance_pct = _sl_tp_result

        quote_balance = balance.get(quote, 0.0)
        position_fraction = params["position_size_fraction"]

        # Desired amount based on fraction of total available quote balance
        desired_amount = quote_balance * position_fraction

        # Apply confidence-based position sizing (LLM-decided weight)
        confidence_sizing_weight = params.get("confidence_sizing_weight", 0.0)
        if confidence_sizing_weight is not None:
            try:
                confidence_sizing_weight = float(confidence_sizing_weight)
            except (TypeError, ValueError):
                confidence_sizing_weight = 0.0
        if confidence_sizing_weight > 0 and signal.confidence < 1.0:
            confidence_multiplier = 1.0 - confidence_sizing_weight * (1.0 - signal.confidence)
            desired_amount *= confidence_multiplier
            logger.info(
                f"Confidence sizing applied: weight={confidence_sizing_weight}, "
                f"confidence={signal.confidence:.2f}, multiplier={confidence_multiplier:.4f}, "
                f"adjusted amount={desired_amount:.2f}"
            )

        # --- Consolidated position sizing: single hard ceiling from all caps ---
        _sizing_result = await self.compute_position_size(
            symbol=symbol,
            display_symbol=display_symbol,
            quote_balance=quote_balance,
            desired_amount=desired_amount,
            params=params,
            sl_pct=sl_pct,
        )
        if _sizing_result is None:
            return
        amount, desired_amount, available = _sizing_result

        # --- Minimum profit check and exchange minimum order size adjustment ---
        amount = await self.check_min_profit_and_order_size(
            symbol=symbol,
            display_symbol=display_symbol,
            quote=quote,
            params=params,
            amount=amount,
            desired_amount=desired_amount,
            available=available,
            tp_pct=tp_pct,
            current_price=current_price,
        )
        if amount is None:
            return

        # --- Determine limit price for BUY ---
        _limit_result = await self.compute_buy_limit_price(
            symbol=symbol,
            display_symbol=display_symbol,
            params=params,
            ticker=ticker,
            atr=atr,
        )
        if _limit_result is None:
            return
        limit_price, time_in_force, need_limit = _limit_result

        # --- Determine order type ---
        order_type = signal.order_type
        if order_type not in ("market", "limit", "stop", "stop_limit", "trailing_stop"):
            # Fallback to existing behaviour: limit if limit_price provided, else market
            if limit_price is not None:
                order_type = "limit"
            else:
                order_type = "market"

        # --- Reserve cycle budget before placing order to prevent race condition ---
        async with engine._cycle_spent_lock:
            available = max(0.0, quote_balance - engine._cycle_spent)
            if amount > available:
                logger.info(
                    f"Skipping BUY {symbol}: cycle budget exhausted "
                    f"(needed {amount:.2f}, available {available:.2f}) due to concurrent order"
                )
                if engine.notifier:
                    await engine.notifier.send_notification(
                        f"⚠️ Skipping BUY {display_symbol}: cycle budget exhausted by concurrent order",
                        summary={"symbol": symbol, "action": "SKIP", "reason": "Cycle budget exhausted (concurrent order)"}
                    )
                return
            engine._cycle_spent += amount

        try:
            if order_type == "market":
                order = await asyncio.to_thread(
                    engine.trader.create_market_buy_order, symbol, amount, fill_timeout,
                    limit_price=None, time_in_force='day'
                )
            elif order_type == "limit":
                order = await asyncio.to_thread(
                    engine.trader.create_market_buy_order, symbol, amount, fill_timeout,
                    limit_price=limit_price, time_in_force=time_in_force
                )
            elif order_type == "stop":
                stop_price = signal.stop_price
                if stop_price is None or stop_price <= 0:
                    raise ValueError("Missing or invalid stop_price for stop order")
                order = await asyncio.to_thread(
                    engine.trader.create_stop_buy_order, symbol, amount, stop_price,
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
                    engine.trader.create_stop_limit_buy_order, symbol, amount,
                    stop_price, limit_price_sl,
                    time_in_force=time_in_force, timeout=fill_timeout
                )
            elif order_type == "trailing_stop":
                trail_offset = signal.trail_offset
                if trail_offset is None or trail_offset <= 0:
                    raise ValueError("Missing or invalid trail_offset for trailing_stop order")
                order = await asyncio.to_thread(
                    engine.trader.create_trailing_stop_buy_order, symbol, amount,
                    trail_offset,
                    time_in_force=time_in_force, timeout=fill_timeout
                )
            else:
                raise ValueError(f"Unknown order_type: {order_type}")
            if order.get('status') == 'open':
                price_str = f" at {limit_price}" if limit_price is not None else ""
                logger.info(f"BUY {order_type} order for {symbol} queued{price_str}")
                queued_entry = {
                    'symbol': symbol,
                    'side': 'buy',
                    'amount': amount,
                    'original_amount': amount,
                    'limit_price': limit_price if order_type in ("limit", "stop_limit") else None,
                    'stop_price': signal.stop_price if order_type in ("stop", "stop_limit") else None,
                    'trail_offset': signal.trail_offset if order_type == "trailing_stop" else None,
                    'order_type': order_type,
                    'time_in_force': time_in_force,
                    'signal': asdict(signal),
                    'timeframe': timeframe,
                    'atr': atr,
                    'order_id': order['id'],
                    'queued_at': time.time(),
                    'filled_qty': 0,
                    'filled_cost': 0.0,
                }
                async with engine._queued_orders_lock:
                    engine.queued_orders.append(queued_entry)
                if engine.notifier:
                    await engine.notifier.send_notification(
                        f"⏳ BUY {order_type} order for {display_symbol} queued{price_str}",
                        summary={"symbol": symbol, "action": "QUEUE", "reason": "Order not yet filled"}
                    )
                return
            if order.get('status') == 'rejected':
                async with engine._cycle_spent_lock:
                    engine._cycle_spent = max(0.0, engine._cycle_spent - amount)
                logger.warning(f"BUY order rejected for {symbol}")
                if engine.notifier:
                    await engine.notifier.send_notification(
                        f"❌ BUY order rejected for {display_symbol}",
                        summary={"symbol": symbol, "action": "REJECT", "reason": "Order rejected by simulator"}
                    )
                return
            logger.info(f"BUY {symbol}: {order}")
            # Queue remaining partial market order for polling
            if order.get("remaining_order_id"):
                queued_entry = {
                    'symbol': symbol,
                    'side': 'buy',
                    'amount': amount - order['cost'],
                    'original_amount': amount - order['cost'],
                    'limit_price': order['price'],
                    'stop_price': None,
                    'trail_offset': None,
                    'order_type': 'limit',
                    'time_in_force': 'day',
                    'signal': asdict(signal),
                    'timeframe': timeframe,
                    'atr': atr,
                    'order_id': order['remaining_order_id'],
                    'queued_at': time.time(),
                    'filled_qty': 0,
                    'filled_cost': 0.0,
                }
                async with engine._queued_orders_lock:
                    engine.queued_orders.append(queued_entry)
            await self.update_or_create_buy_position(
                symbol=symbol,
                order=order,
                signal=signal,
                params=params,
                quote=quote,
                base=base,
                sl_pct=sl_pct,
                tp_pct=tp_pct,
                trailing_stop=trailing_stop,
                trailing_stop_distance_pct=trailing_stop_distance_pct,
                order_type=order_type,
                timeframe=timeframe,
            )
            await self.record_buy_fill_and_notify(
                symbol=symbol,
                display_symbol=display_symbol,
                order=order,
                signal=signal,
                timeframe=timeframe,
                atr=atr,
            )
        except (RuntimeError, ValueError, ConnectionError, KeyError) as e:
            async with engine._cycle_spent_lock:
                engine._cycle_spent = max(0.0, engine._cycle_spent - amount)
            logger.error(f"Buy order failed for {symbol}: {e}")
            if engine.notifier:
                await engine.notifier.send_notification(
                    f"❌ Buy order failed for {display_symbol}: {e}",
                    summary={
                        "symbol": symbol,
                        "action": "ERROR",
                        "reason": f"Buy order failed: {e}"[:200],
                    }
                )

    async def execute_sell(
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
            await engine._cancel_exit_orders(symbol)

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
            quotes = await engine._get_quotes_async([base], timeout=45.0)
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
                asset = await engine._get_asset_info(symbol)
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

        need_limit = not engine._is_regular_hours()
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
                raw = await asyncio.to_thread(engine.redis.get, "trading:limit_price_max_distance_pct")
                if raw:
                    max_distance = float(raw)
            except (TypeError, ValueError, RuntimeError):
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
                    await engine._remove_symbol_if_paused(symbol)
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
                    try:
                        await engine._place_exit_orders(symbol, _dummy_signal, _exit_prices, engine.positions[symbol].get("timeframe"))
                    except (TypeError, ValueError, RuntimeError, AttributeError) as _e:
                        logger.warning(f"Failed to place replacement exit orders after partial sell for {symbol}: {_e}")
            else:
                # Full sell: remove position
                async with engine._positions_lock:
                    engine.positions.pop(symbol, None)
                engine._strategy_intervals.pop(symbol, None)
                engine._last_strategy_eval.pop(symbol, None)
                engine._last_decisions.pop(symbol, None)
                engine._pending_entries.pop(symbol, None)
                await engine._remove_symbol_if_paused(symbol)
            engine._append_trade(order)
            engine._balance_cache = None
            await asyncio.to_thread(insert_trade, order)
            await engine._save_state(force=True)
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
                stock_name = await engine._get_stock_name(symbol)
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

    async def compute_buy_limit_price(
        self,
        symbol: str,
        display_symbol: str,
        params: Dict[str, Any],
        ticker: Dict[str, Any],
        atr: Optional[float],
    ) -> Optional[Tuple[Optional[float], str, bool]]:
        """Determine the limit price, time-in-force, and order path for a BUY.

        Returns (limit_price, time_in_force, need_limit) or None if the
        order should be skipped (invalid limit price).
        """
        engine = self.engine
        need_limit = not engine._is_regular_hours()
        limit_price = None
        time_in_force = "day"
        # If LLM provided a limit_price, use it even during regular hours
        llm_limit_price = params.get("limit_price")
        if llm_limit_price is not None and llm_limit_price > 0:
            limit_price = llm_limit_price
            time_in_force = params.get("time_in_force", "day")
            need_limit = True  # force limit order path
            # Validate that the limit price is within a reasonable distance from the market
            # Read LLM-controlled limit price max distance (fallback to static setting)
            max_distance = settings.LIMIT_PRICE_MAX_DISTANCE_PCT
            try:
                raw = await asyncio.to_thread(engine.redis.get, "trading:limit_price_max_distance_pct")
                if raw:
                    max_distance = float(raw)
            except (TypeError, ValueError, RuntimeError):
                pass
            if ticker and ticker.get('ask') and max_distance > 0:
                ask = ticker['ask']
                if limit_price < ask * (1 - max_distance):
                    logger.warning(
                        f"LLM limit_price {limit_price} for {symbol} is >{max_distance*100:.0f}% below ask {ask}. "
                        f"Rejecting BUY to avoid indefinite queuing."
                    )
                    if engine.notifier:
                        await engine.notifier.send_notification(
                            f"⚠️ Skipping BUY {display_symbol}: limit price {limit_price} too far below ask {ask}.",
                            summary={"symbol": symbol, "action": "SKIP", "reason": "Limit price too far from market"}
                        )
                    return None
        elif need_limit:
            limit_price = self._default_limit_price(symbol, "BUY", ticker, atr=atr)
            time_in_force = params.get("time_in_force", "day")
            if limit_price is None:
                logger.error(f"Cannot place limit order for {symbol}: no limit price available.")
                return None

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
            return None

        return limit_price, time_in_force, need_limit

    async def update_or_create_buy_position(
        self,
        symbol: str,
        order: Dict[str, Any],
        signal: Signal,
        params: Dict[str, Any],
        quote: str,
        base: str,
        sl_pct: float,
        tp_pct: float,
        trailing_stop: bool,
        trailing_stop_distance_pct: Optional[float],
        order_type: str,
        timeframe: Optional[str],
    ) -> None:
        """Update an existing position or create a new one after a filled BUY order."""
        engine = self.engine
        # Extract fee info for cost basis tracking
        fee = order.get('fee', {})
        fee_cost = float(fee.get('cost', 0.0) or 0.0)
        fee_currency = fee.get('currency', '')

        cost_basis = order['cost'] + (fee_cost if fee_currency == quote else 0.0)
        net_base = order['amount'] - (fee_cost if fee_currency == base else 0.0)

        # Risk parameters are guaranteed by the validator
        # sl_pct, tp_pct, trailing_stop, trailing_stop_distance_pct are set above

        if symbol in engine.positions:
            # Accumulate: weighted average price with cost basis
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
            # them from where the LLM originally intended. The LLM can
            # still update SL/TP via _update_position_params (which uses
            # current_price, not the new average).
            engine.positions[symbol]["take_profit_atr_multiple"] = params.get("take_profit_atr_multiple")
            engine.positions[symbol]["trailing_stop"] = trailing_stop
            engine.positions[symbol]["trailing_stop_distance_pct"] = trailing_stop_distance_pct
            engine.positions[symbol]["trailing_stop_atr_multiple"] = params.get("trailing_stop_atr_multiple")
            engine.positions[symbol]["max_hold_time_seconds"] = params.get("max_hold_time_seconds")
            engine.positions[symbol]["trailing_stop_activation_pct"] = params.get("trailing_stop_activation_pct")
            engine.positions[symbol]["trailing_take_profit"] = params.get("trailing_take_profit", False)
            engine.positions[symbol]["trailing_take_profit_distance_pct"] = params.get("trailing_take_profit_distance_pct")
            engine.positions[symbol]["breakeven_activation_pct"] = params.get("breakeven_activation_pct")
            # Multiple partial take-profit levels
            partial_levels = params.get("partial_take_profit_levels")
            if partial_levels:
                engine.positions[symbol]["partial_take_profit_levels"] = partial_levels
                engine.positions[symbol]["partial_tp_levels_triggered"] = []
                engine.positions[symbol]["partial_tp_depth_wait_start"] = {}
                # Clear single-level fields to avoid confusion
                engine.positions[symbol]["partial_take_profit_pct"] = None
                engine.positions[symbol]["partial_take_profit_fraction"] = None
                engine.positions[symbol]["partial_tp_triggered"] = None
            else:
                engine.positions[symbol]["partial_take_profit_pct"] = params.get("partial_take_profit_pct")
                engine.positions[symbol]["partial_take_profit_fraction"] = params.get("partial_take_profit_fraction")
                engine.positions[symbol]["partial_tp_triggered"] = False
            engine.positions[symbol]["cooldown_after_loss_seconds"] = params["cooldown_after_loss_seconds"]
            engine.positions[symbol]["news_sentiment_exit_threshold"] = params.get("news_sentiment_exit_threshold")
            engine.positions[symbol]["max_unrealized_loss_pct"] = params.get("max_unrealized_loss_pct")
            custom_interval = params.get("strategy_interval_seconds")
            if custom_interval is not None:
                engine._strategy_intervals[symbol] = custom_interval
            engine.positions[symbol]["timeframe"] = timeframe
            engine.positions[symbol]["indicator_config"] = signal.indicator_config
            engine.positions[symbol]["entry_order_type"] = order_type
            engine.positions[symbol]["buy_confidence"] = signal.confidence
            engine.positions[symbol]["buy_reasoning"] = (signal.reasoning or "")[:200]
        else:
            entry_price = cost_basis / net_base if net_base > 0 else order["price"]
            engine.positions[symbol] = {
                "symbol": symbol,
                "side": "buy",
                "amount": net_base,
                "price": entry_price,
                "timestamp": order["timestamp"],
                "stop_loss": entry_price * (1 - sl_pct),
                "take_profit": entry_price * (1 + tp_pct),
                "take_profit_atr_multiple": params.get("take_profit_atr_multiple"),
                "cost_basis": cost_basis,
                "net_base": net_base,
                "buy_confidence": signal.confidence,
                "buy_reasoning": (signal.reasoning or "")[:200],
                "trailing_stop": trailing_stop,
                "trailing_stop_distance_pct": trailing_stop_distance_pct,
                "trailing_stop_atr_multiple": params.get("trailing_stop_atr_multiple"),
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
                "cooldown_after_loss_seconds": params["cooldown_after_loss_seconds"],
                "news_sentiment_exit_threshold": params.get("news_sentiment_exit_threshold"),
                "max_unrealized_loss_pct": params.get("max_unrealized_loss_pct"),
                "timeframe": timeframe,
                "indicator_config": signal.indicator_config,
                "entry_order_type": order_type,
            }
            custom_interval = params.get("strategy_interval_seconds")
            if custom_interval is not None:
                engine._strategy_intervals[symbol] = custom_interval

    async def record_buy_fill_and_notify(
        self,
        symbol: str,
        display_symbol: str,
        order: Dict[str, Any],
        signal: Signal,
        timeframe: Optional[str],
        atr: Optional[float],
    ) -> None:
        """Place exit orders, record the trade, and send BUY notification after a fill."""
        engine = self.engine
        # --- Place native exit orders (OCO) if LLM specified them ---
        current_entry = engine.positions[symbol]["price"]
        exit_prices = self.compute_exit_order_prices(
            entry_price=current_entry,
            signal=signal,
            atr=atr,
        )
        await self.place_exit_orders(symbol, signal, exit_prices, timeframe)
        order["strategy_type"] = signal.strategy_type
        order["timeframe"] = timeframe
        order["buy_confidence"] = signal.confidence
        order["buy_reasoning"] = (signal.reasoning or "")[:200]
        if hasattr(signal, 'backtest_summary') and signal.backtest_summary:
            order["backtest_summary"] = signal.backtest_summary
        engine._append_trade(order)
        engine._balance_cache = None  # force refresh on next fetch
        await asyncio.to_thread(insert_trade, order)
        await engine._save_state(force=True)
        engine._portfolio_exposure_cache = None
        if engine.notifier:
            buy_msg = f"🟢 BUY {display_symbol}: {order['amount']:.6f} @ {order['price']:.4f}"
            buy_summary = {
                "symbol": symbol,
                "action": "BUY",
                "price": order["price"],
                "amount": order["amount"],
                "confidence": signal.confidence,
                "reason": signal.reasoning[:200],
                "strategy_type": signal.strategy_type,
                "indicators": {
                    "atr": atr,
                },
            }
            if signal.model_type:
                buy_summary["model_type"] = signal.model_type
            if signal.llm_provider:
                buy_summary["llm_provider"] = signal.llm_provider
            if signal.llm_model:
                buy_summary["llm_model"] = signal.llm_model
            await engine.notifier.send_notification(
                buy_msg,
                summary=buy_summary,
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
            quotes = await engine._get_quotes_async([base], timeout=45.0)
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
            pos = engine.positions.get(queued["symbol"])
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
                logger.warning(f"Failed to cancel OCO order {oco_pair_id}: {e}")
            # Remove the cancelled take-profit from queued_orders (with lock)
            async with engine._queued_orders_lock:
                engine.queued_orders = [
                    q for q in engine.queued_orders
                    if q.get("order_id") != oco_pair_id
                ]
            # Clear OCO reference so we don't try again
            queued["oco_pair"] = None
            # Clear take-profit order ID from position
            pos = engine.positions.get(queued["symbol"])
            if pos:
                pos.pop("take_profit_order_id", None)
            # Notify user
            if engine.notifier:
                stock_name = await engine._get_stock_name(queued["symbol"])
                display_symbol = engine._format_symbol_display(
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
        engine._state_dirty = True
        return True

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
            stock_name = await engine._get_stock_name(queued['symbol'])
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
            stock_name = await engine._get_stock_name(queued['symbol'])
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
                stock_name = await engine._get_stock_name(queued["symbol"])
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
            except (TypeError, ValueError, RuntimeError, AttributeError) as e:
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
        except (RuntimeError, ValueError, ConnectionError) as e:
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
        except (KeyError, RuntimeError, ConnectionError, ValueError) as e:
            logger.warning(f"Dust sweep: could not fetch price for {symbol}: {e}")
            return

        # Fetch minimum order size from asset info
        try:
            asset = await engine._get_asset_info(symbol)
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

        need_limit = not engine._is_regular_hours()
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
        except (AttributeError, TypeError, ValueError):
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
            except (RuntimeError, ValueError, ConnectionError) as e:
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
        except (RuntimeError, ValueError, ConnectionError, KeyError) as e:
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
            except (RuntimeError, ValueError, ConnectionError):
                pass
            async with engine._queued_orders_lock:
                engine.queued_orders = [q for q in engine.queued_orders if q.get("order_id") != oco_pair_id]
        pos.pop("stop_loss_order_id", None)
        pos.pop("take_profit_order_id", None)
        pos.pop("stop_loss_order_type", None)
        pos.pop("_native_stop_price", None)
