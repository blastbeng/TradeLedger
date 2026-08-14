"""Manual trade logging component for the TradingEngine.

Handles logging of manually executed trades in notify mode.
Extracted from OrderExecutor to reduce class size and improve maintainability.
"""
import asyncio
import logging
import time

from src.config.settings import settings
from src.database import insert_trade

logger = logging.getLogger(__name__)


class ManualTradeLogger:
    """Handles manual trade logging for the TradingEngine."""

    def __init__(self, engine, event_bus):
        self.engine = engine
        self.shared_state = engine.shared_state
        self.event_bus = event_bus
        self.event_bus.subscribe("log_manual_trade", self.log_manual_trade)

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
            if symbol in self.shared_state.positions:
                old_pos = self.shared_state.positions[symbol]
                old_cost_basis = old_pos.get("cost_basis", old_pos["amount"] * old_pos["price"])
                old_net_base = old_pos.get("net_base", old_pos["amount"])
                new_cost_basis = old_cost_basis + cost_basis
                new_net_base = old_net_base + net_base
                new_price = new_cost_basis / new_net_base if new_net_base > 0 else 0.0
                self.shared_state.positions[symbol]["amount"] = new_net_base
                self.shared_state.positions[symbol]["price"] = new_price
                self.shared_state.positions[symbol]["cost_basis"] = new_cost_basis
                self.shared_state.positions[symbol]["net_base"] = new_net_base
            else:
                entry_price = cost_basis / net_base if net_base > 0 else price
                self.shared_state.positions[symbol] = {
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
                    "_needs_risk_params": True,
                    "_needs_risk_params_attempts": 0,
                }
                # Force immediate re-evaluation so the LLM can provide risk parameters
                self.shared_state._force_eval[symbol] = True
                self.shared_state._last_strategy_eval.pop(symbol, None)
            self.shared_state._balance_cache = None

            # Update virtual cash balance
            with engine.trader._lock:
                engine.trader._balances[quote] = engine.trader._balances.get(quote, 0.0) - cost_basis
                engine.trader._balances[base] = engine.trader._balances.get(base, 0.0) + net_base
                engine.trader._balances_dirty = True
            await asyncio.to_thread(engine.trader._save_balances)
        elif side == "sell":
            pos = self.shared_state.positions.get(symbol)
            if pos:
                cost_basis = pos.get("cost_basis", pos["amount"] * pos["price"])
                net_quote = cost - fee
                realized_pnl = net_quote - cost_basis
                trade["realized_pnl"] = realized_pnl
                trade["cost_basis"] = cost_basis
                trade["exit_reason"] = "manual_sell"
                if "timestamp" in pos:
                    trade["hold_time_seconds"] = (timestamp - pos["timestamp"]) / 1000.0
                self.shared_state.positions.pop(symbol, None)
                self.shared_state._balance_cache = None

                # Update virtual cash balance
                with engine.trader._lock:
                    engine.trader._balances[base] = engine.trader._balances.get(base, 0.0) - quantity
                    engine.trader._balances[quote] = engine.trader._balances.get(quote, 0.0) + net_quote
                    engine.trader._balances_dirty = True
                await asyncio.to_thread(engine.trader._save_balances)
            else:
                # Check if the user actually holds enough of the base asset
                with engine.trader._lock:
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

        self.shared_state.append_trade(trade, settings.MAX_TRADES_IN_MEMORY)
        await asyncio.to_thread(insert_trade, trade)
        await self.event_bus.publish("save_state", force=True)
        self.shared_state._portfolio_exposure_cache = None
        logger.info(f"Manual trade logged: {side} {quantity} {symbol} @ {price:.4f}")
        return {"status": "ok", "trade": trade}
