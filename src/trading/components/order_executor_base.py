"""Base class for order executors, providing shared logic."""
import logging
from typing import Any, Dict, Optional, Tuple

from src.strategies.base import Signal

logger = logging.getLogger(__name__)


class OrderExecutorBase:
    """Base class for order executors, providing shared logic for order types, limit prices, and fees."""

    def __init__(self, engine, event_bus):
        self.engine = engine
        self.shared_state = engine.shared_state
        self.event_bus = event_bus

    def _determine_order_type(self, signal: Signal, limit_price: Optional[float]) -> str:
        """Determine the order type, falling back to limit or market if invalid."""
        order_type = signal.order_type
        if order_type not in ("market", "limit", "stop", "stop_limit", "trailing_stop"):
            if limit_price is not None:
                order_type = "limit"
            else:
                order_type = "market"
        return order_type

    def _round_limit_price(self, limit_price: Optional[float]) -> Optional[float]:
        """Round limit price to valid tick size ($0.01 for >=$1, $0.0001 for <$1)."""
        if limit_price is not None:
            if limit_price >= 1.0:
                limit_price = round(limit_price, 2)
            else:
                limit_price = round(limit_price, 4)
        return limit_price

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

    def _extract_fee(self, order: Dict[str, Any]) -> Tuple[float, str]:
        """Extract fee cost and currency from an order dictionary."""
        fee = order.get('fee', {})
        fee_cost = float(fee.get('cost', 0.0) or 0.0)
        fee_currency = fee.get('currency', '')
        return fee_cost, fee_currency

    def _compute_pnl_and_proration(
        self, pos: Optional[Dict[str, Any]], sold_amount: float, net_quote: float
    ) -> Tuple[float, float, float, float]:
        """Compute prorated cost basis and realized P&L for a sell.
        Returns: (realized_pnl, prorated_cost_basis, cost_basis, net_base)
        """
        if not pos:
            return 0.0, 0.0, 0.0, 0.0
        cost_basis = pos.get("cost_basis", pos["amount"] * pos["price"])
        net_base = pos.get("net_base", pos["amount"])
        prorated_cost_basis = cost_basis * (sold_amount / pos["amount"]) if pos["amount"] > 0 else 0.0
        realized_pnl = net_quote - prorated_cost_basis
        return realized_pnl, prorated_cost_basis, cost_basis, net_base

    async def _send_buy_notification(
        self,
        symbol: str,
        display_symbol: str,
        order: Dict[str, Any],
        signal: Signal,
        atr: Optional[float] = None,
    ) -> None:
        """Format and send the BUY notification."""
        engine = self.engine
        if not engine.notifier:
            return
        buy_msg = f"🟢 BUY {display_symbol}: {order['amount']:.6f} @ {order['price']:.4f}"
        buy_summary = {
            "symbol": symbol,
            "action": "BUY",
            "price": order["price"],
            "amount": order["amount"],
            "confidence": signal.confidence,
            "reason": (signal.reasoning or '')[:200],
            "strategy_type": signal.strategy_type,
        }
        if atr is not None:
            buy_summary["indicators"] = {"atr": atr}
        if signal.model_type:
            buy_summary["model_type"] = signal.model_type
        if signal.llm_provider:
            buy_summary["llm_provider"] = signal.llm_provider
        if signal.llm_model:
            buy_summary["llm_model"] = signal.llm_model
        await engine.notifier.send_notification(buy_msg, summary=buy_summary)
