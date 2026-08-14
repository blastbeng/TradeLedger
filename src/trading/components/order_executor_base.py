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
