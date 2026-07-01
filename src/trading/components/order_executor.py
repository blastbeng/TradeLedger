"""Order execution component for the TradingEngine.

Handles order creation, fill processing, and exit order management.
Extracted from TradingEngine to reduce class size and improve maintainability.
"""
import logging
from typing import Any, Dict, Optional

from src.strategies.base import Signal

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
