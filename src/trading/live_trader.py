import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class LiveTrader:
    """Dummy trader. Alpaca is no longer used. Will be replaced by PaperTrader in Step 4."""

    def __init__(self, trading_client=None):
        self.trading_client = trading_client

    def get_balance(self, currency: str) -> float:
        return 0.0

    def fetch_balance(self) -> Dict[str, float]:
        return {}

    def create_market_buy_order(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("Live trading is disabled. Use paper mode.")

    def create_market_sell_order(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("Live trading is disabled. Use paper mode.")

    def create_stop_buy_order(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("Live trading is disabled. Use paper mode.")

    def create_stop_sell_order(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("Live trading is disabled. Use paper mode.")

    def create_stop_limit_buy_order(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("Live trading is disabled. Use paper mode.")

    def create_stop_limit_sell_order(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("Live trading is disabled. Use paper mode.")

    def create_trailing_stop_buy_order(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("Live trading is disabled. Use paper mode.")

    def create_trailing_stop_sell_order(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("Live trading is disabled. Use paper mode.")

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        return []

    def get_order(self, order_id: str):
        """Return None – no real orders exist in notify mode."""
        return None

    def cancel_order(self, order_id: str) -> bool:
        return False

    def get_trade_history(self) -> List[Dict[str, Any]]:
        return []
