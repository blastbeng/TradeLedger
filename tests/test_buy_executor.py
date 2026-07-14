import pytest
from unittest.mock import AsyncMock, MagicMock

from src.trading.components.buy_executor import BuyExecutor


def test_compute_position_size():
    mock_dependency = MagicMock()
    executor = BuyExecutor(mock_dependency)
    amount = executor.compute_position_size(
        symbol="AAPL",
        display_symbol="AAPL",
        quote_balance=10000.0,
        desired_amount=5000.0,
        params={},
        sl_pct=0.05,
        atr=2.0,
        current_price=100.0,
    )
    assert isinstance(amount, float)
    assert amount <= 10000.0
