import pytest
from unittest.mock import MagicMock
from src.strategies.base import Signal
from src.trading.components.exit_order_manager import ExitOrderManager


def test_compute_exit_order_prices_stop():
    mock_engine = MagicMock()
    mock_event_bus = MagicMock()
    manager = ExitOrderManager(mock_engine, mock_event_bus)
    signal = Signal(
        action="BUY",
        confidence=0.8,
        reasoning="Test",
        stop_loss_order_type="stop",
        stop_loss_stop_price=95.0,
        take_profit_order_type="limit",
        take_profit_limit_price=110.0,
    )
    result = manager.compute_exit_order_prices(entry_price=100.0, signal=signal)
    assert result["stop_loss_price"] == 95.0
    assert result["take_profit_price"] == 110.0


def test_compute_exit_order_prices_fallback():
    mock_engine = MagicMock()
    mock_event_bus = MagicMock()
    manager = ExitOrderManager(mock_engine, mock_event_bus)
    signal = Signal(
        action="BUY",
        confidence=0.8,
        reasoning="Test",
        strategy_params={"stop_loss_pct": 0.05, "take_profit_pct": 0.10},
    )
    result = manager.compute_exit_order_prices(entry_price=100.0, signal=signal)
    assert result["stop_loss_price"] == 95.0
    assert result["take_profit_price"] == 110.0
