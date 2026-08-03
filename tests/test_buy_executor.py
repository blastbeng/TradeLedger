import pytest
from unittest.mock import AsyncMock, MagicMock

from src.trading.components.buy_executor import BuyExecutor


@pytest.mark.asyncio
async def test_compute_position_size():
    mock_engine = MagicMock()
    mock_event_bus = MagicMock()
    executor = BuyExecutor(mock_engine, mock_event_bus)
    
    executor.shared_state.positions = {}
    executor.shared_state._cycle_spent = 0.0
    executor.shared_state._cycle_spent_lock = AsyncMock()
    
    mock_engine._get_cached_position_tickers = AsyncMock(return_value={})
    mock_engine.redis.get = AsyncMock(return_value=None)
    mock_engine.config_service.get_config = AsyncMock(return_value=None)
    mock_engine._get_global_risk_multiplier = AsyncMock(return_value=1.0)
    
    result = await executor.compute_position_size(
        symbol="AAPL",
        display_symbol="AAPL",
        quote_balance=10000.0,
        desired_amount=5000.0,
        params={},
        sl_pct=0.05,
        atr=2.0,
        current_price=100.0,
    )
    assert isinstance(result, tuple)
    assert isinstance(result[0], float)
    assert result[0] <= 10000.0
