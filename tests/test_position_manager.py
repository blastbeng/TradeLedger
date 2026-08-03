import pytest
from unittest.mock import AsyncMock, MagicMock

from src.trading.components.position_manager import PositionManager


@pytest.mark.asyncio
async def test_compute_portfolio_exposure_summary():
    mock_engine = MagicMock()
    mock_event_bus = MagicMock()
    manager = PositionManager(mock_engine, mock_event_bus)
    
    # Configure shared_state mock
    manager.shared_state.positions = {}
    manager.shared_state._cycle_spent = 0.0
    manager.shared_state._cycle_spent_lock = AsyncMock()
    
    # Mock market data manager method
    mock_engine._market_data_manager._get_all_position_tickers = AsyncMock(return_value={})
    
    summary = await manager.compute_portfolio_exposure_summary(base_balance=10000.0)
    assert isinstance(summary, dict)
    assert "portfolio_total_value" in summary
    assert "portfolio_exposure" in summary
    assert isinstance(summary["portfolio_exposure"], float)
    assert isinstance(summary["portfolio_exposure_pct"], float)
