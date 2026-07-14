import pytest
from unittest.mock import AsyncMock, MagicMock

from src.trading.components.position_manager import PositionManager


def test_compute_portfolio_exposure_summary():
    mock_dependency = MagicMock()
    manager = PositionManager(mock_dependency)
    summary = manager.compute_portfolio_exposure_summary(base_balance=10000.0)
    assert isinstance(summary, dict)
    assert "total_exposure" in summary
    assert "exposure_pct" in summary
    assert isinstance(summary["total_exposure"], float)
    assert isinstance(summary["exposure_pct"], float)
