"""Tests for the fail-closed behavior of the portfolio drawdown circuit breaker.

B2 fix: when Redis is unavailable (or the breaker cannot evaluate), the breaker
must treat the drawdown state as UNKNOWN and pause new BUY decisions locally
(fail-closed) instead of proceeding as if no drawdown exists. Risk-reducing
SELLs remain allowed (local pause is only enforced in the BUY path).
"""
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from src.utils import pause_utils
from src.utils.pause_utils import (
    set_local_pause,
    clear_local_pause,
    is_locally_paused,
    get_local_pause_reason,
)


@pytest.fixture(autouse=True)
def _reset_local_pause():
    clear_local_pause()
    yield
    clear_local_pause()


def _make_risk_manager(redis_unavailable=False):
    """Build a RiskManager via __new__ with a mocked engine."""
    from src.trading.components.risk_manager import RiskManager

    engine = MagicMock()
    engine.initial_balance = 1000.0
    engine.redis = MagicMock()
    engine.notifier = None
    if redis_unavailable:
        engine.redis.get = MagicMock(return_value=None)

    shared_state = MagicMock()
    shared_state._trade_history_lock = __import__("threading").Lock()
    shared_state.trade_history = []
    shared_state._realized_pnl_offset = 0.0
    shared_state.positions = {}
    shared_state._positions_lock = __import__("asyncio").Lock()

    rm = RiskManager.__new__(RiskManager)
    rm.engine = engine
    rm.shared_state = shared_state
    rm.event_bus = MagicMock()
    return rm


def test_local_pause_helpers_basic():
    assert not is_locally_paused()
    assert set_local_pause("redis_unavailable") is True
    assert is_locally_paused()
    assert get_local_pause_reason() == "redis_unavailable"
    # Second call does not transition and keeps original reason
    assert set_local_pause("other") is False
    assert get_local_pause_reason() == "redis_unavailable"
    clear_local_pause()
    assert not is_locally_paused()


def test_breaker_fails_closed_when_redis_unavailable():
    """Redis down => breaker state unknown => local fail-safe pause activates."""
    rm = _make_risk_manager(redis_unavailable=True)
    with patch("src.trading.components.risk_manager.is_redis_available", return_value=False):
        asyncio.run(rm.check_risk_management())
    assert is_locally_paused()
    assert get_local_pause_reason() == "redis_unavailable"


def test_breaker_exception_fails_closed():
    """If the breaker raises (Redis/DB error), pause is activated fail-closed."""
    rm = _make_risk_manager(redis_unavailable=False)
    with patch.object(
        rm, "_check_portfolio_drawdown_circuit_breaker",
        side_effect=ConnectionError("redis down"),
    ), patch("src.trading.components.risk_manager.is_redis_available", return_value=True):
        # check_risk_management calls the breaker directly (not awaited sub-call);
        # simulate via the breaker method raising inside asyncio.run
        with pytest.raises(ConnectionError):
            asyncio.run(rm._check_portfolio_drawdown_circuit_breaker())

    # Now the real fail-closed path: make the breaker catch the exception itself
    rm2 = _make_risk_manager(redis_unavailable=False)

    def _raise_inside(*a, **kw):
        raise ConnectionError("redis down")

    # Directly exercise the except-block behavior by calling the method with
    # redis.get raising inside the try body.
    rm2.engine.redis.get = MagicMock(side_effect=ConnectionError("redis down"))
    with patch("src.trading.components.risk_manager.is_redis_available", return_value=True):
        asyncio.run(rm2._check_portfolio_drawdown_circuit_breaker())
    assert is_locally_paused()
    assert get_local_pause_reason() == "drawdown_breaker_error"


def test_breaker_healthy_redis_no_pause():
    """With Redis healthy and normal drawdown, no local pause is set."""
    rm = _make_risk_manager(redis_unavailable=False)
    rm.engine.redis.get = MagicMock(return_value=None)  # no peak, no pause keys
    with patch("src.trading.components.risk_manager.is_redis_available", return_value=True), \
         patch("src.trading.components.risk_manager.get_peak_total_equity", return_value=None), \
         patch("src.trading.components.risk_manager.save_peak_total_equity", return_value=None), \
         patch("src.trading.components.risk_manager.settings") as mock_settings:
        mock_settings.PAUSE_FORCE_RESUME_MAX_DRAWDOWN_PCT = 20.0
        asyncio.run(rm._check_portfolio_drawdown_circuit_breaker())
    assert not is_locally_paused()


def test_buy_executor_blocked_by_local_pause():
    """The BUY executor must refuse BUYs while the local fail-safe pause is active."""
    from src.trading.components.buy_executor import BuyExecutor

    set_local_pause("redis_unavailable")

    engine = MagicMock()
    engine.redis = MagicMock()
    engine.redis.get = MagicMock(return_value=None)  # Redis-backed pause: unset
    shared_state = MagicMock()
    shared_state.positions = {}
    engine.shared_state = shared_state

    be = BuyExecutor.__new__(BuyExecutor)
    be.engine = engine
    be.shared_state = shared_state

    signal = MagicMock()
    signal.strategy_params = {}

    asyncio.run(
        be.execute_buy(
            symbol="ENI.MI/EUR",
            display_symbol="ENI",
            signal=signal,
            timeframe="1d",
            exit_reason=None,
            atr=None,
            balance={"EUR": 1000.0},
        )
    )
    # No quote fetch, no order execution occurred
    engine._market_data_manager._get_quotes_async.assert_not_called()


def test_notification_sent_once_on_fail_closed_activation():
    """Notifier fires once on first activation, not on subsequent checks."""
    rm = _make_risk_manager(redis_unavailable=True)
    rm.engine.notifier = MagicMock()
    rm.engine.notifier.send_notification = AsyncMock()
    with patch("src.trading.components.risk_manager.is_redis_available", return_value=False):
        asyncio.run(rm.check_risk_management())
        assert rm.engine.notifier.send_notification.await_count == 1
        # Second check: already paused locally, no new notification
        asyncio.run(rm.check_risk_management())
        assert rm.engine.notifier.send_notification.await_count == 1
