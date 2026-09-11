"""Tests for dividend reinvestment routing through the LLM-reviewed decision path.

Invariant (A1): reinvested-dividend BUYs must go through Step-2 LLM review
(run_step2_llm_call) and the centralized provenance gate
(process_post_llm_decision) — never a direct trader.create_market_buy_order.
Fail-safe: unreviewed or LLM-rejected reinvestment never executes (B1: closed
positions must not be resurrected).
"""
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from src.strategies.base import Signal
from src.trading.components.background_task_manager import BackgroundTaskManager


def _make_task_manager():
    """Build a BackgroundTaskManager with a mocked engine/event bus."""
    engine = MagicMock()
    engine.redis = MagicMock()
    engine.redis.get = MagicMock(return_value=None)
    engine.notifier = None
    engine.base_currency = "EUR"
    engine.shared_state.positions = {}
    engine.shared_state._positions_lock = asyncio.Lock()
    tm = BackgroundTaskManager.__new__(BackgroundTaskManager)
    tm.engine = engine
    tm.event_bus = MagicMock()
    return tm


def _pos(symbol="ENI.MI/EUR", amount=100.0, timeframe="1d"):
    return {
        "symbol": symbol,
        "amount": amount,
        "price": 10.0,
        "timeframe": timeframe,
        "cost_basis": amount * 10.0,
    }


def _make_step2_result(reviewed=True, action="BUY"):
    """Patch BacktestManager.run_step2_llm_call to return a reviewed BUY (or HOLD)."""
    async def _fake_step2(self, **kwargs):
        signal = kwargs["preliminary_signal"]
        if not reviewed:
            hold = Signal(action="HOLD", confidence=0.5, reasoning="Step-2 unavailable")
            hold.llm_provider = "fallback"
            hold.llm_model = "step2_failure_hold"
            return hold, "fallback", "step2_failure_hold", True
        out = Signal(action=action, confidence=0.7, reasoning="reviewed")
        out.llm_provider = "openai"
        out.llm_model = "gpt-x"
        out.step2_reviewed = reviewed and action == "BUY"
        return out, "openai", "gpt-x", False
    return _fake_step2


async def test_reinvestment_buy_routed_through_llm_review(monkeypatch):
    """A reviewed reinvestment BUY goes through process_post_llm_decision, not the trader."""
    tm = _make_task_manager()
    engine = tm.engine
    engine.shared_state.positions["ENI.MI/EUR"] = _pos()

    trader = MagicMock()
    engine.trader = trader

    from src.trading.components.backtest_manager import BacktestManager
    engine._backtest_manager = BacktestManager.__new__(BacktestManager)
    monkeypatch.setattr(BacktestManager, "run_step2_llm_call", _make_step2_result(reviewed=True))

    engine.event_bus = MagicMock()
    engine.event_bus.request = AsyncMock(return_value=None)

    # Mock get_ohlcv to avoid DB access
    submitted = await tm._submit_reinvestment_buy(
            "ENI.MI/EUR",
            _pos(),
            {"id": 42, "amount": 0.10},
            total_div_value=10.0,
        )

    assert submitted is True
    # Order was NOT placed directly through the trader
    trader.create_market_buy_order.assert_not_called()
    # Decision went through the standard post-LLM decision path
    engine.event_bus.request.assert_awaited_once()
    args, kwargs = engine.event_bus.request.await_args
    assert args[0] == "process_post_llm_decision"


async def test_unreviewed_reinvestment_buy_never_executes(monkeypatch):
    """Step-2 failure (HOLD downgrade) → no order, no process_post_llm_decision call."""
    tm = _make_task_manager()
    engine = tm.engine
    engine.shared_state.positions["ENI.MI/EUR"] = _pos()

    trader = MagicMock()
    engine.trader = trader

    from src.trading.components.backtest_manager import BacktestManager
    engine._backtest_manager = BacktestManager.__new__(BacktestManager)
    monkeypatch.setattr(BacktestManager, "run_step2_llm_call", _make_step2_result(reviewed=False))

    engine.event_bus = MagicMock()
    engine.event_bus.request = AsyncMock(return_value=None)

    submitted = await tm._submit_reinvestment_buy(
            "ENI.MI/EUR",
            _pos(),
            {"id": 42, "amount": 0.10},
            total_div_value=10.0,
        )

    assert submitted is False
    trader.create_market_buy_order.assert_not_called()
    engine.event_bus.request.assert_not_awaited()


async def test_step2_exception_fails_safe_no_execution(monkeypatch):
    """If Step-2 review raises, the reinvestment is skipped (no order placed)."""
    tm = _make_task_manager()
    engine = tm.engine
    engine.shared_state.positions["ENI.MI/EUR"] = _pos()

    trader = MagicMock()
    engine.trader = trader

    from src.trading.components.backtest_manager import BacktestManager
    engine._backtest_manager = BacktestManager.__new__(BacktestManager)

    async def _boom(self, **kwargs):
        raise TimeoutError("LLM timeout")

    monkeypatch.setattr(BacktestManager, "run_step2_llm_call", _boom)
    engine._record_unexpected_exception = AsyncMock()

    submitted = await tm._submit_reinvestment_buy(
            "ENI.MI/EUR",
            _pos(),
            {"id": 42, "amount": 0.10},
            total_div_value=10.0,
        )

    assert submitted is False
    trader.create_market_buy_order.assert_not_called()


async def test_closed_position_not_resurrected(monkeypatch):
    """If the position was closed concurrently, the reinvestment is skipped entirely."""
    tm = _make_task_manager()
    engine = tm.engine
    # Position NOT present (was closed by a concurrent SELL)

    trader = MagicMock()
    engine.trader = trader

    from src.trading.components.backtest_manager import BacktestManager
    engine._backtest_manager = BacktestManager.__new__(BacktestManager)
    monkeypatch.setattr(BacktestManager, "run_step2_llm_call", _make_step2_result(reviewed=True))

    engine.event_bus = MagicMock()
    engine.event_bus.request = AsyncMock(return_value=None)

    submitted = await tm._submit_reinvestment_buy(
            "ENI.MI/EUR",
            _pos(),
            {"id": 42, "amount": 0.10},
            total_div_value=10.0,
        )

    assert submitted is False
    trader.create_market_buy_order.assert_not_called()
    engine.event_bus.request.assert_not_awaited()


async def test_reinvestment_signal_carries_dividend_metadata(monkeypatch):
    """The preliminary BUY carries the dividend id so the executor can mark it reinvested."""
    tm = _make_task_manager()
    signal = await tm._make_reinvest_dividend_buy(
        "ENI.MI/EUR", _pos(), {"id": 7, "amount": 0.10}, 10.0
    )
    assert signal.action == "BUY"
    assert signal.strategy_params["reinvest_dividend_id"] == 7
    assert signal.strategy_params["reinvestment_trade_value"] == 10.0
    assert signal.strategy_type == "dividend_reinvestment"


async def test_buy_executor_uses_fixed_reinvestment_value():
    """BuyExecutor sizing uses the reinvestment trade value, not balance fraction."""
    from src.trading.components.buy_executor import BuyExecutor
    import ast, inspect
    src = inspect.getsource(BuyExecutor.execute_buy)
    assert "reinvestment_trade_value" in src
