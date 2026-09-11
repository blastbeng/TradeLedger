"""Tests for the centralized LLM-provenance gate in PostDecisionManager.

Invariant enforced: every trading decision (BUY/SELL) must be made or reviewed
by an LLM. Signals without real LLM provenance are forced to HOLD with an
alert + metric. Risk-manager circuit-breaker SELLs (origin="risk_manager")
are the accepted exemption. Fallback HOLDs pass through without promotion.
"""
import asyncio
from unittest.mock import MagicMock, AsyncMock

import pytest

from src.strategies.base import Signal
from src.trading.components.post_decision_manager import PostDecisionManager


def _make_manager(monkeypatch=None):
    """Build a PostDecisionManager with mocked engine/event bus."""
    engine = MagicMock()
    engine.redis = MagicMock()
    engine.redis.incr = MagicMock(return_value=1)
    engine.redis.expire = MagicMock()
    engine.notifier = None
    event_bus = MagicMock()
    # Avoid double-subscription side effects
    event_bus.subscribe = MagicMock()
    manager = PostDecisionManager.__new__(PostDecisionManager)
    manager.engine = engine
    manager.shared_state = engine.shared_state
    manager.event_bus = event_bus
    return manager


def _make_context(signal, llm_provider="openai", llm_model="gpt-x", **kwargs):
    """Build a minimal DecisionContext-like object."""
    from src.trading.components.signal_processor import DecisionContext
    defaults = dict(
        symbol="ENI.MI/EUR",
        display_symbol="ENI",
        stock_name="ENI",
        assigned_tf="1d",
        tf_seconds=86400,
        ticker={},
        signal=signal,
        llm_provider=llm_provider,
        llm_model=llm_model,
        trading_paused=False,
        base_balance=1000.0,
        current_price=10.0,
        atr=0.2,
        rsi=50.0,
        macd=0.0,
        macd_signal=0.0,
        macd_hist=0.0,
        bb_upper=11.0,
        bb_middle=10.0,
        bb_lower=9.0,
        ema_9=10.0,
        ema_21=10.0,
        stochastic_k=50.0,
        stochastic_d=50.0,
        adx=25.0,
        plus_di=20.0,
        minus_di=15.0,
        obv=1000.0,
        mfi=50.0,
        cci=0.0,
        williams_r=-50.0,
        ichimoku=None,
        donchian_channels=None,
        parabolic_sar=9.5,
        keltner_channels=None,
        aggregate_sentiment=None,
        market_regime="neutral",
        min_stop_atr_mult=1.0,
        min_hold_time_mult=1.0,
        global_min_rr=1.0,
        max_hold_expired=False,
        stop_loss_triggered=False,
        take_profit_triggered=False,
        partial_tp_triggered=False,
        dust_sweep_triggered=False,
        strategy_model_type="mind",
        is_fallback=False,
    )
    defaults.update(kwargs)
    return DecisionContext(**defaults)


def _reviewed_buy():
    signal = Signal(action="BUY", confidence=0.8, reasoning="LLM says buy")
    signal.llm_provider = "openai"
    signal.llm_model = "gpt-x"
    signal.step2_reviewed = True
    return signal


def _unreviewed_sell():
    signal = Signal(action="SELL", confidence=1.0, reasoning="Max symbol tenure reached")
    signal.llm_provider = "openai"
    signal.llm_model = "gpt-x"
    signal.step2_reviewed = False
    return signal


def test_gate_reviewed_buy_has_real_provenance():
    manager = _make_manager()
    data = _make_context(_reviewed_buy())
    assert manager.check_llm_provenance(data) is None


def test_gate_blocks_unreviewed_sell_to_hold():
    manager = _make_manager()
    signal = _unreviewed_sell()
    data = _make_context(signal)

    violation = manager.check_llm_provenance(data)
    assert violation is not None

    asyncio.get_event_loop() if False else None
    asyncio.run(manager._enforce_llm_provenance(data, violation))

    assert data.signal.action == "HOLD"
    assert "provenance" in data.signal.reasoning.lower() or "review" in data.signal.reasoning.lower()
    # Symbol context preserved
    assert data.symbol == "ENI.MI/EUR"


def test_gate_blocks_unreviewed_sell_via_process_entry():
    """End-to-end through process_post_llm_decision: unreviewed SELL never reaches execution."""
    manager = _make_manager()
    manager._enforce_llm_provenance = AsyncMock()
    manager.log_and_notify_decision = AsyncMock()
    manager.handle_triggered_flags = AsyncMock(return_value=False)
    manager.check_trade_filters = AsyncMock(return_value=False)
    manager.check_sector_concentration = AsyncMock(return_value=False)
    manager.handle_entry_condition = AsyncMock(return_value=False)
    manager.event_bus.publish = AsyncMock()

    signal = _unreviewed_sell()
    data = _make_context(signal)

    asyncio.run(manager.process_post_llm_decision(data))

    manager._enforce_llm_provenance.assert_awaited_once()
    # Gate returns early — no execution, no flag handling
    manager.handle_triggered_flags.assert_not_awaited()
    manager.event_bus.publish.assert_not_awaited()


def test_gate_risk_manager_sell_exempt():
    """Risk-manager circuit-breaker SELLs (origin='risk_manager') pass the gate."""
    manager = _make_manager()
    signal = _unreviewed_sell()
    signal.origin = "risk_manager"
    data = _make_context(signal)
    assert manager.check_llm_provenance(data) is None


def test_gate_blocks_fallback_provider():
    manager = _make_manager()
    data = _make_context(_reviewed_buy(), llm_provider="fallback")
    violation = manager.check_llm_provenance(data)
    assert violation is not None and "llm_provider" in violation


def test_gate_blocks_default_hold_model():
    manager = _make_manager()
    data = _make_context(_reviewed_buy(), llm_model="default_hold")
    violation = manager.check_llm_provenance(data)
    assert violation is not None and "llm_model" in violation


def test_gate_tolerates_fallback_hold_without_promotion():
    """Fallback HOLD signals are non-executing: tolerated, never flagged for execution."""
    manager = _make_manager()
    signal = Signal(action="HOLD", confidence=0.0, reasoning="Step-2 unavailable")
    signal.llm_provider = "fallback"
    signal.llm_model = "default_hold"
    data = _make_context(signal, llm_provider="fallback", llm_model="default_hold")
    assert manager.check_llm_provenance(data) is None

    manager._enforce_llm_provenance = AsyncMock()
    manager.log_and_notify_decision = AsyncMock()
    manager.handle_triggered_flags = AsyncMock(return_value=True)
    asyncio.run(manager.process_post_llm_decision(data))
    # HOLD passes through unchanged, never promoted
    manager._enforce_llm_provenance.assert_not_awaited()
    assert data.signal.action == "HOLD"


def test_gate_blocks_buy_without_step2_review():
    manager = _make_manager()
    signal = _reviewed_buy()
    signal.step2_reviewed = None
    data = _make_context(signal)
    violation = manager.check_llm_provenance(data)
    assert violation is not None and "step2_reviewed" in violation


def test_step2_failure_path_not_marked_reviewed():
    """run_step2_llm_call failure paths must NOT set step2_reviewed=True."""
    from src.trading.components.backtest_manager import BacktestManager

    manager = BacktestManager.__new__(BacktestManager)
    manager.engine = MagicMock()
    manager.shared_state = MagicMock()
    manager.shared_state.positions = {}
    manager.event_bus = MagicMock()

    prelim = Signal(action="SELL", confidence=0.7, reasoning="preliminary sell")

    async def _cb_active(*a, **kw):
        return True

    import src.trading.components.backtest_manager as bm
    orig_cb = bm.is_llm_circuit_breaker_active
    bm.is_llm_circuit_breaker_active = _cb_active
    try:
        signal, provider, model, is_fallback = asyncio.run(
            manager.run_step2_llm_call(
                symbol="ENI.MI/EUR", assigned_tf="1d", preliminary_signal=prelim,
                backtest_results=[], combined_bt_summary="", ticker={},
                trading_paused=False, strategy_model_type="mind", effective_temp=0.2,
                llm_provider=None, llm_model=None,
            )
        )
    finally:
        bm.is_llm_circuit_breaker_active = orig_cb

    assert getattr(signal, "step2_reviewed", None) is not True


def test_risk_manager_helper_tags_origin():
    """The risk manager's circuit-breaker signal helper sets origin='risk_manager'."""
    from src.trading.components.risk_manager import RiskManager
    sig = RiskManager._circuit_breaker_sell_signal("Test")
    assert sig.action == "SELL"
    assert sig.origin == "risk_manager"
