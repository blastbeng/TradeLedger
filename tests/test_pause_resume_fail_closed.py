"""Fail-closed tests for the pause/resume LLM decision path.

When LLM pause-decision calls fail consecutively, the bot must STAY PAUSED
(never force-resume). This is the invariant: all trading decisions must be
LLM-reviewed; when in doubt, fail-closed (paused).
"""
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.trading.components.pause_resume_manager import PauseResumeManager


def _make_manager():
    """Build a PauseResumeManager via __new__ with a mocked engine."""
    from src.trading.components.signal_processor import SignalProcessor

    engine = MagicMock()
    engine._symbol_reeval_lock = asyncio.Lock()
    engine.redis = MagicMock()
    engine.redis.get = MagicMock(return_value=None)
    engine.redis.incr = MagicMock(return_value=3)
    engine.redis.expire = MagicMock(return_value=True)
    engine.redis.delete = MagicMock(return_value=True)
    engine.notifier = MagicMock()
    engine.notifier.send_notification = AsyncMock()
    engine.config_service = MagicMock()
    engine.config_service.get_config = AsyncMock(return_value=None)
    engine._set_global_risk_multiplier = AsyncMock()
    engine._is_market_open = AsyncMock(return_value=True)
    engine._reeval_trigger = MagicMock()
    engine._market_data_manager = MagicMock()
    engine._market_data_manager._get_quotes_async = AsyncMock(return_value={})

    sp = MagicMock()
    sp.engine = engine
    sp.event_bus = MagicMock()
    sp.event_bus.request = AsyncMock(
        return_value={"equity_curve": {"daily_pnl": 0.0, "total_pnl": 0.0, "consecutive_losses": 0, "drawdown_pct": 0.0}}
    )
    engine.event_bus = sp.event_bus
    sp.model_tier_manager = MagicMock()
    sp.model_tier_manager.compute_prompt_complexity = MagicMock(return_value=0.1)
    sp.model_tier_manager._get_effective_temperature = MagicMock(return_value=0.2)
    sp.model_tier_manager._compute_reasoning_effort = MagicMock(return_value=0)

    shared_state = MagicMock()
    shared_state._market_breadth = None
    shared_state.positions = {}
    engine.shared_state = shared_state

    m = PauseResumeManager.__new__(PauseResumeManager)
    m.sp = sp
    m.engine = engine
    m.shared_state = shared_state
    m.event_bus = sp.event_bus
    return m, engine


def _redis_values(engine, source=b"llm"):
    def get(key):
        return {
            "trading:paused": b"1",
            "trading:pause_source": source,
            "trading:pause_reason": b"test",
            "market:breadth:full": None,
            "trading:pause:keep_count": None,
            "trading:last_auto_resume": None,
            "trading:llm_pause_time": None,
        }.get(key)
    engine.redis.get = MagicMock(side_effect=get)


@pytest.mark.asyncio
async def test_llm_failure_stays_paused_fail_closed():
    """3 consecutive LLM pause-decision failures must NOT resume trading."""
    m, engine = _make_manager()
    _redis_values(engine)

    with patch("src.trading.components.pause_resume_manager.get_cached_llm_response",
               side_effect=RuntimeError("LLM down")), \
         patch("src.trading.components.pause_resume_manager.compute_market_hash",
               return_value="h"), \
         patch("src.utils.pause_utils.clear_trading_pause_keys") as clear_keys:
        await m.check_pause_resume_decision()

    # Trading must remain paused: pause keys must not be cleared
    clear_keys.assert_not_called()
    # Risk multiplier must not be touched and no re-eval trigger
    engine._set_global_risk_multiplier.assert_not_called()
    engine._reeval_trigger.set.assert_not_called()
    # A fail-closed ERROR must be logged
    notifier_msgs = [c.args[0] for c in engine.notifier.send_notification.await_args_list]
    assert any("fail-closed" in msg for msg in notifier_msgs)


@pytest.mark.asyncio
async def test_llm_resume_decision_still_works():
    """Successful LLM decision to resume must still resume (behavior unchanged)."""
    m, engine = _make_manager()
    _redis_values(engine)

    response = {"response": json.dumps({"resume_trading": True, "reason": "ok"}),
                "provider": "p", "model": "m"}
    with patch("src.trading.components.pause_resume_manager.get_cached_llm_response",
               return_value=response), \
         patch("src.trading.components.pause_resume_manager.compute_market_hash",
               return_value="h"), \
         patch("src.utils.pause_utils.clear_trading_pause_keys") as clear_keys:
        await m.check_pause_resume_decision()

    clear_keys.assert_called_once()
    engine._reeval_trigger.set.assert_called_once()
