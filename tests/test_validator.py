"""Unit tests for signal validation."""
import pytest
from src.strategies.base import Signal
from src.strategies.validator import validate_signal


def _make_buy_signal(**overrides):
    """Create a valid BUY signal with optional overrides."""
    params = {
        "stop_loss_pct": 0.05,
        "take_profit_pct": 0.10,
        "trailing_stop": False,
        "position_size_fraction": 0.25,
        "max_hold_time_seconds": 86400,
        "cooldown_after_loss_seconds": 3600,
    }
    params.update(overrides.pop("strategy_params", {}))
    return Signal(
        action="BUY",
        confidence=0.8,
        reasoning="Test signal",
        strategy_params=params,
        **overrides,
    )


class TestHoldSignal:
    def test_hold_passes_through(self):
        signal = Signal(action="HOLD", confidence=0.5, reasoning="No action")
        result = validate_signal(signal)
        assert result.action == "HOLD"
        assert result.confidence == 0.5


class TestBuyValidation:
    def test_valid_buy_signal(self):
        signal = _make_buy_signal()
        result = validate_signal(signal)
        assert result.action == "BUY"

    def test_missing_stop_loss(self):
        signal = _make_buy_signal(strategy_params={"stop_loss_pct": None})
        # Remove stop_loss_pct entirely
        del signal.strategy_params["stop_loss_pct"]
        result = validate_signal(signal)
        assert result.action == "HOLD"
        assert "stop_loss_pct" in result.reasoning

    def test_invalid_stop_loss_zero(self):
        signal = _make_buy_signal(strategy_params={"stop_loss_pct": 0.0})
        result = validate_signal(signal)
        assert result.action == "HOLD"

    def test_invalid_stop_loss_negative(self):
        signal = _make_buy_signal(strategy_params={"stop_loss_pct": -0.05})
        result = validate_signal(signal)
        assert result.action == "HOLD"

    def test_missing_take_profit(self):
        signal = _make_buy_signal()
        del signal.strategy_params["take_profit_pct"]
        result = validate_signal(signal)
        assert result.action == "HOLD"
        assert "take_profit_pct" in result.reasoning

    def test_tp_not_greater_than_sl(self):
        signal = _make_buy_signal(strategy_params={
            "stop_loss_pct": 0.10,
            "take_profit_pct": 0.05,
        })
        result = validate_signal(signal)
        assert result.action == "HOLD"
        assert "take_profit_pct must be greater than stop_loss_pct" in result.reasoning

    def test_missing_position_size(self):
        signal = _make_buy_signal()
        del signal.strategy_params["position_size_fraction"]
        result = validate_signal(signal)
        assert result.action == "HOLD"
        assert "position_size_fraction" in result.reasoning

    def test_invalid_position_size_zero(self):
        signal = _make_buy_signal(strategy_params={"position_size_fraction": 0.0})
        result = validate_signal(signal)
        assert result.action == "HOLD"

    def test_invalid_position_size_above_one(self):
        signal = _make_buy_signal(strategy_params={"position_size_fraction": 1.5})
        result = validate_signal(signal)
        assert result.action == "HOLD"

    def test_missing_max_hold_time(self):
        signal = _make_buy_signal()
        del signal.strategy_params["max_hold_time_seconds"]
        result = validate_signal(signal)
        assert result.action == "HOLD"
        assert "max_hold_time_seconds" in result.reasoning

    def test_missing_cooldown(self):
        signal = _make_buy_signal()
        del signal.strategy_params["cooldown_after_loss_seconds"]
        result = validate_signal(signal)
        assert result.action == "HOLD"
        assert "cooldown_after_loss_seconds" in result.reasoning

    def test_trailing_stop_requires_distance(self):
        signal = _make_buy_signal(strategy_params={
            "trailing_stop": True,
            # Missing trailing_stop_distance_pct
        })
        result = validate_signal(signal)
        assert result.action == "HOLD"
        assert "trailing_stop_distance_pct" in result.reasoning

    def test_trailing_stop_distance_must_be_less_than_sl(self):
        signal = _make_buy_signal(strategy_params={
            "trailing_stop": True,
            "trailing_stop_distance_pct": 0.10,  # >= stop_loss_pct (0.05)
            "stop_loss_pct": 0.05,
        })
        result = validate_signal(signal)
        assert result.action == "HOLD"
        assert "trailing_stop_distance_pct must be less than stop_loss_pct" in result.reasoning


class TestATRBasedStop:
    def test_atr_multiple_missing_pct_fallback(self):
        """When using atr_multiple method, stop_loss_pct is still required as fallback."""
        signal = _make_buy_signal(strategy_params={
            "stop_loss_method": "atr_multiple",
            "stop_loss_atr_multiple": 2.0,
        })
        del signal.strategy_params["stop_loss_pct"]
        result = validate_signal(signal)
        assert result.action == "HOLD"
        assert "stop_loss_pct" in result.reasoning

    def test_atr_multiple_valid(self):
        signal = _make_buy_signal(strategy_params={
            "stop_loss_method": "atr_multiple",
            "stop_loss_atr_multiple": 2.0,
            "stop_loss_pct": 0.04,  # fallback
        })
        result = validate_signal(signal)
        assert result.action == "BUY"

    def test_atr_multiple_invalid_value(self):
        signal = _make_buy_signal(strategy_params={
            "stop_loss_method": "atr_multiple",
            "stop_loss_atr_multiple": -1.0,
            "stop_loss_pct": 0.04,
        })
        result = validate_signal(signal)
        assert result.action == "HOLD"

    def test_invalid_stop_loss_method(self):
        signal = _make_buy_signal(strategy_params={
            "stop_loss_method": "invalid",
        })
        result = validate_signal(signal)
        assert result.action == "HOLD"
        assert "Invalid stop_loss_method" in result.reasoning


class TestMinHoldTime:
    def test_hold_time_too_short(self):
        signal = _make_buy_signal(strategy_params={
            "max_hold_time_seconds": 60,  # 1 minute
        })
        result = validate_signal(signal, timeframe_seconds=3600)  # 1h candles
        assert result.action == "HOLD"
        assert "too short" in result.reasoning

    def test_hold_time_sufficient(self):
        signal = _make_buy_signal(strategy_params={
            "max_hold_time_seconds": 7200,  # 2 hours
        })
        result = validate_signal(signal, timeframe_seconds=3600)
        assert result.action == "BUY"


class TestRiskRewardRatio:
    def test_ratio_below_minimum(self):
        signal = _make_buy_signal(strategy_params={
            "stop_loss_pct": 0.10,
            "take_profit_pct": 0.12,  # ratio = 1.2
        })
        result = validate_signal(signal, global_min_risk_reward_ratio=1.5)
        assert result.action == "HOLD"
        assert "Risk/reward ratio" in result.reasoning

    def test_ratio_above_minimum(self):
        signal = _make_buy_signal(strategy_params={
            "stop_loss_pct": 0.05,
            "take_profit_pct": 0.10,  # ratio = 2.0
        })
        result = validate_signal(signal, global_min_risk_reward_ratio=1.5)
        assert result.action == "BUY"
