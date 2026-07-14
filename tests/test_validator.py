import pytest

from src.strategies.base import Signal
from src.strategies.validator import validate_signal


def test_validate_signal_valid_buy():
    signal = Signal(
        action="BUY",
        price=100.0,
        sl=95.0,
        tp=110.0,
        confidence=0.8,
    )
    result = validate_signal(signal, price=100.0, atr=2.0, timeframe_seconds=3600)
    assert result.action == "BUY"
    assert result.sl < result.price < result.tp


def test_validate_signal_invalid_action():
    signal = Signal(
        action="INVALID",
    )
    result = validate_signal(signal)
    assert result.action == "HOLD"
