"""Regression tests for Signal.from_dict fail-safe action handling (issue A2)."""
import logging

import pytest

from src.strategies.base import Signal


def test_from_dict_missing_action_defaults_to_hold():
    """A missing action must default to HOLD (non-executing), never BUY."""
    signal = Signal.from_dict({})
    assert signal.action == "HOLD"


def test_from_dict_explicit_action_preserved_and_uppercased():
    signal = Signal.from_dict({"action": "buy"})
    assert signal.action == "BUY"


def test_from_dict_invalid_action_maps_to_hold_with_warning(caplog):
    with caplog.at_level(logging.WARNING):
        signal = Signal.from_dict({"action": "FROB"})
    assert signal.action == "HOLD"
    assert any("invalid action" in r.message for r in caplog.records)


def test_from_dict_valid_actions_preserved():
    for action in ("BUY", "SELL", "HOLD"):
        assert Signal.from_dict({"action": action}).action == action
