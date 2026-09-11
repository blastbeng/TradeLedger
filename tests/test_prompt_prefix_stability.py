"""Prompt prefix stability tests (provider prompt caching).

Invariant: the system message must be fully static (identical across calls
for the same task_type), and per-call volatile content must appear only at
the END of the user message. Static decision rules / JSON schema must come
BEFORE volatile market data in the Step-2 final decision prompt.
"""
import pytest

from src.llm.backtest_prompts import build_final_decision_prompt, build_final_decision_messages
from src.llm.system_prompt import build_system_prompt, get_past_mistakes_block


class _FakeRedis:
    def __init__(self, value=None):
        self._value = value

    def get(self, key):
        return self._value


def test_system_prompt_stable_across_volatile_redis_content(monkeypatch):
    """Past-mistakes content must not leak into the system prompt."""
    monkeypatch.setattr("src.llm.system_prompt.get_redis_client",
                        lambda: _FakeRedis(b"mistake analysis v1"))
    p1 = build_system_prompt(task_type="trading")
    monkeypatch.setattr("src.llm.system_prompt.get_redis_client",
                        lambda: _FakeRedis(b"mistake analysis v2 COMPLETELY DIFFERENT"))
    p2 = build_system_prompt(task_type="trading")
    assert p1 == p2, "system prompt must be identical regardless of Redis mistake-analysis content"
    assert "mistake analysis v1" not in p1 and "Past Mistakes Analysis" not in p1


def test_past_mistakes_block_returns_volatile_content(monkeypatch):
    monkeypatch.setattr("src.llm.system_prompt.get_redis_client",
                        lambda: _FakeRedis(b"avoid chasing breakouts"))
    block = get_past_mistakes_block()
    assert "avoid chasing breakouts" in block
    # Empty Redis -> empty block, no exception
    monkeypatch.setattr("src.llm.system_prompt.get_redis_client", lambda: _FakeRedis(None))
    assert get_past_mistakes_block() == ""


def test_final_decision_messages_static_prefix_stable(monkeypatch):
    """System message + leading user-message rules must be identical across calls."""
    monkeypatch.setattr("src.llm.system_prompt.get_redis_client", lambda: _FakeRedis(None))
    ticker_a = {"last": 10.0}
    ticker_b = {"last": 999.99}
    msgs_a = build_final_decision_messages(
        symbol="ISP.MI", ticker=ticker_a,
        preliminary_decision={"action": "BUY", "confidence": 0.6, "reasoning": "rsi", "strategy_params": {}, "timeframe": "1d"},
        backtest_results=[], base_currency="EUR",
    )
    msgs_b = build_final_decision_messages(
        symbol="ENEL.MI", ticker=ticker_b,
        preliminary_decision={"action": "HOLD", "confidence": 0.1, "reasoning": "flat", "strategy_params": {}, "timeframe": "1w"},
        backtest_results=[], base_currency="EUR",
    )
    assert msgs_a[0] == msgs_b[0], "system message must be byte-identical across calls"
    # Static decision-rule header (before volatile data) must match
    head_a = msgs_a[1]["content"].split("Symbol:")[0]
    head_b = msgs_b[1]["content"].split("Symbol:")[0]
    assert head_a == head_b, "static rules must precede all volatile data"
    assert "ISP.MI" in msgs_a[1]["content"] and "999.99" in msgs_b[1]["content"]


def test_final_decision_prompt_keeps_required_information():
    """Same information is present; only ordering changed."""
    monkeypatch_guard = None
    bt = {"variant_params": {"stop_loss_pct": 0.02, "take_profit_pct": 0.05},
          "summary": "ok", "stats": {"total_trades": 10, "win_rate": 0.6,
          "total_pnl_pct": 0.03, "max_drawdown_pct": 0.02}}
    prompt = build_final_decision_prompt(
        symbol="ISP.MI", ticker={"last": 10.0},
        preliminary_decision={"action": "BUY", "confidence": 0.5, "reasoning": "r", "strategy_params": {}, "timeframe": "1d"},
        backtest_results=[bt], base_currency="EUR",
    )
    for token in ("ISP.MI", "Step 2", "Return JSON", "action: BUY|SELL|HOLD",
                  "positive expectancy", "10.00", "Preliminary Action"):
        assert token in prompt, f"missing token: {token}"
    # Static rules must come before the volatile symbol data
    assert prompt.index("Return JSON") < prompt.index("ISP.MI") or \
           prompt.index("action: BUY|SELL|HOLD") < prompt.index("**Local Python Backtest Results")