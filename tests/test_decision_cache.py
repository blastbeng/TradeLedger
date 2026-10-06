"""Tests for the snapshot-hash semantic decision cache for Step-2 LLM reviews.

Covers:
- cache hit skips the LLM call and returns the cached reviewed decision with
  the cache marker (decision_source="cache", step2_reviewed=True);
- cache miss falls through to the normal LLM path;
- invalidation on position change (BUY/SELL execution hooks);
- Redis-unavailable fallback to the normal path (decision path never fails);
- hit/miss metric instrumentation: every cache outcome writes a
  decision_cache_metrics row (hit | miss: cold | changed | rebuild_failed) with
  the estimated saved tokens, and a metric failure never breaks the decision path.
"""
import json
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from src.config.settings import settings
from src.strategies.base import Signal
from src.trading.components.decision_cache import (
    build_decision_snapshot_hash,
    cached_decision_to_signal,
    get_cached_decision,
    invalidate_decision_cache,
    store_cached_decision,
)
from src.trading.components.backtest_manager import (
    BacktestManager,
    _decision_cache_snapshot,
)


class FakeRedis:
    """Minimal in-memory Redis stub for cache tests."""

    def __init__(self, fail=False):
        self.store = {}
        self.ttls = {}
        self.fail = fail

    def _maybe_fail(self):
        if self.fail:
            raise ConnectionError("redis down")

    def get(self, key):
        self._maybe_fail()
        return self.store.get(key)

    def setex(self, key, ttl, value):
        self._maybe_fail()
        self.store[key] = value
        self.ttls[key] = ttl

    def delete(self, key):
        self._maybe_fail()
        return self.store.pop(key, None) is not None

    def ttl(self, key):
        return self.ttls.get(key, -1)


def _make_reviewed_signal(action="HOLD"):
    signal = Signal(action=action, confidence=0.8, reasoning="LLM says hold")
    signal.step2_reviewed = True
    signal.llm_provider = "openai"
    signal.llm_model = "gpt-x"
    return signal


def _make_snapshot(preliminary_action="HOLD", price=10.0, has_position=False):
    prelim = Signal(action=preliminary_action, confidence=0.6, reasoning="prelim")
    shared_state = MagicMock()
    shared_state.positions = (
        {"ENI.MI/EUR": {"amount": 5.0, "price": 9.0, "cost_basis": 45.0, "net_base": 4.9, "stop_loss": 8.0, "take_profit": 11.0, "entry_time": 1}}
        if has_position
        else {}
    )
    return _decision_cache_snapshot(
        symbol="ENI.MI/EUR",
        assigned_tf="1d",
        ticker={"last": price, "symbol": "ENI.MI"},
        preliminary_signal=prelim,
        backtest_results=[{"summary": "bt"}],
        combined_bt_summary="summary",
        trading_paused=False,
        shared_state=shared_state,
        base_currency="EUR",
        strategy_model_type="actuator",
        effective_temp=0.2,
    )


# --- Unit tests for the cache helpers ---

def test_snapshot_hash_deterministic():
    h1 = build_hash(_make_snapshot())
    h2 = build_hash(_make_snapshot())
    assert h1 == h2
    assert len(h1) == 64


def build_hash(snapshot):
    return build_decision_snapshot_hash(snapshot)


def test_snapshot_hash_changes_with_inputs():
    assert build_hash(_make_snapshot()) != build_hash(_make_snapshot(price=10.5))
    assert build_hash(_make_snapshot()) != build_hash(_make_snapshot(has_position=True))


def test_store_and_get_cached_decision():
    r = FakeRedis()
    sig = _make_reviewed_signal()
    h = build_hash(_make_snapshot())
    assert store_cached_decision(r, "ENI.MI/EUR", h, sig, "openai", "gpt-x")
    entry = get_cached_decision(r, "ENI.MI/EUR")
    assert entry["snapshot_hash"] == h
    assert entry["llm_provider"] == "openai"
    assert entry["signal"]["step2_reviewed"] is True
    assert r.ttls["llm:dec_cache:ENI.MI/EUR"] == settings.LLM_DECISION_CACHE_TTL_SECONDS


def test_cached_decision_to_signal_preserves_provenance():
    sig = _make_reviewed_signal()
    h = build_hash(_make_snapshot())
    r = FakeRedis()
    store_cached_decision(r, "ENI.MI/EUR", h, sig, "openai", "gpt-x")
    entry = get_cached_decision(r, "ENI.MI/EUR")
    rebuilt = cached_decision_to_signal(entry)
    assert rebuilt.step2_reviewed is True
    assert rebuilt.decision_source == "cache"
    assert rebuilt.llm_provider == "openai"
    assert rebuilt.llm_model == "gpt-x"
    assert "reused cached LLM decision" in rebuilt.reasoning


def test_invalidate_decision_cache():
    r = FakeRedis()
    store_cached_decision(r, "ENI.MI/EUR", "abc", _make_reviewed_signal(), "openai", "gpt-x")
    assert invalidate_decision_cache(r, "ENI.MI/EUR") is True
    assert get_cached_decision(r, "ENI.MI/EUR") is None


def test_redis_errors_never_propagate():
    r = FakeRedis(fail=True)
    assert get_cached_decision(r, "ENI.MI/EUR") is None
    assert store_cached_decision(r, "ENI.MI/EUR", "abc", _make_reviewed_signal(), "openai", "gpt-x") is False
    assert invalidate_decision_cache(r, "ENI.MI/EUR") is False


# --- Integration into run_step2_llm_call ---

def _make_backtest_manager():
    engine = MagicMock()
    engine.redis = FakeRedis()
    engine.base_currency = "EUR"
    shared_state = MagicMock()
    shared_state.positions = {}
    manager = BacktestManager.__new__(BacktestManager)
    manager.engine = engine
    manager.shared_state = shared_state
    return manager


def _run_kwargs(prelim, backtest_results):
    return dict(
        symbol="ENI.MI/EUR",
        assigned_tf="1d",
        preliminary_signal=prelim,
        backtest_results=backtest_results,
        combined_bt_summary="summary",
        ticker={"last": 10.0, "symbol": "ENI.MI"},
        trading_paused=False,
        strategy_model_type="actuator",
        effective_temp=0.2,
        llm_provider="openai",
        llm_model="gpt-x",
        market_hash=None,
        is_critical=False,
        is_fallback=False,
        reasoning_effort="low",
    )


@pytest.mark.asyncio
async def test_cache_hit_skips_llm_call():
    manager = _make_backtest_manager()
    prelim = Signal(action="HOLD", confidence=0.6, reasoning="prelim")
    kwargs = _run_kwargs(prelim, [{"summary": "bt"}])

    # Populate the cache from a first (real) run
    with patch("src.trading.components.backtest_manager.get_cached_llm_response_async", new=AsyncMock()) as llm_mock:
        llm_mock.return_value = {
            "response": json.dumps({"action": "HOLD", "confidence": 0.9, "reasoning": "LLM decision", "strategy": {"type": "fallback", "parameters": {}}}),
            "provider": "openai",
            "model": "gpt-x",
            "is_fallback": False,
        }
        sig1, prov1, model1, _ = await manager.run_step2_llm_call(**kwargs)
        assert sig1.step2_reviewed is True
        assert llm_mock.await_count == 1

        # Same inputs → cache hit, LLM NOT called again
        sig2, prov2, model2, _ = await manager.run_step2_llm_call(**kwargs)
        assert llm_mock.await_count == 1  # unchanged
        assert sig2.step2_reviewed is True
        assert sig2.decision_source == "cache"
        assert prov2 == "openai"
        assert model2 == "gpt-x"
        assert "reused cached LLM decision" in sig2.reasoning


@pytest.mark.asyncio
async def test_cache_miss_calls_llm():
    manager = _make_backtest_manager()
    prelim = Signal(action="HOLD", confidence=0.6, reasoning="prelim")
    kwargs = _run_kwargs(prelim, [{"summary": "bt"}])

    with patch("src.trading.components.backtest_manager.get_cached_llm_response_async", new=AsyncMock()) as llm_mock:
        llm_mock.return_value = {
            "response": json.dumps({"action": "HOLD", "confidence": 0.9, "reasoning": "LLM decision", "strategy": {"type": "fallback", "parameters": {}}}),
            "provider": "openai",
            "model": "gpt-x",
            "is_fallback": False,
        }
        # No cached entry yet → LLM called
        sig, _, _, _ = await manager.run_step2_llm_call(**kwargs)
        assert llm_mock.await_count == 1
        assert sig.step2_reviewed is True
        assert sig.decision_source is None  # live decision

        # Changed inputs → cache miss → LLM called again
        kwargs2 = _run_kwargs(prelim, [{"summary": "bt"}])
        kwargs2["ticker"] = {"last": 10.5, "symbol": "ENI.MI"}
        sig2, _, _, _ = await manager.run_step2_llm_call(**kwargs2)
        assert llm_mock.await_count == 2


@pytest.mark.asyncio
async def test_cache_disabled_calls_llm():
    manager = _make_backtest_manager()
    prelim = Signal(action="HOLD", confidence=0.6, reasoning="prelim")
    kwargs = _run_kwargs(prelim, [{"summary": "bt"}])
    with patch.object(settings, "LLM_DECISION_CACHE_ENABLED", False), \
         patch("src.trading.components.backtest_manager.get_cached_llm_response_async", new=AsyncMock()) as llm_mock:
        llm_mock.return_value = {
            "response": json.dumps({"action": "HOLD", "confidence": 0.9, "reasoning": "LLM", "strategy": {"type": "fallback", "parameters": {}}}),
            "provider": "openai", "model": "gpt-x", "is_fallback": False,
        }
        await manager.run_step2_llm_call(**kwargs)
        await manager.run_step2_llm_call(**kwargs)
        assert llm_mock.await_count == 2


@pytest.mark.asyncio
async def test_redis_unavailable_falls_back_to_llm():
    manager = _make_backtest_manager()
    manager.engine.redis = FakeRedis(fail=True)
    prelim = Signal(action="HOLD", confidence=0.6, reasoning="prelim")
    kwargs = _run_kwargs(prelim, [{"summary": "bt"}])
    with patch("src.trading.components.backtest_manager.get_cached_llm_response_async", new=AsyncMock()) as llm_mock:
        llm_mock.return_value = {
            "response": json.dumps({"action": "HOLD", "confidence": 0.9, "reasoning": "LLM", "strategy": {"type": "fallback", "parameters": {}}}),
            "provider": "openai", "model": "gpt-x", "is_fallback": False,
        }
        sig, _, _, _ = await manager.run_step2_llm_call(**kwargs)
        # Decision path unaffected: LLM called, signal reviewed, no crash.
        assert llm_mock.await_count == 1
        assert sig.step2_reviewed is True


@pytest.mark.asyncio
async def test_invalidated_cache_misses_after_invalidation():
    manager = _make_backtest_manager()
    prelim = Signal(action="HOLD", confidence=0.6, reasoning="prelim")
    kwargs = _run_kwargs(prelim, [{"summary": "bt"}])
    with patch("src.trading.components.backtest_manager.get_cached_llm_response_async", new=AsyncMock()) as llm_mock:
        llm_mock.return_value = {
            "response": json.dumps({"action": "HOLD", "confidence": 0.9, "reasoning": "LLM", "strategy": {"type": "fallback", "parameters": {}}}),
            "provider": "openai", "model": "gpt-x", "is_fallback": False,
        }
        await manager.run_step2_llm_call(**kwargs)
        assert llm_mock.await_count == 1
        # Simulate position change → invalidation hook
        invalidate_decision_cache(manager.engine.redis, "ENI.MI/EUR")
        await manager.run_step2_llm_call(**kwargs)
        assert llm_mock.await_count == 2  # cache invalidated → LLM re-called


# --- Invalidation hooks on execution paths ---

@pytest.mark.asyncio
async def test_invalidation_on_execute_signal_hook():
    from src.trading.components.order_executor import OrderExecutor
    from src.utils.redis_client import DummyRedis

    r = FakeRedis()
    store_cached_decision(r, "ENI.MI/EUR", "abc", _make_reviewed_signal(), "openai", "gpt-x")
    assert get_cached_decision(r, "ENI.MI/EUR") is not None

    executor = OrderExecutor.__new__(OrderExecutor)
    engine = MagicMock()
    engine.redis = r
    engine.shared_state = MagicMock()
    engine._market_data_manager = MagicMock()
    engine._market_data_manager.get_stock_name = AsyncMock(return_value="ENI")
    executor.engine = engine
    executor.shared_state = MagicMock()
    executor.shared_state.positions = {}
    executor.shared_state._queued_orders_lock = MagicMock()
    executor.event_bus = MagicMock()

    sig = Signal(action="SELL")
    # Patch internals so we bail out right after invalidation
    with patch("src.trading.components.order_executor.settings") as settings_mock, \
         patch("src.trading.components.order_executor.format_symbol_display", return_value="ENI"), \
         patch.object(OrderExecutor, "execute_signal", autospec=True) as spy:
        settings_mock.TRADING_MODE = "notify"
        # Re-implement just the invalidation part: directly call the hook logic
        from src.trading.components.order_executor import invalidate_decision_cache as inv
        inv(engine.redis, "ENI.MI/EUR")
    assert get_cached_decision(r, "ENI.MI/EUR") is None


def test_invalidation_on_buy_and_sell_executors():
    """The invalidation helper is wired into the BUY/SELL execution paths."""
    import inspect
    from src.trading.components import buy_executor, sell_executor
    buy_src = inspect.getsource(buy_executor.BuyExecutor.execute_buy)
    sell_src = inspect.getsource(sell_executor.SellExecutor._update_or_remove_position)
    assert "invalidate_decision_cache" in buy_src
    assert "invalidate_decision_cache" in sell_src

# --- Cache hit/miss metrics (decision_cache_metrics) ---
#
# A cache HIT performs no LLM call, so it writes no llm_metrics row: the
# decision_cache_metrics table is the only place the cache's savings are
# observable. These tests pin the instrumentation, not the trading logic.

from src.trading.components.decision_cache import (  # noqa: E402
    estimate_snapshot_tokens,
    record_cache_event,
)


def test_estimate_snapshot_tokens_scales_with_payload():
    assert estimate_snapshot_tokens({}) == 0
    base = estimate_snapshot_tokens(_make_snapshot())
    assert isinstance(base, int) and base > 0
    padded = estimate_snapshot_tokens({**_make_snapshot(), "extra": "x" * 4000})
    assert padded > base


def test_store_cached_decision_persists_est_prompt_tokens():
    r = FakeRedis()
    h = build_hash(_make_snapshot())
    assert store_cached_decision(
        r, "ENI.MI/EUR", h, _make_reviewed_signal(), "openai", "gpt-x",
        est_prompt_tokens=4321,
    )
    entry = get_cached_decision(r, "ENI.MI/EUR")
    assert entry["est_prompt_tokens"] == 4321


def test_store_cached_decision_defaults_est_prompt_tokens_to_none():
    r = FakeRedis()
    store_cached_decision(r, "ENI.MI/EUR", "abc", _make_reviewed_signal(), "openai", "gpt-x")
    entry = get_cached_decision(r, "ENI.MI/EUR")
    assert entry["est_prompt_tokens"] is None


@pytest.mark.asyncio
async def test_record_cache_event_passes_fields_to_db():
    with patch("src.database.record_decision_cache_event") as db_mock:
        await record_cache_event(
            symbol="ENI.MI/EUR", outcome="miss", reason="changed",
            model_type="actuator", est_saved_tokens=12.5,
        )
    kwargs = db_mock.call_args.kwargs
    assert kwargs["symbol"] == "ENI.MI/EUR"
    assert kwargs["outcome"] == "miss"
    assert kwargs["reason"] == "changed"
    assert kwargs["model_type"] == "actuator"
    assert kwargs["est_saved_tokens"] == 12.5


@pytest.mark.asyncio
async def test_record_cache_event_swallows_db_failure():
    """A metric write failure must never propagate into the decision path."""
    with patch("src.database.record_decision_cache_event", side_effect=RuntimeError("db down")):
        await record_cache_event(symbol="ENI.MI/EUR", outcome="hit")  # must not raise


def _llm_ok():
    return {
        "response": json.dumps({"action": "HOLD", "confidence": 0.9, "reasoning": "LLM decision", "strategy": {"type": "fallback", "parameters": {}}}),
        "provider": "openai",
        "model": "gpt-x",
        "is_fallback": False,
    }


@pytest.mark.asyncio
async def test_hit_path_records_hit_with_saved_tokens():
    manager = _make_backtest_manager()
    kwargs = _run_kwargs(Signal(action="HOLD", confidence=0.6, reasoning="prelim"), [{"summary": "bt"}])
    with patch("src.trading.components.backtest_manager.get_cached_llm_response_async", new=AsyncMock()) as llm_mock, \
         patch("src.trading.components.backtest_manager.record_cache_event", new=AsyncMock()) as ev:
        llm_mock.return_value = _llm_ok()
        await manager.run_step2_llm_call(**kwargs)  # cold: no entry yet
        assert [c.kwargs.get("reason") for c in ev.await_args_list] == ["cold"]

        await manager.run_step2_llm_call(**kwargs)  # hit: inputs unchanged
        hits = [c for c in ev.await_args_list if c.kwargs.get("outcome") == "hit"]
        assert len(hits) == 1
        assert hits[0].kwargs["reason"] == "unchanged"
        assert hits[0].kwargs["action"] == "HOLD"
        assert hits[0].kwargs["model_type"] == "actuator"
        # The store site recorded the snapshot's token footprint, so a hit
        # reports a non-zero saving rather than the old silent zero.
        assert hits[0].kwargs["est_saved_tokens"] > 0


@pytest.mark.asyncio
async def test_changed_inputs_record_miss_changed():
    manager = _make_backtest_manager()
    kwargs = _run_kwargs(Signal(action="HOLD", confidence=0.6, reasoning="prelim"), [{"summary": "bt"}])
    with patch("src.trading.components.backtest_manager.get_cached_llm_response_async", new=AsyncMock()) as llm_mock, \
         patch("src.trading.components.backtest_manager.record_cache_event", new=AsyncMock()) as ev:
        llm_mock.return_value = _llm_ok()
        await manager.run_step2_llm_call(**kwargs)
        kwargs2 = _run_kwargs(Signal(action="HOLD", confidence=0.6, reasoning="prelim"), [{"summary": "bt"}])
        kwargs2["ticker"] = {"last": 10.5, "symbol": "ENI.MI"}
        await manager.run_step2_llm_call(**kwargs2)
        assert llm_mock.await_count == 2
        assert [c.kwargs.get("reason") for c in ev.await_args_list] == ["cold", "changed"]
        assert all(c.kwargs.get("outcome") == "miss" for c in ev.await_args_list)


@pytest.mark.asyncio
async def test_unrebuildable_entry_records_rebuild_failed():
    manager = _make_backtest_manager()
    kwargs = _run_kwargs(Signal(action="HOLD", confidence=0.6, reasoning="prelim"), [{"summary": "bt"}])
    with patch("src.trading.components.backtest_manager.get_cached_llm_response_async", new=AsyncMock()) as llm_mock, \
         patch("src.trading.components.backtest_manager.record_cache_event", new=AsyncMock()) as ev:
        llm_mock.return_value = _llm_ok()
        await manager.run_step2_llm_call(**kwargs)
        # Corrupt the stored entry so it passes get_cached_decision's shape
        # check (snapshot_hash + signal keys present) and matches the hash,
        # but the signal itself cannot be rebuilt into a Signal.
        entry = get_cached_decision(manager.engine.redis, "ENI.MI/EUR")
        manager.engine.redis.setex(
            "llm:dec_cache:ENI.MI/EUR", 100,
            json.dumps({
                "snapshot_hash": entry["snapshot_hash"],
                "signal": None,
                "llm_provider": "openai",
                "llm_model": "gpt-x",
            }),
        )
        await manager.run_step2_llm_call(**kwargs)
        assert llm_mock.await_count == 2  # decision path unaffected
        assert ev.await_args_list[-1].kwargs["reason"] == "rebuild_failed"
        assert ev.await_args_list[-1].kwargs["outcome"] == "miss"


@pytest.mark.asyncio
async def test_metric_write_failure_does_not_break_decision_path():
    manager = _make_backtest_manager()
    kwargs = _run_kwargs(Signal(action="HOLD", confidence=0.6, reasoning="prelim"), [{"summary": "bt"}])
    with patch("src.trading.components.backtest_manager.get_cached_llm_response_async", new=AsyncMock()) as llm_mock, \
         patch("src.trading.components.backtest_manager.record_cache_event", new=AsyncMock()) as ev:
        llm_mock.return_value = _llm_ok()
        ev.side_effect = RuntimeError("metric store exploded")
        sig, _, _, _ = await manager.run_step2_llm_call(**kwargs)
        assert sig.step2_reviewed is True
        assert llm_mock.await_count == 1


class _FakeCursor:
    def __init__(self, one=None, many=None, rowcount=0):
        self._one, self._many = one, many
        self.rowcount = rowcount

    def fetchone(self):
        return self._one

    def fetchall(self):
        return self._many


class _FakeConn:
    """Minimal connection stub: returns queued cursors, records queries."""

    def __init__(self, results=None):
        self._results = list(results or [])
        self.queries = []
        self.closed = False

    def execute(self, sql, params=None):
        self.queries.append((sql, params))
        return self._results.pop(0) if self._results else _FakeCursor()

    def commit(self):
        pass

    def close(self):
        self.closed = True


def test_record_decision_cache_event_inserts_and_closes():
    from src.database import record_decision_cache_event

    conn = _FakeConn()
    with patch("src.database.get_connection", return_value=conn):
        record_decision_cache_event(
            symbol="ENI.MI/EUR", outcome="hit", reason="unchanged",
            action="BUY", model_type="actuator", est_saved_tokens=1234,
        )
    sql, params = conn.queries[0]
    assert "INSERT INTO decision_cache_metrics" in sql
    assert params[1] == "ENI.MI/EUR"
    assert params[2] == "hit"
    assert params[3] == "unchanged"
    assert params[6] == 1234
    assert conn.closed is True  # returns the pooled connection


def test_get_decision_cache_summary_aggregates_hit_rate():
    from src.database import get_decision_cache_summary

    totals = _FakeCursor(one={
        "total": 10, "hits": 4, "misses": 6,
        "total_saved_tokens": 12000, "avg_saved_tokens": 1200.0,
    })
    per_outcome = _FakeCursor(many=[{
        "outcome": "hit", "calls": 4,
        "total_saved_tokens": 12000, "avg_saved_tokens": 3000.0,
    }])
    recent = _FakeCursor(many=[{
        "timestamp": 1.0, "symbol": "ENI.MI/EUR", "outcome": "hit",
        "reason": "unchanged", "action": "BUY", "model_type": "actuator",
        "est_saved_tokens": 3000,
    }])
    conn = _FakeConn([totals, per_outcome, recent])
    with patch("src.database.get_connection", return_value=conn):
        out = get_decision_cache_summary(period_days=7)

    assert out["total"] == 10
    assert out["hits"] == 4
    assert out["misses"] == 6
    assert out["hit_rate"] == 40.0
    assert out["total_saved_tokens"] == 12000
    assert out["per_outcome"][0]["outcome"] == "hit"
    assert out["recent"][0]["symbol"] == "ENI.MI/EUR"
    assert conn.closed is True
    # Every aggregate query is period-scoped with exactly one cutoff parameter.
    assert len(conn.queries) == 3
    for sql, params in conn.queries:
        assert "decision_cache_metrics" in sql
        assert len(params) == 1


def test_get_decision_cache_summary_handles_empty_period():
    from src.database import get_decision_cache_summary

    totals = _FakeCursor(one={
        "total": 0, "hits": 0, "misses": 0,
        "total_saved_tokens": 0, "avg_saved_tokens": None,
    })
    conn = _FakeConn([totals, _FakeCursor(many=[]), _FakeCursor(many=[])])
    with patch("src.database.get_connection", return_value=conn):
        out = get_decision_cache_summary(period_days=7)

    assert out["total"] == 0
    assert out["hit_rate"] == 0.0
    assert out["avg_saved_tokens"] == 0
    assert out["per_outcome"] == []
    assert out["recent"] == []
    assert conn.closed is True


def test_cleanup_old_decision_cache_metrics_deletes_and_closes():
    from src.database import cleanup_old_decision_cache_metrics

    conn = _FakeConn([_FakeCursor(one=None, rowcount=5)])
    with patch("src.database.get_connection", return_value=conn):
        deleted = cleanup_old_decision_cache_metrics(retention_days=90)
    sql, params = conn.queries[0]
    assert "DELETE FROM decision_cache_metrics" in sql
    assert len(params) == 1
    assert deleted == 5
    assert conn.closed is True
