import asyncio
import pytest
import threading
from src.trading.components.shared_state import SharedState
from src.utils.event_bus import EventBus


# ---------- SharedState Concurrency ----------

@pytest.mark.asyncio
async def test_shared_state_concurrent_positions():
    state = SharedState()
    
    async def write_position(i):
        await state.set_position(f"SYM{i}", {"price": i})
        
    await asyncio.gather(*[write_position(i) for i in range(100)])
    
    positions = await state.get_all_positions()
    assert len(positions) == 100
    assert positions["SYM50"]["price"] == 50


@pytest.mark.asyncio
async def test_shared_state_concurrent_cycle_spent():
    state = SharedState()
    
    async def add_spent():
        await state.add_cycle_spent(10.0)
        
    await asyncio.gather(*[add_spent() for _ in range(100)])
    
    total = await state.get_cycle_spent()
    assert total == 1000.0


@pytest.mark.asyncio
async def test_shared_state_concurrent_queued_orders():
    state = SharedState()
    
    async def add_order(i):
        await state.append_queued_order({"id": i})
        
    await asyncio.gather(*[add_order(i) for i in range(100)])
    
    orders = await state.get_queued_orders()
    assert len(orders) == 100


def test_shared_state_concurrent_append_trade():
    state = SharedState()
    
    def add_trade(i):
        state.append_trade({"timestamp": i, "side": "sell", "realized_pnl": 1.0})
        
    threads = [threading.Thread(target=add_trade, args=(i,)) for i in range(100)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
        
    assert len(state.trade_history) == 100
    assert state._trade_history_version == 100


# ---------- EventBus Concurrency ----------

@pytest.mark.asyncio
async def test_event_bus_concurrent_subscribe():
    bus = EventBus()
    
    async def handler():
        pass
        
    async def subscribe():
        bus.subscribe("test_event", handler)
        
    await asyncio.gather(*[subscribe() for _ in range(100)])
    
    assert len(bus._subscribers["test_event"]) == 100


@pytest.mark.asyncio
async def test_event_bus_concurrent_publish():
    bus = EventBus()
    counter = {"count": 0}
    
    async def handler():
        counter["count"] += 1
        
    bus.subscribe("test_event", handler)
    
    async def publish():
        await bus.publish("test_event")
        
    await asyncio.gather(*[publish() for _ in range(100)])
    
    assert counter["count"] == 100
