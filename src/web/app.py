import asyncio
import json
import logging
import math
import os
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from src.config.settings import settings
from src.utils.redis_client import get_redis_client, check_redis_connection
from src.llm.prompts import get_cached_news_summary
from src.exchanges.market_data import get_quotes, get_multi_timeframe_bars
from typing import Optional
from pydantic import BaseModel

async def _get_display_symbol(engine, symbol: str, timeframe: Optional[str] = None) -> str:
    """Return a formatted display string for the given symbol and timeframe."""
    try:
        name = await engine._get_stock_name(symbol)
    except Exception:
        name = symbol.split("/")[0] if "/" in symbol else symbol
    return engine._format_symbol_display(symbol, name, timeframe)

class ManualTradeRequest(BaseModel):
    ticker: str
    side: str  # "buy" or "sell"
    quantity: float
    money_spent: float
    fee: float = 0.0

app = FastAPI(title="Trade Ledger")

logger = logging.getLogger(__name__)

# Serve static files (dashboard)
app.mount("/static", StaticFiles(directory="src/web/static"), name="static")

# Global engine reference
_engine = None

# WebSocket payload cache – shared across all connected clients to avoid
# redundant API calls and SQLite queries when multiple tabs are open.
_ws_payload_cache: Optional[dict] = None
_ws_payload_cache_time: float = 0.0
_WS_PAYLOAD_TTL: float = 2.0  # seconds — matches the send interval

def set_engine(engine):
    global _engine
    _engine = engine
    logger.info("Trading engine attached to web server")

def get_engine():
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    return _engine

@app.get("/")
async def root():
    return FileResponse("src/web/static/index.html")

@app.get("/health")
async def health():
    redis_ok = check_redis_connection()
    return {
        "status": "ok" if redis_ok else "degraded",
        "redis": "connected" if redis_ok else "disconnected",
    }

@app.get("/api/status")
async def status():
    engine = get_engine()
    redis = get_redis_client()
    paused_raw = await asyncio.to_thread(redis.get, "trading:paused")
    paused = paused_raw == "1"
    market_open = await engine._is_market_open()

    current_symbols = []
    for entry in engine.current_symbols:
        entry_copy = dict(entry)
        entry_copy["display"] = await _get_display_symbol(engine, entry["symbol"], entry.get("timeframe"))
        current_symbols.append(entry_copy)

    positions = {}
    for sym, pos in engine.positions.items():
        pos_copy = dict(pos)
        pos_copy["display_symbol"] = await _get_display_symbol(engine, sym, pos.get("timeframe"))
        positions[sym] = pos_copy

    balances = await run_in_threadpool(engine.trader.fetch_balance)
    queued_orders_payload = [
        {k: v for k, v in q.items() if k != "signal"}
        for q in engine.queued_orders
    ]
    return {
        "current_symbols": current_symbols,
        "positions": positions,
        "balances": balances,
        "paused": paused,
        "market_open": market_open,
        "queued_orders": queued_orders_payload,
    }

@app.get("/api/trades")
async def trades(limit: int = 0):
    engine = get_engine()
    open_trades = await run_in_threadpool(engine.get_open_trades)
    for t in open_trades:
        t["display_symbol"] = await _get_display_symbol(engine, t["symbol"], t.get("timeframe"))
    return {"trades": open_trades}

@app.get("/api/profit")
async def profit():
    engine = get_engine()
    return await run_in_threadpool(engine.get_profit_summary)

@app.get("/api/performance")
async def performance():
    engine = get_engine()
    perf = await run_in_threadpool(engine.get_performance_summary)
    if perf.get("rows"):
        async def _add_display_to_row(row):
            row["display_symbol"] = await _get_display_symbol(engine, row["symbol"], row.get("timeframe"))
            return row
        perf["rows"] = await asyncio.gather(*[_add_display_to_row(row) for row in perf["rows"]])
    if perf.get("total"):
        total = perf["total"]
        total["display_symbol"] = "TOTAL"
    return perf

@app.get("/api/risk")
async def risk():
    engine = get_engine()
    return await run_in_threadpool(engine.get_risk_metrics)

@app.get("/api/news")
async def news():
    engine = get_engine()
    symbols = engine.current_symbols
    if not symbols:
        return []

    async def _fetch_news_entry(entry):
        symbol = entry["symbol"]
        try:
            news_data = await run_in_threadpool(get_cached_news_summary, symbol)
            summary = news_data["summary"]
        except Exception:
            summary = "Could not generate summary."
        display = await _get_display_symbol(engine, symbol, entry.get("timeframe"))
        return {"symbol": symbol, "display_symbol": display, "summary": summary}

    result = await asyncio.gather(*[_fetch_news_entry(entry) for entry in symbols])
    return list(result)

@app.get("/api/messages")
async def messages():
    redis = get_redis_client()
    raw_messages = await asyncio.to_thread(redis.lrange, "web:messages", 0, -1)
    messages = []
    for raw in raw_messages:
        try:
            msg = json.loads(raw)
            messages.append(msg)
        except Exception:
            pass
    return messages

@app.get("/api/logs")
async def logs(limit: int = 200):
    """Return the most recent log entries from Redis."""
    redis = get_redis_client()
    try:
        raw = await asyncio.to_thread(redis.lrange, "logs:recent", 0, limit - 1)
        entries = []
        for item in raw:
            try:
                entries.append(json.loads(item))
            except Exception:
                pass
        return entries
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/history")
async def history(limit: int = 50):
    engine = get_engine()
    trades = engine.trade_history[-limit:]
    if trades:
        async def _add_display_to_trade(t):
            t["display_symbol"] = await _get_display_symbol(engine, t["symbol"], t.get("timeframe"))
            return t
        trades = await asyncio.gather(*[_add_display_to_trade(t) for t in trades])
    return trades

@app.post("/api/pause")
async def pause():
    engine = get_engine()
    redis = engine.redis
    await asyncio.to_thread(redis.set, "trading:paused", "1")
    await asyncio.to_thread(redis.set, "trading:pause_source", "manual")
    await asyncio.to_thread(redis.delete, "trading:pause_start")
    await asyncio.to_thread(redis.delete, "trading:pause_duration")
    await asyncio.to_thread(redis.delete, "trading:pause_reason")
    await asyncio.to_thread(redis.delete, "trading:llm_pause_time")
    return {"status": "paused"}

@app.post("/api/resume")
async def resume():
    engine = get_engine()
    if not await engine._is_market_open():
        raise HTTPException(status_code=400, detail="Cannot resume: market is currently closed")
    redis = engine.redis
    keys = [
        "trading:paused",
        "trading:pause_source",
        "trading:pause_start",
        "trading:pause_duration",
        "trading:pause_reason",
        "trading:llm_pause_time",
    ]
    for key in keys:
        await asyncio.to_thread(redis.delete, key)
    return {"status": "resumed"}

@app.post("/api/sell")
async def sell(symbol: str = None):
    engine = get_engine()
    if not await engine._is_market_open():
        raise HTTPException(status_code=400, detail="Cannot sell: market is currently closed")
    if symbol:
        asyncio.create_task(engine.sell_position(symbol))
        return {"status": f"selling {symbol}"}
    else:
        asyncio.create_task(engine.sell_all_positions())
        return {"status": "selling all"}

@app.post("/api/manual-trade")
async def manual_trade(req: ManualTradeRequest):
    engine = get_engine()
    if not await engine._is_market_open():
        raise HTTPException(status_code=400, detail="Cannot log manual trade: market is currently closed")
    req.side = req.side.lower().strip()
    if req.side not in ("buy", "sell"):
        raise HTTPException(status_code=400, detail="Side must be 'buy' or 'sell'")
    if req.quantity <= 0:
        raise HTTPException(status_code=400, detail="Quantity must be positive")
    if req.money_spent <= 0:
        raise HTTPException(status_code=400, detail="Money spent must be positive")
    result = await engine.log_manual_trade(req.ticker, req.side, req.quantity, req.money_spent, req.fee)
    return result

@app.get("/api/manual-trades")
async def get_manual_trades():
    engine = get_engine()
    manual = [t for t in engine.trade_history if t.get("note") == "manual"]
    for t in manual:
        t["display_symbol"] = t["symbol"]
    return manual

@app.get("/api/signals")
async def signals(limit: int = 20):
    engine = get_engine()
    return await run_in_threadpool(engine.get_recent_signals, limit)

@app.post("/api/reload")
async def reload():
    await run_in_threadpool(settings.reload)
    return {"status": "reloaded"}

@app.post("/api/force-reeval")
async def force_reeval():
    engine = get_engine()
    engine.trigger_symbol_reevaluation(force=True, manual=True)
    return {"status": "Forced re-evaluation triggered"}

@app.post("/api/force-download")
async def force_download():
    engine = get_engine()
    asyncio.create_task(engine.force_download_all_assets())
    return {"status": "Force download of all asset OHLCV data triggered"}

@app.post("/api/restart")
def restart():
    """
    Restart the entire application by exiting the process.
    Docker (or the process manager) will bring it back up.
    """
    os._exit(0)

@app.get("/api/config")
def config():
    mind_provider = settings.LLM_MIND_PROVIDER or settings.LLM_PROVIDER
    actuator_provider = settings.LLM_ACTUATOR_PROVIDER or settings.LLM_PROVIDER
    if mind_provider == "ollama":
        mind_model = settings.OLLAMA_MIND_MODEL
    else:
        mind_model = settings.OPENAI_MIND_MODEL
    if actuator_provider == "ollama":
        actuator_model = settings.OLLAMA_ACTUATOR_MODEL
    else:
        actuator_model = settings.OPENAI_ACTUATOR_MODEL

    return {
        "trading_mode": settings.TRADING_MODE,
        "base_currency": settings.BASE_CURRENCY,
        "max_symbols": settings.MAX_SYMBOLS,
        "llm_mind_provider": mind_provider,
        "llm_mind_model": mind_model,
        "llm_actuator_provider": actuator_provider,
        "llm_actuator_model": actuator_model,
        "web_port": settings.WEB_PORT,
    }

@app.get("/api/ohlcv/{symbol:path}")
async def ohlcv(symbol: str, timeframe: str = "1h", limit: int = 24):
    engine = get_engine()
    base_symbol = symbol.split("/")[0]
    try:
        bars = await asyncio.to_thread(
            get_multi_timeframe_bars, base_symbol, [timeframe], limit=limit
        )
        candles = bars.get(timeframe, [])
        result = []
        for candle in candles:
            # Sanitize non-finite floats (NaN, Infinity) to None for JSON compliance
            def _sanitize(v):
                if isinstance(v, float) and not math.isfinite(v):
                    return None
                return v
            result.append({
                "timestamp": candle[0],
                "open": _sanitize(candle[1]),
                "high": _sanitize(candle[2]),
                "low": _sanitize(candle[3]),
                "close": _sanitize(candle[4]),
                "volume": _sanitize(candle[5]),
            })
        return {"symbol": symbol, "timeframe": timeframe, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/ticker/{symbol:path}")
async def ticker(symbol: str):
    base_symbol = symbol.split("/")[0]
    try:
        quotes = await asyncio.to_thread(
            get_quotes, [base_symbol]
        )
        q = quotes.get(base_symbol)
        if q:
            return {
                "symbol": symbol,
                "last": q.get("last"),
                "bid": q.get("bid"),
                "ask": q.get("ask"),
                "change_24h": q.get("change_24h"),
            }
    except Exception as e:
        logger.warning(f"REST ticker fetch failed for {symbol}: {e}")
    return {
        "symbol": symbol,
        "last": None,
        "bid": None,
        "ask": None,
        "change_24h": None,
    }

@app.get("/api/tickers")
async def tickers(symbols: str = ""):
    """Return quotes for a comma-separated list of symbols."""
    if not symbols:
        return {}
    full_symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]
    base_symbol_list = [s.split("/")[0] for s in full_symbol_list]
    result = {}
    try:
        quotes = await asyncio.to_thread(
            get_quotes, base_symbol_list
        )
        for full_sym, base_sym in zip(full_symbol_list, base_symbol_list):
            q = quotes.get(base_sym)
            if q:
                result[full_sym] = {
                    "last": q.get("last"),
                    "bid": q.get("bid"),
                    "ask": q.get("ask"),
                    "change_24h": q.get("change_24h"),
                }
            else:
                result[full_sym] = {"last": None, "bid": None, "ask": None, "change_24h": None}
    except Exception as e:
        logger.warning(f"REST tickers fetch failed: {e}")
        for full_sym in full_symbol_list:
            result[full_sym] = {"last": None, "bid": None, "ask": None, "change_24h": None}
    return result

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket client connected")
    try:
        while True:
            try:
                engine = get_engine()
                redis = get_redis_client()

                # --- Cached payload: share across all WebSocket clients ---
                now = time.time()
                global _ws_payload_cache, _ws_payload_cache_time
                if _ws_payload_cache is not None and (now - _ws_payload_cache_time) < _WS_PAYLOAD_TTL:
                    payload = _ws_payload_cache
                else:
                    # Build current_symbols with display (parallelized to avoid blocking)
                    async def _build_symbol_entry(entry):
                        entry_copy = dict(entry)
                        entry_copy["display"] = await _get_display_symbol(engine, entry["symbol"], entry.get("timeframe"))
                        return entry_copy

                    current_symbols = await asyncio.gather(
                        *[_build_symbol_entry(entry) for entry in engine.current_symbols]
                    ) if engine.current_symbols else []

                    # Build positions with display_symbol (parallelized)
                    async def _build_position_entry(sym, pos):
                        pos_copy = dict(pos)
                        pos_copy["display_symbol"] = await _get_display_symbol(engine, sym, pos.get("timeframe"))
                        return sym, pos_copy

                    position_results = await asyncio.gather(
                        *[_build_position_entry(sym, pos) for sym, pos in engine.positions.items()]
                    ) if engine.positions else []
                    positions = dict(position_results)

                    perf = await run_in_threadpool(engine.get_performance_summary)
                    if perf.get("rows"):
                        async def _add_display_to_row(row):
                            row["display_symbol"] = await _get_display_symbol(engine, row["symbol"], row.get("timeframe"))
                            return row
                        perf["rows"] = await asyncio.gather(
                            *[_add_display_to_row(row) for row in perf["rows"]]
                        )
                    if perf.get("total"):
                        total = perf["total"]
                        total["display_symbol"] = "TOTAL"

                    balances = await run_in_threadpool(engine.trader.fetch_balance)
                    profit_summary = await run_in_threadpool(engine.get_profit_summary)
                    pause_info = await engine.get_pause_status()

                    # Strip large/unserializable fields from queued orders before sending
                    queued_orders_payload = [
                        {k: v for k, v in q.items() if k != "signal"}
                        for q in engine.queued_orders
                    ]

                    try:
                        paused_val = await asyncio.to_thread(redis.get, "trading:paused")
                        is_paused = paused_val == "1"
                    except Exception:
                        is_paused = False

                    market_open = await engine._is_market_open()

                    payload = {
                        "current_symbols": current_symbols,
                        "positions": positions,
                        "balances": balances,
                        "profit": profit_summary,
                        "performance": perf,
                        "paused": is_paused,
                        "pause_info": pause_info,
                        "queued_orders": queued_orders_payload,
                        "market_open": market_open,
                    }
                    _ws_payload_cache = payload
                    _ws_payload_cache_time = now

                await websocket.send_text(json.dumps(payload))
            except HTTPException:
                await websocket.send_text(json.dumps({"status": "initializing"}))
            except Exception as e:
                logger.error(f"WebSocket error: {e}", exc_info=True)
                break
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")

@app.post("/api/simulate/backtest/{symbol:path}")
async def simulate_backtest(symbol: str):
    engine = get_engine()
    return await engine.simulate_backtest(symbol)

@app.post("/api/simulate/decision/{symbol:path}")
async def simulate_decision(symbol: str):
    engine = get_engine()
    return await engine.simulate_decision(symbol)
