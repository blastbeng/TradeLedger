import asyncio
import json
import logging
import math
import os
import secrets
import sys
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, Request, Response, APIRouter, Body
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from collections import defaultdict, deque
from src.config.settings import settings
from src.utils.redis_client import get_redis_client, check_redis_connection
from src.llm.prompts import get_cached_news_summary
from src.exchanges.market_data import get_quotes, get_multi_timeframe_bars
from src.database import get_all_discovered_symbols, get_signals
from typing import Optional
from pydantic import BaseModel

# Rate limiting configuration
RATE_LIMIT_REQUESTS = 100  # max requests per window
RATE_LIMIT_WINDOW = 60    # window size in seconds

# In-memory store for rate limiting: {ip: deque([timestamps])}
_rate_limit_store = defaultdict(deque)

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

async def verify_auth(request: Request):
    """Verify the user is authenticated via a session cookie."""
    if not settings.WEB_USERNAME or not settings.WEB_PASSWORD:
        return True

    token = request.cookies.get("session_token")
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    redis = get_redis_client()
    session = await asyncio.to_thread(redis.get, f"session:{token}")
    if not session:
        raise HTTPException(status_code=401, detail="Invalid or expired session")

    return True

app = FastAPI(title="Trade Ledger")

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()

    # Remove timestamps older than the window
    ip_deque = _rate_limit_store[client_ip]
    while ip_deque and ip_deque[0] < now - RATE_LIMIT_WINDOW:
        ip_deque.popleft()

    # Optional: clean up empty deques to prevent memory bloat
    if not ip_deque:
        _rate_limit_store.pop(client_ip, None)

    if len(ip_deque) >= RATE_LIMIT_REQUESTS:
        return JSONResponse(status_code=429, content={"detail": "Too Many Requests"})

    ip_deque.append(now)
    response = await call_next(request)
    return response

public_router = APIRouter()
http_router = APIRouter(dependencies=[Depends(verify_auth)])

@public_router.post("/api/login")
async def login(request: Request, response: Response, credentials: dict = Body(...)):
    """Authenticate the user and set a session cookie."""
    username = credentials.get("username", "")
    password = credentials.get("password", "")
    if not (settings.WEB_USERNAME and settings.WEB_PASSWORD and
            secrets.compare_digest(username, settings.WEB_USERNAME) and
            secrets.compare_digest(password, settings.WEB_PASSWORD)):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = secrets.token_urlsafe(32)
    redis = get_redis_client()
    await asyncio.to_thread(redis.set, f"session:{token}", "1", ex=86400)  # 1 day expiry
    response.set_cookie(key="session_token", value=token, httponly=True, samesite="lax", max_age=86400)
    return {"status": "ok"}

@public_router.post("/api/logout")
async def logout(request: Request, response: Response):
    """Clear the session cookie and invalidate the token."""
    token = request.cookies.get("session_token")
    if token:
        redis = get_redis_client()
        await asyncio.to_thread(redis.delete, f"session:{token}")
    response.delete_cookie("session_token")
    return {"status": "ok"}

logger = logging.getLogger(__name__)

# Serve static files (dashboard)
app.mount("/static", StaticFiles(directory="src/web/static"), name="static")

# Global engine reference
_engine = None

# WebSocket payload cache – shared across all connected clients to avoid
# redundant API calls and SQLite queries when multiple tabs are open.
_ws_payload_cache: Optional[dict] = None
_ws_payload_cache_time: float = 0.0
_ws_payload_ttl: float = 5.0  # seconds — can be changed via API

def set_engine(engine):
    global _engine
    _engine = engine
    logger.info("Trading engine attached to web server")

def get_engine():
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    return _engine

@public_router.get("/")
async def root():
    return FileResponse("src/web/static/index.html")

@http_router.get("/health")
async def health():
    redis_ok = check_redis_connection()
    from src.llm.llm_client import check_llm_health
    llm_health = await run_in_threadpool(check_llm_health)
    mind_ok = llm_health.get("mind", {}).get("status") == "connected"
    actuator_ok = llm_health.get("actuator", {}).get("status") == "connected"
    all_ok = redis_ok and mind_ok and actuator_ok
    return {
        "status": "ok" if all_ok else "degraded",
        "redis": "connected" if redis_ok else "disconnected",
        "llm_mind": llm_health.get("mind", {}),
        "llm_actuator": llm_health.get("actuator", {}),
    }

@http_router.get("/api/status")
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

@http_router.get("/api/trades")
async def trades(limit: int = 0):
    engine = get_engine()
    open_trades = await engine.event_bus.request("get_open_trades")
    for t in open_trades:
        t["display_symbol"] = await _get_display_symbol(engine, t["symbol"], t.get("timeframe"))
    return {"trades": open_trades}

@http_router.get("/api/profit")
async def profit():
    engine = get_engine()
    return await engine.event_bus.request("get_profit_summary")

@http_router.get("/api/performance")
async def performance():
    engine = get_engine()
    perf = await engine.get_performance_summary()
    if perf.get("rows"):
        async def _add_display_to_row(row):
            row["display_symbol"] = await _get_display_symbol(engine, row["symbol"], row.get("timeframe"))
            return row
        perf["rows"] = await asyncio.gather(*[_add_display_to_row(row) for row in perf["rows"]])
    if perf.get("total"):
        total = perf["total"]
        total["display_symbol"] = "TOTAL"
    return perf

@http_router.get("/api/market-status")
async def market_status_api():
    redis = get_redis_client()
    market_status = None
    try:
        market_status_raw = await asyncio.to_thread(redis.get, "market:status")
        if market_status_raw:
            market_status = json.loads(market_status_raw)
    except Exception:
        pass
    return market_status or {}

@http_router.get("/api/risk")
async def risk():
    engine = get_engine()
    return await engine.get_risk_metrics()

@http_router.get("/api/news")
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

@http_router.get("/api/messages")
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

@http_router.get("/api/logs")
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

@http_router.get("/api/history")
async def history(limit: int = 50):
    engine = get_engine()
    trades = engine.trade_history[-limit:]
    if trades:
        async def _add_display_to_trade(t):
            t["display_symbol"] = await _get_display_symbol(engine, t["symbol"], t.get("timeframe"))
            return t
        trades = await asyncio.gather(*[_add_display_to_trade(t) for t in trades])
    return trades

@http_router.post("/api/pause")
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

@http_router.post("/api/resume")
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

@http_router.post("/api/sell")
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

@http_router.post("/api/manual-trade")
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

    # Validate ticker against discovered symbols (case-insensitive)
    known_symbols = await run_in_threadpool(get_all_discovered_symbols)
    ticker_upper = req.ticker.upper()
    if not any(s.get("symbol", "").upper() == ticker_upper for s in known_symbols):
        raise HTTPException(status_code=400, detail=f"Unknown ticker: {req.ticker}. Please use a valid discovered symbol.")

    result = await engine.log_manual_trade(req.ticker, req.side, req.quantity, req.money_spent, req.fee)
    return result

@http_router.get("/api/manual-trades")
async def get_manual_trades():
    engine = get_engine()
    manual = [t for t in engine.trade_history if t.get("note") == "manual"]
    for t in manual:
        t["display_symbol"] = t["symbol"]
    return manual

@http_router.get("/api/discovered-symbols")
async def discovered_symbols_api():
    """Return all discovered symbols for frontend autocomplete."""
    symbols = await run_in_threadpool(get_all_discovered_symbols)
    return [{"symbol": s.get("symbol"), "name": s.get("name", "")} for s in symbols]

@http_router.get("/api/signals")
async def signals(page: int = 1, limit: int = 5):
    if page < 1:
        page = 1
    offset = (page - 1) * limit
    return await run_in_threadpool(get_signals, limit, offset)

@http_router.post("/api/reload")
async def reload():
    await run_in_threadpool(settings.reload)
    return {"status": "reloaded"}

@http_router.post("/api/config/update-interval")
async def update_interval(data: dict):
    """Update the WebSocket payload cache TTL to match the frontend's chosen interval."""
    global _ws_payload_ttl
    ms = data.get("interval_ms", 5000)
    _ws_payload_ttl = max(1.0, ms / 1000.0)  # convert ms to seconds, minimum 1s
    return {"status": "ok", "ttl_seconds": _ws_payload_ttl}

@http_router.post("/api/force-reeval")
async def force_reeval():
    engine = get_engine()
    engine.trigger_symbol_reevaluation(force=True)
    return {"status": "Forced re-evaluation triggered"}

@http_router.post("/api/force-download")
async def force_download():
    engine = get_engine()
    asyncio.create_task(engine.force_download_tracked_symbols())
    return {"status": "Force download of tracked symbols OHLCV data triggered"}

@http_router.post("/api/force-backfill")
async def force_backfill():
    engine = get_engine()
    asyncio.create_task(engine.force_download_all_assets())
    return {"status": "Force backfill of all discovered symbols triggered"}

@http_router.post("/api/restart")
async def restart():
    """
    Restart the entire application by exiting the process.
    Docker (or the process manager) will bring it back up.
    """
    engine = get_engine()
    await engine.stop()
    sys.exit(0)

@http_router.get("/api/config")
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

@http_router.get("/api/ohlcv/{symbol:path}")
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

@http_router.get("/api/ticker/{symbol:path}")
async def ticker(symbol: str):
    engine = get_engine()
    base_symbol = symbol.split("/")[0]
    try:
        quotes = await engine._get_quotes_async([base_symbol], timeout=300.0)
        q = quotes.get(base_symbol)
        if q:
            return {
                "symbol": symbol,
                "last": q.get("last"),
                "bid": q.get("bid"),
                "ask": q.get("ask"),
                "change_24h": q.get("change_24h"),
                "percentage": q.get("percentage"),
            }
    except Exception as e:
        logger.warning(f"REST ticker fetch failed for {symbol}: {e}")
    return {
        "symbol": symbol,
        "last": None,
        "bid": None,
        "ask": None,
        "change_24h": None,
        "percentage": None,
    }

@http_router.get("/api/tickers")
async def tickers(symbols: str = ""):
    """Return quotes for a comma-separated list of symbols."""
    if not symbols:
        return {}
    engine = get_engine()
    full_symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]
    base_symbol_list = [s.split("/")[0] for s in full_symbol_list]
    result = {}
    try:
        quotes = await engine._get_quotes_async(base_symbol_list, timeout=300.0)
        for full_sym, base_sym in zip(full_symbol_list, base_symbol_list):
            q = quotes.get(base_sym)
            if q:
                result[full_sym] = {
                    "last": q.get("last"),
                    "bid": q.get("bid"),
                    "ask": q.get("ask"),
                    "change_24h": q.get("change_24h"),
                    "percentage": q.get("percentage"),
                }
            else:
                result[full_sym] = {"last": None, "bid": None, "ask": None, "change_24h": None, "percentage": None}
    except Exception as e:
        logger.warning(f"REST tickers fetch failed: {e}")
        for full_sym in full_symbol_list:
            result[full_sym] = {"last": None, "bid": None, "ask": None, "change_24h": None, "percentage": None}
    return result

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    # Verify session for WebSocket
    if settings.WEB_USERNAME and settings.WEB_PASSWORD:
        token = websocket.cookies.get("session_token")
        if not token:
            await websocket.close(code=1008)  # Policy Violation
            return
        redis = get_redis_client()
        session = await asyncio.to_thread(redis.get, f"session:{token}")
        if not session:
            await websocket.close(code=1008)
            return

    await websocket.accept()
    logger.info("WebSocket client connected")
    last_sent_str = None
    try:
        while True:
            try:
                engine = get_engine()
                redis = get_redis_client()

                # Re-check session token to handle expiration during active connection
                if settings.WEB_USERNAME and settings.WEB_PASSWORD:
                    token = websocket.cookies.get("session_token")
                    if not token:
                        await websocket.close(code=1008)  # Policy Violation
                        break
                    session = await asyncio.to_thread(redis.get, f"session:{token}")
                    if not session:
                        await websocket.close(code=1008)
                        break

                # --- Cached payload: share across all WebSocket clients ---
                now = time.time()
                global _ws_payload_cache, _ws_payload_cache_time
                market_open = await engine._is_market_open()
                effective_ttl = _ws_payload_ttl if market_open else max(_ws_payload_ttl, 60.0)
                if _ws_payload_cache is not None and (now - _ws_payload_cache_time) < effective_ttl:
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

                    balances = await run_in_threadpool(engine.trader.fetch_balance)
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

                    payload = {
                        "current_symbols": current_symbols,
                        "positions": positions,
                        "balances": balances,
                        "paused": is_paused,
                        "pause_info": pause_info,
                        "queued_orders": queued_orders_payload,
                        "market_open": market_open,
                    }
                    _ws_payload_cache = payload
                    _ws_payload_cache_time = now

                payload_str = json.dumps(payload)
                if payload_str != last_sent_str:
                    await websocket.send_text(payload_str)
                    last_sent_str = payload_str
            except HTTPException:
                init_str = json.dumps({"status": "initializing"})
                if init_str != last_sent_str:
                    await websocket.send_text(init_str)
                    last_sent_str = init_str
            except WebSocketDisconnect:
                logger.debug("WebSocket client disconnected")
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}", exc_info=True)
                break
            await asyncio.sleep(_ws_payload_ttl)
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")

@http_router.post("/api/simulate/backtest/{symbol:path}")
async def simulate_backtest(symbol: str):
    engine = get_engine()
    return await engine.simulate_backtest(symbol)

@http_router.post("/api/simulate/decision/{symbol:path}")
async def simulate_decision(symbol: str):
    engine = get_engine()
    return await engine.simulate_decision(symbol)

app.include_router(public_router)
app.include_router(http_router)
