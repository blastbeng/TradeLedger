import asyncio
import copy
import logging
import queue
import signal
import sys
import threading
import time as _time
import uvicorn
import uvicorn.config
from src.web.app import app
from src.config.settings import settings
from src.database import init_db, get_telegram_chat_id, set_telegram_chat_id
from src.utils.redis_client import get_redis_client, check_redis_connection, is_redis_available
from src.trading.engine import TradingEngine
from src.news.fetcher import test_rss_feeds
from src.utils.task_supervisor import TaskSupervisor


class HealthEndpointFilter(logging.Filter):
    """Suppress uvicorn access logs for /health."""
    def filter(self, record):
        # Uvicorn access logs store the request details in record.args
        # Checking the formatted message is the most reliable way to match the path
        if '/health' in record.getMessage():
            return False
        return True

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Suppress httpx INFO logs (HTTP request/response lines) unless LOG_LEVEL is DEBUG
if settings.LOG_LEVEL.upper() != "DEBUG":
    logging.getLogger("httpx").setLevel(logging.WARNING)
else:
    logging.getLogger("httpx").setLevel(logging.DEBUG)

# Suppress urllib3 warnings (e.g., "Connection pool is full, discarding connection")
logging.getLogger("urllib3").setLevel(logging.ERROR)

# --- Redis log handler for the web dashboard ---
import logging.handlers
import json as _json
from datetime import datetime, timezone

class RedisLogHandler(logging.Handler):
    """Push log records to a Redis list for the web dashboard.

    Uses a bounded in-memory queue and a background daemon thread to avoid
    blocking the event loop with synchronous Redis I/O on every log record.
    """
    def __init__(self, max_entries=200):
        super().__init__()
        self.max_entries = max_entries
        self.setLevel(logging.INFO)   # only INFO and above to keep the list manageable
        self._queue: queue.Queue = queue.Queue(maxsize=1000)
        self._thread = threading.Thread(
            target=self._flush_loop, daemon=True, name="redis-log-flusher"
        )
        self._thread.start()

    def emit(self, record):
        if not is_redis_available():
            return
        try:
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": self.format(record),
            }
            # Non-blocking put: if the queue is full, drop the entry
            try:
                self._queue.put_nowait(entry)
            except queue.Full:
                pass
        except Exception:
            # Never let a logging handler break the application
            pass

    def _flush_loop(self):
        """Background thread that drains the queue and writes to Redis."""
        while True:
            try:
                entry = self._queue.get(timeout=1.0)
                redis_client = get_redis_client()
                redis_client.lpush("logs:recent", _json.dumps(entry))
                redis_client.ltrim("logs:recent", 0, self.max_entries - 1)
            except queue.Empty:
                continue
            except Exception:
                # Sleep briefly to avoid a tight error loop if Redis is down
                _time.sleep(0.5)

# Create and attach the handler
redis_log_handler = RedisLogHandler(max_entries=200)
# Use a simple format: time - logger - level - message
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
redis_log_handler.setFormatter(formatter)
# Guard against double-addition (e.g., if module is re-imported)
_root_logger = logging.getLogger()
if not any(isinstance(h, RedisLogHandler) for h in _root_logger.handlers):
    _root_logger.addHandler(redis_log_handler)

def _seed_telegram_chat_id():
    """If TELEGRAM_CHAT_ID is set in env and no chat_id is stored, store it."""
    if settings.TELEGRAM_CHAT_ID:
        existing = get_telegram_chat_id()
        if existing is None:
            try:
                chat_id = int(settings.TELEGRAM_CHAT_ID)
                set_telegram_chat_id(chat_id)
                logging.info(f"Seeded Telegram chat ID from env: {chat_id}")
            except ValueError:
                logging.warning("TELEGRAM_CHAT_ID in .env is not a valid integer")



def _validate_startup_settings():
    """Validate that critical settings are properly configured at startup."""
    if not check_redis_connection():
        logging.critical("Redis is not reachable. Exiting.")
        sys.exit(1)

    try:
        settings.validate_llm_settings()
    except ValueError as e:
        logging.critical(f"{e} Exiting.")
        sys.exit(1)

    try:
        from src.database import save_trading_state, load_trading_state
        save_trading_state("__startup_test__", "ok")
        load_trading_state()
    except Exception as e:
        logging.critical(f"Database is not writable: {e}. Exiting.")
        sys.exit(1)

async def main():
    init_db()
    _validate_startup_settings()
    # Pre-create yfinance cache directory to avoid race condition errors
    # when multiple threads try to create it simultaneously.
    import os
    yf_cache_dir = os.path.join(settings.DATA_DIR, ".cache", "py-yfinance")
    os.makedirs(yf_cache_dir, exist_ok=True)
    _seed_telegram_chat_id()
    test_rss_feeds()
    engine = TradingEngine()
    logging.info("Trading engine initialized.")
    from src.web.app import set_engine
    set_engine(engine)

    # Start the web server immediately so the dashboard can connect
    # Customize uvicorn logging: keep internal logs at LOG_LEVEL, but make access logs DEBUG
    log_config = copy.deepcopy(uvicorn.config.LOGGING_CONFIG)
    log_config["loggers"]["uvicorn.access"]["level"] = "DEBUG"
    log_config["loggers"]["uvicorn"]["level"] = settings.LOG_LEVEL.upper()

    # Remove root logger from uvicorn's config to prevent dictConfig from
    # replacing or duplicating our custom handlers (RedisLogHandler and
    # StreamHandler from basicConfig).  Uvicorn's own loggers have
    # propagate=False and their own handlers, so they don't need the root.
    if "" in log_config.get("loggers", {}):
        del log_config["loggers"][""]

    # Add the health endpoint filter
    log_config["filters"] = {
        "health_filter": {
            "()": "src.main.HealthEndpointFilter"
        }
    }
    log_config["loggers"]["uvicorn.access"]["filters"] = ["health_filter"]

    config = uvicorn.Config(
        app,
        host=settings.WEB_HOST,
        port=settings.WEB_PORT,
        log_config=log_config,
    )
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())
    logging.info(f"Web server started on {settings.WEB_HOST}:{settings.WEB_PORT}")

    # Set up Telegram notifier (object only, no network call yet)
    if settings.TELEGRAM_BOT_TOKEN:
        from src.telegram.bot import TelegramBot
        telegram_bot = TelegramBot(engine)
        engine.set_notifier(telegram_bot)

    # Start the engine as a supervised background task immediately
    logging.info("Creating engine task...")
    engine_supervisor = TaskSupervisor(engine.run, name="TradingEngine.run", max_restarts=10, restart_delay=10.0)
    engine_task = asyncio.create_task(engine_supervisor.run(), name="supervisor:TradingEngine.run")

    def engine_task_done(task: asyncio.Task):
        try:
            task.result()
        except asyncio.CancelledError:
            logging.info("Engine supervisor was cancelled.")
        except Exception as e:
            logging.critical(f"Engine supervisor crashed: {e}", exc_info=True)

    engine_task.add_done_callback(engine_task_done)

    # Start Telegram bot as a supervised background task so it never blocks the engine
    telegram_supervisor = None
    telegram_task = None
    if settings.TELEGRAM_BOT_TOKEN:
        telegram_supervisor = TaskSupervisor(telegram_bot.start, name="TelegramBot.start", max_restarts=10, restart_delay=10.0)
        telegram_task = asyncio.create_task(telegram_supervisor.run(), name="supervisor:TelegramBot.start")

    # Graceful shutdown handling
    shutdown_event = asyncio.Event()

    def _signal_handler():
        logging.info("Shutdown signal received.")
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            signal.signal(sig, lambda s, f: _signal_handler())

    # Wait for shutdown signal
    await shutdown_event.wait()
    logging.info("Shutting down...")

    # Stop the engine
    await engine.stop()
    engine_supervisor.cancel()
    engine_task.cancel()
    try:
        await engine_task
    except asyncio.CancelledError:
        pass

    # Stop Telegram bot if it was started
    if telegram_task is not None:
        telegram_supervisor.cancel()
        telegram_task.cancel()
        try:
            await telegram_task
        except asyncio.CancelledError:
            pass
        # Also run the bot's own cleanup
        await telegram_bot.stop()

    # Stop the server
    server.should_exit = True
    await server_task

if __name__ == "__main__":
    asyncio.run(main())
