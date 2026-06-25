import asyncio
import copy
import logging
import signal
import sys
import uvicorn
import uvicorn.config
from src.web.app import app
from src.config.settings import settings
from src.database import init_db, get_telegram_chat_id, set_telegram_chat_id
from src.utils.redis_client import get_redis_client, check_redis_connection
from src.trading.engine import TradingEngine
from src.news.fetcher import test_rss_feeds


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

if not check_redis_connection():
    logging.critical("Redis is not reachable. Exiting.")
    sys.exit(1)

# --- Redis log handler for the web dashboard ---
import logging.handlers
import json as _json
from datetime import datetime, timezone

class RedisLogHandler(logging.Handler):
    """Push log records to a Redis list for the web dashboard."""
    def __init__(self, max_entries=200):
        super().__init__()
        self.max_entries = max_entries
        self.setLevel(logging.INFO)   # only INFO and above to keep the list manageable
        self._seen_keys: set = set()
        self._seen_keys_max = 500

    def emit(self, record):
        try:
            # Deduplication: skip if we've already processed this exact record.
            # record.created is a float timestamp set when the LogRecord was
            # constructed; it is identical across multiple handler calls for
            # the same record.
            key = (record.created, record.name, record.levelname, record.msg)
            if key in self._seen_keys:
                return
            self._seen_keys.add(key)
            # Prevent unbounded growth
            if len(self._seen_keys) > self._seen_keys_max:
                self._seen_keys.clear()

            redis_client = get_redis_client()
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": self.format(record),
            }
            redis_client.lpush("logs:recent", _json.dumps(entry))
            redis_client.ltrim("logs:recent", 0, self.max_entries - 1)
        except Exception:
            # Never let a logging handler break the application
            pass

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



async def main():
    init_db()
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

    # Start the engine as a background task immediately
    logging.info("Creating engine task...")
    engine_task = asyncio.create_task(engine.run())

    def engine_task_done(task: asyncio.Task):
        try:
            task.result()
        except asyncio.CancelledError:
            logging.info("Engine task was cancelled.")
        except Exception as e:
            logging.critical(f"Engine task crashed: {e}", exc_info=True)

    engine_task.add_done_callback(engine_task_done)

    # Start Telegram bot as a background task so it never blocks the engine
    telegram_task = None
    if settings.TELEGRAM_BOT_TOKEN:
        telegram_task = asyncio.create_task(telegram_bot.start())

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
    engine_task.cancel()
    try:
        await engine_task
    except asyncio.CancelledError:
        pass

    # Stop Telegram bot if it was started
    if telegram_task is not None:
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
