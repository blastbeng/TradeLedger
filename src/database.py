import sqlite3
import json
import os
import logging
import time
import functools
import hashlib
import threading
from typing import Dict, List, Any, Optional

from src.config.settings import settings
from src.utils.redis_client import get_redis_client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend detection and connection management
# ---------------------------------------------------------------------------
_backend = settings.DATABASE_BACKEND   # "sqlite" or "postgresql"
_placeholder = "%s"                    # will be converted to "?" for sqlite
_pg_pool = None

if _backend == "postgresql":
    import psycopg2
    import psycopg2.extras
    import psycopg2.pool
    import psycopg2.errors

    _pg_pool = psycopg2.pool.ThreadedConnectionPool(
        minconn=2,
        maxconn=20,
        host=settings.DB_HOST,
        port=settings.DB_PORT,
        dbname=settings.DB_NAME,
        user=settings.DB_USER,
        password=settings.DB_PASSWORD,
    )
    _placeholder = "%s"
else:
    _placeholder = "?"


def _adapt_sql(sql: str) -> str:
    """Convert %s placeholders to ? for SQLite."""
    if _backend == "sqlite":
        return sql.replace("%s", "?")
    return sql


class _PgConnectionWrapper:
    """Wraps a psycopg2 connection so that close() returns it to the pool."""
    def __init__(self, conn):
        self._conn = conn
        # Use RealDictCursor for dict-like row access
        self._conn.cursor_factory = psycopg2.extras.RealDictCursor

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def close(self):
        try:
            _pg_pool.putconn(self._conn)
        except Exception:
            pass

    def commit(self):
        self._conn.commit()

    def rollback(self):
        self._conn.rollback()

    def cursor(self):
        return self._conn.cursor()

    def execute(self, sql, params=None):
        cur = self._conn.cursor()
        cur.execute(sql, params)
        return cur

    def executemany(self, sql, params_list):
        cur = self._conn.cursor()
        cur.executemany(sql, params_list)
        return cur


_sqlite_local = threading.local()

class _SqliteConnectionWrapper:
    """Wraps a sqlite3 connection so that close() is a no-op, allowing reuse."""
    def __init__(self, conn):
        self._conn = conn

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def close(self):
        # Do not close the persistent thread-local connection
        pass

    def commit(self):
        self._conn.commit()

    def rollback(self):
        self._conn.rollback()

    def cursor(self):
        return self._conn.cursor()

    def execute(self, sql, params=None):
        return self._conn.execute(sql, params)

    def executemany(self, sql, params_list):
        return self._conn.executemany(sql, params_list)

def get_connection():
    """Return a database connection appropriate for the current backend."""
    if _backend == "postgresql":
        conn = _pg_pool.getconn()
        return _PgConnectionWrapper(conn)
    else:
        if not hasattr(_sqlite_local, 'connection'):
            os.makedirs(os.path.dirname(settings.DATABASE_PATH), exist_ok=True)
            conn = sqlite3.connect(settings.DATABASE_PATH, timeout=30)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            _sqlite_local.connection = _SqliteConnectionWrapper(conn)
        return _sqlite_local.connection


def _normalize_symbol(symbol: str) -> str:
    """Extract the base symbol from a trading pair (e.g., 'AAPL/USD' -> 'AAPL')."""
    return symbol.split("/")[0] if "/" in symbol else symbol


# ---------------------------------------------------------------------------
# Retry decorator (handles both SQLite locks and PostgreSQL deadlocks)
# ---------------------------------------------------------------------------
def retry_on_db_lock(max_retries=3, initial_delay=0.5):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except sqlite3.OperationalError as e:
                    if "database is locked" in str(e):
                        last_exc = e
                        if attempt < max_retries:
                            delay = initial_delay * (2 ** attempt)
                            time.sleep(delay)
                            continue
                    raise
                except Exception as e:
                    # Retry on PostgreSQL deadlock (40P01) or serialization failure (40001)
                    pgcode = getattr(e, 'pgcode', None)
                    if pgcode in ('40P01', '40001'):
                        last_exc = e
                        if attempt < max_retries:
                            delay = initial_delay * (2 ** attempt)
                            time.sleep(delay)
                            continue
                    raise
            raise last_exc
        return wrapper
    return decorator


def _get_existing_columns(table_name: str) -> set:
    """Return a set of column names currently in the table."""
    conn = get_connection()
    try:
        if _backend == "postgresql":
            cur = conn.cursor()
            cur.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name = %s",
                (table_name,)
            )
            return {row['column_name'] for row in cur.fetchall()}
        else:
            cur = conn.execute(f"PRAGMA table_info({table_name})")
            return {row[1] for row in cur.fetchall()}
    finally:
        conn.close()


def _migrate_db():
    """Add missing columns to existing tables (schema migrations).

    Failed migrations are automatically retried with exponential backoff
    (up to 3 attempts) to handle transient database locks and deadlocks.
    """
    migrations = [
        ("trade_history", "timeframe", "ALTER TABLE trade_history ADD COLUMN timeframe TEXT"),
        ("trade_history", "cost_basis", "ALTER TABLE trade_history ADD COLUMN cost_basis REAL"),
        ("trade_history", "strategy_type", "ALTER TABLE trade_history ADD COLUMN strategy_type TEXT"),
        ("trade_history", "note", "ALTER TABLE trade_history ADD COLUMN note TEXT"),
        ("trade_history", "status", "ALTER TABLE trade_history ADD COLUMN status TEXT"),
        ("trade_history", "exit_reason", "ALTER TABLE trade_history ADD COLUMN exit_reason TEXT"),
        ("trade_history", "hold_time_seconds", "ALTER TABLE trade_history ADD COLUMN hold_time_seconds REAL"),
        ("trade_history", "buy_confidence", "ALTER TABLE trade_history ADD COLUMN buy_confidence REAL"),
        ("quotes", "quotevolume", "ALTER TABLE quotes ADD COLUMN quotevolume REAL"),
        ("discovered_symbols", "maturity", "ALTER TABLE discovered_symbols ADD COLUMN maturity TEXT"),
        ("discovered_symbols", "coupon", "ALTER TABLE discovered_symbols ADD COLUMN coupon REAL"),
        ("discovered_symbols", "country", "ALTER TABLE discovered_symbols ADD COLUMN country TEXT"),
    ]

    max_retries = 3
    initial_delay = 0.5

    for table, column, sql in migrations:
        existing = _get_existing_columns(table)
        if column not in existing:
            last_exc = None
            for attempt in range(max_retries + 1):
                conn = get_connection()
                try:
                    conn.execute(_adapt_sql(sql))
                    conn.commit()
                    logger.info(f"Migration succeeded: {sql}")
                    last_exc = None
                    break
                except Exception as e:
                    last_exc = e
                    if attempt < max_retries:
                        delay = initial_delay * (2 ** attempt)
                        logger.warning(
                            f"Migration {sql} failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"Migration {sql} failed after {max_retries + 1} attempts: {e}"
                        )
                finally:
                    conn.close()
            # If all retries failed, the column will be retried on next startup
            # because _get_existing_columns will still report it as missing.


def _get_init_statements() -> List[str]:
    """Return a list of CREATE TABLE/INDEX statements adapted for the backend."""
    if _backend == "postgresql":
        pk_type = "SERIAL PRIMARY KEY"
        bigint_type = "BIGINT"
        float_type = "DOUBLE PRECISION"
    else:
        pk_type = "INTEGER PRIMARY KEY AUTOINCREMENT"
        bigint_type = "INTEGER"
        float_type = "REAL"

    statements = [
        f"""
        CREATE TABLE IF NOT EXISTS trading_state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS telegram_state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS trade_history (
            id {pk_type},
            order_id TEXT,
            symbol TEXT NOT NULL,
            timeframe TEXT,
            side TEXT NOT NULL,
            type TEXT,
            amount REAL NOT NULL,
            price REAL NOT NULL,
            cost REAL,
            fee_cost REAL,
            fee_currency TEXT,
            realized_pnl REAL,
            cost_basis REAL,
            strategy_type TEXT,
            note TEXT,
            status TEXT,
            timestamp {bigint_type} NOT NULL,
            exit_reason TEXT,
            hold_time_seconds REAL,
            buy_confidence REAL
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_trade_history_symbol ON trade_history(symbol)",
        "CREATE INDEX IF NOT EXISTS idx_trade_history_timestamp ON trade_history(timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_trade_history_symbol_timeframe ON trade_history(symbol, timeframe)",
        f"""
        CREATE TABLE IF NOT EXISTS news_articles (
            id {pk_type},
            symbol TEXT NOT NULL,
            title TEXT,
            source TEXT,
            url TEXT,
            published_at TEXT,
            summary TEXT,
            sentiment_label TEXT,
            sentiment_compound REAL,
            fetched_at {float_type} NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_news_symbol ON news_articles(symbol)",
        "CREATE INDEX IF NOT EXISTS idx_news_fetched_at ON news_articles(fetched_at)",
        f"""
        CREATE TABLE IF NOT EXISTS market_data (
            id {pk_type},
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            timestamp {bigint_type} NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL NOT NULL
        )
        """,
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_market_data_symbol_tf_ts ON market_data(symbol, timeframe, timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_market_data_symbol_tf_ts_desc ON market_data(symbol, timeframe, timestamp DESC)",
        "CREATE INDEX IF NOT EXISTS idx_market_data_timestamp ON market_data(timestamp)",
        f"""
        CREATE TABLE IF NOT EXISTS indicators (
            id {pk_type},
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            timestamp {bigint_type} NOT NULL,
            indicators_json TEXT NOT NULL,
            computed_at {float_type} NOT NULL,
            UNIQUE(symbol, timeframe)
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS quotes (
            symbol TEXT PRIMARY KEY,
            last REAL,
            bid REAL,
            ask REAL,
            volume REAL,
            change_24h REAL,
            percentage REAL,
            quotevolume REAL,
            updated_at {float_type} NOT NULL
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS discovered_symbols (
            symbol TEXT PRIMARY KEY,
            isin TEXT,
            asset_type TEXT,
            name TEXT,
            maturity TEXT,
            coupon REAL,
            country TEXT,
            discovered_at {float_type} NOT NULL
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS position_pnl (
            id {pk_type},
            symbol TEXT NOT NULL,
            timestamp {bigint_type} NOT NULL,
            unrealized_pnl REAL,
            realized_pnl REAL,
            position_value REAL,
            cost_basis REAL,
            amount REAL,
            current_price REAL,
            pnl_pct REAL
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_position_pnl_symbol ON position_pnl(symbol)",
        "CREATE INDEX IF NOT EXISTS idx_position_pnl_timestamp ON position_pnl(timestamp)",
        f"""
        CREATE TABLE IF NOT EXISTS backtest_results (
            id {pk_type},
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            params_hash TEXT NOT NULL,
            variant_params_json TEXT NOT NULL,
            stats_json TEXT NOT NULL,
            summary TEXT,
            created_at {float_type} NOT NULL
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS signals (
            id {pk_type},
            symbol TEXT NOT NULL,
            display_symbol TEXT,
            stock_name TEXT,
            timeframe TEXT,
            action TEXT NOT NULL,
            confidence REAL,
            reasoning TEXT,
            strategy_type TEXT,
            model_type TEXT,
            llm_provider TEXT,
            llm_model TEXT,
            trade_amount REAL,
            base_currency TEXT,
            timestamp {bigint_type} NOT NULL,
            entry_condition TEXT,
            stop_loss TEXT,
            take_profit TEXT,
            position_size_fraction REAL,
            trailing_stop INTEGER,
            trailing_stop_distance_pct REAL,
            max_hold_time_seconds REAL,
            cooldown_after_loss_seconds REAL,
            order_type TEXT,
            limit_price REAL
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS llm_metrics (
            id {pk_type},
            timestamp {float_type} NOT NULL,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            model_type TEXT NOT NULL,
            prompt_tokens INTEGER NOT NULL DEFAULT 0,
            completion_tokens INTEGER NOT NULL DEFAULT 0,
            total_tokens INTEGER NOT NULL DEFAULT 0,
            cache_hit INTEGER NOT NULL DEFAULT 0,
            latency_ms REAL NOT NULL DEFAULT 0,
            error TEXT
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp DESC)",
        "CREATE INDEX IF NOT EXISTS idx_backtest_results_symbol ON backtest_results(symbol)",
        "CREATE INDEX IF NOT EXISTS idx_backtest_results_symbol_tf_hash ON backtest_results(symbol, timeframe, params_hash, created_at DESC)",
        "CREATE INDEX IF NOT EXISTS idx_backtest_results_created_at ON backtest_results(created_at)",
    ]
    return statements


def init_db():
    """Create tables if they don't exist, then run migrations."""
    conn = get_connection()
    try:
        statements = _get_init_statements()
        for stmt in statements:
            conn.execute(stmt)
        conn.commit()
    finally:
        conn.close()
    _migrate_db()


@retry_on_db_lock()
def insert_trade(trade: Dict[str, Any]):
    """Insert a completed trade into the trade_history table."""
    conn = get_connection()
    try:
        sql = _adapt_sql(
            """
            INSERT INTO trade_history (
                order_id, symbol, timeframe, side, type, amount, price, cost,
                fee_cost, fee_currency, realized_pnl, cost_basis,
                strategy_type, note, status, timestamp,
                exit_reason, hold_time_seconds, buy_confidence
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
        )
        conn.execute(sql, (
            trade.get("id"),
            trade["symbol"],
            trade.get("timeframe"),
            trade["side"],
            trade.get("type"),
            trade["amount"],
            trade["price"],
            trade.get("cost"),
            trade.get("fee", {}).get("cost"),
            trade.get("fee", {}).get("currency"),
            trade.get("realized_pnl"),
            trade.get("cost_basis"),
            trade.get("strategy_type"),
            trade.get("note"),
            trade.get("status", "closed"),
            trade["timestamp"],
            trade.get("exit_reason"),
            trade.get("hold_time_seconds"),
            trade.get("buy_confidence"),
        ))
        conn.commit()
    finally:
        conn.close()


@retry_on_db_lock()
def insert_position_pnl_snapshot(
    symbol: str,
    timestamp: int,
    unrealized_pnl: float,
    realized_pnl: float,
    position_value: float,
    cost_basis: float,
    amount: float,
    current_price: float,
    pnl_pct: float,
):
    """Insert a position-level P&L snapshot into the position_pnl table."""
    conn = get_connection()
    try:
        sql = _adapt_sql(
            """
            INSERT INTO position_pnl (
                symbol, timestamp, unrealized_pnl, realized_pnl,
                position_value, cost_basis, amount, current_price, pnl_pct
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
        )
        conn.execute(sql, (
            symbol, timestamp, unrealized_pnl, realized_pnl,
            position_value, cost_basis, amount, current_price, pnl_pct,
        ))
        conn.commit()
    finally:
        conn.close()


def get_position_pnl_history(symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
    """Retrieve position P&L snapshots for a symbol, ordered by timestamp ascending."""
    conn = get_connection()
    try:
        sql = _adapt_sql(
            """
            SELECT symbol, timestamp, unrealized_pnl, realized_pnl,
                   position_value, cost_basis, amount, current_price, pnl_pct
            FROM position_pnl
            WHERE symbol = %s
            ORDER BY timestamp DESC
            LIMIT %s
            """
        )
        rows = conn.execute(sql, (symbol, limit)).fetchall()
        result = []
        for row in rows:
            result.append({
                "symbol": row["symbol"],
                "timestamp": row["timestamp"],
                "unrealized_pnl": row["unrealized_pnl"],
                "realized_pnl": row["realized_pnl"],
                "position_value": row["position_value"],
                "cost_basis": row["cost_basis"],
                "amount": row["amount"],
                "current_price": row["current_price"],
                "pnl_pct": row["pnl_pct"],
            })
        result.reverse()  # chronological order (oldest first)
        return result
    finally:
        conn.close()


def get_all_position_pnl_latest() -> List[Dict[str, Any]]:
    """Retrieve the latest P&L snapshot for each symbol that has a recorded snapshot."""
    conn = get_connection()
    try:
        if _backend == "postgresql":
            sql = _adapt_sql(
                """
                SELECT DISTINCT ON (symbol) symbol, timestamp, unrealized_pnl,
                       realized_pnl, position_value, cost_basis, amount,
                       current_price, pnl_pct
                FROM position_pnl
                ORDER BY symbol, timestamp DESC
                """
            )
            rows = conn.execute(sql).fetchall()
        else:
            sql = _adapt_sql(
                """
                SELECT p.symbol, p.timestamp, p.unrealized_pnl, p.realized_pnl,
                       p.position_value, p.cost_basis, p.amount, p.current_price,
                       p.pnl_pct
                FROM position_pnl p
                INNER JOIN (
                    SELECT symbol, MAX(timestamp) AS max_ts
                    FROM position_pnl
                    GROUP BY symbol
                ) latest ON p.symbol = latest.symbol AND p.timestamp = latest.max_ts
                """
            )
            rows = conn.execute(sql).fetchall()
        result = []
        for row in rows:
            result.append({
                "symbol": row["symbol"],
                "timestamp": row["timestamp"],
                "unrealized_pnl": row["unrealized_pnl"],
                "realized_pnl": row["realized_pnl"],
                "position_value": row["position_value"],
                "cost_basis": row["cost_basis"],
                "amount": row["amount"],
                "current_price": row["current_price"],
                "pnl_pct": row["pnl_pct"],
            })
        return result
    finally:
        conn.close()


@retry_on_db_lock()
def cleanup_old_position_pnl(retention_days: int = 90):
    """Delete position P&L snapshots older than retention_days."""
    conn = get_connection()
    try:
        cutoff_ms = int((time.time() - retention_days * 24 * 60 * 60) * 1000)
        sql = _adapt_sql("DELETE FROM position_pnl WHERE timestamp < %s")
        deleted = conn.execute(sql, (cutoff_ms,)).rowcount
        conn.commit()
        if deleted:
            logger.info(f"Cleaned up {deleted} old position P&L snapshots (older than {retention_days} days)")
        return deleted
    finally:
        conn.close()


@retry_on_db_lock()
def save_backtest_result(symbol: str, timeframe: str, params_hash: str, variant_params: Dict[str, Any], stats: Dict[str, Any], summary: str):
    """Persist a backtest result to the database for historical analysis."""
    conn = get_connection()
    try:
        sql = _adapt_sql(
            """
            INSERT INTO backtest_results (symbol, timeframe, params_hash, variant_params_json, stats_json, summary, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
        )
        conn.execute(sql, (
            symbol,
            timeframe,
            params_hash,
            json.dumps(variant_params, default=str),
            json.dumps(stats, default=str),
            summary,
            time.time(),
        ))
        conn.commit()
    finally:
        conn.close()


def get_recent_backtest_result(symbol: str, timeframe: str, params_hash: str, max_age_seconds: int = 3600) -> Optional[Dict[str, Any]]:
    """Check for a recent backtest result with matching params hash (dedup within a time window)."""
    conn = get_connection()
    try:
        cutoff = time.time() - max_age_seconds
        sql = _adapt_sql(
            """
            SELECT stats_json, summary FROM backtest_results
            WHERE symbol = %s AND timeframe = %s AND params_hash = %s AND created_at >= %s
            ORDER BY created_at DESC LIMIT 1
            """
        )
        row = conn.execute(sql, (symbol, timeframe, params_hash, cutoff)).fetchone()
        if row:
            return {"stats": json.loads(row["stats_json"]), "summary": row["summary"]}
        return None
    finally:
        conn.close()


def get_backtest_results_for_symbol(symbol: str, timeframe: str = None, limit: int = 10) -> List[Dict[str, Any]]:
    """Retrieve recent backtest results for a symbol from the database."""
    conn = get_connection()
    try:
        if timeframe:
            sql = _adapt_sql(
                """
                SELECT symbol, timeframe, variant_params_json, stats_json, summary, created_at
                FROM backtest_results
                WHERE symbol = %s AND timeframe = %s
                ORDER BY created_at DESC
                LIMIT %s
                """
            )
            rows = conn.execute(sql, (symbol, timeframe, limit)).fetchall()
        else:
            sql = _adapt_sql(
                """
                SELECT symbol, timeframe, variant_params_json, stats_json, summary, created_at
                FROM backtest_results
                WHERE symbol = %s
                ORDER BY created_at DESC
                LIMIT %s
                """
            )
            rows = conn.execute(sql, (symbol, limit)).fetchall()
        result = []
        for row in rows:
            result.append({
                "symbol": row["symbol"],
                "timeframe": row["timeframe"],
                "variant_params": json.loads(row["variant_params_json"]),
                "stats": json.loads(row["stats_json"]),
                "summary": row["summary"],
                "created_at": row["created_at"],
            })
        return result
    finally:
        conn.close()


@retry_on_db_lock()
def cleanup_old_backtest_results(retention_days: int = 90):
    """Delete backtest results older than retention_days."""
    conn = get_connection()
    try:
        cutoff = time.time() - retention_days * 24 * 60 * 60
        sql = _adapt_sql("DELETE FROM backtest_results WHERE created_at < %s")
        deleted = conn.execute(sql, (cutoff,)).rowcount
        conn.commit()
        if deleted:
            logger.info(f"Cleaned up {deleted} old backtest results (older than {retention_days} days)")
        return deleted
    finally:
        conn.close()

@retry_on_db_lock()
def reset_paper_trading_data(keep_trade_history: bool = False):
    """Delete all paper trading related data (trades, positions, orders, balances)."""
    conn = get_connection()
    try:
        if not keep_trade_history:
            conn.execute(_adapt_sql("DELETE FROM trade_history"))
        conn.execute(_adapt_sql("DELETE FROM position_pnl"))
        conn.execute(_adapt_sql("DELETE FROM backtest_results"))
        conn.execute(_adapt_sql("DELETE FROM trading_state WHERE key IN ('paper_balances', 'paper_orders')"))
        conn.commit()
        logger.info("Paper trading data reset in database.")
    finally:
        conn.close()


# ---------- Trading state helpers ----------

def load_trading_state() -> Dict[str, Any]:
    """Load all trading state from the database."""
    conn = get_connection()
    try:
        rows = conn.execute("SELECT key, value FROM trading_state").fetchall()
    finally:
        conn.close()
    state = {}
    for row in rows:
        try:
            state[row["key"]] = json.loads(row["value"])
        except (json.JSONDecodeError, TypeError):
            state[row["key"]] = row["value"]
    return state


@retry_on_db_lock()
def save_trading_state(key: str, value: Any):
    """Insert or update a single trading state key."""
    conn = get_connection()
    try:
        if _backend == "postgresql":
            sql = _adapt_sql(
                "INSERT INTO trading_state (key, value) VALUES (%s, %s) "
                "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value"
            )
        else:
            sql = _adapt_sql(
                "INSERT OR REPLACE INTO trading_state (key, value) VALUES (%s, %s)"
            )
        conn.execute(sql, (key, json.dumps(value)))
        conn.commit()
    finally:
        conn.close()


# ---------- Paper balance helpers ----------

@retry_on_db_lock()
def save_paper_balances(balances: Dict[str, float]):
    """Persist the paper simulator's balances dict."""
    save_trading_state("paper_balances", balances)


def load_paper_balances() -> Dict[str, float]:
    """Load the paper simulator's balances dict. Returns empty dict if not found."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT value FROM trading_state WHERE key = 'paper_balances'"
        ).fetchone()
    finally:
        conn.close()
    if row:
        try:
            return json.loads(row["value"])
        except (json.JSONDecodeError, TypeError):
            pass
    return {}


@retry_on_db_lock()
def save_paper_orders(orders: List[Dict[str, Any]]):
    """Persist the paper simulator's open orders as a JSON list."""
    save_trading_state("paper_orders", orders)


def load_paper_orders() -> List[Dict[str, Any]]:
    """Load the paper simulator's persisted orders. Returns empty list if not found."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT value FROM trading_state WHERE key = 'paper_orders'"
        ).fetchone()
    finally:
        conn.close()
    if row:
        try:
            return json.loads(row["value"])
        except (json.JSONDecodeError, TypeError):
            pass
    return []


# ---------- Telegram state helpers ----------

def get_telegram_chat_id() -> Optional[int]:
    """Retrieve the stored Telegram chat ID."""
    conn = get_connection()
    try:
        row = conn.execute("SELECT value FROM telegram_state WHERE key = 'chat_id'").fetchone()
    finally:
        conn.close()
    if row:
        try:
            return int(row["value"])
        except (ValueError, TypeError):
            return None
    return None

@retry_on_db_lock()
def cleanup_old_ohlcv(retention_days: int = 30):
    """Delete OHLCV candles older than retention_days for all symbols and timeframes."""
    conn = get_connection()
    try:
        cutoff_ms = int((time.time() - retention_days * 24 * 60 * 60) * 1000)
        sql = _adapt_sql("DELETE FROM market_data WHERE timestamp < %s")
        deleted = conn.execute(sql, (cutoff_ms,)).rowcount
        conn.commit()
        if deleted:
            logger.info(f"Cleaned up {deleted} old OHLCV candles (older than {retention_days} days)")
        return deleted
    finally:
        conn.close()


def get_all_trades() -> List[Dict[str, Any]]:
    """Retrieve all trades from the trade_history table, ordered by timestamp ascending."""
    conn = get_connection()
    try:
        sql = _adapt_sql(
            """
            SELECT order_id, symbol, timeframe, side, type, amount, price, cost,
                   fee_cost, fee_currency, realized_pnl, cost_basis,
                   strategy_type, note, status, timestamp,
                   exit_reason, hold_time_seconds, buy_confidence
            FROM trade_history
            ORDER BY timestamp ASC
            """
        )
        rows = conn.execute(sql).fetchall()
        trades = []
        for row in rows:
            trades.append({
                "id": row["order_id"],
                "symbol": row["symbol"],
                "timeframe": row["timeframe"],
                "side": row["side"],
                "type": row["type"],
                "amount": row["amount"],
                "price": row["price"],
                "cost": row["cost"],
                "fee": {"cost": row["fee_cost"], "currency": row["fee_currency"]} if row["fee_cost"] is not None else {},
                "realized_pnl": row["realized_pnl"],
                "cost_basis": row["cost_basis"],
                "strategy_type": row["strategy_type"],
                "note": row["note"],
                "status": row["status"],
                "timestamp": row["timestamp"],
                "exit_reason": row["exit_reason"],
                "hold_time_seconds": row["hold_time_seconds"],
                "buy_confidence": row["buy_confidence"],
            })
        return trades
    finally:
        conn.close()


def get_performance() -> Dict[str, Any]:
    """Return performance summary grouped by symbol and timeframe, plus a TOTAL row."""
    conn = get_connection()
    try:
        rows = conn.execute("""
        SELECT
            symbol,
            timeframe,
            COUNT(*) AS trade_count,
            SUM(realized_pnl) AS total_profit,
            SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) AS wins,
            SUM(CASE WHEN realized_pnl < 0 THEN 1 ELSE 0 END) AS losses,
            SUM(cost_basis) AS total_cost_basis
        FROM trade_history
        WHERE side = 'sell' AND realized_pnl IS NOT NULL
        GROUP BY symbol, timeframe
        ORDER BY symbol, timeframe
    """).fetchall()
    finally:
        conn.close()

    performance = []
    total_trades = 0
    total_profit = 0.0
    total_wins = 0
    total_losses = 0
    total_cost_basis = 0.0

    for row in rows:
        symbol = row["symbol"]
        timeframe = row["timeframe"] or "N/A"
        trade_count = row["trade_count"]
        profit = row["total_profit"] or 0.0
        wins = row["wins"] or 0
        losses = row["losses"] or 0
        cost_basis = row["total_cost_basis"] or 0.0

        profit_pct = (profit / cost_basis * 100) if cost_basis else 0.0
        win_rate = (wins / trade_count * 100) if trade_count else 0.0

        performance.append({
            "symbol": symbol,
            "timeframe": timeframe,
            "trade_count": trade_count,
            "profit": round(profit, 4),
            "profit_pct": round(profit_pct, 2),
            "win_rate": round(win_rate, 2),
        })

        total_trades += trade_count
        total_profit += profit
        total_wins += wins
        total_losses += losses
        total_cost_basis += cost_basis

    total_profit_pct = (total_profit / total_cost_basis * 100) if total_cost_basis else 0.0
    total_win_rate = (total_wins / total_trades * 100) if total_trades else 0.0

    total_row = {
        "symbol": "TOTAL",
        "timeframe": "",
        "trade_count": total_trades,
        "profit": round(total_profit, 4),
        "profit_pct": round(total_profit_pct, 2),
        "win_rate": round(total_win_rate, 2),
    }

    return {
        "rows": performance,
        "total": total_row,
    }


@retry_on_db_lock()
def set_telegram_chat_id(chat_id: int):
    """Store the Telegram chat ID."""
    conn = get_connection()
    try:
        if _backend == "postgresql":
            sql = _adapt_sql(
                "INSERT INTO telegram_state (key, value) VALUES (%s, %s) "
                "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value"
            )
        else:
            sql = _adapt_sql(
                "INSERT OR REPLACE INTO telegram_state (key, value) VALUES (%s, %s)"
            )
        conn.execute(sql, ("chat_id", str(chat_id)))
        conn.commit()
    finally:
        conn.close()


@retry_on_db_lock()
def store_news_articles(symbol: str, articles: List[Dict[str, Any]]):
    """Replace all stored articles for a symbol with a fresh batch."""
    symbol = _normalize_symbol(symbol)
    conn = get_connection()
    try:
        now = time.time()
        # Delete old articles for this symbol
        conn.execute(_adapt_sql("DELETE FROM news_articles WHERE symbol = %s"), (symbol,))
        # Insert new articles
        sql = _adapt_sql(
            """
            INSERT INTO news_articles (
                symbol, title, source, url, published_at, summary,
                sentiment_label, sentiment_compound, fetched_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
        )
        for art in articles:
            sentiment = art.get("sentiment", {})
            conn.execute(sql, (
                symbol,
                art.get("title", ""),
                art.get("source", ""),
                art.get("url", ""),
                art.get("published_at", ""),
                art.get("summary", ""),
                sentiment.get("label", ""),
                sentiment.get("compound", 0.0),
                now,
            ))
        conn.commit()
    finally:
        conn.close()


def get_news_for_symbol(symbol: str, max_age_seconds: int = 900) -> List[Dict[str, Any]]:
    """Retrieve recent news articles for a symbol from the database."""
    symbol = _normalize_symbol(symbol)
    conn = get_connection()
    try:
        cutoff = time.time() - max_age_seconds
        sql = _adapt_sql(
            """
            SELECT title, source, url, published_at, summary, sentiment_label, sentiment_compound
            FROM news_articles
            WHERE symbol = %s AND fetched_at >= %s
            ORDER BY fetched_at DESC
            """
        )
        rows = conn.execute(sql, (symbol, cutoff)).fetchall()
        articles = []
        for row in rows:
            articles.append({
                "title": row["title"],
                "source": row["source"],
                "url": row["url"],
                "published_at": row["published_at"],
                "summary": row["summary"],
                "sentiment": {
                    "label": row["sentiment_label"],
                    "compound": row["sentiment_compound"],
                },
            })
        return articles
    finally:
        conn.close()


def get_news_for_symbols(symbols: List[str], max_age_seconds: int = 900) -> Dict[str, List[Dict[str, Any]]]:
    """Retrieve recent news articles for multiple symbols from the database in a single query.
    Returns a dict mapping symbol -> list of articles.
    """
    if not symbols:
        return {}
    normalized = [_normalize_symbol(s) for s in symbols]
    conn = get_connection()
    try:
        cutoff = time.time() - max_age_seconds
        if _backend == "postgresql":
            # PostgreSQL supports ANY() for IN clauses with large lists
            sql = _adapt_sql(
                """
                SELECT symbol, title, source, url, published_at, summary,
                       sentiment_label, sentiment_compound
                FROM news_articles
                WHERE symbol = ANY(%s) AND fetched_at >= %s
                ORDER BY fetched_at DESC
                """
            )
            rows = conn.execute(sql, (normalized, cutoff)).fetchall()
        else:
            # SQLite: use IN clause with placeholders
            placeholders = ",".join(["?" for _ in normalized])
            sql = _adapt_sql(
                f"""
                SELECT symbol, title, source, url, published_at, summary,
                       sentiment_label, sentiment_compound
                FROM news_articles
                WHERE symbol IN ({placeholders}) AND fetched_at >= %s
                ORDER BY fetched_at DESC
                """
            )
            rows = conn.execute(sql, normalized + [cutoff]).fetchall()

        result: Dict[str, List[Dict[str, Any]]] = {s: [] for s in symbols}
        for row in rows:
            sym = row["symbol"]
            # Map back to the original symbol format
            for orig_sym in symbols:
                if _normalize_symbol(orig_sym) == sym:
                    result[orig_sym].append({
                        "title": row["title"],
                        "source": row["source"],
                        "url": row["url"],
                        "published_at": row["published_at"],
                        "summary": row["summary"],
                        "sentiment": {
                            "label": row["sentiment_label"],
                            "compound": row["sentiment_compound"],
                        },
                    })
                    break
        return result
    finally:
        conn.close()


def get_aggregate_sentiment_from_db(symbol: str, max_age_seconds: int = 900) -> Optional[Dict[str, Any]]:
    """Return aggregate sentiment for a symbol from the database."""
    symbol = _normalize_symbol(symbol)
    articles = get_news_for_symbol(symbol, max_age_seconds)
    if not articles:
        return None
    compounds = [a["sentiment"]["compound"] for a in articles if "sentiment" in a]
    if not compounds:
        return None
    avg_compound = sum(compounds) / len(compounds)
    labels = [a["sentiment"]["label"] for a in articles if "sentiment" in a]
    pos = labels.count("positive")
    neg = labels.count("negative")
    neu = labels.count("neutral")
    return {
        "avg_compound": round(avg_compound, 4),
        "positive": pos,
        "negative": neg,
        "neutral": neu,
        "total_articles": len(articles),
    }


def get_aggregate_sentiment_for_symbols(symbols: List[str], max_age_seconds: int = 900) -> Dict[str, Optional[Dict[str, Any]]]:
    """Return aggregate sentiment for multiple symbols from the database in a single query."""
    if not symbols:
        return {}
    normalized = [_normalize_symbol(s) for s in symbols]
    conn = get_connection()
    try:
        cutoff = time.time() - max_age_seconds
        if _backend == "postgresql":
            sql = _adapt_sql(
                """
                SELECT symbol, sentiment_label, sentiment_compound
                FROM news_articles
                WHERE symbol = ANY(%s) AND fetched_at >= %s
                """
            )
            rows = conn.execute(sql, (normalized, cutoff)).fetchall()
        else:
            placeholders = ",".join(["?" for _ in normalized])
            sql = _adapt_sql(
                f"""
                SELECT symbol, sentiment_label, sentiment_compound
                FROM news_articles
                WHERE symbol IN ({placeholders}) AND fetched_at >= %s
                """
            )
            rows = conn.execute(sql, normalized + [cutoff]).fetchall()

        result: Dict[str, Optional[Dict[str, Any]]] = {s: None for s in symbols}
        articles_by_symbol: Dict[str, list] = {}
        for row in rows:
            sym = row["symbol"]
            if sym not in articles_by_symbol:
                articles_by_symbol[sym] = []
            articles_by_symbol[sym].append({
                "label": row["sentiment_label"],
                "compound": row["sentiment_compound"],
            })

        for orig_sym in symbols:
            norm_sym = _normalize_symbol(orig_sym)
            articles = articles_by_symbol.get(norm_sym, [])
            if not articles:
                continue
            compounds = [a["compound"] for a in articles if a["compound"] is not None]
            if not compounds:
                continue
            avg_compound = sum(compounds) / len(compounds)
            labels = [a["label"] for a in articles if a["label"] is not None]
            pos = labels.count("positive")
            neg = labels.count("negative")
            neu = labels.count("neutral")
            result[orig_sym] = {
                "avg_compound": round(avg_compound, 4),
                "positive": pos,
                "negative": neg,
                "neutral": neu,
                "total_articles": len(articles),
            }
        return result
    finally:
        conn.close()


@retry_on_db_lock()
def cleanup_old_news(retention_seconds: int):
    """Delete news articles older than retention_seconds."""
    conn = get_connection()
    try:
        cutoff = time.time() - retention_seconds
        sql = _adapt_sql("DELETE FROM news_articles WHERE fetched_at < %s")
        deleted = conn.execute(sql, (cutoff,)).rowcount
        conn.commit()
        if deleted:
            logger.info(f"Cleaned up {deleted} old news articles.")
    finally:
        conn.close()


@retry_on_db_lock()
def insert_ohlcv_batch(symbol: str, timeframe: str, candles: List[List]):
    """Insert OHLCV candles into the market_data table, ignoring duplicates."""
    if not candles:
        return
    conn = get_connection()
    try:
        if _backend == "postgresql":
            sql = _adapt_sql(
                """
                INSERT INTO market_data (symbol, timeframe, timestamp, open, high, low, close, volume)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol, timeframe, timestamp) DO NOTHING
                """
            )
        else:
            sql = _adapt_sql(
                """
                INSERT OR IGNORE INTO market_data (symbol, timeframe, timestamp, open, high, low, close, volume)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """
            )

        # Filter out invalid candles before insertion
        valid_candles = []
        for c in candles:
            if len(c) < 6:
                continue
            ts, o, h, l, cl, v = int(c[0]), float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5])
            if o <= 0 or h <= 0 or l <= 0 or cl <= 0 or v < 0:
                continue
            if h < max(o, cl, l) or l > min(o, cl, h):
                continue
            valid_candles.append((symbol, timeframe, ts, o, h, l, cl, v))

        if valid_candles:
            conn.executemany(sql, valid_candles)
            conn.commit()
            logger.debug(f"Inserted {len(valid_candles)} valid OHLCV candles for {symbol} {timeframe}")
    finally:
        conn.close()


def get_ohlcv(symbol: str, timeframe: str, since_ms: int = None, limit: int = 500) -> List[Dict[str, Any]]:
    """Retrieve OHLCV candles from the market_data table.

    When since_ms is provided, returns candles from that timestamp onward
    (oldest first), limited to `limit` rows.

    When since_ms is NOT provided, returns the most RECENT `limit` candles
    in chronological order (oldest first within that recent window).
    """
    conn = get_connection()
    try:
        query = "SELECT timestamp, open, high, low, close, volume FROM market_data WHERE symbol = %s AND timeframe = %s"
        params: list = [symbol, timeframe]
        if since_ms is not None:
            query += " AND timestamp >= %s"
            params.append(since_ms)
            query += " ORDER BY timestamp ASC"
            if limit:
                query += f" LIMIT {int(limit)}"
        else:
            # No since_ms: fetch most recent candles (DESC), then we reverse below
            query += " ORDER BY timestamp DESC"
            if limit:
                query += f" LIMIT {int(limit)}"
        rows = conn.execute(_adapt_sql(query), params).fetchall()
        result = [
            {
                "timestamp": row["timestamp"],
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"],
            }
            for row in rows
        ]
        if since_ms is None:
            result.reverse()  # reverse DESC → chronological (oldest first)
        return result
    finally:
        conn.close()


def get_ohlcv_summary_for_symbols(symbols: List[str], timeframes: List[str], since_ms: int) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Retrieve OHLCV summary (change_pct, high, low, volume, candle_count) for multiple symbols and timeframes.
    Returns a dict: {symbol: {timeframe: summary_dict}}
    """
    if not symbols or not timeframes:
        return {}
    conn = get_connection()
    try:
        result: Dict[str, Dict[str, Dict[str, Any]]] = {s: {} for s in symbols}
        normalized_symbols = [_normalize_symbol(s) for s in symbols]

        for tf in timeframes:
            if _backend == "postgresql":
                sql = _adapt_sql(
                    """
                    WITH bounds AS (
                        SELECT symbol, MIN(timestamp) as min_ts, MAX(timestamp) as max_ts
                        FROM market_data
                        WHERE timeframe = %s AND timestamp >= %s AND symbol = ANY(%s)
                        GROUP BY symbol
                    ),
                    aggs AS (
                        SELECT symbol, MAX(high) as high, MIN(low) as low, SUM(volume) as volume, COUNT(*) as candle_count
                        FROM market_data
                        WHERE timeframe = %s AND timestamp >= %s AND symbol = ANY(%s)
                        GROUP BY symbol
                    )
                    SELECT a.symbol,
                           (SELECT open FROM market_data WHERE symbol = a.symbol AND timeframe = %s AND timestamp = b.min_ts) as open_price,
                           a.high,
                           a.low,
                           (SELECT close FROM market_data WHERE symbol = a.symbol AND timeframe = %s AND timestamp = b.max_ts) as close_price,
                           a.volume,
                           a.candle_count
                    FROM aggs a
                    JOIN bounds b ON a.symbol = b.symbol
                    """
                )
                rows = conn.execute(sql, (tf, since_ms, normalized_symbols, tf, since_ms, normalized_symbols, tf, tf)).fetchall()
            else:
                placeholders = ",".join(["?" for _ in normalized_symbols])
                sql = _adapt_sql(
                    f"""
                    WITH bounds AS (
                        SELECT symbol, MIN(timestamp) as min_ts, MAX(timestamp) as max_ts
                        FROM market_data
                        WHERE timeframe = ? AND timestamp >= ? AND symbol IN ({placeholders})
                        GROUP BY symbol
                    ),
                    aggs AS (
                        SELECT symbol, MAX(high) as high, MIN(low) as low, SUM(volume) as volume, COUNT(*) as candle_count
                        FROM market_data
                        WHERE timeframe = ? AND timestamp >= ? AND symbol IN ({placeholders})
                        GROUP BY symbol
                    )
                    SELECT a.symbol,
                           (SELECT open FROM market_data WHERE symbol = a.symbol AND timeframe = ? AND timestamp = b.min_ts) as open_price,
                           a.high,
                           a.low,
                           (SELECT close FROM market_data WHERE symbol = a.symbol AND timeframe = ? AND timestamp = b.max_ts) as close_price,
                           a.volume,
                           a.candle_count
                    FROM aggs a
                    JOIN bounds b ON a.symbol = b.symbol
                    """
                )
                rows = conn.execute(sql, [tf, since_ms] + normalized_symbols + [tf, since_ms] + normalized_symbols + [tf, tf]).fetchall()

            for row in rows:
                sym = row["symbol"]
                for orig_sym in symbols:
                    if _normalize_symbol(orig_sym) == sym:
                        open_price = row["open_price"]
                        close_price = row["close_price"]
                        if open_price and close_price and row["candle_count"] >= 2:
                            change_pct = ((close_price - open_price) / open_price) * 100 if open_price else 0
                            result[orig_sym][tf] = {
                                "candles": row["candle_count"],
                                "change_pct": round(change_pct, 2),
                                "high": row["high"],
                                "low": row["low"],
                                "volume": row["volume"],
                            }
                        break
        return result
    finally:
        conn.close()


def get_latest_ohlcv_timestamp(symbol: str, timeframe: str) -> Optional[int]:
    """Return the latest timestamp for a symbol/timeframe, or None if no data exists."""
    conn = get_connection()
    try:
        sql = _adapt_sql(
            "SELECT MAX(timestamp) AS ts FROM market_data WHERE symbol = %s AND timeframe = %s"
        )
        row = conn.execute(sql, (symbol, timeframe)).fetchone()
        if row and row["ts"] is not None:
            return row["ts"]
        return None
    finally:
        conn.close()


@retry_on_db_lock()
def save_indicators(symbol: str, timeframe: str, timestamp: int, indicators: Dict[str, Any]):
    """Save computed indicators for a symbol/timeframe (upsert — one row per symbol/timeframe)."""
    conn = get_connection()
    try:
        if _backend == "postgresql":
            sql = _adapt_sql(
                "INSERT INTO indicators (symbol, timeframe, timestamp, indicators_json, computed_at) "
                "VALUES (%s, %s, %s, %s, %s) "
                "ON CONFLICT (symbol, timeframe) DO UPDATE SET "
                "timestamp = EXCLUDED.timestamp, indicators_json = EXCLUDED.indicators_json, "
                "computed_at = EXCLUDED.computed_at"
            )
        else:
            sql = _adapt_sql(
                "INSERT OR REPLACE INTO indicators (symbol, timeframe, timestamp, indicators_json, computed_at) "
                "VALUES (%s, %s, %s, %s, %s)"
            )
        conn.execute(sql, (symbol, timeframe, timestamp, json.dumps(indicators), time.time()))
        conn.commit()
    finally:
        conn.close()


def get_indicators(symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
    """Retrieve the latest computed indicators for a symbol/timeframe from DB, or None if not found.

    The returned dict includes a ``_indicator_timestamp`` key (ms epoch) indicating
    the timestamp of the latest candle used to compute the indicators, so callers
    can detect stale data.
    """
    conn = get_connection()
    try:
        sql = _adapt_sql(
            "SELECT indicators_json, timestamp FROM indicators WHERE symbol = %s AND timeframe = %s"
        )
        row = conn.execute(sql, (symbol, timeframe)).fetchone()
        if row:
            data = json.loads(row["indicators_json"])
            data["_indicator_timestamp"] = row["timestamp"]
            return data
        return None
    finally:
        conn.close()


def get_indicators_for_symbols(symbols: List[str], timeframes: List[str]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Retrieve the latest computed indicators for multiple symbols and timeframes in a single query.
    Returns a dict: {symbol: {timeframe: indicators_dict}}
    """
    if not symbols or not timeframes:
        return {}
    conn = get_connection()
    try:
        # Build a flat list of parameters for the IN clause
        pairs = []
        for sym in symbols:
            for tf in timeframes:
                pairs.append(sym)
                pairs.append(tf)
        placeholders = ",".join(["(%s,%s)"] * (len(symbols) * len(timeframes)))
        sql = _adapt_sql(
            f"""
            SELECT symbol, timeframe, indicators_json, timestamp
            FROM indicators
            WHERE (symbol, timeframe) IN ({placeholders})
            """
        )
        rows = conn.execute(sql, pairs).fetchall()

        result: Dict[str, Dict[str, Dict[str, Any]]] = {s: {} for s in symbols}
        for row in rows:
            sym = row["symbol"]
            tf = row["timeframe"]
            if sym in result:
                data = json.loads(row["indicators_json"])
                data["_indicator_timestamp"] = row["timestamp"]
                result[sym][tf] = data
        return result
    finally:
        conn.close()


@retry_on_db_lock()
def save_quotes_batch(quotes: Dict[str, Dict[str, Any]]):
    """Save or update multiple quotes in the database."""
    if not quotes:
        return
    conn = get_connection()
    try:
        now = time.time()
        if _backend == "postgresql":
            sql = _adapt_sql(
                """
                INSERT INTO quotes (symbol, last, bid, ask, volume, change_24h, percentage, quotevolume, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol) DO UPDATE SET
                    last = EXCLUDED.last,
                    bid = COALESCE(EXCLUDED.bid, quotes.bid),
                    ask = COALESCE(EXCLUDED.ask, quotes.ask),
                    volume = COALESCE(EXCLUDED.volume, quotes.volume),
                    change_24h = EXCLUDED.change_24h,
                    percentage = EXCLUDED.percentage,
                    quotevolume = COALESCE(EXCLUDED.quotevolume, quotes.quotevolume),
                    updated_at = EXCLUDED.updated_at
                """
            )
        else:
            sql = _adapt_sql(
                """
                INSERT OR REPLACE INTO quotes (symbol, last, bid, ask, volume, change_24h, percentage, quotevolume, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
            )
        rows = []
        for sym, q in quotes.items():
            if q.get("last") is not None:
                rows.append((
                    sym, q.get("last"), q.get("bid"), q.get("ask"),
                    q.get("volume"), q.get("change_24h"), q.get("percentage"),
                    q.get("quoteVolume"), now
                ))
        if rows:
            conn.executemany(sql, rows)
            conn.commit()
            logger.debug(f"Saved {len(rows)} quotes to database")
    finally:
        conn.close()


def get_quotes_from_db(symbols: List[str], max_age_seconds: int = 86400) -> Dict[str, Dict[str, Any]]:
    """Retrieve quotes from the database. Returns only quotes newer than max_age_seconds."""
    if not symbols:
        return {}
    conn = get_connection()
    try:
        cutoff = time.time() - max_age_seconds
        if _backend == "postgresql":
            sql = _adapt_sql(
                """
                SELECT symbol, last, bid, ask, volume, change_24h, percentage, quotevolume, updated_at
                FROM quotes WHERE symbol = ANY(%s) AND updated_at >= %s
                """
            )
            rows = conn.execute(sql, (symbols, cutoff)).fetchall()
        else:
            placeholders = ",".join(["?" for _ in symbols])
            sql = _adapt_sql(
                f"""
                SELECT symbol, last, bid, ask, volume, change_24h, percentage, quotevolume, updated_at
                FROM quotes WHERE symbol IN ({placeholders}) AND updated_at >= %s
                """
            )
            rows = conn.execute(sql, symbols + [cutoff]).fetchall()

        result = {}
        for row in rows:
            result[row["symbol"]] = {
                "last": row["last"],
                "bid": row["bid"],
                "ask": row["ask"],
                "volume": row["volume"],
                "change_24h": row["change_24h"],
                "percentage": row["percentage"],
                "quoteVolume": row["quotevolume"],
                "last_update": int(row["updated_at"] * 1000) if row["updated_at"] else None,
                "source": "db_quotes",
            }
        return result
    finally:
        conn.close()


def get_latest_close_prices(symbols: List[str]) -> Dict[str, Dict[str, Any]]:
    """Get the latest 2 close prices and volume for multiple symbols from the market_data table.
    Used as a fallback when yfinance quotes are unavailable.

    Queries the most recent 2 candles across ALL timeframes (not just '1d')
    to ensure a price is available even if only long-term timeframes
    (5Y, 1Y, 1w, etc.) have been downloaded by background tasks.

    The market_data table stores symbols as full pairs (e.g., 'ENI.MI/EUR'),
    but get_quotes passes base symbols (e.g., 'ENI.MI'). This function
    queries all candles and filters by the base part in Python to
    handle the format mismatch.

    Returns a dict mapping base symbol -> {"last": float, "prev_close": float|None, "volume": float|None}.
    """
    if not symbols:
        return {}

    # Try to get from Redis cache
    try:
        redis_client = get_redis_client()
        cache_key = "latest_close_prices:" + hashlib.md5(json.dumps(sorted(symbols)).encode()).hexdigest()
        cached = redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
    except Exception:
        pass

    # Normalize input symbols to base form (strip /currency suffix)
    base_symbols = set(s.split('/')[0] for s in symbols)
    # Construct full pair symbols for exact matching in the database
    full_pairs = [f"{bs}/{settings.BASE_CURRENCY}" for bs in base_symbols]

    conn = get_connection()
    try:
        if _backend == "postgresql":
            sql = _adapt_sql(
                """
                WITH RankedCandles AS (
                    SELECT symbol, close, volume, timestamp, timeframe,
                           ROW_NUMBER() OVER (PARTITION BY symbol, timeframe ORDER BY timestamp DESC) as rn
                    FROM market_data
                    WHERE symbol = ANY(%s)
                ),
                LatestTimeframe AS (
                    SELECT symbol, timeframe
                    FROM (
                        SELECT symbol, timeframe, timestamp,
                               ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY timestamp DESC) as tf_rn
                        FROM RankedCandles
                        WHERE rn = 1
                    ) t
                    WHERE tf_rn = 1
                )
                SELECT r.symbol, r.close, r.volume, r.timestamp, r.timeframe
                FROM RankedCandles r
                JOIN LatestTimeframe l ON r.symbol = l.symbol AND r.timeframe = l.timeframe
                WHERE r.rn <= 2
                ORDER BY r.symbol, r.timestamp DESC
                """
            )
            rows = conn.execute(sql, (full_pairs,)).fetchall()
        else:
            placeholders = ",".join(["?" for _ in full_pairs])
            sql = _adapt_sql(
                f"""
                WITH RankedCandles AS (
                    SELECT symbol, close, volume, timestamp, timeframe,
                           ROW_NUMBER() OVER (PARTITION BY symbol, timeframe ORDER BY timestamp DESC) as rn
                    FROM market_data
                    WHERE symbol IN ({placeholders})
                ),
                LatestTimeframe AS (
                    SELECT symbol, timeframe
                    FROM (
                        SELECT symbol, timeframe, timestamp,
                               ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY timestamp DESC) as tf_rn
                        FROM RankedCandles
                        WHERE rn = 1
                    ) t
                    WHERE tf_rn = 1
                )
                SELECT r.symbol, r.close, r.volume, r.timestamp, r.timeframe
                FROM RankedCandles r
                JOIN LatestTimeframe l ON r.symbol = l.symbol AND r.timeframe = l.timeframe
                WHERE r.rn <= 2
                ORDER BY r.symbol, r.timestamp DESC
                """
            )
            rows = conn.execute(sql, full_pairs).fetchall()

        result = {}
        for row in rows:
            db_symbol = row["symbol"]
            # Strip /currency suffix to get base symbol for matching
            db_base = db_symbol.split('/')[0]
            if db_base in base_symbols and row["close"] is not None and row["close"] > 0:
                if db_base not in result:
                    # Latest row (due to ORDER BY timestamp DESC)
                    result[db_base] = {
                        "last": float(row["close"]),
                        "volume": float(row["volume"]) if row["volume"] is not None else None,
                        "prev_close": None,
                        "candle_timestamp": row["timestamp"],
                        "timeframe": row["timeframe"],
                    }
                else:
                    # Second row (older timestamp) — only use as prev_close
                    # if it's from the SAME timeframe as the latest candle.
                    # Otherwise the percentage change would be meaningless.
                    if row["timeframe"] == result[db_base]["timeframe"]:
                        result[db_base]["prev_close"] = float(row["close"])

        # --- Also fetch the latest 2 daily (1d) candles to compute a proper 24h change.
        # The main query above gets the latest candles across ALL timeframes, which may
        # mix timeframes (e.g., 5Y and 1M), making the prev_close / change_24h
        # calculation incorrect.  We specifically query 1d candles so that prev_close
        # always represents the previous trading day's close.
        if base_symbols:
            daily_pairs = [f"{bs}/{settings.BASE_CURRENCY}" for bs in base_symbols]
            if _backend == "postgresql":
                sql_daily = _adapt_sql(
                    """
                    WITH RankedDaily AS (
                        SELECT symbol, close, timestamp,
                               ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY timestamp DESC) as rn
                        FROM market_data
                        WHERE symbol = ANY(%s) AND timeframe = '1d'
                    )
                    SELECT symbol, close, timestamp
                    FROM RankedDaily
                    WHERE rn <= 2
                    ORDER BY symbol, timestamp DESC
                    """
                )
                daily_rows = conn.execute(sql_daily, (daily_pairs,)).fetchall()
            else:
                placeholders = ",".join(["?" for _ in daily_pairs])
                sql_daily = _adapt_sql(
                    f"""
                    WITH RankedDaily AS (
                        SELECT symbol, close, timestamp,
                               ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY timestamp DESC) as rn
                        FROM market_data
                        WHERE symbol IN ({placeholders}) AND timeframe = '1d'
                    )
                    SELECT symbol, close, timestamp
                    FROM RankedDaily
                    WHERE rn <= 2
                    ORDER BY symbol, timestamp DESC
                    """
                )
                daily_rows = conn.execute(sql_daily, daily_pairs).fetchall()

            # Build a dict: base_symbol -> previous daily close (2nd latest 1d candle)
            daily_seen: set = set()
            daily_prev_close: Dict[str, float] = {}
            for row in daily_rows:
                db_symbol = row["symbol"]
                db_base = db_symbol.split('/')[0]
                if db_base in base_symbols and row["close"] is not None and row["close"] > 0:
                    if db_base not in daily_seen:
                        # First (latest) 1d candle — mark as seen, skip
                        daily_seen.add(db_base)
                    else:
                        # Second 1d candle — this is the previous trading day's close
                        daily_prev_close[db_base] = float(row["close"])

            # Override prev_close with daily data when available
            for db_base, prev_close in daily_prev_close.items():
                if db_base in result:
                    result[db_base]["prev_close"] = prev_close
    finally:
        conn.close()

    # Save to Redis cache
    try:
        redis_client = get_redis_client()
        cache_key = "latest_close_prices:" + hashlib.md5(json.dumps(sorted(symbols)).encode()).hexdigest()
        redis_client.setex(cache_key, 60, json.dumps(result))
    except Exception:
        pass

    return result


@retry_on_db_lock()
def save_discovered_symbol(symbol: str, isin: Optional[str], asset_type: str, name: Optional[str] = None, maturity: Optional[str] = None, coupon: Optional[float] = None, country: Optional[str] = None):
    """Insert or update a discovered symbol with its ISIN, maturity, and coupon.

    Uses COALESCE to preserve existing ISINs — if isin is None, the existing
    ISIN in the database is kept (not overwritten with NULL). This is critical
    because yfinance can return None when rate-limited.
    """
    if isin:
        isin = isin.strip()
        if isin == '-' or not isin:
            isin = None
    if name is not None:
        name = name.strip()
        if not name:
            name = None
    conn = get_connection()
    try:
        sql = _adapt_sql(
            "INSERT INTO discovered_symbols (symbol, isin, asset_type, name, maturity, coupon, country, discovered_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (symbol) DO UPDATE SET "
            "isin = COALESCE(EXCLUDED.isin, discovered_symbols.isin), "
            "asset_type = COALESCE(NULLIF(EXCLUDED.asset_type, ''), discovered_symbols.asset_type), "
            "name = COALESCE(NULLIF(EXCLUDED.name, ''), NULLIF(discovered_symbols.name, '')), "
            "maturity = COALESCE(EXCLUDED.maturity, discovered_symbols.maturity), "
            "coupon = COALESCE(EXCLUDED.coupon, discovered_symbols.coupon), "
            "country = COALESCE(EXCLUDED.country, discovered_symbols.country), "
            "discovered_at = EXCLUDED.discovered_at"
        )
        conn.execute(sql, (symbol, isin, asset_type, name, maturity, coupon, country, time.time()))
        conn.commit()
    finally:
        conn.close()


@retry_on_db_lock()
def save_discovered_symbols_batch(symbols: List[Dict[str, Any]]):
    """Batch insert or update discovered symbols.

    Uses COALESCE to preserve existing ISINs — if isin is None, the existing
    ISIN in the database is kept (not overwritten with NULL).
    """
    if not symbols:
        return
    conn = get_connection()
    try:
        now = time.time()
        for s in symbols:
            isin_val = s.get("isin")
            if isin_val:
                isin_val = isin_val.strip()
                if isin_val == '-' or not isin_val:
                    s["isin"] = None
                else:
                    s["isin"] = isin_val
            name_val = s.get("name")
            if name_val is not None:
                name_val = name_val.strip()
                if not name_val:
                    s["name"] = None
                else:
                    s["name"] = name_val
        sql = _adapt_sql(
            "INSERT INTO discovered_symbols (symbol, isin, asset_type, name, maturity, coupon, country, discovered_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (symbol) DO UPDATE SET "
            "isin = COALESCE(EXCLUDED.isin, discovered_symbols.isin), "
            "asset_type = COALESCE(NULLIF(EXCLUDED.asset_type, ''), discovered_symbols.asset_type), "
            "name = COALESCE(NULLIF(EXCLUDED.name, ''), NULLIF(discovered_symbols.name, '')), "
            "maturity = COALESCE(EXCLUDED.maturity, discovered_symbols.maturity), "
            "coupon = COALESCE(EXCLUDED.coupon, discovered_symbols.coupon), "
            "country = COALESCE(EXCLUDED.country, discovered_symbols.country), "
            "discovered_at = EXCLUDED.discovered_at"
        )
        rows = [(s["symbol"], s.get("isin"), s.get("asset_type", ""), s.get("name", ""), s.get("maturity"), s.get("coupon"), s.get("country"), now) for s in symbols]
        conn.executemany(sql, rows)
        conn.commit()
    finally:
        conn.close()


def get_all_discovered_symbols() -> List[Dict[str, Any]]:
    """Return all discovered symbols from the database."""
    conn = get_connection()
    try:
        sql = _adapt_sql("SELECT symbol, isin, asset_type, name, maturity, coupon, country FROM discovered_symbols")
        rows = conn.execute(sql).fetchall()
        return [
            {
                "symbol": row["symbol"],
                "isin": row["isin"],
                "asset_type": row["asset_type"],
                "name": row["name"],
                "maturity": row["maturity"],
                "coupon": row["coupon"],
                "country": row["country"],
            }
            for row in rows
        ]
    finally:
        conn.close()

@retry_on_db_lock()
def insert_signal(signal: Dict[str, Any]):
    """Insert a trading signal into the signals table."""
    conn = get_connection()
    try:
        sql = _adapt_sql(
            """
            INSERT INTO signals (
                symbol, display_symbol, stock_name, timeframe, action, confidence, reasoning,
                strategy_type, model_type, llm_provider, llm_model, trade_amount, base_currency,
                timestamp, entry_condition, stop_loss, take_profit, position_size_fraction,
                trailing_stop, trailing_stop_distance_pct, max_hold_time_seconds,
                cooldown_after_loss_seconds, order_type, limit_price
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
        )
        conn.execute(sql, (
            signal.get("symbol"),
            signal.get("display_symbol"),
            signal.get("stock_name"),
            signal.get("timeframe"),
            signal.get("action"),
            signal.get("confidence"),
            signal.get("reasoning"),
            signal.get("strategy_type"),
            signal.get("model_type"),
            signal.get("llm_provider"),
            signal.get("llm_model"),
            signal.get("trade_amount"),
            signal.get("base_currency"),
            signal.get("timestamp"),
            signal.get("entry_condition"),
            signal.get("stop_loss"),
            signal.get("take_profit"),
            signal.get("position_size_fraction"),
            1 if signal.get("trailing_stop") else 0,
            signal.get("trailing_stop_distance_pct"),
            signal.get("max_hold_time_seconds"),
            signal.get("cooldown_after_loss_seconds"),
            signal.get("order_type"),
            signal.get("limit_price"),
        ))
        conn.commit()
    finally:
        conn.close()


def get_signals(limit: int = 5, offset: int = 0) -> Dict[str, Any]:
    """Retrieve paginated BUY/SELL signals in descending order by time."""
    conn = get_connection()
    try:
        count_row = conn.execute(
            "SELECT COUNT(*) as total FROM signals WHERE action IN ('BUY', 'SELL')"
        ).fetchone()
        total = count_row["total"] if count_row else 0

        sql = _adapt_sql(
            """
            SELECT symbol, display_symbol, stock_name, timeframe, action, confidence, reasoning,
                   strategy_type, model_type, llm_provider, llm_model, trade_amount, base_currency,
                   timestamp, entry_condition, stop_loss, take_profit, position_size_fraction,
                   trailing_stop, trailing_stop_distance_pct, max_hold_time_seconds,
                   cooldown_after_loss_seconds, order_type, limit_price
            FROM signals
            WHERE action IN ('BUY', 'SELL')
            ORDER BY timestamp DESC
            LIMIT %s OFFSET %s
            """
        )
        rows = conn.execute(sql, (limit, offset)).fetchall()

        signals = []
        for row in rows:
            signals.append({
                "symbol": row["symbol"],
                "display_symbol": row["display_symbol"],
                "stock_name": row["stock_name"],
                "timeframe": row["timeframe"],
                "action": row["action"],
                "confidence": row["confidence"],
                "reasoning": row["reasoning"],
                "strategy_type": row["strategy_type"],
                "model_type": row["model_type"],
                "llm_provider": row["llm_provider"],
                "llm_model": row["llm_model"],
                "trade_amount": row["trade_amount"],
                "base_currency": row["base_currency"],
                "timestamp": row["timestamp"],
                "entry_condition": row["entry_condition"],
                "stop_loss": row["stop_loss"],
                "take_profit": row["take_profit"],
                "position_size_fraction": row["position_size_fraction"],
                "trailing_stop": bool(row["trailing_stop"]),
                "trailing_stop_distance_pct": row["trailing_stop_distance_pct"],
                "max_hold_time_seconds": row["max_hold_time_seconds"],
                "cooldown_after_loss_seconds": row["cooldown_after_loss_seconds"],
                "order_type": row["order_type"],
                "limit_price": row["limit_price"],
            })
        return {"signals": signals, "total": total}
    finally:
        conn.close()


def get_isin_from_db(symbol: str) -> Optional[str]:
    """Return the ISIN for a symbol from the database, or None if not found."""
    conn = get_connection()
    try:
        sql = _adapt_sql("SELECT isin FROM discovered_symbols WHERE symbol = %s")
        row = conn.execute(sql, (symbol,)).fetchone()
        if row and row["isin"]:
            return row["isin"]
        return None
    finally:
        conn.close()


def get_symbol_name_from_db(symbol: str) -> Optional[str]:
    """Return the name for a symbol from the discovered_symbols table, or None if not found."""
    base = symbol.split("/")[0] if "/" in symbol else symbol
    # Strip ticker suffix for DB lookup (DB stores base symbols without suffix)
    suffix = settings.TICKER_SUFFIX
    if suffix and base.endswith(suffix):
        base = base[:-len(suffix)]
    conn = get_connection()
    try:
        sql = _adapt_sql("SELECT name FROM discovered_symbols WHERE symbol = %s AND name IS NOT NULL AND name != ''")
        row = conn.execute(sql, (base,)).fetchone()
        if row and row["name"]:
            return row["name"]
        return None
    finally:
        conn.close()


def get_discovered_symbols_with_names() -> set:
    """Return a set of base symbols from discovered_symbols that have a non-empty name."""
    conn = get_connection()
    try:
        sql = _adapt_sql("SELECT symbol FROM discovered_symbols WHERE name IS NOT NULL AND name != ''")
        rows = conn.execute(sql).fetchall()
        return {row["symbol"] for row in rows}
    finally:
        conn.close()


def get_isin_map_from_db(symbols: List[str]) -> Dict[str, Optional[str]]:
    """Return a dict mapping symbol -> ISIN for multiple symbols in a single query."""
    if not symbols:
        return {}
    conn = get_connection()
    try:
        if _backend == "postgresql":
            sql = _adapt_sql("SELECT symbol, isin FROM discovered_symbols WHERE symbol = ANY(%s)")
            rows = conn.execute(sql, (symbols,)).fetchall()
        else:
            placeholders = ",".join(["?" for _ in symbols])
            sql = _adapt_sql(f"SELECT symbol, isin FROM discovered_symbols WHERE symbol IN ({placeholders})")
            rows = conn.execute(sql, symbols).fetchall()
        result = {s: None for s in symbols}
        for row in rows:
            result[row["symbol"]] = row["isin"]
        return result
    finally:
        conn.close()


def get_btp_details_from_db(symbols: List[str]) -> Dict[str, Dict[str, Optional[Any]]]:
    """Return maturity, coupon, and name for BTP symbols from discovered_symbols.
    Returns a dict: {symbol: {maturity, coupon, name}}.
    """
    if not symbols:
        return {}
    conn = get_connection()
    try:
        if _backend == "postgresql":
            sql = _adapt_sql("SELECT symbol, maturity, coupon, name FROM discovered_symbols WHERE symbol = ANY(%s)")
            rows = conn.execute(sql, (symbols,)).fetchall()
        else:
            placeholders = ",".join(["?" for _ in symbols])
            sql = _adapt_sql(f"SELECT symbol, maturity, coupon, name FROM discovered_symbols WHERE symbol IN ({placeholders})")
            rows = conn.execute(sql, symbols).fetchall()
        result = {}
        for row in rows:
            result[row["symbol"]] = {
                "maturity": row["maturity"],
                "coupon": row["coupon"],
                "name": row["name"],
            }
        return result
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# LLM Metrics helpers
# ---------------------------------------------------------------------------

@retry_on_db_lock()
def save_llm_metrics(metrics: dict):
    """Insert a row of LLM call metrics into the llm_metrics table."""
    conn = get_connection()
    try:
        sql = _adapt_sql(
            """
            INSERT INTO llm_metrics (
                timestamp, provider, model, model_type,
                prompt_tokens, completion_tokens, total_tokens,
                cache_hit, latency_ms, error
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
        )
        conn.execute(sql, (
            metrics.get("timestamp", time.time()),
            metrics.get("provider", ""),
            metrics.get("model", ""),
            metrics.get("model_type", ""),
            metrics.get("prompt_tokens", 0),
            metrics.get("completion_tokens", 0),
            metrics.get("total_tokens", 0),
            1 if metrics.get("cache_hit") else 0,
            metrics.get("latency_ms", 0),
            metrics.get("error"),
        ))
        conn.commit()
    finally:
        conn.close()


def get_llm_metrics_summary() -> dict:
    """Return aggregated LLM metrics for the dashboard."""
    conn = get_connection()
    try:
        # Total calls and tokens
        row = conn.execute(
            "SELECT COUNT(*) as total_calls, "
            "COALESCE(SUM(prompt_tokens),0) as total_prompt_tokens, "
            "COALESCE(SUM(completion_tokens),0) as total_completion_tokens, "
            "COALESCE(SUM(total_tokens),0) as total_tokens, "
            "COALESCE(SUM(cache_hit),0) as cache_hits "
            "FROM llm_metrics"
        ).fetchone()
        total_calls = row["total_calls"] if row else 0
        total_prompt_tokens = row["total_prompt_tokens"] if row else 0
        total_completion_tokens = row["total_completion_tokens"] if row else 0
        total_tokens = row["total_tokens"] if row else 0
        cache_hits = row["cache_hits"] if row else 0
        cache_hit_rate = (cache_hits / total_calls * 100) if total_calls > 0 else 0.0

        # Per-model breakdown
        per_model_rows = conn.execute(
            "SELECT provider, model, model_type, COUNT(*) as calls, "
            "SUM(total_tokens) as tokens, AVG(latency_ms) as avg_latency "
            "FROM llm_metrics GROUP BY provider, model, model_type"
        ).fetchall()
        per_model = []
        for r in per_model_rows:
            per_model.append({
                "provider": r["provider"],
                "model": r["model"],
                "model_type": r["model_type"],
                "calls": r["calls"],
                "tokens": r["tokens"],
                "avg_latency_ms": round(r["avg_latency"], 2) if r["avg_latency"] else 0,
            })

        # Recent calls (last 20)
        recent_rows = conn.execute(
            "SELECT timestamp, provider, model, model_type, prompt_tokens, completion_tokens, "
            "total_tokens, cache_hit, latency_ms, error "
            "FROM llm_metrics ORDER BY timestamp DESC LIMIT 20"
        ).fetchall()
        recent_calls = []
        for r in recent_rows:
            recent_calls.append({
                "timestamp": r["timestamp"],
                "provider": r["provider"],
                "model": r["model"],
                "model_type": r["model_type"],
                "prompt_tokens": r["prompt_tokens"],
                "completion_tokens": r["completion_tokens"],
                "total_tokens": r["total_tokens"],
                "cache_hit": bool(r["cache_hit"]),
                "latency_ms": r["latency_ms"],
                "error": r["error"],
            })

        return {
            "total_calls": total_calls,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "cache_hit_rate": round(cache_hit_rate, 2),
            "per_model": per_model,
            "recent_calls": recent_calls,
        }
    finally:
        conn.close()


def get_llm_metrics_timeseries(hours: int = 24) -> List[Dict[str, Any]]:
    """Return hourly aggregated LLM metrics for the last N hours."""
    conn = get_connection()
    try:
        cutoff = time.time() - hours * 3600
        if _backend == "postgresql":
            sql = _adapt_sql(
                """
                SELECT
                    date_trunc('hour', to_timestamp(timestamp)) AS hour,
                    COUNT(*) as calls,
                    SUM(total_tokens) as tokens,
                    AVG(latency_ms) as avg_latency,
                    SUM(cache_hit) as cache_hits
                FROM llm_metrics
                WHERE timestamp >= %s
                GROUP BY hour
                ORDER BY hour ASC
                """
            )
        else:
            # SQLite: group by integer division of timestamp / 3600
            sql = _adapt_sql(
                """
                SELECT
                    (CAST(timestamp / 3600 AS INTEGER) * 3600) AS hour,
                    COUNT(*) as calls,
                    SUM(total_tokens) as tokens,
                    AVG(latency_ms) as avg_latency,
                    SUM(cache_hit) as cache_hits
                FROM llm_metrics
                WHERE timestamp >= %s
                GROUP BY hour
                ORDER BY hour ASC
                """
            )
        rows = conn.execute(sql, (cutoff,)).fetchall()
        result = []
        for row in rows:
            hour_ts = row["hour"]
            calls = row["calls"]
            cache_hits = row["cache_hits"]
            result.append({
                "hour": hour_ts,
                "calls": calls,
                "tokens": row["tokens"],
                "avg_latency_ms": round(row["avg_latency"], 2) if row["avg_latency"] else 0,
                "cache_hit_rate": round((cache_hits / calls * 100) if calls > 0 else 0, 1),
            })
        return result
    finally:
        conn.close()


def close_pool():
    """Close the PostgreSQL connection pool if it exists."""
    global _pg_pool
    if _pg_pool is not None:
        _pg_pool.closeall()
        _pg_pool = None
        logger.info("PostgreSQL connection pool closed.")
