import sqlite3
import json
import os
import logging
import time
import functools
from datetime import datetime
import hashlib
import threading
import weakref
from typing import Dict, List, Any, Optional, Tuple
from contextlib import contextmanager

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
    import psycopg
    from psycopg import rows as pg_rows
    from psycopg_pool import ConnectionPool

    def _configure_connection(conn):
        """Validate connection without leaving an open transaction."""
        # Use execute with autocommit=True (set in kwargs below) - no transaction started
        # This works on all psycopg3 versions
        conn.execute("SELECT 1")
        conn.execute("SET search_path TO public")

    _pg_pool = ConnectionPool(
        kwargs={
            "host": settings.DB_HOST,
            "port": settings.DB_PORT,
            "dbname": settings.DB_NAME,
            "user": settings.DB_USER,
            "password": settings.DB_PASSWORD,
            "autocommit": True,  # Health check runs in autocommit mode
        },
        min_size=5,
        max_size=20,
        timeout=30.0,
        max_idle=60.0,
        reconnect_timeout=300.0,
        configure=_configure_connection,
        open=True,
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
    """Wraps a psycopg3 connection so that close() returns it to the pool."""
    def __init__(self, conn, pool):
        self._conn = conn
        self._pool = pool
        # Use dict_row for dict-like row access (replaces RealDictCursor)
        self._conn.row_factory = pg_rows.dict_row

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def close(self):
        # Ensure any pending transaction is rolled back before returning to pool
        # This prevents "INTRANS" state leaks when exceptions occur before commit()
        try:
            self._conn.rollback()
        except psycopg.Error:
            pass  # Ignore rollback errors (e.g., no transaction in progress)
        try:
            self._pool.putconn(self._conn)
        except Exception as e:
            logger.warning(f"Failed to return connection to pool: {e}")
            # Try to close the connection directly as fallback
            try:
                self._conn.close()
            except psycopg.Error:
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
        with self._conn.cursor() as cur:
            cur.executemany(sql, params_list)


_sqlite_local = threading.local()

class _SqliteConnectionWrapper:
    """Wraps a sqlite3 connection so that close() is a no-op, allowing reuse.
    
    A finalizer is registered to ensure the underlying SQLite connection is
    closed when the thread dies and the thread-local wrapper is garbage collected.
    """
    _all_wrappers = weakref.WeakSet()

    def __init__(self, conn):
        self._conn = conn
        self._closed = False
        self._finalizer = weakref.finalize(self, self._close_conn, conn)
        _SqliteConnectionWrapper._all_wrappers.add(self)

    @staticmethod
    def _close_conn(conn):
        try:
            conn.close()
        except sqlite3.Error:
            pass

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def close(self):
        # Do not close the persistent thread-local connection during normal operation.
        # The finalizer handles cleanup when the thread dies.
        pass

    def force_close(self):
        """Explicitly close the underlying connection and clear the thread-local reference."""
        if self._closed:
            return
        self._closed = True
        self._finalizer()  # runs _close_conn, which closes the sqlite connection
        # Clear the thread-local attribute if it still points to this wrapper.
        # This only works when called from the owning thread; cross-thread closure
        # is handled by the _closed check in get_connection().
        if getattr(_sqlite_local, 'connection', None) is self:
            del _sqlite_local.connection

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
        try:
            stats = _pg_pool.get_stats()
            if stats.get("pool_size", 0) >= _pg_pool._max_size - 2:
                logger.warning(f"PostgreSQL connection pool near exhaustion: {stats}")
        except Exception:
            pass
        conn = _pg_pool.getconn()
        return _PgConnectionWrapper(conn, _pg_pool)
    else:
        if hasattr(_sqlite_local, 'connection'):
            wrapper = _sqlite_local.connection
            if wrapper._closed:
                del _sqlite_local.connection
            else:
                return wrapper
        os.makedirs(os.path.dirname(settings.DATABASE_PATH), exist_ok=True)
        conn = sqlite3.connect(settings.DATABASE_PATH, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        _sqlite_local.connection = _SqliteConnectionWrapper(conn)
        return _sqlite_local.connection


@contextmanager
def get_connection_ctx():
    """Context manager for database connections.
    
    For PostgreSQL: gets a connection from the pool and returns it on exit.
    For SQLite: returns the thread-local connection (no-op on exit).
    
    Usage:
        with get_connection_ctx() as conn:
            conn.execute(...)
            conn.commit()
    """
    conn = get_connection()
    try:
        yield conn
    finally:
        if _backend == "postgresql":
            conn.close()


def close_all_sqlite_connections():
    """Force-close all active SQLite connections across all threads.

    Useful when the database file needs to be replaced (e.g., during a reset
    or migration). After calling this, each thread will automatically create
    a fresh connection on its next get_connection() call.
    """
    for wrapper in list(_SqliteConnectionWrapper._all_wrappers):
        wrapper.force_close()


def vacuum_database():
    """Close all SQLite connections and run VACUUM to reclaim space."""
    if _backend != "sqlite":
        logger.info("VACUUM is only supported for SQLite backend.")
        return
    close_all_sqlite_connections()
    conn = sqlite3.connect(settings.DATABASE_PATH, timeout=30)
    try:
        conn.execute("VACUUM")
        logger.info("SQLite VACUUM completed successfully.")
    except sqlite3.Error as e:
        logger.error(f"SQLite VACUUM failed: {e}")
    finally:
        conn.close()


def _normalize_symbol(symbol: str) -> str:
    """Extract the base symbol from a trading pair (e.g., 'AAPL/USD' -> 'AAPL')."""
    return symbol.split("/")[0] if "/" in symbol else symbol


# ---------------------------------------------------------------------------
# Retry decorator (handles both SQLite locks and PostgreSQL deadlocks)
# ---------------------------------------------------------------------------
def retry_on_db_lock(max_retries=2, initial_delay=0.1):
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
                            delay = min(initial_delay * (2 ** attempt), 1.0)
                            time.sleep(delay)
                            continue
                    raise
                except psycopg.Error as e:
                    # Retry on PostgreSQL deadlock (40P01) or serialization failure (40001)
                    sqlstate = e.diag.sqlstate if e.diag else None
                    if sqlstate in ('40P01', '40001'):
                        last_exc = e
                        if attempt < max_retries:
                            delay = min(initial_delay * (2 ** attempt), 1.0)
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

    All missing migrations are applied in a single transaction to ensure
    atomicity. If any migration fails, the entire transaction is rolled back.
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
        ("discovered_symbols", "manual_isin", "ALTER TABLE discovered_symbols ADD COLUMN manual_isin TEXT"),
        ("discovered_symbols", "candle_count", "ALTER TABLE discovered_symbols ADD COLUMN candle_count INTEGER NOT NULL DEFAULT 0"),
        ("llm_metrics", "request_type", "ALTER TABLE llm_metrics ADD COLUMN request_type TEXT"),
        ("llm_metrics", "is_fallback", "ALTER TABLE llm_metrics ADD COLUMN is_fallback INTEGER NOT NULL DEFAULT 0"),
        ("dividends", "reinvested", "ALTER TABLE dividends ADD COLUMN reinvested INTEGER NOT NULL DEFAULT 0"),
        ("llm_decision_quality", "is_fallback", "ALTER TABLE llm_decision_quality ADD COLUMN is_fallback INTEGER NOT NULL DEFAULT 0"),
    ]

    conn = get_connection()
    try:
        missing_migrations = []
        for table, column, sql in migrations:
            existing = _get_existing_columns(table)
            if column not in existing:
                missing_migrations.append(sql)

        if not missing_migrations:
            return

        try:
            for sql in missing_migrations:
                conn.execute(_adapt_sql(sql))
            conn.commit()
            logger.info(f"Successfully applied {len(missing_migrations)} database migrations.")
        except (sqlite3.Error, psycopg.Error) as e:
            conn.rollback()
            logger.error(f"Database migration failed, rolling back: {type(e).__name__}: {e}")
            raise
    finally:
        conn.close()


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
        CREATE TABLE IF NOT EXISTS latest_close_prices (
            symbol TEXT PRIMARY KEY,
            last REAL,
            prev_close REAL,
            volume REAL,
            candle_timestamp {bigint_type},
            timeframe TEXT,
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
            manual_isin TEXT,
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
            error TEXT,
            request_type TEXT,
            is_fallback INTEGER NOT NULL DEFAULT 0
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp DESC)",
        "CREATE INDEX IF NOT EXISTS idx_backtest_results_symbol ON backtest_results(symbol)",
        "CREATE INDEX IF NOT EXISTS idx_backtest_results_symbol_tf_hash ON backtest_results(symbol, timeframe, params_hash, created_at DESC)",
        "CREATE INDEX IF NOT EXISTS idx_backtest_results_created_at ON backtest_results(created_at)",
        f"""
        CREATE TABLE IF NOT EXISTS llm_decision_quality (
            id {pk_type},
            symbol TEXT NOT NULL,
            timestamp {bigint_type} NOT NULL,
            action TEXT NOT NULL,
            entry_price REAL,
            timeframe TEXT,
            outcome_price REAL,
            outcome_timestamp {bigint_type},
            outcome_profitable INTEGER,
            evaluated INTEGER DEFAULT 0,
            is_fallback INTEGER NOT NULL DEFAULT 0
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_llm_decision_quality_timestamp ON llm_decision_quality(timestamp)",
        f"""
        CREATE TABLE IF NOT EXISTS dividends (
            id {pk_type},
            symbol TEXT NOT NULL,
            ex_date TEXT NOT NULL,
            amount REAL NOT NULL,
            source TEXT,
            fetched_at {float_type} NOT NULL,
            UNIQUE(symbol, ex_date)
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_dividends_symbol ON dividends(symbol)",
        f"""
        CREATE TABLE IF NOT EXISTS portfolio_equity (
            id INTEGER PRIMARY KEY,
            peak_total_equity REAL NOT NULL,
            updated_at {float_type} NOT NULL
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS llm_model_blacklist (
            model TEXT PRIMARY KEY,
            provider TEXT,
            reason TEXT,
            blacklisted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP
        )
        """,
    ]
    return statements


def _table_exists(conn, table_name: str) -> bool:
    """Check if a table exists in the database."""
    if _backend == "postgresql":
        cur = conn.cursor()
        cur.execute(
            "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = %s)",
            (table_name,)
        )
        return cur.fetchone()["exists"]
    else:
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        return cur.fetchone() is not None

def _flush_redis_cache():
    """Flush the entire Redis cache for the current database."""
    try:
        redis_client = get_redis_client()
        redis_client.flushdb()
        logger.info("Redis cache invalidated due to empty or newly created database.")
    except Exception as e:
        logger.warning(f"Failed to flush Redis cache: {type(e).__name__}: {e}")

def init_db():
    """Create tables if they don't exist, then run migrations."""
    conn = get_connection()
    is_new_db = False
    try:
        if _backend == "postgresql":
            conn.execute("CREATE SCHEMA IF NOT EXISTS public")
        
        # Check if the database is empty or not yet created
        if not _table_exists(conn, "trading_state"):
            is_new_db = True
        
        statements = _get_init_statements()
        for stmt in statements:
            conn.execute(stmt)
        conn.commit()
    finally:
        conn.close()
    
    if is_new_db:
        _flush_redis_cache()
        
    _migrate_db()
    backfill_latest_close_prices()


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


def _update_latest_close_price(conn, full_symbol: str):
    """Compute and upsert the latest close price for a single symbol into the latest_close_prices table."""
    base_symbol = full_symbol.split('/')[0] if '/' in full_symbol else full_symbol

    # Query 1: latest candle across all timeframes
    sql_all_tf = _adapt_sql(
        """
        WITH RankedCandles AS (
            SELECT close, volume, timestamp, timeframe,
                   ROW_NUMBER() OVER (PARTITION BY timeframe ORDER BY timestamp DESC) as rn
            FROM market_data
            WHERE symbol = %s
        ),
        LatestTimeframe AS (
            SELECT timeframe
            FROM (
                SELECT timeframe, timestamp,
                       ROW_NUMBER() OVER (ORDER BY timestamp DESC) as tf_rn
                FROM RankedCandles
                WHERE rn = 1
            ) t
            WHERE tf_rn = 1
        )
        SELECT r.close, r.volume, r.timestamp, r.timeframe
        FROM RankedCandles r
        JOIN LatestTimeframe l ON r.timeframe = l.timeframe
        WHERE r.rn = 1
        """
    )
    row = conn.execute(sql_all_tf, (full_symbol,)).fetchone()

    if not row or row["close"] is None:
        return

    last = float(row["close"])
    volume = float(row["volume"]) if row["volume"] is not None else None
    candle_timestamp = row["timestamp"]
    timeframe = row["timeframe"]

    # Query 2: latest 2 daily candles for prev_close
    sql_daily = _adapt_sql(
        """
        WITH RankedDaily AS (
            SELECT close, timestamp,
                   ROW_NUMBER() OVER (ORDER BY timestamp DESC) as rn
            FROM market_data
            WHERE symbol = %s AND timeframe = '1d'
        )
        SELECT close
        FROM RankedDaily
        WHERE rn = 2
        """
    )
    daily_row = conn.execute(sql_daily, (full_symbol,)).fetchone()
    prev_close = float(daily_row["close"]) if daily_row and daily_row["close"] is not None else None

    # Upsert into latest_close_prices
    if _backend == "postgresql":
        upsert_sql = _adapt_sql(
            """
            INSERT INTO latest_close_prices (symbol, last, prev_close, volume, candle_timestamp, timeframe, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (symbol) DO UPDATE SET
                last = EXCLUDED.last,
                prev_close = EXCLUDED.prev_close,
                volume = EXCLUDED.volume,
                candle_timestamp = EXCLUDED.candle_timestamp,
                timeframe = EXCLUDED.timeframe,
                updated_at = EXCLUDED.updated_at
            """
        )
    else:
        upsert_sql = _adapt_sql(
            """
            INSERT OR REPLACE INTO latest_close_prices (symbol, last, prev_close, volume, candle_timestamp, timeframe, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
        )
    conn.execute(upsert_sql, (base_symbol, last, prev_close, volume, candle_timestamp, timeframe, time.time()))


def backfill_latest_close_prices():
    """Backfill the latest_close_prices table from existing market_data.
    
    This runs once on startup to populate the table for symbols that already
    have OHLCV data but haven't had new candles inserted since the table was created.
    """
    conn = get_connection()
    try:
        # Check if the table is empty
        row = conn.execute("SELECT COUNT(*) as count FROM latest_close_prices").fetchone()
        if row and row["count"] > 0:
            return  # Already populated

        logger.info("Backfilling latest_close_prices table from market_data...")
        
        # Get all unique symbols from market_data
        symbols = [r["symbol"] for r in conn.execute("SELECT DISTINCT symbol FROM market_data").fetchall()]
        
        for symbol in symbols:
            _update_latest_close_price(conn, symbol)
        
        conn.commit()
        logger.info(f"Backfilled latest_close_prices for {len(symbols)} symbols.")
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
            _update_latest_close_price(conn, symbol)
            conn.commit()
            logger.debug(f"Inserted {len(valid_candles)} valid OHLCV candles for {symbol} {timeframe}")
            # Invalidate the latest close price cache for this symbol
            try:
                redis_client = get_redis_client()
                base_symbol = symbol.split('/')[0] if '/' in symbol else symbol
                redis_client.delete(f"latest_close_prices:{base_symbol}")
            except (ValueError, TypeError, ConnectionError, TimeoutError, OSError):
                pass
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


def get_ohlcv_batch(symbols: List[str], timeframes: List[str], limit: int = 50) -> Dict[str, Dict[str, List[List]]]:
    """Retrieve the most recent OHLCV candles for multiple symbols and timeframes.
    Returns a dict: {symbol: {timeframe: [[ts, o, h, l, c, v], ...]}}
    """
    if not symbols or not timeframes:
        return {}
    conn = get_connection()
    try:
        if _backend == "postgresql":
            sql = _adapt_sql(
                """
                WITH RankedCandles AS (
                    SELECT symbol, timeframe, timestamp, open, high, low, close, volume,
                           ROW_NUMBER() OVER (PARTITION BY symbol, timeframe ORDER BY timestamp DESC) as rn
                    FROM market_data
                    WHERE symbol = ANY(%s) AND timeframe = ANY(%s)
                )
                SELECT symbol, timeframe, timestamp, open, high, low, close, volume
                FROM RankedCandles
                WHERE rn <= %s
                ORDER BY symbol, timeframe, timestamp ASC
                """
            )
            rows = conn.execute(sql, (symbols, timeframes, limit)).fetchall()
        else:
            sym_placeholders = ",".join(["?" for _ in symbols])
            tf_placeholders = ",".join(["?" for _ in timeframes])
            sql = _adapt_sql(
                f"""
                WITH RankedCandles AS (
                    SELECT symbol, timeframe, timestamp, open, high, low, close, volume,
                           ROW_NUMBER() OVER (PARTITION BY symbol, timeframe ORDER BY timestamp DESC) as rn
                    FROM market_data
                    WHERE symbol IN ({sym_placeholders}) AND timeframe IN ({tf_placeholders})
                )
                SELECT symbol, timeframe, timestamp, open, high, low, close, volume
                FROM RankedCandles
                WHERE rn <= ?
                ORDER BY symbol, timeframe, timestamp ASC
                """
            )
            rows = conn.execute(sql, symbols + timeframes + [limit]).fetchall()

        result: Dict[str, Dict[str, List[List]]] = {s: {} for s in symbols}
        for row in rows:
            sym = row["symbol"]
            tf = row["timeframe"]
            if sym not in result:
                result[sym] = {}
            if tf not in result[sym]:
                result[sym][tf] = []
            result[sym][tf].append([
                row["timestamp"], row["open"], row["high"], row["low"], row["close"], row["volume"]
            ])
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


def get_latest_ohlcv_timestamps_batch(pairs: List[str], timeframes: List[str]) -> Dict[str, Dict[str, Optional[int]]]:
    """Batch fetch the latest OHLCV timestamp for multiple symbols and timeframes in a single query."""
    if not pairs or not timeframes:
        return {}

    result = {pair: {tf: None for tf in timeframes} for pair in pairs}

    conn = get_connection()
    try:
        if _backend == "postgresql":
            sql = _adapt_sql(
                """
                SELECT symbol, timeframe, MAX(timestamp) AS latest_ts
                FROM market_data
                WHERE symbol = ANY(%s) AND timeframe = ANY(%s)
                GROUP BY symbol, timeframe
                """
            )
            rows = conn.execute(sql, (pairs, timeframes)).fetchall()
        else:
            pair_placeholders = ",".join(["?" for _ in pairs])
            tf_placeholders = ",".join(["?" for _ in timeframes])
            sql = _adapt_sql(
                f"""
                SELECT symbol, timeframe, MAX(timestamp) AS latest_ts
                FROM market_data
                WHERE symbol IN ({pair_placeholders}) AND timeframe IN ({tf_placeholders})
                GROUP BY symbol, timeframe
                """
            )
            rows = conn.execute(sql, pairs + timeframes).fetchall()

        for row in rows:
            sym = row["symbol"]
            tf = row["timeframe"]
            latest_ts = row["latest_ts"]
            if sym in result and tf in result[sym]:
                result[sym][tf] = latest_ts
        return result
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
        if _backend == "postgresql":
            sql = _adapt_sql(
                """
                SELECT symbol, timeframe, indicators_json, timestamp
                FROM indicators
                WHERE symbol = ANY(%s) AND timeframe = ANY(%s)
                """
            )
            rows = conn.execute(sql, (symbols, timeframes)).fetchall()
        else:
            sym_placeholders = ",".join(["?" for _ in symbols])
            tf_placeholders = ",".join(["?" for _ in timeframes])
            sql = _adapt_sql(
                f"""
                SELECT symbol, timeframe, indicators_json, timestamp
                FROM indicators
                WHERE symbol IN ({sym_placeholders}) AND timeframe IN ({tf_placeholders})
                """
            )
            rows = conn.execute(sql, symbols + timeframes).fetchall()

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

    # Try to get from Redis cache (per-symbol keys for granular invalidation)
    base_symbols_list = [s.split('/')[0] for s in symbols]
    cached_result = {}
    missing_symbols = []
    try:
        redis_client = get_redis_client()
        pipe = redis_client.pipeline()
        for bs in base_symbols_list:
            pipe.get(f"latest_close_prices:{bs}")
        cached_results = pipe.execute()
        
        for bs, cached in zip(base_symbols_list, cached_results):
            if cached:
                cached_result[bs] = json.loads(cached)
            else:
                missing_symbols.append(bs)
    except (ValueError, TypeError, ConnectionError, TimeoutError, OSError, json.JSONDecodeError):
        missing_symbols = base_symbols_list

    if not missing_symbols:
        return cached_result

    conn = get_connection()
    try:
        if _backend == "postgresql":
            sql = _adapt_sql(
                "SELECT symbol, last, prev_close, volume, candle_timestamp, timeframe FROM latest_close_prices WHERE symbol = ANY(%s)"
            )
            rows = conn.execute(sql, (missing_symbols,)).fetchall()
        else:
            placeholders = ",".join(["?" for _ in missing_symbols])
            sql = _adapt_sql(
                f"SELECT symbol, last, prev_close, volume, candle_timestamp, timeframe FROM latest_close_prices WHERE symbol IN ({placeholders})"
            )
            rows = conn.execute(sql, missing_symbols).fetchall()

        result = {}
        for row in rows:
            sym = row["symbol"]
            if sym in missing_symbols and row["last"] is not None and row["last"] > 0:
                result[sym] = {
                    "last": float(row["last"]),
                    "prev_close": float(row["prev_close"]) if row["prev_close"] is not None else None,
                    "volume": float(row["volume"]) if row["volume"] is not None else None,
                    "candle_timestamp": row["candle_timestamp"],
                    "timeframe": row["timeframe"],
                }
    finally:
        conn.close()

    # Save to Redis cache (per-symbol keys)
    try:
        redis_client = get_redis_client()
        pipe = redis_client.pipeline()
        for bs, data in result.items():
            pipe.setex(f"latest_close_prices:{bs}", 60, json.dumps(data))
        pipe.execute()
    except (ValueError, TypeError, ConnectionError, TimeoutError, OSError):
        pass

    # Merge cached results with newly fetched results
    final_result = {**cached_result, **result}
    return final_result


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
        sql = _adapt_sql("SELECT symbol, isin, asset_type, name, maturity, coupon, country, manual_isin, candle_count FROM discovered_symbols")
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
                "manual_isin": row["manual_isin"],
                "candle_count": row["candle_count"],
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
    """Return the ISIN for a symbol from the database, or None if not found.

    Returns the manual ISIN if it exists and is not empty, otherwise returns
    the automatically discovered ISIN.
    """
    conn = get_connection()
    try:
        sql = _adapt_sql("SELECT isin, manual_isin FROM discovered_symbols WHERE symbol = %s")
        row = conn.execute(sql, (symbol,)).fetchone()
        if row:
            if row["manual_isin"] and row["manual_isin"].strip():
                return row["manual_isin"]
            if row["isin"]:
                return row["isin"]
        return None
    finally:
        conn.close()


def get_manual_isin_from_db(symbol: str) -> Optional[str]:
    """Return the manual ISIN for a symbol from the database, or None if not found."""
    conn = get_connection()
    try:
        sql = _adapt_sql("SELECT manual_isin FROM discovered_symbols WHERE symbol = %s")
        row = conn.execute(sql, (symbol,)).fetchone()
        if row and row["manual_isin"] and row["manual_isin"].strip():
            return row["manual_isin"]
        return None
    finally:
        conn.close()


@retry_on_db_lock()
def update_manual_isin(symbol: str, isin: Optional[str]):
    """Update or clear the manual ISIN for a discovered symbol."""
    conn = get_connection()
    try:
        if isin:
            isin = isin.strip()
            if not isin:
                isin = None
            sql = _adapt_sql("UPDATE discovered_symbols SET manual_isin = %s WHERE symbol = %s")
            conn.execute(sql, (isin, symbol))
        else:
            sql = _adapt_sql("UPDATE discovered_symbols SET manual_isin = NULL WHERE symbol = %s")
            conn.execute(sql, (symbol,))
        conn.commit()
    finally:
        conn.close()


def get_candle_count_for_symbol(symbol: str) -> int:
    """Return the total number of candles for a symbol across all timeframes."""
    base = symbol.split("/")[0] if "/" in symbol else symbol
    conn = get_connection()
    try:
        sql = _adapt_sql("SELECT COUNT(*) as count FROM market_data WHERE symbol = %s OR symbol LIKE %s")
        row = conn.execute(sql, (base, base + "/%")).fetchone()
        return row["count"] if row else 0
    finally:
        conn.close()


@retry_on_db_lock()
def update_candle_count(symbol: str, count: int):
    """Update the total candle count for a discovered symbol."""
    base = symbol.split("/")[0] if "/" in symbol else symbol
    # Strip ticker suffix for DB lookup (DB stores base symbols without suffix)
    suffix = settings.TICKER_SUFFIX
    if suffix and base.endswith(suffix):
        base = base[:-len(suffix)]
    conn = get_connection()
    try:
        sql = _adapt_sql("UPDATE discovered_symbols SET candle_count = %s WHERE symbol = %s")
        conn.execute(sql, (count, base))
        conn.commit()
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
            sql = _adapt_sql("SELECT symbol, isin, manual_isin FROM discovered_symbols WHERE symbol = ANY(%s)")
            rows = conn.execute(sql, (symbols,)).fetchall()
        else:
            placeholders = ",".join(["?" for _ in symbols])
            sql = _adapt_sql(f"SELECT symbol, isin, manual_isin FROM discovered_symbols WHERE symbol IN ({placeholders})")
            rows = conn.execute(sql, symbols).fetchall()
        result = {s: None for s in symbols}
        for row in rows:
            if row["manual_isin"] and row["manual_isin"].strip():
                result[row["symbol"]] = row["manual_isin"]
            else:
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
                cache_hit, latency_ms, error, request_type, is_fallback
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
            metrics.get("request_type"),
            1 if metrics.get("is_fallback") else 0,
        ))
        conn.commit()
    finally:
        conn.close()


@retry_on_db_lock()
def reset_llm_metrics():
    """Delete all rows from the llm_metrics table."""
    conn = get_connection()
    try:
        conn.execute(_adapt_sql("DELETE FROM llm_metrics"))
        conn.commit()
    finally:
        conn.close()


def get_llm_metrics_summary(model_filter: str = "all") -> dict:
    """Return aggregated LLM metrics for the dashboard.

    model_filter: "all" (no filter), "main" (is_fallback=0), "fallback" (is_fallback=1).
    """
    conn = get_connection()
    try:
        # Build WHERE clause for model filter
        where_clause = ""
        if model_filter == "main":
            where_clause = " WHERE is_fallback = 0"
        elif model_filter == "fallback":
            where_clause = " WHERE is_fallback = 1"

        # Total calls and tokens
        row = conn.execute(
            "SELECT COUNT(*) as total_calls, "
            "COALESCE(SUM(prompt_tokens),0) as total_prompt_tokens, "
            "COALESCE(SUM(completion_tokens),0) as total_completion_tokens, "
            "COALESCE(SUM(total_tokens),0) as total_tokens, "
            "COALESCE(SUM(cache_hit),0) as cache_hits "
            f"FROM llm_metrics{where_clause}"
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
            "SUM(total_tokens) as tokens, "
            "SUM(prompt_tokens) as prompt_tokens, "
            "SUM(completion_tokens) as completion_tokens, "
            "AVG(latency_ms) as avg_latency "
            f"FROM llm_metrics{where_clause} GROUP BY provider, model, model_type"
        ).fetchall()
        per_model = []
        for r in per_model_rows:
            per_model.append({
                "provider": r["provider"],
                "model": r["model"],
                "model_type": r["model_type"],
                "calls": r["calls"],
                "tokens": r["tokens"],
                "prompt_tokens": r["prompt_tokens"] or 0,
                "completion_tokens": r["completion_tokens"] or 0,
                "avg_latency_ms": round(r["avg_latency"], 2) if r["avg_latency"] else 0,
            })

        # Recent calls (last 20)
        recent_rows = conn.execute(
            "SELECT timestamp, provider, model, model_type, prompt_tokens, completion_tokens, "
            "total_tokens, cache_hit, latency_ms, error, request_type "
            f"FROM llm_metrics{where_clause} ORDER BY timestamp DESC LIMIT 20"
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
                "request_type": r["request_type"],
            })

        # Period-based statistics (hour, day, week, month)
        now_ts = time.time()
        period_cutoffs = [
            ("hour", now_ts - 3600),
            ("day", now_ts - 86400),
            ("week", now_ts - 604800),
            ("month", now_ts - 2592000),
        ]
        period_stats = {}
        for period_name, cutoff in period_cutoffs:
            if where_clause:
                period_sql = f"FROM llm_metrics{where_clause} AND timestamp >= %s"
            else:
                period_sql = "FROM llm_metrics WHERE timestamp >= %s"
            period_row = conn.execute(
                _adapt_sql(
                    "SELECT COUNT(*) as calls, "
                    "COALESCE(SUM(total_tokens),0) as tokens, "
                    "COALESCE(SUM(prompt_tokens),0) as prompt_tokens, "
                    "COALESCE(SUM(completion_tokens),0) as completion_tokens, "
                    "COALESCE(SUM(cache_hit),0) as cache_hits, "
                    "COALESCE(AVG(latency_ms),0) as avg_latency "
                    f"{period_sql}"
                ),
                (cutoff,)
            ).fetchone()
            p_calls = period_row["calls"] if period_row else 0
            p_cache_hits = period_row["cache_hits"] if period_row else 0
            period_stats[period_name] = {
                "calls": p_calls,
                "prompt_tokens": period_row["prompt_tokens"] if period_row else 0,
                "completion_tokens": period_row["completion_tokens"] if period_row else 0,
                "tokens": period_row["tokens"] if period_row else 0,
                "cache_hit_rate": round((p_cache_hits / p_calls * 100) if p_calls > 0 else 0, 2),
                "avg_latency_ms": round(period_row["avg_latency"], 2) if period_row and period_row["avg_latency"] else 0,
            }

        return {
            "total_calls": total_calls,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "cache_hit_rate": round(cache_hit_rate, 2),
            "per_model": per_model,
            "recent_calls": recent_calls,
            "period_stats": period_stats,
        }
    finally:
        conn.close()


def get_llm_metrics_timeseries(period: str = "hour", from_date: Optional[str] = None, to_date: Optional[str] = None, model_filter: str = "all") -> List[Dict[str, Any]]:
    """Return aggregated LLM metrics for charting based on period and date range.

    model_filter: "all" (no filter), "main" (is_fallback=0), "fallback" (is_fallback=1).
    """
    conn = get_connection()
    try:
        now_ts = time.time()
        if from_date:
            cutoff = datetime.strptime(from_date, "%Y-%m-%d").timestamp()
        else:
            if period == "hour":
                cutoff = now_ts - 24 * 3600
            elif period == "day":
                cutoff = now_ts - 30 * 86400
            elif period == "week":
                cutoff = now_ts - 12 * 604800
            elif period == "month":
                cutoff = now_ts - 365 * 86400
            else:
                cutoff = now_ts - 24 * 3600

        if to_date:
            end_ts = datetime.strptime(to_date, "%Y-%m-%d").timestamp() + 86400
        else:
            end_ts = now_ts

        # Build model filter clause
        model_clause = ""
        if model_filter == "main":
            model_clause = " AND is_fallback = 0"
        elif model_filter == "fallback":
            model_clause = " AND is_fallback = 1"

        if _backend == "postgresql":
            trunc = period if period in ("hour", "day", "week", "month") else "hour"
            sql = _adapt_sql(
                f"""
                SELECT
                    EXTRACT(EPOCH FROM date_trunc('{trunc}', to_timestamp(timestamp))) AS hour,
                    COUNT(*) as calls,
                    SUM(total_tokens) as tokens,
                    SUM(prompt_tokens) as prompt_tokens,
                    SUM(completion_tokens) as completion_tokens,
                    AVG(latency_ms) as avg_latency,
                    SUM(cache_hit) as cache_hits
                FROM llm_metrics
                WHERE timestamp >= %s AND timestamp <= %s{model_clause}
                GROUP BY hour
                ORDER BY hour ASC
                """
            )
        else:
            if period == "hour":
                divisor = 3600
            elif period == "day":
                divisor = 86400
            elif period == "week":
                divisor = 604800
            elif period == "month":
                divisor = 2592000  # Approximate month as 30 days
            else:
                divisor = 3600
            sql = _adapt_sql(
                f"""
                SELECT
                    (CAST(timestamp / {divisor} AS INTEGER) * {divisor}) AS hour,
                    COUNT(*) as calls,
                    SUM(total_tokens) as tokens,
                    SUM(prompt_tokens) as prompt_tokens,
                    SUM(completion_tokens) as completion_tokens,
                    AVG(latency_ms) as avg_latency,
                    SUM(cache_hit) as cache_hits
                FROM llm_metrics
                WHERE timestamp >= %s AND timestamp <= %s{model_clause}
                GROUP BY hour
                ORDER BY hour ASC
                """
            )
        rows = conn.execute(sql, (cutoff, end_ts)).fetchall()
        result = []
        for row in rows:
            hour_ts = row["hour"]
            calls = row["calls"]
            cache_hits = row["cache_hits"]
            result.append({
                "hour": hour_ts,
                "calls": calls,
                "tokens": row["tokens"],
                "prompt_tokens": row["prompt_tokens"] or 0,
                "completion_tokens": row["completion_tokens"] or 0,
                "avg_latency_ms": round(row["avg_latency"], 2) if row["avg_latency"] else 0,
                "cache_hit_rate": round((cache_hits / calls * 100) if calls > 0 else 0, 1),
            })
        return result
    finally:
        conn.close()


def compute_btp_ytm(coupon: Optional[float], maturity: Optional[str], price: Optional[float]) -> Optional[float]:
    """Compute Yield to Maturity for a BTP bond using Newton-Raphson."""
    if not coupon or not maturity or not price or price <= 0:
        return None
    try:
        from datetime import datetime
        maturity_date = datetime.strptime(maturity, "%Y-%m-%d")
        years_to_maturity = (maturity_date - datetime.now()).days / 365.25
        if years_to_maturity <= 0:
            return None
        
        periods = int(years_to_maturity * 2)
        if periods == 0:
            return None
        
        coupon_payment = (coupon / 100) * 100 / 2  # BTPs pay semi-annually
        par_value = 100.0
        
        ytm = 0.05  # Initial guess
        for _ in range(100):
            price_calc = 0.0
            for i in range(1, periods + 1):
                price_calc += coupon_payment / ((1 + ytm / 2) ** i)
            price_calc += par_value / ((1 + ytm / 2) ** periods)
            
            derivative = 0.0
            for i in range(1, periods + 1):
                derivative -= (i / 2) * coupon_payment / ((1 + ytm / 2) ** (i + 1))
            derivative -= (periods / 2) * par_value / ((1 + ytm / 2) ** (periods + 1))
            
            if derivative == 0:
                break
            new_ytm = ytm - (price_calc - price) / derivative
            if abs(new_ytm - ytm) < 1e-6:
                ytm = new_ytm
                break
            ytm = new_ytm
            
        return round(ytm * 100, 2)
    except (ValueError, TypeError):
        return None


def close_pool():
    """Close the PostgreSQL connection pool if it exists."""
    global _pg_pool
    if _pg_pool is not None:
        try:
            _pg_pool.close()
        except (psycopg.Error, RuntimeError) as e:
            logger.warning(f"Error closing connection pool: {type(e).__name__}: {e}")
        _pg_pool = None
        logger.info("PostgreSQL connection pool closed.")


def get_pool_stats() -> dict:
    """Return current connection pool statistics for monitoring."""
    if _pg_pool is None:
        return {"status": "not_initialized"}
    try:
        stats = _pg_pool.get_stats()
        stats["status"] = "active"
        stats["min_size"] = _pg_pool._min_size
        stats["max_size"] = _pg_pool._max_size
        return stats
    except (psycopg.Error, AttributeError):
        return {"status": "unknown"}


@retry_on_db_lock()
def insert_dividend(symbol: str, ex_date: str, amount: float, source: str = "yahoo"):
    """Insert or update a dividend payment."""
    base = symbol.split("/")[0] if "/" in symbol else symbol
    conn = get_connection()
    try:
        if _backend == "postgresql":
            sql = _adapt_sql(
                "INSERT INTO dividends (symbol, ex_date, amount, source, fetched_at) "
                "VALUES (%s, %s, %s, %s, %s) "
                "ON CONFLICT (symbol, ex_date) DO UPDATE SET amount = EXCLUDED.amount, "
                "source = EXCLUDED.source, fetched_at = EXCLUDED.fetched_at"
            )
        else:
            sql = _adapt_sql(
                "INSERT OR REPLACE INTO dividends (symbol, ex_date, amount, source, fetched_at) "
                "VALUES (%s, %s, %s, %s, %s)"
            )
        conn.execute(sql, (base, ex_date, amount, source, time.time()))
        conn.commit()
    finally:
        conn.close()


def get_total_dividends_for_symbol(symbol: str) -> float:
    """Return total dividends received for a symbol (base symbol without /currency)."""
    base = symbol.split("/")[0] if "/" in symbol else symbol
    conn = get_connection()
    try:
        sql = _adapt_sql("SELECT COALESCE(SUM(amount), 0) as total FROM dividends WHERE symbol = %s")
        row = conn.execute(sql, (base,)).fetchone()
        return float(row["total"]) if row else 0.0
    finally:
        conn.close()


def get_total_dividends_for_symbols(symbols: List[str]) -> Dict[str, float]:
    """Return total dividends for multiple symbols in a single query.
    Accepts full pair symbols (e.g., 'AAPL/USD') and normalizes to base.
    Returns dict mapping the original input symbol -> total dividends.
    """
    if not symbols:
        return {}
    # Build mapping from base symbol to original input symbol(s)
    base_to_orig: Dict[str, list] = {}
    for s in symbols:
        base = s.split("/")[0] if "/" in s else s
        base_to_orig.setdefault(base, []).append(s)
    bases = list(base_to_orig.keys())
    conn = get_connection()
    try:
        if _backend == "postgresql":
            sql = _adapt_sql("SELECT symbol, COALESCE(SUM(amount), 0) as total FROM dividends WHERE symbol = ANY(%s) GROUP BY symbol")
            rows = conn.execute(sql, (bases,)).fetchall()
        else:
            placeholders = ",".join(["?" for _ in bases])
            sql = _adapt_sql(f"SELECT symbol, COALESCE(SUM(amount), 0) as total FROM dividends WHERE symbol IN ({placeholders}) GROUP BY symbol")
            rows = conn.execute(sql, bases).fetchall()
        result = {s: 0.0 for s in symbols}
        for row in rows:
            base = row["symbol"]
            total = float(row["total"])
            for orig in base_to_orig.get(base, []):
                result[orig] = total
        return result
    finally:
        conn.close()


def get_next_ex_dividend_date(symbol: str, days_ahead: int = 60) -> Optional[Tuple[str, int]]:
    """Return (ex_date_str, days_until) for the next upcoming dividend within days_ahead.
    Returns None if no upcoming dividend found.
    """
    base = symbol.split("/")[0] if "/" in symbol else symbol
    conn = get_connection()
    try:
        from datetime import datetime, timedelta
        cutoff_date = (datetime.now() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
        today_str = datetime.now().strftime("%Y-%m-%d")
        sql = _adapt_sql(
            "SELECT ex_date, amount FROM dividends "
            "WHERE symbol = %s AND ex_date >= %s AND ex_date <= %s "
            "ORDER BY ex_date ASC LIMIT 1"
        )
        row = conn.execute(sql, (base, today_str, cutoff_date)).fetchone()
        if row:
            ex_date_str = row["ex_date"]
            ex_dt = datetime.strptime(ex_date_str, "%Y-%m-%d")
            days_until = (ex_dt - datetime.now()).days
            return (ex_date_str, days_until)
        return None
    except (ValueError, TypeError, KeyError, sqlite3.Error, psycopg.Error):
        return None
    finally:
        conn.close()


def get_pending_dividends_for_symbol(symbol: str) -> List[Dict[str, Any]]:
    """Retrieve non-reinvested dividends for a symbol."""
    base = symbol.split("/")[0] if "/" in symbol else symbol
    conn = get_connection()
    try:
        sql = _adapt_sql(
            "SELECT id, symbol, ex_date, amount FROM dividends WHERE symbol = %s AND reinvested = 0"
        )
        rows = conn.execute(sql, (base,)).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


@retry_on_db_lock()
def mark_dividend_reinvested(dividend_id: int):
    """Mark a dividend as reinvested."""
    conn = get_connection()
    try:
        sql = _adapt_sql("UPDATE dividends SET reinvested = 1 WHERE id = %s")
        conn.execute(sql, (dividend_id,))
        conn.commit()
    finally:
        conn.close()


def get_dividend_yields_for_symbols(symbols: List[str], prices: Dict[str, float]) -> Dict[str, float]:
    """Compute trailing 12-month dividend yield for multiple symbols.

    Args:
        symbols: List of full pair symbols (e.g., 'AAPL/USD')
        prices: Dict mapping base symbol -> current price

    Returns dict mapping original input symbol -> yield as a fraction (e.g., 0.032 = 3.2%).
    Only includes symbols with yield > 0.
    """
    if not symbols or not prices:
        return {}
    from datetime import datetime, timedelta
    cutoff_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

    base_to_orig: Dict[str, list] = {}
    for s in symbols:
        base = s.split("/")[0] if "/" in s else s
        base_to_orig.setdefault(base, []).append(s)
    bases = list(base_to_orig.keys())

    conn = get_connection()
    try:
        if _backend == "postgresql":
            sql = _adapt_sql(
                "SELECT symbol, SUM(amount) as total FROM dividends "
                "WHERE symbol = ANY(%s) AND ex_date >= %s GROUP BY symbol"
            )
            rows = conn.execute(sql, (bases, cutoff_date)).fetchall()
        else:
            placeholders = ",".join(["?" for _ in bases])
            sql = _adapt_sql(
                f"SELECT symbol, SUM(amount) as total FROM dividends "
                f"WHERE symbol IN ({placeholders}) AND ex_date >= %s GROUP BY symbol"
            )
            rows = conn.execute(sql, bases + [cutoff_date]).fetchall()

        result = {}
        for row in rows:
            base = row["symbol"]
            total_div = float(row["total"]) if row["total"] else 0.0
            price = prices.get(base, 0.0)
            if total_div > 0 and price > 0:
                yield_frac = total_div / price
                for orig in base_to_orig.get(base, []):
                    result[orig] = round(yield_frac, 4)
        return result
    finally:
        conn.close()


@retry_on_db_lock()
def cleanup_old_dividends(retention_days: int = 365):
    """Delete dividend records older than retention_days."""
    conn = get_connection()
    try:
        cutoff = time.time() - retention_days * 24 * 60 * 60
        sql = _adapt_sql("DELETE FROM dividends WHERE fetched_at < %s")
        deleted = conn.execute(sql, (cutoff,)).rowcount
        conn.commit()
        if deleted:
            logger.info(f"Cleaned up {deleted} old dividend records (older than {retention_days} days)")
        return deleted
    finally:
        conn.close()


@retry_on_db_lock()
def insert_llm_decision(symbol: str, action: str, entry_price: float, timeframe: str, is_fallback: bool = False):
    """Inserts a new LLM decision into the quality tracking table."""
    timestamp = int(time.time() * 1000)
    conn = get_connection()
    try:
        sql = _adapt_sql(
            """
            INSERT INTO llm_decision_quality (symbol, timestamp, action, entry_price, timeframe, evaluated, is_fallback)
            VALUES (%s, %s, %s, %s, %s, 0, %s)
            """
        )
        conn.execute(sql, (symbol, timestamp, action, entry_price, timeframe, 1 if is_fallback else 0))
        conn.commit()
    finally:
        conn.close()


def get_pending_llm_decisions(evaluation_window_seconds: int) -> List[Dict[str, Any]]:
    """Fetches LLM decisions that are ready to be evaluated."""
    cutoff_timestamp = int((time.time() - evaluation_window_seconds) * 1000)
    conn = get_connection()
    try:
        sql = _adapt_sql(
            """
            SELECT id, symbol, timestamp, action, entry_price, timeframe
            FROM llm_decision_quality
            WHERE evaluated = 0 AND timestamp <= %s
            """
        )
        rows = conn.execute(sql, (cutoff_timestamp,)).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


@retry_on_db_lock()
def update_llm_decision_outcome(decision_id: int, outcome_price: float, outcome_profitable: bool):
    """Updates an LLM decision record with the actual outcome."""
    outcome_timestamp = int(time.time() * 1000)
    conn = get_connection()
    try:
        sql = _adapt_sql(
            """
            UPDATE llm_decision_quality
            SET outcome_price = %s, outcome_timestamp = %s, outcome_profitable = %s, evaluated = 1
            WHERE id = %s
            """
        )
        conn.execute(sql, (outcome_price, outcome_timestamp, 1 if outcome_profitable else 0, decision_id))
        conn.commit()
    finally:
        conn.close()


def get_llm_decision_quality_metrics(period_days: int = 7, model_filter: str = "all") -> Dict[str, Any]:
    """Calculates LLM decision accuracy and counts over a given period."""
    cutoff_timestamp = int((time.time() - period_days * 86400) * 1000)
    conn = get_connection()
    try:
        where_clause = " WHERE evaluated = 1 AND timestamp >= %s"
        if model_filter == "main":
            where_clause += " AND is_fallback = 0"
        elif model_filter == "fallback":
            where_clause += " AND is_fallback = 1"

        sql = _adapt_sql(
            f"""
            SELECT 
                COUNT(*) as total_decisions,
                SUM(CASE WHEN outcome_profitable = 1 THEN 1 ELSE 0 END) as profitable_decisions,
                SUM(CASE WHEN action = 'HOLD' THEN 1 ELSE 0 END) as hold_decisions,
                SUM(CASE WHEN action = 'BUY' THEN 1 ELSE 0 END) as buy_decisions,
                SUM(CASE WHEN action = 'SELL' THEN 1 ELSE 0 END) as sell_decisions
            FROM llm_decision_quality
            {where_clause}
            """
        )
        row = conn.execute(sql, (cutoff_timestamp,)).fetchone()
        if not row or row["total_decisions"] == 0:
            return {
                "total_evaluated": 0,
                "accuracy": 0.0,
                "hold_count": 0,
                "buy_count": 0,
                "sell_count": 0
            }
        
        total = row["total_decisions"]
        profitable = row["profitable_decisions"] or 0
        return {
            "total_evaluated": total,
            "accuracy": (profitable / total) * 100 if total > 0 else 0.0,
            "hold_count": row["hold_decisions"] or 0,
            "buy_count": row["buy_decisions"] or 0,
            "sell_count": row["sell_decisions"] or 0
        }
    finally:
        conn.close()


@retry_on_db_lock()
def cleanup_old_llm_decisions(retention_days: int = 30):
    """Delete evaluated LLM decisions older than retention_days."""
    conn = get_connection()
    try:
        cutoff_ms = int((time.time() - retention_days * 24 * 60 * 60) * 1000)
        sql = _adapt_sql("DELETE FROM llm_decision_quality WHERE evaluated = 1 AND timestamp < %s")
        deleted = conn.execute(sql, (cutoff_ms,)).rowcount
        conn.commit()
        if deleted:
            logger.info(f"Cleaned up {deleted} old LLM decision quality records (older than {retention_days} days)")
        return deleted
    finally:
        conn.close()


from datetime import datetime, timezone, timedelta

def add_model_to_blacklist(model: str, provider: str, reason: str, expires_at: datetime) -> None:
    """Add or update a model in the blacklist."""
    conn = get_connection()
    try:
        if _backend == "postgresql":
            sql = _adapt_sql(
                "INSERT INTO llm_model_blacklist (model, provider, reason, blacklisted_at, expires_at) "
                "VALUES (%s, %s, %s, %s, %s) "
                "ON CONFLICT (model) DO UPDATE SET provider = EXCLUDED.provider, reason = EXCLUDED.reason, blacklisted_at = EXCLUDED.blacklisted_at, expires_at = EXCLUDED.expires_at"
            )
        else:
            sql = _adapt_sql(
                "INSERT OR REPLACE INTO llm_model_blacklist (model, provider, reason, blacklisted_at, expires_at) "
                "VALUES (%s, %s, %s, %s, %s)"
            )
        conn.execute(sql, (model, provider, reason, datetime.now(timezone.utc), expires_at))
        conn.commit()
    finally:
        conn.close()

def get_active_blacklisted_models() -> List[Dict[str, Any]]:
    """Return models currently in the blacklist (not expired)."""
    conn = get_connection()
    try:
        sql = _adapt_sql(
            "SELECT model, provider, expires_at FROM llm_model_blacklist WHERE expires_at > %s"
        )
        rows = conn.execute(sql, (datetime.now(timezone.utc),)).fetchall()
        return [{"model": r["model"], "provider": r["provider"], "expires_at": r["expires_at"]} for r in rows]
    finally:
        conn.close()

def remove_model_from_blacklist(model: str) -> None:
    """Remove a model from the blacklist."""
    conn = get_connection()
    try:
        conn.execute(_adapt_sql("DELETE FROM llm_model_blacklist WHERE model = %s"), (model,))
        conn.commit()
    finally:
        conn.close()

@retry_on_db_lock()
def clear_all_blacklisted_models() -> None:
    """Remove all models from the blacklist."""
    conn = get_connection()
    try:
        conn.execute(_adapt_sql("DELETE FROM llm_model_blacklist"))
        conn.commit()
    finally:
        conn.close()

@retry_on_db_lock()
def reset_llm_decision_quality() -> None:
    """Delete all rows from the llm_decision_quality table."""
    conn = get_connection()
    try:
        conn.execute(_adapt_sql("DELETE FROM llm_decision_quality"))
        conn.commit()
    finally:
        conn.close()

def get_all_blacklisted_models() -> List[Dict[str, Any]]:
    """Return all models from the blacklist table, ordered by most recent."""
    conn = get_connection()
    try:
        sql = _adapt_sql(
            "SELECT model, provider, reason, blacklisted_at, expires_at FROM llm_model_blacklist ORDER BY blacklisted_at DESC"
        )
        rows = conn.execute(sql).fetchall()
        return [{"model": r["model"], "provider": r["provider"], "reason": r["reason"], "blacklisted_at": r["blacklisted_at"], "expires_at": r["expires_at"]} for r in rows]
    finally:
        conn.close()


@retry_on_db_lock()
def save_peak_total_equity(peak_equity: float):
    """Persist the peak total equity to the dedicated portfolio_equity table."""
    conn = get_connection()
    try:
        if _backend == "postgresql":
            sql = _adapt_sql(
                "INSERT INTO portfolio_equity (id, peak_total_equity, updated_at) VALUES (1, %s, %s) "
                "ON CONFLICT (id) DO UPDATE SET peak_total_equity = EXCLUDED.peak_total_equity, updated_at = EXCLUDED.updated_at"
            )
        else:
            sql = _adapt_sql(
                "INSERT OR REPLACE INTO portfolio_equity (id, peak_total_equity, updated_at) VALUES (1, %s, %s)"
            )
        conn.execute(sql, (peak_equity, time.time()))
        conn.commit()
    finally:
        conn.close()


def get_peak_total_equity() -> Optional[float]:
    """Retrieve the peak total equity from the dedicated portfolio_equity table."""
    conn = get_connection()
    try:
        sql = _adapt_sql("SELECT peak_total_equity FROM portfolio_equity WHERE id = 1")
        row = conn.execute(sql).fetchone()
        if row:
            return float(row["peak_total_equity"])
        return None
    finally:
        conn.close()
