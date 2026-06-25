import sqlite3
import json
import os
import logging
import time
import functools
from typing import Dict, List, Any, Optional

from src.config.settings import settings

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
        maxconn=10,
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


def get_connection():
    """Return a database connection appropriate for the current backend."""
    if _backend == "postgresql":
        conn = _pg_pool.getconn()
        return _PgConnectionWrapper(conn)
    else:
        os.makedirs(os.path.dirname(settings.DATABASE_PATH), exist_ok=True)
        conn = sqlite3.connect(settings.DATABASE_PATH, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn


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
    """Add missing columns to existing tables (schema migrations)."""
    migrations = [
        ("trade_history", "timeframe", "ALTER TABLE trade_history ADD COLUMN timeframe TEXT"),
        ("trade_history", "cost_basis", "ALTER TABLE trade_history ADD COLUMN cost_basis REAL"),
        ("trade_history", "strategy_type", "ALTER TABLE trade_history ADD COLUMN strategy_type TEXT"),
        ("trade_history", "note", "ALTER TABLE trade_history ADD COLUMN note TEXT"),
        ("trade_history", "status", "ALTER TABLE trade_history ADD COLUMN status TEXT"),
    ]

    for table, column, sql in migrations:
        existing = _get_existing_columns(table)
        if column not in existing:
            conn = get_connection()
            try:
                conn.execute(_adapt_sql(sql))
                conn.commit()
            except Exception as e:
                logger.warning(f"Migration {sql} failed: {e}")
            finally:
                conn.close()


def init_db():
    """Create tables if they don't exist, then run migrations."""
    conn = get_connection()
    try:
        if _backend == "postgresql":
            statements = [
                """
                CREATE TABLE IF NOT EXISTS trading_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS telegram_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS trade_history (
                    id SERIAL PRIMARY KEY,
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
                    timestamp BIGINT NOT NULL
                )
                """,
                "CREATE INDEX IF NOT EXISTS idx_trade_history_symbol ON trade_history(symbol)",
                "CREATE INDEX IF NOT EXISTS idx_trade_history_timestamp ON trade_history(timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_trade_history_symbol_timeframe ON trade_history(symbol, timeframe)",
                """
                CREATE TABLE IF NOT EXISTS news_articles (
                    id SERIAL PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    title TEXT,
                    source TEXT,
                    url TEXT,
                    published_at TEXT,
                    summary TEXT,
                    sentiment_label TEXT,
                    sentiment_compound REAL,
                    fetched_at DOUBLE PRECISION NOT NULL
                )
                """,
                "CREATE INDEX IF NOT EXISTS idx_news_symbol ON news_articles(symbol)",
                "CREATE INDEX IF NOT EXISTS idx_news_fetched_at ON news_articles(fetched_at)",
                """
                CREATE TABLE IF NOT EXISTS market_data (
                    id SERIAL PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    timestamp BIGINT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL
                )
                """,
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_market_data_symbol_tf_ts ON market_data(symbol, timeframe, timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_market_data_timestamp ON market_data(timestamp)",
            ]
            for stmt in statements:
                conn.execute(stmt)
        else:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS trading_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS telegram_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS trade_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
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
                    timestamp INTEGER NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_trade_history_symbol ON trade_history(symbol);
                CREATE INDEX IF NOT EXISTS idx_trade_history_timestamp ON trade_history(timestamp);
                CREATE INDEX IF NOT EXISTS idx_trade_history_symbol_timeframe ON trade_history(symbol, timeframe);

                CREATE TABLE IF NOT EXISTS news_articles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    title TEXT,
                    source TEXT,
                    url TEXT,
                    published_at TEXT,
                    summary TEXT,
                    sentiment_label TEXT,
                    sentiment_compound REAL,
                    fetched_at REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_news_symbol ON news_articles(symbol);
                CREATE INDEX IF NOT EXISTS idx_news_fetched_at ON news_articles(fetched_at);

                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL
                );

                CREATE UNIQUE INDEX IF NOT EXISTS idx_market_data_symbol_tf_ts ON market_data(symbol, timeframe, timestamp);
                CREATE INDEX IF NOT EXISTS idx_market_data_timestamp ON market_data(timestamp);
            """)
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
                strategy_type, note, status, timestamp
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
        ))
        conn.commit()
    finally:
        conn.close()


# ---------- Trading state helpers ----------

def load_trading_state() -> Dict[str, Any]:
    """Load all trading state from the database."""
    conn = get_connection()
    rows = conn.execute("SELECT key, value FROM trading_state").fetchall()
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
    row = conn.execute(
        "SELECT value FROM trading_state WHERE key = 'paper_balances'"
    ).fetchone()
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
    row = conn.execute(
        "SELECT value FROM trading_state WHERE key = 'paper_orders'"
    ).fetchone()
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
    row = conn.execute("SELECT value FROM telegram_state WHERE key = 'chat_id'").fetchone()
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


def get_performance() -> Dict[str, Any]:
    """Return performance summary grouped by symbol and timeframe, plus a TOTAL row."""
    conn = get_connection()
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
        conn.executemany(sql, [
            (symbol, timeframe, int(c[0]), float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5]))
            for c in candles
        ])
        conn.commit()
        logger.debug(f"Inserted {len(candles)} OHLCV candles for {symbol} {timeframe}")
    finally:
        conn.close()


def get_ohlcv(symbol: str, timeframe: str, since_ms: int = None, limit: int = 500) -> List[Dict[str, Any]]:
    """Retrieve OHLCV candles from the market_data table."""
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
        rows = conn.execute(_adapt_sql(query), params).fetchall()
        return [
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


def close_pool():
    """Close the PostgreSQL connection pool if it exists."""
    global _pg_pool
    if _pg_pool is not None:
        _pg_pool.closeall()
        _pg_pool = None
        logger.info("PostgreSQL connection pool closed.")
