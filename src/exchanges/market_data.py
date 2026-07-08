import asyncio
import hashlib
import logging
import re
import threading
import time
import warnings
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional

import pandas as pd
import json
import requests
import yfinance as yf
import httpx
from bs4 import BeautifulSoup

from src.config.settings import settings
from src.utils.redis_client import get_redis_client
from src.utils.symbol_utils import is_btp_isin
from src.database import save_quotes_batch, get_quotes_from_db, get_latest_close_prices
from src.exchanges.proxy_utils import DynamicProxyRotator, _dynamic_rotator, _get_proxies
from src.exchanges.borsa_italiana_utils import (
    _check_bi_circuit,
    _record_bi_error,
    _reset_bi_circuit,
    _get_borsa_italiana_token,
    _invalidate_borsa_token_cache,
    _get_isin_and_info_from_borsa_italiana,
    get_borsa_italiana_quote,
    get_borsa_italiana_candles,
    BORSA_TIMEFRAME_MAP,
    _fetch_btp_details,
    discover_btp_bonds,
)
from src.exchanges.yf_session import (
    _yf_download_with_timeout,
    _check_yf_circuit,
    _record_yf_error,
    _reset_yf_circuit,
    _invalidate_yf_session,
    _get_yf_session,
    _yf_rate_limiter,
    YFinanceRateLimiter,
)
from src.exchanges.candle_utils import (
    _validate_and_clean_candles,
    _aggregate_candles,
    _merge_candles,
)
from src.exchanges.alphavantage_utils import (
    _av_rate_limiter,
    get_alphavantage_quote,
    get_alphavantage_candles,
)
from src.exchanges.iex_utils import (
    _iex_rate_limiter,
    get_iex_quote,
    get_iex_candles,
)

logger = logging.getLogger(__name__)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
_notifier = None

def set_notifier(notifier):
    global _notifier
    _notifier = notifier

_get_quotes_lock = threading.Lock()

def _get_isin_from_yfinance(base_symbol: str) -> Optional[str]:
    """Fetch the ISIN code for a symbol, using DB first, then yfinance as fallback."""
    from src.database import get_isin_from_db, save_discovered_symbol

    # Strip suffix for DB lookup (DB stores base symbols without suffix)
    suffix = settings.TICKER_SUFFIX
    db_symbol = base_symbol
    if suffix and db_symbol.endswith(suffix):
        db_symbol = db_symbol[:-len(suffix)]

    # Check DB first (not Redis)
    cached = get_isin_from_db(db_symbol)
    if cached:
        return cached

    isin = None
    # If yfinance circuit is open, we can't fetch the ISIN from yfinance.
    if not _check_yf_circuit():
        yf_symbol = f"{db_symbol}{suffix}" if suffix and not db_symbol.endswith(suffix) else db_symbol
        try:
            ticker = yf.Ticker(yf_symbol, session=_get_yf_session())
            isin = ticker.isin
            if isin:
                isin = isin.strip()
                if isin == '-' or not isin:
                    isin = None
        except (RuntimeError, ValueError, KeyError, AttributeError, OSError) as e:
            logger.debug(f"Failed to fetch ISIN for {base_symbol} from yfinance: {e}")
            isin = None

    # Fallback to Borsa Italiana search if yfinance failed or circuit is open
    if not isin:
        bi_isin, _, _ = _get_isin_and_info_from_borsa_italiana(db_symbol)
        if bi_isin:
            isin = bi_isin

    if isin:
        # Save to DB with the base symbol (no suffix)
        try:
            save_country = settings.TARGET_COUNTRY if settings.COUNTRY_FILTER_STRICT else None
            save_discovered_symbol(db_symbol, isin, "stock", None, country=save_country)
        except (RuntimeError, ValueError, OSError):
            pass

    return isin


# Map our timeframe strings to yfinance interval strings
TIMEFRAME_MAP = {
    "1h": "60m",
    "1d": "1d",
    "1w": "1wk",
    "1M": "1mo",
    "3M": "3mo",
    "6M": "6mo",
    "1Y": "1y",
    "3Y": "3y",
    "5Y": "5y",
}

TIMEFRAME_MS = {
    "1h": 3600_000,
    "1d": 86_400_000,
    "1w": 604_800_000,
    "1M": 2_592_000_000,
    "3M": 7_776_000_000,
    "6M": 15_552_000_000,
    "1Y": 31_536_000_000,
    "3Y": 94_608_000_000,
    "5Y": 157_680_000_000,
}

def _fetch_info(symbol: str, max_retries: int = 2) -> tuple[Optional[str], Optional[str]]:
    """Fetch the country and name from yfinance info for a symbol, with retries.

    Returns a tuple (country, name) on success, or (None, None) if yfinance
    could not provide the information after all retries.
    """
    country, name = None, None
    if not _check_yf_circuit():
        import time as _time
        for attempt in range(max_retries + 1):
            try:
                ticker = yf.Ticker(symbol, session=_get_yf_session())
                info = ticker.info
                country = info.get("country")
                name = info.get("longName") or info.get("shortName")
                if country or name:
                    break
                # country is None or empty – retry if attempts remain
                if attempt < max_retries:
                    _time.sleep(0.5 * (2 ** attempt))
                    continue
            except (RuntimeError, ValueError, KeyError, AttributeError, OSError) as e:
                logger.debug(f"Failed to fetch info for {symbol} (attempt {attempt + 1}/{max_retries + 1}): {e}")
                if attempt < max_retries:
                    _time.sleep(0.5 * (2 ** attempt))

    # Fallback to Borsa Italiana search if yfinance failed or circuit is open
    if not country or not name:
        # Strip suffix for Borsa Italiana search
        db_symbol = symbol
        if settings.TICKER_SUFFIX and db_symbol.endswith(settings.TICKER_SUFFIX):
            db_symbol = db_symbol[:-len(settings.TICKER_SUFFIX)]

        bi_isin, bi_country, bi_name = _get_isin_and_info_from_borsa_italiana(db_symbol)
        if not country:
            country = bi_country
        if not name:
            name = bi_name

    if country or name:
        return country, name
    return None, None


def _discover_wikipedia_tickers(urls: List[str], index_name: str) -> List[str]:
    """Scrape a Wikipedia constituent list from one or more URLs.

    Returns base symbols (suffix stripped). Tries each URL in order; returns
    the first non‑empty result. Also extracts and caches ISIN codes in Redis.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    redis_client = get_redis_client()
    for url in urls:
        try:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tables = pd.read_html(response.text)
        except (requests.RequestException, ValueError, OSError) as e:
            logger.debug(f"Failed to scrape {url}: {str(e)[:200]}")
            continue

        for table in tables:
            # Flatten multi‑level column names
            if isinstance(table.columns, pd.MultiIndex):
                table.columns = [' '.join(col).strip() for col in table.columns.values]
            
            # Try to find ticker and ISIN columns by name
            ticker_col = None
            isin_col = None
            for col in table.columns:
                col_str = str(col).lower()
                if any(kw in col_str for kw in ("ticker", "symbol", "code", "simbolo", "codice", "yahoo", "borsa")):
                    ticker_col = col
                if "isin" in col_str:
                    isin_col = col
            
            if ticker_col is None:
                # Last resort: look for a column whose values look like tickers
                for col in table.columns:
                    sample = table[col].dropna().astype(str).head(20).tolist()
                    if not sample:
                        continue
                    
                    ticker_like = 0
                    non_empty = 0
                    for s in sample:
                        s_clean = s.strip().upper()
                        if not s_clean:
                            continue
                        non_empty += 1
                        # Match typical ticker patterns like ENI, ENI.MI, etc. (avoid ISINs)
                        if re.match(r'^[A-Z0-9]{1,10}(\.[A-Z]{2})?$', s_clean):
                            ticker_like += 1
                    
                    # If at least 60% of non-empty values look like tickers, use this column
                    if non_empty > 0 and ticker_like >= non_empty * 0.6:
                        ticker_col = col
                        break

            # If no explicit ISIN column, look for a column with ISIN-like values
            if isin_col is None:
                for col in table.columns:
                    if col == ticker_col:
                        continue
                    sample = table[col].dropna().astype(str).head(20).tolist()
                    isin_like = 0
                    non_empty = 0
                    for s in sample:
                        s_clean = s.strip().upper()
                        if not s_clean:
                            continue
                        non_empty += 1
                        if re.match(r"^[A-Z]{2}[A-Z0-9]{9}\d$", s_clean):
                            isin_like += 1
                    if non_empty > 0 and isin_like >= non_empty * 0.8:
                        isin_col = col
                        break
            
            if ticker_col is not None:
                tickers = table[ticker_col].dropna().astype(str).tolist()
                isins = table[isin_col].dropna().astype(str).tolist() if isin_col is not None else []
                base_symbols = []
                for i, t in enumerate(tickers):
                    t = t.strip().upper()
                    # Skip ISINs (e.g., IT0001233417)
                    if re.match(r"^[A-Z]{2}[A-Z0-9]{9}\d$", t):
                        continue
                    # Remove any text after a space or parenthesis (e.g., "ENI.MI (ENI)")
                    t = re.split(r'[\s(]', t)[0]
                    base = t.split(".")[0] if "." in t else t
                    if re.match(r"^[A-Z0-9]+$", base):
                        base_symbols.append(base)
                        # Cache ISIN if available
                        if i < len(isins):
                            isin = isins[i].strip().upper()
                            if re.match(r"^[A-Z]{2}[A-Z0-9]{9}\d$", isin):
                                try:
                                    from src.database import save_discovered_symbol
                                    save_discovered_symbol(base, isin, "stock", None, country="italy")
                                except (RuntimeError, ValueError, OSError):
                                    pass
                
                if base_symbols:
                    logger.info(f"Discovered {len(base_symbols)} {index_name} tickers from {url}")
                    return base_symbols

    logger.warning(f"No ticker column found in any Wikipedia page for {index_name}.")
    return []


def _load_static_tickers() -> List[str]:
    """Load base symbols from a static CSV file if present."""
    import os
    path = os.path.join(settings.DATA_DIR, "italian_tickers.csv")
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r") as f:
            return [line.strip().upper() for line in f if line.strip() and re.match(r"^[A-Z0-9]+$", line.strip())]
    except (OSError, ValueError) as e:
        logger.warning(f"Failed to load static tickers file: {e}")
        return []


def _get_hardcoded_tickers() -> List[str]:
    """Return a hardcoded list of major Italian tickers as a last resort."""
    return [
        "ENI", "ENEL", "ISP", "UCG", "STLA", "TIT", "RACE", "AZM", "BAMI", "MB",
        "LDO", "TEN", "PRY", "SPM", "BPE", "EXO", "NEXI", "A2A", "RNST", "SRG",
        "INW", "DHER", "PST", "BZU", "CPR", "TRN", "BMO", "AQUA", "BRS", "TGY",
        "IWM", "MOL", "HER", "BIA", "CNH", "ST", "UNI", "VBT", "AMP", "BKB"
    ]


def _discover_financedatabase_tickers() -> List[str]:
    """Discover base tickers using the FinanceDatabase library based on TARGET_COUNTRY."""
    try:
        import financedatabase as fd
    except ImportError:
        logger.warning("financedatabase not installed. Skipping FinanceDatabase ticker discovery.")
        return []

    country = settings.TARGET_COUNTRY.capitalize()
    suffix = settings.TICKER_SUFFIX
    try:
        equities = fd.Equities()
        try:
            df = equities.select(country=country)
        except (RuntimeError, ValueError, AttributeError, TypeError):
            df = equities.select()
            # Fallback: filter the dataframe manually if 'country' arg is unsupported
            if df is not None and not df.empty:
                if 'country' in df.columns:
                    df = df[df['country'].str.lower() == country.lower()]
                elif 'exchange' in df.columns:
                    df = df[df['exchange'].str.lower().isin(['mil', 'mta', 'borsa italiana'])]
                else:
                    # If no country/exchange columns, return empty to avoid
                    # overwhelming yfinance with thousands of global equities.
                    logger.warning("FinanceDatabase returned no country/exchange columns; skipping to avoid yfinance overload.")
                    return []
        if df is None or df.empty:
            logger.warning(f"No tickers found in FinanceDatabase for country: {country}")
            return []

        base_symbols = []
        for symbol in df.index:
            # We only want symbols that match our configured suffix (e.g., .MI)
            if suffix and symbol.endswith(suffix):
                base = symbol[:-len(suffix)]
                if re.match(r"^[A-Z0-9]+$", base):
                    base_symbols.append(base)
            elif not suffix:
                # If no suffix is configured, just take the base symbol
                base = symbol.split(".")[0] if "." in symbol else symbol
                if re.match(r"^[A-Z0-9]+$", base):
                    base_symbols.append(base)

        logger.info(f"Discovered {len(base_symbols)} tickers from FinanceDatabase for {country}")
        return base_symbols
    except (ImportError, RuntimeError, ValueError, AttributeError, OSError, TypeError) as e:
        logger.warning(f"FinanceDatabase ticker discovery failed: {e}")
        return []


def discover_italian_ucits_etfs() -> List[str]:
    """Discover Italian-focused UCITS ETFs using FinanceDatabase."""
    redis_client = get_redis_client()
    cache_key = "italian_ucits_etfs"
    try:
        cached = redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
    except (TypeError, ValueError, RuntimeError):
        pass

    try:
        import financedatabase as fd
    except ImportError:
        logger.warning("financedatabase not installed. Skipping ETF discovery.")
        return []

    keywords = [k.strip().lower() for k in settings.ETF_ITALY_KEYWORDS.split(",") if k.strip()]
    if not keywords:
        return []

    try:
        etfs = fd.ETFs()
        # Try to filter by country first, fallback to all if unsupported
        try:
            df = etfs.select(country="Italy")
        except (RuntimeError, ValueError, AttributeError, TypeError):
            df = etfs.select()
            # Fallback: filter the dataframe manually if 'country' arg is unsupported
            if df is not None and not df.empty:
                if 'country' in df.columns:
                    df = df[df['country'].str.lower() == 'italy']
                elif 'exchange' in df.columns:
                    # Filter by Italian exchanges (e.g., MIL, MTA)
                    df = df[df['exchange'].str.lower().isin(['mil', 'mta', 'borsa italiana'])]
                else:
                    # If no country/exchange columns, return empty to avoid
                    # overwhelming yfinance with global ETFs.
                    logger.warning("FinanceDatabase returned no country/exchange columns for ETFs; skipping to avoid yfinance overload.")
                    return []

        if df is None or df.empty:
            return []

        base_symbols = []
        for symbol, row in df.iterrows():
            name = str(row.get('name', '')).lower()
            # MANDATORY SAFETY FILTER: Must contain "UCITS"
            if 'ucits' not in name and 'ucits' not in symbol.lower():
                continue

            # Apply keyword filter to ensure Italian economy focus
            if not any(kw in name for kw in keywords):
                continue

            # Extract short alphanumeric symbol (strip exchange suffix)
            base = symbol.split(".")[0] if "." in symbol else symbol
            if re.match(r"^[A-Z0-9]+$", base):
                base_symbols.append(base)

        logger.info(f"Discovered {len(base_symbols)} Italian UCITS ETFs matching keywords.")
        # Cache for 24 hours
        try:
            redis_client.set(cache_key, json.dumps(base_symbols), ex=86400)
        except (TypeError, ValueError, RuntimeError):
            pass
        # Save ETF symbols to DB
        try:
            from src.database import save_discovered_symbols_batch
            symbols_to_save = [
                {"symbol": sym, "isin": None, "asset_type": "etf", "name": None, "country": "italy"}
                for sym in base_symbols
            ]
            if symbols_to_save:
                save_discovered_symbols_batch(symbols_to_save)
        except (RuntimeError, ValueError, OSError) as e:
            logger.warning(f"Failed to save ETF symbols to DB: {e}")
        return base_symbols
    except (ImportError, RuntimeError, ValueError, AttributeError, OSError, TypeError) as e:
        logger.warning(f"Failed to discover Italian UCITS ETFs: {e}")
        return []


def _save_discovered_assets_to_db(base_symbols: List[str], etf_symbols: List[str] = None):
    """Save discovered stock/ETF base symbols to the database. ISINs are fetched on demand."""
    from src.database import save_discovered_symbols_batch, get_isin_map_from_db
    etf_set = set(etf_symbols or [])
    symbols_to_save = []
    for sym in base_symbols:
        base = sym.split(".")[0] if "." in sym else sym
        if re.match(r'^IT[A-Z0-9]{10}$', base):
            continue  # BTPs are saved separately
        asset_type = "etf" if base in etf_set else "stock"
        symbols_to_save.append({
            "symbol": base,
            "isin": None,
            "asset_type": asset_type,
            "name": None,
            "country": None,
        })
    if not symbols_to_save:
        return
    # Batch check which symbols already have ISINs in DB — single query instead of N
    try:
        sym_list = [s["symbol"] for s in symbols_to_save]
        isin_map = get_isin_map_from_db(sym_list)
        # Filter out symbols that already have an ISIN
        symbols_to_save = [s for s in symbols_to_save if not isin_map.get(s["symbol"])]
    except (RuntimeError, ValueError, OSError):
        pass  # If batch lookup fails, save all (upsert handles it)
    if symbols_to_save:
        try:
            save_discovered_symbols_batch(symbols_to_save)
            logger.info(f"Saved {len(symbols_to_save)} discovered symbols to database")
        except (RuntimeError, ValueError, OSError) as e:
            logger.warning(f"Failed to save discovered symbols to database: {e}")


def get_tradable_assets() -> List[str]:
    """Return a list of tradable Italian equity symbols, filtered by country.

    Discovers base symbols dynamically from the FTSE MIB Wikipedia page and
    from news RSS feeds, then appends the configured ticker suffix and
    verifies via yfinance that each symbol's country matches the configured
    TARGET_COUNTRY. Results are cached in Redis for 24 hours.
    """
    # Discover tickers from Wikipedia (FTSE MIB constituents)
    base_symbols = _discover_wikipedia_tickers(
        ["https://it.wikipedia.org/wiki/FTSE_MIB", "https://en.wikipedia.org/wiki/FTSE_MIB"],
        "FTSE MIB"
    )

    # --- FTSE Italia All-Share constituents ---
    all_share = _discover_wikipedia_tickers(
        ["https://it.wikipedia.org/wiki/FTSE_Italia_All-Share", "https://en.wikipedia.org/wiki/FTSE_Italia_All-Share"],
        "FTSE Italia All-Share"
    )
    if all_share:
        existing = set(base_symbols)
        for t in all_share:
            if t not in existing:
                base_symbols.append(t)
                existing.add(t)

    # --- User-configured additional tickers ---
    extra = settings.ADDITIONAL_TICKERS
    if extra:
        existing = set(base_symbols)
        for t in extra:
            t_clean = t.strip().upper()
            if t_clean and t_clean not in existing:
                base_symbols.append(t_clean)
                existing.add(t_clean)

    # --- FinanceDatabase ticker discovery ---
    if settings.FINANCEDATABASE_TICKER_DISCOVERY_ENABLED:
        fd_tickers = _discover_financedatabase_tickers()
        if fd_tickers:
            existing = set(base_symbols)
            for t in fd_tickers:
                if t not in existing:
                    base_symbols.append(t)
                    existing.add(t)

    # --- Italian UCITS ETF discovery ---
    etf_symbols = discover_italian_ucits_etfs()
    if etf_symbols:
        existing = set(base_symbols)
        for etf in etf_symbols:
            if etf not in existing:
                base_symbols.append(etf)
                existing.add(etf)

    # Discover additional tickers from news RSS feeds
    try:
        from src.news.fetcher import discover_tickers_from_news
        news_tickers = discover_tickers_from_news()
        if news_tickers:
            logger.info(f"Discovered {len(news_tickers)} tickers from news feeds")
            # Merge, ensuring uniqueness
            existing = set(base_symbols)
            for t in news_tickers:
                if t not in existing:
                    base_symbols.append(t)
                    existing.add(t)
    except (RuntimeError, ValueError, OSError) as e:
        logger.warning(f"News ticker discovery failed: {e}")

    # --- Fallback: try static CSV file, then hardcoded list ---
    if not base_symbols:
        static = _load_static_tickers()
        if static:
            logger.info(f"Loaded {len(static)} tickers from static file.")
            base_symbols = static

    if not base_symbols:
        hardcoded = _get_hardcoded_tickers()
        if hardcoded:
            logger.info(f"Loaded {len(hardcoded)} tickers from hardcoded fallback list.")
            base_symbols = hardcoded

    suffix = settings.TICKER_SUFFIX
    target_country = settings.TARGET_COUNTRY.lower()
    if not base_symbols:
        logger.warning("No tickers discovered from Wikipedia, Euronext, or news feeds. Checking Redis cache and DB for previously discovered symbols...")
        # Try Redis cache first (may have symbols from a previous successful run)
        redis_client = get_redis_client()
        cache_key = f"tradable_assets:{settings.TARGET_COUNTRY}"
        try:
            cached = redis_client.get(cache_key)
            if cached:
                import json
                cached_list = json.loads(cached)
                # Merge with DB-saved symbols
                try:
                    from src.database import get_all_discovered_symbols
                    db_symbols = get_all_discovered_symbols()
                    existing_set = set(cached_list)
                    for db_entry in db_symbols:
                        db_sym = db_entry["symbol"]
                        db_country = db_entry.get("country")
                        if db_country is not None and db_country.lower() != target_country:
                            continue
                        if re.match(r'^IT[A-Z0-9]{10}$', db_sym):
                            if db_sym not in existing_set:
                                cached_list.append(db_sym)
                                existing_set.add(db_sym)
                        else:
                            candidate = f"{db_sym}{suffix}" if suffix and not db_sym.endswith(suffix) else db_sym
                            if candidate not in existing_set:
                                cached_list.append(candidate)
                                existing_set.add(candidate)
                    logger.info(f"Discovery failed but recovered {len(cached_list)} symbols from Redis cache + DB")
                except Exception as e:
                    logger.warning(f"Failed to merge DB symbols with cached list: {e}")
                if cached_list:
                    return cached_list
        except (TypeError, ValueError, RuntimeError):
            pass
        # No Redis cache — try DB only
        try:
            from src.database import get_all_discovered_symbols
            db_symbols = get_all_discovered_symbols()
            if db_symbols:
                db_only_list = []
                for db_entry in db_symbols:
                    db_sym = db_entry["symbol"]
                    db_country = db_entry.get("country")
                    # Skip symbols confirmed to be non-Italian
                    if db_country is not None and db_country.lower() != target_country:
                        continue
                    if settings.COUNTRY_FILTER_STRICT and db_country is None and not re.match(r'^IT[A-Z0-9]{10}$', db_sym):
                        continue
                    if is_btp_isin(db_sym):
                        db_only_list.append(db_sym)
                    else:
                        candidate = f"{db_sym}{suffix}" if suffix and not db_sym.endswith(suffix) else db_sym
                        db_only_list.append(candidate)
                if db_only_list:
                    logger.info(f"Discovery failed but recovered {len(db_only_list)} symbols from DB only")
                    return db_only_list
        except (RuntimeError, ValueError, OSError) as e:
            logger.warning(f"Failed to recover symbols from DB: {e}")
        if _notifier:
            msg = "⚠️ Market Data Discovery Failure: All discovery sources (Wikipedia, FinanceDatabase, news feeds, DB) failed to return any tradable assets. The bot will idle."
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(_notifier.send_notification(msg))
            except RuntimeError:
                try:
                    asyncio.run(_notifier.send_notification(msg))
                except Exception:
                    pass
        return []

    # Save discovered symbols to DB.
    # In strict mode, defer saving until AFTER country filtering to avoid
    # polluting the DB with non-Italian symbols.
    if not settings.COUNTRY_FILTER_STRICT:
        try:
            etf_set = set(etf_symbols)
            _save_discovered_assets_to_db(base_symbols, list(etf_set))
        except (RuntimeError, ValueError, OSError) as e:
            logger.warning(f"Failed to save discovered assets to DB: {e}")

    suffix = settings.TICKER_SUFFIX
    candidates = []
    for sym in base_symbols:
        if is_btp_isin(sym):
            candidates.append(sym)          # BTP ISIN – no suffix
        else:
            candidates.append(f"{sym}{suffix}")

    # Check Redis cache
    redis_client = get_redis_client()
    cache_key = f"tradable_assets:{settings.TARGET_COUNTRY}"
    try:
        cached = redis_client.get(cache_key)
        if cached:
            import json
            cached_list = json.loads(cached)
            # Merge with DB-saved symbols even on cache hit
            try:
                from src.database import get_all_discovered_symbols
                db_symbols = get_all_discovered_symbols()
                existing_set = set(cached_list)
                for db_entry in db_symbols:
                    db_sym = db_entry["symbol"]
                    db_country = db_entry.get("country")
                    # Skip symbols confirmed to be non-Italian
                    if db_country is not None and db_country.lower() != target_country:
                        continue
                    if settings.COUNTRY_FILTER_STRICT and db_country is None and not re.match(r'^IT[A-Z0-9]{10}$', db_sym):
                        continue
                    if is_btp_isin(db_sym):
                        if db_sym not in existing_set:
                            cached_list.append(db_sym)
                            existing_set.add(db_sym)
                    else:
                        candidate = f"{db_sym}{suffix}" if suffix and not db_sym.endswith(suffix) else db_sym
                        if candidate not in existing_set:
                            cached_list.append(candidate)
                            existing_set.add(candidate)
            except Exception as e:
                logger.warning(f"Failed to merge DB symbols with cached list: {e}")
            return cached_list
    except (TypeError, ValueError, RuntimeError):
        pass

    # Filter candidates by country using yfinance
    strict = settings.COUNTRY_FILTER_STRICT
    filtered = []
    for symbol in candidates:
        # BTP ISINs start with IT and are Italian bonds, skip yfinance country check
        if is_btp_isin(symbol):
            if target_country == "italy":
                filtered.append(symbol)
            continue

        country, name = _fetch_info(symbol)
        # Save the fetched country and name to the database for future filtering.
        # In strict mode, only save Italian symbols to DB.
        if country is not None and (not settings.COUNTRY_FILTER_STRICT or country.lower() == target_country):
            try:
                from src.database import save_discovered_symbol
                db_base = symbol
                if suffix and db_base.endswith(suffix):
                    db_base = db_base[:-len(suffix)]
                save_discovered_symbol(db_base, None, "stock", name or None, country=country)
            except Exception:
                pass
        elif name and not settings.COUNTRY_FILTER_STRICT:
            # Country is None but name is available — save the name for display
            try:
                from src.database import save_discovered_symbol
                db_base = symbol
                if suffix and db_base.endswith(suffix):
                    db_base = db_base[:-len(suffix)]
                save_discovered_symbol(db_base, None, "stock", name or None, country=None)
            except (RuntimeError, ValueError, OSError):
                pass
        if country is None:
            # yfinance failed to return country info.
            # In lenient mode (default), keep the symbol because it was
            # discovered from Italian sources (Wikipedia FTSE MIB,
            # FinanceDatabase country=Italy, news feeds, etc.).
            # In strict mode, drop it.
            if strict:
                logger.debug(f"Symbol {symbol} skipped (country unknown, strict mode)")
            else:
                filtered.append(symbol)
                logger.debug(f"Symbol {symbol} kept (country unknown, assumed from Italian source)")
        elif country.lower() == target_country:
            filtered.append(symbol)
        else:
            logger.debug(f"Symbol {symbol} skipped (country={country}, target={target_country})")

    # In strict mode, save only confirmed-Italian symbols to DB
    if settings.COUNTRY_FILTER_STRICT:
        try:
            from src.database import save_discovered_symbols_batch
            etf_set = set(etf_symbols)
            symbols_to_save = []
            for symbol in filtered:
                if is_btp_isin(symbol):
                    continue  # BTPs are saved separately
                base = symbol
                if suffix and base.endswith(suffix):
                    base = base[:-len(suffix)]
                asset_type = "etf" if base in etf_set else "stock"
                symbols_to_save.append({
                    "symbol": base,
                    "isin": None,
                    "asset_type": asset_type,
                    "name": None,
                    "country": target_country,
                })
            if symbols_to_save:
                save_discovered_symbols_batch(symbols_to_save)
                logger.info(f"Saved {len(symbols_to_save)} confirmed-Italian symbols to DB (strict mode)")
        except (RuntimeError, ValueError, OSError) as e:
            logger.warning(f"Failed to save filtered assets to DB: {e}")

    # Cache the filtered list for 24 hours
    try:
        import json
        redis_client.set(cache_key, json.dumps(filtered), ex=86400)
    except (TypeError, ValueError, RuntimeError) as e:
        logger.warning(f"Failed to cache tradable assets: {e}")

    # Merge with previously discovered symbols from DB so nothing is lost
    try:
        from src.database import get_all_discovered_symbols
        db_symbols = get_all_discovered_symbols()
        existing_set = set(filtered)
        for db_entry in db_symbols:
            db_sym = db_entry["symbol"]
            db_country = db_entry.get("country")
            # Skip symbols confirmed to be non-Italian
            if db_country is not None and db_country.lower() != target_country:
                continue
            if is_btp_isin(db_sym):
                # BTP ISIN — add as-is (no suffix)
                if db_sym not in existing_set:
                    filtered.append(db_sym)
                    existing_set.add(db_sym)
            else:
                # Stock/ETF — add with suffix
                candidate = f"{db_sym}{suffix}" if suffix and not db_sym.endswith(suffix) else db_sym
                if candidate not in existing_set:
                    filtered.append(candidate)
                    existing_set.add(candidate)
        logger.info(f"Merged {len(db_symbols)} symbols from DB, total: {len(filtered)}")
    except (RuntimeError, ValueError, OSError) as e:
        logger.warning(f"Failed to merge discovered symbols from DB: {e}")

    logger.info(f"Tradable assets for {settings.TARGET_COUNTRY}: {len(filtered)} symbols")
    return filtered


def _enrich_quotes_with_btp_details(result: Dict[str, Dict[str, Any]], symbols: List[str]):
    """Enrich quote results with BTP maturity, coupon, and name from discovered_symbols."""
    btp_symbols = [s for s in symbols if is_btp_isin(s)]
    if not btp_symbols:
        return
    try:
        from src.database import get_btp_details_from_db
        details = get_btp_details_from_db(btp_symbols)
        for sym in btp_symbols:
            if sym in result and details.get(sym):
                d = details[sym]
                if d.get("maturity"):
                    result[sym]["maturity"] = d["maturity"]
                if d.get("coupon") is not None:
                    result[sym]["coupon"] = d["coupon"]
                if not result[sym].get("name") and d.get("name"):
                    result[sym]["name"] = d["name"]
    except (RuntimeError, ValueError, KeyError, OSError) as e:
        logger.debug(f"Failed to enrich BTP details in quotes: {e}")


def get_quotes(symbols: List[str] = None) -> Dict[str, Dict[str, Any]]:
    """Fetch latest quotes for a list of symbols using yfinance batch download.

    Returns a dict mapping symbol -> {last, bid, ask, volume, change_24h, percentage, quoteVolume}.
    Uses yf.download for efficient batch fetching.
    A global lock ensures only one batch download runs at a time to prevent rate limits.
    """
    if symbols is None:
        symbols = []
    if not symbols:
        return {}

    # If a fetch is already in progress, immediately fall back to cache
    # instead of blocking for up to 5 seconds. The background quote refresh
    # loop keeps the cache warm, so serving from cache is acceptable and
    # avoids serializing all callers behind a single slow batch fetch.
    if not _get_quotes_lock.acquire(blocking=False):
        return get_quotes_cached(symbols)

    try:
        return _get_quotes_impl(symbols)
    finally:
        _get_quotes_lock.release()


def _get_quotes_impl(symbols: List[str]) -> Dict[str, Dict[str, Any]]:
    # Sanitize symbols: remove $ prefix and /currency suffix
    symbols = [s.lstrip('$').split('/')[0] for s in symbols]

    redis_client = get_redis_client()
    result = {}
    missing_symbols = []

    # Check per-symbol Redis cache first
    for sym in symbols:
        cache_key = f"quote:{sym}"
        try:
            cached = redis_client.get(cache_key)
            if cached:
                result[sym] = json.loads(cached)
            else:
                missing_symbols.append(sym)
        except (TypeError, ValueError, RuntimeError):
            missing_symbols.append(sym)

    if not missing_symbols:
        return result

    # Check database for quotes not in Redis cache (up to 24 hours old)
    try:
        db_quotes = get_quotes_from_db(missing_symbols, max_age_seconds=86400)
        for sym in list(missing_symbols):
            if sym in db_quotes:
                result[sym] = db_quotes[sym]
                missing_symbols.remove(sym)
                # Refresh Redis cache from DB data
                try:
                    redis_client.set(f"quote:{sym}", json.dumps(db_quotes[sym]), ex=300)
                except Exception:
                    pass
        if db_quotes:
            logger.debug(f"Loaded {len(db_quotes)} quotes from database (Redis miss fallback)")
    except (RuntimeError, ValueError, OSError) as e:
        logger.warning(f"DB quote fetch failed: {e}", exc_info=True)

    # --- Try DB close prices first (fast, no network call) ---
    # This ensures quotes are available even when yfinance is rate-limited or blocked.
    # The OHLCV data is populated by background download tasks using borsaitaliana as primary source.
    if missing_symbols:
        try:
            db_candles = get_latest_close_prices(missing_symbols)
            for sym in list(missing_symbols):
                if sym in db_candles and db_candles[sym].get("last", 0) > 0:
                    last = db_candles[sym]["last"]
                    prev_close = db_candles[sym].get("prev_close")
                    volume = db_candles[sym].get("volume")

                    abs_change = None
                    pct = None
                    if prev_close and prev_close > 0:
                        abs_change = last - prev_close
                        pct = round((abs_change / prev_close) * 100, 4)

                    result[sym] = {
                        "last": last,
                        "bid": last,
                        "ask": last,
                        "volume": volume,
                        "change_24h": abs_change,
                        "percentage": pct,
                        "quoteVolume": volume,
                        "last_update": db_candles[sym].get("candle_timestamp"),
                        "source": "db_close",
                    }
                    # Do not remove from missing_symbols so yfinance can still try to update it
        except (RuntimeError, ValueError, KeyError, OSError) as e:
            logger.warning(f"get_quotes: DB close price fallback failed: {e}")

    if not missing_symbols:
        # All symbols got prices from cache/DB — cache and return
        quotes_to_save = {}
        for sym, q in result.items():
            if q.get("last") is not None:
                try:
                    redis_client.set(f"quote:{sym}", json.dumps(q), ex=300)
                except (TypeError, ValueError, RuntimeError):
                    pass
                quotes_to_save[sym] = q
        if quotes_to_save:
            try:
                save_quotes_batch(quotes_to_save)
            except (RuntimeError, ValueError, OSError) as e:
                logger.warning(f"Failed to save quotes to database: {e}")
        return result

    # Initialize result with None for all still-missing symbols that don't have a DB quote yet
    for sym in missing_symbols:
        if sym not in result or result[sym].get("last") is None:
            result[sym] = {"last": None, "bid": None, "ask": None, "volume": None, "change_24h": None, "percentage": None, "quoteVolume": None}

    # Filter out BTP ISINs as they are not supported by yfinance and should be served from DB
    stock_symbols = [s for s in missing_symbols if not is_btp_isin(s)]

    # --- Batch fetch ALL price data using yf.download (single HTTP request) ---
    # This replaces the slow sequential fast_info calls that caused timeouts.
    # We get last price, volume, and previous close from one batch download.
    # Bid/ask are fetched on-demand by _process_symbol via get_yahoo_quote.
    if stock_symbols and not _check_yf_circuit():
        try:
            # Log proxy status for debugging
            if settings.HTTP_PROXY_ENABLED:
                if settings.HTTP_PROXIES:
                    logger.debug(f"get_quotes: HTTP_PROXY_ENABLED with {len(settings.HTTP_PROXIES)} static proxies")
                else:
                    logger.debug("get_quotes: HTTP_PROXY_ENABLED with dynamic proxy rotator")
            else:
                logger.debug("get_quotes: HTTP_PROXY not enabled")
            batch_hist = _yf_download_with_timeout(
                stock_symbols,
                period="2d",
                interval="1d",
                auto_adjust=False,
                actions=False,
                group_by="ticker",
                session=_get_yf_session(),
            )
            if batch_hist is None or batch_hist.empty:
                logger.warning(
                    f"get_quotes: yf.download returned empty data for {len(stock_symbols)} symbols. "
                    f"Yahoo Finance may be rate-limiting or blocking requests."
                )
            for sym in stock_symbols:
                try:
                    if len(stock_symbols) > 1:
                        if batch_hist is None or batch_hist.empty or sym not in batch_hist.columns.levels[1]:
                            continue
                        sym_data = batch_hist[sym]
                    else:
                        sym_data = batch_hist

                    if len(sym_data) >= 1:
                        last = sym_data["Close"].iloc[-1]
                        if last is not None and not pd.isna(last) and last > 0:
                            result[sym]["last"] = float(last)
                        vol = sym_data["Volume"].iloc[-1] if "Volume" in sym_data.columns else None
                        if vol is not None and not pd.isna(vol):
                            result[sym]["volume"] = float(vol)
                            result[sym]["quoteVolume"] = float(vol)
                    if len(sym_data) >= 2:
                        prev_close = sym_data["Close"].iloc[-2]
                        if prev_close is not None and not pd.isna(prev_close) and prev_close > 0:
                            last_val = result[sym].get("last")
                            if last_val is not None:
                                result[sym]["change_24h"] = last_val - prev_close
                                result[sym]["percentage"] = ((last_val - prev_close) / prev_close) * 100
                except (KeyError, ValueError, AttributeError, IndexError):
                    pass
        except (RuntimeError, ValueError, ConnectionError, OSError) as e:
            logger.warning(f"Batch download failed: {e}")
    elif stock_symbols and _check_yf_circuit():
        logger.warning(
            f"get_quotes: yfinance circuit breaker is OPEN — skipping quote fetch for {len(stock_symbols)} symbols. "
            f"Quotes will be served from Redis cache or database if available."
        )

    # --- Try Alpha Vantage for stocks still missing valid prices ---
    missing_after_yf = [
        sym for sym in missing_symbols
        if result.get(sym, {}).get("last") is None
    ]
    if missing_after_yf:
        for sym in missing_after_yf[:10]:
            av_quote = get_alphavantage_quote(sym)
            if av_quote:
                result[sym].update(av_quote)
                logger.debug(f"get_quotes: Alpha Vantage provided quote for {sym}")

    # --- Try IEX Cloud for stocks still missing valid prices ---
    missing_after_av = [
        sym for sym in missing_symbols
        if result.get(sym, {}).get("last") is None
    ]
    if missing_after_av:
        for sym in missing_after_av[:10]:
            iex_quote = get_iex_quote(sym)
            if iex_quote:
                result[sym].update(iex_quote)
                logger.debug(f"get_quotes: IEX Cloud provided quote for {sym}")

    # --- Try Borsa Italiana for Italian stocks AND BTPs still missing valid prices ---
    missing_after_iex = [
        sym for sym in missing_symbols
        if result.get(sym, {}).get("last") is None
    ]
    if missing_after_iex:
        # Limit to 5 symbols to avoid excessive API calls
        for sym in missing_after_iex[:5]:
            bi_quote = get_borsa_italiana_quote(sym)
            if bi_quote:
                result[sym].update(bi_quote)
                logger.debug(f"get_quotes: Borsa Italiana provided quote for {sym}")

    # --- Final pass: compute change_24h and percentage from DB daily candles ---
    # For ALL symbols with a valid last price, recompute change_24h and percentage
    # from the latest 2 daily candles in the database. This ensures consistency
    # and eliminates NULL values when daily candle data is available.
    # DB candles are the primary source; yfinance is only a fallback for the last price.
    symbols_with_price = [
        sym for sym in result
        if result[sym].get("last") is not None and result[sym]["last"] > 0
    ]
    if symbols_with_price:
        try:
            db_change_data = get_latest_close_prices(symbols_with_price)
            for sym in symbols_with_price:
                if sym in db_change_data:
                    prev_close = db_change_data[sym].get("prev_close")
                    if prev_close and prev_close > 0:
                        last = result[sym]["last"]
                        result[sym]["change_24h"] = last - prev_close
                        result[sym]["percentage"] = round((last - prev_close) / prev_close * 100, 4)
        except (RuntimeError, ValueError, KeyError, OSError) as e:
            logger.warning(f"Failed to recompute change_24h/percentage from DB candles: {e}")

    # Ensure bid/ask are never NULL when last is available — use last as fallback
    for sym in result:
        if result[sym].get("last") is not None and result[sym]["last"] > 0:
            if result[sym].get("bid") is None:
                result[sym]["bid"] = result[sym]["last"]
            if result[sym].get("ask") is None:
                result[sym]["ask"] = result[sym]["last"]

    # Cache the result per-symbol in Redis (5 minutes) and save to database
    quotes_to_save = {}
    for sym in result:
        if result[sym].get("last") is not None:
            try:
                redis_client.set(f"quote:{sym}", json.dumps(result[sym]), ex=300)
            except (TypeError, ValueError, RuntimeError):
                pass
            quotes_to_save[sym] = result[sym]

    # Save to database for persistence (survives Redis flushes and yfinance outages)
    if quotes_to_save:
        try:
            save_quotes_batch(quotes_to_save)
        except (RuntimeError, ValueError, OSError) as e:
            logger.warning(f"Failed to save quotes to database: {e}")

    # Summary log
    valid_count = sum(1 for sym in missing_symbols if result[sym].get("last") is not None)
    if valid_count == 0 and missing_symbols:
        if _check_yf_circuit():
            logger.debug(
                f"get_quotes: 0/{len(missing_symbols)} symbols got valid prices "
                f"(circuit breaker open)."
            )
        else:
            logger.warning(
                f"get_quotes: 0/{len(missing_symbols)} symbols got valid prices. "
                f"Check yfinance connectivity and proxy settings."
            )
    else:
        logger.debug(f"get_quotes: {valid_count}/{len(missing_symbols)} symbols got valid prices")

    # Enrich BTP quotes with maturity, coupon, and name from discovered_symbols
    _enrich_quotes_with_btp_details(result, symbols)

    return result


def get_quotes_cached(symbols: List[str] = None) -> Dict[str, Dict[str, Any]]:
    """Fetch quotes from Redis cache and database only. No network calls.

    Used by the symbol re-evaluation loop which must never block on yfinance
    or Borsa Italiana API calls. The background quote refresh loop is
    responsible for keeping Redis and the database up to date.
    """
    if symbols is None:
        symbols = []
    if not symbols:
        return {}

    # Sanitize symbols: remove $ prefix and /currency suffix
    symbols = [s.lstrip('$').split('/')[0] for s in symbols]

    redis_client = get_redis_client()
    result = {}
    missing_symbols = []

    # Check per-symbol Redis cache first
    for sym in symbols:
        cache_key = f"quote:{sym}"
        try:
            cached = redis_client.get(cache_key)
            if cached:
                result[sym] = json.loads(cached)
            else:
                missing_symbols.append(sym)
        except (TypeError, ValueError, RuntimeError):
            missing_symbols.append(sym)

    if not missing_symbols:
        return result

    # Check database for quotes not in Redis cache (up to 24 hours old)
    try:
        db_quotes = get_quotes_from_db(missing_symbols, max_age_seconds=86400)
        for sym in list(missing_symbols):
            if sym in db_quotes:
                result[sym] = db_quotes[sym]
                missing_symbols.remove(sym)
                # Refresh Redis cache from DB data
                try:
                    redis_client.set(f"quote:{sym}", json.dumps(db_quotes[sym]), ex=300)
                except Exception:
                    pass
    except (RuntimeError, ValueError, OSError) as e:
        logger.warning(f"get_quotes_cached: DB quote fetch failed: {e}", exc_info=True)

    # Try DB close prices for anything still missing
    if missing_symbols:
        try:
            db_candles = get_latest_close_prices(missing_symbols)
            for sym in list(missing_symbols):
                if sym in db_candles and db_candles[sym].get("last", 0) > 0:
                    last = db_candles[sym]["last"]
                    prev_close = db_candles[sym].get("prev_close")
                    volume = db_candles[sym].get("volume")

                    abs_change = None
                    pct = None
                    if prev_close and prev_close > 0:
                        abs_change = last - prev_close
                        pct = round((abs_change / prev_close) * 100, 4)

                    result[sym] = {
                        "last": last,
                        "bid": last,
                        "ask": last,
                        "volume": volume,
                        "change_24h": abs_change,
                        "percentage": pct,
                        "quoteVolume": volume,
                        "last_update": db_candles[sym].get("candle_timestamp"),
                        "source": "db_close",
                    }
                    missing_symbols.remove(sym)
        except (RuntimeError, ValueError, KeyError, OSError) as e:
            logger.warning(f"get_quotes_cached: DB close price fallback failed: {e}")

    # Initialize remaining missing symbols with None values
    for sym in missing_symbols:
        result[sym] = {"last": None, "bid": None, "ask": None, "volume": None,
                       "change_24h": None, "percentage": None, "quoteVolume": None}

    # --- Final pass: compute change_24h and percentage from DB daily candles ---
    # For ALL symbols with a valid last price, recompute change_24h and percentage
    # from the latest 2 daily candles in the database. This ensures consistency
    # and eliminates NULL values when daily candle data is available.
    # DB candles are the primary source; yfinance is only a fallback for the last price.
    symbols_with_price = [
        sym for sym in result
        if result[sym].get("last") is not None and result[sym]["last"] > 0
    ]
    if symbols_with_price:
        try:
            db_change_data = get_latest_close_prices(symbols_with_price)
            for sym in symbols_with_price:
                if sym in db_change_data:
                    prev_close = db_change_data[sym].get("prev_close")
                    if prev_close and prev_close > 0:
                        last = result[sym]["last"]
                        result[sym]["change_24h"] = last - prev_close
                        result[sym]["percentage"] = round((last - prev_close) / prev_close * 100, 4)
        except (RuntimeError, ValueError, KeyError, OSError) as e:
            logger.warning(f"get_quotes_cached: Failed to recompute change_24h/percentage from DB candles: {e}")

    # Ensure bid/ask are never NULL when last is available — use last as fallback
    for sym in result:
        if result[sym].get("last") is not None and result[sym]["last"] > 0:
            if result[sym].get("bid") is None:
                result[sym]["bid"] = result[sym]["last"]
            if result[sym].get("ask") is None:
                result[sym]["ask"] = result[sym]["last"]

    # Persist DB close prices to Redis and the quotes table so that other
    # consumers (web dashboard, re-evaluation, etc.) can access them even
    # when yfinance is unavailable.  This is a fast batch DB write, not a
    # network call, so it respects the "no network calls" contract of this
    # function.
    quotes_to_save = {}
    for sym, q in result.items():
        if q.get("last") is not None:
            try:
                redis_client.set(f"quote:{sym}", json.dumps(q), ex=300)
            except (TypeError, ValueError, RuntimeError):
                pass
            quotes_to_save[sym] = q
    if quotes_to_save:
        try:
            save_quotes_batch(quotes_to_save)
        except (RuntimeError, ValueError, OSError) as e:
            logger.warning(f"get_quotes_cached: Failed to save DB close prices to quotes table: {e}")

    # Enrich BTP quotes with maturity, coupon, and name from discovered_symbols
    _enrich_quotes_with_btp_details(result, symbols)

    return result


def get_multi_timeframe_bars(
    symbol: str = "", timeframes: List[str] = None, limit: int = 24
) -> Dict[str, List[List[float]]]:
    if timeframes is None:
        timeframes = []
    """Fetch OHLCV bars for a symbol across multiple timeframes using yfinance.

    Returns a dict mapping timeframe -> list of candles [timestamp_ms, open, high, low, close, volume].
    """
    if not timeframes:
        return {}

    # Sanitize symbol: remove $ prefix and /currency suffix
    symbol = symbol.lstrip('$').split('/')[0]

    # Format symbol for Yahoo Finance: BTP ISINs are used as-is, stocks get TICKER_SUFFIX if missing
    yf_symbol = symbol
    if not re.match(r'^IT[A-Z0-9]{10}$', symbol) and settings.TICKER_SUFFIX and not symbol.endswith(settings.TICKER_SUFFIX):
        yf_symbol = f"{symbol}{settings.TICKER_SUFFIX}"

    redis_client = get_redis_client()
    cache_ttl = 60 if any(tf in ("1h",) for tf in timeframes) else 300

    result = {}
    for tf in timeframes:
        interval = TIMEFRAME_MAP.get(tf)
        if not interval:
            logger.warning(f"Unsupported timeframe: {tf}")
            continue

        needs_aggregation = tf in ("6M", "1Y", "3Y", "5Y")
        fetch_interval = "1mo" if needs_aggregation else interval

        cache_key = f"ohlcv:{symbol}:{tf}:{limit}"
        try:
            cached = redis_client.get(cache_key)
            if cached:
                result[tf] = json.loads(cached)
                continue
        except (TypeError, ValueError, RuntimeError):
            pass

        # BTPs: only borsaitaliana, no yfinance
        if is_btp_isin(symbol):
            borsa_candles = get_borsa_italiana_candles(symbol, tf, limit=limit)
            result[tf] = borsa_candles or []
            if borsa_candles:
                try:
                    redis_client.set(cache_key, json.dumps(result[tf]), ex=cache_ttl)
                except (TypeError, ValueError, RuntimeError):
                    pass
            continue

        # Check if we have ISIN in DB (strip suffix — DB stores base symbols)
        from src.database import get_isin_from_db
        db_lookup_symbol = symbol
        if settings.TICKER_SUFFIX and db_lookup_symbol.endswith(settings.TICKER_SUFFIX):
            db_lookup_symbol = db_lookup_symbol[:-len(settings.TICKER_SUFFIX)]
        db_isin = get_isin_from_db(db_lookup_symbol)
        has_isin = db_isin is not None

        # If we have ISIN, only use borsaitaliana (skip yfinance to avoid rate limits)
        borsa_candles = None
        if has_isin:
            borsa_candles = get_borsa_italiana_candles(symbol, tf, limit=limit)
            if borsa_candles:
                result[tf] = borsa_candles[-limit:] if limit else borsa_candles
                try:
                    redis_client.set(cache_key, json.dumps(result[tf]), ex=cache_ttl)
                except (TypeError, ValueError, RuntimeError):
                    pass
            else:
                result[tf] = []
            continue

        # No ISIN — use yfinance as fallback, then Alpha Vantage, then IEX
        yf_candles: List[List] = []
        if not _check_yf_circuit():
            yf_symbol = symbol
            if not re.match(r'^IT[A-Z0-9]{10}$', symbol) and settings.TICKER_SUFFIX and not symbol.endswith(settings.TICKER_SUFFIX):
                yf_symbol = f"{symbol}{settings.TICKER_SUFFIX}"

            try:
                ticker = yf.Ticker(yf_symbol, session=_get_yf_session())
                if fetch_interval in ("5m", "15m"):
                    period = "60d"
                elif fetch_interval == "60m":
                    period = "730d"
                else:
                    period = "max"
                hist = ticker.history(period=period, interval=fetch_interval, auto_adjust=False, actions=False)
                if not hist.empty:
                    ohlcv_cols = ["Open", "High", "Low", "Close", "Volume"]
                    hist = hist[[col for col in ohlcv_cols if col in hist.columns]]
                    candles = []
                    for idx, row in hist.iterrows():
                        ts = int(idx.timestamp() * 1000)
                        candles.append([ts, row["Open"], row["High"], row["Low"], row["Close"], row["Volume"]])
                    if needs_aggregation:
                        candles = _aggregate_candles(candles, tf)
                    yf_candles = candles
            except (RuntimeError, ValueError, KeyError, AttributeError, OSError) as e:
                logger.debug(f"yfinance fetch failed for {symbol} {tf}: {e}")

        # Try Alpha Vantage if yfinance returned nothing
        av_candles = None
        if not yf_candles:
            av_candles = get_alphavantage_candles(symbol, tf, limit=limit)
            if av_candles:
                logger.debug(f"Alpha Vantage provided candles for {symbol} {tf}")

        # Try IEX Cloud if both yfinance and Alpha Vantage returned nothing
        iex_candles = None
        if not yf_candles and not av_candles:
            iex_candles = get_iex_candles(symbol, tf, limit=limit)
            if iex_candles:
                logger.debug(f"IEX Cloud provided candles for {symbol} {tf}")

        # Merge all sources (borsa > yf > av > iex precedence by timestamp)
        merged = _merge_candles(borsa_candles, yf_candles)
        if av_candles:
            merged = _merge_candles(av_candles, merged)
        if iex_candles:
            merged = _merge_candles(iex_candles, merged)
        if merged:
            merged = _validate_and_clean_candles(merged, symbol)
            result[tf] = merged[-limit:] if limit else merged
            try:
                redis_client.set(cache_key, json.dumps(result[tf]), ex=cache_ttl)
            except (TypeError, ValueError, RuntimeError):
                pass
        else:
            result[tf] = []
    return result


def get_bars_range(
    symbol: str = "", timeframe: str = "", start_ms: int = 0, limit: int = 500
) -> List[List[float]]:
    """Fetch OHLCV bars from a start timestamp (ms) up to the present using yfinance.

    Returns a list of candles [timestamp_ms, open, high, low, close, volume].
    """
    # Sanitize symbol: remove $ prefix and /currency suffix
    symbol = symbol.lstrip('$').split('/')[0]

    interval = TIMEFRAME_MAP.get(timeframe)
    if not interval:
        logger.warning(f"Unsupported timeframe: {timeframe}")
        return []

    # Yahoo Finance does not support 6mo, 1y, 3y, 5y intervals natively.
    # We fetch 1mo data and aggregate it.
    needs_aggregation = timeframe in ("6M", "1Y", "3Y", "5Y")
    fetch_interval = "1mo" if needs_aggregation else interval

    redis_client = get_redis_client()
    cache_key = f"ohlcv_range:{symbol}:{timeframe}:{start_ms}:{limit}"
    try:
        cached = redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
    except (TypeError, ValueError, RuntimeError):
        pass

    # BTPs: only borsaitaliana, no yfinance
    if is_btp_isin(symbol):
        borsa_candles = get_borsa_italiana_candles(symbol, timeframe, limit=limit, start_ms=start_ms)
        if borsa_candles:
            try:
                redis_client.set(cache_key, json.dumps(borsa_candles), ex=300)
            except (TypeError, ValueError, RuntimeError):
                pass
            return borsa_candles
        return []

    # Check if we have ISIN in DB (strip suffix — DB stores base symbols)
    from src.database import get_isin_from_db
    db_lookup_symbol = symbol
    if settings.TICKER_SUFFIX and db_lookup_symbol.endswith(settings.TICKER_SUFFIX):
        db_lookup_symbol = db_lookup_symbol[:-len(settings.TICKER_SUFFIX)]
    db_isin = get_isin_from_db(db_lookup_symbol)
    has_isin = db_isin is not None

    # If we have ISIN, only use borsaitaliana (skip yfinance to avoid rate limits)
    borsa_candles = None
    if has_isin:
        borsa_candles = get_borsa_italiana_candles(symbol, timeframe, limit=limit, start_ms=start_ms)
        if borsa_candles:
            if limit and len(borsa_candles) > limit:
                borsa_candles = borsa_candles[-limit:]
            try:
                redis_client.set(cache_key, json.dumps(borsa_candles), ex=300)
            except Exception:
                pass
            return borsa_candles
        return []

    # No ISIN — use yfinance as fallback, then Alpha Vantage, then IEX
    yf_candles: List[List] = []
    if not _check_yf_circuit():
        yf_symbol = symbol
        if settings.TICKER_SUFFIX and not symbol.endswith(settings.TICKER_SUFFIX):
            yf_symbol = f"{symbol}{settings.TICKER_SUFFIX}"

        start_dt = datetime.fromtimestamp(start_ms / 1000.0, tz=timezone.utc)
        end_dt = datetime.now(timezone.utc)

        if interval in ("5m", "15m", "60m"):
            earliest_allowed = datetime.now(timezone.utc) - timedelta(days=730)
            if start_dt < earliest_allowed:
                start_dt = earliest_allowed

        try:
            ticker = yf.Ticker(yf_symbol, session=_get_yf_session())
            hist = ticker.history(start=start_dt, end=end_dt, interval=fetch_interval, auto_adjust=False, actions=False)
            if not hist.empty:
                ohlcv_cols = ["Open", "High", "Low", "Close", "Volume"]
                hist = hist[[col for col in ohlcv_cols if col in hist.columns]]
                candles = []
                for idx, row in hist.iterrows():
                    ts = int(idx.timestamp() * 1000)
                    candles.append([ts, row["Open"], row["High"], row["Low"], row["Close"], row["Volume"]])
                if needs_aggregation:
                    candles = _aggregate_candles(candles, timeframe)
                yf_candles = candles
        except (RuntimeError, ValueError, KeyError, AttributeError, OSError) as e:
            logger.debug(f"yfinance fetch failed for {symbol} {timeframe}: {e}")

    # Try Alpha Vantage if yfinance returned nothing
    av_candles = None
    if not yf_candles:
        av_candles = get_alphavantage_candles(symbol, timeframe, limit=limit, start_ms=start_ms)
        if av_candles:
            logger.debug(f"Alpha Vantage provided candles for {symbol} {timeframe}")

    # Try IEX Cloud if both yfinance and Alpha Vantage returned nothing
    iex_candles = None
    if not yf_candles and not av_candles:
        iex_candles = get_iex_candles(symbol, timeframe, limit=limit, start_ms=start_ms)
        if iex_candles:
            logger.debug(f"IEX Cloud provided candles for {symbol} {timeframe}")

    # Merge all sources (borsa > yf > av > iex precedence by timestamp)
    merged = _merge_candles(borsa_candles, yf_candles)
    if av_candles:
        merged = _merge_candles(av_candles, merged)
    if iex_candles:
        merged = _merge_candles(iex_candles, merged)
    merged = _validate_and_clean_candles(merged, symbol)

    if merged:
        if limit and len(merged) > limit:
            merged = merged[-limit:]
        try:
            redis_client.set(cache_key, json.dumps(merged), ex=300)
        except (TypeError, ValueError, RuntimeError):
            pass
        return merged
    return []


