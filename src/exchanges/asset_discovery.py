import asyncio
import json
import logging
import re
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional

import pandas as pd
import requests
import yfinance as yf

from src.config.settings import settings
from src.utils.redis_client import get_redis_client
from src.utils.symbol_utils import is_btp_isin, is_italian_isin
from src.exchanges.yf_session import _check_yf_circuit, _get_yf_session
from src.exchanges.borsa_italiana_utils import _get_isin_and_info_from_borsa_italiana

logger = logging.getLogger(__name__)

_notifier = None
_ddg_lookup_count = 0
MAX_DDG_LOOKUPS = settings.MAX_DDG_LOOKUPS

def set_notifier(notifier):
    global _notifier
    _notifier = notifier

def _fetch_info(symbol: str, max_retries: int = 2) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Fetch the country, name, and ISIN from yfinance info for a symbol, with retries.

    Returns a tuple (country, name, isin) on success, or (None, None, None) if yfinance
    could not provide the information after all retries.
    """
    country, name = None, None
    bi_isin = None
    if not _check_yf_circuit():
        import time as _time
        for attempt in range(max_retries + 1):
            try:
                ticker = yf.Ticker(symbol, session=_get_yf_session())
                info = ticker.info
                country = info.get("country")
                name = info.get("longName") or info.get("shortName")
                if country or name:
                    # Also try to get ISIN from yfinance
                    try:
                        yf_isin = ticker.isin
                        if yf_isin and yf_isin.strip() and yf_isin.strip() != '-':
                            yf_isin = yf_isin.strip()
                            # If strict country filter is enabled, discard non-target ISINs
                            if settings.COUNTRY_FILTER_STRICT and not yf_isin.startswith(settings.TARGET_COUNTRY):
                                logger.debug(f"yfinance returned non-target ISIN {yf_isin} for {symbol}, discarding.")
                            else:
                                bi_isin = yf_isin
                    except (RuntimeError, ValueError, KeyError, AttributeError, OSError):
                        pass
                    break
                # country is None or empty – retry if attempts remain
                if attempt < max_retries:
                    _time.sleep(0.5 * (2 ** attempt))
                    continue
            except (RuntimeError, ValueError, KeyError, AttributeError, OSError) as e:
                logger.debug(f"Failed to fetch info for {symbol} (attempt {attempt + 1}/{max_retries + 1}): {type(e).__name__}: {e}")
                if attempt < max_retries:
                    _time.sleep(0.5 * (2 ** attempt))

    # Fallback to Borsa Italiana search if yfinance failed, circuit is open, or ISIN is missing
    if not country or not name or not bi_isin:
        # Strip suffix for Borsa Italiana search
        db_symbol = symbol
        if settings.TICKER_SUFFIX and db_symbol.endswith(settings.TICKER_SUFFIX):
            db_symbol = db_symbol[:-len(settings.TICKER_SUFFIX)]

        bi_isin_new, bi_country, bi_name = _get_isin_and_info_from_borsa_italiana(db_symbol)
        if bi_isin_new:
            bi_isin = bi_isin_new
        # In strict mode, Borsa Italiana is the source of truth for country
        if settings.COUNTRY_FILTER_STRICT and bi_country:
            country = bi_country
        elif not country:
            country = bi_country
        if not name:
            name = bi_name

    # Fallback to DuckDuckGo AI Chat API if ISIN is still missing
    global _ddg_lookup_count
    if not bi_isin and _ddg_lookup_count < MAX_DDG_LOOKUPS:
        _ddg_lookup_count += 1
        try:
            from src.exchanges.duckduckgo_utils import get_isin_from_duckduckgo
            ddg_isin = get_isin_from_duckduckgo(db_symbol, name)
            if ddg_isin:
                bi_isin = ddg_isin
        except Exception as e:
            logger.debug(f"DuckDuckGo ISIN fetch failed for {db_symbol}: {type(e).__name__}: {e}")

    if country or name:
        return country, name, bi_isin
    return None, None, None


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
            logger.debug(f"Failed to scrape {url}: {type(e).__name__}: {str(e)[:200]}")
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
    except (TypeError, ValueError, RuntimeError) as e:
        logger.debug(f"discover_italian_ucits_etfs: failed to read/write Redis cache: {type(e).__name__}: {e}")

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
        etf_names: Dict[str, str] = {}
        for symbol, row in df.iterrows():
            raw_name = str(row.get('name', ''))
            name = raw_name.lower()
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
                if raw_name:
                    etf_names[base] = raw_name

        logger.info(f"Discovered {len(base_symbols)} Italian UCITS ETFs matching keywords.")
        # Cache for 24 hours
        try:
            redis_client.set(cache_key, json.dumps(base_symbols), ex=86400)
        except (TypeError, ValueError, RuntimeError) as e:
            logger.debug(f"discover_italian_ucits_etfs: failed to write Redis cache: {type(e).__name__}: {e}")
        # Save ETF symbols to DB
        try:
            from src.database import save_discovered_symbols_batch
            symbols_to_save = [
                {"symbol": sym, "isin": None, "asset_type": "etf", "name": etf_names.get(sym), "country": "italy"}
                for sym in base_symbols
            ]
            if symbols_to_save:
                save_discovered_symbols_batch(symbols_to_save)
        except (RuntimeError, ValueError, OSError) as e:
            logger.warning(f"Failed to save ETF symbols to DB: {type(e).__name__}: {e}")
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
    except (RuntimeError, ValueError, OSError) as e:
        logger.debug(f"_save_discovered_assets_to_db: batch ISIN lookup failed: {type(e).__name__}: {e}")
    if symbols_to_save:
        try:
            save_discovered_symbols_batch(symbols_to_save)
            logger.info(f"Saved {len(symbols_to_save)} discovered symbols to database")
        except (RuntimeError, ValueError, OSError) as e:
            logger.warning(f"Failed to save discovered symbols to database: {type(e).__name__}: {e}")


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
        cache_key = f"tradable_assets:{settings.TARGET_COUNTRY}:{settings.COUNTRY_FILTER_STRICT}"
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
                            # Sanitize db_sym: strip /currency suffix, then strip ticker suffix if present
                            clean_sym = db_sym.split("/")[0] if "/" in db_sym else db_sym
                            if suffix and clean_sym.endswith(suffix):
                                clean_sym = clean_sym[:-len(suffix)]
                            candidate = f"{clean_sym}{suffix}" if suffix else clean_sym
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
                    db_isin = db_entry.get("isin")
                    if db_isin is not None and not is_italian_isin(db_isin):
                        continue
                    if is_btp_isin(db_sym):
                        db_only_list.append(db_sym)
                    else:
                        # Sanitize db_sym: strip /currency suffix, then strip ticker suffix if present
                        clean_sym = db_sym.split("/")[0] if "/" in db_sym else db_sym
                        if suffix and clean_sym.endswith(suffix):
                            clean_sym = clean_sym[:-len(suffix)]
                        candidate = f"{clean_sym}{suffix}" if suffix else clean_sym
                        db_only_list.append(candidate)
                if db_only_list:
                    logger.info(f"Discovery failed but recovered {len(db_only_list)} symbols from DB only")
                    return db_only_list
        except (RuntimeError, ValueError, OSError) as e:
            logger.warning(f"Failed to recover symbols from DB: {type(e).__name__}: {e}")
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
            logger.warning(f"Failed to save discovered assets to DB: {type(e).__name__}: {e}")

    suffix = settings.TICKER_SUFFIX
    candidates = []
    for sym in base_symbols:
        if is_btp_isin(sym):
            candidates.append(sym)          # BTP ISIN – no suffix
        else:
            candidates.append(f"{sym}{suffix}")

    # Check Redis cache
    redis_client = get_redis_client()
    cache_key = f"tradable_assets:{settings.TARGET_COUNTRY}:{settings.COUNTRY_FILTER_STRICT}"
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
                    db_isin = db_entry.get("isin")
                    if db_isin is not None and not is_italian_isin(db_isin):
                        continue
                    if is_btp_isin(db_sym):
                        if db_sym not in existing_set:
                            cached_list.append(db_sym)
                            existing_set.add(db_sym)
                    else:
                        # Sanitize db_sym: strip /currency suffix, then strip ticker suffix if present
                        clean_sym = db_sym.split("/")[0] if "/" in db_sym else db_sym
                        if suffix and clean_sym.endswith(suffix):
                            clean_sym = clean_sym[:-len(suffix)]
                        candidate = f"{clean_sym}{suffix}" if suffix else clean_sym
                        if candidate not in existing_set:
                            cached_list.append(candidate)
                            existing_set.add(candidate)
            except Exception as e:
                logger.warning(f"Failed to merge DB symbols with cached list: {type(e).__name__}: {e}")
            return cached_list
    except (TypeError, ValueError, RuntimeError):
        pass

    global _ddg_lookup_count
    _ddg_lookup_count = 0

    # Filter candidates by country using yfinance
    strict = settings.COUNTRY_FILTER_STRICT
    filtered = []
    
    # Separate BTPs and non-BTPs
    btp_candidates = [s for s in candidates if is_btp_isin(s)]
    non_btp_candidates = [s for s in candidates if not is_btp_isin(s)]
    
    if target_country == "italy":
        filtered.extend(btp_candidates)
        
    # Parallelize _fetch_info for non-BTP candidates
    fetch_results = {}
    max_workers = min(10, len(non_btp_candidates)) if non_btp_candidates else 1
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {executor.submit(_fetch_info, sym): sym for sym in non_btp_candidates}
        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                country, name, isin = future.result()
                fetch_results[symbol] = (country, name, isin)
            except Exception as e:
                logger.debug(f"Failed to fetch info for {symbol}: {e}")
                fetch_results[symbol] = (None, None, None)
                
    for symbol in non_btp_candidates:
        country, name, isin = fetch_results.get(symbol, (None, None, None))
        
        # Skip symbols with a non-Italian ISIN
        if isin is not None and not is_italian_isin(isin):
            logger.debug(f"Symbol {symbol} skipped (ISIN {isin} is not Italian)")
            continue

        # Save the fetched country, name, and ISIN to the database for future filtering.
        # In strict mode, only save Italian symbols to DB.
        if country is not None and (not settings.COUNTRY_FILTER_STRICT or country.lower() == target_country):
            try:
                from src.database import save_discovered_symbol
                db_base = symbol
                if suffix and db_base.endswith(suffix):
                    db_base = db_base[:-len(suffix)]
                asset_type = "etf" if db_base in etf_symbols else "stock"
                save_discovered_symbol(db_base, isin, asset_type, name or None, country=country)
            except Exception as e:
                logger.debug(f"get_tradable_assets: failed to save discovered symbol {db_base}: {type(e).__name__}: {e}")
        elif name and not settings.COUNTRY_FILTER_STRICT:
            # Country is None but name is available — save the name for display
            try:
                from src.database import save_discovered_symbol
                db_base = symbol
                if suffix and db_base.endswith(suffix):
                    db_base = db_base[:-len(suffix)]
                asset_type = "etf" if db_base in etf_symbols else "stock"
                save_discovered_symbol(db_base, isin, asset_type, name or None, country=None)
            except (RuntimeError, ValueError, OSError) as e:
                logger.debug(f"get_tradable_assets: failed to save discovered symbol {db_base}: {type(e).__name__}: {e}")
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
        logger.warning(f"Failed to cache tradable assets: {type(e).__name__}: {e}")

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
            db_isin = db_entry.get("isin")
            if db_isin is not None and not is_italian_isin(db_isin):
                continue
            if is_btp_isin(db_sym):
                # BTP ISIN — add as-is (no suffix)
                if db_sym not in existing_set:
                    filtered.append(db_sym)
                    existing_set.add(db_sym)
            else:
                # Stock/ETF — add with suffix
                # Sanitize db_sym: strip /currency suffix, then strip ticker suffix if present
                clean_sym = db_sym.split("/")[0] if "/" in db_sym else db_sym
                if suffix and clean_sym.endswith(suffix):
                    clean_sym = clean_sym[:-len(suffix)]
                candidate = f"{clean_sym}{suffix}" if suffix else clean_sym
                if candidate not in existing_set:
                    filtered.append(candidate)
                    existing_set.add(candidate)
        logger.info(f"Merged {len(db_symbols)} symbols from DB, total: {len(filtered)}")
    except (RuntimeError, ValueError, OSError) as e:
        logger.warning(f"Failed to merge discovered symbols from DB: {type(e).__name__}: {e}")

    logger.info(f"Tradable assets for {settings.TARGET_COUNTRY}: {len(filtered)} symbols")
    return filtered
