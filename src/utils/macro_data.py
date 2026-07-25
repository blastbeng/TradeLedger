"""Fetches and caches macro economic context for LLM prompts."""
import json
import logging
from typing import Dict, Any

from src.utils.redis_client import get_redis_client

logger = logging.getLogger(__name__)

_MACRO_CACHE_KEY = "macro:economic_context"
_MACRO_CACHE_TTL = 3600  # 1 hour

def _fetch_macro_data() -> Dict[str, Any]:
    """Fetch basic macro economic indicators using yfinance."""
    try:
        import yfinance as yf
        from src.exchanges.yf_session import _get_yf_session, _check_yf_circuit
    except ImportError:
        logger.warning("yfinance not installed, cannot fetch macro economic context.")
        return {}

    if _check_yf_circuit():
        return {}

    # Tickers representing key global and European macro indicators
    tickers = {
        "EUR_USD": "EURUSD=X",
        "US_10Y_Yield": "^TNX",
        "Brent_Crude": "BZ=F",
        "Gold": "GC=F",
        "FTSE_MIB": "FTSEMIB.MI",
    }

    data = {}
    try:
        session = _get_yf_session()
        for name, ticker_symbol in tickers.items():
            try:
                ticker = yf.Ticker(ticker_symbol, session=session)
                info = ticker.fast_info
                last_price = info.get("lastPrice")
                if last_price is not None:
                    if name == "US_10Y_Yield":
                        # ^TNX yield is multiplied by 10 in yfinance
                        data[name] = f"{last_price / 10:.2f}%"
                    else:
                        data[name] = f"{last_price:.2f}"
            except Exception as e:
                logger.debug(f"Failed to fetch macro data for {name} ({ticker_symbol}): {type(e).__name__}: {e}")
    except Exception as e:
        logger.warning(f"Failed to fetch macro economic context: {type(e).__name__}: {e}")

    return data


def get_macro_economic_context() -> Dict[str, Any]:
    """Return a dictionary of macro economic indicators, cached in Redis."""
    redis_client = get_redis_client()

    try:
        cached = redis_client.get(_MACRO_CACHE_KEY)
        if cached:
            return json.loads(cached)
    except Exception:
        pass

    data = _fetch_macro_data()

    if data:
        try:
            redis_client.set(_MACRO_CACHE_KEY, json.dumps(data), ex=_MACRO_CACHE_TTL)
        except Exception:
            pass

    return data
