import json
import logging
from typing import Any, Dict, Optional

import yfinance as yf

from src.config.settings import settings
from src.exchanges.market_data import _get_yf_session
from src.utils.redis_client import get_redis_client

logger = logging.getLogger(__name__)


def get_yahoo_quote(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Fetch Level 1 quote (bid, ask, last) from Yahoo Finance for a US stock/ETF.

    Returns a dict with keys 'bid', 'ask', 'last', or None if unavailable.
    Results are cached in Redis for YAHOO_FINANCE_CACHE_SECONDS.
    """
    if not settings.YAHOO_FINANCE_ENABLED:
        return None

    # Normalise symbol: yfinance expects ticker without exchange suffix
    base = symbol.split("/")[0] if "/" in symbol else symbol
    base = base.lstrip('$')

    redis_client = get_redis_client()
    cache_key = f"yahoo_quote:{base}"
    try:
        cached = redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
    except Exception:
        pass

    try:
        ticker = yf.Ticker(base, session=_get_yf_session())
        last = None
        bid = None
        ask = None

        try:
            # fast_info is the quickest way to get current price data
            info = ticker.fast_info
            last = info.get("lastPrice")
            bid = info.get("bid")
            ask = info.get("ask")
        except Exception as e:
            logger.debug(f"fast_info failed for {base}: {e}")

        # Fallback to regular info if fast_info lacks bid/ask or last
        if bid is None or ask is None or last is None:
            try:
                info2 = ticker.info
                bid = bid or info2.get("bid")
                ask = ask or info2.get("ask")
                if last is None:
                    last = info2.get("regularMarketPrice") or info2.get("currentPrice")
            except Exception as e:
                logger.debug(f"info failed for {base}: {e}")

        if last is None:
            # Last resort: get the latest daily close
            hist = ticker.history(period="1d")
            if not hist.empty:
                last = hist["Close"].iloc[-1]

        if last is None:
            return None

        result = {
            "last": last,
            "bid": bid,
            "ask": ask,
        }
        # Cache the result
        ttl = settings.YAHOO_FINANCE_CACHE_SECONDS
        redis_client.setex(cache_key, ttl, json.dumps(result))
        return result
    except Exception as e:
        logger.warning(f"Yahoo Finance quote failed for {base}: {e}")
        return None


def get_yahoo_fundamentals(symbol: str) -> Optional[Dict[str, Any]]:
    """Fetch key fundamentals (P/E, Market Cap, Sector, etc.) from Yahoo Finance."""
    if not settings.YAHOO_FINANCE_ENABLED:
        return None

    base = symbol.split("/")[0] if "/" in symbol else symbol
    base = base.lstrip('$')

    redis_client = get_redis_client()
    cache_key = f"yahoo_fundamentals:{base}"
    try:
        cached = redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
    except Exception:
        pass

    try:
        ticker = yf.Ticker(base, session=_get_yf_session())
        info = ticker.info
        def _safe_float(val):
            if val is None:
                return None
            try:
                return float(val)
            except (ValueError, TypeError):
                return None

        result = {
            "pe_ratio": _safe_float(info.get("trailingPE")),
            "forward_pe": _safe_float(info.get("forwardPE")),
            "market_cap": _safe_float(info.get("marketCap")),
            "dividend_yield": _safe_float(info.get("dividendYield")),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "price_to_book": _safe_float(info.get("priceToBook")),
            "profit_margins": _safe_float(info.get("profitMargins")),
            "return_on_equity": _safe_float(info.get("returnOnEquity")),
        }
        # Cache for 24 hours
        try:
            redis_client.setex(cache_key, 86400, json.dumps(result))
        except Exception:
            pass
        return result
    except Exception as e:
        logger.warning(f"Yahoo Finance fundamentals failed for {base}: {e}")
        return None
