import json
import logging
from typing import Any, Dict, List, Optional

import yfinance as yf

from src.config.settings import settings
from src.exchanges.market_data import _get_yf_session, _check_yf_circuit
from src.utils.redis_client import get_redis_client

logger = logging.getLogger(__name__)


def get_yahoo_quote(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Fetch Level 1 quote (bid, ask, last) from Yahoo Finance for a US stock/ETF.

    Returns a dict with keys 'bid', 'ask', 'last', or None if unavailable.
    Results are cached in Redis for YAHOO_FINANCE_CACHE_SECONDS.
    """
    if not settings.YAHOO_FINANCE_ENABLED or _check_yf_circuit():
        return None

    # Ensure we only fetch quotes if the symbol has a valid Italian ISIN
    from src.exchanges.market_data import _get_isin_from_yfinance
    if _get_isin_from_yfinance(symbol) is None:
        logger.debug(f"Skipping yfinance quote for {symbol}: no valid Italian ISIN.")
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
        redis_client.set(cache_key, json.dumps(result), ex=ttl)
        return result
    except Exception as e:
        logger.warning(f"Yahoo Finance quote failed for {base}: {type(e).__name__}: {e}")
        return None


def _validate_and_clean_fundamentals(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Validates and cleans fundamental data. Returns None if data is entirely empty."""
    if not data or all(v is None for v in data.values()):
        return None

    # yfinance sometimes returns negative values for missing data
    numeric_fields = ["pe_ratio", "forward_pe", "price_to_book"]
    for field in numeric_fields:
        if data.get(field) is not None and data[field] < 0:
            data[field] = None

    if data.get("market_cap") is not None and data["market_cap"] <= 0:
        data["market_cap"] = None

    if data.get("dividend_yield") is not None and (data["dividend_yield"] < 0 or data["dividend_yield"] > 1.0):
        data["dividend_yield"] = None

    return data


def get_yahoo_fundamentals(symbol: str) -> Optional[Dict[str, Any]]:
    """Fetch key fundamentals (P/E, Market Cap, Sector, etc.) from Yahoo Finance."""
    if not settings.YAHOO_FINANCE_ENABLED or _check_yf_circuit():
        return None

    # Ensure we only fetch fundamentals if the symbol has a valid Italian ISIN
    from src.exchanges.market_data import _get_isin_from_yfinance
    if _get_isin_from_yfinance(symbol) is None:
        logger.debug(f"Skipping yfinance fundamentals for {symbol}: no valid Italian ISIN.")
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

        # Validate and clean data before caching
        cleaned_result = _validate_and_clean_fundamentals(result)
        if cleaned_result is None:
            logger.warning(f"Yahoo Finance fundamentals for {base} returned empty/invalid data, skipping cache.")
            return None

        # Cache for 24 hours
        try:
            redis_client.set(cache_key, json.dumps(cleaned_result), ex=86400)
        except Exception:
            pass
        return cleaned_result
    except Exception as e:
        logger.warning(f"Yahoo Finance fundamentals failed for {base}: {type(e).__name__}: {e}")
        return None


def get_yahoo_dividends(symbol: str) -> List[Dict[str, Any]]:
    """Fetch dividend history from Yahoo Finance for a US stock/ETF.
    Returns a list of dicts with keys 'date' (ISO string) and 'amount' (float).
    """
    if not settings.YAHOO_FINANCE_ENABLED or _check_yf_circuit():
        return []

    # Ensure we only fetch dividends if the symbol has a valid Italian ISIN
    from src.exchanges.market_data import _get_isin_from_yfinance
    if _get_isin_from_yfinance(symbol) is None:
        logger.debug(f"Skipping yfinance dividends for {symbol}: no valid Italian ISIN.")
        return []

    base = symbol.split("/")[0] if "/" in symbol else symbol
    base = base.lstrip('$')

    redis_client = get_redis_client()
    cache_key = f"yahoo_dividends:{base}"
    try:
        cached = redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
    except Exception:
        pass

    try:
        ticker = yf.Ticker(base, session=_get_yf_session())
        divs = ticker.dividends
        if divs is None or divs.empty:
            return []
        result = []
        for date, amount in divs.items():
            result.append({"date": date.strftime("%Y-%m-%d"), "amount": float(amount)})
        try:
            redis_client.set(cache_key, json.dumps(result), ex=86400)
        except Exception:
            pass
        return result
    except Exception as e:
        logger.warning(f"Yahoo Finance dividends failed for {base}: {type(e).__name__}: {e}")
        return []


def get_yahoo_insider_transactions(symbol: str) -> Optional[List[Dict[str, Any]]]:
    """Fetch recent insider transactions from Yahoo Finance."""
    if not settings.YAHOO_FINANCE_ENABLED or _check_yf_circuit():
        return None

    from src.exchanges.market_data import _get_isin_from_yfinance
    if _get_isin_from_yfinance(symbol) is None:
        logger.debug(f"Skipping yfinance insider transactions for {symbol}: no valid Italian ISIN.")
        return None

    base = symbol.split("/")[0] if "/" in symbol else symbol
    base = base.lstrip('$')

    redis_client = get_redis_client()
    cache_key = f"yahoo_insider_transactions:{base}"
    try:
        cached = redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
    except Exception:
        pass

    try:
        ticker = yf.Ticker(base, session=_get_yf_session())
        transactions = ticker.insider_transactions

        if transactions is None or transactions.empty:
            return None

        # Get the most recent 5 transactions
        recent = transactions.head(5)
        result = []
        for _, row in recent.iterrows():
            result.append({
                "filer": row.get("Filer", ""),
                "transaction_date": str(row.get("Transaction Date", "")),
                "transaction_type": row.get("Transaction Type", ""),
                "shares": int(row.get("Shares", 0)) if row.get("Shares") is not None else 0,
                "value": float(row.get("Value", 0)) if row.get("Value") is not None else 0.0,
            })

        if not result:
            return None

        try:
            redis_client.set(cache_key, json.dumps(result), ex=86400)  # 24h
        except Exception:
            pass
        return result
    except Exception as e:
        logger.warning(f"Yahoo Finance insider transactions failed for {base}: {type(e).__name__}: {e}")
        return None


def get_yahoo_analyst_ratings(symbol: str) -> Optional[Dict[str, Any]]:
    """Fetch analyst ratings and target prices from Yahoo Finance."""
    if not settings.YAHOO_FINANCE_ENABLED or _check_yf_circuit():
        return None

    from src.exchanges.market_data import _get_isin_from_yfinance
    if _get_isin_from_yfinance(symbol) is None:
        logger.debug(f"Skipping yfinance analyst ratings for {symbol}: no valid Italian ISIN.")
        return None

    base = symbol.split("/")[0] if "/" in symbol else symbol
    base = base.lstrip('$')

    redis_client = get_redis_client()
    cache_key = f"yahoo_analyst_ratings:{base}"
    try:
        cached = redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
    except Exception:
        pass

    try:
        ticker = yf.Ticker(base, session=_get_yf_session())
        info = ticker.info
        
        target_mean_price = info.get("targetMeanPrice")
        target_high_price = info.get("targetHighPrice")
        target_low_price = info.get("targetLowPrice")
        current_price = info.get("currentPrice") or info.get("regularMarketPrice")
        
        recommendations = None
        try:
            recs = ticker.recommendations_summary
            if recs is not None and not recs.empty:
                latest = recs.iloc[0]
                recommendations = {
                    "strong_buy": int(latest.get("strongBuy", 0)),
                    "buy": int(latest.get("buy", 0)),
                    "hold": int(latest.get("hold", 0)),
                    "sell": int(latest.get("sell", 0)),
                    "strong_sell": int(latest.get("strongSell", 0)),
                }
        except Exception as e:
            logger.debug(f"Failed to fetch recommendations for {base}: {e}")

        if target_mean_price is None and recommendations is None:
            return None

        result = {
            "target_mean_price": target_mean_price,
            "target_high_price": target_high_price,
            "target_low_price": target_low_price,
            "current_price": current_price,
            "recommendations": recommendations,
        }

        try:
            redis_client.set(cache_key, json.dumps(result), ex=86400)  # 24h
        except Exception:
            pass
        return result
    except Exception as e:
        logger.warning(f"Yahoo Finance analyst ratings failed for {base}: {type(e).__name__}: {e}")
        return None
