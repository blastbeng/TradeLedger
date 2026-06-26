#!/usr/bin/env python3
"""Debug script for quote calculation from Borsa Italiana candles and database close prices.

Mirrors the quote logic in src/exchanges/market_data.py but with verbose output
and no Redis caching. Tests the full quote pipeline:
  1. Borsa Italiana 1d candle → extract last price + 24h change
  2. Database close prices (from OHLCV table)
  3. yfinance batch download (for comparison)

Usage:
  python debug_quotes.py                          # Test all tradable assets
  python debug_quotes.py ENI ENEL ISP             # Test specific symbols
  python debug_quotes.py --btp IT0005637399       # Test a BTP ISIN
  python debug_quotes.py --compare ENI            # Compare Borsa vs yfinance vs DB
"""

import sys
import json
import time
import logging
import argparse
import re
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress noisy libraries
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


def _get_borsa_italiana_quote(symbol: str) -> Optional[Dict[str, Any]]:
    """Fetch a quote from Borsa Italiana 1d candles (mirrors market_data.py logic)."""
    from src.exchanges.market_data import get_borsa_italiana_candles

    base = symbol.split("/")[0] if "/" in symbol else symbol

    # Skip BTPs — they are handled separately
    if re.match(r'^IT[A-Z0-9]{10}$', base):
        return None

    try:
        borsa_candles = get_borsa_italiana_candles(base, "1d", limit=2)
        if not borsa_candles or len(borsa_candles) == 0:
            logger.warning(f"[BORSA] No candles returned for {base}")
            return None

        last_candle = borsa_candles[-1]
        last_price = float(last_candle[4])
        volume = float(last_candle[5]) if last_candle[5] else None

        if last_price <= 0:
            logger.warning(f"[BORSA] Invalid last_price={last_price} for {base}")
            return None

        quote = {
            "last": last_price,
            "bid": last_price,
            "ask": last_price,
            "volume": volume,
            "change_24h": None,
            "percentage": None,
            "quoteVolume": volume,
            "source": "borsa_italiana",
        }

        if len(borsa_candles) > 1:
            prev_close = float(borsa_candles[-2][4])
            if prev_close > 0:
                change = ((last_price - prev_close) / prev_close) * 100
                quote["change_24h"] = round(change, 4)
                quote["percentage"] = round(change, 4)

        logger.info(f"[BORSA] {base}: last={last_price:.4f}, vol={volume}, "
                     f"change={quote['percentage']}%, candles={len(borsa_candles)}")
        return quote
    except Exception as e:
        logger.error(f"[BORSA] Failed for {base}: {e}")
        return None


def _get_db_close_price(symbol: str) -> Optional[Dict[str, Any]]:
    """Fetch the latest close price from the database OHLCV table."""
    from src.database import get_latest_close_prices

    base = symbol.split("/")[0] if "/" in symbol else symbol

    try:
        db_closes = get_latest_close_prices([base])
        if base in db_closes and db_closes[base] > 0:
            price = db_closes[base]
            logger.info(f"[DB] {base}: close_price={price:.4f}")
            return {
                "last": price,
                "bid": price,
                "ask": price,
                "volume": None,
                "change_24h": None,
                "percentage": None,
                "quoteVolume": None,
                "source": "db_close_price",
            }
        else:
            logger.warning(f"[DB] No close price found for {base}")
            return None
    except Exception as e:
        logger.error(f"[DB] Failed for {base}: {e}")
        return None


def _get_db_quotes(symbol: str) -> Optional[Dict[str, Any]]:
    """Fetch quotes from the database quotes table (up to 24h old)."""
    from src.database import get_quotes_from_db

    base = symbol.split("/")[0] if "/" in symbol else symbol

    try:
        db_quotes = get_quotes_from_db([base], max_age_seconds=86400)
        if base in db_quotes:
            q = db_quotes[base]
            logger.info(f"[DB_QUOTES] {base}: last={q.get('last')}, "
                         f"bid={q.get('bid')}, ask={q.get('ask')}, "
                         f"vol={q.get('volume')}, pct={q.get('percentage')}")
            q["source"] = "db_quotes"
            return q
        else:
            logger.warning(f"[DB_QUOTES] No quote found for {base}")
            return None
    except Exception as e:
        logger.error(f"[DB_QUOTES] Failed for {base}: {e}")
        return None


def _get_yfinance_quote(symbol: str) -> Optional[Dict[str, Any]]:
    """Fetch a quote from yfinance (for comparison)."""
    import yfinance as yf
    from src.config.settings import settings
    from src.exchanges.market_data import _get_yf_session, _check_yf_circuit

    base = symbol.split("/")[0] if "/" in symbol else symbol

    if _check_yf_circuit():
        logger.warning(f"[YF] Circuit breaker open, skipping {base}")
        return None

    suffix = settings.TICKER_SUFFIX
    yf_symbol = f"{base}{suffix}" if suffix and not base.endswith(suffix) else base

    try:
        ticker = yf.Ticker(yf_symbol, session=_get_yf_session())
        info = ticker.fast_info
        last_price = info.last_price if hasattr(info, 'last_price') else None
        prev_close = info.previous_close if hasattr(info, 'previous_close') else None

        if last_price is None or last_price <= 0:
            logger.warning(f"[YF] No valid last_price for {yf_symbol}")
            return None

        change = None
        pct = None
        if prev_close and prev_close > 0:
            change = ((last_price - prev_close) / prev_close) * 100
            pct = round(change, 4)

        quote = {
            "last": float(last_price),
            "bid": float(info.bid) if hasattr(info, 'bid') and info.bid else None,
            "ask": float(info.ask) if hasattr(info, 'ask') and info.ask else None,
            "volume": None,
            "change_24h": pct,
            "percentage": pct,
            "quoteVolume": None,
            "source": "yfinance",
        }
        logger.info(f"[YF] {yf_symbol}: last={last_price:.4f}, "
                     f"bid={quote['bid']}, ask={quote['ask']}, pct={pct}%")
        return quote
    except Exception as e:
        logger.error(f"[YF] Failed for {yf_symbol}: {e}")
        return None


def _get_btp_quote(isin: str) -> Optional[Dict[str, Any]]:
    """Fetch a BTP quote from the Borsa Italiana bond list."""
    from src.exchanges.market_data import discover_btp_bonds

    try:
        btp_bonds = discover_btp_bonds()
        for b in btp_bonds:
            if b["isin"] == isin:
                quote = {
                    "last": b["last_price"],
                    "bid": b["last_price"],
                    "ask": b["last_price"],
                    "volume": 0,
                    "change_24h": b["change_pct"],
                    "percentage": b["change_pct"],
                    "quoteVolume": 0,
                    "name": b.get("name"),
                    "coupon": b.get("coupon"),
                    "maturity": b.get("maturity"),
                    "source": "btp_bond_list",
                }
                logger.info(f"[BTP] {isin}: last={b['last_price']:.4f}, "
                             f"name={b.get('name')}, coupon={b.get('coupon')}, "
                             f"maturity={b.get('maturity')}")
                return quote
        logger.warning(f"[BTP] ISIN {isin} not found in BTP bond list")
        return None
    except Exception as e:
        logger.error(f"[BTP] Failed for {isin}: {e}")
        return None


def _print_quote_table(symbols: List[str], results: Dict[str, Dict[str, Any]]):
    """Print a formatted comparison table."""
    print("\n" + "=" * 120)
    print(f"{'Symbol':<15} {'Source':<18} {'Last':>12} {'Bid':>12} {'Ask':>12} {'Volume':>14} {'Change%':>10} {'QuoteVol':>14}")
    print("-" * 120)
    for sym in symbols:
        q = results.get(sym, {})
        source = q.get("source", "N/A")
        last = f"{q.get('last'):.4f}" if q.get("last") is not None else "N/A"
        bid = f"{q.get('bid'):.4f}" if q.get("bid") is not None else "N/A"
        ask = f"{q.get('ask'):.4f}" if q.get("ask") is not None else "N/A"
        vol = f"{q.get('volume'):,.0f}" if q.get("volume") is not None else "N/A"
        pct = f"{q.get('percentage'):.2f}%" if q.get("percentage") is not None else "N/A"
        qvol = f"{q.get('quoteVolume'):,.0f}" if q.get("quoteVolume") is not None else "N/A"
        print(f"{sym:<15} {source:<18} {last:>12} {bid:>12} {ask:>12} {vol:>14} {pct:>10} {qvol:>14}")
    print("=" * 120 + "\n")


def _print_comparison(symbol: str, borsa_q: Optional[dict], db_q: Optional[dict],
                      db_close: Optional[dict], yf_q: Optional[dict]):
    """Print a detailed comparison for a single symbol."""
    print("\n" + "=" * 80)
    print(f"COMPARISON FOR {symbol}")
    print("=" * 80)

    sources = [
        ("Borsa Italiana", borsa_q),
        ("DB Quotes Table", db_q),
        ("DB Close Prices", db_close),
        ("yfinance", yf_q),
    ]

    for name, q in sources:
        if q is None:
            print(f"\n  [{name}] — No data")
            continue
        print(f"\n  [{name}]")
        print(f"    last:       {q.get('last')}")
        print(f"    bid:        {q.get('bid')}")
        print(f"    ask:        {q.get('ask')}")
        print(f"    volume:     {q.get('volume')}")
        print(f"    change_24h: {q.get('change_24h')}")
        print(f"    percentage: {q.get('percentage')}")
        print(f"    quoteVol:   {q.get('quoteVolume')}")

    # Highlight discrepancies
    prices = {}
    for name, q in sources:
        if q and q.get("last") is not None and q["last"] > 0:
            prices[name] = q["last"]

    if len(prices) >= 2:
        print("\n  --- Price Discrepancies ---")
        names = list(prices.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                diff = abs(prices[names[i]] - prices[names[j]])
                pct_diff = (diff / min(prices[names[i]], prices[names[j]])) * 100
                print(f"    {names[i]} vs {names[j]}: "
                      f"diff={diff:.4f} ({pct_diff:.2f}%)")

    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Debug quote calculation from Borsa Italiana and database.")
    parser.add_argument("symbols", nargs="*", help="Base symbols to test (e.g., ENI ENEL ISP)")
    parser.add_argument("--btp", metavar="ISIN", help="Test a specific BTP ISIN")
    parser.add_argument("--compare", metavar="SYMBOL", help="Compare all sources for a single symbol")
    parser.add_argument("--all", action="store_true", help="Test all tradable assets")
    parser.add_argument("--limit", type=int, default=20, help="Limit number of symbols (default 20)")
    args = parser.parse_args()

    print("\n" + "#" * 80)
    print("# DEBUG QUOTES — Borsa Italiana + Database Close Prices")
    print("#" * 80)

    # --- Initialize database ---
    print("\nInitializing database...")
    from src.database import init_db
    init_db()
    print("Database initialized.")

    # --- Determine symbols to test ---
    if args.btp:
        print(f"\nTesting BTP ISIN: {args.btp}")
        quote = _get_btp_quote(args.btp)
        if quote:
            _print_quote_table([args.btp], {args.btp: quote})
        else:
            print(f"\n❌ No quote found for BTP {args.btp}")
        return

    if args.compare:
        symbol = args.compare
        print(f"\nComparing all quote sources for: {symbol}")
        print("-" * 60)

        borsa_q = _get_borsa_italiana_quote(symbol)
        db_q = _get_db_quotes(symbol)
        db_close = _get_db_close_price(symbol)
        yf_q = _get_yfinance_quote(symbol)

        _print_comparison(symbol, borsa_q, db_q, db_close, yf_q)
        return

    if args.all or not args.symbols:
        print("\nFetching all tradable assets...")
        from src.exchanges.market_data import get_tradable_assets
        all_assets = get_tradable_assets()
        # Strip suffix for display
        symbols = []
        for a in all_assets:
            base = a.split(".")[0] if "." in a else a
            if re.match(r'^IT[A-Z0-9]{10}$', base):
                symbols.append(base)  # BTP ISIN
            else:
                symbols.append(base)
        symbols = symbols[:args.limit]
        print(f"Testing {len(symbols)} symbols (limit={args.limit})")
    else:
        symbols = args.symbols

    if not symbols:
        print("No symbols to test.")
        return

    print(f"\nTesting {len(symbols)} symbols: {', '.join(symbols)}")
    print("-" * 80)

    results: Dict[str, Dict[str, Any]] = {}

    for i, sym in enumerate(symbols):
        print(f"\n[{i+1}/{len(symbols)}] Testing {sym}...")

        # Skip BTPs in the general loop (they need the BTP bond list)
        if re.match(r'^IT[A-Z0-9]{10}$', sym):
            print(f"  BTP ISIN detected, using BTP bond list...")
            q = _get_btp_quote(sym)
            if q:
                results[sym] = q
            continue

        # 1. Try Borsa Italiana candles
        print(f"  Step 1: Borsa Italiana 1d candle...")
        borsa_q = _get_borsa_italiana_quote(sym)

        # 2. Try DB close prices
        print(f"  Step 2: Database close prices...")
        db_close = _get_db_close_price(sym)

        # 3. Try DB quotes table
        print(f"  Step 3: Database quotes table...")
        db_q = _get_db_quotes(sym)

        # Priority: Borsa Italiana > DB quotes > DB close prices
        if borsa_q:
            results[sym] = borsa_q
        elif db_q:
            results[sym] = db_q
        elif db_close:
            results[sym] = db_close
        else:
            results[sym] = {
                "last": None, "bid": None, "ask": None, "volume": None,
                "change_24h": None, "percentage": None, "quoteVolume": None,
                "source": "none",
            }

    # --- Print summary table ---
    _print_quote_table(symbols, results)

    # --- Statistics ---
    total = len(symbols)
    borsa_count = sum(1 for q in results.values() if q.get("source") == "borsa_italiana")
    db_quotes_count = sum(1 for q in results.values() if q.get("source") == "db_quotes")
    db_close_count = sum(1 for q in results.values() if q.get("source") == "db_close_price")
    btp_count = sum(1 for q in results.values() if q.get("source") == "btp_bond_list")
    none_count = sum(1 for q in results.values() if q.get("source") == "none")

    print("\n" + "=" * 60)
    print("STATISTICS")
    print("=" * 60)
    print(f"  Total symbols tested:      {total}")
    print(f"  Borsa Italiana quotes:     {borsa_count}")
    print(f"  DB quotes table:           {db_quotes_count}")
    print(f"  DB close prices:           {db_close_count}")
    print(f"  BTP bond list:             {btp_count}")
    print(f"  No quote found:            {none_count}")
    print(f"  Success rate:              {(total - none_count) / total * 100:.1f}%" if total > 0 else "N/A")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
