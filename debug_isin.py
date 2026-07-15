#!/usr/bin/env python3
"""Debug script for ISIN retrieval and database storage.

Usage:
    python debug_isin.py                          # Interactive menu
    python debug_isin.py --symbol ENI.MI          # Debug a single symbol
    python debug_isin.py --db                     # Show all DB ISINs
    python debug_isin.py --db --symbol ENI.MI     # Show DB ISIN for one symbol
    python debug_isin.py --wikipedia              # Test Wikipedia scraper
    python debug_isin.py --borsa ENI              # Test Borsa Italiana search
    python debug_isin.py --yfinance ENI.MI        # Test yfinance ISIN fetch
    python debug_isin.py --full ENI.MI            # Full pipeline for one symbol
    python debug_isin.py --validate               # Validate all DB ISINs against yfinance
"""
import argparse
import json
import re
import sys
import os

# Ensure src is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config.settings import settings
from src.database import (
    get_isin_from_db,
    get_all_discovered_symbols,
    save_discovered_symbol,
    get_isin_map_from_db,
)
from src.utils.symbol_utils import is_btp_isin


def separator(title: str = ""):
    print("\n" + "=" * 70)
    if title:
        print(f"  {title}")
        print("=" * 70)


def test_yfinance_isin(symbol: str):
    """Test _get_isin_from_yfinance for a single symbol."""
    separator(f"yfinance ISIN fetch: {symbol}")
    from src.exchanges.market_data import _get_isin_from_yfinance

    # Strip suffix for DB lookup
    suffix = settings.TICKER_SUFFIX
    db_symbol = symbol
    if suffix and db_symbol.endswith(suffix):
        db_symbol = db_symbol[:-len(suffix)]

    # 1. Check DB first
    print(f"\n1. DB lookup for '{db_symbol}':")
    db_isin = get_isin_from_db(db_symbol)
    print(f"   DB ISIN: {db_isin}")

    # 2. Check yfinance circuit
    from src.exchanges.yf_session import _check_yf_circuit, _get_yf_session
    circuit_open = _check_yf_circuit()
    print(f"\n2. yfinance circuit breaker open: {circuit_open}")

    # 3. Try yfinance directly
    print(f"\n3. yfinance Ticker.isin:")
    yf_symbol = f"{db_symbol}{suffix}" if suffix and not db_symbol.endswith(suffix) else db_symbol
    print(f"   yf_symbol: {yf_symbol}")
    try:
        import yfinance as yf
        ticker = yf.Ticker(yf_symbol, session=_get_yf_session())
        yf_isin = ticker.isin
        print(f"   yfinance ISIN: {yf_isin}")
        if yf_isin:
            yf_isin = yf_isin.strip()
            if yf_isin == '-' or not yf_isin:
                print("   -> yfinance returned empty/placeholder ISIN")
                yf_isin = None
    except Exception as e:
        print(f"   yfinance error: {type(e).__name__}: {e}")
        yf_isin = None

    # 4. Try the full _get_isin_from_yfinance function
    print(f"\n4. Full _get_isin_from_yfinance('{symbol}'):")
    try:
        result = _get_isin_from_yfinance(symbol)
        print(f"   Result: {result}")
    except Exception as e:
        print(f"   Error: {type(e).__name__}: {e}")

    # 5. Show what would be saved
    if yf_isin:
        print(f"\n5. ISIN to be saved: {yf_isin}")
        print(f"   ISIN format valid: {bool(re.match(r'^[A-Z]{2}[A-Z0-9]{9}\d$', yf_isin))}")
        print(f"   Country prefix: {yf_isin[:2]}")

    return yf_isin


def test_borsa_italiana_search(base_symbol: str):
    """Test _get_isin_and_info_from_borsa_italiana for a single symbol."""
    separator(f"Borsa Italiana search: {base_symbol}")
    from src.exchanges.borsa_italiana_utils import _get_isin_and_info_from_borsa_italiana, _check_bi_circuit

    # Strip suffix
    suffix = settings.TICKER_SUFFIX
    db_symbol = base_symbol
    if suffix and db_symbol.endswith(suffix):
        db_symbol = db_symbol[:-len(suffix)]

    print(f"\nInput symbol: {base_symbol}")
    print(f"Stripped for search: {db_symbol}")
    print(f"BI circuit open: {_check_bi_circuit()}")

    try:
        isin, country, name = _get_isin_and_info_from_borsa_italiana(db_symbol)
        print(f"\nResult:")
        print(f"  ISIN:    {isin}")
        print(f"  Country: {country}")
        print(f"  Name:    {name}")

        if isin:
            print(f"\n  ISIN format valid: {bool(re.match(r'^[A-Z]{2}[A-Z0-9]{9}\d$', isin))}")
            # Check if name matches symbol
            if name:
                print(f"  Name matches symbol: {name.lower() == db_symbol.lower()}")
                print(f"  ⚠️  Name match is the filter used by the scraper — if name != ticker, ISIN is rejected!")

        return isin, country, name
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        return None, None, None


def test_fetch_info(symbol: str):
    """Test _fetch_info from asset_discovery (used during discovery)."""
    separator(f"_fetch_info (asset_discovery): {symbol}")
    from src.exchanges.asset_discovery import _fetch_info

    # Add suffix if missing
    suffix = settings.TICKER_SUFFIX
    if suffix and not symbol.endswith(suffix) and not is_btp_isin(symbol):
        full_symbol = f"{symbol}{suffix}"
    else:
        full_symbol = symbol

    print(f"\nInput: {symbol}")
    print(f"Full symbol (with suffix): {full_symbol}")

    try:
        country, name, isin = _fetch_info(full_symbol)
        print(f"\nResult:")
        print(f"  Country: {country}")
        print(f"  Name:    {name}")
        print(f"  ISIN:    {isin}")

        if isin:
            print(f"\n  ISIN format valid: {bool(re.match(r'^[A-Z]{2}[A-Z0-9]{9}\d$', isin))}")
            print(f"  Country prefix: {isin[:2]}")

        return country, name, isin
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        return None, None, None


def test_wikipedia_scraper():
    """Test the Wikipedia ticker scraper and check ISIN alignment."""
    separator("Wikipedia scraper test")
    from src.exchanges.asset_discovery import _discover_wikipedia_tickers

    urls = [
        "https://it.wikipedia.org/wiki/FTSE_MIB",
        "https://en.wikipedia.org/wiki/FTSE_MIB",
    ]

    for url in urls:
        print(f"\n--- Scraping: {url} ---")
        try:
            import requests
            import pandas as pd
            import warnings
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tables = pd.read_html(response.text)

            print(f"Found {len(tables)} tables")

            for idx, table in enumerate(tables):
                if isinstance(table.columns, pd.MultiIndex):
                    table.columns = [' '.join(col).strip() for col in table.columns.values]

                # Find ticker and ISIN columns
                ticker_col = None
                isin_col = None
                for col in table.columns:
                    col_str = str(col).lower()
                    if any(kw in col_str for kw in ("ticker", "symbol", "code", "simbolo", "codice", "yahoo", "borsa")):
                        ticker_col = col
                    if "isin" in col_str:
                        isin_col = col

                if ticker_col is not None:
                    print(f"\n  Table {idx}: ticker_col='{ticker_col}', isin_col='{isin_col}'")
                    print(f"  Columns: {list(table.columns)}")
                    # Show first 10 rows with ticker + ISIN
                    show_cols = [ticker_col]
                    if isin_col:
                        show_cols.append(isin_col)
                    # Also show name column if present
                    for col in table.columns:
                        if "name" in str(col).lower() or "nome" in str(col).lower() or "company" in str(col).lower():
                            show_cols.append(col)
                            break
                    print(table[show_cols].head(10).to_string(index=False))

                    # Check alignment: does the ISIN look correct for the ticker?
                    if isin_col:
                        print(f"\n  --- ISIN alignment check (first 10 rows) ---")
                        for _, row in table.head(10).iterrows():
                            ticker = str(row[ticker_col]).strip().upper()
                            isin_val = str(row[isin_col]).strip().upper() if pd.notna(row[isin_col]) else "N/A"
                            # Split ticker to get base
                            base = re.split(r'[\s(]', ticker)[0].split(".")[0] if "." in ticker else re.split(r'[\s(]', ticker)[0]
                            isin_valid = bool(re.match(r'^[A-Z]{2}[A-Z0-9]{9}\d$', isin_val)) if isin_val != "N/A" else False
                            print(f"    {base:12s} -> ISIN: {isin_val:15s} valid={isin_valid}")
                    else:
                        print(f"\n  (No ISIN column found in this table)")

        except Exception as e:
            print(f"Error scraping {url}: {type(e).__name__}: {e}")


def show_db_isins(symbol: str = None):
    """Show ISINs stored in the database."""
    separator("Database ISINs")

    if symbol:
        suffix = settings.TICKER_SUFFIX
        db_symbol = symbol
        if suffix and db_symbol.endswith(suffix):
            db_symbol = db_symbol[:-len(suffix)]

        print(f"\nLooking up: {db_symbol}")
        isin = get_isin_from_db(db_symbol)
        print(f"  ISIN: {isin}")

        # Also check with suffix
        if symbol != db_symbol:
            isin2 = get_isin_from_db(symbol)
            print(f"  ISIN (with suffix '{symbol}'): {isin2}")
    else:
        print("\nAll discovered symbols with ISINs:")
        symbols = get_all_discovered_symbols()
        with_isin = [s for s in symbols if s.get("isin")]
        without_isin = [s for s in symbols if not s.get("isin")]

        print(f"\nTotal symbols: {len(symbols)}")
        print(f"With ISIN: {len(with_isin)}")
        print(f"Without ISIN: {len(without_isin)}")

        print(f"\n--- Symbols WITH ISIN (first 50) ---")
        for s in with_isin[:50]:
            isin = s["isin"]
            valid = bool(re.match(r'^[A-Z]{2}[A-Z0-9]{9}\d$', isin)) if isin else False
            country_prefix = isin[:2] if isin else "??"
            flag = "✓" if valid else "✗"
            print(f"  {flag} {s['symbol']:20s} ISIN={isin:15s} country={country_prefix} type={s.get('asset_type', '?'):6s} name={s.get('name', '')[:30]}")

        if len(with_isin) > 50:
            print(f"  ... and {len(with_isin) - 50} more")

        # Show suspicious ISINs (non-IT prefix for Italian stocks)
        print(f"\n--- Suspicious ISINs (non-IT prefix for stocks) ---")
        suspicious = [s for s in with_isin if s.get("isin") and s["isin"][:2] != "IT" and s.get("asset_type") in ("stock", "etf")]
        if suspicious:
            for s in suspicious:
                print(f"  ⚠️  {s['symbol']:20s} ISIN={s['isin']:15s} country_prefix={s['isin'][:2]} type={s.get('asset_type')}")
            print(f"\n  Total suspicious: {len(suspicious)}")
        else:
            print("  None found.")

        # Show BTP ISINs
        print(f"\n--- BTP ISINs (first 10) ---")
        btps = [s for s in with_isin if s.get("asset_type") == "btp"]
        for s in btps[:10]:
            print(f"  {s['symbol']:20s} ISIN={s['isin']:15s} name={s.get('name', '')[:30]}")
        if len(btps) > 10:
            print(f"  ... and {len(btps) - 10} more")


def test_full_pipeline(symbol: str):
    """Run the full ISIN retrieval pipeline for a symbol."""
    separator(f"Full pipeline: {symbol}")

    suffix = settings.TICKER_SUFFIX
    db_symbol = symbol
    if suffix and db_symbol.endswith(suffix):
        db_symbol = db_symbol[:-len(suffix)]

    print(f"\nSymbol: {symbol}")
    print(f"DB symbol (stripped): {db_symbol}")

    # Step 1: DB
    print(f"\n--- Step 1: Database lookup ---")
    db_isin = get_isin_from_db(db_symbol)
    print(f"  DB ISIN: {db_isin}")

    # Step 2: yfinance
    print(f"\n--- Step 2: yfinance ---")
    yf_isin = test_yfinance_isin(symbol)

    # Step 3: Borsa Italiana
    print(f"\n--- Step 3: Borsa Italiana ---")
    bi_isin, bi_country, bi_name = test_borsa_italiana_search(db_symbol)

    # Step 4: _fetch_info (asset_discovery)
    print(f"\n--- Step 4: _fetch_info (asset_discovery) ---")
    fi_country, fi_name, fi_isin = test_fetch_info(db_symbol)

    # Summary
    separator("Summary")
    print(f"\n  DB ISIN:          {db_isin}")
    print(f"  yfinance ISIN:    {yf_isin}")
    print(f"  Borsa IT ISIN:    {bi_isin}  (name: {bi_name})")
    print(f"  _fetch_info ISIN: {fi_isin}  (name: {fi_name})")

    # Check for mismatches
    isins = [x for x in [db_isin, yf_isin, bi_isin, fi_isin] if x]
    if len(set(isins)) > 1:
        print(f"\n  ⚠️  MISMATCH DETECTED: different ISINs from different sources!")
        print(f"      Unique ISINs: {set(isins)}")
    elif len(isins) == 1:
        print(f"\n  ✓ All sources agree on ISIN: {isins[0]}")
    else:
        print(f"\n  No ISIN found from any source.")


def validate_all_db_isins():
    """Validate all DB ISINs by re-fetching from yfinance and comparing."""
    separator("Validate all DB ISINs")
    from src.exchanges.market_data import _get_isin_from_yfinance

    symbols = get_all_discovered_symbols()
    with_isin = [s for s in symbols if s.get("isin") and s.get("asset_type") in ("stock", "etf")]

    print(f"\nValidating {len(with_isin)} stock/ETF ISINs from DB against yfinance...")
    print("(This may take a while due to yfinance rate limits)\n")

    mismatches = []
    matches = 0
    errors = 0

    for i, s in enumerate(with_isin):
        db_sym = s["symbol"]
        db_isin = s["isin"]

        # Skip BTPs
        if is_btp_isin(db_sym):
            continue

        # Add suffix for yfinance
        suffix = settings.TICKER_SUFFIX
        yf_sym = f"{db_sym}{suffix}" if suffix and not db_sym.endswith(suffix) else db_sym

        try:
            # Temporarily bypass DB cache by checking yfinance directly
            from src.exchanges.yf_session import _check_yf_circuit, _get_yf_session
            if _check_yf_circuit():
                print(f"  [{i+1}/{len(with_isin)}] {db_sym:15s} DB={db_isin} -> yfinance circuit open, skipping")
                errors += 1
                continue

            import yfinance as yf
            ticker = yf.Ticker(yf_sym, session=_get_yf_session())
            yf_isin = ticker.isin
            if yf_isin:
                yf_isin = yf_isin.strip()
                if yf_isin == '-' or not yf_isin:
                    yf_isin = None

            if yf_isin is None:
                print(f"  [{i+1}/{len(with_isin)}] {db_sym:15s} DB={db_isin} -> yfinance returned None")
                errors += 1
            elif yf_isin == db_isin:
                print(f"  [{i+1}/{len(with_isin)}] {db_sym:15s} DB={db_isin} ✓ matches yfinance")
                matches += 1
            else:
                print(f"  [{i+1}/{len(with_isin)}] {db_sym:15s} DB={db_isin} ✗ yfinance={yf_isin}  MISMATCH!")
                mismatches.append({
                    "symbol": db_sym,
                    "db_isin": db_isin,
                    "yf_isin": yf_isin,
                })

            # Rate limit
            import time
            time.sleep(0.5)

        except Exception as e:
            print(f"  [{i+1}/{len(with_isin)}] {db_sym:15s} DB={db_isin} -> error: {type(e).__name__}: {e}")
            errors += 1

    separator("Validation Results")
    print(f"\n  Total checked:  {len(with_isin)}")
    print(f"  Matches:        {matches}")
    print(f"  Mismatches:     {len(mismatches)}")
    print(f"  Errors/None:    {errors}")

    if mismatches:
        print(f"\n--- Mismatched ISINs ---")
        for m in mismatches:
            print(f"  {m['symbol']:15s}  DB={m['db_isin']}  yfinance={m['yf_isin']}")

        # Offer to fix
        print(f"\nTo fix mismatches, you can run:")
        print(f"  python debug_isin.py --fix-mismatches")


def fix_mismatches():
    """Fix mismatched ISINs by re-fetching from yfinance and updating DB."""
    separator("Fix mismatched ISINs")
    from src.exchanges.market_data import _get_isin_from_yfinance
    from src.exchanges.yf_session import _check_yf_circuit, _get_yf_session
    import yfinance as yf
    import time

    symbols = get_all_discovered_symbols()
    with_isin = [s for s in symbols if s.get("isin") and s.get("asset_type") in ("stock", "etf")]

    suffix = settings.TICKER_SUFFIX
    fixed = 0

    for s in with_isin:
        db_sym = s["symbol"]
        db_isin = s["isin"]

        if is_btp_isin(db_sym):
            continue

        yf_sym = f"{db_sym}{suffix}" if suffix and not db_sym.endswith(suffix) else db_sym

        if _check_yf_circuit():
            print(f"  yfinance circuit open, stopping.")
            break

        try:
            ticker = yf.Ticker(yf_sym, session=_get_yf_session())
            yf_isin = ticker.isin
            if yf_isin:
                yf_isin = yf_isin.strip()
                if yf_isin == '-' or not yf_isin:
                    yf_isin = None

            if yf_isin and yf_isin != db_isin:
                print(f"  Fixing {db_sym}: DB={db_isin} -> yfinance={yf_isin}")
                save_discovered_symbol(db_sym, yf_isin, s.get("asset_type", "stock"), s.get("name"), country=s.get("country"))
                fixed += 1

            time.sleep(0.5)
        except Exception as e:
            print(f"  Error for {db_sym}: {type(e).__name__}: {e}")

    print(f"\nFixed {fixed} ISINs.")


def interactive_menu():
    """Show an interactive menu for debugging."""
    while True:
        separator("ISIN Debug Tool")
        print("\n  1. Test yfinance ISIN fetch for a symbol")
        print("  2. Test Borsa Italiana search for a symbol")
        print("  3. Test _fetch_info (asset_discovery) for a symbol")
        print("  4. Run full pipeline for a symbol")
        print("  5. Show all DB ISINs")
        print("  6. Show DB ISIN for a specific symbol")
        print("  7. Test Wikipedia scraper")
        print("  8. Validate all DB ISINs against yfinance")
        print("  9. Fix mismatched ISINs (update DB with yfinance ISINs)")
        print("  0. Exit")

        choice = input("\n  Choice: ").strip()

        if choice == "1":
            sym = input("  Symbol (e.g., ENI.MI or ENI): ").strip()
            if sym:
                test_yfinance_isin(sym)
        elif choice == "2":
            sym = input("  Base symbol (e.g., ENI): ").strip()
            if sym:
                test_borsa_italiana_search(sym)
        elif choice == "3":
            sym = input("  Symbol (e.g., ENI or ENI.MI): ").strip()
            if sym:
                test_fetch_info(sym)
        elif choice == "4":
            sym = input("  Symbol (e.g., ENI.MI): ").strip()
            if sym:
                test_full_pipeline(sym)
        elif choice == "5":
            show_db_isins()
        elif choice == "6":
            sym = input("  Symbol: ").strip()
            if sym:
                show_db_isins(sym)
        elif choice == "7":
            test_wikipedia_scraper()
        elif choice == "8":
            validate_all_db_isins()
        elif choice == "9":
            confirm = input("  This will overwrite DB ISINs with yfinance values. Continue? (y/n): ").strip().lower()
            if confirm == "y":
                fix_mismatches()
        elif choice == "0":
            print("  Bye!")
            break
        else:
            print("  Invalid choice.")

        input("\n  Press Enter to continue...")


def main():
    parser = argparse.ArgumentParser(description="Debug ISIN retrieval and storage")
    parser.add_argument("--symbol", "-s", type=str, help="Symbol to debug")
    parser.add_argument("--db", action="store_true", help="Show DB ISINs")
    parser.add_argument("--wikipedia", action="store_true", help="Test Wikipedia scraper")
    parser.add_argument("--borsa", type=str, help="Test Borsa Italiana search for a base symbol")
    parser.add_argument("--yfinance", type=str, help="Test yfinance ISIN fetch for a symbol")
    parser.add_argument("--fetch-info", type=str, help="Test _fetch_info for a symbol")
    parser.add_argument("--full", type=str, help="Run full pipeline for a symbol")
    parser.add_argument("--validate", action="store_true", help="Validate all DB ISINs against yfinance")
    parser.add_argument("--fix-mismatches", action="store_true", help="Fix mismatched ISINs in DB")
    args = parser.parse_args()

    if args.db:
        show_db_isins(args.symbol)
    elif args.wikipedia:
        test_wikipedia_scraper()
    elif args.borsa:
        test_borsa_italiana_search(args.borsa)
    elif args.yfinance:
        test_yfinance_isin(args.yfinance)
    elif args.fetch_info:
        test_fetch_info(args.fetch_info)
    elif args.full:
        test_full_pipeline(args.full)
    elif args.validate:
        validate_all_db_isins()
    elif args.fix_mismatches:
        fix_mismatches()
    elif args.symbol:
        test_full_pipeline(args.symbol)
    else:
        interactive_menu()


if __name__ == "__main__":
    main()
