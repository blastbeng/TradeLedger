import sys
import logging

# Configure logging to show debug output
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Ensure the src directory is in the path
sys.path.insert(0, 'src')

from exchanges.market_data import _get_isin_from_yfinance, get_borsa_italiana_candles

def main():
    # List of sample symbols to test (stocks and ETFs)
    test_symbols = [
        "AAPL",   # Apple Inc.
        "MSFT",   # Microsoft Corp.
        "SPY",    # SPDR S&P 500 ETF Trust
        "QQQ",    # Invesco QQQ Trust
        "ENI.MI", # Eni S.p.A. (Italian stock)
    ]

    print("\n--- Testing _get_isin_from_yfinance ---")
    for symbol in test_symbols:
        print(f"\nFetching ISIN for: {symbol}")
        try:
            isin = _get_isin_from_yfinance(symbol)
            if isin:
                print(f"Result for {symbol}: {isin}")
            else:
                print(f"Result for {symbol}: None (ISIN not found)")
        except Exception as e:
            print(f"Error fetching ISIN for {symbol}: {e}")

    print("\n--- Testing get_borsa_italiana_candles ---")
    # Test a few symbols and timeframes
    candle_test_cases = [
        ("ENI.MI", "1d", 5),
        ("ENI.MI", "1M", 5),
        ("ISP.MI", "1d", 5),
        ("AAPL", "1d", 5), # US stock, likely to fail or fallback
    ]
    for symbol, timeframe, limit in candle_test_cases:
        print(f"\nFetching {timeframe} candles for: {symbol} (limit: {limit})")
        try:
            candles = get_borsa_italiana_candles(symbol, timeframe, limit=limit)
            if candles:
                print(f"Result for {symbol} {timeframe}: {len(candles)} candles fetched.")
                # Print the first and last candle for a quick sanity check
                print(f"  First candle: {candles[0]}")
                print(f"  Last candle:  {candles[-1]}")
            else:
                print(f"Result for {symbol} {timeframe}: None (candles not found)")
        except Exception as e:
            print(f"Error fetching candles for {symbol} {timeframe}: {e}")

if __name__ == "__main__":
    main()
