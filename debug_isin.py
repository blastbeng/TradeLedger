import sys
import logging

# Configure logging to show debug output
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Ensure the src directory is in the path
sys.path.insert(0, 'src')

from exchanges.market_data import _get_isin_from_yfinance

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

if __name__ == "__main__":
    main()
