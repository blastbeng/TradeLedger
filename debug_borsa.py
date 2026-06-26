import httpx
import json
import re
import logging
import pandas as pd
from datetime import datetime, timezone
from typing import Optional, List

# Configure basic logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Borsa Italiana timeframe conversion map (daily data → resampled via pandas)
BORSA_TIMEFRAME_MAP = {
    "1d": None,       # Daily native (no conversion needed)
    "1w": "W",         # Weekly
    "1M": "ME",        # Month End
    "3M": "3ME",       # Quarterly
    "6M": "6ME",       # Semi-annual
    "1Y": "YE",        # Year End
    "3Y": "3YE",       # 3-Year
    "5Y": "5YE",       # 5-Year
}

def get_borsa_italiana_candles_debug(
    symbol: str,
    timeframe: str,
    limit: int = 500,
    start_ms: int = None,
) -> Optional[List[List]]:
    """Debug version of get_borsa_italiana_candles. Omits Redis and DB, prints summary."""
    
    if timeframe not in BORSA_TIMEFRAME_MAP:
        logger.error(f"Timeframe {timeframe} not supported. Supported: {list(BORSA_TIMEFRAME_MAP.keys())}")
        return None

    base = symbol.split("/")[0] if "/" in symbol else symbol

    # For BTPs, the symbol IS the ISIN. For stocks, we skip yfinance ISIN lookup in this debug script.
    if re.match(r'^IT[A-Z0-9]{10}$', base):
        isin = base
    else:
        logger.error(f"Symbol {base} is not a valid BTP ISIN. Please provide a valid ISIN (e.g., IT0001234567) for debugging.")
        return None

    api_period = "5Y"
    url = f"https://grafici.borsaitaliana.it/api/instruments/{isin},XMIL,ISIN/history/period?period={api_period}&adjustment=true&add-last-price=true"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "it-IT,it;q=0.9,en;q=0.8",
    }

    logger.info(f"Fetching data from URL: {url}")

    try:
        # Omitting proxy logic for standalone debug
        with httpx.Client(timeout=15.0, follow_redirects=True) as client:
            response = client.get(url, headers=headers)
        response.raise_for_status()

        text = response.text
        try:
            data = response.json()
        except (json.JSONDecodeError, ValueError):
            logger.debug("Response is not pure JSON, trying to extract from <pre> tags...")
            match = re.search(r'<pre>(.*?)</pre>', text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(1))
                except json.JSONDecodeError:
                    logger.error(f"Could not parse JSON from borsaitaliana response for {symbol}")
                    return None
            else:
                logger.error(f"No JSON data found in borsaitaliana response for {symbol}")
                return None

        history = data.get("history", {})
        history_dt = history.get("historyDt", [])

        if not history_dt:
            logger.warning(f"Empty history from borsaitaliana for {symbol} {timeframe}")
            return None

        logger.info(f"Received {len(history_dt)} raw daily candles from API.")

        rows = []
        for item in history_dt:
            dt_str = item.get("dt", "")
            if not dt_str or len(dt_str) != 8:
                continue
            try:
                dt = datetime.strptime(dt_str, "%Y%m%d")
                ts = int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)
                rows.append([
                    ts,
                    float(item["openPx"]),
                    float(item["highPx"]),
                    float(item["lowPx"]),
                    float(item["closePx"]),
                    float(item.get("qty", 0) or 0),
                ])
            except (ValueError, KeyError) as e:
                logger.error(f"Failed to parse borsaitaliana candle for {symbol}: {e}")
                continue

        if not rows:
            logger.error("No valid rows parsed from API response.")
            return None

        rows.sort(key=lambda c: c[0])

        if start_ms is not None:
            rows = [c for c in rows if c[0] >= start_ms]

        pandas_freq = BORSA_TIMEFRAME_MAP.get(timeframe)
        if pandas_freq is not None:
            logger.info(f"Resampling daily candles to {timeframe} using pandas freq '{pandas_freq}'...")
            df = pd.DataFrame(rows, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms')
            df.set_index('Date', inplace=True)
            ohlcv_rules = {
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }
            df = df.resample(pandas_freq).agg(ohlcv_rules)
            df.dropna(subset=['Open'], inplace=True)
            df.reset_index(inplace=True)
            rows = []
            for _, row in df.iterrows():
                ts = int(row['Date'].timestamp() * 1000)
                vol = float(row['Volume']) if pd.notna(row['Volume']) else 0.0
                rows.append([ts, float(row['Open']), float(row['High']), float(row['Low']), float(row['Close']), vol])

        rows.sort(key=lambda c: c[0])

        if limit and len(rows) > limit:
            rows = rows[-limit:]

        if rows:
            logger.info(f"Successfully processed {len(rows)} candles for {symbol} {timeframe}")
            print("\n" + "="*50)
            print(f"SUMMARY FOR {symbol} ({timeframe})")
            print("="*50)
            print(f"Total candles: {len(rows)}")
            print("\nFirst 5 candles:")
            for r in rows[:5]:
                dt = datetime.fromtimestamp(r[0] / 1000.0, tz=timezone.utc)
                print(f"  {dt.strftime('%Y-%m-%d')} | O: {r[1]:.4f} H: {r[2]:.4f} L: {r[3]:.4f} C: {r[4]:.4f} V: {r[5]:.2f}")
            
            print("\nLast 5 candles:")
            for r in rows[-5:]:
                dt = datetime.fromtimestamp(r[0] / 1000.0, tz=timezone.utc)
                print(f"  {dt.strftime('%Y-%m-%d')} | O: {r[1]:.4f} H: {r[2]:.4f} L: {r[3]:.4f} C: {r[4]:.4f} V: {r[5]:.2f}")
            print("="*50 + "\n")
            return rows

        return None

    except Exception as e:
        logger.error(f"Borsaitaliana candle download failed for {symbol} {timeframe}: {e}", exc_info=True)
        return None

if __name__ == "__main__":
    # Replace with a valid BTP ISIN you want to test
    TEST_ISIN = "IT0001234567" 
    TEST_TIMEFRAME = "1d" # Try "1d", "1w", "1M", etc.
    
    print(f"Running debug script for {TEST_ISIN} on {TEST_TIMEFRAME} timeframe...")
    get_borsa_italiana_candles_debug(TEST_ISIN, TEST_TIMEFRAME, limit=10)
