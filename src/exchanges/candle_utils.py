import logging
from datetime import datetime, timezone
from typing import List, Optional

logger = logging.getLogger(__name__)


def _validate_and_clean_candles(candles: List[List], symbol: Optional[str] = None) -> List[List]:
    """Validate and clean OHLCV candles to ensure data quality.

    Removes candles with:
    - Non-positive prices (open, high, low, close)
    - Negative volume
    - Invalid high/low relationships (high < max(open, close, low) or low > min(open, close, high))
    - Duplicate timestamps (keeps the last occurrence)
    """
    if not candles:
        return []

    seen_timestamps = {}

    for c in candles:
        if len(c) < 6:
            continue

        ts, o, h, l, cl, v = c[0], c[1], c[2], c[3], c[4], c[5]

        # Check for non-positive prices
        if o <= 0 or h <= 0 or l <= 0 or cl <= 0:
            continue

        # Check for negative volume
        if v < 0:
            continue

        # Check high/low relationships
        if h < max(o, cl, l) or l > min(o, cl, h):
            continue

        # Track timestamps to remove duplicates (keep last)
        seen_timestamps[ts] = c

    # Sort by timestamp to maintain chronological order
    cleaned = sorted(seen_timestamps.values(), key=lambda x: x[0])

    total_removed = len(candles) - len(cleaned)
    if total_removed > 0:
        sym_str = f" for {symbol}" if symbol else ""
        logger.warning(
            f"Removed {total_removed}/{len(candles)} invalid or duplicate candles{sym_str}."
        )

    return cleaned


def _aggregate_candles(candles: List[List], target_tf: str) -> List[List]:
    """Aggregate monthly candles into larger timeframes (6M, 1Y, 3Y, 5Y)."""
    if not candles or target_tf not in ("6M", "1Y", "3Y", "5Y"):
        return candles

    grouped = {}
    for c in candles:
        ts = c[0]
        dt = datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc)
        year = dt.year
        month = dt.month

        if target_tf == "6M":
            period_key = (year, (month - 1) // 6)
        elif target_tf == "1Y":
            period_key = year
        elif target_tf == "3Y":
            period_key = year // 3
        elif target_tf == "5Y":
            period_key = year // 5

        if period_key not in grouped:
            grouped[period_key] = {
                'timestamp': ts,
                'open': c[1],
                'high': c[2],
                'low': c[3],
                'close': c[4],
                'volume': c[5]
            }
        else:
            grouped[period_key]['high'] = max(grouped[period_key]['high'], c[2])
            grouped[period_key]['low'] = min(grouped[period_key]['low'], c[3])
            grouped[period_key]['close'] = c[4]
            grouped[period_key]['volume'] += c[5]

    result = []
    for key in sorted(grouped.keys()):
        g = grouped[key]
        result.append([
            g['timestamp'],
            g['open'],
            g['high'],
            g['low'],
            g['close'],
            g['volume']
        ])
    return result


def _merge_candles(borsa_candles: Optional[List[List]], yf_candles: Optional[List[List]]) -> List[List]:
    """Merge two candle lists, deduplicating by timestamp (borsaitaliana takes precedence)."""
    if not borsa_candles and not yf_candles:
        return []
    if not borsa_candles:
        return yf_candles
    if not yf_candles:
        return borsa_candles
    merged = {}
    for c in yf_candles:
        merged[c[0]] = c
    for c in borsa_candles:  # borsaitaliana overrides yfinance for same timestamp
        merged[c[0]] = c
    return sorted(merged.values(), key=lambda c: c[0])


def detect_data_quality_issues(candles: List[List], symbol: str) -> Optional[str]:
    """Detects data quality issues like sudden price jumps, gaps, and zero volume."""
    if not candles or len(candles) < 2:
        return None

    issues = []
    # Assuming candle format: [timestamp, open, high, low, close, volume]
    for i in range(1, len(candles)):
        prev_close = candles[i-1][4]
        curr_open = candles[i][1]
        curr_close = candles[i][4]
        curr_volume = candles[i][5]

        if prev_close > 0:
            # 1. Sudden price jump (e.g., > 20%)
            change_pct = abs(curr_close - prev_close) / prev_close * 100
            if change_pct > 20.0:
                issues.append(f"Large price jump of {change_pct:.2f}% at timestamp {candles[i][0]}")

            # 2. Gap detection (e.g., > 10%)
            gap_pct = abs(curr_open - prev_close) / prev_close * 100
            if gap_pct > 10.0:
                issues.append(f"Price gap of {gap_pct:.2f}% at timestamp {candles[i][0]}")

        # 3. Volume anomaly (zero volume)
        if curr_volume == 0:
            issues.append(f"Zero volume at timestamp {candles[i][0]}")

    if issues:
        return f"⚠️ Data quality issues detected for {symbol}:\n" + "\n".join(issues)
    return None
