import re
from typing import Any, Dict, List, Optional

from src.config.settings import settings


def timeframe_to_ms(timeframe: str) -> int:
    """Convert a timeframe string (e.g., '1m', '5m', '1h') to milliseconds."""
    units = {
        'm': 60_000,
        'h': 3_600_000,
        'd': 86_400_000,
        'w': 604_800_000,
        'M': 2_592_000_000,  # approximate (30 days)
        'Y': 31_536_000_000, # approximate (365 days)
    }
    match = re.match(r'^(\d+)([mhdwMY])$', timeframe)
    if not match:
        return 3_600_000  # default to 1h
    amount = int(match.group(1))
    unit = match.group(2)
    return amount * units.get(unit, 3_600_000)


def timeframe_to_seconds(timeframe: str) -> int:
    """Convert a timeframe string (e.g., '5m', '1h') to seconds."""
    return timeframe_to_ms(timeframe) // 1000


def format_symbol_display(symbol: str, stock_name: str, timeframe: Optional[str] = None) -> str:
    """Return a display string like 'AAPL[Apple Inc.]' or 'AAPL[Apple Inc.] (15m)'."""
    base = symbol.split("/")[0] if "/" in symbol else symbol
    if stock_name and stock_name != base:
        display = f"{base}[{stock_name}]"
    else:
        display = base
    if timeframe:
        display += f" ({timeframe})"
    return display


def is_excluded(symbol: str, timeframe: str) -> bool:
    """Return True if (symbol, timeframe) is in the EXCLUDED_SYMBOLS list."""
    for entry in settings.EXCLUDED_SYMBOLS:
        parts = entry.split("/")
        if len(parts) == 2:
            # "BASE/QUOTE" → exclude all timeframes for this pair
            if parts[0] == symbol.split("/")[0] and parts[1] == symbol.split("/")[1]:
                return True
        elif len(parts) == 3:
            # "BASE/QUOTE/TIMEFRAME" → exclude only that specific timeframe
            if (parts[0] == symbol.split("/")[0] and
                parts[1] == symbol.split("/")[1] and
                parts[2] == timeframe):
                return True
    return False


def normalize_llm_symbol(sym: str, sample_pairs: list, base_currency: str) -> Optional[str]:
    """Normalize an LLM-returned symbol to match the format in sample_pairs.

    The LLM may return symbols without the /EUR suffix (e.g., 'ENI.MI' instead
    of 'ENI.MI/EUR'), or with/without exchange-specific suffixes (e.g., 'ENI'
    vs 'ENI.MI'). This method tries multiple formats to find a match.
    Returns the matched pair string, or None if no match is found.
    """
    if sym in sample_pairs:
        return sym
    # Try adding /{base_currency} suffix
    with_suffix = f"{sym}/{base_currency}"
    if with_suffix in sample_pairs:
        return with_suffix
    # Try matching by base symbol (strip any suffix the LLM may have added)
    base = sym.split("/")[0]
    for pair in sample_pairs:
        if pair.split("/")[0] == base:
            return pair
    # Try matching by stripping exchange suffixes from both sides
    # e.g., LLM returns 'ENI' but sample has 'ENI.MI', or vice versa
    configured_suffix = getattr(settings, 'TICKER_SUFFIX', '')

    def _strip_suffix(symbol_base: str) -> str:
        # Strip the configured ticker suffix first
        if configured_suffix and symbol_base.endswith(configured_suffix):
            return symbol_base[:-len(configured_suffix)]
        # Strip common exchange suffixes (e.g., .MI, .PA, .L, .N, .SW)
        parts = symbol_base.rsplit('.', 1)
        if len(parts) == 2 and 1 <= len(parts[1]) <= 3 and parts[1].isalpha() and parts[1].isupper():
            return parts[0]
        return symbol_base

    stripped_base = _strip_suffix(base)
    for pair in sample_pairs:
        pair_base = pair.split("/")[0]
        if stripped_base == _strip_suffix(pair_base):
            return pair
    return None


def get_effective_refresh_interval(base_interval: int, current_symbols: list, loop_type: str = "data") -> int:
    """Scale refresh interval based on the longest tracked timeframe.

    For long-term timeframes (1Y+), use much longer refresh cycles to
    avoid wasting bandwidth and API calls on data that barely changes.

    loop_type: "quotes" for quote refresh, "data" for OHLCV downloads,
               "news" for news downloads.
    """
    if not current_symbols:
        return base_interval

    max_tf_seconds = 0
    for entry in current_symbols:
        tf = entry.get("timeframe", "1d")
        tf_secs = timeframe_to_seconds(tf)
        if tf_secs > max_tf_seconds:
            max_tf_seconds = tf_secs

    if loop_type == "quotes":
        # Quotes: even for long timeframes, prices still move intraday
        if max_tf_seconds >= 31_536_000:  # 1Y+
            return max(base_interval, 3600)  # 1 hour
        elif max_tf_seconds >= 2_592_000:  # 1M+
            return max(base_interval, 1800)  # 30 minutes
        return base_interval
    elif loop_type == "news":
        # News: daily is sufficient for long-term trading
        if max_tf_seconds >= 31_536_000:  # 1Y+
            return max(base_interval, 86400)  # daily
        elif max_tf_seconds >= 2_592_000:  # 1M+
            return max(base_interval, 43200)  # 12 hours
        return base_interval
    else:  # "data" – OHLCV downloads
        if max_tf_seconds >= 31_536_000:  # 1Y+
            return max(base_interval, 86400)  # daily
        elif max_tf_seconds >= 15_552_000:  # 6M+
            return max(base_interval, 43200)  # 12 hours
        elif max_tf_seconds >= 7_776_000:  # 3M+
            return max(base_interval, 21600)  # 6 hours
        elif max_tf_seconds >= 2_592_000:  # 1M+
            return max(base_interval, 10800)  # 3 hours
        elif max_tf_seconds >= 604_800:  # 1w+
            return max(base_interval, 3600)  # 1 hour
        return base_interval
