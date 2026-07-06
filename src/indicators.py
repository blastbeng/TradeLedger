from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import talib


def _compute_simple_atr(candles: List[List], period: int = 14) -> Optional[float]:
    """Compute a simple average True Range when there aren't enough candles for Wilder's ATR.

    Uses the same True Range formula but averages over available candles instead of
    applying Wilder's smoothing. This provides a reasonable ATR estimate for long
    timeframes (5Y, 3Y, etc.) where only a few candles are available.
    """
    if not candles:
        return None
    if len(candles) == 1:
        tr = candles[0][2] - candles[0][3]
        return tr if tr > 0 else None

    true_ranges = []
    for i in range(len(candles)):
        high = candles[i][2]
        low = candles[i][3]
        if i == 0:
            tr = high - low
        else:
            prev_close = candles[i - 1][4]
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        true_ranges.append(tr)

    use_period = min(period, len(true_ranges))
    recent_trs = true_ranges[-use_period:]
    avg_tr = sum(recent_trs) / len(recent_trs)
    return avg_tr if avg_tr > 0 else None


def _compute_simple_atr_series(candles: List[List], period: int = 14) -> List[Optional[float]]:
    """Compute simple average True Range series when there aren't enough candles for Wilder's ATR.

    Returns a full-length list (one value per candle) with no None warmup,
    since the simple average can be computed from the first candle onward.
    """
    if not candles:
        return []

    true_ranges = []
    for i in range(len(candles)):
        high = candles[i][2]
        low = candles[i][3]
        if i == 0:
            tr = high - low
        else:
            prev_close = candles[i - 1][4]
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        true_ranges.append(tr)

    result = []
    for i in range(len(candles)):
        start = max(0, i - period + 1)
        window = true_ranges[start:i + 1]
        avg = sum(window) / len(window)
        result.append(avg if avg > 0 else None)

    return result


def _compute_simple_rsi(closes: List[float], period: int = 14) -> Optional[float]:
    if len(closes) < 2:
        return None
    use_period = min(period, len(closes) - 1)
    gains = []
    losses = []
    for i in range(1, len(closes)):
        change = closes[i] - closes[i-1]
        gains.append(max(0, change))
        losses.append(max(0, -change))
    
    recent_gains = gains[-use_period:]
    recent_losses = losses[-use_period:]
    
    avg_gain = sum(recent_gains) / use_period if use_period > 0 else 0
    avg_loss = sum(recent_losses) / use_period if use_period > 0 else 0
    
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

def _compute_simple_macd(closes: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if len(closes) < 2:
        return None, None, None
    fast_period = min(fast, len(closes))
    slow_period = min(slow, len(closes))
    
    fast_sma = sum(closes[-fast_period:]) / fast_period
    slow_sma = sum(closes[-slow_period:]) / slow_period
    macd_val = fast_sma - slow_sma
    
    return macd_val, macd_val, 0.0

def _compute_simple_bollinger_bands(closes: List[float], period: int = 20, std_dev: float = 2.0) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if not closes:
        return None, None, None
    use_period = min(period, len(closes))
    recent = closes[-use_period:]
    middle = sum(recent) / use_period
    variance = sum((x - middle) ** 2 for x in recent) / use_period
    std = variance ** 0.5
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    return upper, middle, lower

def _compute_simple_stochastic(highs: List[float], lows: List[float], closes: List[float], period: int = 14, smooth_k: int = 3) -> Tuple[Optional[float], Optional[float]]:
    if not closes:
        return None, None
    use_period = min(period, len(closes))
    recent_highs = highs[-use_period:]
    recent_lows = lows[-use_period:]
    highest = max(recent_highs)
    lowest = min(recent_lows)
    current_close = closes[-1]
    if highest == lowest:
        k = 50.0
    else:
        k = ((current_close - lowest) / (highest - lowest)) * 100.0
    return k, k

def _compute_simple_adx(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if len(closes) < 2:
        return None, None, None
    start_price = closes[0]
    end_price = closes[-1]
    price_change = abs(end_price - start_price) / start_price if start_price != 0 else 0
    adx_val = min(100.0, price_change * 100)
    plus_val = 100.0 if end_price > start_price else 0.0
    minus_val = 100.0 if end_price < start_price else 0.0
    return adx_val, plus_val, minus_val

def _compute_simple_ichimoku(highs: List[float], lows: List[float], closes: List[float], tenkan_period: int = 9, kijun_period: int = 26, senkou_b_period: int = 52) -> Optional[Dict[str, Optional[float]]]:
    if not closes:
        return None
    tenkan_p = min(tenkan_period, len(closes))
    kijun_p = min(kijun_period, len(closes))
    senkou_b_p = min(senkou_b_period, len(closes))
    
    tenkan_high = max(highs[-tenkan_p:])
    tenkan_low = min(lows[-tenkan_p:])
    tenkan_sen = (tenkan_high + tenkan_low) / 2
    
    kijun_high = max(highs[-kijun_p:])
    kijun_low = min(lows[-kijun_p:])
    kijun_sen = (kijun_high + kijun_low) / 2
    
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    senkou_b_high = max(highs[-senkou_b_p:])
    senkou_b_low = min(lows[-senkou_b_p:])
    senkou_span_b = (senkou_b_high + senkou_b_low) / 2
    
    chikou_span = closes[-1]
    cloud_top = max(senkou_span_a, senkou_span_b)
    cloud_bottom = min(senkou_span_a, senkou_span_b)
    
    return {
        "tenkan_sen": round(tenkan_sen, 8),
        "kijun_sen": round(kijun_sen, 8),
        "senkou_span_a": round(senkou_span_a, 8),
        "senkou_span_b": round(senkou_span_b, 8),
        "chikou_span": round(chikou_span, 8),
        "cloud_top": round(cloud_top, 8),
        "cloud_bottom": round(cloud_bottom, 8),
    }

def _compute_simple_donchian_channels(highs: List[float], lows: List[float], period: int = 20) -> Optional[Dict[str, float]]:
    if not highs or not lows:
        return None
    use_period = min(period, len(highs))
    upper = max(highs[-use_period:])
    lower = min(lows[-use_period:])
    middle = (upper + lower) / 2.0
    return {
        "upper": round(upper, 8),
        "middle": round(middle, 8),
        "lower": round(lower, 8),
    }

def _compute_simple_mfi(highs: List[float], lows: List[float], closes: List[float], volumes: List[float], period: int = 14) -> Optional[float]:
    if len(closes) < 2:
        return None
    tps = [(h + l + c) / 3.0 for h, l, c in zip(highs, lows, closes)]
    pos_flow = 0.0
    neg_flow = 0.0
    for i in range(1, len(tps)):
        if tps[i] > tps[i-1]:
            pos_flow += tps[i] * volumes[i]
        elif tps[i] < tps[i-1]:
            neg_flow += tps[i] * volumes[i]
    
    if neg_flow == 0:
        return 100.0 if pos_flow > 0 else 50.0
    mfr = pos_flow / neg_flow
    return 100.0 - (100.0 / (1.0 + mfr))

def _compute_simple_cci(highs: List[float], lows: List[float], closes: List[float], period: int = 20) -> Optional[float]:
    if not closes:
        return None
    use_period = min(period, len(closes))
    tp = [(h + l + c) / 3.0 for h, l, c in zip(highs[-use_period:], lows[-use_period:], closes[-use_period:])]
    avg_tp = sum(tp) / use_period
    mean_dev = sum(abs(x - avg_tp) for x in tp) / use_period
    if mean_dev == 0:
        return 0.0
    return (tp[-1] - avg_tp) / (0.015 * mean_dev)

def _compute_simple_williams_r(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Optional[float]:
    if not closes:
        return None
    use_period = min(period, len(closes))
    highest = max(highs[-use_period:])
    lowest = min(lows[-use_period:])
    if highest == lowest:
        return -50.0
    return ((highest - closes[-1]) / (highest - lowest)) * -100.0


def compute_atr(candles: List[List], period: int = 14) -> Optional[float]:
    """Compute Average True Range from OHLCV candles using Wilder's smoothing."""
    if len(candles) < period + 1:
        return None
    highs = np.array([c[2] for c in candles], dtype=float)
    lows = np.array([c[3] for c in candles], dtype=float)
    closes = np.array([c[4] for c in candles], dtype=float)
    result = talib.ATR(highs, lows, closes, timeperiod=period)
    val = result[-1]
    return val if not np.isnan(val) else None


def compute_rsi(closes: List[float], period: int = 14) -> Optional[float]:
    """Compute RSI from closing prices."""
    if len(closes) < period + 1:
        return _compute_simple_rsi(closes, period)
    result = talib.RSI(np.array(closes, dtype=float), timeperiod=period)
    val = result[-1]
    return val if not np.isnan(val) else None


def compute_ema(data: List[float], period: int) -> List[Optional[float]]:
    """Compute Exponential Moving Average. Returns full-length list with None warmup."""
    if not data:
        return []
    if len(data) < period:
        sma = sum(data) / len(data)
        return [sma] * len(data)
    result = talib.EMA(np.array(data, dtype=float), timeperiod=period)
    return [None if np.isnan(v) else float(v) for v in result]


def compute_atr_series(candles: List[List], period: int = 14) -> List[Optional[float]]:
    """Compute ATR series (full-length list with None warmup).

    Falls back to simple average True Range when there aren't enough candles
    for Wilder's smoothing (e.g., long timeframes like 5Y with only a few candles).
    """
    if len(candles) < period + 1:
        return _compute_simple_atr_series(candles, period)
    highs = np.array([c[2] for c in candles], dtype=float)
    lows = np.array([c[3] for c in candles], dtype=float)
    closes = np.array([c[4] for c in candles], dtype=float)
    result = talib.ATR(highs, lows, closes, timeperiod=period)
    return [None if np.isnan(v) else float(v) for v in result]


def compute_adx_series(candles: List[List], period: int = 14) -> List[Optional[float]]:
    """Compute ADX series (full-length list with None warmup)."""
    if len(candles) < period + 1:
        return []
    highs = np.array([c[2] for c in candles], dtype=float)
    lows = np.array([c[3] for c in candles], dtype=float)
    closes = np.array([c[4] for c in candles], dtype=float)
    result = talib.ADX(highs, lows, closes, timeperiod=period)
    return [None if np.isnan(v) else float(v) for v in result]


def compute_rsi_series(candles: List[List], period: int = 14) -> List[Optional[float]]:
    """Compute RSI series (full-length list with None warmup)."""
    if len(candles) < period + 1:
        return []
    closes = np.array([c[4] for c in candles], dtype=float)
    result = talib.RSI(closes, timeperiod=period)
    return [None if np.isnan(v) else float(v) for v in result]


def compute_macd_series(candles: List[List], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
    """Compute MACD line, signal line, and histogram series (full-length lists with None warmup)."""
    if len(candles) < slow + signal:
        return [], [], []
    closes = np.array([c[4] for c in candles], dtype=float)
    macd, macdsignal, macdhist = talib.MACD(closes, fastperiod=fast, slowperiod=slow, signalperiod=signal)
    m = [None if np.isnan(v) else float(v) for v in macd]
    s = [None if np.isnan(v) else float(v) for v in macdsignal]
    h = [None if np.isnan(v) else float(v) for v in macdhist]
    return m, s, h


def compute_stochastic(
    highs: List[float], lows: List[float], closes: List[float],
    period: int = 14, smooth_k: int = 3
) -> Tuple[Optional[float], Optional[float]]:
    """Compute Stochastic Oscillator %K and %D."""
    min_len = period + smooth_k - 1
    if len(closes) < min_len:
        return _compute_simple_stochastic(highs, lows, closes, period, smooth_k)

    fast_k, fast_d = talib.STOCH(
        np.array(highs, dtype=float), 
        np.array(lows, dtype=float), 
        np.array(closes, dtype=float),
        fastk_period=period,
        slowk_period=smooth_k,
        slowk_matype=0,
        slowd_period=smooth_k,
        slowd_matype=0
    )
    k_val = fast_k[-1]
    d_val = fast_d[-1]
    return (k_val if not np.isnan(k_val) else None, 
            d_val if not np.isnan(d_val) else None)


def compute_adx(
    highs: List[float], lows: List[float], closes: List[float], period: int = 14
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Compute ADX, +DI, -DI using Wilder's smoothing."""
    if len(closes) < period + 1:
        return _compute_simple_adx(highs, lows, closes, period)
    
    h = np.array(highs, dtype=float)
    l = np.array(lows, dtype=float)
    c = np.array(closes, dtype=float)
    
    adx = talib.ADX(h, l, c, timeperiod=period)
    plus_di = talib.PLUS_DI(h, l, c, timeperiod=period)
    minus_di = talib.MINUS_DI(h, l, c, timeperiod=period)
    
    adx_val = adx[-1]
    plus_val = plus_di[-1]
    minus_val = minus_di[-1]
    
    return (adx_val if not np.isnan(adx_val) else None,
            plus_val if not np.isnan(plus_val) else None,
            minus_val if not np.isnan(minus_val) else None)


def compute_obv(closes: List[float], volumes: List[float]) -> Optional[float]:
    """Compute On-Balance Volume (latest value)."""
    if len(closes) < 2 or len(volumes) < 2:
        return None
    result = talib.OBV(np.array(closes, dtype=float), np.array(volumes, dtype=float))
    val = result[-1]
    return val if not np.isnan(val) else None


def compute_mfi(
    highs: List[float], lows: List[float], closes: List[float],
    volumes: List[float], period: int = 14
) -> Optional[float]:
    """Compute Money Flow Index."""
    if len(closes) < period + 1:
        return _compute_simple_mfi(highs, lows, closes, volumes, period)
    result = talib.MFI(
        np.array(highs, dtype=float), 
        np.array(lows, dtype=float), 
        np.array(closes, dtype=float), 
        np.array(volumes, dtype=float), 
        timeperiod=period
    )
    val = result[-1]
    return val if not np.isnan(val) else None


def compute_cci(
    highs: List[float], lows: List[float], closes: List[float], period: int = 20
) -> Optional[float]:
    """Compute Commodity Channel Index."""
    if len(closes) < period:
        return _compute_simple_cci(highs, lows, closes, period)
    result = talib.CCI(
        np.array(highs, dtype=float), 
        np.array(lows, dtype=float), 
        np.array(closes, dtype=float), 
        timeperiod=period
    )
    val = result[-1]
    return val if not np.isnan(val) else None


def compute_williams_r(
    highs: List[float], lows: List[float], closes: List[float], period: int = 14
) -> Optional[float]:
    """Compute Williams %R."""
    if len(closes) < period:
        return _compute_simple_williams_r(highs, lows, closes, period)
    result = talib.WILLR(
        np.array(highs, dtype=float), 
        np.array(lows, dtype=float), 
        np.array(closes, dtype=float), 
        timeperiod=period
    )
    val = result[-1]
    return val if not np.isnan(val) else None


def compute_ichimoku(
    highs: List[float], lows: List[float], closes: List[float],
    tenkan_period: int = 9, kijun_period: int = 26, senkou_b_period: int = 52,
) -> Optional[Dict[str, Optional[float]]]:
    """Compute Ichimoku Cloud components.

    Returns dict with tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b,
    chikou_span, cloud_top, cloud_bottom. Returns None if insufficient data.
    """
    if len(closes) < senkou_b_period:
        return _compute_simple_ichimoku(highs, lows, closes, tenkan_period, kijun_period, senkou_b_period)

    # Tenkan-sen (Conversion Line): (highest high + lowest low) / 2 over tenkan_period
    tenkan_high = max(highs[-tenkan_period:])
    tenkan_low = min(lows[-tenkan_period:])
    tenkan_sen = (tenkan_high + tenkan_low) / 2

    # Kijun-sen (Base Line): (highest high + lowest low) / 2 over kijun_period
    kijun_high = max(highs[-kijun_period:])
    kijun_low = min(lows[-kijun_period:])
    kijun_sen = (kijun_high + kijun_low) / 2

    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2

    # Senkou Span B (Leading Span B): (highest high + lowest low) / 2 over senkou_b_period
    senkou_b_high = max(highs[-senkou_b_period:])
    senkou_b_low = min(lows[-senkou_b_period:])
    senkou_span_b = (senkou_b_high + senkou_b_low) / 2

    # Chikou Span (Lagging Span): current close
    chikou_span = closes[-1]

    # Cloud boundaries
    cloud_top = max(senkou_span_a, senkou_span_b)
    cloud_bottom = min(senkou_span_a, senkou_span_b)

    return {
        "tenkan_sen": round(tenkan_sen, 8),
        "kijun_sen": round(kijun_sen, 8),
        "senkou_span_a": round(senkou_span_a, 8),
        "senkou_span_b": round(senkou_span_b, 8),
        "chikou_span": round(chikou_span, 8),
        "cloud_top": round(cloud_top, 8),
        "cloud_bottom": round(cloud_bottom, 8),
    }


def compute_donchian_channels(
    highs: List[float], lows: List[float], period: int = 20
) -> Optional[Dict[str, float]]:
    """Compute Donchian Channels (upper, middle, lower) using TA-Lib MAX/MIN.

    Upper = highest high over N periods
    Lower = lowest low over N periods
    Middle = (upper + lower) / 2

    Returns dict with 'upper', 'middle', 'lower', or None if insufficient data.
    """
    if len(highs) < period or len(lows) < period:
        return _compute_simple_donchian_channels(highs, lows, period)

    upper_arr = talib.MAX(np.array(highs, dtype=float), timeperiod=period)
    lower_arr = talib.MIN(np.array(lows, dtype=float), timeperiod=period)
    
    upper = upper_arr[-1]
    lower = lower_arr[-1]
    
    if np.isnan(upper) or np.isnan(lower):
        return None

    middle = (upper + lower) / 2.0

    return {
        "upper": round(upper, 8),
        "middle": round(middle, 8),
        "lower": round(lower, 8),
    }


def compute_macd(
    closes: List[float], fast: int = 12, slow: int = 26, signal: int = 9
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Compute MACD line, signal line, and histogram."""
    if len(closes) < slow + signal:
        return _compute_simple_macd(closes, fast, slow, signal)
    
    macd, macdsignal, macdhist = talib.MACD(
        np.array(closes, dtype=float), 
        fastperiod=fast, 
        slowperiod=slow, 
        signalperiod=signal
    )
    
    m_val = macd[-1]
    s_val = macdsignal[-1]
    h_val = macdhist[-1]
    
    return (m_val if not np.isnan(m_val) else None,
            s_val if not np.isnan(s_val) else None,
            h_val if not np.isnan(h_val) else None)


def compute_bollinger_bands(
    closes: List[float], period: int = 20, std_dev: float = 2.0
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Compute Bollinger Bands (upper, middle, lower)."""
    if len(closes) < period:
        return _compute_simple_bollinger_bands(closes, period, std_dev)
    
    upper, middle, lower = talib.BBANDS(
        np.array(closes, dtype=float), 
        timeperiod=period, 
        nbdevup=std_dev, 
        nbdevdn=std_dev, 
        matype=0
    )
    
    u_val = upper[-1]
    m_val = middle[-1]
    l_val = lower[-1]
    
    return (u_val if not np.isnan(u_val) else None,
            m_val if not np.isnan(m_val) else None,
            l_val if not np.isnan(l_val) else None)


def compute_all_indicators(
    candles: List[List],
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compute all technical indicators for a list of OHLCV candles.
    candles: list of [timestamp, open, high, low, close, volume]
    config: optional dict with custom periods (e.g., {'rsi_period': 14, ...})
    Returns a dict with keys like 'atr', 'rsi', 'macd', etc.
    Missing indicators are set to None.
    """
    if config is None:
        config = {}

    ind = {}
    if len(candles) < 2:
        return ind

    # ATR — fall back to simple average True Range for long timeframes with few candles
    ind['atr'] = compute_atr(candles)
    if ind['atr'] is None and len(candles) >= 2:
        ind['atr'] = _compute_simple_atr(candles)

    rsi_period = config.get('rsi_period', 14)
    macd_fast = config.get('macd_fast', 12)
    macd_slow = config.get('macd_slow', 26)
    macd_signal_period = config.get('macd_signal', 9)
    bb_period = config.get('bb_period', 20)
    bb_std = config.get('bb_std', 2.0)
    ema_fast = config.get('ema_fast', 9)
    ema_slow = config.get('ema_slow', 21)
    stoch_k_period = config.get('stoch_k_period', 14)
    stoch_d_period = config.get('stoch_d_period', 3)
    adx_period = config.get('adx_period', 14)
    mfi_period = config.get('mfi_period', 14)
    cci_period = config.get('cci_period', 20)
    willr_period = config.get('willr_period', 14)
    ichimoku_tenkan = config.get('ichimoku_tenkan', 9)
    ichimoku_kijun = config.get('ichimoku_kijun', 26)
    ichimoku_senkou_b = config.get('ichimoku_senkou_b', 52)
    donchian_period = config.get('donchian_period', 20)

    # Determine the minimum number of candles required to avoid partial indicator sets
    closes = [c[4] for c in candles]
    highs = [c[2] for c in candles]
    lows = [c[3] for c in candles]
    volumes = [c[5] for c in candles]

    # RSI (needs rsi_period + 1)
    if len(candles) >= rsi_period + 1:
        ind['rsi'] = compute_rsi(closes, period=rsi_period)

    # MACD (needs macd_slow + macd_signal_period)
    if len(candles) >= macd_slow + macd_signal_period:
        macd_val, macd_sig, macd_hist = compute_macd(closes, fast=macd_fast, slow=macd_slow, signal=macd_signal_period)
        ind['macd'] = macd_val
        ind['macd_signal'] = macd_sig
        ind['macd_hist'] = macd_hist

    # Bollinger Bands (needs bb_period)
    if len(candles) >= bb_period:
        bb_upper, bb_middle, bb_lower = compute_bollinger_bands(closes, period=bb_period, std_dev=bb_std)
        ind['bb_upper'] = bb_upper
        ind['bb_middle'] = bb_middle
        ind['bb_lower'] = bb_lower

    # EMA (compute what we can; ema_slow needs ema_slow candles, ema_fast needs ema_fast)
    if len(candles) >= ema_fast:
        ema_9_list = compute_ema(closes, ema_fast)
        last_ema_9 = ema_9_list[-1] if ema_9_list else None
        ind['ema_9'] = last_ema_9 if last_ema_9 is not None and not np.isnan(last_ema_9) else None
    if len(candles) >= ema_slow:
        ema_21_list = compute_ema(closes, ema_slow)
        last_ema_21 = ema_21_list[-1] if ema_21_list else None
        ind['ema_21'] = last_ema_21 if last_ema_21 is not None and not np.isnan(last_ema_21) else None

    # Stochastic (needs stoch_k_period + stoch_d_period - 1)
    min_stoch = stoch_k_period + stoch_d_period - 1
    if len(candles) >= min_stoch:
        stoch_k, stoch_d = compute_stochastic(highs, lows, closes, period=stoch_k_period, smooth_k=stoch_d_period)
        ind['stochastic_k'] = stoch_k
        ind['stochastic_d'] = stoch_d

    # ADX (needs adx_period + 1)
    if len(candles) >= adx_period + 1:
        adx_val, plus_di, minus_di = compute_adx(highs, lows, closes, period=adx_period)
        ind['adx'] = adx_val
        ind['plus_di'] = plus_di
        ind['minus_di'] = minus_di

    # OBV (needs 2)
    if len(candles) >= 2:
        ind['obv'] = compute_obv(closes, volumes)

    # MFI (needs mfi_period + 1)
    if len(candles) >= mfi_period + 1:
        ind['mfi'] = compute_mfi(highs, lows, closes, volumes, period=mfi_period)

    # CCI (needs cci_period)
    if len(candles) >= cci_period:
        ind['cci'] = compute_cci(highs, lows, closes, period=cci_period)

    # Williams %R (needs willr_period)
    if len(candles) >= willr_period:
        ind['williams_r'] = compute_williams_r(highs, lows, closes, period=willr_period)

    # Ichimoku (needs ichimoku_senkou_b — the most data-hungry indicator)
    if len(candles) >= ichimoku_senkou_b:
        ind['ichimoku'] = compute_ichimoku(highs, lows, closes, tenkan_period=ichimoku_tenkan, kijun_period=ichimoku_kijun, senkou_b_period=ichimoku_senkou_b)

    # Donchian Channels (needs donchian_period)
    if len(candles) >= donchian_period:
        ind['donchian_channels'] = compute_donchian_channels(highs, lows, period=donchian_period)

    # Parabolic SAR (needs 2)
    if len(candles) >= 2:
        ind['parabolic_sar'] = compute_parabolic_sar(highs, lows)

    # Keltner Channels (needs bb_period, depends on EMA + ATR)
    if len(candles) >= bb_period:
        ind['keltner_channels'] = compute_keltner_channels(closes, highs, lows, period=bb_period)

    return ind


def compute_parabolic_sar(
    highs: List[float], lows: List[float],
    af_start: float = 0.02, af_max: float = 0.2
) -> Optional[float]:
    """Compute Parabolic SAR for the latest bar using the standard algorithm."""
    if len(highs) < 2:
        return None
    result = talib.SAR(
        np.array(highs, dtype=float), 
        np.array(lows, dtype=float), 
        acceleration=af_start, 
        maximum=af_max
    )
    val = result[-1]
    return val if not np.isnan(val) else None


def compute_keltner_channels(
    closes: List[float], highs: List[float], lows: List[float],
    period: int = 20, atr_mult: float = 2.0
) -> Optional[Dict[str, float]]:
    """Compute Keltner Channels using EMA middle and ATR-based bands."""
    if len(closes) < period:
        return None
    ema_values = compute_ema(closes, period)
    if not ema_values:
        return None
    middle = ema_values[-1]
    if middle is None or np.isnan(middle):
        return None
    candles = [[0, 0, h, l, c, 0] for h, l, c in zip(highs, lows, closes)]
    atr = compute_atr(candles, period)
    if atr is None:
        atr = _compute_simple_atr(candles, period)
    if atr is None:
        return None
    upper = middle + atr_mult * atr
    lower = middle - atr_mult * atr
    return {
        "upper": round(upper, 8),
        "middle": round(middle, 8),
        "lower": round(lower, 8),
    }


def compute_vwap(candles: List[List], period: int = 14) -> Optional[float]:
    """Compute rolling VWAP over the last N periods."""
    if len(candles) < period:
        return None
    recent = candles[-period:]
    total_pv = 0.0
    total_v = 0.0
    for c in recent:
        typical_price = (c[2] + c[3] + c[4]) / 3.0
        volume = c[5]
        total_pv += typical_price * volume
        total_v += volume
    if total_v == 0:
        return None
    return round(total_pv / total_v, 6)


def compute_pivot_points(prev_high: float, prev_low: float, prev_close: float) -> Dict[str, float]:
    """Compute standard pivot points and support/resistance levels."""
    p = (prev_high + prev_low + prev_close) / 3.0
    r1 = 2 * p - prev_low
    s1 = 2 * p - prev_high
    r2 = p + (prev_high - prev_low)
    s2 = p - (prev_high - prev_low)
    r3 = prev_high + 2 * (p - prev_low)
    s3 = prev_low - 2 * (prev_high - p)
    return {
        "pivot": round(p, 4),
        "r1": round(r1, 4),
        "r2": round(r2, 4),
        "r3": round(r3, 4),
        "s1": round(s1, 4),
        "s2": round(s2, 4),
        "s3": round(s3, 4),
    }
