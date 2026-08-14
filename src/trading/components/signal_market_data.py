"""Market data fetching and indicator computation for signal processing."""
import asyncio
import logging
from typing import Any, Dict, List, Optional

from src.config.settings import settings
from src.database import get_indicators_for_symbols
from src.indicators import compute_vwap, compute_pivot_points
from src.trading.engine_utils import timeframe_to_ms

logger = logging.getLogger(__name__)


class SignalMarketDataFetcher:
    """Handles market data fetching and multi-timeframe indicator computation."""

    def __init__(self, engine):
        self.engine = engine

    async def compute_multi_tf_indicators(
        self, symbol: str, ohlcv_data: Dict[str, List[List]], assigned_tf: str
    ) -> Dict[str, Any]:
        """Batch-fetch indicators from DB and extract assigned-timeframe values.

        Returns a dict with keys: multi_tf_indicators, multi_tf_raw_candles,
        atr, rsi, macd, macd_signal, macd_hist, bb_upper, bb_middle, bb_lower,
        ema_9, ema_21, stochastic_k, stochastic_d, adx, plus_di, minus_di,
        obv, mfi, cci, williams_r, ichimoku, donchian_channels, parabolic_sar,
        keltner_channels, vwap, daily_pivot_points.
        """
        engine = self.engine
        multi_tf_indicators: Dict[str, Dict[str, Any]] = {}
        multi_tf_raw_candles: Dict[str, List[List]] = {}
        atr = rsi = macd = macd_signal = macd_hist = None
        bb_upper = bb_middle = bb_lower = ema_9 = ema_21 = None
        stochastic_k = stochastic_d = adx = plus_di = minus_di = None
        obv = mfi = cci = williams_r = ichimoku = donchian_channels = None
        parabolic_sar = keltner_channels = vwap = daily_pivot_points = None

        batch_inds = await asyncio.to_thread(get_indicators_for_symbols, [symbol], settings.OHLCV_TIMEFRAMES)
        symbol_inds = batch_inds.get(symbol, {})

        for tf in settings.OHLCV_TIMEFRAMES:
            ind = symbol_inds.get(tf)
            if tf == assigned_tf:
                if tf in ohlcv_data and ohlcv_data[tf]:
                    candles = ohlcv_data[tf]
                    multi_tf_raw_candles[tf] = candles
                    if ind:
                        # --- Staleness check: recompute if indicators are older than 2× the candle interval ---
                        ind_ts = ind.get("_indicator_timestamp", None)
                        latest_candle_ts = candles[-1][0] if candles else None
                        if ind_ts is not None and latest_candle_ts is not None:
                            tf_ms = timeframe_to_ms(tf)
                            staleness = latest_candle_ts - ind_ts
                            if staleness > 4 * tf_ms:
                                logger.info(
                                    f"Indicators for {symbol} {tf} are severely stale "
                                    f"(indicator ts={ind_ts}, latest candle ts={latest_candle_ts}, "
                                    f"gap={staleness}ms > {4 * tf_ms}ms). Blocking for recomputation."
                                )
                                # Block and await recomputation, then use the fresh indicators
                                updated_ind = await engine._market_data_manager.compute_and_store_indicators(
                                    symbol, tf, candles
                                )
                                if updated_ind:
                                    ind = updated_ind
                            elif staleness > 2 * tf_ms:
                                logger.info(
                                    f"Indicators for {symbol} {tf} are stale "
                                    f"(indicator ts={ind_ts}, latest candle ts={latest_candle_ts}, "
                                    f"gap={staleness}ms > {2 * tf_ms}ms). Scheduling background recomputation."
                                )
                                # Schedule background recomputation — don't block the evaluation loop.
                                asyncio.create_task(
                                    engine._market_data_manager.compute_and_store_indicators(symbol, tf, candles)
                                )
                    multi_tf_indicators[tf] = ind
                    if ind:
                        atr = ind.get('atr')
                        rsi = ind.get('rsi')
                        macd = ind.get('macd')
                        macd_signal = ind.get('macd_signal')
                        macd_hist = ind.get('macd_hist')
                        bb_upper = ind.get('bb_upper')
                        bb_middle = ind.get('bb_middle')
                        bb_lower = ind.get('bb_lower')
                        ema_9 = ind.get('ema_9')
                        ema_21 = ind.get('ema_21')
                        stochastic_k = ind.get('stochastic_k')
                        stochastic_d = ind.get('stochastic_d')
                        adx = ind.get('adx')
                        plus_di = ind.get('plus_di')
                        minus_di = ind.get('minus_di')
                        obv = ind.get('obv')
                        mfi = ind.get('mfi')
                        cci = ind.get('cci')
                        williams_r = ind.get('williams_r')
                        ichimoku = ind.get('ichimoku')
                        donchian_channels = ind.get('donchian_channels')
                        parabolic_sar = ind.get('parabolic_sar')
                        keltner_channels = ind.get('keltner_channels')
                        vwap = compute_vwap(candles)
            else:
                # For non-assigned timeframes, use precomputed indicators without staleness check
                if ind:
                    multi_tf_indicators[tf] = ind

                # If we fetched 1d candles for pivot points, add them to raw candles
                if tf == "1d" and tf in ohlcv_data and ohlcv_data[tf]:
                    multi_tf_raw_candles[tf] = ohlcv_data[tf]

        # Compute daily pivot points from the 1d timeframe (if available)
        if "1d" in multi_tf_raw_candles and len(multi_tf_raw_candles["1d"]) >= 2:
            daily_candles = multi_tf_raw_candles["1d"]
            prev_daily = daily_candles[-2]
            daily_pivot_points = compute_pivot_points(prev_daily[2], prev_daily[3], prev_daily[4])

        return {
            "multi_tf_indicators": multi_tf_indicators,
            "multi_tf_raw_candles": multi_tf_raw_candles,
            "atr": atr, "rsi": rsi, "macd": macd, "macd_signal": macd_signal,
            "macd_hist": macd_hist, "bb_upper": bb_upper, "bb_middle": bb_middle,
            "bb_lower": bb_lower, "ema_9": ema_9, "ema_21": ema_21,
            "stochastic_k": stochastic_k, "stochastic_d": stochastic_d,
            "adx": adx, "plus_di": plus_di, "minus_di": minus_di,
            "obv": obv, "mfi": mfi, "cci": cci, "williams_r": williams_r,
            "ichimoku": ichimoku, "donchian_channels": donchian_channels,
            "parabolic_sar": parabolic_sar, "keltner_channels": keltner_channels,
            "vwap": vwap, "daily_pivot_points": daily_pivot_points,
        }
