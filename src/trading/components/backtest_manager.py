"""Backtest management component for the TradingEngine.

Handles running backtests and the Step 2 LLM call to produce the final signal.
Extracted from TradingEngine to reduce class size and improve maintainability.
"""
import asyncio
import hashlib
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from src.config.settings import settings
from src.database import get_ohlcv, get_recent_backtest_result, save_backtest_result
from src.exchanges.fees import calculate_transaction_costs
from src.indicators import compute_atr_series, compute_adx_series, compute_rsi_series, compute_macd_series
from src.strategies.backtester import backtest_strategy, format_backtest_summary, walk_forward_backtest, format_walk_forward_summary
from src.strategies.base import Signal

logger = logging.getLogger(__name__)


class BacktestManager:
    """Handles backtesting and final decision LLM calls for the TradingEngine."""

    def __init__(self, engine):
        self.engine = engine

    def _prepare_backtest_variants(
        self,
        symbol: str,
        preliminary_signal: Signal,
        historical_ohlcv: Optional[List[List]],
        raw_candles: Optional[List[List]],
    ) -> List[Dict[str, Any]]:
        """Determine which variant param sets to backtest, applying dedup and caps."""
        engine = self.engine
        variants_to_test = []
        if preliminary_signal.backtest_variants:
            variants_to_test = list(preliminary_signal.backtest_variants)
        else:
            # Fallback: use the preliminary signal's own params as a single variant
            fallback_params = dict(preliminary_signal.strategy_params or {})
            if "backtest_entry_config" not in fallback_params:
                fallback_params["backtest_entry_config"] = {
                    "ema_period": 21,
                    "ema_direction": "above",
                    "min_adx": 20,
                    "logic": "and",
                }
            variants_to_test.append(fallback_params)
        # --- Deduplicate variants with identical key risk parameters ---
        variants_to_test = engine._deduplicate_variants(variants_to_test)
        # Safety cap: limit to configured max variants to prevent excessive backtest time
        if len(variants_to_test) > settings.MAX_BACKTEST_VARIANTS:
            logger.warning(
                f"LLM returned {len(variants_to_test)} backtest variants for {symbol}, "
                f"capping to {settings.MAX_BACKTEST_VARIANTS}"
            )
            variants_to_test = variants_to_test[:settings.MAX_BACKTEST_VARIANTS]

        # Limit number of variants based on available data length
        source_candles = historical_ohlcv or raw_candles or []
        if source_candles and len(source_candles) < 50:
            variants_to_test = variants_to_test[:2]
        elif source_candles and len(source_candles) < 100:
            variants_to_test = variants_to_test[:3]

        return variants_to_test

    async def _run_backtest_from_signal(
        self,
        symbol: str,
        signal: Signal,
        atr: Optional[float],
        current_price: float,
        tf_secs: int,
        assigned_tf: str,
        historical_ohlcv: Optional[List[List]],
        raw_candles: Optional[List[List]],
        base_balance: float,
        is_btp: bool,
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        """Run a backtest using the parameters from a signal. Returns (stats, summary)."""
        engine = self.engine
        bt_params = signal.strategy_params or {}
        bt_sl_pct = bt_params.get("stop_loss_pct", 0.02)
        bt_tp_pct = bt_params.get("take_profit_pct", 0.05)
        bt_sl_atr_mult = bt_params.get("stop_loss_atr_multiple")
        bt_tp_atr_mult = bt_params.get("take_profit_atr_multiple")
        bt_max_hold = bt_params.get("max_hold_time_seconds")
        bt_trailing = bt_params.get("trailing_stop", False)
        bt_trail_dist = bt_params.get("trailing_stop_distance_pct")
        bt_trail_act = bt_params.get("trailing_stop_activation_pct")
        bt_entry_config = bt_params.get("backtest_entry_config")

        bt_period_days = bt_params.get("backtest_period_days")
        if bt_period_days is not None:
            bt_period_days = max(30, min(int(bt_period_days), settings.OHLCV_RETENTION_DAYS))
            bt_since_ms = int(time.time() * 1000) - bt_period_days * 24 * 60 * 60 * 1000
            bt_limit = int((bt_period_days * 86400) / tf_secs) + 100
            bt_db_candles = await asyncio.to_thread(
                get_ohlcv, symbol, assigned_tf, since_ms=bt_since_ms, limit=bt_limit
            )
            if bt_db_candles:
                bt_candles = [
                    [c["timestamp"], c["open"], c["high"], c["low"], c["close"], c["volume"]]
                    for c in bt_db_candles
                ]
            else:
                bt_candles = historical_ohlcv or raw_candles
        else:
            bt_candles = historical_ohlcv or raw_candles

        # Early skip: if the assigned timeframe cannot possibly have enough candles
        # given the data retention period, skip backtesting entirely instead of
        # falling back to a much shorter timeframe whose results would be misleading.
        tf_seconds_bt = engine._timeframe_to_seconds(assigned_tf)
        max_possible_candles = (settings.OHLCV_RETENTION_DAYS * 86400) / tf_seconds_bt
        if max_possible_candles < 5:
            return None, (
                f"Backtesting skipped for {assigned_tf}: only ~{int(max_possible_candles)} candles possible "
                f"with {settings.OHLCV_RETENTION_DAYS} days retention (need ≥5). "
                f"Rely on LLM analysis, fundamentals, and multi-timeframe indicators instead."
            )

        # --- Fallback to shorter timeframes when the assigned timeframe has too few candles ---
        MIN_BACKTEST_CANDLES = 20
        backtest_fallback_note = ""
        if bt_candles is None or len(bt_candles) < MIN_BACKTEST_CANDLES:
            if assigned_tf in settings.OHLCV_TIMEFRAMES:
                tf_idx = settings.OHLCV_TIMEFRAMES.index(assigned_tf)
                for shorter_tf in settings.OHLCV_TIMEFRAMES[tf_idx + 1:]:
                    shorter_tf_secs = engine._timeframe_to_seconds(shorter_tf)
                    try:
                        if bt_period_days is not None:
                            fb_since_ms = int(time.time() * 1000) - bt_period_days * 24 * 60 * 60 * 1000
                            fb_limit = int((bt_period_days * 86400) / shorter_tf_secs) + 100
                            fb_db_candles = await asyncio.to_thread(
                                get_ohlcv, symbol, shorter_tf, since_ms=fb_since_ms, limit=fb_limit
                            )
                        else:
                            fb_db_candles = await asyncio.to_thread(
                                get_ohlcv, symbol, shorter_tf, limit=500
                            )
                        if fb_db_candles and len(fb_db_candles) >= MIN_BACKTEST_CANDLES:
                            bt_candles = [
                                [c["timestamp"], c["open"], c["high"], c["low"], c["close"], c["volume"]]
                                for c in fb_db_candles
                            ]
                            backtest_fallback_note = (
                                f" ⚠️ FALLBACK WARNING: Backtest was run on {shorter_tf} candles, NOT {assigned_tf}. "
                                f"The assigned {assigned_tf} timeframe had insufficient candles (< {MIN_BACKTEST_CANDLES}) "
                                f"with {settings.OHLCV_RETENTION_DAYS} days retention. "
                                f"Results from {shorter_tf} may not accurately represent {assigned_tf} behavior — treat with caution."
                            )
                            logger.info(
                                f"Backtest fallback for {symbol}: assigned_tf={assigned_tf} had insufficient candles, "
                                f"using {shorter_tf} ({len(bt_candles)} candles)."
                            )
                            break
                    except Exception as e:
                        logger.debug(f"Backtest fallback to {shorter_tf} failed for {symbol}: {e}")

        bt_position_fraction = bt_params.get("position_size_fraction", 1.0 / engine.effective_max_symbols if engine.effective_max_symbols > 0 else 1.0)
        bt_trade_value = base_balance * bt_position_fraction
        if bt_trade_value > 0:
            buy_costs = calculate_transaction_costs("BUY", 100.0, bt_trade_value / 100.0, symbol=symbol)
            sell_costs = calculate_transaction_costs("SELL", 100.0, bt_trade_value / 100.0, symbol=symbol)
            total_fee_pct = (buy_costs["total_costs"] + sell_costs["total_costs"]) / bt_trade_value
            bt_fee_rate = total_fee_pct / 2
        else:
            bt_fee_rate = 0.006

        atr_series = None
        adx_series = None
        rsi_series = None
        macd_hist_series = None
        if bt_candles and len(bt_candles) >= 2:
            def _compute_bt_indicator_series():
                _atr_series = None
                _adx_series = None
                _rsi_series = None
                _macd_hist_series = None
                try:
                    if bt_params.get("trailing_stop_atr_multiple") or bt_sl_atr_mult or bt_tp_atr_mult:
                        _atr_series = compute_atr_series(bt_candles, period=14)
                    _adx_series = compute_adx_series(bt_candles, period=14)
                    _rsi_series = compute_rsi_series(bt_candles, period=14)
                    _, _, _macd_hist_series = compute_macd_series(bt_candles)
                except Exception:
                    pass
                return _atr_series, _adx_series, _rsi_series, _macd_hist_series

            atr_series, adx_series, rsi_series, macd_hist_series = await asyncio.to_thread(_compute_bt_indicator_series)

        # Fetch LLM-configured thresholds for backtest filters
        bt_max_rsi = 70.0
        try:
            raw = await asyncio.to_thread(engine.redis.get, "trading:skip_eval_rsi_overbought")
            if raw:
                bt_max_rsi = float(raw)
        except Exception:
            pass

        # Fetch portfolio caps for position sizing simulation
        bt_global_risk_mult = 1.0
        bt_max_port_exp = None
        bt_max_port_risk = None
        try:
            raw = await asyncio.to_thread(engine.redis.get, "trading:global_risk_multiplier")
            if raw:
                bt_global_risk_mult = float(raw)
            raw = await asyncio.to_thread(engine.redis.get, "trading:max_portfolio_exposure_pct")
            if raw:
                bt_max_port_exp = float(raw)
            raw = await asyncio.to_thread(engine.redis.get, "trading:max_portfolio_stop_risk_pct")
            if raw:
                bt_max_port_risk = float(raw)
        except Exception:
            pass

        if bt_candles and len(bt_candles) >= 20:
            bt_kwargs = dict(
                stop_loss_pct=bt_sl_pct,
                take_profit_pct=bt_tp_pct,
                stop_loss_atr_multiple=bt_sl_atr_mult,
                take_profit_atr_multiple=bt_tp_atr_mult,
                max_hold_time_seconds=bt_max_hold,
                trailing_stop=bt_trailing,
                trailing_stop_distance_pct=bt_trail_dist,
                trailing_stop_activation_pct=bt_trail_act,
                partial_take_profit_levels=bt_params.get("partial_take_profit_levels"),
                breakeven_activation_pct=bt_params.get("breakeven_activation_pct"),
                trailing_take_profit=bt_params.get("trailing_take_profit", False),
                trailing_take_profit_distance_pct=bt_params.get("trailing_take_profit_distance_pct"),
                trailing_stop_atr_multiple=bt_params.get("trailing_stop_atr_multiple"),
                atr_values=atr_series,
                max_unrealized_loss_pct=bt_params.get("max_unrealized_loss_pct"),
                adx_values=adx_series,
                rsi_values=rsi_series,
                max_rsi=bt_max_rsi,
                macd_hist_values=macd_hist_series,
                fee_rate=bt_fee_rate,
                fee_model="intesa",
                trade_value=bt_trade_value,
                is_btp=is_btp,
                cooldown_after_loss_seconds=bt_params.get("cooldown_after_loss_seconds"),
                slippage_pct=0.001,
                slippage_model="dynamic",
                slippage_base_pct=0.001,
                slippage_max_pct=0.01,
                backtest_entry_config=bt_entry_config,
                direction="long",
                simulate_position_sizing=True,
                initial_balance=base_balance,
                confidence=signal.confidence,
                confidence_sizing_weight=bt_params.get("confidence_sizing_weight", 0.0),
                global_risk_multiplier=bt_global_risk_mult,
                position_size_multiplier=bt_params.get("position_size_multiplier", 1.0),
                max_risk_per_trade_pct=bt_params.get("max_risk_per_trade_pct"),
                max_portfolio_risk_pct=bt_params.get("max_portfolio_risk_pct"),
                max_portfolio_exposure_pct=bt_max_port_exp,
                max_portfolio_stop_risk_pct=bt_max_port_risk,
                position_size_fraction=bt_position_fraction,
                gap_tolerance_mult=1.5,
                on_gaps="warn",
            )
            backtest_stats = await asyncio.to_thread(
                backtest_strategy,
                candles=bt_candles,
                **bt_kwargs,
            )
            bt_entry_config_used = bt_entry_config is not None and isinstance(bt_entry_config, dict) and len(bt_entry_config) > 0
            bt_summary = format_backtest_summary(backtest_stats, entry_config_used=bt_entry_config_used)
            if backtest_fallback_note:
                bt_summary += backtest_fallback_note

            if len(bt_candles) >= 100:
                wf_stats = await asyncio.to_thread(
                    walk_forward_backtest,
                    candles=bt_candles,
                    num_windows=5,
                    **bt_kwargs,
                )
                bt_summary = bt_summary + "\n" + format_walk_forward_summary(wf_stats)

            return backtest_stats, bt_summary
        if backtest_fallback_note:
            return None, f"Insufficient data for backtest (need ≥{MIN_BACKTEST_CANDLES} candles).{backtest_fallback_note}"
        return None, f"Insufficient data for backtest for {assigned_tf} (need ≥{MIN_BACKTEST_CANDLES} candles with {settings.OHLCV_RETENTION_DAYS} days retention)."

    async def _run_backtest_variant(
        self,
        symbol: str,
        variant_params: Dict[str, Any],
        preliminary_signal: Signal,
        atr: Optional[float],
        current_price: float,
        tf_secs: int,
        assigned_tf: str,
        historical_ohlcv: Optional[List[List]],
        raw_candles: Optional[List[List]],
        base_balance: float,
        is_btp: bool,
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        """Run a single backtest variant with database persistence and concurrency limiting."""
        engine = self.engine
        # Build params hash for dedup lookup
        source_candles = historical_ohlcv or raw_candles or []
        last_ts = source_candles[-1][0] if source_candles else 0
        candle_count = len(source_candles)
        params_hash = hashlib.md5(
            json.dumps(variant_params, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]

        # Check database for a recent identical backtest (dedup within 6 hours)
        try:
            recent = await asyncio.to_thread(
                get_recent_backtest_result, symbol, assigned_tf, params_hash, 21600
            )
            if recent:
                logger.debug(f"Backtest DB cache hit for {symbol} {assigned_tf} (params_hash={params_hash})")
                return recent["stats"], recent["summary"]
        except Exception:
            pass

        # Run backtest with concurrency limiting
        async with engine._backtest_semaphore:
            variant_signal = Signal(
                action="BUY",
                confidence=preliminary_signal.confidence,
                reasoning=preliminary_signal.reasoning,
                strategy_params=variant_params,
            )
            bt_stats, bt_summary = await self._run_backtest_from_signal(
                symbol=symbol,
                signal=variant_signal,
                atr=atr,
                current_price=current_price,
                tf_secs=tf_secs,
                assigned_tf=assigned_tf,
                historical_ohlcv=historical_ohlcv,
                raw_candles=raw_candles,
                base_balance=base_balance,
                is_btp=is_btp,
            )

        # Persist the result to the database
        if bt_stats is not None:
            try:
                await asyncio.to_thread(
                    save_backtest_result, symbol, assigned_tf, params_hash,
                    variant_params, bt_stats, bt_summary
                )
            except Exception as e:
                logger.warning(f"Failed to persist backtest result to DB for {symbol}: {e}")

    async def _run_backtest_variants_parallel(
        self,
        symbol: str,
        variants_to_test: List[Dict[str, Any]],
        preliminary_signal: Signal,
        atr: Optional[float],
        current_price: float,
        tf_seconds: int,
        assigned_tf: str,
        historical_ohlcv: Optional[List[List]],
        raw_candles: Optional[List[List]],
        base_balance: float,
        is_btp: bool,
    ) -> List[Dict[str, Any]]:
        """Run all backtest variants in parallel (concurrency-limited by semaphore).

        Returns a list of result dicts, each with keys: variant_params, summary, stats.
        """
        async def _run_single_variant(vp: Dict[str, Any]) -> Dict[str, Any]:
            try:
                bt_stats, bt_summary = await self._run_backtest_variant(
                    symbol=symbol,
                    variant_params=vp,
                    preliminary_signal=preliminary_signal,
                    atr=atr,
                    current_price=current_price,
                    tf_secs=tf_seconds,
                    assigned_tf=assigned_tf,
                    historical_ohlcv=historical_ohlcv,
                    raw_candles=raw_candles,
                    base_balance=base_balance,
                    is_btp=is_btp,
                )
                if bt_stats is not None:
                    return {"variant_params": vp, "summary": bt_summary, "stats": bt_stats}
                else:
                    return {"variant_params": vp, "summary": bt_summary or "Insufficient data for backtest.", "stats": {}}
            except Exception as e:
                logger.warning(f"Backtest variant failed for {symbol}: {e}")
                return {"variant_params": vp, "summary": f"Backtest error: {e}", "stats": {}}

        return list(await asyncio.gather(*[_run_single_variant(vp) for vp in variants_to_test]))
