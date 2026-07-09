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
from src.database import get_ohlcv, get_recent_backtest_result, save_backtest_result, get_backtest_results_for_symbol
from src.exchanges.fees import calculate_transaction_costs
from src.indicators import compute_atr_series, compute_adx_series, compute_rsi_series, compute_macd_series
from src.llm.cache import get_cached_llm_response
from src.llm.backtest_prompts import build_final_decision_messages
from src.strategies.backtester import backtest_strategy, format_backtest_summary, walk_forward_backtest, format_walk_forward_summary, BacktestConfig
from src.strategies.base import Signal
from src.strategies.llm_parser import create_strategy_from_llm

logger = logging.getLogger(__name__)


class BacktestManager:
    """Handles backtesting and final decision LLM calls for the TradingEngine."""

    def __init__(self, engine, event_bus):
        self.engine = engine
        self.event_bus = event_bus
        self.event_bus.subscribe("run_backtest_and_final_decision", self.run_backtest_and_final_decision)

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
            # Ensure each variant has a backtest_entry_config — if missing, default it
            _default_entry_config = {
                "ema_period": 21,
                "ema_direction": "above",
                "min_adx": 20,
                "logic": "and",
            }
            for v in variants_to_test:
                if not isinstance(v, dict):
                    continue
                if not v.get("backtest_entry_config"):
                    # Try to copy from the preliminary signal's params first
                    _prelim_bec = (preliminary_signal.strategy_params or {}).get("backtest_entry_config")
                    v["backtest_entry_config"] = _prelim_bec if isinstance(_prelim_bec, dict) else dict(_default_entry_config)
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
        MIN_STATISTICALLY_SIGNIFICANT_CANDLES = 50
        MIN_BACKTEST_CANDLES = 50
        fallback_tf = None
        if max_possible_candles < MIN_STATISTICALLY_SIGNIFICANT_CANDLES:
            # Try to fall back to a shorter timeframe that has enough candles
            fallback_tfs = ["1w", "1d", "1h"]
            for ft in fallback_tfs:
                if ft == assigned_tf:
                    continue
                ft_secs = engine._timeframe_to_seconds(ft)
                ft_max_candles = (settings.OHLCV_RETENTION_DAYS * 86400) / ft_secs
                if ft_max_candles >= MIN_STATISTICALLY_SIGNIFICANT_CANDLES:
                    fallback_tf = ft
                    break
            if fallback_tf is None:
                return None, (
                    f"Backtesting skipped for {assigned_tf}: only ~{int(max_possible_candles)} candles possible "
                    f"with {settings.OHLCV_RETENTION_DAYS} days retention (need ≥{MIN_STATISTICALLY_SIGNIFICANT_CANDLES}). "
                    f"Rely on LLM analysis, fundamentals, and multi-timeframe indicators instead."
                )
            logger.info(
                f"Backtest timeframe fallback for {symbol}: {assigned_tf} → {fallback_tf} "
                f"(insufficient candles on {assigned_tf}, using {fallback_tf} for backtest validation)"
            )
            # Use the fallback timeframe for backtesting
            tf_seconds_bt = engine._timeframe_to_seconds(fallback_tf)
            bt_since_ms_fallback = int(time.time() * 1000) - settings.OHLCV_RETENTION_DAYS * 24 * 60 * 60 * 1000
            bt_limit_fallback = int((settings.OHLCV_RETENTION_DAYS * 86400) / tf_seconds_bt) + 100
            bt_db_candles_fallback = await asyncio.to_thread(
                get_ohlcv, symbol, fallback_tf, since_ms=bt_since_ms_fallback, limit=bt_limit_fallback
            )
            if bt_db_candles_fallback and len(bt_db_candles_fallback) >= MIN_BACKTEST_CANDLES:
                bt_candles = [
                    [c["timestamp"], c["open"], c["high"], c["low"], c["close"], c["volume"]]
                    for c in bt_db_candles_fallback
                ]
                # Adjust max_hold_time for the fallback timeframe
                if bt_max_hold is not None:
                    original_tf_secs = engine._timeframe_to_seconds(assigned_tf)
                    hold_ratio = bt_max_hold / original_tf_secs if original_tf_secs > 0 else 1.0
                    bt_max_hold = int(hold_ratio * tf_seconds_bt)
            else:
                return None, (
                    f"Backtesting skipped for {assigned_tf}: only ~{int(max_possible_candles)} candles possible "
                    f"with {settings.OHLCV_RETENTION_DAYS} days retention (need ≥{MIN_STATISTICALLY_SIGNIFICANT_CANDLES}). "
                    f"Fallback to {fallback_tf} also insufficient. "
                    f"Rely on LLM analysis, fundamentals, and multi-timeframe indicators instead."
                )

        # --- Skip backtesting if the assigned timeframe has too few candles ---
        if bt_candles is None or len(bt_candles) < MIN_BACKTEST_CANDLES:
            return None, (
                f"Insufficient data for backtest for {assigned_tf} (need ≥{MIN_BACKTEST_CANDLES} candles, "
                f"got {len(bt_candles) if bt_candles else 0} with {settings.OHLCV_RETENTION_DAYS} days retention). "
                f"Rely on LLM analysis, fundamentals, and multi-timeframe indicators instead."
            )

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
            raw = await engine.config_service.get_config("skip_eval_rsi_overbought")
            if raw:
                bt_max_rsi = float(raw)
        except (ValueError, TypeError, ConnectionError, TimeoutError, OSError):
            pass

        # Fetch portfolio caps for position sizing simulation
        bt_global_risk_mult = 1.0
        bt_max_port_exp = None
        bt_max_port_risk = None
        try:
            raw = await engine.config_service.get_config("global_risk_multiplier")
            if raw:
                bt_global_risk_mult = float(raw)
            raw = await engine.config_service.get_config("max_portfolio_exposure_pct")
            if raw:
                bt_max_port_exp = float(raw)
            raw = await engine.config_service.get_config("max_portfolio_stop_risk_pct")
            if raw:
                bt_max_port_risk = float(raw)
        except (ValueError, TypeError, ConnectionError, TimeoutError, OSError):
            pass

        if bt_candles and len(bt_candles) >= 20:
            bt_config = BacktestConfig(
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
                config=bt_config,
            )
            backtest_stats["actual_timeframe"] = fallback_tf if fallback_tf is not None else assigned_tf
            backtest_stats["assigned_timeframe"] = assigned_tf
            bt_entry_config_used = bt_entry_config is not None and isinstance(bt_entry_config, dict) and len(bt_entry_config) > 0
            bt_summary = format_backtest_summary(backtest_stats, entry_config_used=bt_entry_config_used)

            if len(bt_candles) >= settings.WALK_FORWARD_CANDLE_THRESHOLD:
                wf_stats = await asyncio.to_thread(
                    walk_forward_backtest,
                    candles=bt_candles,
                    num_windows=5,
                    config=bt_config,
                )
                bt_summary = bt_summary + "\n" + format_walk_forward_summary(wf_stats)

            return backtest_stats, bt_summary
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

        return bt_stats, bt_summary

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

    async def run_step2_llm_call(
        self,
        symbol: str,
        assigned_tf: str,
        preliminary_signal: Signal,
        backtest_results: List[Dict[str, Any]],
        combined_bt_summary: str,
        ticker: Dict[str, Any],
        trading_paused: bool,
        strategy_model_type: str,
        effective_temp: float,
        llm_provider: Optional[str],
        llm_model: Optional[str],
        market_hash: str = None,
        is_critical: bool = False,
    ) -> Tuple[Signal, Optional[str], Optional[str]]:
        """Run the Step 2 LLM call and carry over execution-critical fields.

        Returns (final_signal, llm_provider, llm_model).
        """
        engine = self.engine

        if not backtest_results:
            logger.info(f"Insufficient data for any backtest for {symbol}. Using preliminary decision.")
            return preliminary_signal, llm_provider, llm_model

        # Build Step 2 prompt with ALL backtest results
        total_variants_proposed = len(preliminary_signal.backtest_variants) if preliminary_signal.backtest_variants else 1
        historical_bt_results = await asyncio.to_thread(
            get_backtest_results_for_symbol, symbol, assigned_tf, 10
        )
        step2_messages = build_final_decision_messages(
            symbol=symbol,
            ticker=ticker,
            preliminary_decision={
                "action": preliminary_signal.action,
                "confidence": preliminary_signal.confidence,
                "reasoning": preliminary_signal.reasoning,
                "strategy_params": preliminary_signal.strategy_params,
                "timeframe": assigned_tf,
            },
            backtest_results=backtest_results,
            base_currency=engine.base_currency,
            trading_paused=trading_paused,
            total_variants_proposed=total_variants_proposed,
            historical_backtest_results=historical_bt_results,
        )
        # Append position info if exists
        if symbol in engine.positions:
            pos = engine.positions[symbol]
            step2_messages[-1]["content"] += (
                f"\n**Existing Position:** You already hold {pos['amount']:.6f} "
                f"at entry {pos['price']:.4f}. A BUY will ADD to this position (scale in).\n"
            )

        # Call LLM for Step 2
        try:
            step2_result = await asyncio.wait_for(
                asyncio.to_thread(
                    get_cached_llm_response,
                    "", "", 60,
                    model_type=strategy_model_type,
                    temperature=effective_temp,
                    symbol=symbol,
                    market_hash=market_hash,
                    messages=step2_messages,
                    request_type="trading_decision_step2",
                ),
                timeout=settings.LLM_TIMEOUT
            )
            step2_response = step2_result["response"]
            llm_provider = step2_result["provider"]
            llm_model = step2_result["model"]
            logger.info(f"LLM Step 2 call completed for {symbol} (provider={llm_provider}, model={llm_model})")

            # Parse Step 2 response
            try:
                final_strategy = create_strategy_from_llm(step2_response)
            except ValueError:
                # Retry with correction prompt
                correction_content = (
                    "Your previous response was not valid JSON. "
                    "Output ONLY a single JSON object. "
                    "Here is the request:\n\n" + step2_messages[-1]["content"]
                )
                retry_messages = [
                    step2_messages[0],
                    {"role": "user", "content": correction_content},
                ]
                try:
                    retry_result = await asyncio.wait_for(
                        asyncio.to_thread(
                            get_cached_llm_response,
                            "", "", 30,
                            model_type="actuator",
                            temperature=effective_temp,
                            symbol=symbol,
                            market_hash=market_hash,
                            messages=retry_messages,
                            request_type="trading_decision_step2_retry",
                        ),
                        timeout=settings.LLM_TIMEOUT
                    )
                    final_strategy = create_strategy_from_llm(retry_result["response"])
                    step2_response = retry_result["response"]
                    llm_provider = retry_result["provider"]
                    llm_model = retry_result["model"]
                except Exception:
                    logger.error(f"Step 2 JSON parse retry failed for {symbol}. Using preliminary decision.")
                    final_strategy = None

            if final_strategy is not None:
                signal = final_strategy.generate_signal({})
                logger.info(f"Step 2 final decision for {symbol}: action={signal.action}, confidence={signal.confidence:.2f}")
                signal.model_type = strategy_model_type
                signal.llm_provider = llm_provider
                signal.llm_model = llm_model
                signal.backtest_summary = combined_bt_summary
            else:
                signal = preliminary_signal
                signal.backtest_summary = combined_bt_summary
            # Carry over execution-critical fields from Step 1 if not provided in Step 2
            if signal.action == "BUY":
                # Execution parameters
                if signal.entry_condition is None and preliminary_signal.entry_condition is not None:
                    signal.entry_condition = preliminary_signal.entry_condition
                if signal.order_type is None and preliminary_signal.order_type is not None:
                    signal.order_type = preliminary_signal.order_type
                if signal.limit_price is None and preliminary_signal.limit_price is not None:
                    signal.limit_price = preliminary_signal.limit_price
                if signal.stop_price is None and preliminary_signal.stop_price is not None:
                    signal.stop_price = preliminary_signal.stop_price
                if signal.stop_loss_order_type is None and preliminary_signal.stop_loss_order_type is not None:
                    signal.stop_loss_order_type = preliminary_signal.stop_loss_order_type
                if signal.stop_loss_stop_price is None and preliminary_signal.stop_loss_stop_price is not None:
                    signal.stop_loss_stop_price = preliminary_signal.stop_loss_stop_price
                if signal.stop_loss_limit_price is None and preliminary_signal.stop_loss_limit_price is not None:
                    signal.stop_loss_limit_price = preliminary_signal.stop_loss_limit_price
                if signal.stop_loss_trail_offset is None and preliminary_signal.stop_loss_trail_offset is not None:
                    signal.stop_loss_trail_offset = preliminary_signal.stop_loss_trail_offset
                if signal.take_profit_order_type is None and preliminary_signal.take_profit_order_type is not None:
                    signal.take_profit_order_type = preliminary_signal.take_profit_order_type
                if signal.take_profit_limit_price is None and preliminary_signal.take_profit_limit_price is not None:
                    signal.take_profit_limit_price = preliminary_signal.take_profit_limit_price
                if signal.trail_offset is None and preliminary_signal.trail_offset is not None:
                    signal.trail_offset = preliminary_signal.trail_offset
                # Risk parameters — carry over from Step 1 if missing in Step 2
                if signal.strategy_params:
                    prelim_params = preliminary_signal.strategy_params or {}
                    for risk_key in (
                        "cooldown_after_loss_seconds",
                        "max_hold_time_seconds",
                        "stop_loss_method",
                        "stop_loss_atr_multiple",
                        "trailing_stop_distance_pct",
                        "trailing_stop_atr_multiple",
                        "trailing_stop_activation_pct",
                        "partial_take_profit_levels",
                        "partial_take_profit_pct",
                        "partial_take_profit_fraction",
                        "breakeven_activation_pct",
                        "trailing_take_profit",
                        "trailing_take_profit_distance_pct",
                        "max_unrealized_loss_pct",
                        "news_sentiment_exit_threshold",
                        "max_risk_per_trade_pct",
                        "max_portfolio_risk_pct",
                        "min_profit_per_trade",
                        "min_risk_reward_ratio",
                        "min_confidence",
                        "position_size_multiplier",
                        "strategy_interval_seconds",
                        "backtest_period_days",
                        "order_fill_timeout_seconds",
                        "time_in_force",
                        "backtest_entry_config",
                    ):
                        if risk_key not in signal.strategy_params and risk_key in prelim_params:
                            signal.strategy_params[risk_key] = prelim_params[risk_key]
                else:
                    # Step 2 returned no params at all — use Step 1's params
                    signal.strategy_params = preliminary_signal.strategy_params
        except Exception as e:
            logger.error(f"LLM Step 2 call failed for {symbol}: {e}. Using preliminary decision.")
            signal = preliminary_signal
            signal.backtest_summary = combined_bt_summary
            # Preserve provider/model from Step 1b as fallback
            if llm_provider is None:
                llm_provider = preliminary_signal.llm_provider
            if llm_model is None:
                llm_model = preliminary_signal.llm_model

        return signal, llm_provider, llm_model

    async def run_backtest_and_final_decision(
        self,
        symbol: str,
        assigned_tf: str,
        tf_seconds: int,
        current_price: float,
        atr: Optional[float],
        historical_ohlcv: Optional[List[List]],
        raw_candles: Optional[List[List]],
        base_balance: float,
        is_btp: bool,
        trading_paused: bool,
        strategy_model_type: str,
        effective_temp: float,
        preliminary_signal: Signal,
        display_symbol: str,
        ticker: Dict[str, Any],
        market_hash: str = None,
        is_critical: bool = False,
    ) -> Tuple[Signal, str, Optional[str], Optional[str]]:
        """Run backtests and the Step 2 LLM call to produce the final signal.

        Returns (final_signal, combined_backtest_summary, llm_provider, llm_model).
        """
        engine = self.engine
        combined_bt_summary = ""
        llm_provider = None
        llm_model = None
        backtest_results = []

        if preliminary_signal.action in ("BUY", "HOLD"):
            variants_to_test = self._prepare_backtest_variants(
                symbol=symbol,
                preliminary_signal=preliminary_signal,
                historical_ohlcv=historical_ohlcv,
                raw_candles=raw_candles,
            )

            backtest_results = await self._run_backtest_variants_parallel(
                symbol=symbol,
                variants_to_test=variants_to_test,
                preliminary_signal=preliminary_signal,
                atr=atr,
                current_price=current_price,
                tf_seconds=tf_seconds,
                assigned_tf=assigned_tf,
                historical_ohlcv=historical_ohlcv,
                raw_candles=raw_candles,
                base_balance=base_balance,
                is_btp=is_btp,
            )

            # Log results after all variants complete
            for i, r in enumerate(backtest_results):
                if r["stats"]:
                    logger.info(f"Backtest variant {i+1}/{len(variants_to_test)} for {symbol}: {r['summary']}")
                else:
                    logger.info(f"Backtest variant {i+1}/{len(variants_to_test)} for {symbol}: insufficient data")

            # Build combined backtest summary for notifications
            summaries = []
            for i, r in enumerate(backtest_results):
                if r.get("stats"):
                    summaries.append(f"V{i+1}: {r['summary']}")
                elif any(br.get("stats") for br in backtest_results):
                    summaries.append(f"V{i+1}: skipped")
            
            if not summaries:
                combined_bt_summary = "Backtest skipped: insufficient data"
            else:
                combined_bt_summary = " | ".join(summaries)

            signal, llm_provider, llm_model = await self.run_step2_llm_call(
                symbol=symbol,
                assigned_tf=assigned_tf,
                preliminary_signal=preliminary_signal,
                backtest_results=backtest_results,
                combined_bt_summary=combined_bt_summary,
                ticker=ticker,
                trading_paused=trading_paused,
                strategy_model_type=strategy_model_type,
                effective_temp=effective_temp,
                llm_provider=llm_provider,
                llm_model=llm_model,
                market_hash=market_hash,
                is_critical=is_critical,
            )
        else:
            # For SELL or HOLD, no backtest needed, use preliminary decision
            signal = preliminary_signal

        # Store raw backtest stats dict on the signal for notification compaction
        if backtest_results and backtest_results[0].get("stats"):
            bt_stats = dict(backtest_results[0]["stats"])
            bt_stats["timeframe"] = assigned_tf
            signal.backtest_stats = bt_stats
        else:
            signal.backtest_stats = None

        return signal, combined_bt_summary, llm_provider, llm_model

    async def run_simulation_step2(
        self,
        symbol: str,
        data: Dict[str, Any],
        preliminary_signal: Signal,
        backtest_results: List[Dict[str, Any]],
        combined_bt_summary: str,
    ) -> Tuple[Optional[str], Optional[str], Optional[Signal], Optional[Dict[str, Any]]]:
        """Run the Step 2 LLM call for simulation.

        Returns (step2_response, error, final_signal, error_dict).
        If error_dict is not None, the caller should return it immediately.
        """
        engine = self.engine
        model_type = data.get("model_type", "mind")
        temperature = data.get("temperature", 0.2)

        total_variants_proposed = len(preliminary_signal.backtest_variants) if preliminary_signal.backtest_variants else 1
        step2_messages = build_final_decision_messages(
            symbol=symbol,
            ticker=data["ticker"],
            preliminary_decision={
                "action": preliminary_signal.action,
                "confidence": preliminary_signal.confidence,
                "reasoning": preliminary_signal.reasoning,
                "strategy_params": preliminary_signal.strategy_params,
                "timeframe": data["assigned_tf"],
            },
            backtest_results=backtest_results,
            base_currency=engine.base_currency,
            trading_paused=False,
            total_variants_proposed=total_variants_proposed,
            historical_backtest_results=data.get("historical_backtest_results"),
        )
        
        # Append position info if exists
        if symbol in engine.positions:
            pos = engine.positions[symbol]
            step2_messages[-1]["content"] += (
                f"\n**Existing Position:** You already hold {pos['amount']:.6f} "
                f"at entry {pos['price']:.4f}. A BUY will ADD to this position (scale in).\n"
            )

        try:
            step2_result = await asyncio.wait_for(
                asyncio.to_thread(
                    get_cached_llm_response,
                    "", "", 60,
                    model_type=model_type,
                    temperature=temperature,
                    symbol=symbol,
                    market_hash=data.get("market_hash"),
                    messages=step2_messages,
                    request_type="simulation_step2",
                ),
                timeout=settings.LLM_TIMEOUT
            )
            step2_response = step2_result["response"]
        except Exception as e:
            return None, f"LLM Step 2 call failed: {e}", None, {
                "step1_response": data.get("step1b_response"),
                "error": f"LLM Step 2 call failed: {e}",
                "action": preliminary_signal.action,
                "backtest_summary": combined_bt_summary,
            }

        # Parse Step 2 response to get the final action
        try:
            final_strategy = create_strategy_from_llm(step2_response)
        except ValueError:
            # Retry with correction prompt
            logger.warning(f"Simulation Step 2 parse failed for {symbol}. Retrying.")
            correction_content = (
                "Your previous response was not valid JSON. "
                "Output ONLY a single JSON object. "
                "Here is the request:\n\n" + step2_messages[-1]["content"]
            )
            retry_messages = [
                step2_messages[0],
                {"role": "user", "content": correction_content},
            ]
            try:
                retry_result = await asyncio.wait_for(
                    asyncio.to_thread(
                        get_cached_llm_response,
                        "", "", 30,
                        model_type="actuator",
                        temperature=temperature,
                        symbol=symbol,
                        market_hash=data.get("market_hash"),
                        messages=retry_messages,
                        request_type="simulation_step2_retry",
                    ),
                    timeout=settings.LLM_TIMEOUT
                )
                step2_response = retry_result["response"]
                final_strategy = create_strategy_from_llm(step2_response)
            except Exception as e2:
                return step2_response, f"Failed to parse LLM Step 2 response after retry: {e2}", None, {
                    "step1_response": data.get("step1b_response"),
                    "step2_response": step2_response,
                    "error": f"Failed to parse LLM Step 2 response after retry: {e2}",
                    "action": preliminary_signal.action,
                    "backtest_summary": combined_bt_summary,
                }

        final_signal = final_strategy.generate_signal({})
        return step2_response, None, final_signal, None

    async def run_simulation_backtests(
        self,
        symbol: str,
        data: Dict[str, Any],
        preliminary_signal: Signal,
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Run backtest variants for simulation and build combined summary.

        Returns (backtest_results, combined_bt_summary).
        """
        variants_to_test = self._prepare_backtest_variants(
            symbol=symbol,
            preliminary_signal=preliminary_signal,
            historical_ohlcv=data.get("historical_ohlcv"),
            raw_candles=data.get("raw_candles"),
        )

        backtest_results = await self._run_backtest_variants_parallel(
            symbol=symbol,
            variants_to_test=variants_to_test,
            preliminary_signal=preliminary_signal,
            atr=data["atr"],
            current_price=data["current_price"],
            tf_seconds=data["tf_seconds"],
            assigned_tf=data["assigned_tf"],
            historical_ohlcv=data["historical_ohlcv"],
            raw_candles=data["raw_candles"],
            base_balance=data["base_balance"],
            is_btp=data["is_btp"],
        )

        summaries = []
        for i, r in enumerate(backtest_results):
            if r.get("stats"):
                summaries.append(f"V{i+1}: {r['summary']}")
            elif any(br.get("stats") for br in backtest_results):
                summaries.append(f"V{i+1}: skipped")
        
        if not summaries:
            combined_bt_summary = "Backtest skipped: insufficient data"
        else:
            combined_bt_summary = " | ".join(summaries)

        return backtest_results, combined_bt_summary
