"""Simulation management for the TradingEngine."""
import asyncio
import json
import logging
from typing import Any, Dict, Optional, Tuple

from src.config.settings import settings
from src.llm.cache import get_cached_llm_response, get_cached_llm_response_async, compute_market_hash
from src.llm.prompts import compact_prompt, build_system_prompt, build_backtest_variants_prompt, build_analysis_messages, BacktestPromptData, StrategyPromptData
from src.llm.backtest_prompts import build_backtest_variants_messages
from src.strategies.llm_parser import create_strategy_from_llm
from src.trading.components.backtest_manager import _backtest_executor
from src.utils.macro_data import get_macro_economic_context

try:
    from src.news.fetcher import detect_upcoming_events
except ImportError:
    detect_upcoming_events = None

logger = logging.getLogger(__name__)


class SimulationManager:
    """Handles simulation data preparation and Step 1 LLM calls."""

    def __init__(self, signal_processor):
        self.sp = signal_processor
        self.engine = signal_processor.engine
        self.shared_state = self.engine.shared_state

    async def run_simulation_step1(
        self,
        symbol: str,
        data: Dict[str, Any],
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[Any], Optional[Dict[str, Any]]]:
        """Run Step 1a and Step 1b for simulation.

        Returns (analysis, step1b_response, preliminary_signal, error_dict).
        If error_dict is not None, the caller should return it immediately.
        """
        engine = self.engine
        model_type = data.get("model_type", "mind")
        temperature = data.get("temperature", 0.2)
        market_hash = data.get("market_hash")

        # Step 1a: Analysis
        try:
            step1a_result = await asyncio.wait_for(
                get_cached_llm_response_async(
                    "", "", 60,
                    market_hash=market_hash,
                    model_type=model_type,
                    temperature=temperature,
                    symbol=symbol,
                    messages=data.get("analysis_messages"),
                    request_type="simulation_step1a",
                ),
                timeout=settings.LLM_TIMEOUT
            )
            step1a_response = step1a_result["response"]
        except (asyncio.TimeoutError, ConnectionError, TimeoutError, OSError, ValueError, TypeError, RuntimeError, json.JSONDecodeError) as e:
            return None, None, None, {"error": f"LLM Step 1a call failed: {type(e).__name__}: {e}"}

        analysis = self.sp._parse_analysis_response(step1a_response)
        if analysis is None:
            return None, None, None, {"error": "Failed to parse Step 1a analysis response", "raw_response": step1a_response}

        # Step 1b: Backtest variants
        prompt_data = BacktestPromptData(
            symbol=symbol,
            analysis=analysis,
            ticker=data["ticker"],
            current_price=data["current_price"],
            atr=data["atr"],
            assigned_timeframe=data["assigned_tf"],
            base_currency=engine.base_currency,
            base_balance=data["base_balance"],
            per_symbol_budget=data["per_symbol_budget"],
            min_order_amount=data.get("min_order_amount"),
            min_order_cost=data.get("min_order_cost"),
            remaining_balance=data.get("remaining_balance"),
            portfolio_total_value=data.get("portfolio_total_value"),
            portfolio_exposure_pct=data.get("portfolio_exposure_pct"),
            portfolio_stop_risk_pct=data.get("portfolio_stop_risk_pct"),
            portfolio_available_capital=data.get("portfolio_available_capital"),
            max_portfolio_exposure_pct=data.get("max_portfolio_exposure_pct"),
            max_portfolio_stop_risk_pct=data.get("max_portfolio_stop_risk_pct"),
            global_risk_multiplier=data.get("global_risk_multiplier"),
            min_stop_atr_mult=data.get("min_stop_atr_mult", 1.0),
            min_hold_time_mult=data.get("min_hold_time_mult", 1.0),
            trading_paused=False,
            has_position=data.get("has_position", False),
            historical_backtest_results=data.get("historical_backtest_results"),
        )
        loop = asyncio.get_running_loop()
        variants_messages = await loop.run_in_executor(
            _backtest_executor,
            build_backtest_variants_messages,
            prompt_data
        )

        try:
            step1b_result = await asyncio.wait_for(
                get_cached_llm_response_async(
                    "", "", 60,
                    market_hash=compute_market_hash({"step": "1b", "analysis": analysis}),
                    model_type=model_type,
                    temperature=temperature,
                    symbol=symbol,
                    messages=variants_messages,
                    request_type="simulation_step1b",
                ),
                timeout=settings.LLM_TIMEOUT
            )
            step1b_response = step1b_result["response"]
        except (asyncio.TimeoutError, ConnectionError, TimeoutError, OSError, ValueError, TypeError, RuntimeError, json.JSONDecodeError) as e:
            return None, None, None, {"error": f"LLM Step 1b call failed: {type(e).__name__}: {e}"}

        try:
            preliminary_strategy = create_strategy_from_llm(step1b_response)
            preliminary_signal = preliminary_strategy.generate_signal({})
        except ValueError as e:
            return None, None, None, {"error": f"Failed to parse Step 1b response: {e}", "raw_response": step1b_response}

        return analysis, step1b_response, preliminary_signal, None

    async def prepare_simulation_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch all necessary data and build the strategy prompt for simulation."""
        engine = self.engine
        symbol_entry = next((e for e in self.shared_state.current_symbols if e["symbol"] == symbol), None)
        if not symbol_entry:
            return {"error": f"Symbol {symbol} not found in current_symbols"}

        assigned_tf = symbol_entry["timeframe"]

        symbol_data = await self.sp.fetch_symbol_market_data(symbol, assigned_tf)
        if symbol_data is None:
            return {"error": "No ticker data"}
        ticker = symbol_data["ticker"]
        current_price = symbol_data["current_price"]
        fundamentals = symbol_data["fundamentals"]
        balance = symbol_data["balance"]
        base_balance = symbol_data["base_balance"]
        ohlcv_data = symbol_data["ohlcv_data"]
        is_btp = symbol_data["is_btp"]
        tf_seconds = symbol_data["tf_seconds"]
        multi_tf_indicators = symbol_data["multi_tf_indicators"]
        multi_tf_raw_candles = symbol_data["multi_tf_raw_candles"]
        atr = symbol_data["atr"]
        rsi = symbol_data["rsi"]
        macd = symbol_data["macd"]
        macd_signal = symbol_data["macd_signal"]
        macd_hist = symbol_data["macd_hist"]
        bb_upper = symbol_data["bb_upper"]
        bb_middle = symbol_data["bb_middle"]
        bb_lower = symbol_data["bb_lower"]
        ema_9 = symbol_data["ema_9"]
        ema_21 = symbol_data["ema_21"]
        stochastic_k = symbol_data["stochastic_k"]
        stochastic_d = symbol_data["stochastic_d"]
        adx = symbol_data["adx"]
        plus_di = symbol_data["plus_di"]
        minus_di = symbol_data["minus_di"]
        obv = symbol_data["obv"]
        mfi = symbol_data["mfi"]
        cci = symbol_data["cci"]
        williams_r = symbol_data["williams_r"]
        ichimoku = symbol_data["ichimoku"]
        donchian_channels = symbol_data["donchian_channels"]
        parabolic_sar = symbol_data["parabolic_sar"]
        keltner_channels = symbol_data["keltner_channels"]
        vwap = symbol_data["vwap"]
        daily_pivot_points = symbol_data["daily_pivot_points"]
        per_symbol_budget = base_balance / engine.effective_max_symbols if engine.effective_max_symbols > 0 else 0.0

        open_positions = [pos for pos in self.shared_state.positions.values() if pos.get("symbol") == symbol]

        _ctx = await self.sp.gather_prompt_context(
            symbol=symbol,
            assigned_tf=assigned_tf,
            tf_seconds=tf_seconds,
            ticker=ticker,
            base_balance=base_balance,
            ohlcv_data=ohlcv_data,
            multi_tf_indicators=multi_tf_indicators,
            multi_tf_raw_candles=multi_tf_raw_candles,
            atr=atr,
            rsi=rsi,
            macd=macd,
            macd_signal=macd_signal,
            macd_hist=macd_hist,
            bb_upper=bb_upper,
            bb_middle=bb_middle,
            bb_lower=bb_lower,
            ema_9=ema_9,
            ema_21=ema_21,
            adx=adx,
            plus_di=plus_di,
            minus_di=minus_di,
        )
        atr_multi_tf = _ctx["atr_multi_tf"]
        atr_percentile = _ctx["atr_percentile"]
        market_regime = _ctx["market_regime"]
        raw_candles = _ctx["raw_candles"]
        historical_ohlcv = _ctx["historical_ohlcv"]
        unrealized_pnl = _ctx["unrealized_pnl"]
        position_info = _ctx["position_info"]
        recent_trades_summary = _ctx["recent_trades_summary"]
        min_order_amount = _ctx["min_order_amount"]
        min_order_cost = _ctx["min_order_cost"]
        past_trades = _ctx["past_trades"]
        aggregate_sentiment = _ctx["aggregate_sentiment"]
        sentiment_trend_val = _ctx["sentiment_trend_val"]
        volume_trend_val = _ctx["volume_trend_val"]
        full_market_breadth = _ctx["full_market_breadth"]
        session_info = _ctx["session_info"]
        minutes_to_market_close = _ctx["minutes_to_market_close"]
        global_risk_mult = _ctx["global_risk_mult"]
        max_port_exp = _ctx["max_port_exp"]
        max_port_risk = _ctx["max_port_risk"]
        partial_tp_executed_levels = _ctx["partial_tp_executed_levels"]
        sim_min_stop_atr_mult = _ctx["min_stop_atr_mult"]
        sim_min_hold_time_mult = _ctx["min_hold_time_mult"]
        historical_backtest_results = _ctx["historical_backtest_results"]

        # --- Emulate _process_symbol context for portfolio exposure ---
        _portfolio = await engine._position_manager.compute_portfolio_exposure_summary(base_balance)
        portfolio_total_value = _portfolio["portfolio_total_value"]
        portfolio_exposure = _portfolio["portfolio_exposure"]
        portfolio_stop_risk = _portfolio["portfolio_stop_risk"]
        portfolio_exposure_pct = _portfolio["portfolio_exposure_pct"]
        portfolio_stop_risk_pct = _portfolio["portfolio_stop_risk_pct"]
        portfolio_available_capital = _portfolio["portfolio_available_capital"]

        # --- Read review limits and position flags (no side effects for simulation) ---
        max_sl_reviews = settings.MAX_STOP_LOSS_REVIEWS
        max_tp_reviews = settings.MAX_TAKE_PROFIT_REVIEWS
        max_partial_tp_reviews = settings.MAX_PARTIAL_TP_REVIEWS
        max_dust_sweep_reviews = settings.MAX_DUST_SWEEP_REVIEWS
        try:
            raw = await engine.config_service.get_config("max_stop_loss_reviews")
            if raw: max_sl_reviews = int(raw)
            raw = await engine.config_service.get_config("max_take_profit_reviews")
            if raw: max_tp_reviews = int(raw)
            raw = await engine.config_service.get_config("max_partial_tp_reviews")
            if raw: max_partial_tp_reviews = int(raw)
            raw = await engine.config_service.get_config("max_dust_sweep_reviews")
            if raw: max_dust_sweep_reviews = int(raw)
        except (ValueError, TypeError, ConnectionError, TimeoutError, OSError):
            pass

        if tf_seconds >= settings.LONG_TERM_TF_SECONDS:
            max_sl_reviews = min(max_sl_reviews, settings.LONG_TERM_MAX_STOP_LOSS_REVIEWS)
        elif tf_seconds >= 604_800:
            max_sl_reviews = min(max_sl_reviews, settings.WEEKLY_MAX_STOP_LOSS_REVIEWS)

        trading_paused = False  # Force False for simulation

        max_hold_expired = False
        max_hold_expired_count = 0
        stop_loss_triggered = False
        stop_loss_review_count = 0
        take_profit_triggered = False
        take_profit_review_count = 0
        partial_tp_triggered = False
        partial_tp_review_count = 0
        partial_tp_triggered_levels = []
        dust_sweep_triggered = False
        dust_sweep_review_count = 0
        if symbol in self.shared_state.positions:
            pos = self.shared_state.positions[symbol]
            max_hold_expired = pos.get("_max_hold_expired", False)
            max_hold_expired_count = pos.get("_max_hold_expired_count", 0)
            stop_loss_triggered = pos.get("_stop_loss_triggered", False)
            stop_loss_review_count = pos.get("_stop_loss_review_count", 0)
            take_profit_triggered = pos.get("_take_profit_triggered", False)
            take_profit_review_count = pos.get("_take_profit_review_count", 0)
            partial_tp_triggered = pos.get("_partial_tp_triggered", False) or pos.get("_partial_tp_triggered_single", False)
            partial_tp_review_count = pos.get("_partial_tp_review_count", 0) or pos.get("_partial_tp_single_review_count", 0)
            partial_tp_triggered_levels = pos.get("_partial_tp_triggered_levels", [])
            dust_sweep_triggered = pos.get("_dust_sweep_triggered", False)
            dust_sweep_review_count = pos.get("_dust_sweep_review_count", 0)

        min_viable_amount = settings.MIN_VIABLE_TRADE_AMOUNT
        try:
            raw = await engine.config_service.get_config("min_viable_trade_amount")
            if raw: min_viable_amount = float(raw)
        except (ValueError, TypeError, ConnectionError, TimeoutError, OSError): pass

        remaining = max(0.0, base_balance - self.shared_state._cycle_spent)

        perf = await engine.event_bus.request("compute_performance_metrics")
        trade_pattern_analysis = await engine.event_bus.request("compute_trade_pattern_analysis")

        symbol_event = None
        if settings.NEWS_ENABLED and detect_upcoming_events is not None:
            try:
                loop = asyncio.get_running_loop()
                symbol_event = await loop.run_in_executor(
                    None,  # Use the default executor (now 100 workers)
                    detect_upcoming_events,
                    symbol
                )
            except (ConnectionError, TimeoutError, OSError, ValueError, TypeError, json.JSONDecodeError) as e:
                logger.debug(f"prepare_simulation_data: failed to detect events for {symbol}: {type(e).__name__}: {e}")

        # Pre-summarize news for the prompt to avoid synchronous LLM calls in prompt builder
        news_section = None
        if settings.NEWS_ENABLED:
            try:
                from src.llm.prompt_utils import get_cached_news_summary_async
                news_summary = await get_cached_news_summary_async(symbol, model_type="weak")
                if news_summary and news_summary.get("summary") and news_summary["summary"] != "No recent news.":
                    news_section = f"Recent news summary for {symbol}: {news_summary['summary']}"
            except Exception as e:
                logger.warning(f"Failed to pre-summarize news for {symbol}: {e}")

        prompt_data = StrategyPromptData(
            symbol=symbol,
            ticker=ticker,
            balance=balance,
            open_positions=open_positions,
            per_symbol_budget=per_symbol_budget,
            max_symbols=engine.effective_max_symbols,
            base_currency=engine.base_currency,
            performance=perf,
            ohlcv_data=ohlcv_data,
            assigned_timeframe=assigned_tf,
            atr=atr,
            atr_multi_tf=atr_multi_tf,
            rsi=rsi,
            macd=macd,
            macd_signal=macd_signal,
            macd_hist=macd_hist,
            bb_upper=bb_upper,
            bb_middle=bb_middle,
            bb_lower=bb_lower,
            ema_9=ema_9,
            ema_21=ema_21,
            stochastic_k=stochastic_k,
            stochastic_d=stochastic_d,
            adx=adx,
            plus_di=plus_di,
            minus_di=minus_di,
            obv=obv,
            mfi=mfi,
            cci=cci,
            williams_r=williams_r,
            ichimoku=ichimoku,
            donchian_channels=donchian_channels,
            parabolic_sar=parabolic_sar,
            keltner_channels=keltner_channels,
            vwap=vwap,
            daily_pivot_points=daily_pivot_points,
            unrealized_pnl=unrealized_pnl,
            position_info=position_info,
            drawdown_pct=perf.get("equity_curve", {}).get("drawdown_pct"),
            raw_candles=raw_candles,
            recent_trades=recent_trades_summary,
            historical_ohlcv=historical_ohlcv,
            min_order_amount=min_order_amount,
            min_order_cost=min_order_cost,
            all_symbols=self.shared_state.current_symbols,
            past_trades=past_trades,
            cycle_spent=self.shared_state._cycle_spent,
            remaining_balance=remaining,
            market_regime=market_regime,
            multi_tf_raw_candles=multi_tf_raw_candles,
            multi_tf_indicators=multi_tf_indicators,
            session_info=session_info,
            sentiment_trend=sentiment_trend_val,
            volume_trend=volume_trend_val,
            market_breadth=self.shared_state._market_breadth,
            full_market_breadth=full_market_breadth,
            atr_percentile=atr_percentile,
            global_risk_multiplier=global_risk_mult,
            trading_paused=trading_paused,
            max_hold_expired=max_hold_expired,
            max_hold_expired_count=max_hold_expired_count,
            stop_loss_triggered=stop_loss_triggered,
            stop_loss_review_count=stop_loss_review_count,
            take_profit_triggered=take_profit_triggered,
            take_profit_review_count=take_profit_review_count,
            partial_tp_triggered=partial_tp_triggered,
            partial_tp_review_count=partial_tp_review_count,
            partial_tp_triggered_levels=partial_tp_triggered_levels if partial_tp_triggered_levels else None,
            partial_tp_executed_levels=partial_tp_executed_levels,
            dust_sweep_triggered=dust_sweep_triggered,
            dust_sweep_review_count=dust_sweep_review_count,
            max_stop_loss_reviews=max_sl_reviews,
            max_take_profit_reviews=max_tp_reviews,
            max_partial_tp_reviews=max_partial_tp_reviews,
            max_dust_sweep_reviews=max_dust_sweep_reviews,
            portfolio_exposure_pct=portfolio_exposure_pct,
            portfolio_stop_risk_pct=portfolio_stop_risk_pct,
            portfolio_total_value=portfolio_total_value,
            portfolio_open_count=len(self.shared_state.positions),
            portfolio_available_capital=portfolio_available_capital,
            last_decision=self.shared_state._last_decisions.get(symbol),
            minutes_to_market_close=minutes_to_market_close,
            current_strategy_interval_seconds=self.shared_state._strategy_intervals.get(symbol, engine._timeframe_to_seconds(assigned_tf)),
            max_portfolio_exposure_pct=max_port_exp,
            max_portfolio_stop_risk_pct=max_port_risk,
            trade_pattern_analysis=trade_pattern_analysis,
            symbol_event=symbol_event,
            queued_orders=self.shared_state.queued_orders,
            fundamentals=fundamentals,
            min_hold_time_mult=sim_min_hold_time_mult,
            min_stop_atr_mult=sim_min_stop_atr_mult,
            min_viable_trade_amount=min_viable_amount,
            historical_backtest_results=historical_backtest_results,
            aggregate_sentiment=aggregate_sentiment,
            news_section=news_section,
            macro_economic_context=get_macro_economic_context(),
        )
        analysis_prompt, market_snapshot, market_hash = await self.sp.build_analysis_prompt_and_snapshot(prompt_data)
        analysis_messages = build_analysis_messages(prompt_data)

        strategy_model_type, effective_temp = self.sp.model_tier_manager.compute_model_tier_and_temperature(
            atr=atr,
            atr_percentile=atr_percentile,
            rsi=rsi,
            macd=macd,
            macd_signal=macd_signal,
            macd_hist=macd_hist,
            bb_upper=bb_upper,
            bb_middle=bb_middle,
            bb_lower=bb_lower,
            ema_9=ema_9,
            ema_21=ema_21,
            stochastic_k=stochastic_k,
            adx=adx,
            plus_di=plus_di,
            minus_di=minus_di,
            mfi=mfi,
            cci=cci,
            williams_r=williams_r,
            ichimoku=ichimoku,
            market_regime=market_regime,
            market_breadth=self.shared_state._market_breadth,
            full_market_breadth=full_market_breadth,
            sentiment_trend_val=sentiment_trend_val,
            volume_trend_val=volume_trend_val,
            unrealized_pnl=unrealized_pnl,
            drawdown_pct=perf.get("equity_curve", {}).get("drawdown_pct"),
            portfolio_exposure_pct=portfolio_exposure_pct,
            portfolio_stop_risk_pct=portfolio_stop_risk_pct,
            is_critical=(max_hold_expired or stop_loss_triggered or take_profit_triggered or partial_tp_triggered or dust_sweep_triggered),
            trading_paused=trading_paused,
            symbol_event=symbol_event,
            fundamentals=fundamentals,
            consecutive_losses=perf.get("equity_curve", {}).get("consecutive_losses", 0),
            current_price=current_price,
            timeframe=assigned_tf,
            num_candidates=len(self.shared_state.current_symbols),
        )

        return {
            "ticker": ticker, "analysis_prompt": analysis_prompt, "atr": atr, "assigned_tf": assigned_tf,
            "tf_seconds": tf_seconds, "historical_ohlcv": historical_ohlcv,
            "raw_candles": raw_candles, "current_price": current_price,
            "base_balance": base_balance, "is_btp": is_btp,
            "model_type": strategy_model_type, "temperature": effective_temp,
            "market_hash": market_hash,
            "per_symbol_budget": per_symbol_budget,
            "min_order_amount": min_order_amount,
            "min_order_cost": min_order_cost,
            "remaining_balance": remaining,
            "portfolio_total_value": portfolio_total_value,
            "portfolio_exposure_pct": portfolio_exposure_pct,
            "portfolio_stop_risk_pct": portfolio_stop_risk_pct,
            "portfolio_available_capital": portfolio_available_capital,
            "max_portfolio_exposure_pct": max_port_exp,
            "max_portfolio_stop_risk_pct": max_port_risk,
            "global_risk_multiplier": global_risk_mult,
            "min_stop_atr_mult": sim_min_stop_atr_mult,
            "min_hold_time_mult": sim_min_hold_time_mult,
            "has_position": symbol in self.shared_state.positions,
            "historical_backtest_results": historical_backtest_results,
            "analysis_messages": analysis_messages,
        }
