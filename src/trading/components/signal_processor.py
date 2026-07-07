"""Signal processing component for the TradingEngine.

Handles per-symbol LLM orchestration, backtesting, validation, and execution.
Extracted from TradingEngine to reduce class size and improve maintainability.
"""
import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from src.config.settings import settings
from src.database import get_latest_ohlcv_timestamp, get_ohlcv, get_indicators, get_backtest_results_for_symbol, get_indicators_for_symbols, get_aggregate_sentiment_from_db, insert_signal
from src.exchanges.yahoo_finance import get_yahoo_quote, get_yahoo_fundamentals
from src.indicators import compute_all_indicators, compute_ema, compute_vwap, compute_pivot_points
from src.llm.cache import get_cached_llm_response, compute_market_hash
from src.llm.prompts import build_analysis_prompt, compact_prompt, build_backtest_variants_prompt, build_system_prompt, get_cached_news_summary, StrategyPromptData, BacktestPromptData
from src.strategies.base import Signal
from src.strategies.llm_parser import create_strategy_from_llm, LLMStrategy
from src.strategies.validator import validate_signal
from src.utils.symbol_utils import is_btp_isin

try:
    from src.news.fetcher import detect_upcoming_events
except ImportError:
    detect_upcoming_events = None

logger = logging.getLogger(__name__)


@dataclass
class DecisionContext:
    symbol: str
    display_symbol: str
    stock_name: str
    assigned_tf: str
    tf_seconds: int
    ticker: Dict[str, Any]
    signal: Signal
    llm_provider: str
    llm_model: str
    trading_paused: bool
    base_balance: float
    current_price: float
    atr: Optional[float]
    rsi: Optional[float]
    macd: Optional[float]
    macd_signal: Optional[float]
    macd_hist: Optional[float]
    bb_upper: Optional[float]
    bb_middle: Optional[float]
    bb_lower: Optional[float]
    ema_9: Optional[float]
    ema_21: Optional[float]
    stochastic_k: Optional[float]
    stochastic_d: Optional[float]
    adx: Optional[float]
    plus_di: Optional[float]
    minus_di: Optional[float]
    obv: Optional[float]
    mfi: Optional[float]
    cci: Optional[float]
    williams_r: Optional[float]
    ichimoku: Optional[Dict[str, Any]]
    donchian_channels: Optional[Dict[str, Any]]
    parabolic_sar: Optional[float]
    keltner_channels: Optional[Dict[str, Any]]
    aggregate_sentiment: Optional[Dict[str, Any]]
    market_regime: str
    min_stop_atr_mult: float
    min_hold_time_mult: float
    global_min_rr: Optional[float]
    max_hold_expired: bool
    stop_loss_triggered: bool
    take_profit_triggered: bool
    partial_tp_triggered: bool
    dust_sweep_triggered: bool
    strategy_model_type: str


class SignalProcessor:
    """Handles per-symbol signal processing for the TradingEngine."""

    def __init__(self, engine, event_bus):
        self.engine = engine
        self.event_bus = event_bus
        self.event_bus.subscribe("process_symbol", self.process_symbol)
        self.event_bus.subscribe("check_pause_resume_decision", self.check_pause_resume_decision)
        self.event_bus.subscribe("detect_entry_signal", self.detect_entry_signal)
        self.event_bus.subscribe("process_pending_entry", self.process_pending_entry)
        self.event_bus.subscribe("check_entry_condition_once", self.check_entry_condition_once)

    @staticmethod
    def _parse_analysis_response(response: str) -> Optional[Dict[str, Any]]:
        """Parse the Step 1a analysis LLM response into a dict.

        Expected fields: action, confidence, reasoning, strategy_direction.
        Returns None if parsing fails.
        """
        try:
            parsed = json.loads(response)
            if not isinstance(parsed, dict):
                return None
            action = parsed.get("action", "").upper()
            if action not in ("BUY", "SELL", "HOLD"):
                return None
            return {
                "action": action,
                "confidence": float(parsed.get("confidence", 0.0)),
                "reasoning": parsed.get("reasoning", ""),
                "strategy_direction": parsed.get("strategy_direction", ""),
            }
        except (json.JSONDecodeError, TypeError, ValueError):
            return None

    async def _get_initial_context(self, symbol: str, symbol_entry: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """Fetch initial context: display symbol, min viable amount, and position flags."""
        engine = self.engine
        stock_name = await engine._get_stock_name(symbol)
        display_symbol = engine._format_symbol_display(symbol, stock_name, symbol_entry["timeframe"])

        min_viable_amount = settings.MIN_VIABLE_TRADE_AMOUNT
        try:
            raw = await engine.config_service.get_config("min_viable_trade_amount")
            if raw:
                min_viable_amount = float(raw)
        except (ValueError, TypeError, ConnectionError, TimeoutError, OSError):
            pass

        _flags = await self.read_position_trigger_flags(symbol, symbol_entry)
        if _flags is None:
            return None

        return {
            "stock_name": stock_name,
            "display_symbol": display_symbol,
            "min_viable_amount": min_viable_amount,
            "flags": _flags,
        }

    async def _fetch_and_validate_data(self, symbol: str, symbol_entry: Dict[str, str], display_symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch market data and check skip conditions. Returns symbol_data or None if skipped."""
        engine = self.engine
        assigned_tf = symbol_entry["timeframe"]
        symbol_data = await self.fetch_symbol_market_data(symbol, assigned_tf)
        if symbol_data is None:
            logger.warning(f"No ticker data for {symbol}, skipping.")
            return None

        ticker = symbol_data["ticker"]
        base_balance = symbol_data["base_balance"]
        ohlcv_data = symbol_data["ohlcv_data"]
        has_position = symbol in engine.positions

        if await self.check_skip_conditions(symbol, display_symbol, ticker, assigned_tf, has_position, base_balance):
            return None
        if base_balance <= 0:
            logger.info(f"Evaluating {symbol} for position management only (base_balance={base_balance:.2f}, no new capital available).")
        if await self.check_no_ohlcv(symbol, display_symbol, assigned_tf, ohlcv_data):
            return None

        return symbol_data

    async def _gather_and_build_prompt(self, symbol: str, symbol_entry: Dict[str, str], symbol_data: Dict[str, Any], min_viable_amount: float, flags: Dict[str, Any]) -> Dict[str, Any]:
        """Gather prompt context and build the analysis prompt. Returns a context dict."""
        engine = self.engine
        assigned_tf = symbol_entry["timeframe"]
        tf_seconds = engine._timeframe_to_seconds(assigned_tf)

        ticker = symbol_data["ticker"]
        current_price = symbol_data["current_price"]
        base_balance = symbol_data["base_balance"]
        ohlcv_data = symbol_data["ohlcv_data"]

        open_positions = [pos for pos in engine.positions.values() if pos.get("symbol") == symbol]
        per_symbol_budget = base_balance / engine.effective_max_symbols if engine.effective_max_symbols > 0 else 0.0

        perf = await engine._compute_performance_metrics()
        trade_pattern_analysis = await engine._compute_trade_pattern_analysis()

        symbol_event = None
        if settings.NEWS_ENABLED and detect_upcoming_events is not None:
            try:
                symbol_event = await asyncio.to_thread(detect_upcoming_events, symbol)
            except (ConnectionError, TimeoutError, OSError, ValueError, TypeError, json.JSONDecodeError):
                pass

        _ctx = await self.gather_prompt_context(
            symbol=symbol, assigned_tf=assigned_tf, tf_seconds=tf_seconds, ticker=ticker,
            base_balance=base_balance, ohlcv_data=ohlcv_data,
            multi_tf_indicators=symbol_data["multi_tf_indicators"],
            multi_tf_raw_candles=symbol_data["multi_tf_raw_candles"],
            atr=symbol_data["atr"], rsi=symbol_data["rsi"], macd=symbol_data["macd"],
            macd_signal=symbol_data["macd_signal"], macd_hist=symbol_data["macd_hist"],
            bb_upper=symbol_data["bb_upper"], bb_middle=symbol_data["bb_middle"], bb_lower=symbol_data["bb_lower"],
            ema_9=symbol_data["ema_9"], ema_21=symbol_data["ema_21"],
            adx=symbol_data["adx"], plus_di=symbol_data["plus_di"], minus_di=symbol_data["minus_di"],
        )

        _portfolio = await engine._position_manager.compute_portfolio_exposure_summary(base_balance)
        async with engine._cycle_spent_lock:
            remaining = max(0.0, base_balance - engine._cycle_spent)

        prompt_data = StrategyPromptData(
            symbol=symbol,
            ticker=ticker,
            balance=symbol_data["balance"],
            open_positions=open_positions,
            per_symbol_budget=per_symbol_budget,
            max_symbols=engine.effective_max_symbols,
            base_currency=engine.base_currency,
            performance=perf,
            ohlcv_data=ohlcv_data,
            assigned_timeframe=assigned_tf,
            atr=symbol_data["atr"],
            atr_multi_tf=_ctx["atr_multi_tf"],
            rsi=symbol_data["rsi"],
            macd=symbol_data["macd"],
            macd_signal=symbol_data["macd_signal"],
            macd_hist=symbol_data["macd_hist"],
            bb_upper=symbol_data["bb_upper"],
            bb_middle=symbol_data["bb_middle"],
            bb_lower=symbol_data["bb_lower"],
            ema_9=symbol_data["ema_9"],
            ema_21=symbol_data["ema_21"],
            stochastic_k=symbol_data["stochastic_k"],
            stochastic_d=symbol_data["stochastic_d"],
            adx=symbol_data["adx"],
            plus_di=symbol_data["plus_di"],
            minus_di=symbol_data["minus_di"],
            obv=symbol_data["obv"],
            mfi=symbol_data["mfi"],
            cci=symbol_data["cci"],
            williams_r=symbol_data["williams_r"],
            ichimoku=symbol_data["ichimoku"],
            donchian_channels=symbol_data["donchian_channels"],
            parabolic_sar=symbol_data["parabolic_sar"],
            keltner_channels=symbol_data["keltner_channels"],
            vwap=symbol_data["vwap"],
            daily_pivot_points=symbol_data["daily_pivot_points"],
            unrealized_pnl=_ctx["unrealized_pnl"],
            position_info=_ctx["position_info"],
            drawdown_pct=perf.get("equity_curve", {}).get("drawdown_pct"),
            raw_candles=_ctx["raw_candles"],
            recent_trades=_ctx["recent_trades_summary"],
            historical_ohlcv=_ctx["historical_ohlcv"],
            min_order_amount=_ctx["min_order_amount"],
            min_order_cost=_ctx["min_order_cost"],
            all_symbols=engine.current_symbols,
            past_trades=_ctx["past_trades"],
            cycle_spent=engine._cycle_spent,
            remaining_balance=remaining,
            market_regime=_ctx["market_regime"],
            multi_tf_raw_candles=symbol_data["multi_tf_raw_candles"],
            multi_tf_indicators=symbol_data["multi_tf_indicators"],
            session_info=_ctx["session_info"],
            sentiment_trend=_ctx["sentiment_trend_val"],
            volume_trend=_ctx["volume_trend_val"],
            market_breadth=getattr(engine, '_market_breadth', None),
            full_market_breadth=_ctx["full_market_breadth"],
            atr_percentile=_ctx["atr_percentile"],
            global_risk_multiplier=_ctx["global_risk_mult"],
            trading_paused=False,
            max_hold_expired=flags["max_hold_expired"],
            max_hold_expired_count=flags["max_hold_expired_count"],
            stop_loss_triggered=flags["stop_loss_triggered"],
            stop_loss_review_count=flags["stop_loss_review_count"],
            take_profit_triggered=flags["take_profit_triggered"],
            take_profit_review_count=flags["take_profit_review_count"],
            partial_tp_triggered=flags["partial_tp_triggered"],
            partial_tp_review_count=flags["partial_tp_review_count"],
            partial_tp_triggered_levels=flags["partial_tp_triggered_levels"] if flags["partial_tp_triggered_levels"] else None,
            partial_tp_executed_levels=_ctx["partial_tp_executed_levels"],
            dust_sweep_triggered=flags["dust_sweep_triggered"],
            dust_sweep_review_count=flags["dust_sweep_review_count"],
            max_stop_loss_reviews=flags["max_sl_reviews_prompt"],
            max_take_profit_reviews=flags["max_tp_reviews_prompt"],
            max_partial_tp_reviews=flags["max_partial_tp_reviews_prompt"],
            max_dust_sweep_reviews=flags["max_dust_sweep_reviews_prompt"],
            portfolio_exposure_pct=_portfolio["portfolio_exposure_pct"],
            portfolio_stop_risk_pct=_portfolio["portfolio_stop_risk_pct"],
            portfolio_total_value=_portfolio["portfolio_total_value"],
            portfolio_open_count=len(engine.positions),
            portfolio_available_capital=_portfolio["portfolio_available_capital"],
            last_decision=engine._last_decisions.get(symbol),
            minutes_to_market_close=_ctx["minutes_to_market_close"],
            current_strategy_interval_seconds=engine._strategy_intervals.get(symbol, engine._timeframe_to_seconds(assigned_tf)),
            max_portfolio_exposure_pct=_ctx["max_port_exp"],
            max_portfolio_stop_risk_pct=_ctx["max_port_risk"],
            trade_pattern_analysis=trade_pattern_analysis,
            symbol_event=symbol_event,
            queued_orders=engine.queued_orders,
            fundamentals=symbol_data["fundamentals"],
            min_hold_time_mult=_ctx["min_hold_time_mult"],
            min_stop_atr_mult=_ctx["min_stop_atr_mult"],
            min_viable_trade_amount=min_viable_amount,
            historical_backtest_results=_ctx["historical_backtest_results"],
            aggregate_sentiment=_ctx["aggregate_sentiment"],
        )
        analysis_prompt, market_snapshot, market_hash = await self.build_analysis_prompt_and_snapshot(prompt_data)

        return {
            "ticker": ticker, "current_price": current_price, "base_balance": base_balance,
            "atr": symbol_data["atr"], "rsi": symbol_data["rsi"], "macd": symbol_data["macd"],
            "macd_signal": symbol_data["macd_signal"], "macd_hist": symbol_data["macd_hist"],
            "bb_upper": symbol_data["bb_upper"], "bb_middle": symbol_data["bb_middle"], "bb_lower": symbol_data["bb_lower"],
            "ema_9": symbol_data["ema_9"], "ema_21": symbol_data["ema_21"],
            "stochastic_k": symbol_data["stochastic_k"], "stochastic_d": symbol_data["stochastic_d"],
            "adx": symbol_data["adx"], "plus_di": symbol_data["plus_di"], "minus_di": symbol_data["minus_di"],
            "obv": symbol_data["obv"], "mfi": symbol_data["mfi"], "cci": symbol_data["cci"], "williams_r": symbol_data["williams_r"],
            "ichimoku": symbol_data["ichimoku"], "donchian_channels": symbol_data["donchian_channels"],
            "parabolic_sar": symbol_data["parabolic_sar"], "keltner_channels": symbol_data["keltner_channels"],
            "aggregate_sentiment": _ctx["aggregate_sentiment"], "market_regime": _ctx["market_regime"],
            "min_stop_atr_mult": _ctx["min_stop_atr_mult"], "min_hold_time_mult": _ctx["min_hold_time_mult"],
            "global_min_rr": _ctx["global_min_rr"],
            "analysis_prompt": analysis_prompt, "market_snapshot": market_snapshot, "market_hash": market_hash,
            "historical_ohlcv": _ctx["historical_ohlcv"], "raw_candles": _ctx["raw_candles"],
            "is_btp": symbol_data["is_btp"], "portfolio_total_value": _portfolio["portfolio_total_value"],
            "portfolio_exposure_pct": _portfolio["portfolio_exposure_pct"], "portfolio_stop_risk_pct": _portfolio["portfolio_stop_risk_pct"],
            "portfolio_available_capital": _portfolio["portfolio_available_capital"], "remaining": remaining,
            "per_symbol_budget": per_symbol_budget, "min_order_amount": _ctx["min_order_amount"],
            "min_order_cost": _ctx["min_order_cost"], "max_port_exp": _ctx["max_port_exp"], "max_port_risk": _ctx["max_port_risk"],
            "global_risk_mult": _ctx["global_risk_mult"], "historical_backtest_results": _ctx["historical_backtest_results"],
            "sentiment_trend_val": _ctx["sentiment_trend_val"],
            "volume_trend_val": _ctx["volume_trend_val"],
            "unrealized_pnl": _ctx["unrealized_pnl"],
            "drawdown_pct": perf.get("equity_curve", {}).get("drawdown_pct"),
            "full_market_breadth": _ctx["full_market_breadth"],
            "symbol_event": symbol_event,
            "consecutive_losses": perf.get("equity_curve", {}).get("consecutive_losses", 0),
            "atr_percentile": _ctx["atr_percentile"],
            "fundamentals": symbol_data["fundamentals"],
        }

    async def _check_skip_and_model_tier(self, symbol: str, ctx: Dict[str, Any], flags: Dict[str, Any]) -> Optional[Tuple[str, float]]:
        """Check if LLM eval should be skipped and compute model tier. Returns (model_type, temp) or None if skipped."""
        engine = self.engine
        is_critical = flags["max_hold_expired"] or flags["stop_loss_triggered"] or flags["take_profit_triggered"] or flags["partial_tp_triggered"] or flags["dust_sweep_triggered"]
        has_position = symbol in engine.positions

        if await self.should_skip_llm_eval(
            symbol=symbol, current_price=ctx["current_price"], atr=ctx["atr"], rsi=ctx["rsi"],
            macd_hist=ctx["macd_hist"], atr_percentile=ctx.get("atr_percentile"), market_regime=ctx["market_regime"],
            sentiment_trend_val=ctx.get("sentiment_trend_val"), timeframe_seconds=engine._timeframe_to_seconds(ctx.get("assigned_tf", "1d")),
            has_position=has_position, is_critical=is_critical,
        ):
            logger.info(f"Skipping LLM for {symbol}: market unchanged, no strong signals.")
            async with engine._eval_state_lock:
                engine._force_eval.pop(symbol, None)
            return None

        strategy_model_type, effective_temp = self.compute_model_tier_and_temperature(
            atr=ctx["atr"], atr_percentile=ctx.get("atr_percentile"), rsi=ctx["rsi"], macd=ctx["macd"],
            macd_signal=ctx["macd_signal"], macd_hist=ctx["macd_hist"], bb_upper=ctx["bb_upper"], bb_middle=ctx["bb_middle"], bb_lower=ctx["bb_lower"],
            ema_9=ctx["ema_9"], ema_21=ctx["ema_21"], stochastic_k=ctx["stochastic_k"], adx=ctx["adx"], plus_di=ctx["plus_di"], minus_di=ctx["minus_di"],
            mfi=ctx["mfi"], cci=ctx["cci"], williams_r=ctx["williams_r"], ichimoku=ctx["ichimoku"], market_regime=ctx["market_regime"],
            market_breadth=getattr(engine, '_market_breadth', None), full_market_breadth=ctx.get("full_market_breadth"),
            sentiment_trend_val=ctx.get("sentiment_trend_val"), volume_trend_val=ctx.get("volume_trend_val"),
            unrealized_pnl=ctx.get("unrealized_pnl"), drawdown_pct=ctx.get("drawdown_pct"),
            portfolio_exposure_pct=ctx["portfolio_exposure_pct"], portfolio_stop_risk_pct=ctx["portfolio_stop_risk_pct"],
            is_critical=is_critical, trading_paused=False, symbol_event=ctx.get("symbol_event"),
            fundamentals=ctx.get("fundamentals"), consecutive_losses=ctx.get("consecutive_losses", 0),
            current_price=ctx["current_price"], num_candidates=len(engine.current_symbols),
        )
        return strategy_model_type, effective_temp

    async def _run_llm_steps(self, symbol: str, display_symbol: str, symbol_entry: Dict[str, str], ctx: Dict[str, Any], strategy_model_type: str, effective_temp: float, flags: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Run Step 1a, 1b, and Step 2 LLM calls. Returns dict with signal/provider/model or None if should return."""
        engine = self.engine
        assigned_tf = symbol_entry["timeframe"]
        tf_seconds = engine._timeframe_to_seconds(assigned_tf)
        is_critical = flags["max_hold_expired"] or flags["stop_loss_triggered"] or flags["take_profit_triggered"] or flags["partial_tp_triggered"] or flags["dust_sweep_triggered"]
        has_position = symbol in engine.positions

        critical_reason = None
        if is_critical:
            critical_reason = "LLM timeout"
            if flags["max_hold_expired"]: critical_reason = "Max hold expired, LLM timeout"
            elif flags["stop_loss_triggered"]: critical_reason = "Stop-loss triggered, LLM timeout"
            elif flags["take_profit_triggered"]: critical_reason = "Take-profit triggered, LLM timeout"
            elif flags["partial_tp_triggered"]: critical_reason = "Partial TP triggered, LLM timeout"
            elif flags["dust_sweep_triggered"]: critical_reason = "Dust sweep triggered, LLM timeout"

        analysis_result, llm_provider, llm_model, _should_return = await self.run_step1a_llm_call(
            symbol=symbol, display_symbol=display_symbol, analysis_prompt=ctx["analysis_prompt"],
            system_prompt=compact_prompt(build_system_prompt()), market_hash=ctx["market_hash"],
            strategy_model_type=strategy_model_type, effective_temp=effective_temp,
            current_price=ctx["current_price"], rsi=ctx["rsi"], macd_hist=ctx["macd_hist"],
            is_critical=is_critical, critical_reason=critical_reason,
        )
        if _should_return:
            return None

        signal, combined_bt_summary, llm_provider, llm_model, _skip_backtest = await self.handle_step1a_fallback(
            symbol=symbol, analysis_result=analysis_result, has_position=has_position,
            strategy_model_type=strategy_model_type, llm_provider=llm_provider, llm_model=llm_model,
        )

        if not _skip_backtest:
            preliminary_signal, llm_provider, llm_model = await self.run_step1b_llm_call(
                symbol=symbol, analysis_result=analysis_result, ticker=ctx["ticker"], current_price=ctx["current_price"],
                atr=ctx["atr"], assigned_tf=assigned_tf, base_balance=ctx["base_balance"], per_symbol_budget=ctx["per_symbol_budget"],
                min_order_amount=ctx["min_order_amount"], min_order_cost=ctx["min_order_cost"], remaining=ctx["remaining"],
                portfolio_total_value=ctx["portfolio_total_value"], portfolio_exposure_pct=ctx["portfolio_exposure_pct"],
                portfolio_stop_risk_pct=ctx["portfolio_stop_risk_pct"], portfolio_available_capital=ctx["portfolio_available_capital"],
                max_port_exp=ctx["max_port_exp"], max_port_risk=ctx["max_port_risk"], global_risk_mult=ctx["global_risk_mult"],
                min_stop_atr_mult=ctx["min_stop_atr_mult"], min_hold_time_mult=ctx["min_hold_time_mult"], trading_paused=False,
                has_position=has_position, strategy_model_type=strategy_model_type, effective_temp=effective_temp,
                market_snapshot=ctx["market_snapshot"], historical_backtest_results=ctx["historical_backtest_results"],
            )
            signal, combined_bt_summary, llm_provider, llm_model = await engine._run_backtest_and_final_decision(
                symbol=symbol, assigned_tf=assigned_tf, tf_seconds=tf_seconds, current_price=ctx["current_price"],
                atr=ctx["atr"], historical_ohlcv=ctx["historical_ohlcv"], raw_candles=ctx["raw_candles"],
                base_balance=ctx["base_balance"], is_btp=ctx["is_btp"], trading_paused=False,
                strategy_model_type=strategy_model_type, effective_temp=effective_temp,
                preliminary_signal=preliminary_signal, display_symbol=display_symbol, ticker=ctx["ticker"],
            )

        return {"signal": signal, "llm_provider": llm_provider, "llm_model": llm_model}

    async def process_symbol(self, symbol_entry: Dict[str, str], trading_paused: bool = False) -> None:
        """Fetch market data, get LLM strategy, validate, and execute."""
        engine = self.engine
        symbol = symbol_entry["symbol"]
        assigned_tf = symbol_entry["timeframe"]
        tf_seconds = engine._timeframe_to_seconds(assigned_tf)

        try:
            _init_ctx = await self._get_initial_context(symbol, symbol_entry)
            if _init_ctx is None:
                return
            stock_name = _init_ctx["stock_name"]
            display_symbol = _init_ctx["display_symbol"]
            min_viable_amount = _init_ctx["min_viable_amount"]
            _flags = _init_ctx["flags"]

            symbol_data = await self._fetch_and_validate_data(symbol, symbol_entry, display_symbol)
            if symbol_data is None:
                return

            ctx = await self._gather_and_build_prompt(symbol, symbol_entry, symbol_data, min_viable_amount, _flags)
            ctx["assigned_tf"] = assigned_tf

            model_tier = await self._check_skip_and_model_tier(symbol, ctx, _flags)
            if model_tier is None:
                return
            strategy_model_type, effective_temp = model_tier

            llm_result = await self._run_llm_steps(symbol, display_symbol, symbol_entry, ctx, strategy_model_type, effective_temp, _flags)
            if llm_result is None:
                return
            signal = llm_result["signal"]
            llm_provider = llm_result["llm_provider"]
            llm_model = llm_result["llm_model"]

            decision_data = DecisionContext(
                symbol=symbol, display_symbol=display_symbol, stock_name=stock_name,
                assigned_tf=assigned_tf, tf_seconds=tf_seconds, ticker=ctx["ticker"], signal=signal,
                llm_provider=llm_provider, llm_model=llm_model, trading_paused=trading_paused,
                base_balance=ctx["base_balance"], current_price=ctx["current_price"], atr=ctx["atr"],
                rsi=ctx["rsi"], macd=ctx["macd"], macd_signal=ctx["macd_signal"], macd_hist=ctx["macd_hist"],
                bb_upper=ctx["bb_upper"], bb_middle=ctx["bb_middle"], bb_lower=ctx["bb_lower"],
                ema_9=ctx["ema_9"], ema_21=ctx["ema_21"], stochastic_k=ctx["stochastic_k"], stochastic_d=ctx["stochastic_d"],
                adx=ctx["adx"], plus_di=ctx["plus_di"], minus_di=ctx["minus_di"], obv=ctx["obv"], mfi=ctx["mfi"],
                cci=ctx["cci"], williams_r=ctx["williams_r"], ichimoku=ctx["ichimoku"], donchian_channels=ctx["donchian_channels"],
                parabolic_sar=ctx["parabolic_sar"], keltner_channels=ctx["keltner_channels"],
                aggregate_sentiment=ctx["aggregate_sentiment"], market_regime=ctx["market_regime"],
                min_stop_atr_mult=ctx["min_stop_atr_mult"], min_hold_time_mult=ctx["min_hold_time_mult"],
                global_min_rr=ctx["global_min_rr"], max_hold_expired=_flags["max_hold_expired"],
                stop_loss_triggered=_flags["stop_loss_triggered"], take_profit_triggered=_flags["take_profit_triggered"],
                partial_tp_triggered=_flags["partial_tp_triggered"], dust_sweep_triggered=_flags["dust_sweep_triggered"],
                strategy_model_type=strategy_model_type,
            )
            await self.process_post_llm_decision(decision_data)
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}", exc_info=True)
            if engine.notifier:
                await engine.notifier.send_notification(
                    f"❌ Error processing {display_symbol}: {e}",
                    summary={"symbol": symbol, "action": "ERROR", "reason": str(e)[:200]}
                )

    async def classify_market_regime(
        self,
        adx: Optional[float],
        plus_di: Optional[float],
        minus_di: Optional[float],
        ema_9: Optional[float],
        ema_21: Optional[float],
        bb_upper: Optional[float],
        bb_lower: Optional[float],
        bb_middle: Optional[float],
        atr: Optional[float],
        atr_percentile: Optional[float],
        current_price: float,
    ) -> str:
        """Classify market regime using multiple indicators."""
        engine = self.engine
        if current_price is None or current_price <= 0:
            return "unknown"

        # Read LLM-decided regime thresholds from Redis (set during stock selection).
        adx_strong = None
        adx_moderate = None
        vol_high_pct = None
        vol_low_pct = None
        bb_squeeze_width = None
        bb_expansion_width = None
        try:
            raw = await engine.config_service.get_config("regime_adx_strong")
            if raw:
                adx_strong = float(raw)
            raw = await engine.config_service.get_config("regime_adx_moderate")
            if raw:
                adx_moderate = float(raw)
            raw = await engine.config_service.get_config("regime_volatility_high_pct")
            if raw:
                vol_high_pct = float(raw)
            raw = await engine.config_service.get_config("regime_volatility_low_pct")
            if raw:
                vol_low_pct = float(raw)
            raw = await engine.config_service.get_config("regime_bb_squeeze_width")
            if raw:
                bb_squeeze_width = float(raw)
            raw = await engine.config_service.get_config("regime_bb_expansion_width")
            if raw:
                bb_expansion_width = float(raw)
        except (ValueError, TypeError, ConnectionError, TimeoutError, OSError):
            pass

        if (
            adx_strong is None
            or adx_moderate is None
            or vol_high_pct is None
            or vol_low_pct is None
            or bb_squeeze_width is None
            or bb_expansion_width is None
        ):
            return "unknown"

        # --- Trend direction and strength ---
        trend_dir = "neutral"
        trend_strength = "weak"
        if adx is not None and plus_di is not None and minus_di is not None:
            if adx > adx_strong:
                trend_strength = "strong"
            elif adx > adx_moderate:
                trend_strength = "moderate"
            else:
                trend_strength = "weak"

            if plus_di > minus_di:
                trend_dir = "uptrend"
            elif minus_di > plus_di:
                trend_dir = "downtrend"
            else:
                trend_dir = "neutral"

        # --- Moving average alignment ---
        ma_alignment = "neutral"
        if ema_9 is not None and ema_21 is not None:
            if ema_9 > ema_21:
                ma_alignment = "bullish"
            else:
                ma_alignment = "bearish"

        # --- Volatility state ---
        volatility = "normal"
        if atr is not None and current_price > 0:
            atr_pct = (atr / current_price) * 100
            if atr_percentile is not None:
                if atr_percentile > vol_high_pct:
                    volatility = "high"
                elif atr_percentile < vol_low_pct:
                    volatility = "low"
                else:
                    volatility = "normal"
            else:
                if atr_pct > (bb_expansion_width * 100):
                    volatility = "high"
                elif atr_pct < (bb_squeeze_width * 100):
                    volatility = "low"

        # --- Bollinger Band squeeze/expansion ---
        bb_state = ""
        if bb_upper is not None and bb_lower is not None and bb_middle is not None and bb_middle > 0:
            bb_width = (bb_upper - bb_lower) / bb_middle
            if bb_width < bb_squeeze_width:
                bb_state = " squeeze"
            elif bb_width > bb_expansion_width:
                bb_state = " expansion"

        # --- Compose final regime string ---
        if trend_strength in ("strong", "moderate") and trend_dir != "neutral":
            regime = f"{trend_strength} {trend_dir}"
        else:
            regime = "ranging"

        regime += f", {volatility} volatility"

        if bb_state:
            regime += bb_state

        if trend_strength == "weak" and ma_alignment != "neutral":
            regime += f" ({ma_alignment} MA bias)"

        return regime

    async def compute_atr_percentile(
        self,
        symbol: str,
        atr: Optional[float],
    ) -> Optional[float]:
        """Compute ATR percentile from Redis-stored history.

        Maintains a rolling window of the last 100 ATR values in Redis
        and returns the percentile rank of the current ATR.
        Returns None if ATR is invalid or insufficient history exists.
        """
        engine = self.engine
        if atr is None or atr <= 0:
            return None

        atr_percentile_key = f"atr_percentile:{symbol}"
        try:
            stored_atr = await asyncio.to_thread(engine.redis.get, atr_percentile_key)
            if stored_atr:
                atr_history = json.loads(stored_atr)
            else:
                atr_history = []
            atr_history.append(atr)
            atr_history = atr_history[-100:]
            await asyncio.to_thread(engine.redis.setex, atr_percentile_key, 7 * 24 * 3600, json.dumps(atr_history))
            if len(atr_history) >= 5:
                sorted_atr = sorted(atr_history)
                rank = sum(1 for v in sorted_atr if v <= atr)
                return round(rank / len(sorted_atr) * 100, 1)
        except (json.JSONDecodeError, ValueError, TypeError, ConnectionError, TimeoutError, OSError) as e:
            logger.info(f"ATR percentile computation failed for {symbol}: {e}")

        return None

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
            if tf in ohlcv_data and ohlcv_data[tf]:
                candles = ohlcv_data[tf]
                multi_tf_raw_candles[tf] = candles
                ind = symbol_inds.get(tf)
                if ind:
                    # --- Staleness check: recompute if indicators are older than 2× the candle interval ---
                    ind_ts = ind.get("_indicator_timestamp", None)
                    latest_candle_ts = candles[-1][0] if candles else None
                    if ind_ts is not None and latest_candle_ts is not None:
                        tf_ms = engine._timeframe_to_ms(tf)
                        if (latest_candle_ts - ind_ts) > 2 * tf_ms:
                            logger.info(
                                f"Indicators for {symbol} {tf} are stale "
                                f"(indicator ts={ind_ts}, latest candle ts={latest_candle_ts}, "
                                f"gap={latest_candle_ts - ind_ts}ms > {2 * tf_ms}ms). Scheduling background recomputation."
                            )
                            # Recompute indicators synchronously to avoid using stale data for the current decision
                            ind = await asyncio.to_thread(compute_all_indicators, candles)
                            # Schedule background storage of the recomputed indicators
                            asyncio.create_task(
                                engine._market_data_manager.compute_and_store_indicators(symbol, tf, candles)
                            )
                    multi_tf_indicators[tf] = ind
                    if tf == assigned_tf:
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

    async def fetch_symbol_market_data(self, symbol: str, assigned_tf: str) -> Optional[Dict[str, Any]]:
        """Fetch all raw market data for a symbol: ticker, fundamentals, balance, OHLCV, and multi-TF indicators.

        Returns a dict with all fetched data, or None if ticker is unavailable.
        """
        engine = self.engine
        base_symbol = symbol.split("/")[0]
        is_btp = is_btp_isin(base_symbol)
        tf_seconds = engine._timeframe_to_seconds(assigned_tf)

        # --- Fetch ticker ---
        async with engine._exchange_semaphore:
            quotes = await engine._get_quotes_async([base_symbol], timeout=45.0)
            ticker = quotes.get(base_symbol)
        if ticker is None:
            return None
        current_price = ticker['last']

        # --- Yahoo Finance fallback for missing bid/ask ---
        if ticker is not None and not is_btp:
            bid = ticker.get('bid')
            ask = ticker.get('ask')
            if bid is None or ask is None:
                yahoo = await asyncio.to_thread(get_yahoo_quote, base_symbol)
                if yahoo:
                    if bid is None:
                        ticker['bid'] = yahoo.get('bid')
                    if ask is None:
                        ticker['ask'] = yahoo.get('ask')
                    logger.info(f"Yahoo Finance quote merged for {symbol}: bid={ticker.get('bid')}, ask={ticker.get('ask')}")

        # --- Fetch fundamental data ---
        fundamentals = None
        if settings.YAHOO_FINANCE_ENABLED and not is_btp:
            fundamentals = await asyncio.to_thread(get_yahoo_fundamentals, base_symbol)

        # --- Fetch balance ---
        balance = await engine._get_cached_balance()
        base_balance = balance.get(engine.base_currency, 0.0)

        # --- Fetch OHLCV from database ---
        ohlcv_data = {}
        if settings.OHLCV_TIMEFRAMES:
            async def _fetch_ohlcv_tf(tf):
                try:
                    if tf == assigned_tf:
                        since_ms = int(time.time() * 1000) - settings.OHLCV_RETENTION_DAYS * 24 * 60 * 60 * 1000
                        tf_seconds = engine._timeframe_to_seconds(tf)
                        hist_limit = int((settings.OHLCV_RETENTION_DAYS * 86400) / tf_seconds) + 100
                        db_candles = await asyncio.to_thread(get_ohlcv, symbol, tf, since_ms=since_ms, limit=hist_limit)
                    else:
                        db_candles = await asyncio.to_thread(get_ohlcv, symbol, tf, limit=100)
                    if db_candles:
                        return tf, [[c["timestamp"], c["open"], c["high"], c["low"], c["close"], c["volume"]] for c in db_candles]
                except (ValueError, TypeError, KeyError, ConnectionError, TimeoutError, OSError) as e:
                    logger.debug(f"DB OHLCV fetch failed for {symbol} {tf}: {e}")
                return tf, None
            ohlcv_results = await asyncio.gather(*[_fetch_ohlcv_tf(tf) for tf in settings.OHLCV_TIMEFRAMES])
            for tf, candles in ohlcv_results:
                if candles:
                    ohlcv_data[tf] = candles

        # --- Compute multi-TF indicators ---
        _inds = await self.compute_multi_tf_indicators(symbol, ohlcv_data, assigned_tf)

        return {
            "ticker": ticker,
            "current_price": current_price,
            "fundamentals": fundamentals,
            "balance": balance,
            "base_balance": base_balance,
            "ohlcv_data": ohlcv_data,
            "is_btp": is_btp,
            "tf_seconds": tf_seconds,
            "multi_tf_indicators": _inds["multi_tf_indicators"],
            "multi_tf_raw_candles": _inds["multi_tf_raw_candles"],
            "atr": _inds["atr"],
            "rsi": _inds["rsi"],
            "macd": _inds["macd"],
            "macd_signal": _inds["macd_signal"],
            "macd_hist": _inds["macd_hist"],
            "bb_upper": _inds["bb_upper"],
            "bb_middle": _inds["bb_middle"],
            "bb_lower": _inds["bb_lower"],
            "ema_9": _inds["ema_9"],
            "ema_21": _inds["ema_21"],
            "stochastic_k": _inds["stochastic_k"],
            "stochastic_d": _inds["stochastic_d"],
            "adx": _inds["adx"],
            "plus_di": _inds["plus_di"],
            "minus_di": _inds["minus_di"],
            "obv": _inds["obv"],
            "mfi": _inds["mfi"],
            "cci": _inds["cci"],
            "williams_r": _inds["williams_r"],
            "ichimoku": _inds["ichimoku"],
            "donchian_channels": _inds["donchian_channels"],
            "parabolic_sar": _inds["parabolic_sar"],
            "keltner_channels": _inds["keltner_channels"],
            "vwap": _inds["vwap"],
            "daily_pivot_points": _inds["daily_pivot_points"],
        }

    async def gather_prompt_context(
        self,
        symbol: str,
        assigned_tf: str,
        tf_seconds: int,
        ticker: Dict[str, Any],
        base_balance: float,
        ohlcv_data: Dict[str, List[List]],
        multi_tf_indicators: Dict[str, Dict[str, Any]],
        multi_tf_raw_candles: Dict[str, List[List]],
        atr: Optional[float],
        rsi: Optional[float],
        macd: Optional[float],
        macd_signal: Optional[float],
        macd_hist: Optional[float],
        bb_upper: Optional[float],
        bb_middle: Optional[float],
        bb_lower: Optional[float],
        ema_9: Optional[float],
        ema_21: Optional[float],
        adx: Optional[float],
        plus_di: Optional[float],
        minus_di: Optional[float],
    ) -> Dict[str, Any]:
        """Gather all additional market context needed for the strategy prompt."""
        engine = self.engine
        # ATR multi-TF
        atr_multi_tf: Dict[str, float] = {}
        for tf in settings.OHLCV_TIMEFRAMES:
            ind = multi_tf_indicators.get(tf, {})
            tf_atr = ind.get('atr')
            if tf_atr is not None and tf_atr > 0:
                atr_multi_tf[tf] = tf_atr

        # ATR Percentile (volatility context)
        atr_percentile = await self.compute_atr_percentile(symbol, atr)

        # Market regime classification
        market_regime = await self.classify_market_regime(
            adx=adx, plus_di=plus_di, minus_di=minus_di,
            ema_9=ema_9, ema_21=ema_21,
            bb_upper=bb_upper, bb_lower=bb_lower, bb_middle=bb_middle,
            atr=atr, atr_percentile=atr_percentile,
            current_price=ticker['last'],
        )

        # Extract raw candles for the assigned timeframe
        raw_candles = multi_tf_raw_candles.get(assigned_tf)

        # Fetch historical OHLCV from DB for backtest analysis
        historical_ohlcv = ohlcv_data.get(assigned_tf)
        if historical_ohlcv and len(historical_ohlcv) >= 2:
            interval_ms = engine._timeframe_to_ms(assigned_tf)
            timestamps = [c[0] for c in historical_ohlcv]
            has_gap = False
            for i in range(len(timestamps) - 1):
                if timestamps[i+1] - timestamps[i] > interval_ms * 1.5:
                    has_gap = True
                    break
            if has_gap:
                logger.warning(
                    f"Historical OHLCV for {symbol} {assigned_tf} contains gaps; "
                    f"passing data to LLM anyway for backtesting."
                )

        # Unrealized P&L for current position
        unrealized_pnl = None
        position_info = None
        if symbol in engine.positions:
            pos = engine.positions[symbol]
            position_info = pos
            current_price = ticker['last']
            entry_price = pos['price']
            amount = pos['amount']
            unrealized_pnl = (current_price - entry_price) * amount

        # Recent trade outcomes (last 5 closed trades)
        recent_trades = [t for t in engine.trade_history if t.get("side") == "sell"][-5:]
        recent_trades_summary = [
            {
                "symbol": t["symbol"],
                "realized_pnl": t.get("realized_pnl", 0.0),
                "strategy": t.get("strategy_type", "unknown"),
            }
            for t in recent_trades
        ]

        # Fetch minimum order size
        try:
            asset = await engine._get_asset_info(symbol)
            min_order_amount = float(asset.min_order_size) if asset.min_order_size else None
        except (ValueError, TypeError, AttributeError, ConnectionError, TimeoutError, OSError):
            min_order_amount = None
        current_price = ticker['last']
        if min_order_amount is not None and current_price:
            min_order_cost = min_order_amount * current_price
        else:
            min_order_cost = None

        # Past trades for this specific symbol (last 10 closed sells)
        past_trades = [
            t for t in engine.trade_history
            if t.get("symbol") == symbol and t.get("side") == "sell"
        ][-10:]

        # Fetch historical backtest results for this symbol
        historical_backtest_results = await asyncio.to_thread(
            get_backtest_results_for_symbol, symbol, assigned_tf, 10
        )

        # Fetch aggregate sentiment
        aggregate_sentiment = None
        if settings.NEWS_ENABLED:
            try:
                aggregate_sentiment = await engine._get_cached_sentiment(symbol)
            except (ValueError, TypeError, KeyError, ConnectionError, TimeoutError, OSError) as e:
                logger.info(f"Could not fetch aggregate sentiment for {symbol}: {e}")

        # Sentiment trend
        sentiment_trend_val = None
        if aggregate_sentiment:
            base_symbol = symbol.split("/")[0] if "/" in symbol else symbol
            current_compound = aggregate_sentiment.get("avg_compound")
            prev_key = f"sentiment:prev:{base_symbol}"
            prev_raw = await asyncio.to_thread(engine.redis.get, prev_key)
            prev_compound = float(prev_raw) if prev_raw else None
            if current_compound is not None:
                await asyncio.to_thread(engine.redis.setex, prev_key, settings.NEWS_CACHE_TTL_SECONDS, str(current_compound))
            if current_compound is not None and prev_compound is not None:
                sentiment_trend_val = round(current_compound - prev_compound, 4)

        # Volume trend
        volume_trend_val = None
        current_volume = ticker.get('quoteVolume', 0) or 0
        if current_volume > 0:
            volume_trend_val = await self._compute_volume_trend(symbol, current_volume, timeframe=assigned_tf)

        # Full market breadth from Redis
        full_market_breadth = None
        try:
            full_breadth_raw = await asyncio.to_thread(engine.redis.get, "market:breadth:full")
            if full_breadth_raw:
                full_market_breadth = json.loads(full_breadth_raw)
        except (ValueError, TypeError, ConnectionError, TimeoutError, OSError, json.JSONDecodeError):
            pass
        session_info = engine._get_session_info()

        # Compute minutes until market close
        now_rome = datetime.now(timezone.utc).astimezone(ZoneInfo(settings.MARKET_TIMEZONE))
        weekday = now_rome.weekday()
        if weekday < 5:
            rome_minutes = now_rome.hour * 60 + now_rome.minute
            close_minutes = settings.MARKET_CLOSE_HOUR * 60 + settings.MARKET_CLOSE_MINUTE
            minutes_to_market_close = close_minutes - rome_minutes
            if minutes_to_market_close < 0:
                minutes_to_market_close = 0
        else:
            minutes_to_market_close = None

        # Global risk multiplier
        global_risk_mult = await engine._get_global_risk_multiplier()

        # Portfolio risk thresholds
        max_port_exp = None
        max_port_risk = None
        try:
            raw = await engine.config_service.get_config("max_portfolio_exposure_pct")
            if raw:
                max_port_exp = float(raw)
            raw = await engine.config_service.get_config("max_portfolio_stop_risk_pct")
            if raw:
                max_port_risk = float(raw)
        except (ValueError, TypeError, ConnectionError, TimeoutError, OSError):
            pass

        partial_tp_executed_levels = engine.positions[symbol].get("partial_tp_levels_triggered", []) if symbol in engine.positions else []

        # Validator multipliers
        min_stop_atr_mult = 1.0
        min_hold_time_mult = 1.0
        global_min_rr = None
        try:
            raw = await engine.config_service.get_config("min_stop_loss_atr_mult")
            if raw:
                min_stop_atr_mult = float(raw)
            raw = await engine.config_service.get_config("min_max_hold_time_mult")
            if raw:
                min_hold_time_mult = float(raw)
            raw = await engine.config_service.get_config("min_risk_reward_ratio")
            if raw:
                global_min_rr = float(raw)
        except (ValueError, TypeError, ConnectionError, TimeoutError, OSError):
            pass

        return {
            "atr_multi_tf": atr_multi_tf,
            "atr_percentile": atr_percentile,
            "market_regime": market_regime,
            "raw_candles": raw_candles,
            "historical_ohlcv": historical_ohlcv,
            "unrealized_pnl": unrealized_pnl,
            "position_info": position_info,
            "recent_trades_summary": recent_trades_summary,
            "min_order_amount": min_order_amount,
            "min_order_cost": min_order_cost,
            "past_trades": past_trades,
            "aggregate_sentiment": aggregate_sentiment,
            "sentiment_trend_val": sentiment_trend_val,
            "volume_trend_val": volume_trend_val,
            "full_market_breadth": full_market_breadth,
            "session_info": session_info,
            "minutes_to_market_close": minutes_to_market_close,
            "global_risk_mult": global_risk_mult,
            "max_port_exp": max_port_exp,
            "max_port_risk": max_port_risk,
            "partial_tp_executed_levels": partial_tp_executed_levels,
            "min_stop_atr_mult": min_stop_atr_mult,
            "min_hold_time_mult": min_hold_time_mult,
            "global_min_rr": global_min_rr,
            "historical_backtest_results": historical_backtest_results,
        }

    async def build_analysis_prompt_and_snapshot(
        self,
        data: StrategyPromptData,
    ) -> Tuple[str, Dict[str, Any], str]:
        """Build the Step 1a analysis prompt, market snapshot, and market hash.

        Returns (analysis_prompt, market_snapshot, market_hash).
        """
        engine = self.engine

        analysis_prompt = await asyncio.to_thread(
            build_analysis_prompt,
            data
        )
        # Add quote staleness warning if the price data is outdated
        staleness_warning = engine._market_data_manager._get_quote_staleness_warning(data.ticker)
        if staleness_warning:
            analysis_prompt += staleness_warning
        # Add auto-resume note so the LLM sees this context in per-symbol decisions
        last_auto_resume_raw = await asyncio.to_thread(engine.redis.get, "trading:last_auto_resume")
        if last_auto_resume_raw:
            try:
                last_auto_resume_ts = float(last_auto_resume_raw)
                seconds_since = time.time() - last_auto_resume_ts
                if seconds_since < engine._symbol_reevaluation_interval * 2:
                    minutes_since = seconds_since / 60
                    analysis_prompt += (
                        f"\n**NOTE:** Trading was auto‑resumed {minutes_since:.1f} minutes ago after a pause. "
                        "Market conditions may not have changed significantly. "
                        "Consider whether conditions have actually improved enough to justify trading. "
                        "If you decide to pause again, set a longer `pause_duration_seconds` (e.g., 1800–7200) "
                        "to allow conditions to evolve; a very short pause will likely lead to the same outcome.\n"
                    )
            except (ValueError, TypeError):
                pass
        logger.info(f"LLM Step 1a analysis prompt for {data.symbol}: {len(analysis_prompt)} chars")
        # Build a market snapshot dict for caching (per-symbol)
        market_snapshot = {
            "symbol": data.symbol,
            "ticker": data.ticker,
            "staleness_warning": staleness_warning,
            "balance": data.balance,
            "open_positions": data.open_positions,
            "per_symbol_budget": data.per_symbol_budget,
            "max_symbols": data.max_symbols,
            "performance": data.performance,
            "ohlcv_data": data.ohlcv_data,
            "assigned_timeframe": data.assigned_timeframe,
            "atr": data.atr,
            "atr_multi_tf": data.atr_multi_tf,
            "rsi": data.rsi,
            "macd": data.macd,
            "macd_signal": data.macd_signal,
            "macd_hist": data.macd_hist,
            "bb_upper": data.bb_upper,
            "bb_middle": data.bb_middle,
            "bb_lower": data.bb_lower,
            "ema_9": data.ema_9,
            "ema_21": data.ema_21,
            "stochastic_k": data.stochastic_k,
            "stochastic_d": data.stochastic_d,
            "adx": data.adx,
            "plus_di": data.plus_di,
            "minus_di": data.minus_di,
            "obv": data.obv,
            "mfi": data.mfi,
            "cci": data.cci,
            "williams_r": data.williams_r,
            "ichimoku": data.ichimoku,
            "donchian_channels": data.donchian_channels,
            "drawdown_pct": data.drawdown_pct,
            "raw_candles": data.raw_candles,
            "recent_trades": data.recent_trades,
            "historical_ohlcv": data.historical_ohlcv,
            "min_order_amount": data.min_order_amount,
            "min_order_cost": data.min_order_cost,
            "all_symbols": data.all_symbols,
            "past_trades": data.past_trades,
            "aggregate_sentiment": data.aggregate_sentiment,
            "cycle_spent": data.cycle_spent,
            "remaining_balance": data.remaining_balance,
            "market_regime": data.market_regime,
            "multi_tf_raw_candles": data.multi_tf_raw_candles,
            "multi_tf_indicators": data.multi_tf_indicators,
            "session_info": data.session_info,
            "sentiment_trend": data.sentiment_trend,
            "volume_trend": data.volume_trend,
            "market_breadth": data.market_breadth,
            "full_market_breadth": data.full_market_breadth,
            "parabolic_sar": data.parabolic_sar,
            "keltner_channels": data.keltner_channels,
            "atr_percentile": data.atr_percentile,
            "global_risk_multiplier": data.global_risk_multiplier,
            "trading_paused": data.trading_paused,
            "last_decision": data.last_decision,
        }
        market_hash = compute_market_hash(market_snapshot)

        return analysis_prompt, market_snapshot, market_hash

    async def read_position_trigger_flags(
        self, symbol: str, symbol_entry: Dict[str, str]
    ) -> Optional[Dict[str, Any]]:
        """Read pre-processing flags and review limits for a symbol.

        Checks max tenure, cooldown after loss, queued orders, and reads
        all position trigger flags (max hold, stop loss, take profit, partial
        TP, dust sweep) and LLM-decided review limits.

        Returns None if the symbol should be skipped (tenure reached,
        cooldown active, or order already queued).
        Otherwise returns a dict with all flags and review limits.
        """
        engine = self.engine
        assigned_tf = symbol_entry["timeframe"]
        tf_seconds = engine._timeframe_to_seconds(assigned_tf)

        # --- Maximum symbol tenure (per-symbol, set by LLM) ---
        max_tenure_hours = symbol_entry.get('max_tenure_hours')
        if max_tenure_hours is not None and max_tenure_hours > 0 and 'entry_time' in symbol_entry:
            tenure_seconds = max_tenure_hours * 3600
            if time.time() - symbol_entry['entry_time'] > tenure_seconds:
                logger.info(f"Max symbol tenure reached for {symbol} ({max_tenure_hours:.1f}h), forcing sell")
                from src.strategies.base import Signal
                signal = Signal(action="SELL", confidence=1.0, reasoning="Max symbol tenure reached")
                await self.event_bus.publish("execute_signal", symbol, signal, exit_reason="max_tenure")
                async with engine._eval_state_lock:
                    engine._force_eval.pop(symbol, None)
                return None

        # --- Cooldown after a losing trade (LLM-defined) ---
        if symbol not in engine.positions:
            last_loss = engine.last_loss_time.get(symbol)
            if last_loss is not None:
                cooldown = engine.cooldown_durations.get(symbol, 0)
                if cooldown > 0:
                    elapsed = time.time() - last_loss
                    if elapsed < cooldown:
                        remaining = cooldown - elapsed
                        logger.info(
                            f"Skipping {symbol}: cooldown active ({remaining:.0f}s remaining after loss)"
                        )
                        async with engine._eval_state_lock:
                            engine._force_eval.pop(symbol, None)
                        return None

        # Skip if there is already a queued order for this symbol
        async with engine._queued_orders_lock:
            has_queued = any(q['symbol'] == symbol for q in engine.queued_orders)
        if has_queued:
            logger.info(f"Skipping {symbol}: order already queued.")
            async with engine._eval_state_lock:
                engine._force_eval.pop(symbol, None)
            return None

        # --- Read position trigger flags ---
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
        if symbol in engine.positions:
            pos = engine.positions[symbol]
            max_hold_expired = pos.get("_max_hold_expired", False)
            max_hold_expired_count = pos.get("_max_hold_expired_count", 1)
            stop_loss_triggered = pos.get("_stop_loss_triggered", False)
            stop_loss_review_count = pos.get("_stop_loss_review_count", 0)
            take_profit_triggered = pos.get("_take_profit_triggered", False)
            take_profit_review_count = pos.get("_take_profit_review_count", 0)
            partial_tp_triggered = pos.get("_partial_tp_triggered", False) or pos.get("_partial_tp_triggered_single", False)
            partial_tp_review_count = pos.get("_partial_tp_review_count", 0) or pos.get("_partial_tp_single_review_count", 0)
            partial_tp_triggered_levels = pos.get("_partial_tp_triggered_levels", [])
            dust_sweep_triggered = pos.get("_dust_sweep_triggered", False)
            dust_sweep_review_count = pos.get("_dust_sweep_review_count", 0)

        # --- Read LLM-decided review limits from Redis ---
        max_sl_reviews_prompt = settings.MAX_STOP_LOSS_REVIEWS
        max_tp_reviews_prompt = settings.MAX_TAKE_PROFIT_REVIEWS
        try:
            raw = await engine.config_service.get_config("max_stop_loss_reviews")
            if raw:
                max_sl_reviews_prompt = int(raw)
            raw = await engine.config_service.get_config("max_take_profit_reviews")
            if raw:
                max_tp_reviews_prompt = int(raw)
        except (ValueError, TypeError, ConnectionError, TimeoutError, OSError):
            pass

        max_partial_tp_reviews_prompt = settings.MAX_PARTIAL_TP_REVIEWS
        max_dust_sweep_reviews_prompt = settings.MAX_DUST_SWEEP_REVIEWS
        try:
            raw = await engine.config_service.get_config("max_partial_tp_reviews")
            if raw:
                max_partial_tp_reviews_prompt = int(raw)
            raw = await engine.config_service.get_config("max_dust_sweep_reviews")
            if raw:
                max_dust_sweep_reviews_prompt = int(raw)
        except (ValueError, TypeError, ConnectionError, TimeoutError, OSError):
            pass

        # Scale stop-loss review limit for long-term timeframes
        if tf_seconds >= settings.LONG_TERM_TF_SECONDS:  # >= 1 month
            max_sl_reviews_prompt = min(max_sl_reviews_prompt, settings.LONG_TERM_MAX_STOP_LOSS_REVIEWS)
        elif tf_seconds >= 604_800:  # >= 1 week
            max_sl_reviews_prompt = min(max_sl_reviews_prompt, settings.WEEKLY_MAX_STOP_LOSS_REVIEWS)

        return {
            "max_hold_expired": max_hold_expired,
            "max_hold_expired_count": max_hold_expired_count,
            "stop_loss_triggered": stop_loss_triggered,
            "stop_loss_review_count": stop_loss_review_count,
            "take_profit_triggered": take_profit_triggered,
            "take_profit_review_count": take_profit_review_count,
            "partial_tp_triggered": partial_tp_triggered,
            "partial_tp_review_count": partial_tp_review_count,
            "partial_tp_triggered_levels": partial_tp_triggered_levels,
            "dust_sweep_triggered": dust_sweep_triggered,
            "dust_sweep_review_count": dust_sweep_review_count,
            "max_sl_reviews_prompt": max_sl_reviews_prompt,
            "max_tp_reviews_prompt": max_tp_reviews_prompt,
            "max_partial_tp_reviews_prompt": max_partial_tp_reviews_prompt,
            "max_dust_sweep_reviews_prompt": max_dust_sweep_reviews_prompt,
        }

    async def check_skip_conditions(
        self,
        symbol: str,
        display_symbol: str,
        ticker: Dict[str, Any],
        assigned_tf: str,
        has_position: bool,
        base_balance: float,
    ) -> bool:
        """Check whether a symbol should be skipped before LLM evaluation.

        Returns True if the symbol should be skipped (caller should return),
        False if processing should continue.
        """
        engine = self.engine

        # --- Staleness guard: skip symbols with stale quotes (unless we have an open position) ---
        if not has_position and await engine._is_quote_too_stale(ticker, assigned_tf):
            logger.info(
                f"Skipping {symbol}: quote data is too stale for timeframe {assigned_tf}."
            )
            stale_notify_key = f"trading:stale_quote_notify:{symbol}"
            should_notify = True
            try:
                last_notify_raw = await asyncio.to_thread(engine.redis.get, stale_notify_key)
                if last_notify_raw:
                    if (time.time() - float(last_notify_raw)) < settings.STALENESS_NOTIFY_THRESHOLD_SECONDS:
                        should_notify = False
            except (ValueError, TypeError, ConnectionError, TimeoutError, OSError):
                pass
            if should_notify and engine.notifier:
                await engine.notifier.send_notification(
                    f"⏸️ Skipping {display_symbol}: quote data is too stale for timeframe {assigned_tf}.",
                    summary={
                        "symbol": symbol,
                        "action": "SKIP",
                        "reason": "Quote data too stale",
                    }
                )
                try:
                    await asyncio.to_thread(engine.redis.setex, stale_notify_key, 3600, str(time.time()))
                except (ValueError, TypeError, ConnectionError, TimeoutError, OSError):
                    pass
            async with engine._eval_state_lock:
                engine._force_eval.pop(symbol, None)
            return True

        # If we have an open position, we must continue evaluating it for SELL signals
        # even when base_balance is 0 (all capital deployed) or effective_max_symbols is 0.
        if not has_position and (base_balance <= 0 or engine.effective_max_symbols == 0):
            logger.warning(
                f"Skipping {symbol}: {engine.base_currency} balance={base_balance:.2f}, "
                f"effective_max_symbols={engine.effective_max_symbols}"
            )
            return True

        return False

    async def check_no_ohlcv(
        self,
        symbol: str,
        display_symbol: str,
        assigned_tf: str,
        ohlcv_data: Dict[str, Any],
    ) -> bool:
        """Check if no OHLCV data is available for the symbol.

        Returns True if the symbol should be skipped (caller should return),
        False if processing should continue.
        """
        engine = self.engine
        no_ohlcv = (
            not ohlcv_data
            or all(len(candles) == 0 for candles in ohlcv_data.values())
        )
        if not no_ohlcv:
            return False

        logger.info(
            f"Skipping {symbol}: no OHLCV data – market data unavailable."
        )
        # Find the most recent OHLCV timestamp across all timeframes
        last_data_ts = None
        last_data_tf = None
        for tf in settings.OHLCV_TIMEFRAMES:
            try:
                ts = await asyncio.to_thread(get_latest_ohlcv_timestamp, symbol, tf)
                if ts is not None and (last_data_ts is None or ts > last_data_ts):
                    last_data_ts = ts
                    last_data_tf = tf
            except (ValueError, TypeError, KeyError, ConnectionError, TimeoutError, OSError):
                pass

        if last_data_ts is not None:
            age_seconds = time.time() - (last_data_ts / 1000.0)
            if age_seconds < 3600:
                age_str = f"{age_seconds/60:.0f} minutes ago"
            elif age_seconds < 86400:
                age_str = f"{age_seconds/3600:.1f} hours ago"
            else:
                age_str = f"{age_seconds/86400:.1f} days ago"
            msg = (
                f"⚠️ Skipping {display_symbol}: no OHLCV data available. "
                f"Last data: {last_data_tf} candle from {age_str}. "
                f"Try a manual force-download via the dashboard or Telegram."
            )
        else:
            msg = (
                f"⚠️ Skipping {display_symbol}: no OHLCV data available. "
                f"No historical data found in database. "
                f"Run a force-download via the dashboard or Telegram to populate market data."
            )

        no_ohlcv_notify_key = f"trading:no_ohlcv_notify:{symbol}"
        should_notify = True
        try:
            last_notify_raw = await asyncio.to_thread(engine.redis.get, no_ohlcv_notify_key)
            if last_notify_raw:
                if (time.time() - float(last_notify_raw)) < settings.STALENESS_NOTIFY_THRESHOLD_SECONDS:
                    should_notify = False
        except (ValueError, TypeError, ConnectionError, TimeoutError, OSError):
            pass

        if should_notify and engine.notifier:
            await engine.notifier.send_notification(
                msg,
                summary={
                    "symbol": symbol,
                    "action": "SKIP",
                    "reason": "No OHLCV data",
                    "last_data_timestamp": last_data_ts,
                    "last_data_timeframe": last_data_tf,
                }
            )
            try:
                await asyncio.to_thread(engine.redis.setex, no_ohlcv_notify_key, 3600, str(time.time()))
            except (ValueError, TypeError, ConnectionError, TimeoutError, OSError):
                pass

        async with engine._eval_state_lock:
            engine._force_eval.pop(symbol, None)
        return True

    async def handle_triggered_flags(
        self,
        symbol: str,
        display_symbol: str,
        signal: Signal,
        validated: Signal,
        assigned_tf: str,
        current_price: float,
        atr: Optional[float],
        ticker: Dict[str, Any],
        max_hold_expired: bool,
        stop_loss_triggered: bool,
        take_profit_triggered: bool,
        partial_tp_triggered: bool,
        dust_sweep_triggered: bool,
        strategy_model_type: str,
        llm_provider: Optional[str],
        llm_model: Optional[str],
    ) -> bool:
        """Handle triggered position flags (max hold, stop loss, take profit, partial TP, dust sweep).

        Returns True if the caller should return immediately (flag was handled).
        Returns False if the caller should continue with normal execution.
        """
        engine = self.engine
        params = signal.strategy_params or {}

        # --- Handle max‑hold‑expired LLM decision ---
        if max_hold_expired and signal.action == "HOLD":
            new_max_hold = params.get("max_hold_time_seconds") if params else None
            if new_max_hold is not None and new_max_hold > 0:
                logger.info(f"LLM extended max hold time for {symbol} to {new_max_hold}s")
                if symbol in engine.positions:
                    async with engine._positions_lock:
                        engine.positions[symbol]["max_hold_time_seconds"] = new_max_hold
                        engine.positions[symbol]["timestamp"] = int(time.time() * 1000)
                        engine.positions[symbol].pop("_max_hold_expired", None)
                        engine.positions[symbol].pop("_max_hold_expired_count", None)
                for symbol_entry in engine.current_symbols:
                    if symbol_entry["symbol"] == symbol:
                        symbol_entry["entry_time"] = time.time()
                        break
                if engine.notifier:
                    await engine.notifier.send_notification(
                        f"⏰ Max hold time for {display_symbol} extended to {new_max_hold}s by LLM.\n"
                        f"Reasoning: {validated.reasoning}",
                        summary={
                            "symbol": symbol,
                            "action": "HOLD",
                            "reason": validated.reasoning,
                            "new_max_hold_seconds": new_max_hold,
                            "model_type": strategy_model_type,
                            "llm_provider": llm_provider,
                            "llm_model": llm_model,
                        }
                    )
                await self.event_bus.publish(
                    "update_position_params",
                    symbol, params, signal.indicator_config, assigned_tf, current_price, atr,
                )
                engine._state_dirty = True
            else:
                logger.warning(
                    f"LLM returned HOLD without new max_hold_time_seconds for {symbol} "
                    f"after max hold expiry – forcing SELL."
                )
                if engine.notifier:
                    await engine.notifier.send_notification(
                        f"⏰ LLM did not extend hold time for {display_symbol} – closing position.",
                        summary={
                            "symbol": symbol, "action": "SELL",
                            "reason": "Max hold expired, LLM did not extend",
                            "exit_reason": "max_hold_time_llm_no_extend",
                            "model_type": strategy_model_type, "llm_provider": llm_provider, "llm_model": llm_model,
                        }
                    )
                await self.event_bus.publish(
                    "execute_signal",
                    symbol, Signal(action="SELL", confidence=1.0, reasoning="Max hold expired, LLM did not extend"),
                    exit_reason="max_hold_time_llm_no_extend"
                )
            return True

        # --- Handle stop-loss-triggered LLM decision ---
        if stop_loss_triggered and signal.action == "HOLD":
            new_params = signal.strategy_params or {}
            new_stop_method = new_params.get("stop_loss_method", "fixed")
            new_stop_pct = None
            if new_stop_method == "atr_multiple" and atr is not None and atr > 0:
                atr_mult = new_params.get("stop_loss_atr_multiple")
                if atr_mult is not None:
                    new_stop_pct = (atr_mult * atr) / current_price
            else:
                new_stop_pct = new_params.get("stop_loss_pct")

            if new_stop_pct is not None and new_stop_pct > 0:
                logger.info(
                    f"LLM decided to hold {symbol} after stop-loss trigger, "
                    f"new stop_loss_pct={new_stop_pct:.4%}"
                )
                if symbol in engine.positions:
                    async with engine._positions_lock:
                        engine.positions[symbol]["stop_loss"] = current_price * (1 - new_stop_pct)
                        engine.positions[symbol].pop("_stop_loss_triggered", None)
                        engine.positions[symbol].pop("_stop_loss_review_count", None)
                    await self.event_bus.publish(
                        "update_position_params",
                        symbol, new_params, signal.indicator_config, assigned_tf, current_price, atr,
                    )
                    engine._state_dirty = True
                if engine.notifier:
                    await engine.notifier.send_notification(
                        f"🔄 {display_symbol}: LLM adjusted stop-loss to {new_stop_pct:.4%} – holding.\n"
                        f"Reasoning: {validated.reasoning}",
                        summary={
                            "symbol": symbol, "action": "HOLD", "reason": validated.reasoning,
                            "new_stop_loss_pct": new_stop_pct, "model_type": strategy_model_type,
                            "llm_provider": llm_provider, "llm_model": llm_model,
                        }
                    )
                return True
            else:
                logger.warning(
                    f"LLM returned HOLD for {symbol} after stop-loss trigger but did not provide "
                    f"a new stop-loss. Forcing SELL."
                )
                if engine.notifier:
                    await engine.notifier.send_notification(
                        f"⛔ {display_symbol}: LLM did not provide new stop-loss – selling.",
                        summary={
                            "symbol": symbol, "action": "SELL",
                            "reason": "Stop-loss triggered, LLM did not provide new stop",
                            "exit_reason": "stop_loss_llm_no_action",
                            "model_type": strategy_model_type, "llm_provider": llm_provider, "llm_model": llm_model,
                        }
                    )
                await self.event_bus.publish(
                    "execute_signal",
                    symbol, Signal(action="SELL", confidence=1.0, reasoning="Stop-loss triggered, LLM did not provide new stop"),
                    exit_reason="stop_loss_llm_no_action"
                )
                return True

        elif stop_loss_triggered and signal.action == "SELL":
            if symbol in engine.positions:
                async with engine._positions_lock:
                    engine.positions[symbol].pop("_stop_loss_triggered", None)
                    engine.positions[symbol].pop("_stop_loss_review_count", None)
            # Continue to normal SELL execution

        # --- Handle take-profit-triggered LLM decision ---
        if take_profit_triggered and signal.action == "HOLD":
            new_params = signal.strategy_params or {}
            new_tp_pct = new_params.get("take_profit_pct")
            if new_tp_pct is not None and new_tp_pct > 0:
                logger.info(
                    f"LLM decided to hold {symbol} after take-profit trigger, "
                    f"new take_profit_pct={new_tp_pct:.4%}"
                )
                if symbol in engine.positions:
                    async with engine._positions_lock:
                        engine.positions[symbol]["take_profit"] = current_price * (1 + new_tp_pct)
                        engine.positions[symbol].pop("_take_profit_triggered", None)
                        engine.positions[symbol].pop("_take_profit_review_count", None)
                    await self.event_bus.publish(
                        "update_position_params",
                        symbol, new_params, signal.indicator_config, assigned_tf, current_price, atr,
                    )
                    engine._state_dirty = True
                if engine.notifier:
                    await engine.notifier.send_notification(
                        f"🔄 {display_symbol}: LLM adjusted take-profit to {new_tp_pct:.4%} – holding.\n"
                        f"Reasoning: {validated.reasoning}",
                        summary={
                            "symbol": symbol, "action": "HOLD", "reason": validated.reasoning,
                            "new_take_profit_pct": new_tp_pct, "model_type": strategy_model_type,
                            "llm_provider": llm_provider, "llm_model": llm_model,
                        }
                    )
                return True
            else:
                logger.warning(
                    f"LLM returned HOLD for {symbol} after take-profit trigger but did not provide "
                    f"a new take-profit. Forcing SELL."
                )
                if engine.notifier:
                    await engine.notifier.send_notification(
                        f"🎯 {display_symbol}: LLM did not provide new take-profit – selling.",
                        summary={
                            "symbol": symbol, "action": "SELL",
                            "reason": "Take-profit triggered, LLM did not provide new take-profit",
                            "exit_reason": "take_profit_llm_no_action",
                            "model_type": strategy_model_type, "llm_provider": llm_provider, "llm_model": llm_model,
                        }
                    )
                await self.event_bus.publish(
                    "execute_signal",
                    symbol, Signal(action="SELL", confidence=1.0, reasoning="Take-profit triggered, LLM did not provide new take-profit"),
                    exit_reason="take_profit_llm_no_action"
                )
                return True

        elif take_profit_triggered and signal.action == "SELL":
            if symbol in engine.positions:
                async with engine._positions_lock:
                    engine.positions[symbol].pop("_take_profit_triggered", None)
                    engine.positions[symbol].pop("_take_profit_review_count", None)
            # Continue to normal SELL execution

        # --- Handle partial TP triggered ---
        if partial_tp_triggered and signal.action == "HOLD":
            new_levels = params.get("partial_take_profit_levels") if params else None
            if new_levels is not None:
                async with engine._positions_lock:
                    engine.positions[symbol]["partial_take_profit_levels"] = new_levels
                    engine.positions[symbol].pop("_partial_tp_triggered", None)
                    engine.positions[symbol].pop("_partial_tp_triggered_single", None)
                    engine.positions[symbol].pop("_partial_tp_review_count", None)
                    engine.positions[symbol].pop("_partial_tp_single_review_count", None)
                    engine.positions[symbol].pop("_partial_tp_triggered_levels", None)
                    engine.positions[symbol]["partial_tp_levels_triggered"] = []
                    engine.positions[symbol]["partial_tp_depth_wait_start"] = {}
                logger.info(f"LLM updated partial TP levels for {symbol}")
                await self.event_bus.publish(
                    "update_position_params",
                    symbol, params, signal.indicator_config, assigned_tf, current_price, atr,
                )
                engine._state_dirty = True
                if engine.notifier:
                    await engine.notifier.send_notification(
                        f"🔄 {display_symbol}: LLM adjusted partial TP levels – holding.",
                        summary={"symbol": symbol, "action": "HOLD", "reason": "Partial TP levels adjusted by LLM", "model_type": strategy_model_type, "llm_provider": llm_provider, "llm_model": llm_model}
                    )
                return True
            else:
                logger.info(f"LLM did not update partial TP levels for {symbol}, executing triggered level(s)")
                if engine.positions[symbol].get("_partial_tp_triggered_single"):
                    await self.event_bus.publish("execute_partial_tp_single", symbol, current_price, None, ticker)
                    async with engine._positions_lock:
                        engine.positions[symbol].pop("_partial_tp_triggered_single", None)
                        engine.positions[symbol].pop("_partial_tp_single_review_count", None)
                if engine.positions[symbol].get("_partial_tp_triggered"):
                    for lvl in engine.positions[symbol].get("_partial_tp_triggered_levels", []):
                        await self.event_bus.publish("execute_partial_tp_level", symbol, lvl, current_price, None, ticker)
                    async with engine._positions_lock:
                        engine.positions[symbol].pop("_partial_tp_triggered", None)
                        engine.positions[symbol].pop("_partial_tp_review_count", None)
                        engine.positions[symbol].pop("_partial_tp_triggered_levels", None)
                return True

        elif partial_tp_triggered and signal.action == "SELL":
            async with engine._positions_lock:
                engine.positions[symbol].pop("_partial_tp_triggered", None)
                engine.positions[symbol].pop("_partial_tp_triggered_single", None)
                engine.positions[symbol].pop("_partial_tp_review_count", None)
                engine.positions[symbol].pop("_partial_tp_single_review_count", None)
                engine.positions[symbol].pop("_partial_tp_triggered_levels", None)
            # Continue to normal SELL execution

        # --- Handle dust sweep triggered ---
        if dust_sweep_triggered and signal.action == "HOLD":
            async with engine._positions_lock:
                engine.positions[symbol].pop("_dust_sweep_triggered", None)
                if engine.positions[symbol].get("_dust_keep_since") is None:
                    engine.positions[symbol]["_dust_keep_since"] = time.time()
            engine._state_dirty = True
            logger.info(f"LLM decided to hold dust for {symbol}")
            if engine.notifier:
                await engine.notifier.send_notification(
                    f"🧹 {display_symbol}: LLM decided to keep dust – holding.",
                    summary={"symbol": symbol, "action": "HOLD", "reason": "Dust kept by LLM"}
                )
            return True
        elif dust_sweep_triggered and signal.action == "SELL":
            async with engine._positions_lock:
                engine.positions[symbol].pop("_dust_sweep_triggered", None)
                engine.positions[symbol].pop("_dust_sweep_review_count", None)
            logger.info(f"LLM decided to sell dust for {symbol}")
            await self.event_bus.publish("sweep_dust", symbol)
            return True

        return False

    async def handle_entry_condition(
        self,
        symbol: str,
        display_symbol: str,
        validated: Signal,
        assigned_tf: str,
        tf_seconds: int,
        trading_paused: bool,
    ) -> bool:
        """Handle entry condition for a BUY signal.

        Returns True if the entry was deferred (caller should return),
        False if no entry condition is present (caller should continue to execute).
        """
        engine = self.engine

        if validated.action != "BUY" or validated.entry_condition is None or trading_paused:
            return False

        etype = validated.entry_condition.get("type")
        if etype == "delay":
            # Delay entries are simple time-based waits – schedule directly
            delay_sec = validated.entry_condition.get("delay_seconds", 0)
            logger.info(f"Scheduling delayed BUY for {symbol} in {delay_sec}s")
            task = asyncio.create_task(
                engine._execute_delayed_entry(symbol, validated, assigned_tf, delay_sec)
            )
            engine._delayed_entry_tasks.add(task)
            task.add_done_callback(engine._delayed_entry_tasks.discard)
            if engine.notifier:
                await engine.notifier.send_notification(
                    f"⏳ Delayed entry for {display_symbol} – executing in {delay_sec}s.",
                    summary={
                        "symbol": symbol,
                        "action": "WAIT",
                        "reason": "Delay entry scheduled",
                        "delay_seconds": delay_sec,
                    }
                )
            return True

        timeout = validated.entry_condition.get("timeout_seconds", 600)
        # Enforce a minimum based on the candle timeframe
        min_timeout = max(300, int(settings.ENTRY_CONDITION_MIN_TIMEOUT_MULT * tf_seconds))
        # Cap the minimum timeout to avoid absurd values for very long timeframes
        min_timeout = min(min_timeout, 15_552_000)  # 180 days
        if timeout < min_timeout:
            logger.info(
                f"Entry condition timeout for {symbol} too short ({timeout}s), "
                f"clamping to minimum {min_timeout}s (timeframe={assigned_tf})"
            )
            timeout = min_timeout
        deadline = time.time() + timeout
        # Store for background checking – do NOT block the main loop
        async with engine._pending_entries_lock:
            engine._pending_entries[symbol] = {
                "signal": validated,
                "deadline": deadline,
                "timeframe": assigned_tf,
                "condition": validated.entry_condition,
            }
        logger.info(
            f"Queued entry condition for {symbol} (type={etype}, deadline in {timeout}s). "
            f"Will monitor in background."
        )
        if engine.notifier:
            await engine.notifier.send_notification(
                f"⏳ Waiting for entry condition on {display_symbol} "
                f"(type={etype}, timeout {timeout}s).",
                summary={
                    "symbol": symbol,
                    "action": "WAIT",
                    "reason": "Entry condition pending",
                }
            )
        return True

    async def check_trade_filters(
        self,
        symbol: str,
        display_symbol: str,
        validated: Signal,
        params: Dict[str, Any],
    ) -> bool:
        """Check trade filters (confidence thresholds, SELL without position).

        Returns True if the trade should be skipped (caller should return),
        False if processing should continue.
        """
        engine = self.engine

        # --- Global confidence rejection threshold (set during stock selection) ---
        if validated.action == "BUY":
            conf_rejection_raw = await engine.config_service.get_config("confidence_rejection_threshold")
            if conf_rejection_raw:
                try:
                    conf_threshold = float(conf_rejection_raw)
                    if conf_threshold > 0 and validated.confidence < conf_threshold:
                        logger.info(f"Skipping {symbol}: confidence {validated.confidence:.2f} below global rejection threshold {conf_threshold:.2f}")
                        if engine.notifier:
                            await engine.notifier.send_notification(
                                f"⚠️ Skipping {display_symbol}: confidence {validated.confidence:.2f} below threshold {conf_threshold:.2f}",
                                summary={
                                    "symbol": symbol,
                                    "action": "SKIP",
                                    "reason": "Confidence below rejection threshold",
                                    "confidence": validated.confidence,
                                    "threshold": conf_threshold,
                                }
                            )
                        return True
                except (ValueError, TypeError):
                    pass

        min_conf = params.get("min_confidence")
        if min_conf is not None and validated.confidence < min_conf:
            logger.info(f"Skipping {symbol}: confidence {validated.confidence:.2f} below LLM min {min_conf:.2f}")
            if engine.notifier:
                await engine.notifier.send_notification(
                    f"⚠️ Skipping {display_symbol}: confidence too low ({validated.confidence:.2f})",
                    summary={
                        "symbol": symbol,
                        "action": "SKIP",
                        "reason": "Confidence too low",
                        "confidence": validated.confidence,
                        "min_confidence": min_conf,
                    }
                )
            return True

        # Prevent SELL without an open position (no shorting)
        if validated.action == "SELL" and symbol not in engine.positions:
            logger.info(f"Skipping SELL for {symbol}: no open position.")
            if engine.notifier:
                await engine.notifier.send_notification(
                    f"⚠️ Skipping SELL for {display_symbol}: no open position.",
                    summary={
                        "symbol": symbol,
                        "action": "SKIP",
                        "reason": "No open position",
                    }
                )
            return True

        return False

    async def check_sector_concentration(
        self,
        symbol: str,
        display_symbol: str,
        assigned_tf: str,
    ) -> bool:
        """Check if buying this symbol would exceed the sector concentration limit.

        Returns True if the trade should be skipped (caller should return),
        False if processing should continue.
        """
        engine = self.engine

        current_sector = None
        for entry in engine.current_symbols:
            if entry["symbol"] == symbol:
                current_sector = entry.get("sector")
                break

        if not current_sector:
            return False

        max_positions_per_sector_raw = await engine.config_service.get_config("max_positions_per_sector")
        if max_positions_per_sector_raw:
            try:
                max_positions_per_sector = int(max_positions_per_sector_raw)
            except ValueError:
                max_positions_per_sector = None
        else:
            max_positions_per_sector = None

        if max_positions_per_sector is None or max_positions_per_sector <= 0:
            return False

        sector_count = 0
        for pos_sym in engine.positions.keys():
            for entry in engine.current_symbols:
                if entry["symbol"] == pos_sym and entry.get("sector") == current_sector:
                    sector_count += 1
                    break

        if sector_count >= max_positions_per_sector:
            logger.info(
                f"Skipping BUY {symbol}: sector '{current_sector}' already has "
                f"{sector_count} open positions (max {max_positions_per_sector})"
            )
            if engine.notifier:
                stock_name = await engine._get_stock_name(symbol)
                display = engine._format_symbol_display(symbol, stock_name, assigned_tf)
                await engine.notifier.send_notification(
                    f"⚠️ Skipping BUY {display}: sector '{current_sector}' concentration limit reached ({sector_count}/{max_positions_per_sector})",
                    summary={
                        "symbol": symbol,
                        "action": "SKIP",
                        "reason": "Sector concentration limit",
                        "sector": current_sector,
                        "sector_count": sector_count,
                        "max_positions_per_sector": max_positions_per_sector,
                    }
                )
            return True

        return False

    async def _get_sentiment_str(self, symbol: str) -> str:
        """Get a short news sentiment string for notifications, including an LLM summary."""
        if not settings.NEWS_ENABLED:
            return ""
        try:
            base_symbol = symbol.split("/")[0] if "/" in symbol else symbol
            agg_sent = await asyncio.to_thread(get_aggregate_sentiment_from_db, base_symbol, max_age_seconds=settings.NEWS_CACHE_TTL_SECONDS)
            if not agg_sent:
                return ""

            compound = agg_sent["avg_compound"]
            sentiment_label = "positive" if compound > 0.05 else "negative" if compound < -0.05 else "neutral"
            total = agg_sent["total_articles"]

            # Try to get an LLM-generated summary of the news
            summary = ""
            try:
                summary_raw = await asyncio.to_thread(get_cached_news_summary, symbol)
                if isinstance(summary_raw, dict):
                    summary = summary_raw.get("summary", "")
                else:
                    summary = summary_raw
                if summary in ("No recent news.", "Could not generate summary."):
                    summary = ""
            except (ValueError, TypeError, KeyError, ConnectionError, TimeoutError, OSError, json.JSONDecodeError):
                pass  # fallback to no summary

            base = f"📰 (sentiment: {compound:+.2f}[{sentiment_label}], {total} articles)"
            if summary:
                return f"{base} – {summary}"
            return base
        except Exception:
            pass
        return ""

    async def log_and_notify_decision(
        self,
        data: DecisionContext,
        validated: Signal,
        backtest_stats: Optional[Dict[str, Any]],
    ) -> None:
        """Log the decision, record it in recent_signals, and send notification."""
        engine = self.engine

        logger.info(f"Decision for {data.symbol}: {validated.action} (confidence: {validated.confidence:.2f})")

        # Store the last decision for the next prompt cycle
        params = data.signal.strategy_params
        engine._last_decisions[data.symbol] = {
            "action": validated.action,
            "confidence": validated.confidence,
            "reasoning": validated.reasoning[:300],
            "strategy_type": data.signal.strategy_type,
            "timestamp": time.time(),
            "stop_loss_pct": params.get("stop_loss_pct") if params else None,
            "take_profit_pct": params.get("take_profit_pct") if params else None,
            "position_size_fraction": params.get("position_size_fraction") if params else None,
            "stop_loss_method": params.get("stop_loss_method") if params else None,
        }
        engine._state_dirty = True

        # Compute trade amount for display in the signals card
        _params = data.signal.strategy_params or {}
        _psf = _params.get("position_size_fraction")
        if validated.action == "BUY" and _psf is not None:
            _trade_amount = data.base_balance * float(_psf)
        elif validated.action == "SELL" and data.symbol in engine.positions:
            _pos = engine.positions[data.symbol]
            _trade_amount = _pos.get("amount", 0) * data.current_price
        else:
            _trade_amount = 0.0

        # Extract strategy parameters for the signal detail modal
        _sig_params = data.signal.strategy_params or {}
        _entry_cond_str = None
        if validated.entry_condition:
            _ec = validated.entry_condition
            _etype = _ec.get("type", "")
            if _etype == "limit_price":
                _entry_cond_str = f"Wait for price to drop to {_ec.get('price', '?')} (timeout: {_ec.get('timeout_seconds', '?')}s)"
            elif _etype == "rsi_threshold":
                _entry_cond_str = f"Wait for RSI(14) to fall below {_ec.get('rsi_below', '?')} (timeout: {_ec.get('timeout_seconds', '?')}s)"
            elif _etype == "delay":
                _entry_cond_str = f"Wait {_ec.get('delay_seconds', '?')}s before executing"
            elif _etype == "indicator_combo":
                _conds = _ec.get("conditions", [])
                _cond_strs = []
                for c in _conds:
                    _cond_strs.append(f"{c.get('indicator','?')} {c.get('direction','?')} {c.get('threshold','?')}")
                _entry_cond_str = f"Wait for ALL: {', '.join(_cond_strs)} (timeout: {_ec.get('timeout_seconds', '?')}s)"
        _sl_method = _sig_params.get("stop_loss_method", "fixed")
        _sl_str = ""
        if _sl_method == "atr_multiple":
            _sl_str = f"ATR × {_sig_params.get('stop_loss_atr_multiple', '?')} (fallback: {_sig_params.get('stop_loss_pct', '?')})"
        else:
            _sl_str = f"{_sig_params.get('stop_loss_pct', '?')}"
        _tp_str = ""
        if _sig_params.get("take_profit_atr_multiple"):
            _tp_str = f"ATR × {_sig_params.get('take_profit_atr_multiple', '?')} (fallback: {_sig_params.get('take_profit_pct', '?')})"
        else:
            _tp_str = f"{_sig_params.get('take_profit_pct', '?')}"

        # Record signal for the web dashboard
        signal_record = {
            "symbol": data.symbol,
            "display_symbol": data.display_symbol,
            "stock_name": data.stock_name,
            "timeframe": data.assigned_tf,
            "action": validated.action,
            "confidence": validated.confidence,
            "reasoning": validated.reasoning or "",
            "strategy_type": data.signal.strategy_type,
            "model_type": getattr(validated, 'model_type', None),
            "llm_provider": data.llm_provider,
            "llm_model": data.llm_model,
            "trade_amount": round(_trade_amount, 2),
            "base_currency": engine.base_currency,
            "timestamp": time.time(),
            "entry_condition": _entry_cond_str,
            "stop_loss": _sl_str,
            "take_profit": _tp_str,
            "position_size_fraction": _sig_params.get("position_size_fraction"),
            "trailing_stop": _sig_params.get("trailing_stop"),
            "trailing_stop_distance_pct": _sig_params.get("trailing_stop_distance_pct"),
            "max_hold_time_seconds": _sig_params.get("max_hold_time_seconds"),
            "cooldown_after_loss_seconds": _sig_params.get("cooldown_after_loss_seconds"),
            "order_type": data.signal.order_type,
            "limit_price": _sig_params.get("limit_price"),
        }
        await asyncio.to_thread(insert_signal, signal_record)

        if engine.notifier:
            emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⏸️"}.get(validated.action, "❓")
            paused_tag = " (PAUSED)" if data.trading_paused and validated.action == "BUY" else ""
            # Build a short indicator summary
            ind_parts = []
            if data.rsi is not None:
                ind_parts.append(f"RSI={data.rsi:.1f}")
            if data.macd is not None and data.macd_signal is not None:
                ind_parts.append(f"MACD={data.macd:.4f}/{data.macd_signal:.4f}")
                if data.macd_hist is not None:
                    ind_parts.append(f"Hist={data.macd_hist:.4f}")
            if data.bb_upper is not None:
                ind_parts.append(f"BB={data.bb_lower:.2f}/{data.bb_middle:.2f}/{data.bb_upper:.2f}")
            if data.ema_9 is not None and data.ema_21 is not None:
                ind_parts.append(f"EMA9/21={data.ema_9:.2f}/{data.ema_21:.2f}")
            if data.stochastic_k is not None:
                ind_parts.append(f"StochK={data.stochastic_k:.1f}")
                if data.stochastic_d is not None:
                    ind_parts.append(f"StochD={data.stochastic_d:.1f}")
            if data.adx is not None:
                ind_parts.append(f"ADX={data.adx:.1f}")
                if data.plus_di is not None and data.minus_di is not None:
                    ind_parts.append(f"+DI={data.plus_di:.1f}/-DI={data.minus_di:.1f}")
            if data.atr is not None:
                ind_parts.append(f"ATR={data.atr:.4f}")
            if data.obv is not None:
                ind_parts.append(f"OBV={data.obv:.2f}")
            if data.mfi is not None:
                ind_parts.append(f"MFI={data.mfi:.2f}")
            if data.cci is not None:
                ind_parts.append(f"CCI={data.cci:.2f}")
            if data.williams_r is not None:
                ind_parts.append(f"WR={data.williams_r:.2f}")
            if data.ichimoku is not None:
                ind_parts.append(f"Ichi T={data.ichimoku['tenkan_sen']:.2f}/K={data.ichimoku['kijun_sen']:.2f}")
                ind_parts.append(f"Cloud={data.ichimoku['cloud_bottom']:.2f}-{data.ichimoku['cloud_top']:.2f}")
            if data.donchian_channels is not None:
                ind_parts.append(f"Donch={data.donchian_channels['lower']:.2f}/{data.donchian_channels['middle']:.2f}/{data.donchian_channels['upper']:.2f}")
            if data.parabolic_sar is not None:
                ind_parts.append(f"SAR={data.parabolic_sar:.4f}")
            if data.keltner_channels is not None:
                ind_parts.append(f"Kelt={data.keltner_channels['lower']:.4f}/{data.keltner_channels['middle']:.4f}/{data.keltner_channels['upper']:.4f}")
            indicator_str = " | ".join(ind_parts) if ind_parts else "No indicators (insufficient OHLCV data)"
            sentiment_str = await self._get_sentiment_str(data.symbol)
            reasoning_str = f" – {validated.reasoning}" if validated.reasoning else ""
            msg = f"{emoji} {data.display_symbol}: {validated.action} (confidence: {validated.confidence:.2f}){reasoning_str}{paused_tag}"
            if sentiment_str:
                msg += f"\n{sentiment_str}"
            if getattr(validated, 'backtest_summary', None):
                msg += f"\n📈 Backtest: {validated.backtest_summary}"
            msg += f"\n📊 {indicator_str}"
            # Build summary dict for logging
            decision_summary = {
                "symbol": data.symbol,
                "action": validated.action,
                "confidence": validated.confidence,
                "reason": validated.reasoning[:200],
                "sentiment": data.aggregate_sentiment,
                "indicators": {
                    "rsi": data.rsi,
                    "macd": data.macd,
                    "macd_signal": data.macd_signal,
                    "atr": data.atr,
                    "adx": data.adx,
                    "bb_upper": data.bb_upper,
                    "bb_lower": data.bb_lower,
                    "ema_9": data.ema_9,
                    "ema_21": data.ema_21,
                    "stochastic_k": data.stochastic_k,
                    "mfi": data.mfi,
                    "cci": data.cci,
                    "williams_r": data.williams_r,
                    "ichimoku": data.ichimoku,
                    "donchian_channels": data.donchian_channels,
                },
                "backtest": backtest_stats,
                "strategy_type": data.signal.strategy_type,
                "market_regime": data.market_regime,
                "model_type": getattr(validated, 'model_type', None),
                "llm_provider": data.llm_provider,
                "llm_model": data.llm_model,
            }
            await engine.notifier.send_notification(msg, summary=decision_summary)

    async def run_step1a_llm_call(
        self,
        symbol: str,
        display_symbol: str,
        analysis_prompt: str,
        system_prompt: str,
        market_hash: str,
        strategy_model_type: str,
        effective_temp: float,
        current_price: float,
        rsi: Optional[float],
        macd_hist: Optional[float],
        is_critical: bool,
        critical_reason: Optional[str],
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str], bool]:
        """Run the Step 1a LLM call and handle timeouts/retries.

        Returns (analysis_result, llm_provider, llm_model, should_return).
        If should_return is True, the caller should return immediately.
        """
        engine = self.engine
        analysis_result = None
        llm_provider = None
        llm_model = None

        # --- LLM circuit breaker: skip calls if too many consecutive failures ---
        cb_active = False
        try:
            cb_raw = await asyncio.to_thread(engine.redis.get, "llm:circuit_breaker")
            if cb_raw:
                cb_data = json.loads(cb_raw)
                if time.time() < cb_data.get("active_until", 0):
                    cb_active = True
        except (ValueError, TypeError, ConnectionError, TimeoutError, OSError, json.JSONDecodeError):
            pass

        if cb_active:
            logger.warning(f"LLM circuit breaker active for {symbol}, skipping LLM call.")
            # Clear _force_eval to break the retry loop
            async with engine._eval_state_lock:
                engine._force_eval.pop(symbol, None)
            return None, None, None, False

        try:
            step1a_result = await asyncio.wait_for(
                asyncio.to_thread(
                    get_cached_llm_response,
                    compact_prompt(analysis_prompt),
                    system_prompt,
                    60,
                    market_hash=market_hash,
                    model_type=strategy_model_type,
                    temperature=effective_temp,
                ),
                timeout=settings.LLM_TIMEOUT
            )
            step1a_response = step1a_result["response"]
            llm_provider = step1a_result["provider"]
            llm_model = step1a_result["model"]
            logger.info(f"LLM Step 1a (analysis) completed for {symbol} (provider={llm_provider}, model={llm_model})")
            analysis_result = self._parse_analysis_response(step1a_response)
            if analysis_result is None:
                logger.warning(f"Failed to parse Step 1a analysis response for {symbol}. Retrying with correction.")
                correction_prompt = (
                    "Your previous response was not valid JSON. "
                    "You MUST output ONLY a single JSON object with fields: "
                    '"action", "confidence", "reasoning", "strategy_direction". '
                    "No markdown fences, no explanations, no extra text. "
                    "Here is the original request:\n\n" + analysis_prompt
                )
                retry_result = await asyncio.wait_for(
                    asyncio.to_thread(
                        get_cached_llm_response,
                        compact_prompt(correction_prompt),
                        system_prompt, 30,
                        model_type="actuator",
                        temperature=effective_temp,
                    ),
                    timeout=settings.LLM_TIMEOUT
                )
                analysis_result = self._parse_analysis_response(retry_result["response"])
                llm_provider = retry_result["provider"]
                llm_model = retry_result["model"]
            # Reset consecutive LLM failure counter on success
            try:
                await asyncio.to_thread(engine.redis.delete, "llm:consecutive_failures")
            except (ConnectionError, TimeoutError, OSError):
                pass
            # Update snapshot after a real LLM call
            self._update_last_eval_snapshot(symbol, current_price, rsi, macd_hist)
            async with engine._eval_state_lock:
                engine._force_eval.pop(symbol, None)
        except asyncio.TimeoutError:
            logger.warning(f"LLM Step 1a (analysis) timed out for {symbol}.")
            if is_critical and critical_reason is not None:
                logger.warning(f"Forcing SELL for {symbol} due to {critical_reason}")
                if engine.notifier:
                    await engine.notifier.send_notification(
                        f"⏱️ LLM timeout for {display_symbol} with critical flag – forcing SELL.",
                        summary={"symbol": symbol, "action": "SELL", "reason": critical_reason, "model_type": strategy_model_type}
                    )
                await self.event_bus.publish(
                    "execute_signal",
                    symbol,
                    Signal(action="SELL", confidence=1.0, reasoning=critical_reason),
                    exit_reason=critical_reason.replace(" ", "_").lower()
                )
                return None, None, None, True
            # Non-critical timeout: fall through to fallback HOLD
            async with engine._eval_state_lock:
                engine._force_eval[symbol] = True  # Force retry on next cycle
            await self._increment_llm_failures()
            # Fall through to fallback HOLD below
        except (ConnectionError, TimeoutError, OSError, ValueError, TypeError, RuntimeError, json.JSONDecodeError) as e:
            logger.error(f"LLM Step 1a failed for {symbol}: {e}")
            async with engine._eval_state_lock:
                engine._force_eval[symbol] = True  # Force retry on next cycle
            await self._increment_llm_failures()
            # Fall through to fallback HOLD below

        return analysis_result, llm_provider, llm_model, False

    async def handle_step1a_fallback(
        self,
        symbol: str,
        analysis_result: Optional[Dict[str, Any]],
        has_position: bool,
        strategy_model_type: str,
        llm_provider: Optional[str],
        llm_model: Optional[str],
    ) -> Tuple[Signal, str, Optional[str], Optional[str], bool]:
        """Handle fallback HOLD signal when Step 1a fails or returns HOLD with no position.

        Returns (signal, combined_bt_summary, llm_provider, llm_model, skip_backtest).
        If signal is None, the caller should proceed with Step 1b and Step 2.
        """
        engine = self.engine
        if analysis_result is None:
            logger.warning(f"Step 1a analysis failed for {symbol} after all retries. Using fallback HOLD.")
            # Check if circuit breaker is active; if so, clear _force_eval to break retry loop
            cb_active = False
            try:
                cb_raw = await asyncio.to_thread(engine.redis.get, "llm:circuit_breaker")
                if cb_raw:
                    cb_data = json.loads(cb_raw)
                    if time.time() < cb_data.get("active_until", 0):
                        cb_active = True
            except (ValueError, TypeError, ConnectionError, TimeoutError, OSError, json.JSONDecodeError):
                pass
            if cb_active:
                async with engine._eval_state_lock:
                    engine._force_eval.pop(symbol, None)
            # Otherwise, keep _force_eval set to retry on the next cycle.
            # Create a fallback HOLD signal so the bot continues functioning
            preliminary_signal = self._create_fallback_hold_signal(
                symbol, "LLM Step 1a analysis failed after retries", strategy_model_type
            )
            signal = preliminary_signal
            llm_provider = "fallback"
            llm_model = "default_hold"
            combined_bt_summary = ""
            _skip_backtest = True
        # If analysis says HOLD with no position, skip parameter selection entirely
        elif analysis_result.get("action") == "HOLD" and not has_position:
            logger.info(f"Step 1a analysis returned HOLD with no position for {symbol}. Skipping Step 1b.")
            # Create a minimal preliminary signal for the notification flow
            preliminary_signal = Signal(
                action="HOLD",
                confidence=analysis_result.get("confidence", 0.0),
                reasoning=analysis_result.get("reasoning", ""),
            )
            preliminary_signal.model_type = strategy_model_type
            preliminary_signal.llm_provider = llm_provider or "fallback"
            preliminary_signal.llm_model = llm_model or "default_hold"
            # Skip backtests and Step 2 — go directly to notification
            signal = preliminary_signal
            combined_bt_summary = ""
            _skip_backtest = True
        else:
            signal = None
            combined_bt_summary = ""
            _skip_backtest = False

        return signal, combined_bt_summary, llm_provider, llm_model, _skip_backtest

    def _update_last_eval_snapshot(self, symbol: str, price: float, rsi: Optional[float], macd_hist: Optional[float]):
        self.engine._last_eval_snapshot[symbol] = {
            "timestamp": time.time(),
            "price": price,
            "rsi": rsi,
            "macd_hist": macd_hist,
        }

    async def _increment_llm_failures(self) -> None:
        """Increment the global LLM failure counter and activate circuit breaker if threshold reached."""
        engine = self.engine
        try:
            fail_count = await asyncio.to_thread(engine.redis.incr, "llm:consecutive_failures")
            await asyncio.to_thread(engine.redis.expire, "llm:consecutive_failures", 3600)

            # Read threshold and cooldown from config with defaults
            cb_threshold = 5
            cb_cooldown = 300
            try:
                raw = await engine.config_service.get_config("llm_circuit_breaker_threshold")
                if raw:
                    cb_threshold = int(raw)
                raw = await engine.config_service.get_config("llm_circuit_breaker_cooldown_seconds")
                if raw:
                    cb_cooldown = int(raw)
            except (ValueError, TypeError, ConnectionError, TimeoutError, OSError):
                pass

            if fail_count >= cb_threshold:
                cb_data = json.dumps({
                    "active_until": time.time() + cb_cooldown,
                    "fail_count": fail_count,
                })
                await asyncio.to_thread(engine.redis.setex, "llm:circuit_breaker", cb_cooldown, cb_data)
                logger.warning(
                    f"LLM circuit breaker activated after {fail_count} consecutive failures. "
                    f"Cooldown: {cb_cooldown}s."
                )
                if engine.notifier:
                    await engine.notifier.send_notification(
                        f"🔌 LLM circuit breaker activated after {fail_count} consecutive failures. "
                        f"LLM calls will be skipped for {cb_cooldown}s.",
                        summary={
                            "action": "CIRCUIT_BREAKER",
                            "reason": f"Consecutive LLM failures: {fail_count}",
                        }
                    )
        except (ConnectionError, TimeoutError, OSError) as e:
            logger.debug(f"Could not increment LLM failure counter: {e}")

    def _create_fallback_hold_signal(
        self, symbol: str, reason: str, strategy_model_type: str
    ) -> Signal:
        """Create a fallback HOLD signal when LLM calls fail or return unparseable JSON."""
        signal = Signal(
            action="HOLD",
            confidence=0.0,
            reasoning=reason,
        )
        signal.model_type = strategy_model_type
        signal.llm_provider = "fallback"
        signal.llm_model = "default_hold"
        return signal

    async def process_post_llm_decision(
        self,
        data: DecisionContext,
    ) -> None:
        """Validate the LLM signal, log/notify, and execute if all checks pass."""
        engine = self.engine

        # Ensure llm_provider and llm_model are never None for notifications/signals
        llm_provider = data.llm_provider or "fallback"
        llm_model = data.llm_model or "default_hold"

        validated = validate_signal(
            data.signal,
            atr=data.atr,
            price=data.current_price,
            timeframe_seconds=data.tf_seconds,
            min_stop_atr_mult=data.min_stop_atr_mult,
            min_hold_time_mult=data.min_hold_time_mult,
            global_min_risk_reward_ratio=data.global_min_rr,
            symbol=data.symbol,
        )
        validated.model_type = getattr(data.signal, 'model_type', None)
        validated.backtest_summary = getattr(data.signal, 'backtest_summary', None)
        validated.backtest_stats = getattr(data.signal, 'backtest_stats', None)

        # Clear _needs_risk_params flag if the LLM has now provided risk parameters
        if data.symbol in engine.positions:
            _pos = engine.positions[data.symbol]
            if _pos.get("_needs_risk_params"):
                if _pos.get("stop_loss") is not None and _pos.get("take_profit") is not None:
                    _pos.pop("_needs_risk_params", None)
                    _pos.pop("_needs_risk_params_attempts", None)
                    logger.info(f"Risk parameters obtained for {data.symbol}; cleared _needs_risk_params flag.")

        await self.log_and_notify_decision(
            data=data,
            validated=validated,
            backtest_stats=getattr(validated, 'backtest_stats', None),
        )

        params = data.signal.strategy_params or {}

        # --- Handle triggered position flags (max hold, stop loss, take profit, partial TP, dust sweep) ---
        if await self.handle_triggered_flags(
            symbol=data.symbol,
            display_symbol=data.display_symbol,
            signal=data.signal,
            validated=validated,
            assigned_tf=data.assigned_tf,
            current_price=data.current_price,
            atr=data.atr,
            ticker=data.ticker,
            max_hold_expired=data.max_hold_expired,
            stop_loss_triggered=data.stop_loss_triggered,
            take_profit_triggered=data.take_profit_triggered,
            partial_tp_triggered=data.partial_tp_triggered,
            dust_sweep_triggered=data.dust_sweep_triggered,
            strategy_model_type=data.strategy_model_type,
            llm_provider=llm_provider,
            llm_model=llm_model,
        ):
            return

        # --- LLM‑controlled trade filters ---
        if await self.check_trade_filters(
            data.symbol, data.display_symbol, validated, params
        ):
            return

        # Apply any updated risk parameters from the LLM to the open position
        if data.symbol in engine.positions and data.signal.strategy_params:
            await self.event_bus.publish(
                "update_position_params",
                data.symbol,
                data.signal.strategy_params,
                data.signal.indicator_config,
                data.assigned_tf,
                data.current_price,
                data.atr,
            )

        if validated.action != "HOLD":
            # --- Sector concentration limit check (only for BUY) ---
            if validated.action == "BUY":
                if await self.check_sector_concentration(
                    data.symbol, data.display_symbol, data.assigned_tf
                ):
                    return

            if await self.handle_entry_condition(
                symbol=data.symbol,
                display_symbol=data.display_symbol,
                validated=validated,
                assigned_tf=data.assigned_tf,
                tf_seconds=data.tf_seconds,
                trading_paused=data.trading_paused,
            ):
                return

            await self.event_bus.publish("execute_signal", data.symbol, validated, timeframe=data.assigned_tf, atr=data.atr)

    async def run_step1b_llm_call(
        self,
        symbol: str,
        analysis_result: Dict[str, Any],
        ticker: Dict[str, Any],
        current_price: float,
        atr: Optional[float],
        assigned_tf: str,
        base_balance: float,
        per_symbol_budget: float,
        min_order_amount: Optional[float],
        min_order_cost: Optional[float],
        remaining: float,
        portfolio_total_value: float,
        portfolio_exposure_pct: float,
        portfolio_stop_risk_pct: float,
        portfolio_available_capital: float,
        max_port_exp: Optional[float],
        max_port_risk: Optional[float],
        global_risk_mult: Optional[float],
        min_stop_atr_mult: float,
        min_hold_time_mult: float,
        trading_paused: bool,
        has_position: bool,
        strategy_model_type: str,
        effective_temp: float,
        market_snapshot: Dict[str, Any],
        historical_backtest_results: Optional[list],
    ) -> Tuple[Signal, Optional[str], Optional[str]]:
        """Run the Step 1b LLM call for backtest variants and parameters.

        Returns (preliminary_signal, llm_provider, llm_model).
        """
        engine = self.engine
        llm_provider = None
        llm_model = None

        # --- Build variants prompt ---
        prompt_data = BacktestPromptData(
            symbol=symbol,
            analysis=analysis_result,
            ticker=ticker,
            current_price=current_price,
            atr=atr,
            assigned_timeframe=assigned_tf,
            base_currency=engine.base_currency,
            base_balance=base_balance,
            per_symbol_budget=per_symbol_budget,
            min_order_amount=min_order_amount,
            min_order_cost=min_order_cost,
            remaining_balance=remaining,
            portfolio_total_value=portfolio_total_value,
            portfolio_exposure_pct=portfolio_exposure_pct,
            portfolio_stop_risk_pct=portfolio_stop_risk_pct,
            portfolio_available_capital=portfolio_available_capital,
            max_portfolio_exposure_pct=max_port_exp,
            max_portfolio_stop_risk_pct=max_port_risk,
            global_risk_multiplier=global_risk_mult,
            min_stop_atr_mult=min_stop_atr_mult,
            min_hold_time_mult=min_hold_time_mult,
            trading_paused=trading_paused,
            has_position=has_position,
            historical_backtest_results=historical_backtest_results,
        )
        variants_prompt = await asyncio.to_thread(
            build_backtest_variants_prompt,
            prompt_data
        )
        logger.info(f"LLM Step 1b variants prompt for {symbol}: {len(variants_prompt)} chars")

        # Use a different market hash for Step 1b (include analysis to differentiate)
        variants_market_hash = compute_market_hash({
            **market_snapshot,
            "step": "1b",
            "analysis": analysis_result,
        })

        try:
            step1b_result = await asyncio.wait_for(
                asyncio.to_thread(
                    get_cached_llm_response,
                    compact_prompt(variants_prompt),
                    compact_prompt(build_system_prompt()),
                    60,
                    market_hash=variants_market_hash,
                    model_type=strategy_model_type,
                    temperature=effective_temp,
                ),
                timeout=settings.LLM_TIMEOUT
            )
            step1b_response = step1b_result["response"]
            llm_provider = step1b_result["provider"]
            llm_model = step1b_result["model"]
            logger.info(f"LLM Step 1b (variants) completed for {symbol} (provider={llm_provider}, model={llm_model})")
        except asyncio.TimeoutError:
            logger.warning(f"LLM Step 1b (variants) timed out for {symbol}. Using Step 1a analysis as fallback.")
            step1b_response = json.dumps({
                "action": analysis_result.get("action", "HOLD"),
                "confidence": analysis_result.get("confidence", 0.0),
                "reasoning": analysis_result.get("reasoning", ""),
                "strategy": {
                    "type": "fallback",
                    "parameters": {},
                },
            })
        except (ConnectionError, TimeoutError, OSError, ValueError, TypeError, RuntimeError, json.JSONDecodeError) as e:
            logger.error(f"LLM Step 1b failed for {symbol}: {e}. Using Step 1a analysis as fallback.")
            step1b_response = json.dumps({
                "action": analysis_result.get("action", "HOLD"),
                "confidence": analysis_result.get("confidence", 0.0),
                "reasoning": analysis_result.get("reasoning", ""),
                "strategy": {
                    "type": "fallback",
                    "parameters": {},
                },
            })

        # --- Parse Step 1b response ---
        try:
            preliminary_strategy = create_strategy_from_llm(step1b_response)
        except ValueError as e:
            logger.warning(f"LLM Step 1b response parse failed for {symbol}: {e}. Retrying with correction prompt.")
            correction_prompt = (
                "Your previous response was not valid JSON. "
                "You MUST output ONLY a single JSON object, with no markdown fences, no explanations, no extra text. "
                "Here is the original request:\n\n" + variants_prompt
            )
            try:
                response2 = await asyncio.wait_for(
                    asyncio.to_thread(
                        get_cached_llm_response, compact_prompt(correction_prompt),
                        compact_prompt(build_system_prompt()),
                        30,
                        model_type="actuator",
                        temperature=effective_temp,
                    ),
                    timeout=settings.LLM_TIMEOUT
                )
                preliminary_strategy = create_strategy_from_llm(response2["response"])
                llm_provider = response2["provider"]
                llm_model = response2["model"]
            except (asyncio.TimeoutError, ConnectionError, TimeoutError, OSError, ValueError, TypeError, RuntimeError, json.JSONDecodeError) as e2:
                logger.error(f"LLM Step 1b response still invalid after retry for {symbol}: {e2}")
                preliminary_strategy = LLMStrategy(self._create_fallback_hold_signal(
                    symbol, "Failed to parse LLM Step 1b response after retry", strategy_model_type
                ))

        preliminary_signal = preliminary_strategy.generate_signal({})
        preliminary_signal.model_type = strategy_model_type
        preliminary_signal.llm_provider = llm_provider
        preliminary_signal.llm_model = llm_model

        return preliminary_signal, llm_provider, llm_model

    def choose_model_tier(
        self,
        atr: Optional[float] = None,
        atr_percentile: Optional[float] = None,
        rsi: Optional[float] = None,
        macd: Optional[float] = None,
        macd_signal: Optional[float] = None,
        macd_hist: Optional[float] = None,
        bb_upper: Optional[float] = None,
        bb_middle: Optional[float] = None,
        bb_lower: Optional[float] = None,
        ema_9: Optional[float] = None,
        ema_21: Optional[float] = None,
        stochastic_k: Optional[float] = None,
        adx: Optional[float] = None,
        plus_di: Optional[float] = None,
        minus_di: Optional[float] = None,
        mfi: Optional[float] = None,
        cci: Optional[float] = None,
        williams_r: Optional[float] = None,
        ichimoku: Optional[Dict[str, Any]] = None,
        market_regime: str = "",
        market_breadth: Optional[Dict[str, Any]] = None,
        full_market_breadth: Optional[Dict[str, Any]] = None,
        sentiment_trend_val: Optional[float] = None,
        volume_trend: Optional[float] = None,
        unrealized_pnl: Optional[float] = None,
        drawdown_pct: Optional[float] = None,
        portfolio_exposure_pct: Optional[float] = None,
        portfolio_stop_risk_pct: Optional[float] = None,
        is_critical: bool = False,
        trading_paused: bool = False,
        symbol_event: Optional[Dict[str, Any]] = None,
        fundamentals: Optional[Dict[str, Any]] = None,
        consecutive_losses: int = 0,
        current_price: Optional[float] = None,
    ) -> str:
        """Return "mind" or "actuator" based on market complexity."""
        if is_critical:
            return "mind"

        score = 0.0
        max_score = 0.0

        # === Critical factors (weight 2.0) ===
        if rsi is not None and macd_hist is not None:
            max_score += 2.0
            if (rsi < settings.MODEL_TIER_RSI_EXTREME and macd_hist < 0) or (rsi > (100 - settings.MODEL_TIER_RSI_EXTREME) and macd_hist > 0):
                score += 2.0

        if all(v is not None for v in (ema_9, ema_21, adx, plus_di, minus_di)):
            max_score += 2.0
            if (ema_9 > ema_21) != (plus_di > minus_di) and adx > settings.MODEL_TIER_ADX_STRONG:
                score += 2.0

        if drawdown_pct is not None:
            max_score += 2.0
            if drawdown_pct > settings.MODEL_TIER_DRAWDOWN_PCT:
                score += 2.0

        if symbol_event is not None:
            max_score += 2.0
            if symbol_event.get("has_event"):
                score += 2.0

        max_score += 2.0
        if consecutive_losses >= settings.MODEL_TIER_CONSECUTIVE_LOSSES:
            score += 2.0

        # === Significant factors (weight 1.5) ===
        if atr_percentile is not None:
            max_score += 1.5
            if atr_percentile > settings.MODEL_TIER_ATR_PERCENTILE_HIGH or atr_percentile < settings.MODEL_TIER_ATR_PERCENTILE_LOW:
                score += 1.5

        if market_regime:
            max_score += 1.5
            if any(kw in market_regime for kw in ("high volatility", "squeeze", "expansion", "ranging")):
                score += 1.5

        if sentiment_trend_val is not None:
            max_score += 1.5
            if abs(sentiment_trend_val) > settings.MODEL_TIER_SENTIMENT_TREND_MAG:
                score += 1.5

        if all(v is not None for v in (bb_upper, bb_lower, bb_middle)) and bb_middle > 0:
            max_score += 1.5
            bb_width = (bb_upper - bb_lower) / bb_middle
            if bb_width < settings.MODEL_TIER_BB_WIDTH_SQUEEZE or bb_width > settings.MODEL_TIER_BB_WIDTH_EXPANSION:
                score += 1.5

        if portfolio_exposure_pct is not None:
            max_score += 1.5
            if portfolio_exposure_pct > settings.MODEL_TIER_PORTFOLIO_EXPOSURE_HIGH:
                score += 1.5

        if portfolio_stop_risk_pct is not None:
            max_score += 1.5
            if portfolio_stop_risk_pct > settings.MODEL_TIER_PORTFOLIO_STOP_RISK_HIGH:
                score += 1.5

        if unrealized_pnl is not None:
            max_score += 1.5
            if unrealized_pnl < 0:
                score += 1.5

        if market_breadth is not None:
            max_score += 1.5
            pos_pct = market_breadth.get("positive_pct", 50)
            if pos_pct > settings.MODEL_TIER_MARKET_BREADTH_EXTREME or pos_pct < (100 - settings.MODEL_TIER_MARKET_BREADTH_EXTREME):
                score += 1.5

        if full_market_breadth is not None:
            max_score += 1.5
            pos_pct = full_market_breadth.get("positive_pct", 50)
            if pos_pct > settings.MODEL_TIER_MARKET_BREADTH_EXTREME or pos_pct < (100 - settings.MODEL_TIER_MARKET_BREADTH_EXTREME):
                score += 1.5

        # === Standard factors (weight 1.0) ===
        if macd is not None and macd_signal is not None and macd != 0:
            max_score += 1.0
            if abs(macd - macd_signal) < settings.MODEL_TIER_MACD_HIST_CHANGE * abs(macd):
                score += 1.0

        if stochastic_k is not None:
            max_score += 1.0
            if stochastic_k < settings.MODEL_TIER_STOCH_EXTREME or stochastic_k > (100 - settings.MODEL_TIER_STOCH_EXTREME):
                score += 1.0

        if mfi is not None:
            max_score += 1.0
            if mfi < settings.MODEL_TIER_MFI_EXTREME or mfi > (100 - settings.MODEL_TIER_MFI_EXTREME):
                score += 1.0

        if cci is not None:
            max_score += 1.0
            if cci < -settings.MODEL_TIER_CCI_EXTREME or cci > settings.MODEL_TIER_CCI_EXTREME:
                score += 1.0

        if williams_r is not None:
            max_score += 1.0
            if williams_r < -(100 - settings.MODEL_TIER_WILLIAMS_R_EXTREME) or williams_r > -settings.MODEL_TIER_WILLIAMS_R_EXTREME:
                score += 1.0

        if ichimoku is not None and current_price is not None:
            cloud_top = ichimoku.get("cloud_top")
            cloud_bottom = ichimoku.get("cloud_bottom")
            if cloud_top is not None and cloud_bottom is not None:
                max_score += 1.0
                if cloud_bottom <= current_price <= cloud_top:
                    score += 1.0

        if volume_trend is not None:
            max_score += 1.0
            if volume_trend > settings.MODEL_TIER_VOLUME_TREND_HIGH:
                score += 1.0

        if fundamentals is not None:
            pe = fundamentals.get("pe_ratio")
            if pe is not None:
                max_score += 1.0
                if pe > settings.MODEL_TIER_PE_HIGH or pe < 0:
                    score += 1.0
            margins = fundamentals.get("profit_margins")
            if margins is not None:
                max_score += 1.0
                if margins < 0:
                    score += 1.0

        if max_score == 0:
            return "actuator"
        normalized_score = score / max_score
        return "mind" if normalized_score >= settings.LLM_MIND_MODEL_THRESHOLD else "actuator"

    def compute_prompt_complexity(
        self,
        # Candidate/portfolio context
        num_candidates: int = 0,
        # Volatility
        volatility_percentile: Optional[float] = None,
        # Core indicators
        rsi: Optional[float] = None,
        macd: Optional[float] = None,
        macd_signal: Optional[float] = None,
        macd_hist: Optional[float] = None,
        bb_upper: Optional[float] = None,
        bb_middle: Optional[float] = None,
        bb_lower: Optional[float] = None,
        ema_9: Optional[float] = None,
        ema_21: Optional[float] = None,
        stochastic_k: Optional[float] = None,
        adx: Optional[float] = None,
        plus_di: Optional[float] = None,
        minus_di: Optional[float] = None,
        mfi: Optional[float] = None,
        cci: Optional[float] = None,
        williams_r: Optional[float] = None,
        ichimoku: Optional[Dict[str, Any]] = None,
        # Market context
        market_breadth: Optional[Dict[str, Any]] = None,
        full_market_breadth: Optional[Dict[str, Any]] = None,
        sentiment_trend_magnitude: Optional[float] = None,
        volume_trend: Optional[float] = None,
        market_regime: str = "",
        # Portfolio context
        unrealized_pnl: Optional[float] = None,
        drawdown_pct: Optional[float] = None,
        portfolio_exposure_pct: Optional[float] = None,
        portfolio_stop_risk_pct: Optional[float] = None,
        # Position state
        is_critical: bool = False,
        trading_paused: bool = False,
        # Events & fundamentals
        symbol_event: Optional[Dict[str, Any]] = None,
        fundamentals: Optional[Dict[str, Any]] = None,
        # Past performance
        consecutive_losses: int = 0,
        # Current price
        current_price: Optional[float] = None,
        # Legacy params (kept for backward compat with other call sites)
        fear_greed: Optional[Dict[str, Any]] = None,
        conflicting_signals: bool = False,
    ) -> float:
        """Return a complexity score between 0.0 (simple) and 1.0 (very complex)."""
        # Category-based scoring: within each category, take the MAX contribution
        # (not the sum) so that multiple factors in the same category don't
        # dominate the score.  Category maxima sum to 1.0 exactly.

        # === Category 1: Technical indicator extremes (max 0.25) ===
        tech_score = 0.0
        if rsi is not None and (rsi < settings.MODEL_TIER_RSI_EXTREME or rsi > (100 - settings.MODEL_TIER_RSI_EXTREME)):
            tech_score = max(tech_score, 0.15)
        if stochastic_k is not None and (stochastic_k < settings.MODEL_TIER_STOCH_EXTREME or stochastic_k > (100 - settings.MODEL_TIER_STOCH_EXTREME)):
            tech_score = max(tech_score, 0.12)
        if mfi is not None and (mfi < settings.MODEL_TIER_MFI_EXTREME or mfi > (100 - settings.MODEL_TIER_MFI_EXTREME)):
            tech_score = max(tech_score, 0.12)
        if cci is not None and (cci < -settings.MODEL_TIER_CCI_EXTREME or cci > settings.MODEL_TIER_CCI_EXTREME):
            tech_score = max(tech_score, 0.12)
        if williams_r is not None and (williams_r < -(100 - settings.MODEL_TIER_WILLIAMS_R_EXTREME) or williams_r > -settings.MODEL_TIER_WILLIAMS_R_EXTREME):
            tech_score = max(tech_score, 0.12)
        if macd is not None and macd_signal is not None and macd != 0:
            if abs(macd - macd_signal) < settings.MODEL_TIER_MACD_HIST_CHANGE * abs(macd):
                tech_score = max(tech_score, 0.10)
        if bb_upper is not None and bb_lower is not None and bb_middle is not None and bb_middle > 0:
            bb_width = (bb_upper - bb_lower) / bb_middle
            if bb_width < settings.MODEL_TIER_BB_WIDTH_SQUEEZE or bb_width > settings.MODEL_TIER_BB_WIDTH_EXPANSION:
                tech_score = max(tech_score, 0.15)
        if ichimoku is not None and current_price is not None:
            cloud_top = ichimoku.get("cloud_top")
            cloud_bottom = ichimoku.get("cloud_bottom")
            if cloud_top is not None and cloud_bottom is not None:
                if cloud_bottom <= current_price <= cloud_top:
                    tech_score = max(tech_score, 0.12)

        # === Category 2: Conflicting signals (max 0.20) ===
        conflict_score = 0.0
        if rsi is not None and macd_hist is not None:
            if (rsi < settings.MODEL_TIER_RSI_EXTREME and macd_hist < 0) or (rsi > (100 - settings.MODEL_TIER_RSI_EXTREME) and macd_hist > 0):
                conflict_score = max(conflict_score, 0.20)
        if ema_9 is not None and ema_21 is not None and adx is not None and plus_di is not None and minus_di is not None:
            ema_bullish = ema_9 > ema_21
            di_bullish = plus_di > minus_di
            if ema_bullish != di_bullish and adx > settings.MODEL_TIER_ADX_STRONG:
                conflict_score = max(conflict_score, 0.15)
        if conflicting_signals:
            conflict_score = max(conflict_score, 0.10)

        # === Category 3: Market context (max 0.20) ===
        market_score = 0.0
        if volatility_percentile is not None and (volatility_percentile > settings.MODEL_TIER_ATR_PERCENTILE_HIGH or volatility_percentile < settings.MODEL_TIER_ATR_PERCENTILE_LOW):
            market_score = max(market_score, 0.15)
        if market_regime and any(kw in market_regime for kw in ("high volatility", "squeeze", "expansion", "ranging")):
            market_score = max(market_score, 0.12)
        if market_breadth:
            pos_pct = market_breadth.get("positive_pct", 50)
            if pos_pct > settings.MODEL_TIER_MARKET_BREADTH_EXTREME or pos_pct < (100 - settings.MODEL_TIER_MARKET_BREADTH_EXTREME):
                market_score = max(market_score, 0.12)
        if full_market_breadth:
            pos_pct = full_market_breadth.get("positive_pct", 50)
            if pos_pct > settings.MODEL_TIER_MARKET_BREADTH_EXTREME or pos_pct < (100 - settings.MODEL_TIER_MARKET_BREADTH_EXTREME):
                market_score = max(market_score, 0.10)
        if sentiment_trend_magnitude is not None and sentiment_trend_magnitude > settings.MODEL_TIER_SENTIMENT_TREND_MAG:
            market_score = max(market_score, 0.12)
        if volume_trend is not None and volume_trend > settings.MODEL_TIER_VOLUME_TREND_HIGH:
            market_score = max(market_score, 0.10)

        # === Category 4: Portfolio stress (max 0.15) ===
        portfolio_score = 0.0
        if portfolio_exposure_pct is not None and portfolio_exposure_pct > settings.MODEL_TIER_PORTFOLIO_EXPOSURE_HIGH:
            portfolio_score = max(portfolio_score, 0.15)
        if portfolio_stop_risk_pct is not None and portfolio_stop_risk_pct > settings.MODEL_TIER_PORTFOLIO_STOP_RISK_HIGH:
            portfolio_score = max(portfolio_score, 0.15)
        if drawdown_pct is not None and drawdown_pct > settings.MODEL_TIER_DRAWDOWN_PCT:
            portfolio_score = max(portfolio_score, 0.15)
        if unrealized_pnl is not None and unrealized_pnl < 0:
            portfolio_score = max(portfolio_score, 0.10)
        if consecutive_losses >= settings.MODEL_TIER_CONSECUTIVE_LOSSES:
            portfolio_score = max(portfolio_score, 0.12)

        # === Category 5: Critical & events (max 0.15) ===
        critical_score = 0.0
        if is_critical:
            critical_score = max(critical_score, 0.15)
        if symbol_event is not None and symbol_event.get("has_event"):
            critical_score = max(critical_score, 0.10)
        if fundamentals is not None:
            pe = fundamentals.get("pe_ratio")
            if pe is not None and (pe > settings.MODEL_TIER_PE_HIGH or pe < 0):
                critical_score = max(critical_score, 0.08)
            margins = fundamentals.get("profit_margins")
            if margins is not None and margins < 0:
                critical_score = max(critical_score, 0.08)
        if trading_paused:
            critical_score = max(critical_score, 0.05)

        # === Category 6: Candidate count (max 0.03) ===
        candidate_score = 0.0
        if num_candidates > 20:
            candidate_score = 0.03
        elif num_candidates > 10:
            candidate_score = 0.02

        # === Category 7: Legacy fear_greed (max 0.02) ===
        legacy_score = 0.0
        if fear_greed:
            fg = fear_greed.get("value", 50)
            if fg <= 25 or fg >= 75:
                legacy_score = 0.02

        total = tech_score + conflict_score + market_score + portfolio_score + critical_score + candidate_score + legacy_score
        return min(1.0, total)

    def _get_effective_temperature(self, model_type: str, complexity: float) -> float:
        """Return the temperature to use for a given model_type and complexity score (0-1)."""
        from src.config.settings import Settings
        raw = settings.LLM_MIND_TEMPERATURE if model_type == "mind" else settings.LLM_ACTUATOR_TEMPERATURE
        parsed = Settings.parse_temperature_range(raw)
        if parsed is None:
            # Fall back to global LLM_TEMPERATURE
            return settings.LLM_TEMPERATURE
        lo, hi = parsed
        if lo == hi:
            return lo
        # Map complexity 0→lo, 1→hi
        return lo + (hi - lo) * complexity

    def compute_model_tier_and_temperature(
        self,
        atr: Optional[float],
        atr_percentile: Optional[float],
        rsi: Optional[float],
        macd: Optional[float],
        macd_signal: Optional[float],
        macd_hist: Optional[float],
        bb_upper: Optional[float],
        bb_middle: Optional[float],
        bb_lower: Optional[float],
        ema_9: Optional[float],
        ema_21: Optional[float],
        stochastic_k: Optional[float],
        adx: Optional[float],
        plus_di: Optional[float],
        minus_di: Optional[float],
        mfi: Optional[float],
        cci: Optional[float],
        williams_r: Optional[float],
        ichimoku: Optional[Dict[str, Any]],
        market_regime: str,
        market_breadth: Optional[Dict[str, Any]],
        full_market_breadth: Optional[Dict[str, Any]],
        sentiment_trend_val: Optional[float],
        volume_trend_val: Optional[float],
        unrealized_pnl: Optional[float],
        drawdown_pct: Optional[float],
        portfolio_exposure_pct: float,
        portfolio_stop_risk_pct: float,
        is_critical: bool,
        trading_paused: bool,
        symbol_event: Optional[Dict[str, Any]],
        fundamentals: Optional[Dict[str, Any]],
        consecutive_losses: int,
        current_price: float,
        num_candidates: int,
    ) -> Tuple[str, float]:
        """Compute the strategy model type and effective temperature.

        Returns (strategy_model_type, effective_temp).
        """
        engine = self.engine

        strategy_model_type = self.choose_model_tier(
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
            market_breadth=market_breadth,
            full_market_breadth=full_market_breadth,
            sentiment_trend_val=sentiment_trend_val,
            volume_trend=volume_trend_val,
            unrealized_pnl=unrealized_pnl,
            drawdown_pct=drawdown_pct,
            portfolio_exposure_pct=portfolio_exposure_pct,
            portfolio_stop_risk_pct=portfolio_stop_risk_pct,
            is_critical=is_critical,
            trading_paused=trading_paused,
            symbol_event=symbol_event,
            fundamentals=fundamentals,
            consecutive_losses=consecutive_losses,
            current_price=current_price,
        )

        # Compute prompt complexity for temperature selection
        _conflicting = False
        if rsi is not None and macd_hist is not None:
            if (rsi < settings.MODEL_TIER_RSI_EXTREME and macd_hist < 0) or (rsi > (100 - settings.MODEL_TIER_RSI_EXTREME) and macd_hist > 0):
                _conflicting = True
        strategy_complexity = self.compute_prompt_complexity(
            num_candidates=num_candidates,
            volatility_percentile=atr_percentile,
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
            market_breadth=market_breadth,
            full_market_breadth=full_market_breadth,
            sentiment_trend_magnitude=abs(sentiment_trend_val) if sentiment_trend_val is not None else None,
            volume_trend=volume_trend_val,
            market_regime=market_regime,
            unrealized_pnl=unrealized_pnl,
            drawdown_pct=drawdown_pct,
            portfolio_exposure_pct=portfolio_exposure_pct,
            portfolio_stop_risk_pct=portfolio_stop_risk_pct,
            is_critical=is_critical,
            trading_paused=trading_paused,
            symbol_event=symbol_event,
            fundamentals=fundamentals,
            consecutive_losses=consecutive_losses,
            current_price=current_price,
            conflicting_signals=_conflicting,
        )
        effective_temp = self._get_effective_temperature(strategy_model_type, strategy_complexity)

        return strategy_model_type, effective_temp

    async def check_pause_resume_decision(self) -> None:
        """When trading is paused, ask the LLM whether to resume (lightweight)."""
        engine = self.engine
        async with engine._symbol_reeval_lock:
            # Only run if actually paused
            paused_raw = await asyncio.to_thread(engine.redis.get, "trading:paused")
            if not paused_raw or paused_raw != "1":
                return

            # Only handle LLM-initiated pauses. Manual pauses are not subject to auto-resume logic.
            source_raw = await asyncio.to_thread(engine.redis.get, "trading:pause_source")
            source = source_raw.decode() if isinstance(source_raw, bytes) else (source_raw or "")
            if source != "llm":
                logger.info("Pause/resume check skipped: pause was not initiated by LLM (source=%s).", source or "unknown")
                return

            # Read LLM-decided pause recovery settings from Redis
            max_keep = settings.PAUSE_MAX_CONSECUTIVE_KEEP
            force_resume_mult = settings.PAUSE_FORCE_RESUME_RISK_MULTIPLIER
            try:
                raw = await engine.config_service.get_config("pause_max_consecutive_keep")
                if raw:
                    max_keep = int(raw)
                raw = await engine.config_service.get_config("pause_force_resume_risk_multiplier")
                if raw:
                    force_resume_mult = float(raw)
            except (ValueError, TypeError, ConnectionError, TimeoutError, OSError):
                pass

            # Gather minimal market context
            benchmark_price = None
            try:
                tickers_map = await engine._get_quotes_async([settings.BENCHMARK_SYMBOL], timeout=45.0)
                benchmark_ticker = tickers_map.get(settings.BENCHMARK_SYMBOL)
                benchmark_price = benchmark_ticker.get("last") if benchmark_ticker else None
            except (ConnectionError, TimeoutError, OSError, ValueError, TypeError, KeyError):
                pass

            # Market breadth from Redis (already computed by background task)
            full_market_breadth = None
            try:
                raw = await asyncio.to_thread(engine.redis.get, "market:breadth:full")
                if raw:
                    full_market_breadth = json.loads(raw)
            except (ValueError, TypeError, ConnectionError, TimeoutError, OSError, json.JSONDecodeError):
                pass
            market_breadth = getattr(engine, '_market_breadth', None)

            # Current pause reason
            reason_raw = await asyncio.to_thread(engine.redis.get, "trading:pause_reason")
            pause_reason = reason_raw.decode() if isinstance(reason_raw, bytes) else (reason_raw or "")

            # --- Consecutive "keep paused" counter ---
            keep_key = "trading:pause:keep_count"
            keep_count_raw = await asyncio.to_thread(engine.redis.get, keep_key)
            try:
                keep_count = int(keep_count_raw) if keep_count_raw else 0
            except (ValueError, TypeError):
                keep_count = 0

            # Build a richer prompt with performance context
            perf = await asyncio.to_thread(engine._compute_performance_metrics)
            daily_pnl = perf["equity_curve"].get("daily_pnl", 0.0)
            total_pnl = perf["equity_curve"].get("total_pnl", 0.0)
            consecutive_losses = perf["equity_curve"].get("consecutive_losses", 0)
            drawdown_pct = perf["equity_curve"].get("drawdown_pct", 0.0)

            prompt_parts = [
                "Trading is currently paused.",
            ]
            if pause_reason:
                prompt_parts.append(f"Pause reason: {pause_reason}")
            prompt_parts.append(f"Account P&L: daily={daily_pnl:.4f}, total={total_pnl:.4f}, drawdown={drawdown_pct:.2f}%")
            if consecutive_losses > 0:
                prompt_parts.append(f"Consecutive losing trades: {consecutive_losses}")
            if benchmark_price is not None:
                prompt_parts.append(f"Benchmark ({settings.BENCHMARK_SYMBOL}) price: {benchmark_price}")
            if market_breadth:
                prompt_parts.append(f"Market breadth (top stocks): {market_breadth['positive_pct']}% positive")
            if full_market_breadth:
                prompt_parts.append(f"Full market breadth: {full_market_breadth['positive_pct']}% positive")

            # Check if this is a recent auto-resume situation
            last_auto_resume_raw = await asyncio.to_thread(engine.redis.get, "trading:last_auto_resume")
            if last_auto_resume_raw:
                try:
                    last_auto_resume_ts = float(last_auto_resume_raw)
                    seconds_since = time.time() - last_auto_resume_ts
                    if seconds_since < 3600:  # within the last hour
                        minutes_since = seconds_since / 60
                        prompt_parts.append(
                            f"Trading was auto‑resumed {minutes_since:.1f} minutes ago. "
                            "Market conditions may not have changed significantly. "
                            "Only resume if there is clear, concrete improvement in the data above."
                        )
                except (ValueError, TypeError):
                    pass

            # --- Consecutive keep warning and recovery nudge ---
            if keep_count > 0:
                prompt_parts.append(
                    f"You have chosen to keep trading paused {keep_count} time(s) in a row. "
                    f"If you keep it paused {max_keep} times consecutively, the engine will "
                    f"force‑resume trading with a reduced global risk multiplier of "
                    f"{force_resume_mult} to attempt recovery."
                )

            prompt_parts.append(
                "If the account is in drawdown or has consecutive losses, consider resuming "
                "with a **reduced global risk multiplier** (e.g., 0.3–0.5) instead of staying "
                "paused indefinitely. This allows the bot to cautiously seek small profitable "
                "trades to recover, while limiting downside. You can provide an optional "
                "`global_risk_multiplier` field in your JSON response (0.0–1.0) to set the "
                "risk level upon resume. If you omit it, the current multiplier (or 1.0) will be used. "
                "If you are even slightly uncertain, **resume trading** with a very small "
                "`global_risk_multiplier` (e.g., 0.15–0.25). This lets the bot probe for "
                "opportunities with minimal risk. Staying paused forever guarantees zero recovery."
            )

            prompt = (
                "\n".join(prompt_parts)
                + "\n\nShould we resume trading now? Reply with a JSON object: "
                '{"resume_trading": true/false, "reason": "short explanation", '
                '"global_risk_multiplier": 0.0-1.0 (optional)}'
                + "\n\n**Important:** Only resume if you see specific, high‑confidence opportunities. "
                "If conditions are still poor, you may keep trading paused, but remember that "
                "staying paused forever prevents any recovery. A cautious resume with a low risk "
                "multiplier is often better than doing nothing."
            )

            pause_resume_complexity = self.compute_prompt_complexity(
                num_candidates=0,
                market_breadth=market_breadth,
                fear_greed=None,
                volatility_percentile=None,
                sentiment_trend_magnitude=None,
                conflicting_signals=False,
                is_critical=False,
            )
            effective_temp = self._get_effective_temperature("actuator", pause_resume_complexity)

            try:
                pause_result = await asyncio.wait_for(
                    asyncio.to_thread(
                        get_cached_llm_response, compact_prompt(prompt), compact_prompt(build_system_prompt()), 120,
                        model_type="actuator",
                        temperature=effective_temp,
                    ),
                    timeout=settings.LLM_TIMEOUT
                )
                response = pause_result["response"]
                llm_provider = pause_result["provider"]
                llm_model = pause_result["model"]
                decision = json.loads(response)
            except (asyncio.TimeoutError, ConnectionError, TimeoutError, OSError, ValueError, TypeError, RuntimeError, json.JSONDecodeError) as e:
                logger.warning(f"Pause/resume LLM call failed: {e}")
                # Track consecutive failures in Redis
                fail_key = "trading:pause:llm_fail_count"
                current_fails = await asyncio.to_thread(engine.redis.incr, fail_key)
                await asyncio.to_thread(engine.redis.expire, fail_key, 3600)
                _min_pause = settings.MIN_LLM_PAUSE_DURATION
                try:
                    raw = await engine.config_service.get_config("min_llm_pause_duration")
                    if raw:
                        _min_pause = int(raw)
                except (ValueError, TypeError, ConnectionError, TimeoutError, OSError):
                    pass
                if engine.notifier:
                    await engine.notifier.send_notification(
                        f"⚠️ Could not reach LLM to decide pause/resume (failure #{current_fails}). "
                        f"Auto‑resume will be attempted after {_min_pause}s if LLM stays silent.",
                        summary={"action": "INFO", "reason": "LLM pause-resume call failed"}
                    )
                # If we failed 3 times in a row, force‑resume (optional but safe)
                if current_fails >= 3:
                    # Double-check source before force-resuming
                    fail_source = await asyncio.to_thread(engine.redis.get, "trading:pause_source")
                    if fail_source and (fail_source.decode() if isinstance(fail_source, bytes) else fail_source) != "llm":
                        logger.warning("Force-resume on LLM failure skipped: pause source is not LLM.")
                        return
                    pause_keys = [
                        "trading:paused",
                        "trading:pause_source",
                        "trading:pause_start",
                        "trading:pause_duration",
                        "trading:pause_reason",
                        "trading:llm_pause_time",
                    ]
                    for key in pause_keys:
                        await asyncio.to_thread(engine.redis.delete, key)
                    await asyncio.to_thread(engine.redis.delete, fail_key)
                    # --- Also reset keep counter and set force‑resume risk multiplier ---
                    await asyncio.to_thread(engine.redis.delete, keep_key)
                    await engine._set_global_risk_multiplier(force_resume_mult)
                    engine._reeval_trigger.set()
                    if engine.notifier:
                        await engine.notifier.send_notification(
                            "▶️ Trading auto‑resumed because LLM could not be reached for pause decision. "
                            f"Global risk multiplier set to {force_resume_mult}.",
                            summary={"action": "RESUME", "reason": "LLM pause-resume failures exceeded limit"}
                        )
                return

            resume_trading = decision.get("resume_trading")
            reason = decision.get("reason", "")

            if resume_trading is True:
                # Source is already verified as "llm" by the early check at the top of this method.

                # Check minimum LLM pause duration
                llm_pause_time_raw = await asyncio.to_thread(engine.redis.get, "trading:llm_pause_time")
                if llm_pause_time_raw:
                    try:
                        llm_pause_time = float(llm_pause_time_raw)
                        _min_pause = settings.MIN_LLM_PAUSE_DURATION
                        try:
                            raw = await engine.config_service.get_config("min_llm_pause_duration")
                            if raw:
                                _min_pause = int(raw)
                        except (ValueError, TypeError, ConnectionError, TimeoutError, OSError):
                            pass
                        if time.time() - llm_pause_time < _min_pause:
                            remaining = _min_pause - (time.time() - llm_pause_time)
                            logger.info(f"Ignoring LLM resume request: minimum pause duration not elapsed ({remaining:.0f}s remaining).")
                            if engine.notifier:
                                await engine.notifier.send_notification(
                                    f"⏸️ LLM resume request ignored: minimum pause duration "
                                    f"({_min_pause}s) not yet elapsed ({remaining:.0f}s remaining).",
                                    summary={"action": "RESUME", "reason": f"LLM resume blocked by minimum pause duration ({_min_pause}s)", "model_type": "actuator"}
                                )
                            return
                    except (ValueError, TypeError):
                        pass

                # --- Apply optional global_risk_multiplier from LLM ---
                global_mult_raw = decision.get("global_risk_multiplier")
                applied_mult = None
                if global_mult_raw is not None:
                    try:
                        mult_val = float(global_mult_raw)
                        if 0.0 <= mult_val <= 1.0:
                            await engine._set_global_risk_multiplier(mult_val)
                            logger.info(f"LLM set global risk multiplier on resume: {mult_val}")
                            applied_mult = mult_val
                        else:
                            logger.warning(f"Invalid global_risk_multiplier in resume decision: {global_mult_raw}")
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid global_risk_multiplier value: {global_mult_raw}")

                # Resume trading
                pause_keys = [
                    "trading:paused",
                    "trading:pause_source",
                    "trading:pause_start",
                    "trading:pause_duration",
                    "trading:pause_reason",
                    "trading:llm_pause_time",
                ]
                for key in pause_keys:
                    await asyncio.to_thread(engine.redis.delete, key)
                # Reset the keep counter
                await asyncio.to_thread(engine.redis.delete, keep_key)
                logger.info("LLM decided to resume trading.")
                engine._reeval_trigger.set()
                if engine.notifier:
                    reason_text = f" – {reason}" if reason else ""
                    mult_text = f" (risk multiplier: {applied_mult})" if applied_mult is not None else ""
                    await engine.notifier.send_notification(
                        f"▶️ Trading resumed by LLM decision{reason_text}{mult_text}",
                        summary={"action": "RESUME", "reason": f"LLM resume request: {reason}" if reason else "LLM resume request", "model_type": "actuator", "llm_provider": llm_provider, "llm_model": llm_model}
                    )
            elif resume_trading is False:
                # LLM wants to stay paused – optionally update reason
                if reason:
                    await asyncio.to_thread(engine.redis.set, "trading:pause_reason", reason)

                # Increment consecutive keep counter
                new_keep_count = await asyncio.to_thread(engine.redis.incr, keep_key)
                # Set a TTL so it doesn't persist forever (e.g., 24h)
                await asyncio.to_thread(engine.redis.expire, keep_key, 86400)

                if new_keep_count >= max_keep:
                    # Double-check that the pause is still LLM-initiated (should always be true here)
                    current_source = await asyncio.to_thread(engine.redis.get, "trading:pause_source")
                    if current_source and (current_source.decode() if isinstance(current_source, bytes) else current_source) != "llm":
                        logger.warning("Force-resume skipped: pause source changed to non-LLM.")
                        return

                    # --- Drawdown circuit breaker: do not force-resume in significant drawdown ---
                    max_drawdown = settings.PAUSE_FORCE_RESUME_MAX_DRAWDOWN_PCT
                    if drawdown_pct >= max_drawdown:
                        logger.warning(
                            f"Force-resume blocked: account drawdown {drawdown_pct:.2f}% "
                            f"exceeds circuit breaker threshold {max_drawdown:.2f}%. "
                            f"Keeping trading paused for safety."
                        )
                        if engine.notifier:
                            await engine.notifier.send_notification(
                                f"⛔ Force-resume blocked: drawdown {drawdown_pct:.2f}% exceeds "
                                f"circuit breaker threshold {max_drawdown:.2f}%. "
                                f"Trading stays paused to protect capital. "
                                f"LLM has kept it paused {new_keep_count} time(s).",
                                summary={
                                    "action": "PAUSE",
                                    "reason": f"Force-resume blocked by drawdown circuit breaker ({drawdown_pct:.2f}% >= {max_drawdown:.2f}%)",
                                    "model_type": "actuator",
                                    "llm_provider": llm_provider,
                                    "llm_model": llm_model,
                                }
                            )
                        # Do NOT force-resume; let the LLM continue deciding
                        return

                    logger.warning(
                        f"LLM kept trading paused {new_keep_count} times consecutively – "
                        f"forcing resume with risk multiplier {force_resume_mult}."
                    )
                    # Force resume
                    pause_keys = [
                        "trading:paused",
                        "trading:pause_source",
                        "trading:pause_start",
                        "trading:pause_duration",
                        "trading:pause_reason",
                        "trading:llm_pause_time",
                    ]
                    for key in pause_keys:
                        await asyncio.to_thread(engine.redis.delete, key)
                    await asyncio.to_thread(engine.redis.delete, keep_key)
                    await engine._set_global_risk_multiplier(force_resume_mult)
                    engine._reeval_trigger.set()
                    if engine.notifier:
                        await engine.notifier.send_notification(
                            f"▶️ Trading force‑resumed after {new_keep_count} consecutive pauses. "
                            f"Global risk multiplier set to {force_resume_mult}.",
                            summary={
                                "action": "RESUME",
                                "reason": f"Force resume after {new_keep_count} consecutive keep-paused decisions",
                                "model_type": "actuator",
                            }
                        )
                else:
                    logger.info(f"LLM decided to keep trading paused. Reason: {reason} (keep count: {new_keep_count}/{max_keep})")
                    if engine.notifier:
                        reason_text = f" – {reason}" if reason else ""
                        await engine.notifier.send_notification(
                            f"⏸️ LLM decided to keep trading paused{reason_text} "
                            f"({new_keep_count}/{max_keep} consecutive keeps)",
                            summary={"action": "PAUSE", "reason": f"LLM keep paused: {reason}" if reason else "LLM keep paused", "model_type": "actuator", "llm_provider": llm_provider, "llm_model": llm_model}
                        )
            else:
                logger.warning(f"Invalid resume_trading value in LLM response: {resume_trading}")

    async def should_skip_llm_eval(
        self,
        symbol: str,
        current_price: float,
        atr: Optional[float],
        rsi: Optional[float],
        macd_hist: Optional[float],
        atr_percentile: Optional[float],
        market_regime: str,
        sentiment_trend_val: Optional[float],
        timeframe_seconds: float,
        has_position: bool,
        is_critical: bool,
    ) -> bool:
        """Return True if it's safe to skip the LLM call and just HOLD."""
        engine = self.engine
        # If a force evaluation was requested (entry signal detected), never skip
        async with engine._eval_state_lock:
            force_eval = engine._force_eval.get(symbol, False)
        if force_eval:
            return False
        # Never skip critical situations (max hold, stop-loss, take-profit triggered)
        if is_critical:
            return False

        # ATR is used for price-change comparison but is not strictly required.
        # When ATR is None (common for long timeframes like 1Y/3Y/5Y), we fall
        # back to a fixed percentage threshold so the skip logic still works
        # and we don't waste LLM calls every cycle.

        snapshot = engine._last_eval_snapshot.get(symbol)
        if snapshot is None:
            # First evaluation – must call
            return False

        now = time.time()
        last_time = snapshot.get("timestamp", 0)
        last_price = snapshot.get("price", 0)

        # Always call if enough time has passed (3× the effective interval)
        # For medium/long-term, be more patient before forcing an evaluation
        effective_interval = timeframe_seconds * settings.STRATEGY_INTERVAL_MULTIPLIER
        # Cap the safety net at the configured max skip interval so the bot
        # never skips LLM evaluations indefinitely, even for very long
        # timeframes (e.g., 1Y where 3× the interval would be ~3 years).
        # Cap the safety net at a value proportional to the timeframe,
        # but never less than the configured MAX_SKIP_INTERVAL_SECONDS.
        # This prevents excessively frequent forced evaluations for long
        # timeframes (e.g., 1Y candles should not be forced every 7 days).
        max_skip = max(settings.MAX_SKIP_INTERVAL_SECONDS, int(timeframe_seconds))
        if now - last_time > min(3 * effective_interval, max_skip):
            return False

        # Fetch LLM-driven skip thresholds from Redis.
        # Fall back to sensible hardcoded defaults when the LLM has not
        # configured them, so the skip logic is functional even before the
        # LLM provides values. The LLM can override these at any time via
        # its stock selection response.
        skip_price_mult_raw = await engine.config_service.get_config("skip_eval_price_change_atr_mult")
        skip_price_mult = float(skip_price_mult_raw) if skip_price_mult_raw else 1.0

        skip_rsi_raw = await engine.config_service.get_config("skip_eval_rsi_change")
        skip_rsi = float(skip_rsi_raw) if skip_rsi_raw else 5.0

        skip_macd_raw = await engine.config_service.get_config("skip_eval_macd_hist_change")
        skip_macd = float(skip_macd_raw) if skip_macd_raw else 0.0005

        # Price change since last evaluation
        if last_price > 0:
            price_change_pct = abs(current_price - last_price) / last_price
            # If price moved less than skip_price_mult × ATR (in %), it's boring
            atr_pct = (atr / current_price) if (atr and atr > 0) else 0.005
            if price_change_pct > atr_pct * skip_price_mult:
                return False   # enough movement to warrant a new look

        # Indicator changes
        last_rsi = snapshot.get("rsi")
        last_macd_hist = snapshot.get("macd_hist")
        if rsi is not None and last_rsi is not None:
            if abs(rsi - last_rsi) > skip_rsi:
                return False
        if macd_hist is not None and last_macd_hist is not None:
            if abs(macd_hist - last_macd_hist) > skip_macd:
                return False

        # MACD histogram sign change (crossover) — momentum shift
        if macd_hist is not None and last_macd_hist is not None:
            if (macd_hist > 0) != (last_macd_hist > 0):
                return False

        # If we have no open position and nothing is screaming, skip
        if not has_position:
            # Only call if there is a potential entry signal (extreme RSI, MACD crossover, etc.)
            # RSI extreme? (thresholds are LLM-decided)
            # RSI extremes are optional – only use them if the LLM has set them.
            rsi_oversold = None
            rsi_overbought = None
            try:
                raw = await engine.config_service.get_config("skip_eval_rsi_oversold")
                if raw:
                    rsi_oversold = float(raw)
                raw = await engine.config_service.get_config("skip_eval_rsi_overbought")
                if raw:
                    rsi_overbought = float(raw)
            except (ValueError, TypeError, ConnectionError, TimeoutError, OSError):
                pass
            if (
                rsi is not None
                and rsi_oversold is not None
                and rsi_overbought is not None
                and (rsi < rsi_oversold or rsi > rsi_overbought)
            ):
                return False
            # MACD histogram direction change? (harder to detect without previous sign – skip for simplicity)
            # Otherwise, no strong signal → skip
            return True

        # Have an open position – skip if price far from stop/tp and indicators calm
        # (the risk management loop will handle stop/tp)
        return True

    async def _compute_volume_trend(self, symbol: str, current_volume: float, timeframe: Optional[str] = None) -> Optional[float]:
        """Compute volume trend as ratio of current 24h volume to EMA of past volumes.

        Returns the ratio (e.g., 2.0 means current volume is 2× the average).
        Returns None if volume data is unavailable.

        For long-term timeframes (>= 1 day), the ratio is cached in Redis
        to avoid recomputing on every evaluation cycle.
        """
        engine = self.engine
        if current_volume <= 0:
            return None

        # Determine cache TTL based on timeframe
        cache_ttl = 0  # no caching by default
        if timeframe is not None:
            tf_seconds = engine._timeframe_to_seconds(timeframe)
            if tf_seconds >= 2_592_000:  # >= 1 month
                cache_ttl = 3600       # 1 hour
            elif tf_seconds >= 604_800:  # >= 1 week
                cache_ttl = 1800       # 30 minutes
            elif tf_seconds >= 86_400:  # >= 1 day
                cache_ttl = 900        # 15 minutes

        # Check ratio cache first (long timeframes only)
        ratio_cache_key = f"volume_trend:ratio:{symbol}"
        if cache_ttl > 0:
            try:
                cached_ratio = await asyncio.to_thread(engine.redis.get, ratio_cache_key)
                if cached_ratio is not None:
                    return round(float(cached_ratio), 3)
            except (ValueError, TypeError, ConnectionError, TimeoutError, OSError):
                pass

        redis_key = f"volume_trend:ema:{symbol}"
        alpha = 0.3  # EMA smoothing factor

        try:
            stored = await asyncio.to_thread(engine.redis.get, redis_key)
            if stored is not None:
                old_avg = float(stored)
                new_avg = alpha * current_volume + (1 - alpha) * old_avg
                ratio = current_volume / old_avg if old_avg > 0 else 1.0
                # Store the updated average with 7-day TTL
                await asyncio.to_thread(engine.redis.setex, redis_key, 7 * 24 * 3600, str(new_avg))
                # Cache the ratio for long-term timeframes
                if cache_ttl > 0:
                    await asyncio.to_thread(engine.redis.setex, ratio_cache_key, cache_ttl, str(ratio))
                return round(ratio, 3)
            else:
                # First observation: initialize with current volume, ratio = 1.0
                await asyncio.to_thread(engine.redis.setex, redis_key, 7 * 24 * 3600, str(current_volume))
                if cache_ttl > 0:
                    await asyncio.to_thread(engine.redis.setex, ratio_cache_key, cache_ttl, "1.0")
                return 1.0
        except (ValueError, TypeError, ConnectionError, TimeoutError, OSError) as e:
            logger.info(f"Volume trend computation failed for {symbol}: {e}")
            return None

    async def detect_entry_signal(self, symbol: str, timeframe: str) -> bool:
        """Return True if a favourable entry condition is detected for the symbol.
        Uses recent OHLCV data from the database and compares with previous state."""
        engine = self.engine
        # Fetch pre-computed indicators from DB
        ind = await asyncio.to_thread(get_indicators, symbol, timeframe)

        # Still need candles for volume EMA computation and fallback
        # indicator computation
        db_candles = await asyncio.to_thread(
            get_ohlcv, symbol, timeframe, limit=50
        )
        if len(db_candles) < 26:
            return False

        # If DB indicators are missing (common for long timeframes like
        # 1Y where indicators may not be stored), compute them
        # on-the-fly from the OHLCV candles so entry signal detection
        # still works.
        if not ind:
            raw_candles = [
                [c["timestamp"], c["open"], c["high"], c["low"], c["close"], c["volume"]]
                for c in db_candles
            ]
            try:
                ind = await asyncio.to_thread(compute_all_indicators, raw_candles)
            except (ValueError, TypeError, IndexError, ZeroDivisionError) as e:
                logger.debug(
                    f"Failed to compute indicators on-the-fly for {symbol} {timeframe}: {e}"
                )
                return False
            if not ind:
                return False

        closes = [c["close"] for c in db_candles]
        volumes = [c["volume"] for c in db_candles]

        # Retrieve previous state
        prev = engine._entry_signal_state.get(symbol, {})

        # Current values
        rsi = ind.get("rsi")
        macd_hist = ind.get("macd_hist")
        macd_val = ind.get("macd")
        macd_signal = ind.get("macd_signal")
        stoch_k = ind.get("stochastic_k")
        adx = ind.get("adx")
        plus_di = ind.get("plus_di")
        minus_di = ind.get("minus_di")
        bb_upper = ind.get("bb_upper")
        bb_lower = ind.get("bb_lower")
        bb_middle = ind.get("bb_middle")
        ema_9 = ind.get("ema_9")
        ema_21 = ind.get("ema_21")
        parabolic_sar = ind.get("parabolic_sar")
        ichimoku = ind.get("ichimoku")
        current_close = closes[-1] if closes else None

        # Volume EMA for spike detection (using talib via compute_ema).
        # Exclude the latest candle (which may be incomplete for intraday
        # timeframes) by using the second-to-last EMA value.
        volume_ema_list = compute_ema(volumes, 20)
        volume_ema = volume_ema_list[-2] if len(volume_ema_list) >= 2 else 0.0

        # Store current state for next cycle
        new_state = {
            "rsi": rsi,
            "macd_hist": macd_hist,
            "macd_val": macd_val,
            "macd_signal": macd_signal,
            "stoch_k": stoch_k,
            "adx": adx,
            "plus_di": plus_di,
            "minus_di": minus_di,
            "bb_upper": bb_upper,
            "bb_lower": bb_lower,
            "bb_middle": bb_middle,
            "ema_9": ema_9,
            "ema_21": ema_21,
            "parabolic_sar": parabolic_sar,
            "ichimoku_cloud_top": ichimoku["cloud_top"] if ichimoku else None,
            "ichimoku_cloud_bottom": ichimoku["cloud_bottom"] if ichimoku else None,
            "close": current_close,
            "volume_ema": volume_ema,
        }
        engine._entry_signal_state[symbol] = new_state

        # --- Read LLM-defined thresholds from Redis (fallback to defaults) ---
        rsi_oversold = 30.0
        rsi_overbought = 70.0
        adx_moderate = 25.0
        bb_squeeze_width = 0.02
        try:
            raw = await engine.config_service.get_config("skip_eval_rsi_oversold")
            if raw:
                rsi_oversold = float(raw)
            raw = await engine.config_service.get_config("skip_eval_rsi_overbought")
            if raw:
                rsi_overbought = float(raw)
            raw = await engine.config_service.get_config("regime_adx_moderate")
            if raw:
                adx_moderate = float(raw)
            raw = await engine.config_service.get_config("regime_bb_squeeze_width")
            if raw:
                bb_squeeze_width = float(raw)
        except (ValueError, TypeError, ConnectionError, TimeoutError, OSError):
            pass

        prev_close = prev.get("close")

        # --- Long-term timeframe detection (>= 1 month) ---
        # Use trend reversal and regime shift logic instead of short-term
        # crossovers, which fire on every candle for long timeframes because
        # indicators change dramatically between candles (e.g., RSI 20→80).
        tf_seconds = engine._timeframe_to_seconds(timeframe)
        if tf_seconds >= 2_592_000:  # >= 1 month (1M, 3M, 6M, 1Y)
            # 1. Trend direction reversal: +DI crosses above -DI
            prev_plus_di = prev.get("plus_di")
            prev_minus_di = prev.get("minus_di")
            if (prev_plus_di is not None and prev_minus_di is not None
                    and plus_di is not None and minus_di is not None
                    and prev_plus_di <= prev_minus_di and plus_di > minus_di):
                return True

            # 2. Trend initiation: ADX crosses above moderate threshold
            prev_adx = prev.get("adx")
            if (prev_adx is not None and adx is not None
                    and prev_adx <= adx_moderate and adx > adx_moderate
                    and plus_di is not None and minus_di is not None
                    and plus_di > minus_di):
                return True

            # 3. Major breakout: price breaks above Donchian upper channel
            donchian = ind.get("donchian_channels")
            if (donchian is not None and prev_close is not None
                    and current_close is not None):
                dc_upper = donchian.get("upper")
                if dc_upper is not None and prev_close <= dc_upper and current_close > dc_upper:
                    return True

            # 4. Ichimoku cloud breakout: price crosses above cloud top
            prev_cloud_top = prev.get("ichimoku_cloud_top")
            if (prev_cloud_top is not None and ichimoku is not None
                    and prev_close is not None and current_close is not None):
                cloud_top = ichimoku.get("cloud_top")
                if cloud_top is not None and prev_close <= cloud_top and current_close > cloud_top:
                    return True

            # 5. MACD zero-line crossover (long-term momentum shift)
            # Require a minimum magnitude to avoid firing on tiny swings
            # that are common on long timeframes (1M, 3M, 6M, 1Y).
            # Use a lower threshold (0.05× ATR) so low-volatility assets
            # like BTPs are not filtered out.
            prev_macd_val = prev.get("macd_val")
            _atr = ind.get("atr")
            if (prev_macd_val is not None and macd_val is not None
                    and prev_macd_val <= 0 and macd_val > 0
                    and _atr is not None and _atr > 0
                    and abs(macd_val) > settings.ENTRY_SIGNAL_MACD_ATR_MULT * _atr):
                return True

            # 6. EMA golden cross (valid for long timeframes — major trend shift)
            prev_ema_9 = prev.get("ema_9")
            prev_ema_21 = prev.get("ema_21")
            if (prev_ema_9 is not None and prev_ema_21 is not None
                    and ema_9 is not None and ema_21 is not None
                    and prev_ema_9 <= prev_ema_21 and ema_9 > ema_21):
                return True

            # No long-term entry signal detected
            return False

        # --- Condition checks ---
        # 1. RSI oversold
        if rsi is not None and rsi < rsi_oversold:
            return True

        # 2. MACD histogram bullish crossover (was negative, now positive)
        prev_macd_hist = prev.get("macd_hist")
        if (prev_macd_hist is not None and macd_hist is not None
                and prev_macd_hist <= 0 and macd_hist > 0):
            return True

        # 3. RSI leaving oversold (momentum shift)
        prev_rsi = prev.get("rsi")
        if (prev_rsi is not None and rsi is not None
                and prev_rsi < rsi_oversold and rsi >= rsi_oversold):
            return True

        # 4. MACD line crossing above signal line (bullish crossover)
        prev_macd_val = prev.get("macd_val")
        prev_macd_signal = prev.get("macd_signal")
        if (prev_macd_val is not None and prev_macd_signal is not None
                and macd_val is not None and macd_signal is not None
                and prev_macd_val <= prev_macd_signal and macd_val > macd_signal):
            return True

        # 6. ADX rising above moderate threshold and +DI > -DI (trend start)
        prev_adx = prev.get("adx")
        if (adx is not None and plus_di is not None and minus_di is not None
                and plus_di > minus_di
                and prev_adx is not None and prev_adx <= adx_moderate and adx > adx_moderate):
            return True

        # 7. Bollinger Band squeeze breakout
        prev_bb_upper = prev.get("bb_upper")
        prev_bb_lower = prev.get("bb_lower")
        prev_bb_middle = prev.get("bb_middle")
        if (prev_bb_upper is not None and prev_bb_lower is not None and prev_bb_middle is not None
                and bb_upper is not None and bb_lower is not None and bb_middle is not None
                and prev_bb_middle > 0 and bb_middle > 0):
            prev_width = (prev_bb_upper - prev_bb_lower) / prev_bb_middle
            curr_width = (bb_upper - bb_lower) / bb_middle
            if prev_width < bb_squeeze_width and current_close is not None and current_close > bb_upper:
                return True

        # 8. Volume spike (last COMPLETE candle volume > 3 * EMA of volume)
        # Use the second-to-last candle to avoid false signals from the
        # latest candle which may still be forming (incomplete volume).
        if len(volumes) >= 2 and volume_ema > 0 and volumes[-2] > 3.0 * volume_ema:
            return True

        # 9. EMA9 crossing above EMA21 (golden cross)
        prev_ema_9 = prev.get("ema_9")
        prev_ema_21 = prev.get("ema_21")
        if (prev_ema_9 is not None and prev_ema_21 is not None
                and ema_9 is not None and ema_21 is not None
                and prev_ema_9 <= prev_ema_21 and ema_9 > ema_21):
            return True

        # 11. Parabolic SAR flip (from above price to below price → uptrend)
        prev_sar = prev.get("parabolic_sar")
        if (prev_sar is not None and parabolic_sar is not None
                and prev_close is not None and current_close is not None
                and prev_sar > prev_close and parabolic_sar < current_close):
            return True

        # 12. Ichimoku: price crossing above cloud
        prev_cloud_top = prev.get("ichimoku_cloud_top")
        prev_cloud_bottom = prev.get("ichimoku_cloud_bottom")
        if (prev_cloud_top is not None and prev_cloud_bottom is not None
                and ichimoku is not None
                and prev_close is not None and current_close is not None):
            cloud_top = ichimoku["cloud_top"]
            cloud_bottom = ichimoku["cloud_bottom"]
            # Previous close was below or inside cloud, current close above cloud top
            if prev_close <= cloud_top and current_close > cloud_top:
                return True

        return False

    async def check_entry_condition_once(
        self, symbol: str, condition: Dict[str, Any], timeframe: str
    ) -> bool:
        """Check a single entry condition immediately. Return True if met."""
        engine = self.engine
        etype = condition.get("type")
        if etype == "limit_price":
            target_price = condition["price"]
            try:
                tickers_map = await engine._get_quotes_async([symbol.split("/")[0]], timeout=45.0)
                ticker = tickers_map.get(symbol.split("/")[0])
            except (ConnectionError, TimeoutError, OSError, ValueError, TypeError, KeyError):
                return False
            current_price = ticker.get("last", 0) if ticker else 0
            return current_price > 0 and current_price <= target_price

        elif etype == "rsi_threshold":
            target_rsi = condition["rsi_below"]
            ind = await asyncio.to_thread(get_indicators, symbol, timeframe)
            if ind:
                rsi = ind.get("rsi")
                return rsi is not None and rsi <= target_rsi
            return False

        elif etype == "delay":
            # Delay conditions are handled by _execute_delayed_entry, not the
            # pending-entries system. If we somehow reach here, treat as not met
            # so the deadline handler can deal with it.
            return False

        elif etype == "indicator_combo":
            conditions = condition["conditions"]
            ind = await asyncio.to_thread(get_indicators, symbol, timeframe)
            if not ind:
                return False
            # Mapping of indicator names the LLM can use to DB keys.
            # All scalar indicators stored in the indicators table are supported.
            _INDICATOR_KEYS = {
                "rsi": "rsi",
                "macd": "macd",
                "macd_signal": "macd_signal",
                "macd_hist": "macd_hist",
                "bb_upper": "bb_upper",
                "bb_middle": "bb_middle",
                "bb_lower": "bb_lower",
                "ema_9": "ema_9",
                "ema_21": "ema_21",
                "stochastic_k": "stochastic_k",
                "stochastic_d": "stochastic_d",
                "adx": "adx",
                "plus_di": "plus_di",
                "minus_di": "minus_di",
                "obv": "obv",
                "mfi": "mfi",
                "cci": "cci",
                "williams_r": "williams_r",
                "parabolic_sar": "parabolic_sar",
                "atr": "atr",
            }
            for cond in conditions:
                indicator_name = cond["indicator"]
                thresh = cond["threshold"]
                direction = cond["direction"]
                db_key = _INDICATOR_KEYS.get(indicator_name)
                if db_key is None:
                    logger.warning(
                        f"Unsupported indicator '{indicator_name}' in indicator_combo "
                        f"entry condition for {symbol}"
                    )
                    return False
                val = ind.get(db_key)
                if val is None:
                    return False
                if direction == "below" and val > thresh:
                    return False
                if direction == "above" and val < thresh:
                    return False
            return True

        return False

    async def process_pending_entry(self, symbol: str, now: float) -> None:
        """Process a single pending entry: check timeout and condition, execute if met."""
        engine = self.engine
        async with engine._pending_entries_lock:
            entry = engine._pending_entries.get(symbol)
            if entry is None:
                return
            entry_tf = entry.get("timeframe")
            stock_name = await engine._get_stock_name(symbol)
            display_symbol = engine._format_symbol_display(symbol, stock_name, entry_tf)
            if now >= entry["deadline"]:
                # Timeout – clear and notify
                logger.info(f"Entry condition timeout for {symbol}")
                del engine._pending_entries[symbol]
                engine._state_dirty = True
                _timed_out = True
                _signal = None
            else:
                # Check the condition (non‑blocking)
                condition_met = await self.check_entry_condition_once(
                    symbol, entry["condition"], entry["timeframe"]
                )
                if condition_met:
                    logger.info(f"Entry condition met for {symbol}, executing BUY")
                    # Remove from pending before executing to avoid re‑trigger
                    _signal = entry["signal"]
                    del engine._pending_entries[symbol]
                    engine._state_dirty = True
                    _timed_out = False
                else:
                    return  # condition not met, keep pending

        if _timed_out:
            if engine.notifier:
                await engine.notifier.send_notification(
                    f"⏭️ Entry condition timeout for {display_symbol} – skipping BUY.",
                    summary={
                        "symbol": symbol,
                        "action": "SKIP",
                        "reason": "Entry condition timeout",
                    }
                )
            return

        # condition_met is True here
        # Check trading pause again (may have changed)
        paused = await asyncio.to_thread(engine.redis.get, "trading:paused")
        if paused:
            logger.info(f"Ignoring queued BUY {symbol}: trading is now paused.")
            if engine.notifier:
                await engine.notifier.send_notification(
                    f"⏸️ Queued BUY for {display_symbol} skipped – trading paused.",
                    summary={"symbol": symbol, "action": "SKIP", "reason": "Trading paused"}
                )
        else:
            await self.event_bus.publish(
                "execute_signal",
                symbol,
                _signal,
                timeframe=entry_tf,
                atr=None,
            )

    async def run_simulation_step1(
        self,
        symbol: str,
        data: Dict[str, Any],
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[Signal], Optional[Dict[str, Any]]]:
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
                asyncio.to_thread(
                    get_cached_llm_response,
                    compact_prompt(data["analysis_prompt"]),
                    compact_prompt(build_system_prompt()), 60,
                    market_hash=market_hash,
                    model_type=model_type,
                    temperature=temperature,
                ),
                timeout=settings.LLM_TIMEOUT
            )
            step1a_response = step1a_result["response"]
        except (asyncio.TimeoutError, ConnectionError, TimeoutError, OSError, ValueError, TypeError, RuntimeError, json.JSONDecodeError) as e:
            return None, None, None, {"error": f"LLM Step 1a call failed: {e}"}

        analysis = self._parse_analysis_response(step1a_response)
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
        variants_prompt = await asyncio.to_thread(
            build_backtest_variants_prompt,
            prompt_data
        )

        try:
            step1b_result = await asyncio.wait_for(
                asyncio.to_thread(
                    get_cached_llm_response,
                    compact_prompt(variants_prompt),
                    compact_prompt(build_system_prompt()), 60,
                    market_hash=compute_market_hash({"step": "1b", "analysis": analysis}),
                    model_type=model_type,
                    temperature=temperature,
                ),
                timeout=settings.LLM_TIMEOUT
            )
            step1b_response = step1b_result["response"]
        except (asyncio.TimeoutError, ConnectionError, TimeoutError, OSError, ValueError, TypeError, RuntimeError, json.JSONDecodeError) as e:
            return None, None, None, {"error": f"LLM Step 1b call failed: {e}"}

        try:
            preliminary_strategy = create_strategy_from_llm(step1b_response)
            preliminary_signal = preliminary_strategy.generate_signal({})
        except ValueError as e:
            return None, None, None, {"error": f"Failed to parse Step 1b response: {e}", "raw_response": step1b_response}

        return analysis, step1b_response, preliminary_signal, None

    async def prepare_simulation_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch all necessary data and build the strategy prompt for simulation."""
        engine = self.engine
        symbol_entry = next((e for e in engine.current_symbols if e["symbol"] == symbol), None)
        if not symbol_entry:
            return {"error": f"Symbol {symbol} not found in current_symbols"}
        
        assigned_tf = symbol_entry["timeframe"]

        symbol_data = await self.fetch_symbol_market_data(symbol, assigned_tf)
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

        open_positions = [pos for pos in engine.positions.values() if pos.get("symbol") == symbol]

        _ctx = await self.gather_prompt_context(
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
        if symbol in engine.positions:
            pos = engine.positions[symbol]
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

        remaining = max(0.0, base_balance - engine._cycle_spent)

        perf = await engine._compute_performance_metrics()
        trade_pattern_analysis = await engine._compute_trade_pattern_analysis()

        symbol_event = None
        if settings.NEWS_ENABLED and detect_upcoming_events is not None:
            try:
                symbol_event = await asyncio.to_thread(detect_upcoming_events, symbol)
            except (ConnectionError, TimeoutError, OSError, ValueError, TypeError, json.JSONDecodeError):
                pass

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
            all_symbols=engine.current_symbols,
            past_trades=past_trades,
            cycle_spent=engine._cycle_spent,
            remaining_balance=remaining,
            market_regime=market_regime,
            multi_tf_raw_candles=multi_tf_raw_candles,
            multi_tf_indicators=multi_tf_indicators,
            session_info=session_info,
            sentiment_trend=sentiment_trend_val,
            volume_trend=volume_trend_val,
            market_breadth=getattr(engine, '_market_breadth', None),
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
            portfolio_open_count=len(engine.positions),
            portfolio_available_capital=portfolio_available_capital,
            last_decision=engine._last_decisions.get(symbol),
            minutes_to_market_close=minutes_to_market_close,
            current_strategy_interval_seconds=engine._strategy_intervals.get(symbol, engine._timeframe_to_seconds(assigned_tf)),
            max_portfolio_exposure_pct=max_port_exp,
            max_portfolio_stop_risk_pct=max_port_risk,
            trade_pattern_analysis=trade_pattern_analysis,
            symbol_event=symbol_event,
            queued_orders=engine.queued_orders,
            fundamentals=fundamentals,
            min_hold_time_mult=sim_min_hold_time_mult,
            min_stop_atr_mult=sim_min_stop_atr_mult,
            min_viable_trade_amount=min_viable_amount,
            historical_backtest_results=historical_backtest_results,
            aggregate_sentiment=aggregate_sentiment,
        )
        analysis_prompt, market_snapshot, market_hash = await self.build_analysis_prompt_and_snapshot(prompt_data)

        strategy_model_type, effective_temp = self.compute_model_tier_and_temperature(
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
            market_breadth=getattr(engine, '_market_breadth', None),
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
            num_candidates=len(engine.current_symbols),
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
            "has_position": symbol in engine.positions,
            "historical_backtest_results": historical_backtest_results,
        }
