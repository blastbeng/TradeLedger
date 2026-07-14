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
from src.utils.btp_policy import BTPPolicy

from src.trading.components.signal_market_data import SignalMarketDataFetcher
from src.trading.components.model_tier_manager import ModelTierManager
from src.trading.components.entry_signal_manager import EntrySignalManager
from src.trading.components.llm_step_manager import LLMStepManager
from src.trading.components.simulation_manager import SimulationManager
from src.trading.components.post_decision_manager import PostDecisionManager
from src.trading.components.pause_resume_manager import PauseResumeManager

try:
    from src.news.fetcher import detect_upcoming_events, get_upcoming_earnings
except ImportError:
    detect_upcoming_events = None
    get_upcoming_earnings = None

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
        self.shared_state = engine.shared_state
        self.event_bus = event_bus
        self.market_data_fetcher = SignalMarketDataFetcher(engine)
        self.model_tier_manager = ModelTierManager(engine)
        self.entry_signal_manager = EntrySignalManager(engine, event_bus)
        self.llm_step_manager = LLMStepManager(self)
        self.simulation_manager = SimulationManager(self)
        self.post_decision_manager = PostDecisionManager(engine, event_bus)
        self.pause_resume_manager = PauseResumeManager(self)
        self.event_bus.subscribe("process_symbol", self.process_symbol)
        self.event_bus.subscribe("check_pause_resume_decision", self.pause_resume_manager.check_pause_resume_decision)
        self.event_bus.subscribe("detect_entry_signal", self.entry_signal_manager.detect_entry_signal)
        self.event_bus.subscribe("process_pending_entry", self.entry_signal_manager.process_pending_entry)
        self.event_bus.subscribe("check_entry_condition_once", self.entry_signal_manager.check_entry_condition_once)
        self._skip_config_cache: Dict[str, float] = {}
        self._skip_config_cache_time: float = 0.0
        self._skip_config_cache_ttl: float = 300.0  # 5 minutes

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
        stock_name = await engine._market_data_manager.get_stock_name(symbol)
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
        has_position = symbol in self.shared_state.positions

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

        open_positions = [pos for pos in self.shared_state.positions.values() if pos.get("symbol") == symbol]
        per_symbol_budget = base_balance / engine.effective_max_symbols if engine.effective_max_symbols > 0 else 0.0

        perf = await engine.event_bus.request("compute_performance_metrics")
        trade_pattern_analysis = await engine.event_bus.request("compute_trade_pattern_analysis")

        symbol_event = None
        if settings.NEWS_ENABLED and detect_upcoming_events is not None:
            try:
                symbol_event = await asyncio.to_thread(detect_upcoming_events, symbol)
            except (ConnectionError, TimeoutError, OSError, ValueError, TypeError, json.JSONDecodeError):
                pass

        upcoming_earnings = None
        if get_upcoming_earnings is not None and symbol in self.shared_state.positions:
            try:
                upcoming_earnings = await asyncio.to_thread(get_upcoming_earnings, symbol)
            except (ValueError, TypeError, KeyError, ConnectionError, TimeoutError, OSError):
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
        async with self.shared_state._cycle_spent_lock:
            remaining = max(0.0, base_balance - self.shared_state._cycle_spent)

        # Pre-summarize news for the prompt to avoid synchronous LLM calls in prompt builder
        news_section = None
        if settings.NEWS_ENABLED:
            try:
                from src.llm.prompt_utils import get_cached_news_summary_async
                news_summary = await get_cached_news_summary_async(symbol, model_type="weak")
                if news_summary and news_summary.get("summary") and news_summary["summary"] != "No recent news.":
                    news_section = f"Recent news summary for {symbol}: {news_summary['summary']}"
            except (ValueError, TypeError, KeyError, ConnectionError, TimeoutError, OSError, json.JSONDecodeError) as e:
                logger.warning(f"Failed to pre-summarize news for {symbol}: {type(e).__name__}: {e}")

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
            all_symbols=self.shared_state.current_symbols,
            past_trades=_ctx["past_trades"],
            cycle_spent=self.shared_state._cycle_spent,
            remaining_balance=remaining,
            market_regime=_ctx["market_regime"],
            multi_tf_raw_candles=symbol_data["multi_tf_raw_candles"],
            multi_tf_indicators=symbol_data["multi_tf_indicators"],
            session_info=_ctx["session_info"],
            sentiment_trend=_ctx["sentiment_trend_val"],
            volume_trend=_ctx["volume_trend_val"],
            market_breadth=self.shared_state._market_breadth,
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
            portfolio_open_count=len(self.shared_state.positions),
            portfolio_available_capital=_portfolio["portfolio_available_capital"],
            last_decision=self.shared_state._last_decisions.get(symbol),
            minutes_to_market_close=_ctx["minutes_to_market_close"],
            current_strategy_interval_seconds=self.shared_state._strategy_intervals.get(symbol, engine._timeframe_to_seconds(assigned_tf)),
            max_portfolio_exposure_pct=_ctx["max_port_exp"],
            max_portfolio_stop_risk_pct=_ctx["max_port_risk"],
            trade_pattern_analysis=trade_pattern_analysis,
            symbol_event=symbol_event,
            upcoming_earnings=upcoming_earnings,
            queued_orders=self.shared_state.queued_orders,
            fundamentals=symbol_data["fundamentals"],
            min_hold_time_mult=_ctx["min_hold_time_mult"],
            min_stop_atr_mult=_ctx["min_stop_atr_mult"],
            min_viable_trade_amount=min_viable_amount,
            historical_backtest_results=_ctx["historical_backtest_results"],
            aggregate_sentiment=_ctx["aggregate_sentiment"],
            ytm=symbol_data["ytm"],
            dividend_yield=_ctx.get("dividend_yield"),
            next_ex_dividend=_ctx.get("next_ex_dividend"),
            news_section=news_section,
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
            "upcoming_earnings": upcoming_earnings,
            "consecutive_losses": perf.get("equity_curve", {}).get("consecutive_losses", 0),
            "atr_percentile": _ctx["atr_percentile"],
            "fundamentals": symbol_data["fundamentals"],
        }

    async def _check_skip_and_model_tier(self, symbol: str, ctx: Dict[str, Any], flags: Dict[str, Any]) -> Optional[Tuple[str, float]]:
        """Check if LLM eval should be skipped and compute model tier. Returns (model_type, temp) or None if skipped."""
        engine = self.engine
        is_critical = flags["max_hold_expired"] or flags["stop_loss_triggered"] or flags["take_profit_triggered"] or flags["partial_tp_triggered"] or flags["dust_sweep_triggered"]
        has_position = symbol in self.shared_state.positions

        if await self.should_skip_llm_eval(
            symbol=symbol, current_price=ctx["current_price"], atr=ctx["atr"], rsi=ctx["rsi"],
            macd_hist=ctx["macd_hist"], atr_percentile=ctx.get("atr_percentile"), market_regime=ctx["market_regime"],
            sentiment_trend_val=ctx.get("sentiment_trend_val"), timeframe_seconds=engine._timeframe_to_seconds(ctx.get("assigned_tf", "1d")),
            has_position=has_position, is_critical=is_critical,
        ):
            logger.info(f"Skipping LLM for {symbol}: market unchanged, no strong signals.")
            async with self.shared_state._eval_state_lock:
                self.shared_state._force_eval.pop(symbol, None)
            # Do NOT update the snapshot timestamp here — it must only be updated
            # after a real LLM call (done in LLMStepManager._update_last_eval_snapshot).
            # Updating it on skip would reset the clock and defeat the time-based
            # fallback that ensures periodic re-evaluation.
            return None

        strategy_model_type, effective_temp = self.model_tier_manager.compute_model_tier_and_temperature(
            atr=ctx["atr"], atr_percentile=ctx.get("atr_percentile"), rsi=ctx["rsi"], macd=ctx["macd"],
            macd_signal=ctx["macd_signal"], macd_hist=ctx["macd_hist"], bb_upper=ctx["bb_upper"], bb_middle=ctx["bb_middle"], bb_lower=ctx["bb_lower"],
            ema_9=ctx["ema_9"], ema_21=ctx["ema_21"], stochastic_k=ctx["stochastic_k"], adx=ctx["adx"], plus_di=ctx["plus_di"], minus_di=ctx["minus_di"],
            mfi=ctx["mfi"], cci=ctx["cci"], williams_r=ctx["williams_r"], ichimoku=ctx["ichimoku"], market_regime=ctx["market_regime"],
            market_breadth=self.shared_state._market_breadth, full_market_breadth=ctx.get("full_market_breadth"),
            sentiment_trend_val=ctx.get("sentiment_trend_val"), volume_trend_val=ctx.get("volume_trend_val"),
            unrealized_pnl=ctx.get("unrealized_pnl"), drawdown_pct=ctx.get("drawdown_pct"),
            portfolio_exposure_pct=ctx["portfolio_exposure_pct"], portfolio_stop_risk_pct=ctx["portfolio_stop_risk_pct"],
            is_critical=is_critical, trading_paused=False, symbol_event=ctx.get("symbol_event"),
            fundamentals=ctx.get("fundamentals"), consecutive_losses=ctx.get("consecutive_losses", 0),
            current_price=ctx["current_price"], timeframe=ctx.get("assigned_tf"),
            num_candidates=len(self.shared_state.current_symbols),
        )
        return strategy_model_type, effective_temp

    async def _run_llm_steps(self, symbol: str, display_symbol: str, symbol_entry: Dict[str, str], ctx: Dict[str, Any], strategy_model_type: str, effective_temp: float, flags: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Run Step 1a, 1b, and Step 2 LLM calls. Returns dict with signal/provider/model or None if should return."""
        engine = self.engine
        assigned_tf = symbol_entry["timeframe"]
        tf_seconds = engine._timeframe_to_seconds(assigned_tf)
        is_critical = flags["max_hold_expired"] or flags["stop_loss_triggered"] or flags["take_profit_triggered"] or flags["partial_tp_triggered"] or flags["dust_sweep_triggered"]
        has_position = symbol in self.shared_state.positions

        critical_reason = None
        if is_critical:
            critical_reason = "LLM timeout"
            if flags["max_hold_expired"]: critical_reason = "Max hold expired, LLM timeout"
            elif flags["stop_loss_triggered"]: critical_reason = "Stop-loss triggered, LLM timeout"
            elif flags["take_profit_triggered"]: critical_reason = "Take-profit triggered, LLM timeout"
            elif flags["partial_tp_triggered"]: critical_reason = "Partial TP triggered, LLM timeout"
            elif flags["dust_sweep_triggered"]: critical_reason = "Dust sweep triggered, LLM timeout"

        analysis_result, llm_provider, llm_model, _should_return = await self.llm_step_manager.run_step1a_llm_call(
            symbol=symbol, display_symbol=display_symbol, analysis_prompt=ctx["analysis_prompt"],
            system_prompt=compact_prompt(build_system_prompt()), market_hash=ctx["market_hash"],
            strategy_model_type=strategy_model_type, effective_temp=effective_temp,
            current_price=ctx["current_price"], rsi=ctx["rsi"], macd_hist=ctx["macd_hist"],
            is_critical=is_critical, critical_reason=critical_reason,
            tf_seconds=tf_seconds,
        )
        if _should_return:
            return None

        signal, combined_bt_summary, llm_provider, llm_model, _skip_backtest = await self.llm_step_manager.handle_step1a_fallback(
            symbol=symbol, analysis_result=analysis_result, has_position=has_position,
            strategy_model_type=strategy_model_type, llm_provider=llm_provider, llm_model=llm_model,
        )

        if not _skip_backtest:
            preliminary_signal, llm_provider, llm_model = await self.llm_step_manager.run_step1b_llm_call(
                symbol=symbol, analysis_result=analysis_result, ticker=ctx["ticker"], current_price=ctx["current_price"],
                atr=ctx["atr"], assigned_tf=assigned_tf, base_balance=ctx["base_balance"], per_symbol_budget=ctx["per_symbol_budget"],
                min_order_amount=ctx["min_order_amount"], min_order_cost=ctx["min_order_cost"], remaining=ctx["remaining"],
                portfolio_total_value=ctx["portfolio_total_value"], portfolio_exposure_pct=ctx["portfolio_exposure_pct"],
                portfolio_stop_risk_pct=ctx["portfolio_stop_risk_pct"], portfolio_available_capital=ctx["portfolio_available_capital"],
                max_port_exp=ctx["max_port_exp"], max_port_risk=ctx["max_port_risk"], global_risk_mult=ctx["global_risk_mult"],
                min_stop_atr_mult=ctx["min_stop_atr_mult"], min_hold_time_mult=ctx["min_hold_time_mult"], trading_paused=False,
                has_position=has_position, strategy_model_type=strategy_model_type, effective_temp=effective_temp,
                market_snapshot=ctx["market_snapshot"], historical_backtest_results=ctx["historical_backtest_results"],
                is_critical=is_critical,
            )
            signal, combined_bt_summary, llm_provider, llm_model = await engine.event_bus.request(
                "run_backtest_and_final_decision",
                symbol=symbol, assigned_tf=assigned_tf, tf_seconds=tf_seconds, current_price=ctx["current_price"],
                atr=ctx["atr"], historical_ohlcv=ctx["historical_ohlcv"], raw_candles=ctx["raw_candles"],
                base_balance=ctx["base_balance"], is_btp=ctx["is_btp"], trading_paused=False,
                strategy_model_type=strategy_model_type, effective_temp=effective_temp,
                preliminary_signal=preliminary_signal, display_symbol=display_symbol, ticker=ctx["ticker"],
                market_hash=ctx["market_hash"],
                is_critical=is_critical,
            )

        # --- Automatic adjustment for upcoming earnings ---
        upcoming_earnings = ctx.get("upcoming_earnings")
        if upcoming_earnings and signal:
            try:
                earnings_date = datetime.strptime(upcoming_earnings, '%Y-%m-%d').date()
                days_to_earnings = (earnings_date - datetime.now(timezone.utc).date()).days
                if 0 <= days_to_earnings <= 3:
                    logger.info(f"Upcoming earnings for {symbol} in {days_to_earnings} days. Adjusting signal for earnings risk.")
                    if signal.action == "BUY":
                        original_size = signal.position_size if signal.position_size is not None else 1.0
                        signal.position_size = original_size * 0.5
                        logger.info(f"Reduced position size for {symbol} from {original_size:.2f} to {signal.position_size:.2f} due to upcoming earnings.")
                    if signal.stop_loss is not None:
                        original_sl = signal.stop_loss
                        signal.stop_loss = min(signal.stop_loss, 0.03)
                        if signal.stop_loss != original_sl:
                            logger.info(f"Tightened stop_loss for {symbol} from {original_sl:.3f} to {signal.stop_loss:.3f} due to upcoming earnings.")
                    if signal.stop_loss_atr_multiple is not None:
                        original_sl_atr = signal.stop_loss_atr_multiple
                        signal.stop_loss_atr_multiple = min(signal.stop_loss_atr_multiple, 1.0)
                        if signal.stop_loss_atr_multiple != original_sl_atr:
                            logger.info(f"Tightened stop_loss_atr_multiple for {symbol} from {original_sl_atr:.2f} to {signal.stop_loss_atr_multiple:.2f} due to upcoming earnings.")
            except (ValueError, TypeError):
                pass

        return {"signal": signal, "llm_provider": llm_provider, "llm_model": llm_model}

    async def process_symbol(self, symbol_entry: Dict[str, str], trading_paused: bool = False) -> None:
        """Fetch market data, get LLM strategy, validate, and execute."""
        engine = self.engine
        symbol = symbol_entry["symbol"]
        assigned_tf = symbol_entry["timeframe"]
        tf_seconds = engine._timeframe_to_seconds(assigned_tf)
        display_symbol = symbol  # Initialize to prevent NameError in except block

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
            await self.event_bus.request("process_post_llm_decision", decision_data)
        except asyncio.CancelledError:
            raise
        except (ConnectionError, TimeoutError, OSError) as e:
            logger.warning(f"Network/IO error processing {symbol}: {type(e).__name__}: {e}")
        except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, json.JSONDecodeError) as e:
            logger.error(f"Data/logic error processing {symbol}: {type(e).__name__}: {e}", exc_info=True)
            await self.engine._record_unexpected_exception("process_symbol", e)
            if engine.notifier:
                await engine.notifier.send_notification(
                    f"❌ Error processing {display_symbol}: {e}",
                    summary={"symbol": symbol, "action": "ERROR", "reason": str(e)[:200]}
                )
        except Exception as e:
            logger.error(f"Error processing {symbol}: {type(e).__name__}: {e}", exc_info=True)
            await self.engine._record_unexpected_exception("process_symbol", e)
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
            logger.warning(f"ATR percentile computation failed for {symbol}: {type(e).__name__}: {e}")

        return None

    async def fetch_symbol_market_data(self, symbol: str, assigned_tf: str) -> Optional[Dict[str, Any]]:
        """Fetch all raw market data for a symbol: ticker, fundamentals, balance, OHLCV, and multi-TF indicators.

        Returns a dict with all fetched data, or None if ticker is unavailable.
        """
        engine = self.engine
        base_symbol = symbol.split("/")[0]
        is_btp = BTPPolicy.is_btp(base_symbol)
        tf_seconds = engine._timeframe_to_seconds(assigned_tf)

        # --- Fetch ticker ---
        async with engine._exchange_semaphore:
            quotes = await engine._market_data_manager._get_quotes_async([base_symbol], timeout=45.0)
            ticker = quotes.get(base_symbol)
        if ticker is None:
            return None
        current_price = ticker['last']

        # --- Compute YTM for BTP bonds ---
        ytm = None
        if is_btp:
            from src.database import get_btp_details_from_db, compute_btp_ytm
            btp_details = await asyncio.to_thread(get_btp_details_from_db, [base_symbol])
            details = btp_details.get(base_symbol)
            if details:
                ytm = compute_btp_ytm(details.get("coupon"), details.get("maturity"), ticker.get("last"))

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

        # --- Fetch OHLCV from database (only for assigned timeframe) ---
        ohlcv_data = {}
        if settings.OHLCV_TIMEFRAMES:
            try:
                since_ms = int(time.time() * 1000) - settings.OHLCV_RETENTION_DAYS * 24 * 60 * 60 * 1000
                tf_seconds = engine._timeframe_to_seconds(assigned_tf)
                hist_limit = int((settings.OHLCV_RETENTION_DAYS * 86400) / tf_seconds) + 100
                db_candles = await asyncio.to_thread(get_ohlcv, symbol, assigned_tf, since_ms=since_ms, limit=hist_limit)
                if db_candles:
                    ohlcv_data[assigned_tf] = [[c["timestamp"], c["open"], c["high"], c["low"], c["close"], c["volume"]] for c in db_candles]
            except (ValueError, TypeError, KeyError, ConnectionError, TimeoutError, OSError) as e:
                logger.debug(f"DB OHLCV fetch failed for {symbol} {assigned_tf}: {type(e).__name__}: {e}")

            # Fetch last 2 daily candles for pivot points if assigned_tf is not '1d'
            if assigned_tf != "1d" and "1d" in settings.OHLCV_TIMEFRAMES:
                try:
                    daily_candles = await asyncio.to_thread(get_ohlcv, symbol, "1d", limit=2)
                    if daily_candles:
                        ohlcv_data["1d"] = [[c["timestamp"], c["open"], c["high"], c["low"], c["close"], c["volume"]] for c in daily_candles]
                except (ValueError, TypeError, KeyError, ConnectionError, TimeoutError, OSError) as e:
                    logger.debug(f"DB OHLCV fetch failed for {symbol} 1d: {type(e).__name__}: {e}")

        # --- Compute multi-TF indicators ---
        _inds = await self.market_data_fetcher.compute_multi_tf_indicators(symbol, ohlcv_data, assigned_tf)

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
            "ytm": ytm,
        }

    async def _fetch_dividend_data(self, symbol: str, ticker: Dict[str, Any]) -> Tuple[Optional[float], Optional[Any]]:
        """Fetch dividend yield and next ex-dividend date concurrently."""
        if BTPPolicy.is_btp(symbol.split("/")[0]):
            return None, None
        from src.database import get_dividend_yields_for_symbols, get_next_ex_dividend_date
        base = symbol.split("/")[0] if "/" in symbol else symbol
        prices = {base: ticker['last']} if ticker.get('last') else {}
        dividend_yield = None
        if prices:
            div_yields = await asyncio.to_thread(get_dividend_yields_for_symbols, [symbol], prices)
            if symbol in div_yields:
                dividend_yield = div_yields[symbol]
        next_ex_dividend = await asyncio.to_thread(get_next_ex_dividend_date, symbol)
        return dividend_yield, next_ex_dividend

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
        if symbol in self.shared_state.positions:
            pos = self.shared_state.positions[symbol]
            position_info = pos
            current_price = ticker['last']
            entry_price = pos['price']
            amount = pos['amount']
            unrealized_pnl = (current_price - entry_price) * amount

        # Recent trade outcomes (last 5 closed trades)
        recent_trades = [t for t in self.shared_state.trade_history if t.get("side") == "sell"][-5:]
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
            asset = await engine._market_data_manager.get_asset_info(symbol)
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
            t for t in self.shared_state.trade_history
            if t.get("symbol") == symbol and t.get("side") == "sell"
        ][-10:]

        # Run independent I/O-bound tasks concurrently
        (
            historical_backtest_results,
            dividend_data,
            aggregate_sentiment,
            global_risk_mult,
            full_breadth_raw,
            max_port_exp_raw,
            max_port_risk_raw,
            min_stop_atr_mult_raw,
            min_hold_time_mult_raw,
            global_min_rr_raw,
        ) = await asyncio.gather(
            asyncio.to_thread(get_backtest_results_for_symbol, symbol, assigned_tf, 10),
            self._fetch_dividend_data(symbol, ticker),
            engine._get_cached_sentiment(symbol) if settings.NEWS_ENABLED else None,
            engine._get_global_risk_multiplier(),
            asyncio.to_thread(engine.redis.get, "market:breadth:full"),
            engine.config_service.get_config("max_portfolio_exposure_pct"),
            engine.config_service.get_config("max_portfolio_stop_risk_pct"),
            engine.config_service.get_config("min_stop_loss_atr_mult"),
            engine.config_service.get_config("min_max_hold_time_mult"),
            engine.config_service.get_config("min_risk_reward_ratio"),
            return_exceptions=True,
        )

        # Process results with original error handling
        if isinstance(historical_backtest_results, Exception):
            historical_backtest_results = []
        dividend_yield, next_ex_dividend = dividend_data if not isinstance(dividend_data, Exception) else (None, None)
        if isinstance(aggregate_sentiment, Exception):
            logger.warning(f"Could not fetch aggregate sentiment for {symbol}: {type(aggregate_sentiment).__name__}: {aggregate_sentiment}")
            aggregate_sentiment = None
        if isinstance(global_risk_mult, Exception):
            global_risk_mult = None

        full_market_breadth = None
        if not isinstance(full_breadth_raw, Exception) and full_breadth_raw:
            try:
                full_market_breadth = json.loads(full_breadth_raw)
            except (ValueError, TypeError, json.JSONDecodeError):
                pass

        max_port_exp = float(max_port_exp_raw) if not isinstance(max_port_exp_raw, Exception) and max_port_exp_raw else None
        max_port_risk = float(max_port_risk_raw) if not isinstance(max_port_risk_raw, Exception) and max_port_risk_raw else None

        min_stop_atr_mult = float(min_stop_atr_mult_raw) if not isinstance(min_stop_atr_mult_raw, Exception) and min_stop_atr_mult_raw else 1.0
        min_hold_time_mult = float(min_hold_time_mult_raw) if not isinstance(min_hold_time_mult_raw, Exception) and min_hold_time_mult_raw else 1.0
        global_min_rr = float(global_min_rr_raw) if not isinstance(global_min_rr_raw, Exception) and global_min_rr_raw else None

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

        session_info = engine._market_data_manager._get_session_info()

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

        partial_tp_executed_levels = self.shared_state.positions[symbol].get("partial_tp_levels_triggered", []) if symbol in self.shared_state.positions else []

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
            "dividend_yield": dividend_yield,
            "next_ex_dividend": next_ex_dividend,
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
        # Exclude high-cardinality fields from the cache hash to prevent
        # frequent cache misses caused by volatile data like raw candles.
        _high_cardinality_keys = {
            "ohlcv_data", "raw_candles", "recent_trades", "historical_ohlcv",
            "past_trades", "multi_tf_raw_candles"
        }
        hash_snapshot = {k: v for k, v in market_snapshot.items() if k not in _high_cardinality_keys}
        market_hash = compute_market_hash(hash_snapshot)

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
                async with self.shared_state._eval_state_lock:
                    self.shared_state._force_eval.pop(symbol, None)
                return None

        # --- Cooldown after a losing trade (LLM-defined) ---
        if symbol not in self.shared_state.positions:
            last_loss = self.shared_state.last_loss_time.get(symbol)
            if last_loss is not None:
                cooldown = self.shared_state.cooldown_durations.get(symbol, 0)
                if cooldown > 0:
                    elapsed = time.time() - last_loss
                    if elapsed < cooldown:
                        remaining = cooldown - elapsed
                        logger.info(
                            f"Skipping {symbol}: cooldown active ({remaining:.0f}s remaining after loss)"
                        )
                        async with self.shared_state._eval_state_lock:
                            self.shared_state._force_eval.pop(symbol, None)
                        return None

        # Skip if there is already a queued order for this symbol
        async with self.shared_state._queued_orders_lock:
            has_queued = any(q['symbol'] == symbol for q in self.shared_state.queued_orders)
        if has_queued:
            logger.info(f"Skipping {symbol}: order already queued.")
            async with self.shared_state._eval_state_lock:
                self.shared_state._force_eval.pop(symbol, None)
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
        if symbol in self.shared_state.positions:
            pos = self.shared_state.positions[symbol]
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
            async with self.shared_state._eval_state_lock:
                self.shared_state._force_eval.pop(symbol, None)
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

        async with self.shared_state._eval_state_lock:
            self.shared_state._force_eval.pop(symbol, None)
        return True

    async def _get_skip_eval_config(self) -> Dict[str, float]:
        """Fetch LLM-driven skip thresholds from Redis, with a 5-minute cache."""
        engine = self.engine
        now = time.time()
        if now - self._skip_config_cache_time < self._skip_config_cache_ttl:
            return self._skip_config_cache

        cache = {
            "skip_price_mult": 1.0,
            "skip_rsi": 5.0,
            "skip_macd": 0.0005,
            "rsi_oversold": 30.0,
            "rsi_overbought": 70.0,
        }
        try:
            raw = await engine.config_service.get_config("skip_eval_price_change_atr_mult")
            if raw:
                cache["skip_price_mult"] = float(raw)
            raw = await engine.config_service.get_config("skip_eval_rsi_change")
            if raw:
                cache["skip_rsi"] = float(raw)
            raw = await engine.config_service.get_config("skip_eval_macd_hist_change")
            if raw:
                cache["skip_macd"] = float(raw)
            raw = await engine.config_service.get_config("skip_eval_rsi_oversold")
            if raw:
                cache["rsi_oversold"] = float(raw)
            raw = await engine.config_service.get_config("skip_eval_rsi_overbought")
            if raw:
                cache["rsi_overbought"] = float(raw)
        except (ValueError, TypeError, ConnectionError, TimeoutError, OSError):
            pass

        # Sanity-check: if thresholds are degenerate, fall back to defaults
        if cache["rsi_oversold"] <= 0 or cache["rsi_oversold"] >= 50:
            cache["rsi_oversold"] = 30.0
        if cache["rsi_overbought"] >= 100 or cache["rsi_overbought"] <= 50:
            cache["rsi_overbought"] = 70.0

        self._skip_config_cache = cache
        self._skip_config_cache_time = now
        return cache

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
        async with self.shared_state._eval_state_lock:
            force_eval = self.shared_state._force_eval.get(symbol, False)
        if force_eval:
            return False
        # Never skip critical situations (max hold, stop-loss, take-profit triggered)
        if is_critical:
            return False

        # ATR is used for price-change comparison but is not strictly required.
        # When ATR is None (common for long timeframes like 1Y/3Y/5Y), we fall
        # back to a fixed percentage threshold so the skip logic still works
        # and we don't waste LLM calls every cycle.

        snapshot = self.shared_state._last_eval_snapshot.get(symbol)
        if snapshot is None:
            # First evaluation – must call
            return False

        # If the timeframe has changed, the old snapshot is stale and
        # cannot be compared against the new timeframe's indicators.
        if snapshot.get("timeframe_seconds") != timeframe_seconds:
            return False

        now = time.time()
        last_time = snapshot.get("timestamp", 0)
        last_price = snapshot.get("price", 0)

        # Always call if enough time has passed (3× the effective interval)
        # For medium/long-term, be more patient before forcing an evaluation
        effective_interval = timeframe_seconds * settings.STRATEGY_INTERVAL_MULTIPLIER
        # Cap the safety net at a value proportional to the timeframe,
        # but never greater than the configured MAX_SKIP_INTERVAL_SECONDS.
        # This prevents excessively long skip durations for long timeframes
        # (e.g., 5Y candles should not skip evaluation for 5 years).
        max_skip = min(settings.MAX_SKIP_INTERVAL_SECONDS, effective_interval)
        if now - last_time > min(3 * effective_interval, max_skip):
            return False

        # Fetch LLM-driven skip thresholds from Redis (cached for 5 minutes).
        cfg = await self._get_skip_eval_config()
        skip_price_mult = cfg["skip_price_mult"]
        skip_rsi = cfg["skip_rsi"]
        skip_macd = cfg["skip_macd"]

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
            rsi_oversold = cfg["rsi_oversold"]
            rsi_overbought = cfg["rsi_overbought"]
            if rsi is not None and (rsi < rsi_oversold or rsi > rsi_overbought):
                return False
            # MACD histogram direction change? (harder to detect without previous sign – skip for simplicity)
            # Ensure we still evaluate at least once per strategy interval
            # even if no significant changes are detected
            if now - last_time >= effective_interval:
                logger.info(f"Forcing LLM eval for {symbol}: interval elapsed ({now - last_time:.0f}s >= {effective_interval:.0f}s)")
                return False
            # Otherwise, no strong signal → skip
            logger.info(f"Skipping LLM eval for {symbol}: no significant market changes detected (rsi={rsi})")
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
            logger.warning(f"Volume trend computation failed for {symbol}: {type(e).__name__}: {e}")
            return None
