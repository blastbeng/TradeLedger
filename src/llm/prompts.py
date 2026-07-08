import json
import logging
import re
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from src.config.settings import settings
from src.utils.symbol_utils import is_btp_isin
from src.database import get_news_for_symbol, get_aggregate_sentiment_from_db
from src.llm.stock_selection_prompts import build_stock_selection_prompt, build_final_selection_prompt
from src.exchanges.market_data import TIMEFRAME_MAP
from src.llm.prompt_utils import (
    _timeframe_to_seconds,
    compact_prompt,
    _summarize_ohlcv,
    _format_raw_candles_compact,
    _format_trade_pattern_analysis,
    _format_news_for_prompt,
    get_cached_news_summary,
)
logger = logging.getLogger(__name__)
from src.llm.system_prompt import build_system_prompt, SYSTEM_PROMPT_TEMPLATE
from src.llm.backtest_prompts import BacktestPromptData, build_backtest_variants_prompt, build_final_decision_prompt

@dataclass
class StrategyPromptData:
    symbol: str
    ticker: Dict[str, Any]
    balance: Dict[str, float]
    open_positions: List[Dict[str, Any]]
    per_symbol_budget: float
    max_symbols: int
    base_currency: str
    performance: Optional[Dict[str, Any]] = None
    ohlcv_data: Optional[Dict[str, List]] = None
    assigned_timeframe: Optional[str] = None
    atr: Optional[float] = None
    atr_multi_tf: Optional[Dict[str, float]] = None
    rsi: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_hist: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None
    ema_9: Optional[float] = None
    ema_21: Optional[float] = None
    stochastic_k: Optional[float] = None
    stochastic_d: Optional[float] = None
    adx: Optional[float] = None
    plus_di: Optional[float] = None
    minus_di: Optional[float] = None
    obv: Optional[float] = None
    mfi: Optional[float] = None
    cci: Optional[float] = None
    williams_r: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    position_info: Optional[Dict[str, Any]] = None
    drawdown_pct: Optional[float] = None
    raw_candles: Optional[List[List]] = None
    recent_trades: Optional[List[Dict[str, Any]]] = None
    historical_ohlcv: Optional[List[List]] = None
    min_order_amount: Optional[float] = None
    min_order_cost: Optional[float] = None
    all_symbols: Optional[List[Dict[str, str]]] = None
    past_trades: Optional[List[Dict[str, Any]]] = None
    cycle_spent: Optional[float] = None
    remaining_balance: Optional[float] = None
    market_regime: Optional[str] = None
    multi_tf_raw_candles: Optional[Dict[str, List[List]]] = None
    multi_tf_indicators: Optional[Dict[str, Dict[str, Any]]] = None
    session_info: Optional[Dict[str, Any]] = None
    sentiment_trend: Optional[float] = None
    volume_trend: Optional[float] = None
    ichimoku: Optional[Dict[str, Optional[float]]] = None
    market_breadth: Optional[Dict[str, Any]] = None
    full_market_breadth: Optional[Dict[str, Any]] = None
    keltner_channels: Optional[Dict[str, float]] = None
    donchian_channels: Optional[Dict[str, float]] = None
    parabolic_sar: Optional[float] = None
    atr_percentile: Optional[float] = None
    global_risk_multiplier: Optional[float] = None
    trading_paused: bool = False
    max_hold_expired: bool = False
    max_hold_expired_count: int = 0
    stop_loss_triggered: bool = False
    stop_loss_review_count: int = 0
    take_profit_triggered: bool = False
    take_profit_review_count: int = 0
    partial_tp_triggered: bool = False
    partial_tp_review_count: int = 0
    partial_tp_triggered_levels: Optional[List[int]] = None
    partial_tp_executed_levels: Optional[List[int]] = None
    dust_sweep_triggered: bool = False
    dust_sweep_review_count: int = 0
    max_stop_loss_reviews: int = 10
    max_take_profit_reviews: int = 10
    max_partial_tp_reviews: int = 10
    max_dust_sweep_reviews: int = 10
    portfolio_exposure_pct: Optional[float] = None
    portfolio_stop_risk_pct: Optional[float] = None
    portfolio_total_value: Optional[float] = None
    portfolio_open_count: int = 0
    portfolio_available_capital: Optional[float] = None
    last_decision: Optional[Dict[str, Any]] = None
    minutes_to_market_close: Optional[int] = None
    current_strategy_interval_seconds: Optional[int] = None
    max_portfolio_exposure_pct: Optional[float] = None
    max_portfolio_stop_risk_pct: Optional[float] = None
    trade_pattern_analysis: Optional[Dict[str, Any]] = None
    symbol_event: Optional[Dict[str, Any]] = None
    queued_orders: Optional[List[Dict[str, Any]]] = None
    fundamentals: Optional[Dict[str, Any]] = None
    aggregate_sentiment: Optional[Dict[str, Any]] = None
    vwap: Optional[float] = None
    daily_pivot_points: Optional[Dict[str, float]] = None
    min_hold_time_mult: float = 1.0
    min_stop_atr_mult: float = 1.0
    min_viable_trade_amount: float = 0.0
    historical_backtest_results: Optional[List[Dict[str, Any]]] = None


def build_strategy_prompt(
    data: StrategyPromptData,
) -> str:
    """Build a prompt to generate a trading strategy for a specific stock/ETF."""
    symbol = data.symbol
    ticker = data.ticker
    balance = data.balance
    open_positions = data.open_positions
    per_symbol_budget = data.per_symbol_budget
    max_symbols = data.max_symbols
    base_currency = data.base_currency
    performance = data.performance
    ohlcv_data = data.ohlcv_data
    assigned_timeframe = data.assigned_timeframe
    atr = data.atr
    atr_multi_tf = data.atr_multi_tf
    rsi = data.rsi
    macd = data.macd
    macd_signal = data.macd_signal
    macd_hist = data.macd_hist
    bb_upper = data.bb_upper
    bb_middle = data.bb_middle
    bb_lower = data.bb_lower
    ema_9 = data.ema_9
    ema_21 = data.ema_21
    stochastic_k = data.stochastic_k
    stochastic_d = data.stochastic_d
    adx = data.adx
    plus_di = data.plus_di
    minus_di = data.minus_di
    obv = data.obv
    mfi = data.mfi
    cci = data.cci
    williams_r = data.williams_r
    unrealized_pnl = data.unrealized_pnl
    position_info = data.position_info
    drawdown_pct = data.drawdown_pct
    raw_candles = data.raw_candles
    recent_trades = data.recent_trades
    historical_ohlcv = data.historical_ohlcv
    min_order_amount = data.min_order_amount
    min_order_cost = data.min_order_cost
    all_symbols = data.all_symbols
    past_trades = data.past_trades
    cycle_spent = data.cycle_spent
    remaining_balance = data.remaining_balance
    market_regime = data.market_regime
    multi_tf_raw_candles = data.multi_tf_raw_candles
    multi_tf_indicators = data.multi_tf_indicators
    session_info = data.session_info
    sentiment_trend = data.sentiment_trend
    volume_trend = data.volume_trend
    ichimoku = data.ichimoku
    market_breadth = data.market_breadth
    full_market_breadth = data.full_market_breadth
    keltner_channels = data.keltner_channels
    donchian_channels = data.donchian_channels
    parabolic_sar = data.parabolic_sar
    atr_percentile = data.atr_percentile
    global_risk_multiplier = data.global_risk_multiplier
    trading_paused = data.trading_paused
    max_hold_expired = data.max_hold_expired
    max_hold_expired_count = data.max_hold_expired_count
    stop_loss_triggered = data.stop_loss_triggered
    stop_loss_review_count = data.stop_loss_review_count
    take_profit_triggered = data.take_profit_triggered
    take_profit_review_count = data.take_profit_review_count
    partial_tp_triggered = data.partial_tp_triggered
    partial_tp_review_count = data.partial_tp_review_count
    partial_tp_triggered_levels = data.partial_tp_triggered_levels
    partial_tp_executed_levels = data.partial_tp_executed_levels
    dust_sweep_triggered = data.dust_sweep_triggered
    dust_sweep_review_count = data.dust_sweep_review_count
    max_stop_loss_reviews = data.max_stop_loss_reviews
    max_take_profit_reviews = data.max_take_profit_reviews
    max_partial_tp_reviews = data.max_partial_tp_reviews
    max_dust_sweep_reviews = data.max_dust_sweep_reviews
    portfolio_exposure_pct = data.portfolio_exposure_pct
    portfolio_stop_risk_pct = data.portfolio_stop_risk_pct
    portfolio_total_value = data.portfolio_total_value
    portfolio_open_count = data.portfolio_open_count
    portfolio_available_capital = data.portfolio_available_capital
    last_decision = data.last_decision
    minutes_to_market_close = data.minutes_to_market_close
    current_strategy_interval_seconds = data.current_strategy_interval_seconds
    max_portfolio_exposure_pct = data.max_portfolio_exposure_pct
    max_portfolio_stop_risk_pct = data.max_portfolio_stop_risk_pct
    trade_pattern_analysis = data.trade_pattern_analysis
    symbol_event = data.symbol_event
    queued_orders = data.queued_orders
    fundamentals = data.fundamentals
    vwap = data.vwap
    daily_pivot_points = data.daily_pivot_points
    min_hold_time_mult = data.min_hold_time_mult
    min_stop_atr_mult = data.min_stop_atr_mult
    min_viable_trade_amount = data.min_viable_trade_amount
    historical_backtest_results = data.historical_backtest_results
    # Trim large lists to prevent context window overflow
    if recent_trades and len(recent_trades) > 20:
        recent_trades = recent_trades[-20:]
    if past_trades and len(past_trades) > 20:
        past_trades = past_trades[-20:]
    if historical_backtest_results and len(historical_backtest_results) > 5:
        historical_backtest_results = historical_backtest_results[-5:]
    if multi_tf_raw_candles:
        multi_tf_raw_candles = {tf: candles[-200:] for tf, candles in multi_tf_raw_candles.items()}
    if raw_candles and len(raw_candles) > 500:
        raw_candles = raw_candles[-500:]
    if historical_ohlcv and len(historical_ohlcv) > 1000:
        historical_ohlcv = historical_ohlcv[-1000:]
    current_price = ticker.get("last") if ticker else None
    if assigned_timeframe and assigned_timeframe not in TIMEFRAME_MAP:
        logger.warning(f"Assigned timeframe {assigned_timeframe} is not supported by yfinance. Falling back to default.")
        assigned_timeframe = "1d" if "1d" in TIMEFRAME_MAP else list(TIMEFRAME_MAP.keys())[0]
    tf_seconds = _timeframe_to_seconds(assigned_timeframe) if assigned_timeframe else 86400
    _ticker_compact = {
        k: ticker.get(k) for k in ("last", "bid", "ask", "volume", "quoteVolume", "name", "coupon", "maturity")
        if k in ticker
    }
    # Rename "percentage" to "change_24h" so the LLM understands what it represents
    _pct = ticker.get("percentage")
    if _pct is not None:
        _ticker_compact["change_24h"] = _pct
    prompt = f"""Symbol: {symbol}
Current ticker: {json.dumps(_ticker_compact)}
Current balances: {json.dumps(balance)}
"""
    # Explicitly highlight the 24h change so the LLM uses it in its analysis
    _change_24h = ticker.get("percentage")
    if _change_24h is not None:
        prompt += f"24h price change: {_change_24h:+.2f}%\n"
    else:
        prompt += "24h price change: N/A (no daily candle data available)\n"
    # --- Portfolio context: total base balance and all tracked symbols ---
    base_balance = balance.get(base_currency, 0.0)
    prompt += f"\nTotal {base_currency} balance available: {base_balance:.2f}\n"
    if all_symbols:
        other_symbols = [s for s in all_symbols if s["symbol"] != symbol]
        if other_symbols:
            symbol_list_str = ", ".join(f"{s['symbol']}({s['timeframe']})" for s in other_symbols)
            prompt += f"Other symbols being traded: {symbol_list_str}\n"
            prompt += (
                f"**CRITICAL:** Concentrate your trading decision entirely on THIS single selected ticker ({symbol}) "
                "where a signal has been detected. You may allocate a larger fraction of the available balance to this "
                "ticker if you have high conviction, but leave some budget for other promising setups if possible.\n"
            )
        else:
            prompt += "This is the only symbol being traded; you may use the full available balance.\n"
    _positions_compact = [
        {
            "symbol": p.get("symbol"),
            "entry": p.get("price"),
            "amount": p.get("amount"),
            "sl": p.get("stop_loss"),
            "tp": p.get("take_profit"),
        }
        for p in open_positions
    ]
    prompt += f"""Open positions: {json.dumps(_positions_compact)}
Total available {base_currency} balance: {base_balance:.2f}
Suggested equal share per symbol: {per_symbol_budget:.2f} {base_currency}
Maximum symbols to trade: {max_symbols}

**Focus:** Concentrate your analysis and trading decision entirely on this single ticker ({symbol}). You may use up to the total available balance for this trade if your conviction is high, provided it does not exceed the remaining balance.
"""
    # --- Portfolio exposure summary ---
    if portfolio_total_value is not None:
        prompt += f"\n**Portfolio Exposure Summary:**\n"
        prompt += f"  Total portfolio value: {portfolio_total_value:.2f} {base_currency}\n"
        prompt += f"  Open positions: {portfolio_open_count}\n"
        if portfolio_exposure_pct is not None:
            prompt += f"  Capital deployed: {portfolio_exposure_pct:.1f}%\n"
        if portfolio_stop_risk_pct is not None:
            prompt += f"  Total stop-loss risk: {portfolio_stop_risk_pct:.2f}% (loss if ALL stops hit)\n"
        if portfolio_available_capital is not None:
            prompt += f"  Available capital: {portfolio_available_capital:.2f} {base_currency}\n"
        if max_portfolio_exposure_pct is not None and max_portfolio_stop_risk_pct is not None:
            prompt += (
                f"Use this to decide `position_size_fraction`. If deployment is high (>{max_portfolio_exposure_pct*100:.0f}%) "
                f"or stop-loss risk is elevated (>{max_portfolio_stop_risk_pct*100:.0f}%), reduce size or HOLD. "
                "If low exposure/risk, allocate more to high-conviction trades.\n"
            )
        else:
            prompt += (
                "Use this to decide `position_size_fraction`. If deployment is high or stop-loss risk is elevated, "
                "reduce size or HOLD. If low exposure/risk, allocate more to high-conviction trades.\n"
            )
    if cycle_spent is not None and remaining_balance is not None:
        prompt += (
            f"Amount allocated to other symbols this cycle: {cycle_spent:.2f} {base_currency}\n"
            f"Remaining available for this symbol: {remaining_balance:.2f} {base_currency}\n"
            "Your `position_size_fraction` must not exceed the remaining balance. If low, reduce fraction or HOLD.\n"
        )
        max_possible_amount = remaining_balance
        prompt += (
            f"Max amount allocatable to this trade: {max_possible_amount:.2f} {base_currency} "
            f"(the full remaining balance). You are NOT limited to the per-symbol budget. If setting `min_profit_per_trade`, "
            f"ensure it is ≤ `max_possible_amount * take_profit_pct`.\n"
        )
    if global_risk_multiplier is not None and global_risk_multiplier < 1.0:
        prompt += (
            f"\n**Global risk multiplier: {global_risk_multiplier}.** "
            "Actual amount used = `position_size_fraction × total_balance × global_risk_multiplier`. "
            "Adjust `position_size_fraction` to compensate if you want a specific exposure.\n"
        )
    # --- Queued orders for this symbol ---
    if queued_orders:
        symbol_queued = [q for q in queued_orders if q.get('symbol') == symbol]
        if symbol_queued:
            prompt += "\n**Queued orders for this symbol (already waiting to fill):**\n"
            now = time.time()
            current_price = ticker.get('last') if ticker else None
            for q in symbol_queued:
                side = q.get('side', '?').upper()
                limit_price = q.get('limit_price')
                queued_at = q.get('queued_at')
                age_str = ""
                if queued_at is not None:
                    age_sec = now - queued_at
                    if age_sec < 60: age_str = f" (placed {age_sec:.0f}s ago)"
                    elif age_sec < 3600: age_str = f" (placed {age_sec/60:.1f}m ago)"
                    else: age_str = f" (placed {age_sec/3600:.1f}h ago)"
                dist_str = ""
                if current_price is not None and limit_price is not None and current_price > 0:
                    if side == 'BUY':
                        dist_pct = ((limit_price - current_price) / current_price) * 100
                        dist_str = f" (limit is {dist_pct:+.2f}% from current {current_price:.4f})"
                    else:
                        dist_pct = ((current_price - limit_price) / current_price) * 100
                        dist_str = f" (limit is {-dist_pct:+.2f}% from current {current_price:.4f})"
                filled_qty = q.get('filled_qty', 0.0)
                original_amount = q.get('original_amount')
                partial_str = ""
                if filled_qty > 0 and original_amount is not None and original_amount > 0:
                    remaining_base = original_amount - filled_qty
                    partial_str = f" (partial fill: {filled_qty:.6f} filled, {remaining_base:.6f} remaining)"

                prompt += f"  - {side} limit @ {limit_price}{age_str}{dist_str}{partial_str}\n"

            prompt += (
                "**Do NOT output a new BUY or SELL signal while a queued order exists.** The engine ignores new signals "
                "until the queued order fills or is cancelled. Partial fills are handled automatically. To change the "
                "order, output HOLD and explain in reasoning.\n"
            )
    base_symbol = symbol
    quote_currency = base_currency
    if min_order_amount is not None or min_order_cost is not None:
        prompt += f"\nMin order size for {symbol}:"
        if min_order_amount is not None:
            prompt += f" {min_order_amount} {base_symbol}"
        if min_order_cost is not None:
            prompt += f" (or {min_order_cost} {quote_currency} cost)"
        prompt += (
            ". Your `position_size_fraction` must meet both minimums. Use current price to convert.\n"
        )
    if assigned_timeframe:
        prompt += f"\nAssigned timeframe: {assigned_timeframe}. Base your decision PRIMARILY on this timeframe's OHLCV data.\n"
    if market_regime:
        prompt += f"Market regime: {market_regime}\n"

    if session_info:
        prompt += f"Current UTC hour: {session_info['utc_hour']} ({session_info['session']} session)\n"
    if minutes_to_market_close is not None:
        if minutes_to_market_close > 0:
            prompt += f"  Minutes to market close (5:30 PM Rome): {minutes_to_market_close}\n"
        else:
            prompt += "  Market is currently closed.\n"
    if current_strategy_interval_seconds is not None:
        prompt += f"  Strategy eval interval: {current_strategy_interval_seconds}s\n"

    # --- Volatility, order book imbalance, and position P&L context ---
    if atr is not None:
        prompt += f"ATR (14-period, {assigned_timeframe or 'default'}): {atr:.6f}\n"
    if atr is not None and current_price is not None and current_price > 0:
        atr_pct = atr / current_price
        min_sl = min_stop_atr_mult * atr_pct
        prompt += (
            f"\n**Current ATR%: {atr_pct:.4%}**. Validator enforces min fixed stop-loss of "
            f"{min_stop_atr_mult} × ATR% = {min_sl:.4%}. Your `stop_loss_pct` must be ≥ this value.\n"
        )
    if atr_percentile is not None:
        prompt += f"ATR percentile (last 100 obs): {atr_percentile:.1f}%\n"
    if atr_multi_tf:
        prompt += f"ATR across timeframes: {json.dumps(atr_multi_tf)}\n"
    # --- Transaction cost break-even calculation ---
    _is_btp = is_btp_isin(symbol)
    # Use the full remaining balance (or total balance) for fee break-even calculation
    # so the LLM understands it can use more than the per-symbol budget if needed.
    trade_value = remaining_balance if remaining_balance is not None and remaining_balance > 0 else base_balance
    if trade_value > 0:
        if _is_btp:
            if settings.BTP_IS_PRIMARY_ISSUANCE:
                buy_fee = sell_fee = 0.0
            else:
                buy_fee = max(settings.BTP_MIN_FEE, trade_value * settings.BTP_FEE_PERC)
                sell_fee = max(settings.BTP_MIN_FEE, trade_value * settings.BTP_FEE_PERC)
            total_fees = buy_fee + sell_fee
            break_even_pct = total_fees / trade_value
            prompt += (
                f"\n**Transaction Cost Break-Even (BTP) — Example using full available balance:**\n"
                f"  If you use the full available balance (~{trade_value:.2f} {quote_currency}) for this trade:\n"
                f"  Total round-trip fees: {total_fees:.2f} {quote_currency} ({break_even_pct*100:.2f}%)\n"
                f"  `take_profit_pct` MUST be > {break_even_pct*100:.2f}% to be profitable.\n"
                f"  **You are NOT limited to the per-symbol budget.** You may use up to the full remaining balance for this single trade if your conviction is high.\n"
            )
        else:
            buy_fee = max(settings.STOCK_FEE_MIN, trade_value * settings.STOCK_FEE_PERC) + settings.STOCK_FEE_FIXED + (trade_value * settings.TOBIN_TAX_RATE)
            sell_fee = max(settings.STOCK_FEE_MIN, trade_value * settings.STOCK_FEE_PERC) + settings.STOCK_FEE_FIXED
            total_fees = buy_fee + sell_fee
            break_even_pct = total_fees / trade_value
            prompt += (
                f"\n**Transaction Cost Break-Even — Example using full available balance:**\n"
                f"  If you use the full available balance (~{trade_value:.2f} {quote_currency}) for this trade:\n"
                f"  Total round-trip fees: {total_fees:.2f} {quote_currency} ({break_even_pct*100:.2f}%)\n"
                f"  `take_profit_pct` MUST be > {break_even_pct*100:.2f}% to be profitable.\n"
                f"  **You are NOT limited to the per-symbol budget.** You may use up to the full remaining balance for this single trade if your conviction is high.\n"
            )
    # --- Show the LLM its previous decision for this symbol ---
    if last_decision:
        age_seconds = time.time() - last_decision.get("timestamp", 0)
        prompt += (
            f"\n**Previous decision for {symbol} ({age_seconds:.0f}s ago):**\n"
            f"  Action: {last_decision.get('action')}, Confidence: {last_decision.get('confidence', 0):.2f}\n"
            f"  Reasoning: {last_decision.get('reasoning', '')}\n"
        )
        sl_pct = last_decision.get("stop_loss_pct")
        tp_pct = last_decision.get("take_profit_pct")
        psf = last_decision.get("position_size_fraction")
        sl_method = last_decision.get("stop_loss_method")
        if sl_method: prompt += f"  SL method: {sl_method}\n"
        if sl_pct is not None: prompt += f"  SL pct: {sl_pct}\n"
        if tp_pct is not None: prompt += f"  TP pct: {tp_pct}\n"
        if psf is not None: prompt += f"  Size fraction: {psf}\n"
    if unrealized_pnl is not None and position_info:
        prompt += f"Current position unrealized P&L: {unrealized_pnl:.2f} {base_currency}\n"
        entry_price = position_info.get('price', 0)
        amount = position_info.get('amount', 0)
        prompt += f"Position: entry {entry_price}, amount {amount}\n"
        prompt += f"\n**You hold {amount:.6f} {base_symbol} @ {entry_price:.4f}.**\n"
        prompt += "BUY = ADD to position (scale in). SELL = close ENTIRE position.\n"
        if entry_price > 0 and amount > 0:
            cost_basis = entry_price * amount
            if cost_basis > 0:
                pnl_pct = (unrealized_pnl / cost_basis) * 100
                prompt += f"Unrealized P&L: {pnl_pct:+.2f}%\n"
        current_sl = position_info.get('stop_loss')
        current_tp = position_info.get('take_profit')
        if current_sl is not None: prompt += f"Current SL price: {current_sl:.6f}\n"
        if current_tp is not None: prompt += f"Current TP price: {current_tp:.6f}\n"
        if current_price and current_price > 0:
            if current_sl is not None:
                sl_distance_pct = ((current_price - current_sl) / current_price) * 100
                prompt += f"Distance to SL: {sl_distance_pct:.2f}% below current\n"
            if current_tp is not None:
                tp_distance_pct = ((current_tp - current_price) / current_price) * 100
                prompt += f"Distance to TP: {tp_distance_pct:.2f}% above current\n"
        trailing_active = position_info.get('trailing_stop', False)
        if trailing_active:
            trailing_dist = position_info.get('trailing_stop_distance_pct')
            trailing_act = position_info.get('trailing_stop_activation_pct')
            prompt += f"Trailing stop: enabled (dist={trailing_dist}, act={trailing_act})\n"
        max_hold = position_info.get('max_hold_time_seconds')
        if max_hold is not None and max_hold > 0:
            entry_ts = position_info.get('timestamp', 0) / 1000.0
            elapsed = time.time() - entry_ts if entry_ts > 0 else 0
            remaining = max(0, max_hold - elapsed)
            prompt += f"Max hold: {max_hold:.0f}s total, {remaining:.0f}s remaining\n"

    # --- Multi-timeframe OHLCV summary and indicators ---
    if multi_tf_raw_candles:
        tf_summaries = []
        for tf in settings.OHLCV_TIMEFRAMES:
            if tf in multi_tf_raw_candles:
                summary = _summarize_ohlcv(multi_tf_raw_candles[tf])
                if summary:
                    tf_summaries.append(
                        f"  [{tf}] chg={summary['change_pct']}%, H={summary['high']}, L={summary['low']}, "
                        f"vol={summary['volume']}, candles={summary['candle_count']}"
                    )
        if tf_summaries:
            prompt += "\nMulti-timeframe OHLCV summary:\n" + "\n".join(tf_summaries) + "\n"
            prompt += "Use these to assess momentum/trend across timeframes. Align your decision with the long-term trend.\n"
    if multi_tf_indicators:
        ind_lines = []
        for tf in settings.OHLCV_TIMEFRAMES:
            if tf in multi_tf_indicators:
                ind = multi_tf_indicators[tf]
                ind_compact = {}
                if ind.get('rsi') is not None: ind_compact['rsi'] = round(ind['rsi'], 2)
                if ind.get('macd') is not None:
                    ind_compact['macd'] = round(ind['macd'], 4)
                    ind_compact['macd_sig'] = round(ind['macd_signal'], 4)
                    ind_compact['macd_h'] = round(ind['macd_hist'], 4)
                if ind.get('bb_upper') is not None:
                    ind_compact['bb_u'] = round(ind['bb_upper'], 4)
                    ind_compact['bb_m'] = round(ind['bb_middle'], 4)
                    ind_compact['bb_l'] = round(ind['bb_lower'], 4)
                if ind.get('ema_9') is not None:
                    ind_compact['ema9'] = round(ind['ema_9'], 4)
                    if ind.get('ema_21') is not None: ind_compact['ema21'] = round(ind['ema_21'], 4)
                if ind.get('stochastic_k') is not None:
                    ind_compact['stoch_k'] = round(ind['stochastic_k'], 2)
                    if ind.get('stochastic_d') is not None: ind_compact['stoch_d'] = round(ind['stochastic_d'], 2)
                if ind.get('adx') is not None:
                    ind_compact['adx'] = round(ind['adx'], 2)
                    if ind.get('plus_di') is not None: ind_compact['+di'] = round(ind['plus_di'], 2)
                    if ind.get('minus_di') is not None: ind_compact['-di'] = round(ind['minus_di'], 2)
                if ind.get('obv') is not None: ind_compact['obv'] = round(ind['obv'], 2)
                if ind.get('mfi') is not None: ind_compact['mfi'] = round(ind['mfi'], 2)
                if ind.get('cci') is not None: ind_compact['cci'] = round(ind['cci'], 2)
                if ind.get('williams_r') is not None: ind_compact['wr'] = round(ind['williams_r'], 2)
                if ind.get('ichimoku') is not None:
                    ich = ind['ichimoku']
                    ind_compact['ich'] = {"t": round(ich['tenkan_sen'], 4), "k": round(ich['kijun_sen'], 4),
                                          "sa": round(ich['senkou_span_a'], 4), "sb": round(ich['senkou_span_b'], 4),
                                          "cb": round(ich['cloud_bottom'], 4), "ct": round(ich['cloud_top'], 4)}
                if ind.get('donchian_channels') is not None:
                    dc = ind['donchian_channels']
                    ind_compact['dc'] = {"u": round(dc['upper'], 4), "m": round(dc['middle'], 4), "l": round(dc['lower'], 4)}
                if ind.get('atr') is not None: ind_compact['atr'] = round(ind['atr'], 6)
                if ind.get('parabolic_sar') is not None: ind_compact['sar'] = round(ind['parabolic_sar'], 6)
                if ind.get('keltner_channels') is not None:
                    kc = ind['keltner_channels']
                    ind_compact['kc'] = {"u": round(kc['upper'], 6), "m": round(kc['middle'], 6), "l": round(kc['lower'], 6)}
                

                if not ind_compact: continue
                ind_lines.append(f"[{tf}] {json.dumps(ind_compact, separators=(',', ':'))}")
        if ind_lines:
            prompt += "\nComputed indicators per timeframe:\n" + "\n".join(ind_lines) + "\n"
    elif raw_candles:
        summary = _summarize_ohlcv(raw_candles)
        if summary:
            prompt += (
                f"\nOHLCV summary ({assigned_timeframe}): chg={summary['change_pct']}%, H={summary['high']}, "
                f"L={summary['low']}, vol={summary['volume']}, candles={summary['candle_count']}\n"
            )
            has_indicators = any(v is not None for v in [rsi, macd, bb_upper, ema_9])
            if has_indicators:
                prompt += "Indicators (RSI, MACD, BB, EMA) are pre-computed. Use them to time entries/exits and explain in reasoning.\n"
    if historical_ohlcv:
        hist_summary = _summarize_ohlcv(historical_ohlcv)
        if hist_summary:
            closes = [c[4] for c in historical_ohlcv]
            volumes = [c[5] for c in historical_ohlcv]
            last_20 = closes[-20:] if len(closes) >= 20 else closes
            avg_close = sum(last_20) / len(last_20) if last_20 else 0
            max_close = max(last_20) if last_20 else 0
            min_close = min(last_20) if last_20 else 0
            avg_volume = sum(volumes[-20:]) / min(len(volumes), 20) if volumes else 0
            last_5_closes = closes[-5:] if len(closes) >= 5 else closes
            recent_momentum_pct = ((last_5_closes[-1] - last_5_closes[0]) / last_5_closes[0]) * 100 if len(last_5_closes) >= 2 and last_5_closes[0] > 0 else 0.0

            prompt += (
                f"\nHistorical OHLCV summary ({hist_summary['candle_count']} candles, {assigned_timeframe or 'default'}):\n"
                f"  Overall chg: {hist_summary['change_pct']:.2f}%, High: {hist_summary['high']:.4f}, Low: {hist_summary['low']:.4f}\n"
                f"  Last 20 — avg close: {avg_close:.4f}, max: {max_close:.4f}, min: {min_close:.4f}, avg vol: {avg_volume:.0f}\n"
                f"  Recent momentum (last 5): {recent_momentum_pct:+.2f}%\n"
            )
            prompt += (
                f"\n**Available Historical Data:** Up to {settings.OHLCV_RETENTION_DAYS} days "
                f"({settings.OHLCV_RETENTION_DAYS // 30} months) on {assigned_timeframe or 'default'} timeframe.\n"
                "You MUST include `backtest_period_days` in strategy parameters. Choose a relevant period:\n"
                "- 1w candles: 365–730 days\n- 1M candles: 730 days (all available)\n- 1d candles: 90–365 days\n"
                f"Default: {settings.OHLCV_RETENTION_DAYS} days.\n"
            )
        prompt += (
            "**Step 1: Propose Multiple Backtest Variants**\n"
            "Propose **multiple** sets of strategy parameters for backtesting. Each set is a \"backtest variant\" — a complete set of trading parameters "
            "(stop_loss_pct, take_profit_pct, max_hold_time_seconds, trailing_stop, position_size_fraction, etc.).\n"
            "**Backtest Entry Logic (REQUIRED):** You MUST include a `backtest_entry_config` object in every backtest variant. If omitted, the backtester will NOT run and will return an error.\n"
            "The `backtest_entry_config` object supports these fields (all optional — defaults shown):\n"
            "- `ema_period` (int, default 0): EMA period for trend filter. Set to 0 to disable EMA filter.\n"
            "- `ema_direction` (\"above\" or \"below\", default \"above\"): enter when close is above/below the EMA.\n"
            "- `min_adx` (float, default 0): minimum ADX to enter. Set to 0 to disable ADX filter.\n"
            "- `max_rsi` (float, default 100): maximum RSI to enter (avoids overbought). Set to 100 to disable.\n"
            "- `min_rsi` (float, default 0): minimum RSI to enter (avoids oversold entries). Set to 0 to disable.\n"
            "- `macd_filter` (\"positive\", \"negative\", or \"none\", default \"none\"): require MACD histogram above/below 0.\n"
            "- `logic` (\"and\" or \"or\", default \"and\"): combine all enabled filters with AND or OR logic.\n"
            "Example: `{\"backtest_entry_config\": {\"ema_period\": 21, \"ema_direction\": \"above\", \"min_adx\": 25, \"max_rsi\": 65, \"macd_filter\": \"positive\", \"logic\": \"and\"}}`\n"
            "**Slippage Model:** The backtester uses **dynamic slippage** based on each candle's relative volume and volatility (ATR%). Low-volume candles incur higher slippage (up to 3× base), and high-volatility candles add proportional slippage. This means strategies that trade in thin or volatile markets will show more realistic execution costs. The base slippage is 0.1%, capped at 1%.\n"
            "**CRITICAL — `backtest_entry_config` is REQUIRED:** If you omit `backtest_entry_config`, the backtester will NOT run and will return an error. You MUST include a `backtest_entry_config` object in every backtest variant that matches your intended entry conditions (e.g., EMA trend filter, ADX strength, RSI range, MACD direction). Without it, backtest results would be misleading because entering every candle does not reflect any actual entry strategy.\n"
            "Your goal is to find parameters that would have been profitable given your chosen entry logic.\n"
            "**Key Recommendations:**\n"
            "- Prefer ATR-based stops and take-profits to adapt to volatility.\n"
            "- Avoid very tight stops (< 1.5x ATR) as they will be triggered by normal noise.\n"
            "- Set a reasonable `cooldown_after_loss_seconds` (e.g., 1-3 candle periods) to avoid consecutive losses.\n"
            "- Use `trailing_stop` to lock in profits during strong trends.\n"
            f"Return these as a `backtest_variants` array in your JSON output. You decide how many variants to return "
            f"(minimum 1, recommended 3–5, maximum {settings.MAX_BACKTEST_VARIANTS}). If you provide more than {settings.MAX_BACKTEST_VARIANTS}, only the first {settings.MAX_BACKTEST_VARIANTS} will be tested. "
            "Each variant should explore a different hypothesis:\n"
            "- e.g., tight stop vs wide stop\n"
            "- short hold vs long hold\n"
            "- trailing stop on vs off\n"
            "- different take-profit targets\n"
            "- different position sizes\n"
            "The engine will run a local Python backtest for EACH variant sequentially. "
            "Running just one backtest may not be enough to intercept profitable configurations, "
            "so provide several diverse variants to maximize the chance of finding a winning strategy.\n"
            "You may also include a preliminary `action` and `confidence`, but your final decision will be made in Step 2 "
            "after reviewing ALL backtest results.\n"
        )
    if drawdown_pct is not None:
        prompt += f"Current account drawdown: {drawdown_pct}%\n"
    if recent_trades:
        _recent_compact = [
            {
                "sym": t.get("symbol"),
                "pnl": t.get("realized_pnl"),
                "reason": t.get("exit_reason"),
                "hold": t.get("hold_time_seconds"),
            }
            for t in recent_trades
        ]
        prompt += f"\nRecent closed trades (last {len(recent_trades)}):\n{json.dumps(_recent_compact)}\n"
        prompt += "If recent trades are losing, become more conservative.\n"

    # --- Past trades for this symbol ---
    if past_trades:
        prompt += f"\nPast closed trades for {symbol} (last {len(past_trades)}):\n"
        for t in past_trades:
            entry_price = t.get("price", 0.0)
            exit_price = t.get("exit_price", 0.0)
            amount = t.get("amount", 0.0)
            pnl = t.get("realized_pnl", 0.0)
            exit_reason = t.get("exit_reason", "unknown")
            hold_time = t.get("hold_time_seconds", None)
            strategy = t.get("strategy_type", "unknown")
            cost_basis = t.get("cost_basis", amount * entry_price)
            pnl_pct = (pnl / cost_basis * 100) if cost_basis > 0 else 0.0
            hold_str = f"{hold_time:.0f}s" if hold_time is not None else "N/A"
            prompt += f"- Entry:{entry_price:.4f} Exit:{exit_price:.4f} P&L:{pnl:+.2f} ({pnl_pct:+.1f}%) Reason:{exit_reason} Hold:{hold_str} Strat:{strategy}\n"

    if historical_backtest_results:
        prompt += f"\n**Historical Backtest Results for {symbol} (last {len(historical_backtest_results)} tests):**\n"
        for bt in historical_backtest_results:
            age_hours = (time.time() - bt.get("created_at", 0)) / 3600
            stats = bt.get("stats", {})
            params = bt.get("variant_params", {})
            prompt += (
                f"  [{bt.get('timeframe', '?')}] {age_hours:.1f}h ago: "
                f"SL={params.get('stop_loss_pct', '?')}, TP={params.get('take_profit_pct', '?')}, "
                f"trades={stats.get('total_trades', 0)}, win_rate={stats.get('win_rate', 0)*100:.1f}%, "
                f"total_pnl={stats.get('total_pnl_pct', 0)*100:+.2f}%, "
                f"max_dd={stats.get('max_drawdown_pct', 0)*100:.2f}%, "
                f"profit_factor={stats.get('profit_factor', 0):.2f}\n"
            )

    # --- Aggregate sentiment summary ---
    if sentiment_trend is not None:
        prompt += f"\nSentiment trend (change in compound score since last cycle): {sentiment_trend:+.4f}\n"
        prompt += "Positive delta = sentiment improving, negative = deteriorating. Adjust confidence and risk parameters accordingly.\n"
        prompt += (
            "\n**News Sentiment Exit Threshold:**\n"
            "You may include `\"news_sentiment_exit_threshold\"` in your strategy parameters (range -1.0 to 0.0). "
            "If set, the engine will automatically close the position when the aggregate news sentiment compound score "
            "drops below this threshold. **This value MUST be negative** (e.g., -0.3) — a positive or zero value would "
            "cause the position to exit even when sentiment is neutral or mildly positive, which is almost certainly "
            "not your intent. Use a more negative value (e.g., -0.5) for a stricter exit (only exit on very negative "
            "sentiment) or a less negative value (e.g., -0.1) for a more sensitive exit. Omit this field if you do not "
            "want a sentiment-based exit.\n"
        )
    if volume_trend is not None:
        prompt += f"\nVolume trend: {volume_trend:.2f}x (current daily volume relative to recent average)\n"
        prompt += "Ratio > 1.0 = volume above average, > 2.0 = significant spike. Elevated volume confirms price move strength. Low volume during breakout may signal fakeout.\n"
    if market_breadth:
        prompt += (
            f"\nMarket breadth: {market_breadth['positive_pct']}% of {market_breadth['total_count']} "
            f"candidate stocks have a positive daily change ({market_breadth['positive_count']} positive).\n"
            "High breadth (>70%) = broad market strength (risk-on); low breadth (<30%) = weakness (risk-off). Adjust selection and risk accordingly.\n"
        )
    if full_market_breadth:
        prompt += (
            f"\nFull market breadth (all available symbols): {full_market_breadth['positive_pct']}% of "
            f"{full_market_breadth['total_count']} symbols have a positive daily change "
            f"({full_market_breadth['positive_count']} positive).\n"
            "Broader measure of market health. If full breadth is very low (<25%) while candidate breadth is moderate, market may be more fragile than it appears.\n"
        )
    if donchian_channels:
        prompt += (
            f"\nDonchian Channels ({assigned_timeframe or 'default'}): "
            f"Upper={donchian_channels['upper']:.6f}, "
            f"Middle={donchian_channels['middle']:.6f}, "
            f"Lower={donchian_channels['lower']:.6f}\n"
        )
        prompt += "Donchian Channels: highest high/lowest low over lookback period. Breakout above upper = new high (bullish), below lower = new low (bearish). Narrow channel = low volatility (squeeze).\n"

    if parabolic_sar is not None:
        prompt += f"\nParabolic SAR ({assigned_timeframe or 'default'}): {parabolic_sar:.6f}\n"
        prompt += "Parabolic SAR: dots above price = downtrend, below price = uptrend. Use for trailing stop or trend confirmation.\n"

    if vwap is not None:
        prompt += f"\nVWAP (14-period, {assigned_timeframe or 'default'}): {vwap:.6f}\n"
        prompt += "VWAP is the volume-weighted average price. Price above VWAP = bullish, below = bearish. Use as dynamic support/resistance.\n"

    if daily_pivot_points:
        prompt += f"\n**Daily Pivot Points (Support/Resistance):**\n"
        prompt += f"  Pivot: {daily_pivot_points['pivot']}\n"
        prompt += f"  Resistances: R1={daily_pivot_points['r1']}, R2={daily_pivot_points['r2']}, R3={daily_pivot_points['r3']}\n"
        prompt += f"  Supports: S1={daily_pivot_points['s1']}, S2={daily_pivot_points['s2']}, S3={daily_pivot_points['s3']}\n"
        prompt += "Use these levels for potential entry (near supports) and exit (near resistances).\n"

    # --- News section (detailed articles) ---
    news_section = ""
    if settings.NEWS_ENABLED:
        articles = get_news_for_symbol(symbol, max_age_seconds=settings.NEWS_CACHE_TTL_SECONDS)
        if articles:
            raw_news = "Recent news articles for this stock:\n" + _format_news_for_prompt(articles)
            # Summarize the news section using the weak model to save tokens
            try:
                from src.llm.summarizer import summarize_text
                news_section = summarize_text(raw_news, context="strategy news", max_length=500)
            except Exception:
                news_section = raw_news
    if news_section:
        prompt += f"\n{news_section}\n"

    # Cap the validator minimum for long timeframes to avoid rejecting reasonable hold times
    if assigned_timeframe in ("3Y", "5Y"):
        validator_min = 31_536_000  # ~1 year minimum for 3Y/5Y
    elif assigned_timeframe in ("1Y", "6M"):
        validator_min = min(int(min_hold_time_mult * tf_seconds), 31_536_000)  # cap at ~1 year
    else:
        validator_min = int(min_hold_time_mult * tf_seconds)

    # Use validator_min as the "reasonable minimum" in the prompt so the LLM
    # is told the same value the validator actually enforces.
    prompt += f"""
**For the {assigned_timeframe or 'default'} timeframe, a reasonable minimum max_hold_time_seconds is {validator_min} seconds. The validator enforces a hard minimum of {validator_min} seconds for this timeframe. Your max_hold_time_seconds must be at least this value.

You are trading spot only (no shorting). Only output SELL if you currently hold the asset.
"""
    # --- Fundamental Data ---
    if fundamentals:
        prompt += "\n**Fundamental Data (Medium/Long-Term Context):**\n"
        if fundamentals.get("sector"):
            prompt += f"  Sector: {fundamentals['sector']}\n"
        if fundamentals.get("industry"):
            prompt += f"  Industry: {fundamentals['industry']}\n"
        if fundamentals.get("market_cap") is not None:
            try:
                mc = float(fundamentals["market_cap"])
                if mc >= 1e12:
                    mc_str = f"{mc/1e12:.2f}T"
                elif mc >= 1e9:
                    mc_str = f"{mc/1e9:.2f}B"
                elif mc >= 1e6:
                    mc_str = f"{mc/1e6:.2f}M"
                else:
                    mc_str = str(mc)
                prompt += f"  Market Cap: {mc_str}\n"
            except (TypeError, ValueError):
                pass

        def _safe_fmt(val, mult=1.0, suffix="", fmt=".2f"):
            try:
                return f"{float(val) * mult:{fmt}}{suffix}"
            except (TypeError, ValueError):
                return "N/A"

        if fundamentals.get("pe_ratio") is not None:
            prompt += f"  P/E Ratio (trailing): {_safe_fmt(fundamentals['pe_ratio'])}\n"
        if fundamentals.get("forward_pe") is not None:
            prompt += f"  Forward P/E: {_safe_fmt(fundamentals['forward_pe'])}\n"
        if fundamentals.get("dividend_yield") is not None:
            prompt += f"  Dividend Yield: {_safe_fmt(fundamentals['dividend_yield'], mult=100, suffix='%')}\n"
        if fundamentals.get("price_to_book") is not None:
            prompt += f"  Price/Book: {_safe_fmt(fundamentals['price_to_book'])}\n"
        if fundamentals.get("profit_margins") is not None:
            prompt += f"  Profit Margins: {_safe_fmt(fundamentals['profit_margins'], mult=100, suffix='%')}\n"
        if fundamentals.get("return_on_equity") is not None:
            prompt += f"  Return on Equity: {_safe_fmt(fundamentals['return_on_equity'], mult=100, suffix='%')}\n"
    prompt += (
        "\n**Entry Condition (REQUIRED for every BUY):**\n"
        "You MUST include an `entry_condition` object in your JSON output for every BUY action. "
        "This tells the bot the **exact moment** to enter the trade. "
        "If you omit this field, the trade will be executed immediately at the current market price. "
        "The object must have a `\"type\"` field and, except for `\"delay\"`, a `\"timeout_seconds\"` field.\n"
        "Supported types:\n"
        "- `\"limit_price\"`: wait for the price to drop to or below `\"price\"`.\n"
        "  Example: {\"type\": \"limit_price\", \"price\": 1.23, \"timeout_seconds\": 3600}\n"
        "- `\"rsi_threshold\"`: wait for RSI(14) to fall below `\"rsi_below\"`.\n"
        "  Example: {\"type\": \"rsi_threshold\", \"rsi_below\": 30, \"timeout_seconds\": 7200}\n"
        "- `\"delay\"`: simply wait `\"delay_seconds\"` before executing.\n"
        "  Example: {\"type\": \"delay\", \"delay_seconds\": 3600}\n"
        "- `\"indicator_combo\"`: wait until ALL listed indicator conditions are met.\n"
        "  Supported indicators: `rsi`, `macd`, `macd_signal`, `macd_hist`, `bb_upper`, `bb_middle`, `bb_lower`, `ema_9`, `ema_21`, `stochastic_k`, `stochastic_d`, `adx`, `plus_di`, `minus_di`, `obv`, `mfi`, `cci`, `williams_r`, `parabolic_sar`, `atr`.\n"
        "  Example: {\"type\": \"indicator_combo\", \"conditions\": [ {\"indicator\": \"rsi\", \"threshold\": 30, \"direction\": \"below\"}, {\"indicator\": \"macd_hist\", \"threshold\": 0, \"direction\": \"above\"} ], \"timeout_seconds\": 7200}\n"
        "If a timeout expires without the condition being met, the trade is skipped entirely.\n"
        "**Important:** The engine enforces a minimum timeout of 300 seconds or "
        f"{settings.ENTRY_CONDITION_MIN_TIMEOUT_MULT}× the candle timeframe, whichever is larger. "
        "Set `timeout_seconds` to at least this value, and prefer longer timeouts for higher timeframes "
        "(e.g., 3600–7200 s for 1d candles).\n"
        "For 1w candles, consider timeouts of 86400–604800 s (1–7 days); "
        "for 1M candles, 604800–2592000 s (1–4 weeks).\n"
    )
    prompt += (
        "\n**Output ONLY the raw JSON object as specified.**\n\n"
        "Return a JSON object with these **required** fields:\n"
        "- `action`: one of BUY, SELL, HOLD\n"
        "- `confidence`: a float between 0.0 and 1.0\n"
        "- `reasoning`: a string explaining **why** you chose this action and this confidence level. "
        "Include the key factors (indicators, sentiment, market regime, etc.) that led to your decision. "
        "You MUST also include the decided price (the current market price or your specified `limit_price`) in the reasoning message.\n"
        "- `strategy`: an object containing `type` (string) and `parameters` (object).\n"
        "  The `parameters` object MUST include ALL required trading parameters:\n"
        "  `stop_loss_pct`, `take_profit_pct`, `position_size_fraction`, `confidence_sizing_weight`, `trailing_stop`, `max_hold_time_seconds`, `cooldown_after_loss_seconds`, `backtest_period_days`,\n"
        "  and `backtest_entry_config` (REQUIRED for BUY actions — the same entry logic object used in your backtest variants), etc.\n"
        "- `backtest_variants`: a JSON array of objects, each containing a complete set of strategy parameters for backtesting. "
        "Each variant MUST include at minimum: `stop_loss_pct`, `take_profit_pct`, `max_hold_time_seconds`, `trailing_stop`, "
        "`position_size_fraction`, and `backtest_period_days`. You decide how many variants to return (minimum 1, recommended 3–5). "
        "Each variant should explore a different hypothesis (e.g., tight vs wide stop, short vs long hold, trailing on vs off, etc.). "
        "The engine will run a backtest for EACH variant and present ALL results in Step 2.\n"
        "- `entry_condition`: REQUIRED for every BUY action. An object specifying the exact moment to enter the trade (see Entry Condition section below for format).\n"
        "- `limit_price`: optional, a specific limit price for the order.\n"
        "- `time_in_force`: optional, \"day\" or \"gtc\". Default \"day\".\n"
    )
    if trading_paused:
        prompt += (
            "\n**Trading is currently PAUSED.** You can still output BUY, SELL, or HOLD actions. "
            "However, any BUY signals will NOT be executed; they will only be sent as notifications. "
            "SELL signals for existing positions will be executed normally if the market is open. "
            "Please continue to analyze the market and generate signals as you normally would.\n"
        )
    if performance:
        stock_perf = performance.get("stock_performance", {}).get(symbol, {})
        strategy_perf = performance.get("strategy_performance", {})
        equity = performance.get("equity_curve", {})
        perf_lines = ["Historical Performance:"]
        if stock_perf:
            perf_lines.append(f"- This stock's past performance: {json.dumps(stock_perf)} (stop_loss_hits = number of times stop-loss was triggered; avg_hold_time_seconds = average trade duration)")
        if equity:
            perf_lines.append(f"- Overall equity curve: {json.dumps(equity)}")
        if strategy_perf:
            perf_lines.append(f"- Strategy performance summary: {json.dumps(strategy_perf)}")
        
        if len(perf_lines) > 1:
            prompt += "\n".join(perf_lines) + "\n"
        daily_pnl = equity.get("daily_pnl", 0.0)
        total_pnl = equity.get("total_pnl", 0.0)
        consecutive_losses = equity.get("consecutive_losses", 0)
        if daily_pnl is not None:
            prompt += f"Today's realized P&L: {daily_pnl:.4f} {base_currency}\n"
        if consecutive_losses > 0:
            prompt += f"⚠️ You have {consecutive_losses} consecutive losing trades. Consider reducing risk or skipping this trade.\n"
        prompt += f"\n**Account P&L**: Total realized P&L = {total_pnl:.4f} {base_currency}.\n"
    # --- Trade pattern analysis ---
    if trade_pattern_analysis:
        prompt += "\n" + _format_trade_pattern_analysis(trade_pattern_analysis) + "\n"
    if max_hold_expired:
        prompt += (
            f"\n**IMPORTANT: The max hold time for your current position in {symbol} has expired "
            f"(this is occurrence #{max_hold_expired_count}).**\n"
            "You must decide immediately whether to SELL now or to extend the hold time.\n"
            "- If you believe the position still has profit potential, output a **HOLD** action "
            "and provide a new `max_hold_time_seconds` in the `parameters` object (you may also "
            "update stop‑loss, take‑profit, or any other parameters).\n"
            "- If you decide to exit, output a **SELL** action.\n"
            "**Do NOT output HOLD without a new `max_hold_time_seconds`** – that will be treated "
            "as a decision to sell immediately.\n"
        )
    if stop_loss_triggered:
        prompt += (
            f"\n**⚠️ STOP-LOSS TRIGGERED (review {stop_loss_review_count}/{max_stop_loss_reviews}):** "
            f"Your stop-loss level was triggered for {symbol}.\n"
            "You must decide immediately:\n"
            "- **SELL**: output a SELL action to close the position.\n"
            "- **HOLD with adjusted stop**: output a HOLD action and provide a **new, lower stop-loss** "
            "(via `stop_loss_pct` or `stop_loss_atr_multiple`). You may also update other parameters "
            "(e.g., take-profit, trailing stop).\n"
            "**If you output HOLD without a new stop-loss, the engine will force-sell the position.**\n"
            "Choose the option that you believe will maximise profit or minimise loss given the current "
            "market conditions and indicators.\n"
        )
    if take_profit_triggered:
        prompt += (
            f"\n**🎯 TAKE-PROFIT TRIGGERED (review {take_profit_review_count}/{max_take_profit_reviews}):** "
            f"Your take-profit level was reached for {symbol}.\n"
            "You must decide immediately:\n"
            "- **SELL**: output a SELL action to take the profit.\n"
            "- **HOLD with adjusted take-profit**: output a HOLD action and provide a **new, higher take-profit** "
            "(via `take_profit_pct`). You may also update other parameters (e.g., stop-loss, trailing stop).\n"
            "**If you output HOLD without a new `take_profit_pct`, the engine will force-sell the position.**\n"
            "Choose the option that you believe will maximise profit given the current "
            "market conditions and indicators.\n"
        )
    # --- Partial take-profit triggered ---
    if partial_tp_triggered:
        levels_str = ", ".join(str(i) for i in partial_tp_triggered_levels) if partial_tp_triggered_levels else "unknown"
        prompt += (
            f"\n**⚠️ PARTIAL TAKE‑PROFIT TRIGGERED (review {partial_tp_review_count}/{max_partial_tp_reviews}):** "
            f"Partial take‑profit level(s) {levels_str} for {symbol} have been reached.\n"
            "You must decide immediately:\n"
            "- **Execute**: let the partial sell(s) happen as originally planned. "
            "Output HOLD **without** changing the `partial_take_profit_levels` array.\n"
            "- **Adjust**: output HOLD and provide an **updated** `partial_take_profit_levels` array "
            "with a new `take_profit_pct` for the triggered level(s), or remove them entirely.\n"
            "- **Sell All**: output SELL to close the **entire** position.\n"
            "If you output HOLD without updating `partial_take_profit_levels`, the partial sell(s) will execute.\n"
        )
    # --- Dust sweep triggered ---
    if dust_sweep_triggered:
        prompt += (
            f"\n**🧹 DUST SWEEP TRIGGERED (review {dust_sweep_review_count}/{max_dust_sweep_reviews}):** "
            f"The remaining position size for {symbol} is below the minimum trade amount and cannot be sold normally.\n"
            "You must decide immediately:\n"
            "- **Sell Dust**: output SELL to sell the remaining dust (a market sell will be attempted).\n"
            "- **Hold**: output HOLD to keep the dust. It may become tradeable again if the price rises.\n"
            "If you output HOLD, the dust will be kept. If you output SELL, the dust will be sold.\n"
        )
    if symbol_event and symbol_event.get("has_event"):
        prompt += (
            f"\n**⚠️ Upcoming Corporate Event Detected for {symbol}:**\n"
            f"  Event types: {', '.join(symbol_event.get('event_types', []))}\n"
            f"  Detected keywords: {', '.join(symbol_event.get('keywords', [])[:5])}\n"
        )
    return prompt


def build_analysis_prompt(data: StrategyPromptData) -> str:
    """Build a focused prompt for Step 1a: Market analysis only.

    Reuses build_strategy_prompt for all market data context,
    but appends a simpler output format instruction at the end.
    No trading parameters, backtest variants, or entry conditions are requested.
    """
    full_prompt = build_strategy_prompt(data)

    analysis_output = (
        "\n\n**IMPORTANT — Step 1a: Analysis Only**\n"
        "For this step, IGNORE the output format instructions above. "
        "Instead, output ONLY a raw JSON object with these fields:\n"
        '- "action": one of BUY, SELL, HOLD\n'
        '- "confidence": a float between 0.0 and 1.0\n'
        '- "reasoning": a string explaining your analysis. Include the key factors '
        '(indicators, sentiment, market regime, fundamentals, news, portfolio context) '
        'that led to your decision. You MUST include the current market price in the reasoning.\n'
        '- "strategy_direction": a short string describing your intended strategy approach '
        '(e.g., "momentum_breakout", "mean_reversion", "trend_following", "range_trading", "hold")\n\n'
        "Focus ONLY on analyzing the market data and deciding the direction. "
        "Do NOT output trading parameters, backtest variants, or entry conditions — "
        "those will be requested in the next step.\n\n"
        "Output ONLY the raw JSON object."
    )

    return full_prompt + analysis_output


