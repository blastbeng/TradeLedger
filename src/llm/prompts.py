import json
import logging
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from src.config.settings import settings
from src.utils.symbol_utils import is_btp_isin
from src.database import get_news_for_symbol
from src.llm.stock_selection_prompts import build_stock_selection_prompt, build_final_selection_prompt, build_stock_selection_messages, build_final_selection_messages
from src.exchanges.market_data import TIMEFRAME_MAP
from src.llm.prompt_utils import (
    _timeframe_to_seconds,
    compact_prompt,
    _summarize_ohlcv,
    _format_trade_pattern_analysis,
    _format_news_for_prompt,
    get_cached_news_summary,
    _round_floats,
    _to_toon,
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
    ytm: Optional[float] = None
    dividend_yield: Optional[float] = None
    next_ex_dividend: Optional[Tuple[str, int]] = None


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
    ytm = data.ytm
    dividend_yield = data.dividend_yield
    next_ex_dividend = data.next_ex_dividend
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
    _ticker_compact = {}
    for k in ("last", "bid", "ask", "volume", "quoteVolume", "name", "coupon", "maturity"):
        if k in ticker:
            val = ticker.get(k)
            if k in ("last", "bid", "ask") and isinstance(val, (int, float)):
                _ticker_compact[k] = round(val, 2)
            elif k in ("volume", "quoteVolume") and isinstance(val, (int, float)):
                _ticker_compact[k] = round(val)
            else:
                _ticker_compact[k] = val
    # Rename "percentage" to "change_24h" so the LLM understands what it represents
    _pct = ticker.get("percentage")
    if _pct is not None:
        _ticker_compact["change_24h"] = _pct
    _balance_compact = {k: round(v, 2) if isinstance(v, (int, float)) else v for k, v in balance.items()}
    prompt = f"""Symbol: {symbol}
Current ticker: {_to_toon(_ticker_compact)}
Current balances: {_to_toon(_balance_compact)}
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
        else:
            prompt += "This is the only symbol being traded; you may use the full available balance.\n"
    _positions_compact = []
    for p in open_positions:
        pos = {
            "symbol": p.get("symbol"),
        }
        _amt = p.get("amount")
        pos["amount"] = round(_amt, 4) if isinstance(_amt, (int, float)) else _amt
        for key in ("price", "stop_loss", "take_profit"):
            val = p.get(key)
            if isinstance(val, (int, float)):
                pos[key] = round(val, 2)
            else:
                pos[key] = val
        _positions_compact.append({
            "symbol": pos["symbol"],
            "entry": pos["price"],
            "amount": pos["amount"],
            "sl": pos["stop_loss"],
            "tp": pos["take_profit"],
        })
    prompt += f"""Open positions: {_to_toon(_positions_compact)}
Total available {base_currency} balance: {base_balance:.2f}
Suggested equal share per symbol: {per_symbol_budget:.2f} {base_currency}
Maximum symbols to trade: {max_symbols}

**Focus:** Trade only {symbol}. May use full remaining balance if high conviction.
"""
    # --- Portfolio exposure summary ---
    if portfolio_total_value is not None:
        prompt += f"\n**Portfolio:** Val={portfolio_total_value:.2f},{portfolio_open_count}pos,Dep={portfolio_exposure_pct:.1f}%,Risk={portfolio_stop_risk_pct:.2f}%,Avail={portfolio_available_capital:.2f}\n"
        if max_portfolio_exposure_pct is not None and max_portfolio_stop_risk_pct is not None:
            prompt += f"Limits: Exp<{max_portfolio_exposure_pct*100:.0f}%,Risk<{max_portfolio_stop_risk_pct*100:.0f}%. Reduce size if exceeded.\n"
    if cycle_spent is not None and remaining_balance is not None:
        prompt += f"CycleSpent:{cycle_spent:.0f},Remaining:{remaining_balance:.0f}\n"
        prompt += f"MaxTrade:{remaining_balance:.2f} (full remaining). position_size_fraction must not exceed remaining.\n"
    if global_risk_multiplier is not None and global_risk_multiplier < 1.0:
        prompt += f"\n**GlobalRiskMult:{global_risk_multiplier:.2f}** (amount=frac×balance×mult)\n"
    # --- Queued orders for this symbol ---
    if queued_orders:
        symbol_queued = [q for q in queued_orders if q.get('symbol') == symbol]
        if symbol_queued:
            now = time.time()
            prompt += "\n**QueuedOrders:**\n"
            for q in symbol_queued:
                side = q.get('side', '?').upper()
                limit_price = q.get('limit_price')
                age_sec = now - q.get('queued_at', 0) if q.get('queued_at') else 0
                age_min = int(age_sec / 60)
                lp_str = f"{limit_price:.2f}" if isinstance(limit_price, (int, float)) else str(limit_price)
                prompt += f"  {side} @ {lp_str} ({age_min}m ago)\n"
            prompt += "Do NOT output new BUY/SELL while queued order exists. Output HOLD to change.\n"
    base_symbol = symbol
    quote_currency = base_currency
    if min_order_amount is not None or min_order_cost is not None:
        prompt += f"\nMinOrder: "
        if min_order_amount is not None: prompt += f"{min_order_amount} {base_symbol}"
        if min_order_cost is not None: prompt += f" (or {min_order_cost} {quote_currency})"
        prompt += "\n"
    if assigned_timeframe:
        prompt += f"TF:{assigned_timeframe}\n"
    if market_regime:
        prompt += f"Regime:{market_regime}\n"

    if session_info:
        prompt += f"UTC:{session_info['utc_hour']}h({session_info['session']})\n"
    if minutes_to_market_close is not None:
        if minutes_to_market_close > 0:
            prompt += f"MktClose:{minutes_to_market_close}min\n"
        else:
            prompt += "Mkt:Closed\n"
    if current_strategy_interval_seconds is not None:
        prompt += f"EvalInt:{current_strategy_interval_seconds}s\n"

    # --- Volatility, order book imbalance, and position P&L context ---
    if atr is not None:
        prompt += f"ATR(14,{assigned_timeframe or 'default'}):{atr:.2f}\n"
    if atr is not None and current_price is not None and current_price > 0:
        atr_pct = atr / current_price
        min_sl = min_stop_atr_mult * atr_pct
        prompt += f"ATR%={atr_pct:.2%},minSL={min_sl:.2%}({min_stop_atr_mult}×ATR%)\n"
    if atr_percentile is not None:
        prompt += f"ATR percentile (last 100 obs): {atr_percentile:.1f}%\n"
    if atr_multi_tf:
        prompt += f"ATR across timeframes: {_to_toon(_round_floats(atr_multi_tf))}\n"
    # --- Transaction cost break-even calculation ---
    _is_btp = is_btp_isin(symbol)
    # Use per_symbol_budget as the representative trade size — the LLM typically
    # trades a fraction of the full balance, and fixed fees make small trades
    # proportionally more expensive.
    trade_value = per_symbol_budget if per_symbol_budget is not None and per_symbol_budget > 0 else (remaining_balance if remaining_balance is not None and remaining_balance > 0 else base_balance)
    if trade_value > 0:
        if _is_btp:
            if settings.BTP_IS_PRIMARY_ISSUANCE:
                total_fees = 0.0
                prompt += f"\n**Fees:** Primary issuance — zero fees.\n"
            else:
                buy_fee = max(settings.BTP_MIN_FEE, trade_value * settings.BTP_FEE_PERC)
                sell_fee = max(settings.BTP_MIN_FEE, trade_value * settings.BTP_FEE_PERC)
                total_fees = buy_fee + sell_fee
                break_even_pct = total_fees / trade_value
                prompt += (
                    f"\n**Fees:** FeePerc={settings.BTP_FEE_PERC*100:.2f}%,MinFee={settings.BTP_MIN_FEE:.2f}. "
                    f"Round-trip@{trade_value:.0f}={total_fees:.2f} ({break_even_pct*100:.2f}%). "
                    f"Smaller trades → higher % due to min fee. TP must be > {break_even_pct*100:.2f}%.\n"
                )
        else:
            buy_fee = max(settings.STOCK_FEE_MIN, trade_value * settings.STOCK_FEE_PERC) + settings.STOCK_FEE_FIXED + (trade_value * settings.TOBIN_TAX_RATE)
            sell_fee = max(settings.STOCK_FEE_MIN, trade_value * settings.STOCK_FEE_PERC) + settings.STOCK_FEE_FIXED
            total_fees = buy_fee + sell_fee
            break_even_pct = total_fees / trade_value
            prompt += (
                f"\n**Fees:** Perc={settings.STOCK_FEE_PERC*100:.2f}%,Min={settings.STOCK_FEE_MIN:.2f},"
                f"Fixed={settings.STOCK_FEE_FIXED:.2f},Tobin={settings.TOBIN_TAX_RATE*100:.2f}%. "
                f"Round-trip@{trade_value:.0f}={total_fees:.2f} ({break_even_pct*100:.2f}%). "
                f"Smaller trades → higher % due to fixed+min fees. TP must be > {break_even_pct*100:.2f}%.\n"
            )
    if ytm is not None:
        prompt += f"YTM:{ytm:.2f}%\n"
    if dividend_yield is not None:
        prompt += f"DivYield:{dividend_yield*100:.2f}%\n"
    if next_ex_dividend is not None:
        prompt += f"NextExDiv:{next_ex_dividend[0]}({next_ex_dividend[1]}d)\n"
    # --- Show the LLM its previous decision for this symbol ---
    if last_decision:
        age_seconds = time.time() - last_decision.get("timestamp", 0)
        prompt += f"\n**PrevDecision({int(age_seconds // 60)}m ago):** {last_decision.get('action')},conf={last_decision.get('confidence', 0):.2f}"
        sl_pct = last_decision.get("stop_loss_pct")
        tp_pct = last_decision.get("take_profit_pct")
        psf = last_decision.get("position_size_fraction")
        parts = []
        if sl_pct is not None: parts.append(f"SL={sl_pct:.2f}")
        if tp_pct is not None: parts.append(f"TP={tp_pct:.2f}")
        if psf is not None: parts.append(f"Size={psf:.2f}")
        if parts: prompt += f" ({','.join(parts)})"
        prompt += f" R:{last_decision.get('reasoning', '')[:80]}\n"
    if unrealized_pnl is not None and position_info:
        prompt += f"Current position unrealized P&L: {unrealized_pnl:.0f} {base_currency}\n"
        entry_price = position_info.get('price', 0)
        amount = position_info.get('amount', 0)
        prompt += f"Position: entry {entry_price:.2f}, amount {amount:.4f}\n"
        prompt += f"\n**You hold {amount:.4f} {base_symbol} @ {entry_price:.2f}.**\n"
        prompt += "BUY = ADD to position (scale in). SELL = close ENTIRE position.\n"
        if entry_price > 0 and amount > 0:
            cost_basis = entry_price * amount
            if cost_basis > 0:
                pnl_pct = (unrealized_pnl / cost_basis) * 100
                prompt += f"Unrealized P&L: {pnl_pct:+.0f}%\n"
        current_sl = position_info.get('stop_loss')
        current_tp = position_info.get('take_profit')
        if current_sl is not None: prompt += f"Current SL price: {current_sl:.2f}\n"
        if current_tp is not None: prompt += f"Current TP price: {current_tp:.2f}\n"
        if current_price and current_price > 0:
            if current_sl is not None:
                sl_distance_pct = ((current_price - current_sl) / current_price) * 100
                prompt += f"Distance to SL: {sl_distance_pct:.0f}% below current\n"
            if current_tp is not None:
                tp_distance_pct = ((current_tp - current_price) / current_price) * 100
                prompt += f"Distance to TP: {tp_distance_pct:.0f}% above current\n"
        trailing_active = position_info.get('trailing_stop', False)
        if trailing_active:
            trailing_dist = position_info.get('trailing_stop_distance_pct')
            trailing_act = position_info.get('trailing_stop_activation_pct')
            prompt += f"Trailing stop: enabled (dist={trailing_dist:.2f}, act={trailing_act:.2f})\n"
        max_hold = position_info.get('max_hold_time_seconds')
        if max_hold is not None and max_hold > 0:
            entry_ts = position_info.get('timestamp', 0) / 1000.0
            elapsed = time.time() - entry_ts if entry_ts > 0 else 0
            remaining = max(0, max_hold - elapsed)
            prompt += f"Max hold: {max_hold:.0f}s total, {int(remaining // 60)}m remaining\n"

    # --- Multi-timeframe OHLCV summary and indicators ---
    if multi_tf_raw_candles:
        tf_summaries = []
        for tf in settings.OHLCV_TIMEFRAMES:
            if tf in multi_tf_raw_candles:
                summary = _summarize_ohlcv(multi_tf_raw_candles[tf])
                if summary:
                    tf_summaries.append(
                        f"  [{tf}] chg={summary['change_pct']}%, H={summary['high']:.2f}, L={summary['low']:.2f}, "
                        f"vol={summary['volume']}, candles={summary['candle_count']}"
                    )
        if tf_summaries:
            prompt += "\nMTF OHLCV:\n" + "\n".join(tf_summaries) + "\n"
    if multi_tf_indicators:
        ind_lines = []
        for tf in settings.OHLCV_TIMEFRAMES:
            if tf in multi_tf_indicators:
                ind = multi_tf_indicators[tf]
                ind_compact = {}
                if ind.get('rsi') is not None: ind_compact['rsi'] = round(ind['rsi'], 2)
                if ind.get('macd') is not None:
                    ind_compact['macd'] = round(ind['macd'], 2)
                    ind_compact['macd_sig'] = round(ind['macd_signal'], 2)
                    ind_compact['macd_h'] = round(ind['macd_hist'], 2)
                if ind.get('bb_upper') is not None:
                    ind_compact['bb_u'] = round(ind['bb_upper'], 2)
                    ind_compact['bb_m'] = round(ind['bb_middle'], 2)
                    ind_compact['bb_l'] = round(ind['bb_lower'], 2)
                if ind.get('ema_9') is not None:
                    ind_compact['ema9'] = round(ind['ema_9'], 2)
                    if ind.get('ema_21') is not None: ind_compact['ema21'] = round(ind['ema_21'], 2)
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
                    ind_compact['ich'] = {"t": round(ich['tenkan_sen'], 2), "k": round(ich['kijun_sen'], 2),
                                          "sa": round(ich['senkou_span_a'], 2), "sb": round(ich['senkou_span_b'], 2),
                                          "cb": round(ich['cloud_bottom'], 2), "ct": round(ich['cloud_top'], 2)}
                if ind.get('donchian_channels') is not None:
                    dc = ind['donchian_channels']
                    ind_compact['dc'] = {"u": round(dc['upper'], 2), "m": round(dc['middle'], 2), "l": round(dc['lower'], 2)}
                if ind.get('atr') is not None: ind_compact['atr'] = round(ind['atr'], 2)
                if ind.get('parabolic_sar') is not None: ind_compact['sar'] = round(ind['parabolic_sar'], 2)
                if ind.get('keltner_channels') is not None:
                    kc = ind['keltner_channels']
                    ind_compact['kc'] = {"u": round(kc['upper'], 2), "m": round(kc['middle'], 2), "l": round(kc['lower'], 2)}
                

                if not ind_compact: continue
                ind_lines.append(f"[{tf}] {_to_toon(ind_compact)}")
        if ind_lines:
            prompt += "\nComputed indicators per timeframe:\n" + "\n".join(ind_lines) + "\n"
    elif raw_candles:
        summary = _summarize_ohlcv(raw_candles)
        if summary:
            prompt += (
                f"\nOHLCV summary ({assigned_timeframe}): chg={summary['change_pct']}%, H={summary['high']:.2f}, "
                f"L={summary['low']:.2f}, vol={summary['volume']}, candles={summary['candle_count']}\n"
            )
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
                f"\nHistOHLCV ({hist_summary['candle_count']}c,{assigned_timeframe or 'default'}):"
                f"chg={hist_summary['change_pct']:.2f}%,H={hist_summary['high']:.2f},L={hist_summary['low']:.2f}\n"
                f"Last20:avg={avg_close:.2f},max={max_close:.2f},min={min_close:.2f},vol={avg_volume:.0f},mom5={recent_momentum_pct:+.2f}%\n"
            )
        prompt += (
            f"\n**Available Historical Data:** Up to {settings.OHLCV_RETENTION_DAYS} days on {assigned_timeframe or 'default'}.\n"
            "Set `backtest_period_days` (1w:365-730, 1M:730, 1d:90-365). Default: "
            f"{settings.OHLCV_RETENTION_DAYS}.\n\n"
            "**Step 1: Propose Multiple Backtest Variants**\n"
            "Propose 3-5 backtest variants (max "
            f"{settings.MAX_BACKTEST_VARIANTS}). Each variant = complete param set (SL,TP,hold,trailing,size,entry_config).\n"
            "**REQUIRED:** `backtest_entry_config` in EVERY variant. Fields: ema_period,ema_direction,min_adx,max_rsi,min_rsi,macd_filter,logic.\n"
            "Example: {\"backtest_entry_config\":{\"ema_period\":21,\"ema_direction\":\"above\",\"min_adx\":25,\"logic\":\"and\"}}\n"
            "Dynamic slippage: low-volume candles → higher slippage (up to 3×). Base 0.1%, cap 1%.\n"
            "Explore different hypotheses: tight vs wide SL, short vs long hold, trailing on/off, different TP targets.\n"
        )
    if drawdown_pct is not None:
        prompt += f"Drawdown:{drawdown_pct:.0f}%\n"
    if recent_trades:
        _recent_compact = [
            {
                "sym": t.get("symbol"),
                "pnl": round(t.get("realized_pnl", 0), 2) if isinstance(t.get("realized_pnl"), (int, float)) else t.get("realized_pnl"),
                "reason": t.get("exit_reason"),
                "hold": int(t.get("hold_time_seconds", 0) / 60) if t.get("hold_time_seconds") is not None else None,
            }
            for t in recent_trades
        ]
        prompt += f"\nRecentTrades: {_to_toon(_recent_compact)}\n"

    # --- Past trades for this symbol ---
    if past_trades:
        _past_compact = [
            {"e": round(t.get("price", 0.0), 2), "x": round(t.get("exit_price", 0.0), 2), "pnl": round(t.get("realized_pnl", 0.0), 2),
             "r": t.get("exit_reason", ""), "h": int(t.get("hold_time_seconds", 0) / 60) if t.get("hold_time_seconds") is not None else None, "s": t.get("strategy_type", "")}
            for t in past_trades
        ]
        prompt += f"\nPastTrades({symbol}): {_to_toon(_past_compact)}\n"

    if historical_backtest_results:
        _ht_compact = [
            {"tf": bt.get('timeframe', '?'), "h": int((time.time() - bt.get("created_at", 0)) / 3600),
             "SL": bt.get("variant_params", {}).get('stop_loss_pct', '?'),
             "TP": bt.get("variant_params", {}).get('take_profit_pct', '?'),
             "t": bt.get("stats", {}).get('total_trades', 0),
             "wr": round(bt.get("stats", {}).get('win_rate', 0), 2),
             "pnl": round(bt.get("stats", {}).get('total_pnl_pct', 0), 2),
             "dd": round(bt.get("stats", {}).get('max_drawdown_pct', 0), 2),
             "pf": round(bt.get("stats", {}).get('profit_factor', 0), 2)}
            for bt in historical_backtest_results
        ]
        prompt += f"\n**HistBT({symbol}):** {_to_toon(_ht_compact)}\n"

    # --- Aggregate sentiment summary ---
    if sentiment_trend is not None:
        prompt += f"\nSentimentTrend: {sentiment_trend:+.2f} (delta compound since last cycle)\n"
        prompt += "Set `news_sentiment_exit_threshold` (-1.0 to 0.0, MUST be negative) to auto-exit on negative sentiment. Omit to disable.\n"
    if volume_trend is not None:
        prompt += f"\nVolTrend: {volume_trend:.2f}x (current vs avg). >2.0=spike, <1.0=low.\n"
    if market_breadth:
        prompt += f"\nMktBreadth: {market_breadth['positive_pct']}% pos ({market_breadth['positive_count']}/{market_breadth['total_count']})\n"
    if full_market_breadth:
        prompt += f"FullMktBreadth: {full_market_breadth['positive_pct']}% pos ({full_market_breadth['positive_count']}/{full_market_breadth['total_count']})\n"
    if donchian_channels:
        prompt += (
            f"\nDonchian ({assigned_timeframe or 'default'}): "
            f"U={donchian_channels['upper']:.2f},M={donchian_channels['middle']:.2f},"
            f"L={donchian_channels['lower']:.2f}\n"
        )

    if parabolic_sar is not None:
        prompt += f"\nSAR ({assigned_timeframe or 'default'}): {parabolic_sar:.2f}\n"

    if vwap is not None:
        prompt += f"\nVWAP ({assigned_timeframe or 'default'}): {vwap:.2f}\n"

    if daily_pivot_points:
        prompt += f"\nPivots: P={daily_pivot_points['pivot']:.2f},R1={daily_pivot_points['r1']:.2f},R2={daily_pivot_points['r2']:.2f},S1={daily_pivot_points['s1']:.2f},S2={daily_pivot_points['s2']:.2f}\n"

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
    prompt += f"\nMin max_hold_time for {assigned_timeframe or 'default'}: {validator_min}s. Spot only (no shorting). SELL only if holding.\n"
    # --- Fundamental Data ---
    if fundamentals:
        prompt += "\n**Fundamentals:**"
        parts = []
        if fundamentals.get("sector"): parts.append(f"Sector={fundamentals['sector']}")
        if fundamentals.get("industry"): parts.append(f"Ind={fundamentals['industry']}")
        if fundamentals.get("market_cap") is not None:
            try:
                mc = float(fundamentals["market_cap"])
                if mc >= 1e12: mc_str = f"{mc/1e12:.2f}T"
                elif mc >= 1e9: mc_str = f"{mc/1e9:.2f}B"
                elif mc >= 1e6: mc_str = f"{mc/1e6:.2f}M"
                else: mc_str = str(mc)
                parts.append(f"MC={mc_str}")
            except (TypeError, ValueError): pass
        if fundamentals.get("pe_ratio") is not None: parts.append(f"PE={fundamentals['pe_ratio']:.2f}")
        if fundamentals.get("forward_pe") is not None: parts.append(f"FwdPE={fundamentals['forward_pe']:.2f}")
        if fundamentals.get("dividend_yield") is not None: parts.append(f"DivY={fundamentals['dividend_yield']*100:.2f}%")
        if fundamentals.get("price_to_book") is not None: parts.append(f"PB={fundamentals['price_to_book']:.2f}")
        if fundamentals.get("profit_margins") is not None: parts.append(f"Margin={fundamentals['profit_margins']*100:.2f}%")
        if fundamentals.get("return_on_equity") is not None: parts.append(f"ROE={fundamentals['return_on_equity']*100:.2f}%")
        if parts: prompt += " " + ", ".join(parts) + "\n"
    if trading_paused:
        prompt += "\n**PAUSED:** BUY=notify only (not executed). SELL=executed if market open.\n"
    if performance:
        stock_perf = _round_floats(performance.get("stock_performance", {}).get(symbol, {}))
        equity = _round_floats(performance.get("equity_curve", {}))
        strategy_perf = _round_floats(performance.get("strategy_performance", {}))
        parts = []
        if stock_perf:
            parts.append(f"StockPerf={_to_toon(stock_perf)}")
        if equity:
            parts.append(f"Equity={_to_toon(equity)}")
        if strategy_perf:
            parts.append(f"StratPerf={_to_toon(strategy_perf)}")
        if parts:
            prompt += "\n" + " | ".join(parts) + "\n"
        daily_pnl = equity.get("daily_pnl", 0.0)
        total_pnl = equity.get("total_pnl", 0.0)
        consecutive_losses = equity.get("consecutive_losses", 0)
        prompt += f"P&L: Today={daily_pnl:.0f},Total={total_pnl:.0f}"
        if consecutive_losses > 0:
            prompt += f",⚠️{consecutive_losses} consec losses"
        prompt += "\n"
    # --- Trade pattern analysis ---
    if trade_pattern_analysis:
        prompt += "\n" + _format_trade_pattern_analysis(trade_pattern_analysis) + "\n"
    if max_hold_expired:
        prompt += (
            f"\n**⏰ MAX HOLD EXPIRED (#{max_hold_expired_count}):** SELL now OR HOLD with new `max_hold_time_seconds`. "
            "HOLD without new max_hold_time = auto-sell.\n"
        )
    if stop_loss_triggered:
        prompt += (
            f"\n**⛔ SL TRIGGERED ({stop_loss_review_count}/{max_stop_loss_reviews}):** SELL or HOLD with new lower SL. "
            "HOLD without new SL = auto-sell.\n"
        )
    if take_profit_triggered:
        prompt += (
            f"\n**🎯 TP TRIGGERED ({take_profit_review_count}/{max_take_profit_reviews}):** SELL or HOLD with new higher TP. "
            "HOLD without new TP = auto-sell.\n"
        )
    # --- Partial take-profit triggered ---
    if partial_tp_triggered:
        levels_str = ",".join(str(i) for i in partial_tp_triggered_levels) if partial_tp_triggered_levels else "?"
        prompt += (
            f"\n**🔸 PARTIAL TP ({partial_tp_review_count}/{max_partial_tp_reviews}) L{levels_str}:** "
            "HOLD=execute planned partials, or update `partial_take_profit_levels`, or SELL all.\n"
        )
    # --- Dust sweep triggered ---
    if dust_sweep_triggered:
        prompt += (
            f"\n**🧹 DUST ({dust_sweep_review_count}/{max_dust_sweep_reviews}):** SELL to sweep dust or HOLD to keep.\n"
        )
    if symbol_event and symbol_event.get("has_event"):
        prompt += f"\n**⚠️ Event({symbol}):** {', '.join(symbol_event.get('event_types', []))} [{', '.join(symbol_event.get('keywords', [])[:5])}]\n"
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
        '- "reasoning": EXTREMELY short (max 50 chars). Use abbreviations/symbols only (e.g., "RSI<30+MACD↑"). NO full sentences. NO price.\n'
        '- "strategy_direction": a short string describing your intended strategy approach '
        '(e.g., "momentum_breakout", "mean_reversion", "trend_following", "range_trading", "hold")\n\n'
        "Focus ONLY on analyzing the market data and deciding the direction. "
        "Do NOT output trading parameters, backtest variants, or entry conditions — "
        "those will be requested in the next step.\n\n"
        "Output ONLY the raw JSON object."
    )

    return full_prompt + analysis_output


def build_strategy_messages(data: StrategyPromptData) -> List[Dict[str, str]]:
    """Build a list of messages (system + user) for prompt caching."""
    return [
        {"role": "system", "content": compact_prompt(build_system_prompt(task_type="trading"))},
        {"role": "user", "content": compact_prompt(build_strategy_prompt(data))},
    ]


def build_analysis_messages(data: StrategyPromptData) -> List[Dict[str, str]]:
    """Build a list of messages (system + user) for prompt caching."""
    return [
        {"role": "system", "content": compact_prompt(build_system_prompt(task_type="trading"))},
        {"role": "user", "content": compact_prompt(build_analysis_prompt(data))},
    ]


