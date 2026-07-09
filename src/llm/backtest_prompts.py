import json
import logging
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from src.config.settings import settings
from src.utils.symbol_utils import is_btp_isin
from src.llm.prompt_utils import _timeframe_to_seconds

logger = logging.getLogger(__name__)


@dataclass
class BacktestPromptData:
    symbol: str
    analysis: Dict[str, Any]
    ticker: Dict[str, Any]
    current_price: float
    atr: Optional[float]
    assigned_timeframe: str
    base_currency: str
    base_balance: float
    per_symbol_budget: float
    min_order_amount: Optional[float] = None
    min_order_cost: Optional[float] = None
    remaining_balance: Optional[float] = None
    portfolio_total_value: Optional[float] = None
    portfolio_exposure_pct: Optional[float] = None
    portfolio_stop_risk_pct: Optional[float] = None
    portfolio_available_capital: Optional[float] = None
    max_portfolio_exposure_pct: Optional[float] = None
    max_portfolio_stop_risk_pct: Optional[float] = None
    global_risk_multiplier: Optional[float] = None
    min_stop_atr_mult: float = 1.0
    min_hold_time_mult: float = 1.0
    trading_paused: bool = False
    has_position: bool = False
    historical_backtest_results: Optional[List[Dict[str, Any]]] = None


def build_backtest_variants_prompt(data: BacktestPromptData) -> str:
    """Build a focused prompt for Step 1b: Parameter selection and backtest variants.

    Given the analysis from Step 1a, asks the LLM to propose backtest variants
    with full parameters, entry conditions, and preliminary strategy parameters.
    The LLM does not need to re-analyze the market — it translates its analysis
    into concrete trading parameters.
    """
    from src.config.settings import settings as _settings
    import re as _re
    import time as _time

    symbol = data.symbol
    analysis = data.analysis
    current_price = data.current_price
    atr = data.atr
    assigned_timeframe = data.assigned_timeframe
    base_currency = data.base_currency
    base_balance = data.base_balance
    per_symbol_budget = data.per_symbol_budget
    remaining_balance = data.remaining_balance
    portfolio_total_value = data.portfolio_total_value
    portfolio_exposure_pct = data.portfolio_exposure_pct
    portfolio_stop_risk_pct = data.portfolio_stop_risk_pct
    portfolio_available_capital = data.portfolio_available_capital
    max_portfolio_exposure_pct = data.max_portfolio_exposure_pct
    max_portfolio_stop_risk_pct = data.max_portfolio_stop_risk_pct
    global_risk_multiplier = data.global_risk_multiplier
    min_stop_atr_mult = data.min_stop_atr_mult
    min_hold_time_mult = data.min_hold_time_mult
    trading_paused = data.trading_paused
    has_position = data.has_position
    historical_backtest_results = data.historical_backtest_results

    tf_seconds = _timeframe_to_seconds(assigned_timeframe)

    # Cap validator minimum for long timeframes
    if assigned_timeframe in ("3Y", "5Y"):
        validator_min = 31_536_000
    elif assigned_timeframe in ("1Y", "6M"):
        validator_min = min(int(min_hold_time_mult * tf_seconds), 31_536_000)
    else:
        validator_min = int(min_hold_time_mult * tf_seconds)

    # Detect BTP for fee calculation
    _is_btp = is_btp_isin(symbol)
    # Use the full remaining balance (or total balance) for fee break-even calculation
    trade_value = remaining_balance if remaining_balance is not None and remaining_balance > 0 else base_balance

    prompt = f"""**Step 1b: Parameter Selection & Backtest Variants**

Symbol: {symbol}
Current price: {current_price}
Assigned timeframe: {assigned_timeframe}
Base currency: {base_currency}

**Your Step 1a Analysis (you already made this decision):**
- Action: {analysis.get("action", "HOLD")}
- Confidence: {analysis.get("confidence", 0.0)}
- Reasoning: {analysis.get("reasoning", "")}
- Strategy Direction: {analysis.get("strategy_direction", "unknown")}

**Key Market Context for Parameter Selection:**
- ATR (14-period, {assigned_timeframe}): {f"{atr:.6f}" if atr is not None else "N/A"}
"""
    if atr is not None and current_price and current_price > 0:
        atr_pct = atr / current_price
        min_sl = min_stop_atr_mult * atr_pct
        prompt += f"- ATR%: {atr_pct:.4%}\n"
        prompt += f"- Minimum stop-loss (validator enforces): {min_sl:.4%} ({min_stop_atr_mult} × ATR%)\n"

    prompt += f"- Total {base_currency} balance: {base_balance:.2f}\n"
    prompt += f"- Suggested per-symbol budget: {per_symbol_budget:.2f} {base_currency}\n"
    if remaining_balance is not None:
        prompt += f"- Remaining available for this symbol: {remaining_balance:.2f} {base_currency}\n"
    if portfolio_total_value is not None:
        prompt += f"- Total portfolio value: {portfolio_total_value:.2f} {base_currency}\n"
    if portfolio_exposure_pct is not None:
        prompt += f"- Current capital deployed: {portfolio_exposure_pct:.1f}%\n"
    if portfolio_stop_risk_pct is not None:
        prompt += f"- Total stop-loss risk: {portfolio_stop_risk_pct:.2f}%\n"
    if portfolio_available_capital is not None:
        prompt += f"- Available capital: {portfolio_available_capital:.2f} {base_currency}\n"
    if max_portfolio_exposure_pct is not None:
        prompt += f"- Max portfolio exposure: {max_portfolio_exposure_pct*100:.0f}%\n"
    if max_portfolio_stop_risk_pct is not None:
        prompt += f"- Max portfolio stop risk: {max_portfolio_stop_risk_pct*100:.0f}%\n"
    if global_risk_multiplier is not None and global_risk_multiplier < 1.0:
        prompt += f"- Global risk multiplier: {global_risk_multiplier}\n"
    if data.min_order_amount is not None:
        prompt += f"- Min order amount: {data.min_order_amount}\n"
    if data.min_order_cost is not None:
        prompt += f"- Min order cost: {data.min_order_cost:.2f} {base_currency}\n"

    # Transaction cost break-even
    if trade_value > 0:
        if _is_btp:
            if _settings.BTP_IS_PRIMARY_ISSUANCE:
                buy_fee = 0.0
                sell_fee = 0.0
            else:
                buy_fee = max(_settings.BTP_MIN_FEE, trade_value * _settings.BTP_FEE_PERC)
                sell_fee = max(_settings.BTP_MIN_FEE, trade_value * _settings.BTP_FEE_PERC)
        else:
            buy_fee = max(_settings.STOCK_FEE_MIN, trade_value * _settings.STOCK_FEE_PERC) + _settings.STOCK_FEE_FIXED + (trade_value * _settings.TOBIN_TAX_RATE)
            sell_fee = max(_settings.STOCK_FEE_MIN, trade_value * _settings.STOCK_FEE_PERC) + _settings.STOCK_FEE_FIXED
        total_fees = buy_fee + sell_fee
        break_even_pct = total_fees / trade_value
        prompt += f"\n**Fees:** Round-trip={total_fees:.2f} ({break_even_pct*100:.2f}%). TP must be > {break_even_pct*100:.2f}%. You may use full remaining balance.\n"

    if historical_backtest_results:
        prompt += "\n**Historical Backtest Results (learn from past tests):**\n"
        for bt in historical_backtest_results[:5]:
            stats = bt.get("stats", {})
            params = bt.get("variant_params", {})
            prompt += (
                f"  SL={params.get('stop_loss_pct', '?')}, TP={params.get('take_profit_pct', '?')}, "
                f"trades={stats.get('total_trades', 0)}, win_rate={stats.get('win_rate', 0)*100:.1f}%, "
                f"total_pnl={stats.get('total_pnl_pct', 0)*100:+.2f}%, "
                f"max_dd={stats.get('max_drawdown_pct', 0)*100:.2f}%\n"
            )
        prompt += "Avoid repeating failed combinations. Prefer parameters similar to historically profitable ones.\n\n"

    prompt += "\n**ATR fallback:** Always include `stop_loss_pct` and `take_profit_pct` even if using ATR methods. Set to estimated ATR-based %.\n"
    prompt += "\n**Backtest Entry Config (REQUIRED):** Include `backtest_entry_config` in every variant. Fields: ema_period,ema_direction,min_adx,max_rsi,min_rsi,macd_filter,logic. Example: {\"backtest_entry_config\":{\"ema_period\":21,\"ema_direction\":\"above\",\"min_adx\":25,\"max_rsi\":65,\"macd_filter\":\"positive\",\"logic\":\"and\"}}\n"
    prompt += f"\n**Constraints:** min max_hold={validator_min}s, min SL={min_stop_atr_mult}×ATR%, TP>SL.\n"
    prompt += f"""
Return JSON:
- action: BUY|SELL|HOLD
- confidence: float 0-1
- reasoning: str max 80 chars
- strategy: {{type, parameters{{stop_loss_pct,take_profit_pct,position_size_fraction,confidence_sizing_weight,trailing_stop,max_hold_time_seconds,cooldown_after_loss_seconds,backtest_period_days,backtest_entry_config(REQUIRED for BUY)}}}}
- backtest_variants: array of objects with same params (min 1, max {_settings.MAX_BACKTEST_VARIANTS})
- entry_condition: object (REQUIRED for BUY)
- limit_price: float? (optional)
- time_in_force: "day"|"gtc"? (optional)
"""
    if has_position:
        prompt += (
            "\n**You currently hold a position in this symbol.** "
            "If you output BUY, you will ADD to the existing position (scale in). "
            "If you output SELL, you will close the ENTIRE position.\n"
        )
    if trading_paused:
        prompt += (
            "\n**Trading is currently PAUSED.** BUY signals will NOT be executed; "
            "they will only be sent as notifications. SELL signals for existing positions "
            "will be executed normally if the market is open.\n"
        )
    return prompt


def build_final_decision_prompt(
    symbol: str,
    ticker: Dict[str, Any],
    preliminary_decision: Dict[str, Any],
    backtest_results: List[Dict[str, Any]],
    base_currency: str,
    trading_paused: bool = False,
    total_variants_proposed: Optional[int] = None,
    historical_backtest_results: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Build a prompt to ask the LLM for its final decision after reviewing backtest results."""
    current_price = ticker.get("last") if ticker else None

    # Build a combined backtest results section showing ALL variants
    backtest_sections = []
    for i, bt_result in enumerate(backtest_results):
        variant_params = bt_result.get("variant_params", {})
        bt_summary = bt_result.get("summary", "No backtest summary available.")
        bt_stats = bt_result.get("stats", {})

        # Explicitly highlight timeframe fallback if it occurred
        fallback_warning = ""
        actual_tf = bt_stats.get("actual_timeframe")
        assigned_tf = bt_stats.get("assigned_timeframe")
        if actual_tf and assigned_tf and actual_tf != assigned_tf:
            fallback_warning = (
                f"⚠️ TIMEFRAME FALLBACK: This backtest was run on {actual_tf} candles, NOT the assigned {assigned_tf} timeframe. "
                f"Results may not accurately represent {assigned_tf} behavior — treat with caution.\n"
            )

        backtest_sections.append(
            f"**Variant {i+1}:**\n"
            f"Parameters: {json.dumps(variant_params, separators=(',', ':'))}\n"
            f"{fallback_warning}"
            f"Summary: {bt_summary}\n"
            f"Full statistics: {json.dumps(bt_stats, separators=(',', ':'))}\n"
        )
    all_backtests_text = "\n".join(backtest_sections)

    prompt = f"""**Step 2: Final Trading Decision**

Symbol: {symbol}
Current price: {current_price}
Base currency: {base_currency}

**Your Step 1 Preliminary Decision:**
- Preliminary Action: {preliminary_decision.get("action", "HOLD")}
- Confidence: {preliminary_decision.get("confidence", 0.0)}
- Reasoning: {preliminary_decision.get("reasoning", "")}
- Proposed Strategy Parameters: {json.dumps(preliminary_decision.get("strategy_params", {}), separators=(',', ':'))}

**Local Python Backtest Results ({len(backtest_results)} variant(s) tested):**
{all_backtests_text}

Compare variants. Choose best-performing or combine insights. If all poor, output HOLD. Benchmark: buy_and_hold_pct. Only BUY if strategy beats buy-and-hold or reduces drawdown.
"""
    prompt += f"\nBacktests on {preliminary_decision.get('timeframe', 'assigned')} timeframe, varying periods.\n"
    if total_variants_proposed is not None and total_variants_proposed > len(backtest_results):
        prompt += f"\nProposed {total_variants_proposed} variants, only {len(backtest_results)} tested (max {settings.MAX_BACKTEST_VARIANTS}).\n"
    if historical_backtest_results:
        prompt += "\n**Historical Backtest Results for this symbol (past tests):**\n"
        for bt in historical_backtest_results[:5]:
            stats = bt.get("stats", {})
            params = bt.get("variant_params", {})
            prompt += (
                f"  SL={params.get('stop_loss_pct', '?')}, TP={params.get('take_profit_pct', '?')}, "
                f"trades={stats.get('total_trades', 0)}, win_rate={stats.get('win_rate', 0)*100:.1f}%, "
                f"total_pnl={stats.get('total_pnl_pct', 0)*100:+.2f}%, "
                f"max_dd={stats.get('max_drawdown_pct', 0)*100:.2f}%\n"
            )
        prompt += "Consider these historical results when making your final decision.\n"
    prompt += (
        "Return JSON:\n"
        "- action: BUY|SELL|HOLD\n"
        "- confidence: float 0-1\n"
        "- reasoning: str max 80 chars\n"
        "- strategy: {type, parameters{stop_loss_pct,take_profit_pct,position_size_fraction,confidence_sizing_weight,trailing_stop,max_hold_time_seconds,cooldown_after_loss_seconds,backtest_entry_config(REQUIRED for BUY)}}\n"
        "- entry_condition: object (REQUIRED for BUY)\n"
        "- limit_price: float? (optional)\n"
        "- time_in_force: \"day\"|\"gtc\"? (optional)\n"
    )
    if trading_paused:
        prompt += (
            "\n**Trading is currently PAUSED.** You can still output BUY, SELL, or HOLD actions. "
            "However, any BUY signals will NOT be executed; they will only be sent as notifications. "
            "SELL signals for existing positions will be executed normally if the market is open. "
            "Please continue to analyze the market and generate signals as you normally would.\n"
        )
    return prompt


def build_backtest_variants_messages(data: BacktestPromptData) -> List[Dict[str, str]]:
    """Build a list of messages (system + user) for prompt caching."""
    from src.llm.system_prompt import build_system_prompt
    from src.llm.prompt_utils import compact_prompt
    return [
        {"role": "system", "content": compact_prompt(build_system_prompt(task_type="trading"))},
        {"role": "user", "content": compact_prompt(build_backtest_variants_prompt(data))},
    ]


def build_final_decision_messages(
    symbol: str,
    ticker: Dict[str, Any],
    preliminary_decision: Dict[str, Any],
    backtest_results: List[Dict[str, Any]],
    base_currency: str,
    trading_paused: bool = False,
    total_variants_proposed: Optional[int] = None,
    historical_backtest_results: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, str]]:
    """Build a list of messages (system + user) for prompt caching."""
    from src.llm.system_prompt import build_system_prompt
    from src.llm.prompt_utils import compact_prompt
    return [
        {"role": "system", "content": compact_prompt(build_system_prompt(task_type="trading"))},
        {"role": "user", "content": compact_prompt(build_final_decision_prompt(
            symbol=symbol,
            ticker=ticker,
            preliminary_decision=preliminary_decision,
            backtest_results=backtest_results,
            base_currency=base_currency,
            trading_paused=trading_paused,
            total_variants_proposed=total_variants_proposed,
            historical_backtest_results=historical_backtest_results,
        ))},
    ]
