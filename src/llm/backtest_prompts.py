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
        prompt += (
            f"\n**Transaction Cost Break-Even — Example using full available balance:**\n"
            f"  If you use the full available balance (~{trade_value:.2f} {base_currency}) for this trade:\n"
            f"  Total round-trip fees: {total_fees:.2f} {base_currency} ({break_even_pct*100:.2f}%)\n"
            f"  Your take_profit_pct MUST be > {break_even_pct*100:.2f}% to be profitable.\n"
            f"  **You are NOT limited to the per-symbol budget.** You may use up to the full remaining balance for this single trade if your conviction is high.\n"
        )

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

    prompt += (
        "\n**CRITICAL — Fallback Parameters for ATR Methods:**\n"
        "If you set `stop_loss_method: \"atr_multiple\"`, you MUST also include `stop_loss_pct` as a fallback "
        "(set it to the estimated ATR-based stop percentage: `stop_loss_atr_multiple × ATR / current_price`). "
        "Similarly, if you use `take_profit_atr_multiple`, you MUST also include `take_profit_pct` as a fallback. "
        "If you omit these fallbacks, the validator will reject your signal.\n"
    )
    prompt += f"""
**Backtest Entry Logic (REQUIRED):**
You MUST include a `backtest_entry_config` object in EVERY backtest variant. If omitted, the backtest will fail with an error and no results will be produced.
Supported fields: ema_period, ema_direction, min_adx, max_rsi, min_rsi, macd_filter, logic.
Example: {{"backtest_entry_config": {{"ema_period": 21, "ema_direction": "above", "min_adx": 25, "max_rsi": 65, "macd_filter": "positive", "logic": "and"}}}}

**Validator Constraints:**
- Minimum max_hold_time_seconds for {assigned_timeframe}: {validator_min} seconds
- Minimum stop-loss: {min_stop_atr_mult} × ATR% (if ATR available)
- take_profit_pct MUST be strictly greater than stop_loss_pct

**Output ONLY the raw JSON object as specified.**

Return a JSON object with these **required** fields:
- `action`: one of BUY, SELL, HOLD (should match your Step 1a analysis)
- `confidence`: a float between 0.0 and 1.0 (should match your Step 1a analysis)
- `reasoning`: a string explaining your parameter choices. You MUST include the current market price.
- `strategy`: an object containing `type` (string) and `parameters` (object).
  The `parameters` object MUST include ALL required trading parameters:
  `stop_loss_pct`, `take_profit_pct`, `position_size_fraction`, `confidence_sizing_weight`,
  `trailing_stop`, `max_hold_time_seconds`, `cooldown_after_loss_seconds`, `backtest_period_days`,
  and `backtest_entry_config` (the same entry logic object you use in your backtest variants — REQUIRED for BUY actions), etc.
- `backtest_variants`: a JSON array of objects, each containing a complete set of strategy parameters
  for backtesting. Each variant MUST include at minimum: `stop_loss_pct`, `take_profit_pct`,
  `max_hold_time_seconds`, `trailing_stop`, `position_size_fraction`, and `backtest_period_days`.
  You decide how many variants to return (minimum 1, recommended 3–5, maximum {_settings.MAX_BACKTEST_VARIANTS}).
  Each variant should explore a different hypothesis based on your Step 1a analysis.
- `entry_condition`: REQUIRED for every BUY action. An object specifying the exact moment to enter.
- `limit_price`: optional, a specific limit price for the order.
- `time_in_force`: optional, "day" or "gtc". Default "day".
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
            f"Parameters: {json.dumps(variant_params, indent=2)}\n"
            f"{fallback_warning}"
            f"Summary: {bt_summary}\n"
            f"Full statistics: {json.dumps(bt_stats, indent=2)}\n"
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
- Proposed Strategy Parameters: {json.dumps(preliminary_decision.get("strategy_params", {}), indent=2)}

**Local Python Backtest Results ({len(backtest_results)} variant(s) tested):**
{all_backtests_text}

You have received the results of ALL {len(backtest_results)} backtest variant(s) above. 
Compare the variants and choose the best-performing one (or combine insights from multiple variants) to inform your final decision.
If ALL backtests show poor performance (e.g., negative total P&L, low win rate, high drawdown), you should reconsider and likely output HOLD or adjust your parameters.
If ANY backtest variant confirms a strategy is viable, you may output your final action (BUY, SELL, or HOLD) using the best-performing variant's parameters.
**Benchmark:** The backtest results include a `buy_and_hold_pct` field, which represents the return of simply buying and holding the asset over the same period. If your strategy's `total_pnl_pct` is lower than `buy_and_hold_pct`, it means your active trading strategy is worse than doing nothing. Only proceed with a BUY if your strategy is better than buy-and-hold or if it significantly reduces drawdown.
"""
    prompt += (
        f"\n**Backtest Period:** The backtests were run using historical data on the {preliminary_decision.get('timeframe', 'assigned')} timeframe. "
        f"Each variant may have used a different `backtest_period_days` value (see individual variant parameters above).\n"
    )
    if total_variants_proposed is not None and total_variants_proposed > len(backtest_results):
        prompt += (
            f"\n**Note:** You proposed {total_variants_proposed} backtest variants in Step 1, but only the first "
            f"{len(backtest_results)} were tested (maximum {settings.MAX_BACKTEST_VARIANTS} variants per cycle). The results above cover all "
            f"tested variants. To avoid truncation in future cycles, limit your `backtest_variants` array to at most {settings.MAX_BACKTEST_VARIANTS} entries.\n"
        )
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
        "**Output ONLY the raw JSON object as specified.**\n"
        "Return a JSON object with these **required** fields:\n"
        "- `action`: one of BUY, SELL, HOLD\n"
        "- `confidence`: a float between 0.0 and 1.0\n"
        "- `reasoning`: a string explaining your final decision, specifically referencing the backtest results. "
        "You MUST also include the decided price (the current market price or your specified `limit_price`) in the reasoning message.\n"
        "- `strategy`: an object containing `type` and `parameters`.\n"
        "  The `parameters` object MUST include ALL required trading parameters (same as Step 1):\n"
        "  `stop_loss_pct`, `take_profit_pct`, `position_size_fraction`, `confidence_sizing_weight`, `trailing_stop`, `max_hold_time_seconds`, `cooldown_after_loss_seconds`,\n"
        "  and `backtest_entry_config` (REQUIRED for BUY actions — copy it from your best-performing backtest variant), etc.\n"
        "  You may adjust `position_size_fraction` based on backtest performance (e.g., reduce size if drawdown is high).\n"
    )
    prompt += (
        "\nIf your final action is BUY, you MUST also include an `entry_condition` object (same format as Step 1).\n"
        "You may also include `order_type`, `limit_price`, `time_in_force`, and other execution parameters.\n"
    )
    if trading_paused:
        prompt += (
            "\n**Trading is currently PAUSED.** You can still output BUY, SELL, or HOLD actions. "
            "However, any BUY signals will NOT be executed; they will only be sent as notifications. "
            "SELL signals for existing positions will be executed normally if the market is open. "
            "Please continue to analyze the market and generate signals as you normally would.\n"
        )
    return prompt
