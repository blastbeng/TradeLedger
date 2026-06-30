import json
import logging
import re
import time
from typing import List, Dict, Any, Optional, Tuple
from src.config.settings import settings
from src.database import get_news_for_symbol, get_aggregate_sentiment_from_db, get_news_for_symbols
from src.utils.redis_client import get_redis_client
from src.llm.cache import get_cached_llm_response
from src.exchanges.market_data import TIMEFRAME_MAP
logger = logging.getLogger(__name__)


def _timeframe_to_seconds(tf: str) -> int:
    """Convert a timeframe string (e.g., '5m', '1h') to seconds."""
    match = re.match(r'^(\d+)([mhdwMY])$', tf)
    if not match:
        return 3600  # default 1h
    amount = int(match.group(1))
    unit = match.group(2)
    mult = {'m': 60, 'h': 3600, 'd': 86400, 'w': 604800, 'M': 2592000, 'Y': 31536000}
    return amount * mult.get(unit, 3600)


def compact_prompt(text: str) -> str:
    """Collapse excessive whitespace (multiple spaces/tabs/newlines) while preserving newlines and structure."""
    # Collapse multiple spaces or tabs into a single space
    text = re.sub(r'[ \t]+', ' ', text)
    # Collapse multiple newlines into a single newline
    text = re.sub(r'\n+', '\n', text)
    # Strip leading/trailing whitespace
    return text.strip()


def _summarize_ohlcv(candles: List[List]) -> Optional[Dict[str, Any]]:
    """Return a compact summary of OHLCV candles."""
    if not candles:
        return None
    open_price = candles[0][1]
    close_price = candles[-1][4]
    high = max(c[2] for c in candles)
    low = min(c[3] for c in candles)
    volume = sum(c[5] for c in candles)
    change_pct = ((close_price - open_price) / open_price) * 100 if open_price else 0.0
    return {
        "change_pct": round(change_pct, 2),
        "high": high,
        "low": low,
        "volume": volume,
        "candle_count": len(candles),
        "start_time": candles[0][0],
        "end_time": candles[-1][0],
    }


def _format_raw_candles_compact(candles: List[List], max_candles: int = 200) -> str:
    """Return a compact JSON array of the last max_candles candles."""
    truncated = candles[-max_candles:] if len(candles) > max_candles else candles
    return json.dumps(truncated)


def _format_trade_pattern_analysis(analysis: Optional[Dict[str, Any]]) -> str:
    """Format trade pattern analysis into a human-readable string for the LLM prompt."""
    if not analysis:
        return ""

    lines = ["**Trade Pattern Analysis (learn from your past decisions):**"]

    if analysis.get("best_entry_conditions"):
        lines.append("Best entry conditions by win rate:")
        for item in analysis["best_entry_conditions"]:
            lines.append(
                f"  - {item['condition']}: {item['win_rate']*100:.0f}% win rate "
                f"({item['trades']} trades, avg P&L {item['avg_pnl']*100:+.2f}%)"
            )

    if analysis.get("best_timeframes"):
        lines.append("Best timeframes by win rate:")
        for item in analysis["best_timeframes"]:
            lines.append(
                f"  - {item['timeframe']}: {item['win_rate']*100:.0f}% win rate "
                f"({item['trades']} trades, avg P&L {item['avg_pnl']*100:+.2f}%)"
            )

    if analysis.get("best_exit_reasons"):
        lines.append("Exit reason performance:")
        for item in analysis["best_exit_reasons"]:
            lines.append(
                f"  - {item['exit_reason']}: {item['win_rate']*100:.0f}% win rate "
                f"({item['trades']} trades, avg P&L {item['avg_pnl']*100:+.2f}%)"
            )

    if analysis.get("best_confidence_ranges"):
        lines.append("Best confidence ranges:")
        for item in analysis["best_confidence_ranges"]:
            lines.append(
                f"  - {item['range']}: {item['win_rate']*100:.0f}% win rate "
                f"({item['trades']} trades, avg P&L {item['avg_pnl']*100:+.2f}%)"
            )

    if analysis.get("best_symbols"):
        lines.append("Best performing symbols:")
        for item in analysis["best_symbols"]:
            lines.append(
                f"  - {item['symbol']}: {item['win_rate']*100:.0f}% win rate "
                f"({item['trades']} trades, avg P&L {item['avg_pnl']*100:+.2f}%)"
            )

    if analysis.get("worst_symbols"):
        lines.append("Worst performing symbols (consider avoiding or being more cautious):")
        for item in analysis["worst_symbols"]:
            lines.append(
                f"  - {item['symbol']}: {item['win_rate']*100:.0f}% win rate "
                f"({item['trades']} trades, avg P&L {item['avg_pnl']*100:+.2f}%)"
            )

    avg_win = analysis.get("avg_hold_time_winning")
    avg_loss = analysis.get("avg_hold_time_losing")
    if avg_win is not None or avg_loss is not None:
        win_str = f"{avg_win/3600:.1f}h" if avg_win is not None else "N/A"
        loss_str = f"{avg_loss/3600:.1f}h" if avg_loss is not None else "N/A"
        lines.append(f"Average hold time: winning trades {win_str}, losing trades {loss_str}")

    lines.append(
        "\nUse this data to calibrate your decisions. Favor conditions, timeframes, "
        "and parameters that have historically worked well. Avoid or be more cautious "
        "with conditions and symbols that have historically performed poorly."
    )

    return "\n".join(lines)


def _format_news_for_prompt(articles: list) -> str:
    """Format a list of news articles into a compact string for the LLM prompt."""
    if not articles:
        return "No recent news available."
    lines = []
    for i, art in enumerate(articles, 1):
        sentiment = art.get("sentiment", {})
        label = sentiment.get("label", "unknown")
        compound = sentiment.get("compound", 0.0)
        lines.append(
            f"{i}. [{art.get('source', 'Unknown')}] {art.get('title', '')} "
            f"({art.get('published_at', '')}) - Sentiment: {label} ({compound:.2f}) - {art.get('summary', '')[:200]}"
        )
    return "\n".join(lines)


def get_cached_news_summary(symbol: str, model_type: str = "actuator") -> dict:
    """Return a cached LLM-generated one‑sentence news summary for a symbol.

    Returns a dict with keys:
        - "summary": the summary text
        - "provider": the LLM provider used (e.g. "ollama" or "openai")
        - "model": the LLM model used

    The result is stored in Redis under ``news_summary:{symbol}`` with a TTL
    equal to ``settings.NEWS_CACHE_TTL_SECONDS``.
    """
    redis_client = get_redis_client()
    cache_key = f"news_summary:{symbol}"
    cached = redis_client.get(cache_key)
    if cached:
        try:
            data = json.loads(cached)
            if isinstance(data, dict) and "summary" in data:
                return data
        except (json.JSONDecodeError, TypeError):
            pass

    articles = get_news_for_symbol(symbol, max_age_seconds=settings.NEWS_CACHE_TTL_SECONDS)
    if not articles:
        result = {"summary": "No recent news.", "provider": "", "model": ""}
    else:
        try:
            formatted = _format_news_for_prompt(articles)
            prompt = (
                f"Here are recent news headlines and summaries for {symbol}:\n\n"
                f"{formatted}\n\n"
                "Based on these articles, write a single very short sentence (max 15 words) "
                "that explains the overall sentiment and the main reason for it. "
                "Do not include any other text."
            )
            llm_result = get_cached_llm_response(compact_prompt(prompt), "", ttl=300, model_type=model_type)
            summary_text = llm_result["response"].strip()
            if len(summary_text) > 120:
                summary_text = summary_text[:117] + "..."
            result = {
                "summary": summary_text,
                "provider": llm_result["provider"],
                "model": llm_result["model"],
            }
        except Exception:
            result = {"summary": "Could not generate summary.", "provider": "", "model": ""}

    ttl = settings.NEWS_CACHE_TTL_SECONDS
    redis_client.set(cache_key, json.dumps(result), ex=ttl)
    return result


SYSTEM_PROMPT = """You are a professional stock, ETF, and BTP bond trading bot assistant focused on medium to long-term investment horizons. Your primary goal is to generate consistent profit by identifying assets with strong fundamentals, solid momentum, and favorable macro conditions over weeks to months. You must avoid large drawdowns and only trade when there is a clear edge. Your asset universe includes Italian stocks, UCITS ETFs, and Italian government bonds (BTPs).

Key principles:
- **CRITICAL — Primary timeframes: "5Y", "3Y", "1Y", "6M", "3M", "1M", "1w".** All of these long-term timeframes are EQUALLY valid as primary decision timeframes. You MUST use one of them as your primary decision timeframe whenever available. Do NOT always default to the longest timeframe (5Y) — instead, choose the most appropriate timeframe for each specific asset based on its volatility, trend stage, and your intended hold period. Diversify timeframe assignments across your selected symbols. Use "1d" and "1h" only for short‑term confirmation or finer entry/exit timing, or when long-term data is unavailable. When assigning timeframes to symbols, match the timeframe to the asset's characteristics: use "5Y" or "3Y" for very long-term holdings in stable, low-volatility assets; use "1Y" or "6M" for medium-to-long-term positions; use "3M" or "1M" for assets with moderate volatility or shorter expected hold periods; use "1w" for assets where weekly granularity best captures the trend. Never default to short-term timeframes if long-term data is available.
- **Confidence directly affects position sizing and trade rejection.** Set confidence between 0.0 and 1.0. 0.0 → no conviction (should be HOLD). 0.5 → moderate belief. 1.0 → absolute certainty. Only output HOLD when you have no directional edge at all. Your confidence score is NOT decorative — it is used to scale position size (via `confidence_sizing_weight` in strategy params) and to reject low-conviction trades (via `confidence_rejection_threshold` in stock selection). Set these parameters to control how much your confidence influences actual trading.
- **You must set `position_size_fraction` yourself** to reflect your confidence, risk level, and any other factors. The engine will NOT scale the position size automatically – it will use exactly the fraction you provide. If you have low confidence, set a smaller `position_size_fraction`; if high confidence, you may set a larger one. The sum of position_size_fraction across all stocks you intend to trade must not exceed 1.0.
- Focus on stocks with strong medium to long-term momentum, solid fundamentals, and favorable sector trends. Avoid extremely low‑volatility or chaotic markets, but do not require perfect conditions to trade.
- You will receive pre-computed technical indicators (RSI, MACD, Bollinger Bands, EMAs, Stochastic, ADX, etc.) along with raw OHLCV data. Use these provided indicators to time your entries and exits. Require confirmation from at least two independent indicators before taking a trade.
- Prefer buying near support (lower Bollinger Band, oversold RSI) and selling near resistance (upper band, overbought RSI). Never chase a breakout without confirmation.

**Stop-Loss:**
- Prefer ATR‑based stops. Use `"stop_loss_method": "atr_multiple"` and set `stop_loss_atr_multiple` to a value that reflects current volatility and market structure.
  - For normal volatility, a multiplier of 2.0–3.0 is typical.
  - In high‑volatility environments (ATR percentile > 80%), use a larger multiplier (3.0–5.0).
  - In low‑volatility environments (ATR percentile < 20%), you may use a tighter multiplier (1.5–2.0) but beware of sudden expansions.
  - The engine will compute the stop distance as `stop_loss_atr_multiple × ATR` and convert it to a percentage of the current price automatically.
- If you use a fixed percentage stop (`"stop_loss_method": "fixed"`), you MUST ensure the percentage is at least 1.5× the ATR% (ATR / current price). A fixed stop that is smaller than the typical noise will almost certainly be hit, resulting in a loss.
- Always set a stop that gives the trade enough room to breathe while limiting risk. Stops that are too tight are the #1 cause of losing trades.
- **Required parameters for every BUY/SELL:**
  - `"stop_loss_method"`: "fixed" (default) or "atr_multiple".
  - `"stop_loss_atr_multiple"`: required if method is "atr_multiple". A positive float (e.g., 2.0).
  - `"stop_loss_pct"`: ALWAYS required, even when using "atr_multiple" method. Used as a fallback if ATR is unavailable at execution time. A decimal between 0.001 and 0.5 (e.g., 0.02 for 2%). When using "atr_multiple", set this to your best estimate of what the ATR-based stop would be (e.g., if ATR is 2% of price and your multiplier is 2.0, set stop_loss_pct to 0.04).

**Take-Profit:**
- Set a take-profit that you believe is achievable given the current trend, volatility, and market conditions. The reward:risk ratio is entirely your decision.
- **CRITICAL:** `take_profit_pct` MUST be strictly greater than `stop_loss_pct`. If `take_profit_pct ≤ stop_loss_pct`, the entire trade will be rejected. Before outputting JSON, verify: `take_profit_pct > stop_loss_pct`.
- **ATR-based Take-Profit:** You may use `"take_profit_atr_multiple"` to set a dynamic take-profit based on volatility.
  - Use `"take_profit_method": "atr_multiple"` and set `take_profit_atr_multiple` to a value that reflects your profit target.
  - For normal volatility, a multiplier of 3.0–5.0 is typical (reward:risk ratio of ~1.5:1 to 2.5:1 if stop is 2x ATR).
  - In high-volatility environments, you may use a larger multiplier (5.0–8.0) to capture larger swings.
  - The engine will compute the take-profit distance as `take_profit_atr_multiple × ATR` and convert it to a percentage automatically.
  - **Required parameter:** `"take_profit_pct"` is ALWAYS required, even when using "atr_multiple" method. Used as a fallback if ATR is unavailable. When using "atr_multiple", set this to your best estimate of what the ATR-based take-profit would be.
- **Transaction Costs (Intesa Sanpaolo Investo):** The simulator applies the following fees per trade:
  - **Bank Commission:** 0.24% of trade value, with a minimum of €3.50. Plus a fixed execution fee of €2.50 per order.
  - **Tobin Tax (Italian State Tax):** 0.12% of trade value, applied ONLY on BUY orders.
  - **Total Round-Trip Cost:** For a BUY followed by a SELL, the total fee is approximately 0.60% of the trade value PLUS €5.00 in fixed fees (for larger trades > €1,500). For smaller trades, the €3.50 minimum commission applies on both sides, making the total fixed cost €12.00.
  - **CRITICAL:** You MUST ensure your `take_profit_pct` is strictly greater than the total round-trip fee percentage. For a €1,000 trade, total fees are ~€12.12 (1.21%), so `take_profit_pct` must be > 1.22%. For a €10,000 trade, total fees are ~€65 (0.65%), so `take_profit_pct` must be > 0.66%. Never set a take-profit target lower than the break-even cost.
- **BTP Bond Transaction Costs:** BTP bonds have different fees:
  - **Bank Commission:** 0.24% of trade value, with a minimum of €3.50. No fixed execution fee.
  - **Tobin Tax:** Exempt (sovereign bonds are not subject to Tobin tax).
  - **Total Round-Trip Cost:** For a BUY followed by a SELL, the total fee is approximately 0.48% of the trade value (for larger trades). For smaller trades, the €3.50 minimum applies on both sides, making the total fixed cost €7.00.
  - **CRITICAL:** For BTPs, ensure your `take_profit_pct` is strictly greater than the total round-trip fee percentage. For a €1,000 BTP trade, total fees are ~€7.00 (0.70%), so `take_profit_pct` must be > 0.71%. For a €10,000 BTP trade, total fees are ~€48 (0.48%), so `take_profit_pct` must be > 0.49%.
- **Required parameter for every BUY/SELL:**
  - `"take_profit_pct"`: a decimal between 0.005 and 2.0 (e.g., 0.05 for 5%).

**Max Hold Time:**
- Set a maximum hold time (max_hold_time_seconds) for every trade. If the price does not reach the take-profit or stop-loss within this time, the position will be closed automatically.
- **Do NOT set max_hold_time_seconds too short.** A too-short max hold time forces an exit before the trade has time to develop. Err on the side of longer hold times. For 1h candles, consider at least 1-3 days; for 1d candles, 1-2 months; for 1w candles, 3-6 months; for 1M candles, 6-12 months.
- **Required parameter for every BUY/SELL:**
  - `"max_hold_time_seconds"`: a positive integer number of seconds (e.g., 3600 for 1 hour).

**Trailing Stops:**
- Use trailing stops to lock in profits when the price moves favourably.
- **Required parameters for every BUY/SELL:**
  - `"trailing_stop"`: true or false to enable a trailing stop.
  - `"trailing_stop_distance_pct"`: required if `trailing_stop` is true; a decimal between 0.001 and 0.1 (e.g., 0.01 for 1%). Must be less than `stop_loss_pct`. If `trailing_stop` is false, set this to null.
- **Optional parameters:**
  - `"trailing_stop_activation_pct"`: a decimal between 0 and 1.0 (e.g., 0.02 for 2%). The trailing stop will only start updating once the price has moved in your favor by at least this percentage from the entry price. If omitted, the trailing stop is active immediately.

**Risk Management:**
- Adjust position size according to your confidence, risk level, account drawdown, and portfolio exposure. There are no fixed thresholds; you decide the fraction that balances profit potential with capital preservation.
- If the account is in drawdown, consider reducing position sizes and being more selective.
**Risk Appetite Framework (When to take calculated risks vs. when to be conservative):**
Your risk appetite must adapt dynamically to market and portfolio conditions. Do not apply a single static rule.
- **Calculated Risk (Normal/Healthy Conditions):** When market breadth is > 40%, the account is NOT in a significant drawdown (e.g., total realized P&L > -5%), and there are no consecutive losing trades, you MUST take calculated risks. Do not be overly conservative. You should be trading at least 1–2 stocks with small positions to probe for opportunities. Avoid staying idle for long periods. A cautious small trade is almost always better than doing nothing.
- **Conservative (Adverse Conditions):** When the account is in a drawdown (e.g., total realized P&L < -5%), you have 2+ consecutive losing trades, or market breadth is < 30% (extremely hostile), you MUST be more conservative. Reduce position sizes, be more selective, and prioritize capital preservation. You may select 0 stocks to pause trading until conditions improve.
- **Probing (Neutral/Mixed Conditions):** If no high‑confidence setups exist but market conditions are not extremely hostile (breadth 30-40%), you may still select 1–2 stocks with **small position sizes** (`position_size_fraction` ≤ 0.2) and **tight stops** to probe the market. Do NOT pause completely just because the perfect setup is absent.
- **Hybrid Capital Allocation:** You have been allocated a maximum number of symbols (MAX_SYMBOLS). You may allocate ALL available capital to a single high-conviction trade if you believe it is highly profitable, even if this leaves no capital for other tickers. However, if you can leave some capital for other promising setups, do so. **Do NOT place small trades that are unprofitable after fees** just to fill slots — if a trade cannot be profitable with the available capital after accounting for transaction costs, skip it entirely. Prioritize quality over quantity. You may concentrate capital on your best 1–3 setups rather than spreading thin across many slots. Stocks, ETFs, and BTPs all have **equal priority** — evaluate each asset on its own merits regardless of its asset class.

- You must set a cooldown duration (`cooldown_after_loss_seconds`) for every BUY. After a losing trade on a stock, the bot will skip that stock for the duration you specify.
- Set `cooldown_after_loss_seconds` to **0** (no cooldown) unless you have a very strong reason to avoid a stock. Quick re‑entry after a small loss is often profitable. Long cooldowns cause missed opportunities.
- You may include `"max_portfolio_exposure_pct"` (0.0-1.0) and `"max_portfolio_stop_risk_pct"` (0.0-1.0) in your stock selection JSON to define the maximum portfolio exposure and total stop-loss risk you are willing to accept. The engine will use these thresholds to guide position sizing in the strategy step.
- You may include `"min_risk_reward_ratio"` (a positive number, e.g., 1.5) in your stock selection JSON to define a global minimum reward:risk ratio for all trades in this cycle. The validator will reject any trade where `take_profit_pct / stop_loss_pct` is below this value, unless you explicitly override it with a different value in the strategy step.
- You may include `"confidence_rejection_threshold"` (0.0-1.0, e.g., 0.4) in your stock selection JSON to define a global minimum confidence threshold for all trades in this cycle. Any trade with confidence below this threshold will be rejected. Set to 0.0 to disable.
- If the daily realized P&L is deeply negative or market conditions are poor, you may select 0 stocks in the stock selection step to pause trading. Always set a meaningful `pause_duration_seconds` (≥ 1800) to avoid an immediate re‑pause. (See the Risk Appetite Framework above for exact thresholds).
- **Required parameter for every BUY/SELL:**
  - `"cooldown_after_loss_seconds"`: a non-negative integer (0 or more). If the trade results in a loss, the bot will avoid this stock for this many seconds before considering it again. Set 0 to allow immediate re-entry.
- **Optional parameters:**
  - `"position_size_fraction"`: a decimal between 0.01 and 1.0 representing the fraction of your **total available cash balance** to allocate to this trade. Must be > 0 and ≤ 1. Even very small values (0.01–0.05) are valid for low-conviction trades or when capital is limited. The sum of this fraction across all stocks you trade should not exceed 1.0.
  - `"max_risk_per_trade_pct"`: a decimal between 0 and 1.0 (e.g., 0.02 for 2% of portfolio). The position size will be limited so that the potential loss (entry - stop) does not exceed this fraction of your total portfolio value.
  - `"max_portfolio_risk_pct"`: an optional decimal between 0 and 1.0 (e.g., 0.06 for 6% of portfolio). If set, the bot will calculate the total potential loss of all open positions plus the potential loss of this new trade. If this total exceeds this percentage of your total portfolio value, the trade will be skipped.
  - `"min_profit_per_trade"`: an optional non-negative number (in base currency, e.g., 0.5). If set, the bot will skip the trade if the expected gross profit (position size × take_profit_pct) is below this value. Set `min_profit_per_trade` to **0** (or a very small value like 0.01) to allow tiny profits. Do not block trades because the expected profit is small – a small profit is still profit.
  - `"min_risk_reward_ratio"`: an optional positive number (e.g., 1.5). If set, the validator will reject the trade unless take_profit_pct / stop_loss_pct >= this value.
  - `"position_size_multiplier"`: an optional decimal between 0.0 and 1.0 (e.g., 0.5 for 50%). If set, the final position size for this trade will be further multiplied by this factor, after the global risk multiplier.
  - `"confidence_sizing_weight"`: an optional decimal between 0.0 and 1.0 (e.g., 0.5). Controls how much your confidence score scales the position size. The effective position size is multiplied by `(1.0 - confidence_sizing_weight × (1.0 - confidence))`. Set to 0.0 to disable confidence-based sizing (position size is unaffected by confidence). Set to 1.0 to make position size directly proportional to confidence (e.g., confidence 0.5 → half the position size). This makes your confidence score meaningful: higher confidence → larger position, lower confidence → smaller position.
  - `"min_confidence"`: an optional decimal between 0.0 and 1.0 (e.g., 0.6). If set, the bot will skip the trade if your confidence is below this threshold.
- `"portfolio_risk_adjustment_factor"`: an optional decimal between 0.1 and 1.0 (e.g., 0.5). This is your per-symbol "vote" on the overall portfolio risk for the current cycle. The engine will take the **minimum** of this factor across all symbols evaluated in the current cycle and apply it as a global multiplier to all position sizes. Use a lower value (e.g., 0.3–0.5) if you detect high volatility, unfavorable market regime shifts, or elevated risk for this symbol. Use 1.0 if conditions are normal and you see no reason to reduce overall portfolio risk. This allows you to dynamically adjust the global trading risk based on the latest per-symbol market data, rather than relying solely on the periodic stock selection phase.

- **Position Sizing — Your Full Responsibility:** You MUST decide the exact currency amount to trade by setting `position_size_fraction`. The engine will NOT automatically reduce your position size based on ATR or fixed risk limits. You must calculate the appropriate size yourself considering ALL of the following:
  1. **Risk per share**: For ATR-based stops, `risk_per_share = stop_loss_atr_multiple × ATR`. For fixed stops, `risk_per_share = stop_loss_pct × current_price`.
  2. **Max risk amount**: `max_risk_amount = total_portfolio_value × max_risk_per_trade_pct` (if you set `max_risk_per_trade_pct`).
  3. **Max quantity**: `max_quantity = max_risk_amount / risk_per_share`.
  4. **Position size fraction**: `position_size_fraction = (max_quantity × current_price) / total_portfolio_value`.
  5. Also consider: transaction costs (fees), your confidence level, backtest results (win rate, drawdown, profit factor), market conditions (volatility, regime, breadth), portfolio exposure, and concentration.
  Example: Portfolio €10,000, risk 1% (€100), ATR €0.50, stop = 2×ATR (€1.00 risk/share) → max 100 shares. At €25/share, `position_size_fraction = (100 × 25) / 10000 = 0.25`.
  If you prefer not to use risk-based sizing, you may set `position_size_fraction` based on confidence and setup quality. The engine respects your decision as long as it does not exceed available balance or exchange minimums.

**Pause/Resume:**
````

src/llm/prompts.py
````python
<<<<<<< SEARCH
    if atr is not None and current_price is not None and current_price > 0:
        atr_pct = atr / current_price
        min_sl = min_stop_atr_mult * atr_pct
        prompt += (
            f"\n**Current ATR%: {atr_pct:.4%}**. "
            f"The validator enforces a minimum fixed stop-loss of {min_stop_atr_mult} × ATR% = {min_sl:.4%}. "
            f"Your fixed stop_loss_pct must be at least this value.\n"
            f"\n**Position Sizing Guidance (you decide the final value):**\n"
            f"  Total portfolio value: ~{portfolio_total_value:.2f} {base_currency}\n"
            f"  Current price: {current_price:.4f}\n"
            f"  ATR: {atr:.6f}\n"
            f"  If you use ATR-based stop (multiplier M), risk per share = M × ATR = M × {atr:.6f}.\n"
            f"  If you want to risk R% of portfolio: max_quantity = (portfolio_value × R%) / (M × ATR).\n"
            f"  position_size_fraction = (max_quantity × current_price) / portfolio_value.\n"
            f"  Example: M=2, R=1% → risk/share = {2*atr:.6f}, max_qty = {portfolio_total_value*0.01/(2*atr):.2f}, "
            f"fraction = {(portfolio_total_value*0.01/(2*atr)*current_price)/portfolio_total_value:.4f}.\n"
            f"  Adjust R and M based on your confidence, backtest results, fees, and market conditions.\n"
        )
- You may include `"pause_trading"` (boolean) in your stock selection JSON to pause/resume trading. Always include a `"pause_reason"` string when setting pause_trading. You may also set `"pause_duration_seconds"` (positive integer) to auto-resume after a delay.
- If you pause because of consecutive losses, drawdown, or lack of high‑confidence setups, you MUST set a longer pause_duration_seconds (at least 1800–7200 seconds). A very short pause will almost certainly result in the same market conditions and an immediate re‑pause.
- Use shorter pauses (e.g., 600–1800s) only when you expect a specific short‑term event to pass.
- If you omit pause_duration_seconds, the engine will default to a 30‑minute pause.

**Learn from Past Trades:**
- After a losing trade on a stock, avoid that stock for at least several evaluation cycles. The prompt will include a list of recent closed trades for the current stock. Use this to avoid repeating mistakes and to reinforce successful patterns. If a stock has a string of losses, be more cautious or avoid it.
- Learn from historical performance: avoid stocks and strategies with poor win rates or negative average P&L.
- Calibrate your confidence: if high-confidence trades are losing, lower confidence for similar setups; if low-confidence trades are winning, consider raising confidence.

You will receive news sentiment data for each stock. Use it to gauge market sentiment and catalysts: prefer stocks with positive sentiment; be cautious with negative sentiment. If sentiment conflicts with technicals, give more weight to technicals but explain your reasoning.

Output strict JSON only. The response must start with '{' or '[' and end with '}' or ']'. No markdown fences, no explanations, no extra text.

**Stock & ETF Market Specifics:**
- **Earnings & Corporate Events:** Stocks can experience large price gaps due to earnings reports, FDA decisions, or other corporate events. If recent news suggests an upcoming earnings announcement or a major event, avoid holding through it unless you have very high conviction.
- **ETFs:** ETFs generally have lower volatility and smoother trends than individual stocks. Be aware of decay in leveraged ETFs if held long.
- **BTP Bonds (Italian Sovereign Bonds):** The asset universe may also include BTPs identified by their ISIN code (e.g., IT0001234567). BTPs are fixed-income securities with significantly lower volatility compared to stocks. When trading BTPs, use wider stop-losses (or ATR-based stops if ATR is available), longer max hold times, and smaller take-profit targets relative to stocks. They are suitable for capital preservation and steady income.
  - **Yield to Maturity (YTM) Assessment:** The `ticker` object includes `coupon` (annual coupon rate as a decimal, e.g., 0.0725 for 7.25%) and `maturity` (expiration date). Bond prices are quoted as a percentage of par value (e.g., a price of 101.68 means 101.68% of face value). To assess if a BTP is a good buy, calculate its approximate Yield to Maturity (YTM):
    - `Annual Coupon = coupon × 100` (e.g., 7.25)
    - `Years to Maturity = (maturity_date - current_date).days / 365`
    - `Approximate YTM = (Annual Coupon + (100 - Current Price) / Years to Maturity) / ((100 + Current Price) / 2)`
    - Compare the YTM to current Italian government bond yields (e.g., 10-year BTP yield) or your required return. If YTM is attractive relative to current market yields, the bond is a good buy. If the price is well above par (e.g., >110) and YTM is low, the upside is limited and there is higher downside risk if interest rates rise.

- **Two-Step Decision Process with Multiple Backtest Variants:** You will now operate in two steps. 
  1. In the first step, you will analyze the market data, indicators, and statistical summaries, and propose **multiple** sets of strategy parameters for backtesting. Each set is called a "backtest variant" and should explore a different hypothesis (e.g., tight stop vs wide stop, short hold vs long hold, trailing stop on/off, different take-profit targets, etc.). You decide how many variants to return (minimum 1, recommended 3–5, maximum __MAX_BACKTEST_VARIANTS__). The engine will run a local Python backtest for EACH variant sequentially. If you provide more than __MAX_BACKTEST_VARIANTS__ variants, only the first __MAX_BACKTEST_VARIANTS__ will be tested. Running just one backtes may not be enough to intercept profitable configurations, so provide several diverse variants to maximize the chance of finding a winning strategy.
  2. In the second step, you will receive ALL backtest results (one per variant) and be asked to make your final trading decision (BUY, SELL, or HOLD) based on the full set of results. You should compare the variants and choose the best-performing one (or combine insights from multiple variants) to inform your final decision and final strategy parameters.

**Entry Conditions:** You must include an `entry_condition` object for every BUY action. The strategy prompt provides full details and examples.

"""

# Replace placeholder with the configurable max backtest variants value
SYSTEM_PROMPT = SYSTEM_PROMPT.replace(
    "__MAX_BACKTEST_VARIANTS__", str(settings.MAX_BACKTEST_VARIANTS)
)

def build_stock_selection_prompt(
    available_symbols: List[str],
    current_symbols: List[Dict[str, str]],
    max_symbols: int,
    base_currency: str,
    tickers: Dict[str, Any],
    base_balance: float,
    per_symbol_budget: float,
    market_limits: Dict[str, Dict[str, Any]],
    performance: Optional[Dict[str, Any]] = None,
    ohlcv_summary: Optional[Dict[str, Dict[str, Any]]] = None,
    market_trend: Optional[Dict[str, Any]] = None,
    symbol_indicators: Optional[Dict[str, Dict[str, Any]]] = None,
    daily_pnl: Optional[float] = None,
    historical_ohlcv_summary: Optional[Dict[str, Dict[str, Any]]] = None,
    correlation_matrix: Optional[Dict[str, Dict[str, float]]] = None,
    session_info: Optional[Dict[str, Any]] = None,
    sentiment_trend: Optional[Dict[str, Optional[float]]] = None,
    trading_paused: Optional[bool] = None,
    open_positions: Optional[Dict[str, Dict[str, Any]]] = None,
    symbol_tenure: Optional[Dict[str, float]] = None,
    symbol_max_tenure: Optional[Dict[str, Optional[float]]] = None,
    vix: Optional[float] = None,
    trade_pattern_analysis: Optional[Dict[str, Any]] = None,
    symbol_events: Optional[Dict[str, Dict[str, Any]]] = None,
    symbol_trend_scores: Optional[Dict[str, float]] = None,
    market_breadth: Optional[Dict[str, Any]] = None,
    min_hold_time_mult: float = 1.0,
    min_stop_atr_mult: float = 1.0,
    min_viable_trade_amount: float = 0.0,
) -> str:
    """Build a prompt to ask the LLM which stocks/ETFs to trade."""
    # Summarize tickers and limits for the prompt
    # --- Batch-fetch sentiment for all symbols to avoid sequential DB queries ---
    batch_sentiment: Dict[str, Optional[Dict[str, Any]]] = {}
    if settings.NEWS_ENABLED:
        from src.database import get_aggregate_sentiment_for_symbols
        batch_sentiment = get_aggregate_sentiment_for_symbols(available_symbols, max_age_seconds=settings.NEWS_CACHE_TTL_SECONDS)

    ticker_summary = {}
    for symbol in available_symbols:
        if symbol in tickers:
            t = tickers[symbol]
            limits = market_limits.get(symbol, {})
            ticker_summary[symbol] = {
                "last": t.get("last"),
                "percentage_24h": t.get("percentage"),
                "volume": t.get("quoteVolume"),
                "min_trade_cost": limits.get("min_cost"),  # now always a number
                "name": t.get("name"),
                "coupon": t.get("coupon"),
                "maturity": t.get("maturity"),
            }
            if settings.NEWS_ENABLED:
                agg = batch_sentiment.get(symbol)
                if agg:
                    ticker_summary[symbol]["sentiment"] = agg

    # --- News section ---
    news_section = ""
    if settings.NEWS_ENABLED:
        news_lines = []
        # Limit news to top 20 candidates to avoid exceeding LLM context window
        symbols_to_check = available_symbols[:20]
        batch_news = get_news_for_symbols(symbols_to_check, max_age_seconds=settings.NEWS_CACHE_TTL_SECONDS)
        for sym in symbols_to_check:
            articles = batch_news.get(sym, [])
            if articles:
                formatted = _format_news_for_prompt(articles)
                news_lines.append(f"**{sym}**\n{formatted}")
        if news_lines:
            news_section = "Recent news for all candidate stocks:\n\n" + "\n\n".join(news_lines)

    available_timeframes = [tf for tf in settings.OHLCV_TIMEFRAMES if tf in TIMEFRAME_MAP]
    prompt = f"""Current base currency: {base_currency}
Your available {base_currency} balance: {base_balance:.2f}
Maximum number of stocks to trade: {max_symbols}
Reference equal-share budget per stock (suggestion only — you decide actual allocations): {per_symbol_budget:.2f} {base_currency}
Available timeframes: {json.dumps(available_timeframes)}
Currently tracked stocks (with assigned timeframes): {json.dumps(current_symbols) if current_symbols else "None"}"""

    prompt += (
        f"\n**Capital Allocation — Fully Dynamic:**\n"
        f"Your total available capital is {base_balance:.2f} {base_currency}.\n"
        f"The 'Budget per stock' figure above is ONLY a reference (equal split). "
        f"You are NOT required to allocate equally. You MUST decide the actual `position_size_fraction` "
        f"for each stock dynamically based on your confidence, the quality of the setup, volatility, "
        f"and all other parameters provided.\n"
        f"Key rules:\n"
        f"- The sum of `position_size_fraction` across ALL stocks you select must NOT exceed 1.0 "
        f"(i.e., total allocated capital must not exceed your available {base_currency} balance).\n"
        f"- Each trade amount is UNIQUE — do not default to equal splits. "
        f"Allocate more to high-conviction setups and less to speculative ones.\n"
        f"- **HYBRID ALLOCATION:** You may allocate ALL available capital to a single high-conviction trade "
        f"if you believe it is highly profitable, even if this leaves no capital for other tickers. "
        f"However, if you can leave some capital for other promising setups, do so. "
        f"**Do NOT place small trades that are unprofitable after fees** just to fill slots — "
        f"if a trade cannot be profitable with the available capital after accounting for transaction costs, "
        f"skip it entirely rather than allocating a tiny unprofitable amount.\n"
        f"- Even with a large number of symbols and a low portfolio value, you should still attempt trades. "
        f"Use small `position_size_fraction` values (e.g., 0.01–0.05) if needed, "
        f"but ONLY if the trade is still profitable after fees at that size.\n"
        f"- If allocating capital to one high-conviction trade leaves no room for others, that is acceptable. "
        f"Prioritize quality over quantity. You may concentrate capital on your best 1–3 setups.\n"
        f"- The only hard floor is the exchange minimum order size (shown as `min_trade_cost` per symbol). "
        f"Your `position_size_fraction × total_balance` must be ≥ that symbol's `min_trade_cost`.\n"
        f"- If your available balance cannot meet the exchange minimum for any symbol, select 0 stocks.\n"
    )

    # --- Open positions summary ---
    if open_positions:
        prompt += "\n**Open positions (these will continue to be managed even if trading is paused):**\n"
        for sym, pos in open_positions.items():
            entry = pos.get("price", "?")
            amount = pos.get("amount", "?")
            sl = pos.get("stop_loss", "?")
            tp = pos.get("take_profit", "?")
            prompt += (
                f"  {sym}: entry={entry}, amount={amount}, "
                f"stop_loss={sl}, take_profit={tp}\n"
            )
        prompt += (
            "When deciding to pause or resume trading, consider these open positions. "
            "If you pause, no new positions will be opened, but existing positions will still be "
            "managed with their stop-loss/take-profit levels. "
            "If you resume, new positions can be opened alongside these.\n"
        )

    if symbol_tenure:
        prompt += "\n**Stock tenure (how long each stock has been continuously tracked, in seconds):**\n"
        for sym, sec in symbol_tenure.items():
            prompt += f"  {sym}: {sec:.0f}s\n"
    if symbol_max_tenure:
        prompt += "\n**Current max tenure per stock (hours, if set):**\n"
        for sym, hours in symbol_max_tenure.items():
            if hours is not None:
                prompt += f"  {sym}: {hours:.1f}h\n"

    prompt += f"""
Available symbols with market data and minimum trade cost (in {base_currency}):
{json.dumps(ticker_summary)}

Select between {settings.MIN_SYMBOLS if settings.MIN_SYMBOLS > 0 else 0} and {max_symbols} assets (stocks, ETFs, or BTP bonds) to trade. The available symbols may include Italian BTP bonds identified by their ISIN (e.g., IT0001234567). The `name` field in the market data contains the bond's description, including maturity and coupon (e.g., 'Btp-1nv26 7,25%' means November 2026 maturity, 7.25% coupon). You can select them alongside stocks. If market conditions are extremely unfavorable (e.g., high losses, poor momentum, negative sentiment), you may select 0 assets to pause trading until the next evaluation. **You should actively try to select as many symbols as possible up to {max_symbols} when there are profitable opportunities.** Do not default to selecting only 2–3 symbols when you have {max_symbols} slots available — each unused slot is a missed opportunity. Use small position_size_fraction values (e.g., 0.02–0.05) to fit more symbols within your available capital. You MUST only select assets where your total available balance ({base_balance:.2f} {base_currency}) is greater than or equal to the asset's min_trade_cost. Since you decide position_size_fraction dynamically, you can allocate any portion of your total balance to any asset — the equal-share budget is NOT a constraint. Skip only assets whose min_trade_cost exceeds your total available balance. **Stocks, ETFs, and BTPs all have EQUAL priority.** Do not favor stocks over ETFs or BTPs, or vice versa. Evaluate each asset on its own merits (trend, momentum, sentiment, fundamentals) regardless of its asset class. Consider a mix of stocks, ETFs, and BTP bonds for diversification. BTPs offer lower volatility and steady income, making them excellent for capital preservation or hedging against market volatility. ETFs provide broad market exposure. Do not ignore BTPs and ETFs just because they have lower volume than individual stocks; include them in your portfolio when they offer a favorable risk/reward profile or when market conditions warrant a more conservative stance. You may keep some current assets if they are still promising and meet the budget requirement, or replace them. **Prefer to keep assets that have been tracked for a while** – they have more historical data and the bot has already invested in learning their behaviour. Only drop an asset if it shows clear deterioration (e.g., negative momentum on all timeframes, poor win rate, or strongly negative sentiment). For assets already being tracked, re-evaluate their assigned timeframe. If the market regime has changed (e.g., a stock that was trending on 1d is now choppy and better suited to 1w), update the timeframe. If you change the timeframe for an asset with an open position, the bot will switch to managing the position using the new timeframe.

**Important:** Unless the market is in a clear crisis (e.g., breadth < 20%, account in deep drawdown), you MUST select at least 1–2 stocks with **small position sizes** (position_size_fraction ≤ 0.15) and **tight stops**. Doing nothing guarantees zero profit; a cautious small trade at least gives a chance. Only select 0 stocks if conditions are truly hostile. **When you have {max_symbols} slots available, you should aim to fill as many as possible** — select up to {max_symbols} stocks/ETFs/BTPs to diversify your portfolio and capture more opportunities. Do not artificially limit yourself to 1-2 stocks if the market offers more valid setups. Use small position_size_fraction values (0.02–0.05) to fit more symbols within your capital. A portfolio of 20+ small positions across stocks, ETFs, and BTPs is strongly preferred over concentrating on 2–3 large positions, as it spreads risk and increases the probability of capturing profitable moves.
"""
    if settings.MIN_SYMBOLS > 0:
        prompt += (
            f"\n**MANDATORY:** You MUST select at least {settings.MIN_SYMBOLS} symbols. "
            f"Selecting fewer than {settings.MIN_SYMBOLS} is NOT allowed unless you are pausing trading entirely (pause_trading=true). "
            f"If you cannot find {settings.MIN_SYMBOLS} high-conviction setups, select the next best symbols with small position_size_fraction (0.01-0.05) and tight stops. "
            f"Do NOT select fewer than {settings.MIN_SYMBOLS} — the engine will override your selection and add more symbols automatically.\n"
        )
    prompt += """

Each symbol can only appear once in your selection. Choose the single best timeframe for each stock based on the multi-timeframe OHLCV data.

**CRITICAL — You MUST use long-term timeframes ("5Y", "3Y", "1Y", "6M", "3M", "1M", "1w") for long‑term trading.** All of these timeframes are EQUALLY valid as primary timeframes. Do NOT always select "5Y" — choose the most appropriate timeframe for each asset based on its characteristics (volatility, trend clarity, sector, expected hold period). Diversify timeframe assignments across your selected symbols rather than assigning the same timeframe to all. Use "1d" or "1h" ONLY if the stock shows high short‑term volatility or you need finer entry timing AND long-term data is unavailable. The biggest profits in this asset universe come from identifying strong long-term trends and holding through them.

**Output ONLY the raw JSON object as specified.**

Return a JSON object with the following fields:
- "stocks": a JSON array of objects, each with "symbol", "timeframe" (the timeframe must be one of the available timeframes, e.g., {', '.join([repr(tf) for tf in available_timeframes])}), and "sector" (a string representing the stock's sector, e.g., "Technology", "Healthcare", "Financials", "Energy", "Consumer Discretionary", "Consumer Staples", "Industrials", "Materials", "Real Estate", "Utilities", "Communication Services"). Each object may optionally include "max_tenure_hours" (a positive float, hours) to force-sell the stock after that many hours in the portfolio. Omit or set to null for no limit.
- "max_stocks": an integer between 0 and {max_symbols} indicating how many stocks you actually want to trade. Set to 0 to pause trading. This must equal the length of the "stocks" array. **You should set this as high as possible (up to {max_symbols}) when there are profitable opportunities — do not default to small numbers like 2 or 3 when you have {max_symbols} slots available.**
- "max_positions_per_sector": an integer between 1 and {max_symbols} indicating the maximum number of open positions allowed in the same sector at the same time. This helps diversify risk across different sectors. You decide this value based on current market volatility and your confidence in specific sectors.
- "skip_eval_price_change_atr_mult": a float (e.g., 0.5) indicating the minimum price change (as a multiple of ATR%) required to trigger a new LLM strategy evaluation for a stock. If the price moves less than this, the LLM is skipped to save costs.
- "skip_eval_rsi_change": a float (e.g., 5.0) indicating the minimum absolute RSI change required to trigger a new LLM evaluation.
- "skip_eval_rsi_oversold": a float (e.g., 30.0) indicating the RSI level below which the bot should always trigger a new LLM evaluation (potential oversold buy signal), even if nothing else changed.
- "skip_eval_rsi_overbought": a float (e.g., 70.0) indicating the RSI level above which the bot should always trigger a new LLM evaluation (potential overbought sell signal), even if nothing else changed.
- "skip_eval_macd_hist_change": a float (e.g., 0.0005) indicating the minimum absolute MACD histogram change required to trigger a new LLM evaluation.
- "regime_adx_strong": a float (e.g., 40.0) indicating the ADX level above which a trend is considered strong.
- "regime_adx_moderate": a float (e.g., 25.0) indicating the ADX level above which a trend is considered moderate.
- "regime_volatility_high_pct": a float (e.g., 80.0) indicating the ATR percentile above which volatility is considered high.
- "regime_volatility_low_pct": a float (e.g., 20.0) indicating the ATR percentile below which volatility is considered low.
- "regime_bb_squeeze_width": a float (e.g., 0.02) indicating the Bollinger Band width below which the market is considered in a squeeze.
- "regime_bb_expansion_width": a float (e.g., 0.08) indicating the Bollinger Band width above which the market is considered in expansion.
- "min_stop_loss_atr_mult": a float (e.g., 1.5) indicating the minimum stop-loss as a multiple of ATR%. Trades with a fixed stop below this multiple of ATR will be rejected. Lower values allow tighter stops; higher values require wider stops.
- "min_max_hold_time_mult": a float (e.g., 2.0) indicating the minimum max_hold_time_seconds as a multiple of the candle timeframe (in seconds). Trades with max hold time below this multiple will be rejected.
- "max_stop_loss_reviews": an integer between 1 and 20 (e.g., 3). The maximum number of times the LLM can review a triggered stop-loss before the engine force-sells the position. Lower values = quicker exit on stop-loss; higher values = more LLM discretion.
- "max_take_profit_reviews": an integer between 1 and 20 (e.g., 3). The maximum number of times the LLM can review a triggered take-profit before the engine force-sells the position. Lower values = quicker profit-taking; higher values = more LLM discretion.
- "max_partial_tp_reviews": an integer between 1 and 20 (e.g., 3). The maximum number of times the LLM can review a triggered partial take-profit before the engine force-executes the partial sell.
- "max_dust_sweep_reviews": an integer between 1 and 20 (e.g., 3). The maximum number of times the LLM can review a triggered dust sweep before the engine force-sells the remaining dust.
- "min_llm_pause_duration_seconds": an integer between 300 and 14400 (e.g., 3600). The minimum duration in seconds that the LLM must wait before it can resume trading after pausing. This prevents rapid pause/resume cycles. Lower values = faster resume capability; higher values = more conservative cooldown.
- "pause_max_consecutive_keep": an integer between 1 and 10 (e.g., 3). The maximum number of consecutive "keep paused" decisions the LLM can make before the engine force-resumes trading with a reduced risk multiplier. Lower values = quicker force-resume; higher values = more patience with LLM's pause decisions.
- "pause_force_resume_risk_multiplier": a float between 0.0 and 1.0 (e.g., 0.3). The global risk multiplier applied when the engine force-resumes trading after too many consecutive "keep paused" decisions. Lower values = more conservative forced resume; higher values = more aggressive.
- "max_portfolio_exposure_pct": a float between 0.0 and 1.0 (e.g., 0.7 for 70%). The maximum percentage of total portfolio value that can be deployed in open positions.
- "max_portfolio_stop_risk_pct": a float between 0.0 and 1.0 (e.g., 0.05 for 5%). The maximum total stop-loss risk as a percentage of portfolio value.
- "min_risk_reward_ratio": a positive number (e.g., 1.5). The minimum reward:risk ratio required for all trades. Trades with a lower ratio will be rejected.
- "confidence_rejection_threshold": a float between 0.0 and 1.0 (e.g., 0.4). The minimum confidence required for any trade to be executed. Trades with confidence below this threshold will be rejected. Set to 0.0 to disable. This is a global threshold that applies to all trades in the cycle.
- "limit_price_max_distance_pct": an optional float between 0.0 and 1.0 (e.g., 0.05 for 5%). The maximum allowed distance of a limit price from the current best bid/ask. Orders with a limit price further away than this are rejected to avoid indefinite queuing. Set to 0.0 to disable the check entirely. If omitted, the engine uses its default (0.05).
- "min_viable_trade_amount": an optional positive number (e.g., 500.0) indicating the minimum amount in {base_currency} that should be allocated to a single trade for it to be profitable after fees. This is a SUGGESTION only — the engine does NOT block trades below this value. You decide the actual position size dynamically. Set to 0 to allow trades of any size (only exchange minimums apply).
- "reasoning": a short string (max 200 characters) explaining why you selected these specific stocks and timeframes. This will be shown to the user, so make it informative.

You may optionally include the following fields:
- "stock_revaluation_interval_seconds" (integer >= 3600, i.e., at least 1 hour) to change how often the bot re-evaluates the stock list. The bot also re-evaluates automatically before market open and when unusual market conditions are detected (significant news, extreme indicators, unusually active market).
- "pause_trading" (boolean): true to pause trading, false to resume. Always include "pause_reason" (string) when setting pause_trading. You may also set "pause_duration_seconds" (positive integer) to auto-resume after a delay.

Example: {{"stocks": [{{"symbol": "ENI.MI/EUR", "timeframe": "1Y", "sector": "Energy", "max_tenure_hours": 8760}}, {{"symbol": "ENEL.MI/EUR", "timeframe": "6M", "sector": "Utilities"}}, {{"symbol": "STM.MI/EUR", "timeframe": "3Y", "sector": "Technology"}}], "max_stocks": 3, "max_positions_per_sector": 2, "skip_eval_price_change_atr_mult": 0.5, "skip_eval_rsi_change": 5.0, "skip_eval_rsi_oversold": 30.0, "skip_eval_rsi_overbought": 70.0, "skip_eval_macd_hist_change": 0.0005, "regime_adx_strong": 40.0, "regime_adx_moderate": 25.0, "regime_volatility_high_pct": 80.0, "regime_volatility_low_pct": 20.0, "regime_bb_squeeze_width": 0.02, "regime_bb_expansion_width": 0.08, "min_stop_loss_atr_mult": 1.5, "min_max_hold_time_mult": 1.5, "max_stop_loss_reviews": 3, "max_take_profit_reviews": 3, "min_llm_pause_duration_seconds": 3600, "pause_max_consecutive_keep": 3, "pause_force_resume_risk_multiplier": 0.3, "max_partial_tp_reviews": 3, "max_dust_sweep_reviews": 3, "reasoning": "ENI shows strong uptrend on 1Y with high volume; ENEL has bullish MACD crossover on 6M.", "stock_revaluation_interval_seconds": 300, "max_portfolio_exposure_pct": 0.8, "max_portfolio_stop_risk_pct": 0.1, "min_risk_reward_ratio": 1.5, "confidence_rejection_threshold": 0.4, "limit_price_max_distance_pct": 0.05, "pause_trading": false, "pause_reason": "Market conditions are favorable"}}

Set `max_portfolio_exposure_pct` to at least **0.8** and `max_portfolio_stop_risk_pct` to at least **0.1** unless you have a very strong reason to be more conservative. Higher limits allow the bot to take more positions simultaneously and capture more opportunities. Do not unnecessarily restrict capital deployment."""
    # --- Enhanced pause/resume guidance ---
    if trading_paused:
        prompt += (
            "\n**Trading is currently PAUSED.**\n"
            "You may resume trading by setting `\"pause_trading\": false` if you see clear profit opportunities.\n"
            "Do NOT resume just because market conditions have improved slightly; only resume if you identify specific "
            "stocks with strong setups (positive sentiment, solid technicals, favorable trend) that are likely to be profitable.\n"
            "If you keep trading paused, include a `\"pause_reason\"` field explaining why.\n"
            "\nAlso consider the news sentiment data below when deciding whether to resume.\n"
        )
    else:
        prompt += (
            "\n**Trading is currently ACTIVE.**\n"
            "You may pause trading by setting `\"pause_trading\": true` if conditions warrant.\n"
            "However, do NOT pause solely because of a bad market index (e.g., high fear, low breadth). "
            "First, check the news sentiment and technical indicators. If there are stocks with "
            "strong positive sentiment and clear technical signals, you may still trade them profitably even in a down market.\n"
            "Only pause if NO such opportunities exist, or if the account is in significant drawdown with no high‑confidence setups.\n"
            "\nAlso consider the news sentiment data below when deciding whether to pause.\n"
        )
    if symbol_trend_scores:
        prompt += "\nTrend quality scores (0-1, higher = cleaner trend; combines ADX strength, EMA alignment, RSI consistency, MACD direction, +DI/-DI confirmation):\n"
        for sym in available_symbols:
            if sym in symbol_trend_scores:
                prompt += f"  {sym}: {symbol_trend_scores[sym]:.3f}\n"
        prompt += "High trend quality (>0.7) = strong, clean trend suitable for momentum/breakout strategies. Low score (<0.3) = choppy or ranging, better for mean reversion or avoid.\n"
    if ohlcv_summary:
        filtered_ohlcv_summary = {}
        for sym, tfs in ohlcv_summary.items():
            valid_tfs = {tf: data for tf, data in tfs.items() if data}
            if valid_tfs:
                filtered_ohlcv_summary[sym] = valid_tfs
        if filtered_ohlcv_summary:
            prompt += f"\nMulti-timeframe OHLCV summary (price change %, high, low, volume):\n{json.dumps(filtered_ohlcv_summary)}\n"
        else:
            prompt += (
                "\n**Note:** No OHLCV data is available for any candidate symbol. "
                "You must base your selection entirely on ticker data (price, 24h change, volume), "
                "news sentiment, trend quality scores, and other provided metrics. "
                "Do not pause trading solely due to missing OHLCV if other indicators suggest strong opportunities.\n"
            )
    else:
        prompt += (
            "\n**Note:** No OHLCV data is available for any candidate symbol. "
            "You must base your selection entirely on ticker data (price, 24h change, volume), "
            "news sentiment, trend quality scores, and other provided metrics. "
            "Do not pause trading solely due to missing OHLCV if other indicators suggest strong opportunities.\n"
        )
    prompt += (
        "\n**CRITICAL — Only assign timeframes that have OHLCV data.** "
        "The Multi-timeframe OHLCV summary above shows exactly which timeframes have data for each symbol. "
        "You MUST only assign a timeframe to a symbol if that timeframe appears in the symbol's OHLCV summary. "
        "If a symbol has no OHLCV data at all, do NOT select it — there is nothing to analyze.\n"
    )
    if correlation_matrix:
        # Trim to only include symbols that appear in the candidate list
        candidate_set = set(available_symbols)
        trimmed = {}
        for sym_a, row in correlation_matrix.items():
            if sym_a not in candidate_set:
                continue
            trimmed[sym_a] = {sym_b: v for sym_b, v in row.items() if sym_b in candidate_set}
        if trimmed:
            prompt += (
                "\nPairwise correlation matrix (Pearson correlation of daily returns, range -1 to +1):\n"
                f"{json.dumps(trimmed)}\n"
            )
    if symbol_indicators:
        prompt += "\nTechnical indicators for candidate assets (stocks, ETFs, BTPs):\n"
        # Only include key long-term timeframes to keep prompt size manageable
        key_timeframes = {"5Y", "3Y", "1Y", "6M", "3M", "1M", "1w"}
        for sym, tf_indicators in symbol_indicators.items():
            lines = [f"{sym}:"]
            for tf, ind in tf_indicators.items():
                if tf not in key_timeframes:
                    continue
                tf_lines = []
                if ind.get('rsi') is not None:
                    tf_lines.append(f"    RSI(14)={ind['rsi']:.2f}")
                if ind.get('macd') is not None:
                    tf_lines.append(f"    MACD={ind['macd']:.4f} Signal={ind['macd_signal']:.4f} Hist={ind['macd_hist']:.4f}")
                if ind.get('bb_upper') is not None:
                    tf_lines.append(f"    BB Upper={ind['bb_upper']:.4f} Middle={ind['bb_middle']:.4f} Lower={ind['bb_lower']:.4f}")
                if ind.get('ema_9') is not None:
                    ema9_str = f"EMA9={ind['ema_9']:.4f}"
                    ema21_str = f" EMA21={ind['ema_21']:.4f}" if ind.get('ema_21') is not None else ""
                    tf_lines.append(f"    {ema9_str}{ema21_str}")
                if ind.get('stochastic_k') is not None:
                    d_str = f"{ind['stochastic_d']:.2f}" if ind['stochastic_d'] is not None else "N/A"
                    tf_lines.append(f"    Stoch %K={ind['stochastic_k']:.2f} %D={d_str}")
                if ind.get('adx') is not None:
                    tf_lines.append(f"    ADX(14)={ind['adx']:.2f} +DI={ind['plus_di']:.2f} -DI={ind['minus_di']:.2f}")
                if ind.get('obv') is not None:
                    tf_lines.append(f"    OBV={ind['obv']:.2f}")
                if ind.get('mfi') is not None:
                    tf_lines.append(f"    MFI(14)={ind['mfi']:.2f}")
                if ind.get('cci') is not None:
                    tf_lines.append(f"    CCI(20)={ind['cci']:.2f}")
                if ind.get('williams_r') is not None:
                    tf_lines.append(f"    Williams %R(14)={ind['williams_r']:.2f}")
                if ind.get('ichimoku') is not None:
                    ich = ind['ichimoku']
                    tf_lines.append(f"    Ichimoku: Tenkan={ich['tenkan_sen']:.4f} Kijun={ich['kijun_sen']:.4f} SpanA={ich['senkou_span_a']:.4f} SpanB={ich['senkou_span_b']:.4f} Cloud={ich['cloud_bottom']:.4f}-{ich['cloud_top']:.4f}")
                if ind.get('donchian_channels') is not None:
                    dc = ind['donchian_channels']
                    tf_lines.append(f"    Donchian: Upper={dc['upper']:.4f} Middle={dc['middle']:.4f} Lower={dc['lower']:.4f}")
                if ind.get('parabolic_sar') is not None:
                    tf_lines.append(f"    SAR={ind['parabolic_sar']:.6f}")
                if ind.get('keltner_channels') is not None:
                    kc = ind['keltner_channels']
                    tf_lines.append(f"    Keltner: Upper={kc['upper']:.6f} Middle={kc['middle']:.6f} Lower={kc['lower']:.6f}")
                
                if tf_lines:
                    lines.append(f"  [{tf}]")
                    lines.extend(tf_lines)
            if len(lines) > 1:
                prompt += "\n".join(lines) + "\n"
    if market_trend:
        prompt += f"\nOverall market trend ({market_trend['symbol']}): daily change {market_trend.get('change_24h')}%, last price {market_trend.get('last')}\n"
    if session_info:
        prompt += f"\nCurrent UTC hour: {session_info['utc_hour']} ({session_info['session']} session)\n"
    # VIX is not available for the Italian market — omitted.
    # --- Market regime summary (based on breadth only, VIX not available) ---
    regime_label = "neutral"
    if market_breadth:
        breadth_pct = market_breadth.get("positive_pct", 50)
        if breadth_pct > 60:
            regime_label = "RISK-ON (broad market strength)"
        elif breadth_pct < 40:
            regime_label = "RISK-OFF (broad market weakness)"
    prompt += f"\n**Market Regime: {regime_label}**\n"
    if sentiment_trend:
        prompt += "\nSentiment trend (change in compound score since last cycle):\n"
        for base, delta in sentiment_trend.items():
            if delta is not None:
                prompt += f"  {base}: {delta:+.4f}\n"
    if news_section:
        prompt += f"\n{news_section}\n"
        prompt += (
            "**IMPORTANT:** Do not rely on pre-computed sentiment scores. Read the news headlines and summaries above "
            "and use your own understanding of financial context to assess the sentiment and potential impact for each stock. "
            "Factor this assessment into your stock selection and reasoning.\n"
        )
    if performance:
        perf_lines = ["Historical Performance Data:"]
        equity_curve = performance.get('equity_curve', {})
        stock_perf = performance.get('stock_performance', {})
        strategy_perf = performance.get('strategy_performance', {})
        
        if equity_curve:
            perf_lines.append(f"Overall equity curve: {json.dumps(equity_curve)}")
        if stock_perf:
            perf_lines.append(f"Per-stock performance (win rate, avg P&L, total trades): {json.dumps(stock_perf)}")
        if strategy_perf:
            perf_lines.append(f"Per-strategy performance: {json.dumps(strategy_perf)}")
        
        if len(perf_lines) > 1:
            prompt += "\n".join(perf_lines) + "\n"
        if daily_pnl is not None:
            prompt += f"Today's realized P&L: {daily_pnl:.4f} {base_currency}\n"
        consecutive_losses = performance.get("equity_curve", {}).get("consecutive_losses", 0)
        if consecutive_losses > 0:
            prompt += f"⚠️ You have {consecutive_losses} consecutive losing trades. Consider pausing or reducing risk.\n"
    # --- Trade pattern analysis ---
    if trade_pattern_analysis:
        prompt += "\n" + _format_trade_pattern_analysis(trade_pattern_analysis) + "\n"
    # --- Account P&L context ---
    if performance:
        daily_pnl = performance.get("equity_curve", {}).get("daily_pnl", 0.0)
        total_pnl = performance.get("equity_curve", {}).get("total_pnl", 0.0)
        prompt += (
            f"\n**Account P&L**: Today's realized P&L = {daily_pnl:.4f} {base_currency}, "
            f"Total realized P&L = {total_pnl:.4f} {base_currency}.\n"
        )
    if symbol_events:
        prompt += "\n**Upcoming Corporate Events (detected from news):**\n"
        prompt += "These symbols have upcoming or recent corporate events. Consider the risk of holding through these events.\n"
        for sym, event in symbol_events.items():
            types = ", ".join(event.get("event_types", []))
            kws = ", ".join(event.get("keywords", [])[:5])
            prompt += f"  {sym}: {types} (keywords: {kws})\n"
        prompt += (
            "Stocks with upcoming earnings or major events can gap significantly. "
            "You may choose to avoid these stocks, reduce position sizes, or set wider stops. "
            "The decision is yours.\n"
        )
    return prompt


def build_final_selection_prompt(
    chunk_results: List[Dict[str, Any]],
    current_symbols: List[Dict[str, str]],
    max_symbols: int,
    base_currency: str,
    base_balance: float,
    per_symbol_budget: float,
    performance: Optional[Dict[str, Any]] = None,
    open_positions: Optional[Dict[str, Dict[str, Any]]] = None,
    market_breadth: Optional[Dict[str, Any]] = None,
    full_market_breadth: Optional[Dict[str, Any]] = None,
    market_trend: Optional[Dict[str, Any]] = None,
    session_info: Optional[Dict[str, Any]] = None,
    trading_paused: Optional[bool] = None,
    symbol_tenure: Optional[Dict[str, float]] = None,
    symbol_max_tenure: Optional[Dict[str, Optional[float]]] = None,
    trade_pattern_analysis: Optional[Dict[str, Any]] = None,
    daily_pnl: Optional[float] = None,
    vix: Optional[float] = None,
    min_viable_trade_amount: float = 0.0,
    available_timeframes: Optional[List[str]] = None,
    market_limits: Optional[Dict[str, Dict[str, Any]]] = None,
    available_timeframes_by_symbol: Optional[Dict[str, List[str]]] = None,
) -> str:
    """Build a prompt for the final symbol selection from chunk results.

    After evaluating candidates in chunks, this prompt presents the combined
    shortlist from all chunks and asks the LLM to make the final selection.
    """
    # Build shortlist from all chunk results
    shortlist = []
    for i, chunk in enumerate(chunk_results):
        stocks = chunk.get("stocks", [])
        reasoning = chunk.get("reasoning", "")
        for stock in stocks:
            sym = stock.get("symbol")
            entry = {
                "symbol": sym,
                "timeframe": stock.get("timeframe"),
                "sector": stock.get("sector"),
                "chunk_reasoning": reasoning[:200] if reasoning else "",
            }
            if stock.get("max_tenure_hours") is not None:
                entry["max_tenure_hours"] = stock.get("max_tenure_hours")
            # Include min_trade_cost so the LLM can avoid selecting unaffordable symbols
            if market_limits and sym in market_limits:
                min_cost = market_limits[sym].get("min_cost")
                if min_cost is not None:
                    entry["min_trade_cost"] = min_cost
            shortlist.append(entry)

    # Deduplicate by symbol
    seen = set()
    deduped_shortlist = []
    for s in shortlist:
        sym = s.get("symbol")
        if sym and sym not in seen:
            seen.add(sym)
            deduped_shortlist.append(s)
    shortlist = deduped_shortlist

    if not available_timeframes:
        available_timeframes = ["5Y", "3Y", "1Y", "6M", "3M", "1M", "1w", "1d", "1h"]

    prompt = f"""**Final Symbol Selection — Step 2**

You have evaluated {len(chunk_results)} batches of candidate symbols and selected a combined shortlist of {len(shortlist)} symbols.
Your task is to make the FINAL selection of up to {max_symbols} symbols from this shortlist.

Current base currency: {base_currency}
Your available {base_currency} balance: {base_balance:.2f}
Maximum number of stocks to trade: {max_symbols}
Reference equal-share budget per stock (suggestion only): {per_symbol_budget:.2f} {base_currency}
Available timeframes: {json.dumps(available_timeframes)}
Currently tracked stocks (with assigned timeframes): {json.dumps(current_symbols) if current_symbols else "None"}

**Combined Shortlist from All Batches (deduplicated):**
{json.dumps(shortlist)}

"""
    if available_timeframes_by_symbol:
        prompt += "**Available timeframes per symbol (ONLY select from these):**\n"
        for sym, tfs in available_timeframes_by_symbol.items():
            if tfs:
                prompt += f"  {sym}: {', '.join(tfs)}\n"
        prompt += (
            "\n**CRITICAL — Only assign timeframes that have OHLCV data.** "
            "You MUST only assign a timeframe to a symbol if it appears in the list above for that symbol. "
            "If a symbol has no available timeframes, do NOT select it.\n\n"
        )
    prompt += f"""Select between {settings.MIN_SYMBOLS if settings.MIN_SYMBOLS > 0 else 0} and {max_symbols} assets from the shortlist above. You may keep current assets if they are still promising, or replace them. Each symbol can only appear once. Choose the single best timeframe for each stock.

**Important:** Unless the market is in a clear crisis, you MUST select at least 1-2 stocks. **HYBRID ALLOCATION:** You may allocate ALL available capital to a single high-conviction trade if you believe it is highly profitable, even if this leaves no capital for other tickers. However, if you can leave some capital for other promising setups, do so. **Do NOT place small trades that are unprofitable after fees** just to fill slots — if a trade cannot be profitable with the available capital after accounting for transaction costs, skip it entirely. Prioritize quality over quantity. You may concentrate capital on your best 1–3 setups.

"""
    if settings.MIN_SYMBOLS > 0:
        prompt += (
            f"\n**MANDATORY:** You MUST select at least {settings.MIN_SYMBOLS} symbols. "
            f"Selecting fewer than {settings.MIN_SYMBOLS} is NOT allowed unless you are pausing trading entirely. "
            f"Use small position_size_fraction values (0.01-0.05) for lower-conviction symbols to fill the remaining slots.\n"
        )

    # Add open positions
    if open_positions:
        prompt += "\n**Open positions (these will continue to be managed even if trading is paused):**\n"
        for sym, pos in open_positions.items():
            entry = pos.get("price", "?")
            amount = pos.get("amount", "?")
            sl = pos.get("stop_loss", "?")
            tp = pos.get("take_profit", "?")
            prompt += f"  {sym}: entry={entry}, amount={amount}, stop_loss={sl}, take_profit={tp}\n"
        prompt += "When deciding to pause or resume, consider these open positions.\n"

    # Add symbol tenure
    if symbol_tenure:
        shortlist_syms = {s.get("symbol") for s in shortlist}
        current_syms = {c.get("symbol") for c in current_symbols} if current_symbols else set()
        relevant_syms = shortlist_syms | current_syms
        prompt += "\n**Stock tenure (seconds):**\n"
        for sym, sec in symbol_tenure.items():
            if sym in relevant_syms:
                prompt += f"  {sym}: {sec:.0f}s\n"
    if symbol_max_tenure:
        prompt += "\n**Current max tenure per stock (hours, if set):**\n"
        for sym, hours in symbol_max_tenure.items():
            if hours is not None and (sym in {s.get("symbol") for s in shortlist} or sym in {c.get("symbol") for c in current_symbols}):
                prompt += f"  {sym}: {hours:.1f}h\n"

    # Add performance
    if performance:
        perf_lines = ["Historical Performance Data:"]
        equity_curve = performance.get('equity_curve', {})
        stock_perf = performance.get('stock_performance', {})
        strategy_perf = performance.get('strategy_performance', {})
        
        if equity_curve:
            perf_lines.append(f"Overall equity curve: {json.dumps(equity_curve)}")
        if stock_perf:
            perf_lines.append(f"Per-stock performance (win rate, avg P&L, total trades): {json.dumps(stock_perf)}")
        if strategy_perf:
            perf_lines.append(f"Per-strategy performance: {json.dumps(strategy_perf)}")
        
        if len(perf_lines) > 1:
            prompt += "\n".join(perf_lines) + "\n"
        if daily_pnl is not None:
            prompt += f"Today's realized P&L: {daily_pnl:.4f} {base_currency}\n"
        consecutive_losses = equity_curve.get("consecutive_losses", 0)
        if consecutive_losses > 0:
            prompt += f"⚠️ You have {consecutive_losses} consecutive losing trades. Consider pausing or reducing risk.\n"

    # Add trade pattern analysis
    if trade_pattern_analysis:
        prompt += "\n" + _format_trade_pattern_analysis(trade_pattern_analysis) + "\n"

    # Add market context
    if market_breadth:
        prompt += f"\nMarket breadth: {market_breadth.get('positive_pct', 50)}% positive ({market_breadth.get('positive_count', 0)}/{market_breadth.get('total_count', 0)})\n"
    if full_market_breadth:
        prompt += f"Full market breadth: {full_market_breadth.get('positive_pct', 50)}% positive\n"
    if market_trend:
        prompt += f"\nOverall market trend ({market_trend['symbol']}): daily change {market_trend.get('change_24h')}%, last price {market_trend.get('last')}\n"
    if session_info:
        prompt += f"\nCurrent UTC hour: {session_info['utc_hour']} ({session_info['session']} session)\n"

    # Market regime
    regime_label = "neutral"
    if market_breadth:
        breadth_pct = market_breadth.get("positive_pct", 50)
        if breadth_pct > 60:
            regime_label = "RISK-ON (broad market strength)"
        elif breadth_pct < 40:
            regime_label = "RISK-OFF (broad market weakness)"
    prompt += f"\n**Market Regime: {regime_label}**\n"

    # Pause/resume guidance
    if trading_paused:
        prompt += (
            "\n**Trading is currently PAUSED.**\n"
            "You may resume trading by setting pause_trading to false if you see clear profit opportunities.\n"
            "If you keep trading paused, include a pause_reason field.\n"
        )
    else:
        prompt += (
            "\n**Trading is currently ACTIVE.**\n"
            "You may pause trading by setting pause_trading to true if conditions warrant.\n"
        )

    # Account P&L
    if performance:
        daily_pnl_val = performance.get("equity_curve", {}).get("daily_pnl", 0.0)
        total_pnl = performance.get("equity_curve", {}).get("total_pnl", 0.0)
        prompt += (
            f"\n**Account P&L**: Today's realized P&L = {daily_pnl_val:.4f} {base_currency}, "
            f"Total realized P&L = {total_pnl:.4f} {base_currency}.\n"
        )

    # Output format
    prompt += f"""
Return a JSON object with the following fields:
- "stocks": a JSON array of objects, each with "symbol", "timeframe" (one of {', '.join([repr(tf) for tf in available_timeframes])}), "sector", and optionally "max_tenure_hours"
- "max_stocks": an integer between 0 and {max_symbols}
- "max_positions_per_sector": an integer between 1 and {max_symbols}
- "reasoning": a short string (max 200 characters) explaining your final selection
- "skip_eval_price_change_atr_mult": a float (e.g., 0.5)
- "skip_eval_rsi_change": a float (e.g., 5.0)
- "skip_eval_rsi_oversold": a float (e.g., 30.0)
- "skip_eval_rsi_overbought": a float (e.g., 70.0)
- "skip_eval_macd_hist_change": a float (e.g., 0.0005)
- "regime_adx_strong": a float (e.g., 40.0)
- "regime_adx_moderate": a float (e.g., 25.0)
- "regime_volatility_high_pct": a float (e.g., 80.0)
- "regime_volatility_low_pct": a float (e.g., 20.0)
- "regime_bb_squeeze_width": a float (e.g., 0.02)
- "regime_bb_expansion_width": a float (e.g., 0.08)
- "min_stop_loss_atr_mult": a float (e.g., 1.5)
- "min_max_hold_time_mult": a float (e.g., 2.0)
- "max_stop_loss_reviews": an integer between 1 and 20
- "max_take_profit_reviews": an integer between 1 and 20
- "max_partial_tp_reviews": an integer between 1 and 20
- "max_dust_sweep_reviews": an integer between 1 and 20
- "min_llm_pause_duration_seconds": an integer between 300 and 14400
- "pause_max_consecutive_keep": an integer between 1 and 10
- "pause_force_resume_risk_multiplier": a float between 0.0 and 1.0
- "max_portfolio_exposure_pct": a float between 0.0 and 1.0
- "max_portfolio_stop_risk_pct": a float between 0.0 and 1.0
- "min_risk_reward_ratio": a positive number
- "confidence_rejection_threshold": a float between 0.0 and 1.0
- "limit_price_max_distance_pct": an optional float between 0.0 and 1.0
- "min_viable_trade_amount": an optional positive number
- "stock_revaluation_interval_seconds": an optional integer >= 3600
- "pause_trading": optional boolean
- "pause_reason": optional string
- "pause_duration_seconds": optional positive integer
- "global_risk_multiplier": optional float (0.0-1.0)

Set max_portfolio_exposure_pct to at least 0.8 and max_portfolio_stop_risk_pct to at least 0.1 unless you have a very strong reason to be more conservative.

Output ONLY the raw JSON object."""
    return prompt


def build_strategy_prompt(
    symbol: str,
    ticker: Dict[str, Any],
    balance: Dict[str, float],
    open_positions: List[Dict[str, Any]],
    per_symbol_budget: float,
    max_symbols: int,
    base_currency: str,
    performance: Optional[Dict[str, Any]] = None,
    ohlcv_data: Optional[Dict[str, List]] = None,
    assigned_timeframe: Optional[str] = None,
    atr: Optional[float] = None,
    atr_multi_tf: Optional[Dict[str, float]] = None,
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
    stochastic_d: Optional[float] = None,
    adx: Optional[float] = None,
    plus_di: Optional[float] = None,
    minus_di: Optional[float] = None,
    obv: Optional[float] = None,
    mfi: Optional[float] = None,
    cci: Optional[float] = None,
    williams_r: Optional[float] = None,
    unrealized_pnl: Optional[float] = None,
    position_info: Optional[Dict[str, Any]] = None,
    drawdown_pct: Optional[float] = None,
    raw_candles: Optional[List[List]] = None,
    recent_trades: Optional[List[Dict[str, Any]]] = None,
    historical_ohlcv: Optional[List[List]] = None,
    min_order_amount: Optional[float] = None,
    min_order_cost: Optional[float] = None,
    all_symbols: Optional[List[Dict[str, str]]] = None,
    past_trades: Optional[List[Dict[str, Any]]] = None,
    cycle_spent: Optional[float] = None,
    remaining_balance: Optional[float] = None,
    market_regime: Optional[str] = None,
    multi_tf_raw_candles: Optional[Dict[str, List[List]]] = None,
    multi_tf_indicators: Optional[Dict[str, Dict[str, Any]]] = None,
    session_info: Optional[Dict[str, Any]] = None,
    sentiment_trend: Optional[float] = None,
    volume_trend: Optional[float] = None,
    ichimoku: Optional[Dict[str, Optional[float]]] = None,
    market_breadth: Optional[Dict[str, Any]] = None,
    full_market_breadth: Optional[Dict[str, Any]] = None,
    keltner_channels: Optional[Dict[str, float]] = None,
    donchian_channels: Optional[Dict[str, float]] = None,
    parabolic_sar: Optional[float] = None,
    atr_percentile: Optional[float] = None,
    global_risk_multiplier: Optional[float] = None,
    trading_paused: bool = False,
    max_hold_expired: bool = False,
    max_hold_expired_count: int = 0,
    stop_loss_triggered: bool = False,
    stop_loss_review_count: int = 0,
    take_profit_triggered: bool = False,
    take_profit_review_count: int = 0,
    partial_tp_triggered: bool = False,
    partial_tp_review_count: int = 0,
    partial_tp_triggered_levels: Optional[List[int]] = None,
    partial_tp_executed_levels: Optional[List[int]] = None,
    dust_sweep_triggered: bool = False,
    dust_sweep_review_count: int = 0,
    max_stop_loss_reviews: int = 10,
    max_take_profit_reviews: int = 10,
    max_partial_tp_reviews: int = 10,
    max_dust_sweep_reviews: int = 10,
    portfolio_exposure_pct: Optional[float] = None,
    portfolio_stop_risk_pct: Optional[float] = None,
    portfolio_total_value: Optional[float] = None,
    portfolio_open_count: int = 0,
    portfolio_available_capital: Optional[float] = None,
    last_decision: Optional[Dict[str, Any]] = None,
    minutes_to_market_close: Optional[int] = None,
    current_strategy_interval_seconds: Optional[int] = None,
    max_portfolio_exposure_pct: Optional[float] = None,
    max_portfolio_stop_risk_pct: Optional[float] = None,
    trade_pattern_analysis: Optional[Dict[str, Any]] = None,
    symbol_event: Optional[Dict[str, Any]] = None,
    queued_orders: Optional[List[Dict[str, Any]]] = None,
    fundamentals: Optional[Dict[str, Any]] = None,
    vwap: Optional[float] = None,
    daily_pivot_points: Optional[Dict[str, float]] = None,
    min_hold_time_mult: float = 1.0,
    min_stop_atr_mult: float = 1.0,
    min_viable_trade_amount: float = 0.0,
    historical_backtest_results: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Build a prompt to generate a trading strategy for a specific stock/ETF."""
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
            prompt += f"Other symbols being traded (you must leave budget for them): {symbol_list_str}\n"
        else:
            prompt += "This is the only symbol being traded; you may use the full budget.\n"
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
Your total available {base_currency} balance: {base_balance:.2f}
Suggested equal share per symbol (balance / max_symbols): {per_symbol_budget:.2f} {base_currency}
Maximum symbols to trade: {max_symbols}
"""
    # --- Portfolio exposure summary ---
    if portfolio_total_value is not None:
        prompt += f"\n**Portfolio Exposure Summary:**\n"
        prompt += f"  Total portfolio value: {portfolio_total_value:.2f} {base_currency}\n"
        prompt += f"  Open positions: {portfolio_open_count}\n"
        if portfolio_exposure_pct is not None:
            prompt += f"  Capital deployed: {portfolio_exposure_pct:.1f}% of portfolio\n"
        if portfolio_stop_risk_pct is not None:
            prompt += f"  Total stop-loss risk: {portfolio_stop_risk_pct:.2f}% of portfolio (loss if ALL stops hit)\n"
        if portfolio_available_capital is not None:
            prompt += f"  Available capital for new positions: {portfolio_available_capital:.2f} {base_currency}\n"
        if max_portfolio_exposure_pct is not None and max_portfolio_stop_risk_pct is not None:
            prompt += (
                f"Use this summary to decide position_size_fraction. If capital deployment is already high "
                f"(>{max_portfolio_exposure_pct*100:.0f}%) or total stop-loss risk is elevated "
                f"(>{max_portfolio_stop_risk_pct*100:.0f}%), reduce your position_size_fraction or output HOLD. "
                "If you have low exposure and low risk, you may allocate more capital to high-conviction trades.\n"
            )
        else:
            prompt += (
                "Use this summary to decide position_size_fraction. If capital deployment is already high "
                "or total stop-loss risk is elevated, reduce your position_size_fraction or output HOLD. "
                "If you have low exposure and low risk, you may allocate more capital to high-conviction trades.\n"
            )
    prompt += (
        "\n**Position Sizing — Single Hard Ceiling:** Your `position_size_fraction` is the PRIMARY driver. "
        "The engine computes a single hard ceiling from all portfolio risk caps (exposure limit, stop-loss risk limit, "
        "per-trade risk limit, remaining cycle budget) and caps your trade at that amount. "
        "The global risk multiplier and per-symbol multiplier scale your desired amount. "
        "Check the available capital and exposure/risk budgets above to ensure your fraction results in a "
        "profitable trade after fees. If the available capital is too small for a profitable trade, output HOLD.\n"
    )
    # --- Dynamic portfolio risk adjustment ---
    prompt += (
        "\n**Dynamic Portfolio Risk Adjustment:**\n"
        "You can include a `\"portfolio_risk_adjustment_factor\"` (0.1–1.0) in your strategy parameters. "
        "This is your per-symbol vote on the overall portfolio risk for the current cycle. "
        "The engine will take the **minimum** of this factor across all symbols evaluated in this cycle "
        "and apply it as a global multiplier to all position sizes. "
        "Use a lower value if you detect high volatility, an unfavorable market regime shift, or elevated risk. "
        "Use 1.0 (or omit the field) if conditions are normal. "
        "This gives you direct control over the global trading risk based on the latest market data.\n"
    )
    if cycle_spent is not None and remaining_balance is not None:
        prompt += (
            f"Amount already allocated to other symbols in this cycle: {cycle_spent:.2f} {base_currency}\n"
            f"Remaining available for this symbol: {remaining_balance:.2f} {base_currency}\n"
            "Your position_size_fraction must not require more than the remaining balance. "
            "If the remaining balance is low, reduce your fraction accordingly or output HOLD.\n"
        )
        # Help the LLM set min_profit_per_trade realistically
        max_possible_amount = min(per_symbol_budget, remaining_balance)
        prompt += (
            f"The maximum amount that can actually be allocated to this trade is "
            f"{max_possible_amount:.2f} {base_currency} (the smaller of the per‑symbol budget and the remaining balance). "
            "If you set `min_profit_per_trade`, ensure it is not larger than "
            "`max_possible_amount * take_profit_pct`. Otherwise the trade will be skipped.\n"
        )
    if global_risk_multiplier is not None and global_risk_multiplier < 1.0:
        prompt += (
            f"\n**Global risk multiplier is currently {global_risk_multiplier}.** "
            "All position sizes will be multiplied by this factor. "
            "The actual amount used will be: position_size_fraction × total_balance × global_risk_multiplier. "
            "Adjust your position_size_fraction accordingly – if you want a certain exposure, "
            "you may need to set a higher fraction to compensate, or accept the reduced size.\n"
        )
    prompt += (
        f"**position_size_fraction** represents a fraction of your **total {base_currency} balance** (0.01 to 1.0). "
        f"You decide the exact fraction dynamically — there is no equal-split requirement. "
        f"Allocate more to high-conviction setups and less to speculative ones. "
        f"**Important:** The sum of position_size_fraction across all stocks you intend to trade must not exceed 1.0 "
        f"(total allocated capital must not exceed your available {base_currency} balance). "
        f"Even with many symbols and limited capital, use small fractions (0.01–0.05) rather than skipping trades entirely.\n"
    )
    prompt += (
        "If you are uncertain about a trade, prefer a **small position** "
        "(`position_size_fraction` ≤ 0.1) with tight stops over outputting HOLD. "
        "Multiple small positions diversify risk and increase the chance of capturing profitable moves. "
        "Doing nothing guarantees zero profit.\n"
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
                    if age_sec < 60:
                        age_str = f" (placed {age_sec:.0f}s ago)"
                    elif age_sec < 3600:
                        age_str = f" (placed {age_sec/60:.1f}m ago)"
                    else:
                        age_str = f" (placed {age_sec/3600:.1f}h ago)"
                dist_str = ""
                if current_price is not None and limit_price is not None and current_price > 0:
                    if side == 'BUY':
                        dist_pct = ((limit_price - current_price) / current_price) * 100
                        if dist_pct > 0:
                            dist_str = f" (limit is {dist_pct:.2f}% above current price {current_price:.4f})"
                        else:
                            dist_str = f" (limit is {abs(dist_pct):.2f}% below current price – should be marketable)"
                    else:  # SELL
                        dist_pct = ((current_price - limit_price) / current_price) * 100
                        if dist_pct > 0:
                            dist_str = f" (limit is {dist_pct:.2f}% below current price {current_price:.4f})"
                        else:
                            dist_str = f" (limit is {abs(dist_pct):.2f}% above current price – should be marketable)"
                # --- Partial fill info ---
                filled_qty = q.get('filled_qty', 0.0)
                original_amount = q.get('original_amount')
                partial_str = ""
                if filled_qty > 0 and original_amount is not None and original_amount > 0:
                    if side == 'BUY':
                        # For buys, original_amount is quote amount; filled_qty is base quantity.
                        # We can show filled base qty and remaining base qty.
                        remaining_base = original_amount - filled_qty
                        partial_str = f" (partially filled: {filled_qty:.6f} base filled, {remaining_base:.6f} remaining)"
                    else:
                        # For sells, original_amount is base amount; filled_qty is base quantity.
                        remaining_base = original_amount - filled_qty
                        partial_str = f" (partially filled: {filled_qty:.6f} base filled, {remaining_base:.6f} remaining)"

                prompt += f"  - {side} limit order at {limit_price}{age_str}{dist_str}{partial_str}\n"

            prompt += (
                "A queued order means the bot has already placed a limit order that is waiting "
                "for the market price to reach the limit. **Do NOT output a new BUY or SELL signal "
                "for this symbol while a queued order exists.** The engine will ignore any new signal "
                "until the queued order fills or is cancelled. "
                "If the order is partially filled, the engine will automatically fill the remaining "
                "amount when the market price crosses the limit – you do not need to take any action. "
                "If you want to change the order, you must first cancel it (not possible via JSON) – "
                "instead, output HOLD and explain in reasoning.\n"
            )
    base_symbol = symbol
    quote_currency = base_currency
    if min_order_amount is not None or min_order_cost is not None:
        prompt += f"\nMinimum order size for {symbol}:"
        if min_order_amount is not None:
            prompt += f" {min_order_amount} {base_symbol}"
        if min_order_cost is not None:
            prompt += f" (or {min_order_cost} {quote_currency} cost)"
        prompt += (
            ". Your position_size_fraction must result in an order that meets both the minimum amount "
            "and the minimum cost. Use the current price to convert between amount and cost.\n"
        )
    if assigned_timeframe:
        prompt += f"\nAssigned trading timeframe for this stock: {assigned_timeframe}. Base your decision PRIMARILY on the OHLCV data for this timeframe.\n"
        if assigned_timeframe in ("1w", "1M", "3M", "6M", "1Y", "3Y", "5Y"):
            prompt += (
                f"**CRITICAL: {assigned_timeframe} is a PRIMARY timeframe.** "
                "All long-term timeframes (5Y, 3Y, 1Y, 6M, 3M, 1M, 1w) are equally valid primary timeframes and capture the largest, most reliable trends. "
                "You MUST focus on this timeframe to identify the primary long-term trend direction and strength. "
                "The largest profits come from holding positions that are in a strong long-term uptrend. "
                "Set max_hold_time_seconds appropriate for this timeframe (e.g., several months to years).\n"
            )
        elif assigned_timeframe in ("1d", "1h"):
            prompt += (
                f"**WARNING: {assigned_timeframe} is a short-term timeframe.** "
                "You should only be using this timeframe if long-term data (5Y, 3Y, 1Y, 6M, 3M, 1M) was unavailable. "
                "If long-term data IS available in the multi-timeframe section, you MUST base your primary decision on those longer timeframes instead.\n"
            )
    if market_regime:
        prompt += f"\nMarket regime: {market_regime}\n"

    if session_info:
        prompt += f"\nCurrent UTC hour: {session_info['utc_hour']} ({session_info['session']} session)\n"
    if minutes_to_market_close is not None:
        if minutes_to_market_close > 0:
            prompt += f"  Minutes until market close (5:30 PM Rome): {minutes_to_market_close}\n"
        else:
            prompt += "  Market is currently closed.\n"
    if current_strategy_interval_seconds is not None:
        prompt += f"  Current strategy evaluation interval for this symbol: {current_strategy_interval_seconds}s\n"

    # --- Volatility, order book imbalance, and position P&L context ---
    if atr is not None:
        prompt += f"ATR (14-period, {assigned_timeframe or 'default'}): {atr:.6f}\n"
    if atr is not None and current_price is not None and current_price > 0:
        atr_pct = atr / current_price
        min_sl = min_stop_atr_mult * atr_pct
        prompt += (
            f"\n**Current ATR%: {atr_pct:.4%}**. "
            f"The validator enforces a minimum fixed stop-loss of {min_stop_atr_mult} × ATR% = {min_sl:.4%}. "
            f"Your fixed stop_loss_pct must be at least this value.\n"
        )
    if atr_percentile is not None:
        prompt += f"ATR percentile (relative to last 100 observations): {atr_percentile:.1f}%\n"
    if atr_multi_tf:
        prompt += f"ATR across timeframes: {json.dumps(atr_multi_tf)}\n"
    # --- Transaction cost break-even calculation ---
    # Detect BTP bonds (ISIN format) to apply the correct fee structure.
    _is_btp = bool(re.match(r'^IT[A-Z0-9]{10}', symbol.split("/")[0]))
    trade_value = min(per_symbol_budget, remaining_balance if remaining_balance is not None else per_symbol_budget)
    if trade_value > 0:
        if _is_btp:
            # BTP fees: BTP_FEE_PERC with BTP_MIN_FEE, no Tobin tax, no fixed execution fee
            btp_fee_perc = settings.BTP_FEE_PERC
            btp_min_fee = settings.BTP_MIN_FEE
            if settings.BTP_IS_PRIMARY_ISSUANCE:
                buy_fee = 0.0
                sell_fee = 0.0
            else:
                buy_fee = max(btp_min_fee, trade_value * btp_fee_perc)
                sell_fee = max(btp_min_fee, trade_value * btp_fee_perc)
            total_fees = buy_fee + sell_fee
            break_even_pct = total_fees / trade_value
            prompt += (
                f"\n**Transaction Cost Break-Even (BTP Bond – Intesa Sanpaolo Investo):**\n"
                f"For a trade size of ~{trade_value:.2f} {quote_currency}:\n"
                f"  Estimated Buy Fee: {buy_fee:.2f} {quote_currency} (no Tobin tax)\n"
                f"  Estimated Sell Fee: {sell_fee:.2f} {quote_currency}\n"
                f"  Total Round-Trip Fees: {total_fees:.2f} {quote_currency} ({break_even_pct*100:.2f}% of trade value)\n"
                f"**Your `take_profit_pct` MUST be strictly greater than {break_even_pct*100:.2f}% to be profitable.**\n"
                f"BTP bonds have lower fees than stocks (no Tobin tax, no fixed execution fee).\n"
                f"Set your `take_profit_pct` comfortably above this break-even percentage.\n"
            )
        else:
            # Standard stock/ETF fees: max(3.50, V*0.0024) + 2.50 + V*0.0012 (buy)
            # max(3.50, V*0.0024) + 2.50 (sell)
            buy_fee = max(3.50, trade_value * 0.0024) + 2.50 + (trade_value * 0.0012)
            sell_fee = max(3.50, trade_value * 0.0024) + 2.50
            total_fees = buy_fee + sell_fee
            break_even_pct = total_fees / trade_value
            prompt += (
                f"\n**Transaction Cost Break-Even (Intesa Sanpaolo Investo):**\n"
                f"For a trade size of ~{trade_value:.2f} {quote_currency}:\n"
                f"  Estimated Buy Fee: {buy_fee:.2f} {quote_currency}\n"
                f"  Estimated Sell Fee: {sell_fee:.2f} {quote_currency}\n"
                f"  Total Round-Trip Fees: {total_fees:.2f} {quote_currency} ({break_even_pct*100:.2f}% of trade value)\n"
                f"**Your `take_profit_pct` MUST be strictly greater than {break_even_pct*100:.2f}% to be profitable.**\n"
                f"Set your `take_profit_pct` comfortably above this break-even percentage.\n"
            )
    prompt += (
        f"\n**Capital Allocation — Fully Dynamic:**\n"
        f"Your total available {base_currency} balance is {base_balance:.2f}.\n"
        f"The 'Suggested equal share per symbol' is ONLY a reference. You decide the actual `position_size_fraction`.\n"
        f"- Set `position_size_fraction` based on your confidence, the quality of this setup, volatility (ATR), "
        f"portfolio exposure, drawdown, and all other parameters. Each trade amount should be UNIQUE.\n"
        f"- **HYBRID ALLOCATION:** You may allocate ALL available capital to this single trade "
        f"if you believe it is highly profitable, even if this leaves no capital for other tickers. "
        f"However, if you can leave some capital for other promising setups, do so. "
        f"**Do NOT place a small trade that is unprofitable after fees** just because the equal-share budget seems small — "
        f"if the trade cannot be profitable with the available capital after accounting for transaction costs, "
        f"output HOLD instead of placing a tiny unprofitable trade.\n"
        f"- Even a very small `position_size_fraction` (e.g., 0.01–0.05) is valid if your conviction is low or "
        f"capital is limited, BUT ONLY if the trade is still profitable after fees at that size.\n"
        f"- The sum of `position_size_fraction` across all stocks must not exceed 1.0 (total available balance).\n"
    )
    prompt += (
        "\n**Confidence-Based Position Sizing:** You can set `confidence_sizing_weight` (0.0–1.0) in your strategy parameters "
        "to make your confidence score directly affect the position size. The effective position size is: "
        "`position_size_fraction × total_balance × (1.0 - confidence_sizing_weight × (1.0 - confidence))`. "
        "Set to 0.0 to disable (default). Set to 1.0 to make position size directly proportional to confidence. "
        "This makes your confidence score meaningful: high confidence → larger position, low confidence → smaller position.\n"
    )
    prompt += (
        f"- The only hard floor is the exchange minimum order size: your `position_size_fraction × total_balance` "
        f"must be ≥ the minimum order cost for this symbol (shown above as min_order_cost).\n"
        f"- If the remaining balance is too small to meet the exchange minimum, reduce your fraction or output HOLD.\n"
    )
    # --- Show the LLM its previous decision for this symbol ---
    if last_decision:
        age_seconds = time.time() - last_decision.get("timestamp", 0)
        prompt += (
            f"\n**Your previous decision for {symbol} (made {age_seconds:.0f}s ago):**\n"
            f"  Action: {last_decision.get('action')}\n"
            f"  Confidence: {last_decision.get('confidence', 0):.2f}\n"
            f"  Reasoning: {last_decision.get('reasoning', '')}\n"
        )
        sl_pct = last_decision.get("stop_loss_pct")
        tp_pct = last_decision.get("take_profit_pct")
        psf = last_decision.get("position_size_fraction")
        sl_method = last_decision.get("stop_loss_method")
        if sl_method:
            prompt += f"  Stop-loss method: {sl_method}\n"
        if sl_pct is not None:
            prompt += f"  Stop-loss pct: {sl_pct}\n"
        if tp_pct is not None:
            prompt += f"  Take-profit pct: {tp_pct}\n"
        if psf is not None:
            prompt += f"  Position size fraction: {psf}\n"
    if unrealized_pnl is not None and position_info:
        prompt += f"Current position unrealized P&L: {unrealized_pnl:.2f} {base_currency}\n"
        entry_price = position_info.get('price', 0)
        amount = position_info.get('amount', 0)
        prompt += f"Position details: entry price {entry_price}, amount {amount}\n"
        prompt += f"\n**You currently hold {amount:.6f} {base_symbol} at an average entry of {entry_price:.4f}.**\n"
        prompt += "If you output BUY, you will ADD to this existing position (scale in). If you output SELL, you will close the ENTIRE position.\n"
        # Compute and show P&L percentage explicitly so the LLM doesn't have to calculate it
        if entry_price > 0 and amount > 0:
            cost_basis = entry_price * amount
            if cost_basis > 0:
                pnl_pct = (unrealized_pnl / cost_basis) * 100
                prompt += f"Unrealized P&L percentage: {pnl_pct:+.2f}%\n"
        # Explicitly show current risk levels (these are otherwise buried in the open_positions JSON)
        current_sl = position_info.get('stop_loss')
        current_tp = position_info.get('take_profit')
        if current_sl is not None:
            prompt += f"Current stop-loss price: {current_sl:.6f}\n"
        if current_tp is not None:
            prompt += f"Current take-profit price: {current_tp:.6f}\n"
        # Show distance from current price to stop/TP as percentages
        if current_price and current_price > 0:
            if current_sl is not None:
                sl_distance_pct = ((current_price - current_sl) / current_price) * 100
                prompt += f"Distance to stop-loss: {sl_distance_pct:.2f}% below current price\n"
            if current_tp is not None:
                tp_distance_pct = ((current_tp - current_price) / current_price) * 100
                prompt += f"Distance to take-profit: {tp_distance_pct:.2f}% above current price\n"
        # Show trailing stop status
        trailing_active = position_info.get('trailing_stop', False)
        if trailing_active:
            trailing_dist = position_info.get('trailing_stop_distance_pct')
            trailing_act = position_info.get('trailing_stop_activation_pct')
            prompt += f"Trailing stop: enabled (distance={trailing_dist}, activation={trailing_act})\n"
        # Show max hold time remaining
        max_hold = position_info.get('max_hold_time_seconds')
        if max_hold is not None and max_hold > 0:
            entry_ts = position_info.get('timestamp', 0) / 1000.0
            elapsed = time.time() - entry_ts if entry_ts > 0 else 0
            remaining = max(0, max_hold - elapsed)
            prompt += f"Max hold time: {max_hold:.0f}s total, {remaining:.0f}s remaining\n"

    # --- Multi-timeframe OHLCV summary and indicators ---
    if multi_tf_raw_candles:
        tf_summaries = []
        for tf in settings.OHLCV_TIMEFRAMES:
            if tf in multi_tf_raw_candles:
                summary = _summarize_ohlcv(multi_tf_raw_candles[tf])
                if summary:
                    tf_summaries.append(
                        f"  [{tf}] change={summary['change_pct']}%, "
                        f"high={summary['high']}, low={summary['low']}, "
                        f"volume={summary['volume']}, candles={summary['candle_count']}"
                    )
        if tf_summaries:
            prompt += "\nMulti-timeframe OHLCV summary (price change %, high, low, volume, candle count):\n"
            prompt += "\n".join(tf_summaries) + "\n"
            prompt += (
                "Use these summaries to assess momentum and trend across timeframes. "
                "**CRITICAL: All long-term timeframes (5Y, 3Y, 1Y, 6M, 3M, 1M, 1w) are your PRIMARY timeframes and are equally important** — they show the dominant long-term trends that drive the largest profits. "
                "The 1d (daily) and 1h timeframes provide additional context for entry and exit timing only. "
                "You MUST always align your trading decision with the long-term trend direction.\n"
            )
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
                    if ind.get('ema_21') is not None:
                        ind_compact['ema21'] = round(ind['ema_21'], 4)
                if ind.get('stochastic_k') is not None:
                    ind_compact['stoch_k'] = round(ind['stochastic_k'], 2)
                    if ind.get('stochastic_d') is not None:
                        ind_compact['stoch_d'] = round(ind['stochastic_d'], 2)
                if ind.get('adx') is not None:
                    ind_compact['adx'] = round(ind['adx'], 2)
                    if ind.get('plus_di') is not None:
                        ind_compact['+di'] = round(ind['plus_di'], 2)
                    if ind.get('minus_di') is not None:
                        ind_compact['-di'] = round(ind['minus_di'], 2)
                if ind.get('obv') is not None: ind_compact['obv'] = round(ind['obv'], 2)
                if ind.get('mfi') is not None: ind_compact['mfi'] = round(ind['mfi'], 2)
                if ind.get('cci') is not None: ind_compact['cci'] = round(ind['cci'], 2)
                if ind.get('williams_r') is not None: ind_compact['wr'] = round(ind['williams_r'], 2)
                if ind.get('ichimoku') is not None:
                    ich = ind['ichimoku']
                    ind_compact['ich'] = {
                        "t": round(ich['tenkan_sen'], 4),
                        "k": round(ich['kijun_sen'], 4),
                        "sa": round(ich['senkou_span_a'], 4),
                        "sb": round(ich['senkou_span_b'], 4),
                        "cb": round(ich['cloud_bottom'], 4),
                        "ct": round(ich['cloud_top'], 4),
                    }
                if ind.get('donchian_channels') is not None:
                    dc = ind['donchian_channels']
                    ind_compact['dc'] = {"u": round(dc['upper'], 4), "m": round(dc['middle'], 4), "l": round(dc['lower'], 4)}
                if ind.get('atr') is not None: ind_compact['atr'] = round(ind['atr'], 6)
                if ind.get('parabolic_sar') is not None: ind_compact['sar'] = round(ind['parabolic_sar'], 6)
                if ind.get('keltner_channels') is not None:
                    kc = ind['keltner_channels']
                    ind_compact['kc'] = {"u": round(kc['upper'], 6), "m": round(kc['middle'], 6), "l": round(kc['lower'], 6)}
                
                if not ind_compact:
                    continue
                ind_lines.append(f"[{tf}] {json.dumps(ind_compact)}")
        if ind_lines:
            prompt += "\nComputed technical indicators per timeframe:\n"
            prompt += "**CRITICAL: Pay closest attention to the PRIMARY timeframes ([5Y], [3Y], [1Y], [6M], [3M], [1M], [1w]) — they are all equally important and define the primary long-term trend that must drive your decision.**\n"
            prompt += "\n".join(ind_lines) + "\n"
    elif raw_candles:
        summary = _summarize_ohlcv(raw_candles)
        if summary:
            prompt += (
                f"\nOHLCV summary for {assigned_timeframe} timeframe: "
                f"change={summary['change_pct']}%, high={summary['high']}, low={summary['low']}, "
                f"volume={summary['volume']}, candles={summary['candle_count']}\n"
            )
            # Only claim indicators are available if at least one key indicator is present
            has_indicators = any(v is not None for v in [rsi, macd, bb_upper, ema_9])
            if has_indicators:
                prompt += (
                    "The technical indicators (RSI, MACD, Bollinger Bands, EMA) have already been computed for you from this data. "
                    "Use them together with the summary to time entries and exits. "
                    "Explain in your reasoning how the indicators support your decision.\n"
                )
    if historical_ohlcv:
        # Provide a statistical summary instead of raw candles to reduce prompt size
        # and avoid "lost in the middle" syndrome. The Python backtester handles
        # the deterministic validation independently.
        hist_summary = _summarize_ohlcv(historical_ohlcv)
        if hist_summary:
            # Compute additional statistics for the last N candles
            closes = [c[4] for c in historical_ohlcv]
            volumes = [c[5] for c in historical_ohlcv]
            last_20 = closes[-20:] if len(closes) >= 20 else closes
            avg_close = sum(last_20) / len(last_20) if last_20 else 0
            max_close = max(last_20) if last_20 else 0
            min_close = min(last_20) if last_20 else 0
            avg_volume = sum(volumes[-20:]) / min(len(volumes), 20) if volumes else 0
            # Recent momentum (last 5 candles)
            last_5_closes = closes[-5:] if len(closes) >= 5 else closes
            if len(last_5_closes) >= 2 and last_5_closes[0] > 0:
                recent_momentum_pct = ((last_5_closes[-1] - last_5_closes[0]) / last_5_closes[0]) * 100
            else:
                recent_momentum_pct = 0.0

            prompt += (
                f"\nHistorical OHLCV statistical summary (last {hist_summary['candle_count']} candles, "
                f"{assigned_timeframe or 'default'} timeframe):\n"
                f"  Overall change: {hist_summary['change_pct']:.2f}%\n"
                f"  High: {hist_summary['high']:.4f}  Low: {hist_summary['low']:.4f}\n"
                f"  Total volume: {hist_summary['volume']:.0f}\n"
                f"  Last 20 candles — avg close: {avg_close:.4f}, max: {max_close:.4f}, min: {min_close:.4f}\n"
                f"  Avg volume (last 20): {avg_volume:.0f}\n"
                f"  Recent momentum (last 5 candles): {recent_momentum_pct:+.2f}%\n"
            )
            prompt += (
                f"\n**Available Historical Data for Backtest:** Up to {settings.OHLCV_RETENTION_DAYS} days "
                f"({settings.OHLCV_RETENTION_DAYS // 30} months) of historical OHLCV data is available on the "
                f"{assigned_timeframe or 'default'} timeframe. The statistical summary above covers "
                f"{hist_summary['candle_count']} candles.\n"
                "You MUST include `backtest_period_days` in your strategy parameters to tell the engine "
                "how many days of history to use for the backtest. Choose a period relevant to your strategy:\n"
                "- Long-term position (1w candles): 365–730 days\n"
                "- Very long-term analysis (1M candles): 730 days (all available data)\n"
                "- Medium-term swing (1d candles): 90–365 days\n"
                f"If omitted, the engine defaults to {settings.OHLCV_RETENTION_DAYS} days (all available data).\n"
            )
        prompt += (
            "**Step 1: Propose Multiple Backtest Variants**\n"
            "Based on the indicators and statistical summaries above, propose **multiple** sets of strategy parameters "
            "for backtesting. Each set is a \"backtest variant\" — a complete set of trading parameters "
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
        prompt += "Use these outcomes to adapt your strategy. If recent trades are losing, become more conservative.\n"

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
        prompt += (
            "Use these past outcomes to avoid repeating mistakes and reinforce successful patterns. "
            "Calibrate your confidence: if high-confidence trades are losing, lower confidence for similar setups; "
            "if low-confidence trades are winning, consider raising confidence.\n"
        )

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
        prompt += (
            "Use these historical results to avoid repeating failed parameter combinations "
            "and to build on strategies that have historically performed well for this symbol.\n"
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
            news_section = "Recent news articles for this stock:\n" + _format_news_for_prompt(articles)
    if news_section:
        prompt += f"\n{news_section}\n"
        prompt += (
            "**IMPORTANT:** Do not rely on pre-computed sentiment scores. Read the news headlines and summaries above "
            "and use your own understanding of financial context to assess the sentiment and potential impact. "
            "Factor this assessment into your confidence, position size, and reasoning.\n"
        )

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
**For the {assigned_timeframe or 'default'} timeframe, a reasonable minimum max_hold_time_seconds is {validator_min} seconds. Do not set it lower unless you have a very specific, justified reason (e.g., medium-term with a very tight stop and high confidence). For long-term candles (1w, 1M, 3M, 6M, 1Y, 3Y, 5Y), prefer max_hold_time_seconds of 2,592,000–94,608,000 seconds (1–36 months or more) to allow long-term trends to fully develop. The most profitable trades in stocks, ETFs, and BTPs come from holding positions for many months or years on long-term candles.

The validator enforces a hard minimum of {validator_min} seconds for this timeframe. Your max_hold_time_seconds must be at least this value.

You are trading spot only (no shorting). Only output SELL if you currently hold the asset.

**Note on BTP Bonds:** If the symbol is a BTP bond (ISIN format like IT0001234567), adjust your strategy for lower volatility: use longer max hold times, smaller take-profit targets, and ensure stop-losses are wide enough to avoid being triggered by normal bond price fluctuations. The `ticker` object includes `name`, `coupon` (annual coupon rate), and `maturity` (expiration date). Bond prices are quoted as a percentage of par value (e.g., a price of 101.68 means 101.68% of face value). 
**BTP Valuation:** To decide if this BTP is a good buy, calculate its approximate Yield to Maturity (YTM):
- `Annual Coupon = coupon × 100`
- `Years to Maturity = (maturity_date - current_date).days / 365`
- `Approximate YTM = (Annual Coupon + (100 - Current Price) / Years to Maturity) / ((100 + Current Price) / 2)`
If the YTM is attractive relative to current Italian government bond yields, it is a good buy. If the price is well above par (e.g., >110) and YTM is low, the upside is limited and there is higher downside risk if interest rates rise. ATR and OHLCV data are available for BTPs via yfinance and should be used for technical analysis like any other asset.
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
        prompt += "Use this fundamental data to assess valuation and long-term viability. For medium/long-term trades, prefer stocks with reasonable P/E ratios, strong profit margins, and solid return on equity. Avoid stocks that appear significantly overvalued unless there is a strong growth catalyst.\n"
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
        "  `stop_loss_pct`, `take_profit_pct`, `position_size_fraction`, `confidence_sizing_weight`, `trailing_stop`, `max_hold_time_seconds`, `cooldown_after_loss_seconds`, `backtest_period_days`, etc.\n"
        "- `backtest_variants`: a JSON array of objects, each containing a complete set of strategy parameters for backtesting. "
        "Each variant MUST include at minimum: `stop_loss_pct`, `take_profit_pct`, `max_hold_time_seconds`, `trailing_stop`, "
        "`position_size_fraction`, and `backtest_period_days`. You decide how many variants to return (minimum 1, recommended 3–5). "
        "Each variant should explore a different hypothesis (e.g., tight vs wide stop, short vs long hold, trailing on vs off, etc.). "
        "The engine will run a backtest for EACH variant and present ALL results in Step 2.\n"
        "- `entry_condition`: REQUIRED for every BUY action. An object specifying the exact moment to enter the trade (see Entry Condition section below for format).\n"
        "- `limit_price`: optional, a specific limit price for the order.\n"
        "- `time_in_force`: optional, \"day\" or \"gtc\". Default \"day\".\n"
        "You may include `\"portfolio_risk_adjustment_factor\"` (0.1–1.0) in the strategy parameters "
        "to vote on the overall portfolio risk for this cycle.\n"
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
            perf_text = "\n".join(perf_lines) + "\n\nUse this data to decide whether to BUY, SELL, or HOLD. If the stock has a poor win rate or the overall equity curve is declining, be more conservative. Prefer strategies that have worked well historically.\n"
        else:
            perf_text = ""
        perf_text += (
            "Calibrate parameters based on this data: low win rate/negative P&L → reduce size, widen stop, shorten hold time; "
            "high win rate/positive P&L → may increase size and tighten stops; high stop_loss_hits → wider stop or longer timeframe; "
            "use avg_hold_time_seconds to set realistic max_hold_time_seconds.\n"
        )
        prompt += perf_text
        daily_pnl = equity.get("daily_pnl", 0.0)
        total_pnl = equity.get("total_pnl", 0.0)
        consecutive_losses = equity.get("consecutive_losses", 0)
        if daily_pnl is not None:
            prompt += f"Today's realized P&L: {daily_pnl:.4f} {base_currency}\n"
        if consecutive_losses > 0:
            prompt += f"⚠️ You have {consecutive_losses} consecutive losing trades. Consider reducing risk or skipping this trade.\n"
        prompt += f"\n**Account P&L**: Total realized P&L = {total_pnl:.4f} {base_currency}.\n"
        if total_pnl < 0:
            prompt += "Account in loss – be more conservative (prefer HOLD unless exceptional opportunity).\n"
        else:
            prompt += "Account in profit – take calculated risks but only trade clear setups.\n"
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
            "This stock has upcoming or recent corporate events (e.g., earnings, FDA decision, merger). "
            "Such events can cause significant price gaps. You should decide whether to:\n"
            "- Avoid entering a new position before the event.\n"
            "- Reduce position size to limit gap risk.\n"
            "- Set wider stop-loss to accommodate event volatility.\n"
            "- Exit an existing position before the event if the risk is too high.\n"
            "The decision is entirely yours based on your assessment of the event's impact.\n"
        )
    return prompt


def build_analysis_prompt(**kwargs) -> str:
    """Build a focused prompt for Step 1a: Market analysis only.

    Reuses build_strategy_prompt for all market data context,
    but appends a simpler output format instruction at the end.
    No trading parameters, backtest variants, or entry conditions are requested.
    """
    full_prompt = build_strategy_prompt(**kwargs)

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


def build_backtest_variants_prompt(
    symbol: str,
    analysis: Dict[str, Any],
    ticker: Dict[str, Any],
    current_price: float,
    atr: Optional[float],
    assigned_timeframe: str,
    base_currency: str,
    base_balance: float,
    per_symbol_budget: float,
    min_order_amount: Optional[float] = None,
    min_order_cost: Optional[float] = None,
    remaining_balance: Optional[float] = None,
    portfolio_total_value: Optional[float] = None,
    portfolio_exposure_pct: Optional[float] = None,
    portfolio_stop_risk_pct: Optional[float] = None,
    portfolio_available_capital: Optional[float] = None,
    max_portfolio_exposure_pct: Optional[float] = None,
    max_portfolio_stop_risk_pct: Optional[float] = None,
    global_risk_multiplier: Optional[float] = None,
    min_stop_atr_mult: float = 1.0,
    min_hold_time_mult: float = 1.0,
    trading_paused: bool = False,
    has_position: bool = False,
    historical_backtest_results: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Build a focused prompt for Step 1b: Parameter selection and backtest variants.

    Given the analysis from Step 1a, asks the LLM to propose backtest variants
    with full parameters, entry conditions, and preliminary strategy parameters.
    The LLM does not need to re-analyze the market — it translates its analysis
    into concrete trading parameters.
    """
    from src.config.settings import settings as _settings
    import re as _re
    import time as _time

    tf_seconds = _timeframe_to_seconds(assigned_timeframe)

    # Cap validator minimum for long timeframes
    if assigned_timeframe in ("3Y", "5Y"):
        validator_min = 31_536_000
    elif assigned_timeframe in ("1Y", "6M"):
        validator_min = min(int(min_hold_time_mult * tf_seconds), 31_536_000)
    else:
        validator_min = int(min_hold_time_mult * tf_seconds)

    # Detect BTP for fee calculation
    _is_btp = bool(_re.match(r'^IT[A-Z0-9]{10}', symbol.split("/")[0]))
    trade_value = min(per_symbol_budget, remaining_balance if remaining_balance is not None else per_symbol_budget)

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
    if min_order_amount is not None:
        prompt += f"- Min order amount: {min_order_amount}\n"
    if min_order_cost is not None:
        prompt += f"- Min order cost: {min_order_cost:.2f} {base_currency}\n"

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
            buy_fee = max(3.50, trade_value * 0.0024) + 2.50 + (trade_value * 0.0012)
            sell_fee = max(3.50, trade_value * 0.0024) + 2.50
        total_fees = buy_fee + sell_fee
        break_even_pct = total_fees / trade_value
        prompt += (
            f"\n**Transaction Cost Break-Even:**\n"
            f"  Trade size: ~{trade_value:.2f} {base_currency}\n"
            f"  Total round-trip fees: {total_fees:.2f} {base_currency} ({break_even_pct*100:.2f}%)\n"
            f"  Your take_profit_pct MUST be > {break_even_pct*100:.2f}% to be profitable.\n"
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

    prompt += f"""
**Backtest Entry Logic (REQUIRED):**
You MUST include a `backtest_entry_config` object in EVERY backtest variant. If omitted, the backtest will fail with an error and no results will be produced.
Supported fields: ema_period, ema_direction, min_adx, max_rsi, min_rsi, macd_filter, logic.
Example: {{"backtest_entry_config": {{"ema_period": 21, "ema_direction": "above", "min_adx": 25, "max_rsi": 65, "macd_filter": "positive", "logic": "and"}}}}

**Validator Constraints:**
- Minimum max_hold_time_seconds for {assigned_timeframe}: {validator_min} seconds
- Minimum stop-loss: {min_stop_atr_mult} × ATR% (if ATR available)
- take_profit_pct MUST be strictly greater than stop_loss_pct

**Position Sizing:**
- position_size_fraction represents a fraction of your total {base_currency} balance (0.01 to 1.0)
- The sum of position_size_fraction across all stocks must not exceed 1.0
- Use small fractions (0.01–0.05) for low conviction, larger for high conviction
- If remaining balance is too small for a profitable trade after fees, set action to HOLD

**Entry Condition (REQUIRED for every BUY):**
Include an `entry_condition` object. Supported types: limit_price, rsi_threshold, delay, indicator_combo.
Example: {{"type": "limit_price", "price": 1.23, "timeout_seconds": 3600}}
Minimum timeout: max(300, {_settings.ENTRY_CONDITION_MIN_TIMEOUT_MULT} × candle timeframe seconds).

**Output ONLY the raw JSON object as specified.**

Return a JSON object with these **required** fields:
- `action`: one of BUY, SELL, HOLD (should match your Step 1a analysis)
- `confidence`: a float between 0.0 and 1.0 (should match your Step 1a analysis)
- `reasoning`: a string explaining your parameter choices. You MUST include the current market price.
- `strategy`: an object containing `type` (string) and `parameters` (object).
  The `parameters` object MUST include ALL required trading parameters:
  `stop_loss_pct`, `take_profit_pct`, `position_size_fraction`, `confidence_sizing_weight`,
  `trailing_stop`, `max_hold_time_seconds`, `cooldown_after_loss_seconds`, `backtest_period_days`, etc.
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
        backtest_sections.append(
            f"**Variant {i+1}:**\n"
            f"Parameters: {json.dumps(variant_params, indent=2)}\n"
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
        "  `stop_loss_pct`, `take_profit_pct`, `position_size_fraction`, `confidence_sizing_weight`, `trailing_stop`, `max_hold_time_seconds`, `cooldown_after_loss_seconds`, etc.\n"
        "  You may adjust `position_size_fraction` based on backtest performance (e.g., reduce size if drawdown is high).\n"
        "\n**CRITICAL — Confidence must reflect ALL available data:**\n"
        "Your `confidence` value must be your holistic assessment based on EVERYTHING provided to you:\n"
        "- **Backtest results**: win rate, profit factor, max drawdown, Sharpe ratio, total P&L\n"
        "- **Technical indicators**: RSI, MACD, Bollinger Bands, EMAs, ADX, Stochastic, MFI, CCI, Williams %R, Ichimoku, Parabolic SAR, VWAP, Donchian Channels, Keltner Channels, ATR percentile\n"
        "- **News sentiment**: aggregate compound score, sentiment trend, individual article headlines and summaries\n"
        "- **Market regime**: trend strength, volatility level, Bollinger Band state\n"
        "- **Market breadth**: percentage of positive stocks, full market breadth\n"
        "- **Fundamentals**: P/E ratio, market cap, dividend yield, profit margins, return on equity (when available)\n"
        "- **Historical performance**: your past win rate and P&L for this symbol and strategy type\n"
        "- **Trade pattern analysis**: which conditions, timeframes, and confidence ranges have historically worked\n"
        "- **Portfolio context**: current exposure, stop-loss risk, drawdown, consecutive losses\n"
        "Do NOT set confidence arbitrarily. If the backtest shows poor results, lower your confidence. "
        "If indicators conflict with your proposed direction, lower your confidence. "
        "If news sentiment strongly opposes your direction, lower your confidence. "
        "Only set high confidence (≥0.7) when backtest results, indicators, AND sentiment all align favorably.\n"
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
