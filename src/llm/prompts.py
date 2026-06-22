import json
import logging
import re
import time
from typing import List, Dict, Any, Optional, Tuple
from src.config.settings import settings
from src.database import get_news_for_symbol, get_aggregate_sentiment_from_db
from src.utils.redis_client import get_redis_client
from src.llm.cache import get_cached_llm_response
from src.exchanges.market_data import TIMEFRAME_MAP
from src.indicators import (
    compute_atr,
    compute_rsi,
    compute_ema,
    compute_stochastic,
    compute_adx,
    compute_obv,
    compute_mfi,
    compute_cci,
    compute_williams_r,
    compute_vwap,
    compute_ichimoku,
    compute_parabolic_sar,
    compute_keltner_channels,
    compute_pivot_points,
    compute_donchian_channels,
    compute_macd,
    compute_bollinger_bands,
)

logger = logging.getLogger(__name__)


def _timeframe_to_seconds(tf: str) -> int:
    """Convert a timeframe string (e.g., '5m', '1h') to seconds."""
    match = re.match(r'^(\d+)([mhdwM])$', tf)
    if not match:
        return 3600  # default 1h
    amount = int(match.group(1))
    unit = match.group(2)
    mult = {'m': 60, 'h': 3600, 'd': 86400, 'w': 604800, 'M': 2592000}
    return amount * mult.get(unit, 3600)


def compact_prompt(text: str) -> str:
    """Collapse all whitespace sequences to a single space and strip."""
    return re.sub(r'\s+', ' ', text).strip()


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
    redis_client.setex(cache_key, ttl, json.dumps(result))
    return result


SYSTEM_PROMPT = """You are a professional stock and ETF trading bot assistant focused on medium to long-term investment horizons. Your primary goal is to generate consistent profit by identifying stocks with strong fundamentals, solid momentum, and favorable macro conditions over weeks to months. You must avoid large drawdowns and only trade when there is a clear edge.

Key principles:
- **Confidence is your directional conviction, not a trade gate.** Set confidence between 0.0 and 1.0. 0.0 → no conviction (should be HOLD). 0.5 → moderate belief. 1.0 → absolute certainty. Only output HOLD when you have no directional edge at all.
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
- **Note on costs:** The simulator applies a configurable fee per trade. Ensure your `take_profit_pct` is large enough to cover the fees.
- **Required parameter for every BUY/SELL:**
  - `"take_profit_pct"`: a decimal between 0.005 and 2.0 (e.g., 0.05 for 5%).

**Max Hold Time:**
- Set a maximum hold time (max_hold_time_seconds) for every trade. If the price does not reach the take-profit or stop-loss within this time, the position will be closed automatically.
- **Do NOT set max_hold_time_seconds too short.** A too-short max hold time forces an exit before the trade has time to develop. Err on the side of longer hold times. For 1h candles, consider at least 1-3 days; for 4h candles, 1-2 weeks; for 1d candles, 1-2 months.
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
You are a professional trading bot. Your primary goal is to generate consistent profit over medium to long-term horizons. Do not be overly conservative – take calculated risks. If market conditions are not extremely hostile (e.g., market breadth > 30%), you should be trading at least 1–2 stocks with small positions to probe for opportunities. Avoid staying idle for long periods. A cautious small trade is almost always better than doing nothing.

- You must set a cooldown duration (`cooldown_after_loss_seconds`) for every BUY. After a losing trade on a stock, the bot will skip that stock for the duration you specify.
- Set `cooldown_after_loss_seconds` to **0** (no cooldown) unless you have a very strong reason to avoid a stock. Quick re‑entry after a small loss is often profitable. Long cooldowns cause missed opportunities.
- You may include `"max_portfolio_exposure_pct"` (0.0-1.0) and `"max_portfolio_stop_risk_pct"` (0.0-1.0) in your stock selection JSON to define the maximum portfolio exposure and total stop-loss risk you are willing to accept. The engine will use these thresholds to guide position sizing in the strategy step.
- You may include `"min_risk_reward_ratio"` (a positive number, e.g., 1.5) in your stock selection JSON to define a global minimum reward:risk ratio for all trades in this cycle. The validator will reject any trade where `take_profit_pct / stop_loss_pct` is below this value, unless you explicitly override it with a different value in the strategy step.
- If the daily realized P&L is deeply negative or market conditions are poor, you may select 0 stocks in the stock selection step. This will pause trading until the next evaluation cycle. When you do this, always set a meaningful `pause_duration_seconds` (≥ 1800) to avoid an immediate re‑pause.
- If no high‑confidence setups exist but market conditions are not extremely hostile, you may still select 1–2 stocks with **small position sizes** (`position_size_fraction` ≤ 0.2) and **tight stops** to probe the market. Do NOT pause completely just because the perfect setup is absent – a small, cautious trade is often better than idling.
- **Required parameter for every BUY/SELL:**
  - `"cooldown_after_loss_seconds"`: a non-negative integer (0 or more). If the trade results in a loss, the bot will avoid this stock for this many seconds before considering it again. Set 0 to allow immediate re-entry.
- **Optional parameters:**
  - `"position_size_fraction"`: a decimal between 0.1 and 1.0 representing the fraction of your **total available cash balance** to allocate to this trade. Must be > 0 and ≤ 1. The sum of this fraction across all stocks you trade should not exceed 1.0.
  - `"max_risk_per_trade_pct"`: a decimal between 0 and 1.0 (e.g., 0.02 for 2% of portfolio). The position size will be limited so that the potential loss (entry - stop) does not exceed this fraction of your total portfolio value.
  - `"max_portfolio_risk_pct"`: an optional decimal between 0 and 1.0 (e.g., 0.06 for 6% of portfolio). If set, the bot will calculate the total potential loss of all open positions plus the potential loss of this new trade. If this total exceeds this percentage of your total portfolio value, the trade will be skipped.
  - `"min_profit_per_trade"`: an optional non-negative number (in base currency, e.g., 0.5). If set, the bot will skip the trade if the expected gross profit (position size × take_profit_pct) is below this value. Set `min_profit_per_trade` to **0** (or a very small value like 0.01) to allow tiny profits. Do not block trades because the expected profit is small – a small profit is still profit.
  - `"min_risk_reward_ratio"`: an optional positive number (e.g., 1.5). If set, the validator will reject the trade unless take_profit_pct / stop_loss_pct >= this value.
  - `"position_size_multiplier"`: an optional decimal between 0.0 and 1.0 (e.g., 0.5 for 50%). If set, the final position size for this trade will be further multiplied by this factor, after the global risk multiplier.
  - `"min_confidence"`: an optional decimal between 0.0 and 1.0 (e.g., 0.6). If set, the bot will skip the trade if your confidence is below this threshold.
- `"portfolio_risk_adjustment_factor"`: an optional decimal between 0.1 and 1.0 (e.g., 0.5). This is your per-symbol "vote" on the overall portfolio risk for the current cycle. The engine will take the **minimum** of this factor across all symbols evaluated in the current cycle and apply it as a global multiplier to all position sizes. Use a lower value (e.g., 0.3–0.5) if you detect high volatility, unfavorable market regime shifts, or elevated risk for this symbol. Use 1.0 if conditions are normal and you see no reason to reduce overall portfolio risk. This allows you to dynamically adjust the global trading risk based on the latest per-symbol market data, rather than relying solely on the periodic stock selection phase.

**Pause/Resume:**
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

**Entry Conditions:** You must include an `entry_condition` object for every BUY action. The strategy prompt provides full details and examples.

"""

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
    ohlcv_data: Optional[Dict[str, Dict[str, List]]] = None,
    market_trend: Optional[Dict[str, Any]] = None,
    news_sentiment: Optional[Dict[str, Dict[str, Any]]] = None,
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
) -> str:
    """Build a prompt to ask the LLM which stocks/ETFs to trade."""
    # Summarize tickers and limits for the prompt
    ticker_summary = {}
    for symbol in available_symbols:
        if symbol in tickers:
            t = tickers[symbol]
            limits = market_limits.get(symbol, {})
            ticker_summary[symbol] = {
                "last": t.get("last"),
                "change_24h": t.get("percentage"),
                "volume": t.get("quoteVolume"),
                "min_trade_cost": limits.get("min_cost"),  # now always a number
            }
            if settings.NEWS_ENABLED:
                agg = get_aggregate_sentiment_from_db(symbol, max_age_seconds=settings.NEWS_CACHE_TTL_SECONDS)
                if agg:
                    ticker_summary[symbol]["sentiment"] = agg

    # Build OHLCV summary if provided
    ohlcv_summary = {}
    if ohlcv_data:
        for symbol in available_symbols:
            if symbol in ohlcv_data:
                tf_data = ohlcv_data[symbol]
                summary = {}
                for tf, candles in tf_data.items():
                    if not candles:
                        continue
                    open_price = candles[0][1]
                    close_price = candles[-1][4]
                    high = max(c[2] for c in candles)
                    low = min(c[3] for c in candles)
                    volume = sum(c[5] for c in candles)
                    change_pct = ((close_price - open_price) / open_price) * 100 if open_price else 0
                    summary[tf] = {
                        "change_pct": round(change_pct, 2),
                        "high": high,
                        "low": low,
                        "volume": volume,
                    }
                ohlcv_summary[symbol] = summary

    # --- News section ---
    news_section = ""
    if settings.NEWS_ENABLED:
        news_lines = []
        symbols_to_check = available_symbols[:50]
        for sym in symbols_to_check:
            articles = get_news_for_symbol(sym, max_age_seconds=settings.NEWS_CACHE_TTL_SECONDS)
            if articles:
                formatted = _format_news_for_prompt(articles)
                news_lines.append(f"**{sym}**\n{formatted}")
        if news_lines:
            news_section = "Recent news for top stocks:\n\n" + "\n\n".join(news_lines)

    available_timeframes = [tf for tf in settings.OHLCV_TIMEFRAMES if tf in TIMEFRAME_MAP]
    prompt = f"""Current base currency: {base_currency}
Your available {base_currency} balance: {base_balance:.2f}
Maximum number of stocks to trade: {max_symbols}
Budget per stock: {per_symbol_budget:.2f} {base_currency}
Available timeframes: {json.dumps(available_timeframes)}
Currently tracked stocks (with assigned timeframes): {json.dumps(current_symbols) if current_symbols else "None"}"""

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
{json.dumps(ticker_summary, indent=2)}

Select between 0 and {max_symbols} stocks to trade. If market conditions are extremely unfavorable (e.g., high losses, poor momentum, negative sentiment), you may select 0 stocks to pause trading until the next evaluation. You decide the exact number based on how many high‑quality opportunities you see. If market conditions are poor, you may choose fewer stocks (even 0 or 1) to concentrate capital on the best setup. If many strong setups exist, you may select up to {max_symbols}. You MUST only select stocks where the per-stock budget ({per_symbol_budget:.2f} {base_currency}) is greater than or equal to the stock's min_trade_cost. Skip any stock that does not meet this requirement. Prefer stocks with high volume and positive momentum. You may keep some current stocks if they are still promising and meet the budget requirement, or replace them. **Prefer to keep stocks that have been tracked for a while** – they have more historical data and the bot has already invested in learning their behaviour. Only drop a stock if it shows clear deterioration (e.g., negative momentum on all timeframes, poor win rate, or strongly negative sentiment). For stocks already being tracked, re-evaluate their assigned timeframe. If the market regime has changed (e.g., a stock that was trending on 1h is now choppy and better suited to 15m), update the timeframe. If you change the timeframe for a stock with an open position, the bot will switch to managing the position using the new timeframe.

**Important:** Unless the market is in a clear crisis (e.g., breadth < 20%, account in deep drawdown), you MUST select at least 1–2 stocks with **small position sizes** (position_size_fraction ≤ 0.15) and **tight stops**. Doing nothing guarantees zero profit; a cautious small trade at least gives a chance. Only select 0 stocks if conditions are truly hostile.

Each symbol can only appear once in your selection. Choose the single best timeframe for each stock based on the multi-timeframe OHLCV data.

**Output ONLY the raw JSON object as specified.**

Return a JSON object with the following fields:
- "stocks": a JSON array of objects, each with "symbol", "timeframe" (the timeframe must be one of the available timeframes, e.g., {', '.join([repr(tf) for tf in available_timeframes])}), and "sector" (a string representing the stock's sector, e.g., "Technology", "Healthcare", "Financials", "Energy", "Consumer Discretionary", "Consumer Staples", "Industrials", "Materials", "Real Estate", "Utilities", "Communication Services"). Each object may optionally include "max_tenure_hours" (a positive float, hours) to force-sell the stock after that many hours in the portfolio. Omit or set to null for no limit.
- "max_stocks": an integer between 0 and {max_symbols} indicating how many stocks you actually want to trade. Set to 0 to pause trading. This must equal the length of the "stocks" array.
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
- "min_llm_pause_duration_seconds": an integer between 300 and 14400 (e.g., 3600). The minimum duration in seconds that the LLM must wait before it can resume trading after pausing. This prevents rapid pause/resume cycles. Lower values = faster resume capability; higher values = more conservative cooldown.
- "pause_max_consecutive_keep": an integer between 1 and 10 (e.g., 3). The maximum number of consecutive "keep paused" decisions the LLM can make before the engine force-resumes trading with a reduced risk multiplier. Lower values = quicker force-resume; higher values = more patience with LLM's pause decisions.
- "pause_force_resume_risk_multiplier": a float between 0.0 and 1.0 (e.g., 0.3). The global risk multiplier applied when the engine force-resumes trading after too many consecutive "keep paused" decisions. Lower values = more conservative forced resume; higher values = more aggressive.
- "max_portfolio_exposure_pct": a float between 0.0 and 1.0 (e.g., 0.7 for 70%). The maximum percentage of total portfolio value that can be deployed in open positions.
- "max_portfolio_stop_risk_pct": a float between 0.0 and 1.0 (e.g., 0.05 for 5%). The maximum total stop-loss risk as a percentage of portfolio value.
- "min_risk_reward_ratio": a positive number (e.g., 1.5). The minimum reward:risk ratio required for all trades. Trades with a lower ratio will be rejected.
- "limit_price_max_distance_pct": an optional float between 0.0 and 1.0 (e.g., 0.05 for 5%). The maximum allowed distance of a limit price from the current best bid/ask. Orders with a limit price further away than this are rejected to avoid indefinite queuing. Set to 0.0 to disable the check entirely. If omitted, the engine uses its default (0.05).
- "reasoning": a short string (max 200 characters) explaining why you selected these specific stocks and timeframes. This will be shown to the user, so make it informative.

You may optionally include "stock_revaluation_interval_seconds" (integer >= 60) to change how often the bot re-evaluates the stock list.

Example: {{"stocks": [{{"symbol": "ENI.MI/EUR", "timeframe": "1h", "sector": "Energy", "max_tenure_hours": 48}}, {{"symbol": "ENEL.MI/EUR", "timeframe": "15m", "sector": "Utilities"}}, {{"symbol": "STM.MI/EUR", "timeframe": "5m", "sector": "Technology"}}], "max_stocks": 2, "max_positions_per_sector": 2, "skip_eval_price_change_atr_mult": 0.5, "skip_eval_rsi_change": 5.0, "skip_eval_rsi_oversold": 30.0, "skip_eval_rsi_overbought": 70.0, "skip_eval_macd_hist_change": 0.0005, "regime_adx_strong": 40.0, "regime_adx_moderate": 25.0, "regime_volatility_high_pct": 80.0, "regime_volatility_low_pct": 20.0, "regime_bb_squeeze_width": 0.02, "regime_bb_expansion_width": 0.08, "min_stop_loss_atr_mult": 1.5, "min_max_hold_time_mult": 1.5, "max_stop_loss_reviews": 3, "max_take_profit_reviews": 3, "min_llm_pause_duration_seconds": 3600, "pause_max_consecutive_keep": 3, "pause_force_resume_risk_multiplier": 0.3, "max_partial_tp_reviews": 3, "max_dust_sweep_reviews": 3, 
"reasoning": "ENI shows strong uptrend on 1h with high volume; ENEL has bullish MACD crossover on 15m.", "stock_revaluation_interval_seconds": 300, "max_portfolio_exposure_pct": 0.8, "max_portfolio_stop_risk_pct": 0.1, "min_risk_reward_ratio": 1.5, "limit_price_max_distance_pct": 0.05, "pause_trading": false, "pause_reason": "Market conditions are favorable"}}

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
        prompt += f"\nMulti-timeframe OHLCV summary (price change %, high, low, volume):\n{json.dumps(ohlcv_summary, indent=2)}\n"
    else:
        prompt += (
            "\n**Note:** No OHLCV data is available for any candidate symbol. "
            "You must base your selection entirely on ticker data (price, 24h change, volume), "
            "news sentiment, trend quality scores, and other provided metrics. "
            "Do not pause trading solely due to missing OHLCV if other indicators suggest strong opportunities.\n"
        )
    if correlation_matrix:
        # Trim to only include symbols that appear in the candidate list
        trimmed = {}
        for sym_a, row in correlation_matrix.items():
            trimmed[sym_a] = {sym_b: v for sym_b, v in row.items()}
        prompt += (
            "\nPairwise correlation matrix (Pearson correlation of daily returns, range -1 to +1):\n"
            f"{json.dumps(trimmed, indent=2)}\n"
        )
    if historical_ohlcv_summary:
        prompt += (
            "\nHistorical OHLCV summary from database (up to 30 days, price change %, high, low, volume, candle count):\n"
            f"{json.dumps(historical_ohlcv_summary, indent=2)}\n"
        )
    if symbol_indicators:
        prompt += "\nTechnical indicators for candidate stocks:\n"
        for sym, tf_indicators in symbol_indicators.items():
            lines = [f"{sym}:"]
            for tf, ind in tf_indicators.items():
                lines.append(f"  [{tf}]")
                if ind.get('rsi') is not None:
                    lines.append(f"    RSI(14)={ind['rsi']:.2f}")
                if ind.get('macd') is not None:
                    lines.append(f"    MACD={ind['macd']:.4f} Signal={ind['macd_signal']:.4f} Hist={ind['macd_hist']:.4f}")
                if ind.get('bb_upper') is not None:
                    lines.append(f"    BB Upper={ind['bb_upper']:.4f} Middle={ind['bb_middle']:.4f} Lower={ind['bb_lower']:.4f}")
                if ind.get('ema_9') is not None:
                    lines.append(f"    EMA9={ind['ema_9']:.4f} EMA21={ind['ema_21']:.4f}")
                if ind.get('stochastic_k') is not None:
                    d_str = f"{ind['stochastic_d']:.2f}" if ind['stochastic_d'] is not None else "N/A"
                    lines.append(f"    Stoch %K={ind['stochastic_k']:.2f} %D={d_str}")
                if ind.get('adx') is not None:
                    lines.append(f"    ADX(14)={ind['adx']:.2f} +DI={ind['plus_di']:.2f} -DI={ind['minus_di']:.2f}")
                if ind.get('obv') is not None:
                    lines.append(f"    OBV={ind['obv']:.2f}")
                if ind.get('mfi') is not None:
                    lines.append(f"    MFI(14)={ind['mfi']:.2f}")
                if ind.get('cci') is not None:
                    lines.append(f"    CCI(20)={ind['cci']:.2f}")
                if ind.get('williams_r') is not None:
                    lines.append(f"    Williams %R(14)={ind['williams_r']:.2f}")
                if ind.get('ichimoku') is not None:
                    ich = ind['ichimoku']
                    lines.append(f"    Ichimoku: Tenkan={ich['tenkan_sen']:.4f} Kijun={ich['kijun_sen']:.4f} SpanA={ich['senkou_span_a']:.4f} SpanB={ich['senkou_span_b']:.4f} Cloud={ich['cloud_bottom']:.4f}-{ich['cloud_top']:.4f}")
                if ind.get('donchian_channels') is not None:
                    dc = ind['donchian_channels']
                    lines.append(f"    Donchian: Upper={dc['upper']:.4f} Middle={dc['middle']:.4f} Lower={dc['lower']:.4f}")
                if ind.get('keltner_channels') is not None:
                    kc = ind['keltner_channels']
                    lines.append(f"    Keltner: Upper={kc['upper']:.6f} Middle={kc['middle']:.6f} Lower={kc['lower']:.6f}")
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
    if news_sentiment:
        prompt += "\n## News Sentiment\n"
        prompt += "Aggregate sentiment from recent news articles (compound score -1 to +1, higher = more positive):\n"
        for sym in available_symbols:
            if sym in news_sentiment:
                ns = news_sentiment[sym]
                prompt += (
                    f"- {sym}: compound={ns['avg_compound']}, "
                    f"positive={ns['positive']}, negative={ns['negative']}, "
                    f"neutral={ns['neutral']}, total_articles={ns['total_articles']}\n"
                )
    if sentiment_trend:
        prompt += "\nSentiment trend (change in compound score since last cycle):\n"
        for base, delta in sentiment_trend.items():
            if delta is not None:
                prompt += f"  {base}: {delta:+.4f}\n"
    if news_section:
        prompt += f"\n{news_section}\n"
    if performance:
        perf_text = f"""
Historical Performance Data:
Overall equity curve: {json.dumps(performance.get('equity_curve', {}))}
Per-stock performance (win rate, avg P&L, total trades): {json.dumps(performance.get('stock_performance', {}), indent=2)}
Per-strategy performance: {json.dumps(performance.get('strategy_performance', {}), indent=2)}
"""
        prompt += perf_text
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
    fee_rate: Optional[float] = None,
    drawdown_pct: Optional[float] = None,
    raw_candles: Optional[List[List]] = None,
    recent_trades: Optional[List[Dict[str, Any]]] = None,
    historical_ohlcv: Optional[List[List]] = None,
    min_order_amount: Optional[float] = None,
    min_order_cost: Optional[float] = None,
    all_symbols: Optional[List[Dict[str, str]]] = None,
    past_trades: Optional[List[Dict[str, Any]]] = None,
    aggregate_sentiment: Optional[Dict[str, Any]] = None,
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
) -> str:
    """Build a prompt to generate a trading strategy for a specific stock/ETF."""
    current_price = ticker.get("last") if ticker else None
    if assigned_timeframe and assigned_timeframe not in TIMEFRAME_MAP:
        logger.warning(f"Assigned timeframe {assigned_timeframe} is not supported by yfinance. Falling back to default.")
        assigned_timeframe = "1h" if "1h" in TIMEFRAME_MAP else list(TIMEFRAME_MAP.keys())[0]
    tf_seconds = _timeframe_to_seconds(assigned_timeframe) if assigned_timeframe else 3600
    min_hold = 2 * tf_seconds
    prompt = f"""Symbol: {symbol}
Current ticker: {json.dumps(ticker)}
Current balances: {json.dumps(balance)}
"""
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
    prompt += f"""Open positions: {json.dumps(open_positions)}
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
        f"**position_size_fraction** now represents a fraction of your **total {base_currency} balance** (0.1 to 1.0). "
        f"You may allocate more than the equal share for high‑confidence/high‑profit opportunities, and less for riskier ones. "
        f"**Important:** The sum of position_size_fraction across all stocks you intend to trade must not exceed 1.0, "
        f"so that you leave enough capital for other stocks. Plan your allocations accordingly.\n"
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
        prompt += f"\nAssigned trading timeframe for this stock: {assigned_timeframe}. Base your decision primarily on the OHLCV data for this timeframe.\n"
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
        min_sl = 1.5 * atr_pct
        prompt += f"\n**Current ATR%: {atr_pct:.4%}**. Minimum fixed stop: {min_sl:.4%}.\n"
    if atr_percentile is not None:
        prompt += f"ATR percentile (relative to last 100 observations): {atr_percentile:.1f}%\n"
    if atr_multi_tf:
        prompt += f"ATR across timeframes: {json.dumps(atr_multi_tf)}\n"
    if fee_rate is not None:
        prompt += f"Taker fee rate for this symbol: {fee_rate*100:.2f}%\n"
        # Calculate exact break-even take-profit percentage
        # Round trip: buy at P, sell at P*(1+TP). Fees: P*fee + P*(1+TP)*fee
        # Break even: P*(1+TP) - P*(1+TP)*fee - P - P*fee = 0
        # 1 + TP - fee - TP*fee - 1 - fee = 0
        # TP(1 - fee) = 2*fee
        # TP = 2*fee / (1 - fee)
        if fee_rate < 1.0:
            break_even_tp_pct = (2 * fee_rate) / (1 - fee_rate)
        else:
            break_even_tp_pct = 0.0
        
        min_profitable_tp_pct = break_even_tp_pct
        
        prompt += f"**Break-even take_profit_pct (fees + spread): {min_profitable_tp_pct:.4%}. Set your take_profit_pct strictly above this.**\n"
    # Help the LLM set min_profit_per_trade by showing the expected profit for a 1% take-profit
    example_tp = 0.01
    example_profit = per_symbol_budget * example_tp
    prompt += (
        f"For reference, a 1% take-profit on the per-symbol budget ({per_symbol_budget:.2f} {quote_currency}) "
        f"would yield ~{example_profit:.4f} {quote_currency} gross profit. "
        "Set min_profit_per_trade accordingly, and ensure it is not larger than your expected profit.\n"
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
                # Show net P&L after exit fees so the LLM knows the real profit/loss if it sells now
                if fee_rate is not None and current_price and current_price > 0:
                    current_position_value = amount * current_price
                    exit_fee_cost = current_position_value * fee_rate
                    # Also account for the entry fee already paid (stored in cost_basis)
                    # cost_basis already includes entry fee, so net = sell_value - exit_fee - cost_basis
                    net_pnl_after_fees = unrealized_pnl - exit_fee_cost
                    net_pnl_pct_after_fees = (net_pnl_after_fees / cost_basis) * 100 if cost_basis > 0 else 0.0
                    prompt += f"Exit fee if sold now: {exit_fee_cost:.4f} {base_currency}\n"
                    prompt += f"Net P&L after exit fees: {net_pnl_after_fees:+.4f} {base_currency} ({net_pnl_pct_after_fees:+.2f}%)\n"
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
        prompt += "\nMulti-timeframe OHLCV summary (price change %, high, low, volume, candle count):\n"
        for tf in settings.OHLCV_TIMEFRAMES:
            if tf in multi_tf_raw_candles:
                summary = _summarize_ohlcv(multi_tf_raw_candles[tf])
                if summary:
                    prompt += (
                        f"  [{tf}] change={summary['change_pct']}%, "
                        f"high={summary['high']}, low={summary['low']}, "
                        f"volume={summary['volume']}, candles={summary['candle_count']}\n"
                    )
        prompt += (
            "Use these summaries to assess short‑term momentum and trend across timeframes. "
            "The higher timeframes (1h, 4h) show the larger trend, while lower timeframes "
            "provide additional context for entry and exit timing.\n"
        )
    if multi_tf_indicators:
        prompt += "\nComputed technical indicators per timeframe:\n"
        for tf in settings.OHLCV_TIMEFRAMES:
            if tf in multi_tf_indicators:
                ind = multi_tf_indicators[tf]
                lines = [f"[{tf}]"]
                if ind.get('rsi') is not None:
                    lines.append(f"  RSI={ind['rsi']:.2f}")
                if ind.get('macd') is not None:
                    lines.append(f"  MACD={ind['macd']:.4f} Signal={ind['macd_signal']:.4f} Hist={ind['macd_hist']:.4f}")
                if ind.get('bb_upper') is not None:
                    lines.append(f"  BB Upper={ind['bb_upper']:.4f} Middle={ind['bb_middle']:.4f} Lower={ind['bb_lower']:.4f}")
                if ind.get('ema_9') is not None:
                    lines.append(f"  EMA9={ind['ema_9']:.4f} EMA21={ind['ema_21']:.4f}")
                if ind.get('stochastic_k') is not None:
                    lines.append(f"  Stoch %K={ind['stochastic_k']:.2f} %D={ind['stochastic_d']:.2f}")
                if ind.get('adx') is not None:
                    lines.append(f"  ADX={ind['adx']:.2f} +DI={ind['plus_di']:.2f} -DI={ind['minus_di']:.2f}")
                if ind.get('obv') is not None:
                    lines.append(f"  OBV={ind['obv']:.2f}")
                if ind.get('mfi') is not None:
                    lines.append(f"  MFI={ind['mfi']:.2f}")
                if ind.get('cci') is not None:
                    lines.append(f"  CCI={ind['cci']:.2f}")
                if ind.get('williams_r') is not None:
                    lines.append(f"  Williams %R={ind['williams_r']:.2f}")
                if ind.get('ichimoku') is not None:
                    ich = ind['ichimoku']
                    lines.append(f"  Ichimoku: Tenkan={ich['tenkan_sen']:.4f} Kijun={ich['kijun_sen']:.4f} SpanA={ich['senkou_span_a']:.4f} SpanB={ich['senkou_span_b']:.4f} Cloud={ich['cloud_bottom']:.4f}-{ich['cloud_top']:.4f}")
                if ind.get('donchian_channels') is not None:
                    dc = ind['donchian_channels']
                    lines.append(f"  Donchian: Upper={dc['upper']:.4f} Middle={dc['middle']:.4f} Lower={dc['lower']:.4f}")
                if ind.get('atr') is not None:
                    lines.append(f"  ATR(14)={ind['atr']:.6f}")
                if ind.get('keltner_channels') is not None:
                    kc = ind['keltner_channels']
                    lines.append(f"  Keltner: Upper={kc['upper']:.6f} Middle={kc['middle']:.6f} Lower={kc['lower']:.6f}")
                prompt += "\n".join(lines) + "\n"
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
        # Provide raw candles for backtesting
        raw_candles_str = _format_raw_candles_compact(historical_ohlcv, max_candles=200)
        prompt += (
            f"\nRaw historical OHLCV candles for backtesting "
            f"(last {min(len(historical_ohlcv), 200)} candles, "
            f"format: [timestamp_ms, open, high, low, close, volume]):\n"
            f"{raw_candles_str}\n"
        )
        prompt += (
            "**REQUIRED:** Perform a backtest on these candles using your proposed strategy. "
            "You MUST include a `backtest_summary` field in your JSON output with the results "
            "(e.g., \"3 wins, 2 losses, net +2.3%\"). If there is not enough data, explain why in the field.\n"
        )
    if drawdown_pct is not None:
        prompt += f"Current account drawdown: {drawdown_pct}%\n"
    if recent_trades:
        prompt += f"\nRecent closed trades (last {len(recent_trades)}):\n{json.dumps(recent_trades)}\n"
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
            buy_conf = t.get("buy_confidence", 0.0)
            buy_reason = t.get("buy_reasoning", "")
            conf_str = f", Buy Conf: {buy_conf:.2f}" if buy_conf else ""
            reason_str = f", Buy Reason: {buy_reason}" if buy_reason else ""
            prompt += (
                f"- Entry: {entry_price:.4f}, Exit: {exit_price:.4f}, Amount: {amount:.6f}, "
                f"P&L: {pnl:+.4f} ({pnl_pct:+.2f}%), Reason: {exit_reason}, "
                f"Hold: {hold_str}, Strategy: {strategy}{conf_str}{reason_str}\n"
            )
        prompt += (
            "Use these past outcomes to avoid repeating mistakes and reinforce successful patterns. "
            "Calibrate your confidence: if high-confidence trades are losing, lower confidence for similar setups; "
            "if low-confidence trades are winning, consider raising confidence.\n"
        )

    # --- Aggregate sentiment summary ---
    if aggregate_sentiment:
        prompt += (
            f"\nAggregate news sentiment for {symbol}:\n"
            f"  Compound score: {aggregate_sentiment['avg_compound']:.2f}  (range -1 to +1)\n"
            f"  Positive articles: {aggregate_sentiment['positive']}\n"
            f"  Negative articles: {aggregate_sentiment['negative']}\n"
            f"  Neutral articles: {aggregate_sentiment['neutral']}\n"
            f"  Total articles: {aggregate_sentiment['total_articles']}\n"
        )
        prompt += "Use aggregate sentiment to adjust confidence, position size, and risk. Strong positive = higher confidence/larger positions; strong negative = more cautious or skip.\n"
    if sentiment_trend is not None:
        prompt += f"\nSentiment trend (change in compound score since last cycle): {sentiment_trend:+.4f}\n"
        prompt += "Positive delta = sentiment improving, negative = deteriorating. Adjust confidence and risk parameters accordingly.\n"
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
    if keltner_channels:
        prompt += (
            f"\nKeltner Channels (20 EMA, 2× ATR): "
            f"Upper={keltner_channels['upper']:.6f}, "
            f"Middle={keltner_channels['middle']:.6f}, "
            f"Lower={keltner_channels['lower']:.6f}\n"
        )
        prompt += "Keltner Channels: volatility-based envelopes. Price near upper = overbought, near lower = oversold. Squeeze precedes large moves.\n"
    if donchian_channels:
        prompt += (
            f"\nDonchian Channels ({assigned_timeframe or 'default'}): "
            f"Upper={donchian_channels['upper']:.6f}, "
            f"Middle={donchian_channels['middle']:.6f}, "
            f"Lower={donchian_channels['lower']:.6f}\n"
        )
        prompt += "Donchian Channels: highest high/lowest low over lookback period. Breakout above upper = new high (bullish), below lower = new low (bearish). Narrow channel = low volatility (squeeze).\n"

    # --- News section (detailed articles) ---
    news_section = ""
    if settings.NEWS_ENABLED:
        articles = get_news_for_symbol(symbol, max_age_seconds=settings.NEWS_CACHE_TTL_SECONDS)
        if articles:
            news_section = "Recent news articles for this stock:\n" + _format_news_for_prompt(articles)
    if news_section:
        prompt += f"\n{news_section}\n"
        prompt += "Consider the news headlines above when setting confidence, position size, and max hold time.\n"

    prompt += f"""
**For the {assigned_timeframe or 'default'} timeframe, a reasonable minimum max_hold_time_seconds is {min_hold} seconds. Do not set it lower unless you have a very specific, justified reason (e.g., medium-term with a very tight stop and high confidence).**

You are trading spot only (no shorting). Only output SELL if you currently hold the stock.

"""
    prompt += (
        "\n**Entry Condition (REQUIRED for every BUY):**\n"
        "You MUST include an `entry_condition` object in your JSON output for every BUY action. "
        "This tells the bot the **exact moment** to enter the trade. "
        "If you omit this field, the trade will be executed immediately at the current market price. "
        "The object must have a `\"type\"` field and, except for `\"delay\"`, a `\"timeout_seconds\"` field.\n"
        "Supported types:\n"
        "- `\"limit_price\"`: wait for the price to drop to or below `\"price\"`.\n"
        "  Example: {\"type\": \"limit_price\", \"price\": 1.23, \"timeout_seconds\": 300}\n"
        "- `\"rsi_threshold\"`: wait for RSI(14) to fall below `\"rsi_below\"`.\n"
        "  Example: {\"type\": \"rsi_threshold\", \"rsi_below\": 30, \"timeout_seconds\": 600}\n"
        "- `\"delay\"`: simply wait `\"delay_seconds\"` before executing.\n"
        "  Example: {\"type\": \"delay\", \"delay_seconds\": 60}\n"
        "- `\"indicator_combo\"`: wait until ALL listed indicator conditions are met.\n"
        "  Example: {\"type\": \"indicator_combo\", \"conditions\": [ {\"indicator\": \"rsi\", \"threshold\": 30, \"direction\": \"below\"}, {\"indicator\": \"macd_hist\", \"threshold\": 0, \"direction\": \"above\"} ], \"timeout_seconds\": 600}\n"
        "If a timeout expires without the condition being met, the trade is skipped entirely.\n"
        "**Important:** The engine enforces a minimum timeout of 300 seconds or "
        f"{settings.ENTRY_CONDITION_MIN_TIMEOUT_MULT}× the candle timeframe, whichever is larger. "
        "Set `timeout_seconds` to at least this value, and prefer longer timeouts for higher timeframes "
        "(e.g., 900–1800 s for 1h candles).\n"
    )
    prompt += (
        "\n**Output ONLY the raw JSON object as specified.**\n\n"
        "Return a JSON object with these **required** fields:\n"
        "- `action`: one of BUY, SELL, HOLD\n"
        "- `confidence`: a float between 0.0 and 1.0\n"
        "- `reasoning`: a string explaining **why** you chose this action and this confidence level. "
        "Include the key factors (indicators, sentiment, market regime, etc.) that led to your decision.\n"
        "- `limit_price`: optional, a specific limit price for the order.\n"
        "- `time_in_force`: optional, \"day\" or \"gtc\". Default \"day\".\n"
        "You may include `\"portfolio_risk_adjustment_factor\"` (0.1–1.0) in the strategy parameters "
        "to vote on the overall portfolio risk for this cycle.\n"
    )
    if trading_paused:
        prompt += (
            "\n**Trading is currently PAUSED.** You may ONLY output SELL or HOLD actions. "
            "Do NOT output BUY under any circumstances. "
            "If you hold this stock, decide whether to continue holding (HOLD) or exit (SELL) "
            "based on current market conditions, risk parameters, and profit/loss status.\n"
        )
    if performance:
        stock_perf = performance.get("stock_performance", {}).get(symbol, {})
        strategy_perf = performance.get("strategy_performance", {})
        equity = performance.get("equity_curve", {})
        perf_text = f"""
Historical Performance:
- This stock's past performance: {json.dumps(stock_perf)} (stop_loss_hits = number of times stop-loss was triggered; avg_hold_time_seconds = average trade duration)
- Overall equity curve: {json.dumps(equity)}
- Strategy performance summary: {json.dumps(strategy_perf)}

Use this data to decide whether to BUY, SELL, or HOLD. If the stock has a poor win rate or the overall equity curve is declining, be more conservative. Prefer strategies that have worked well historically.
"""
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
