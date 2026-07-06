import json
import logging
import re
import time
from typing import List, Dict, Any, Optional, Tuple
from src.config.settings import settings
from src.utils.symbol_utils import is_btp_isin
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


def get_cached_news_summary(symbol: str, model_type: str = "weak") -> dict:
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


SYSTEM_PROMPT_TEMPLATE = """You are a professional stock, ETF, and BTP bond trading bot assistant focused on medium to long-term investment horizons. Your primary goal is to generate consistent profit by identifying assets with strong fundamentals, solid momentum, and favorable macro conditions over weeks to months. You must avoid large drawdowns and only trade when there is a clear edge. Your asset universe includes Italian stocks, UCITS ETFs, and Italian government bonds (BTPs).

## Key Principles
- **Primary Timeframes:** "5Y", "3Y", "1Y", "6M", "3M", "1M", "1w" are all equally valid primary timeframes. Choose the most appropriate one for each asset based on volatility, trend stage, and intended hold period. Diversify timeframe assignments. Use "1d" and "1h" only for short-term confirmation or when long-term data is unavailable. Match timeframe to asset: "5Y"/"3Y" for stable, long-term holdings; "1Y"/"6M" for medium-term; "3M"/"1M" for moderate volatility; "1w" for weekly trend capture.
- **Confidence & Sizing:** Set confidence between 0.0 (no conviction, HOLD) and 1.0 (absolute certainty). Confidence scales position size (via `confidence_sizing_weight`) and rejects low-conviction trades (via `confidence_rejection_threshold`).
- **Position Sizing:** You MUST set `position_size_fraction` (0.01 to 1.0) to reflect your confidence and risk. The engine uses exactly the fraction you provide. The sum across all traded stocks must not exceed 1.0.
- **Trade Selection:** Focus on strong medium/long-term momentum, solid fundamentals, and favorable sector trends. Require confirmation from at least two independent indicators (e.g., RSI + MACD). Prefer buying near support (lower Bollinger Band, oversold RSI) and selling near resistance. Never chase breakouts without confirmation.

## Stop-Loss
- Prefer ATR-based stops (`"stop_loss_method": "atr_multiple"`). Typical multipliers: 2.0–3.0 (normal vol), 3.0–5.0 (high vol/ATR > 80%), 1.5–2.0 (low vol/ATR < 20%). The engine converts `stop_loss_atr_multiple × ATR` to a percentage.
- If using fixed stops (`"stop_loss_method": "fixed"`), ensure `stop_loss_pct` is at least 1.5× ATR% to avoid being stopped out by noise.
- **Required parameters for every BUY/SELL:**
  - `"stop_loss_method"`: "fixed" or "atr_multiple".
  - `"stop_loss_atr_multiple"`: Float (e.g., 2.0). Required if method is "atr_multiple".
  - `"stop_loss_pct"`: Float 0.001–0.5. ALWAYS required (fallback for ATR). If using "atr_multiple", set to the estimated ATR-based stop percentage.

## Take-Profit
- Set an achievable take-profit based on trend, volatility, and market conditions. The reward:risk ratio is your decision.
- **CRITICAL:** `take_profit_pct` MUST be strictly greater than `stop_loss_pct`. If `take_profit_pct ≤ stop_loss_pct`, the trade will be rejected.
- **ATR-based Take-Profit:** You may use `"take_profit_method": "atr_multiple"` and set `take_profit_atr_multiple`. Typical multipliers: 3.0–5.0 (normal vol), 5.0–8.0 (high vol). The engine converts `take_profit_atr_multiple × ATR` to a percentage.
__STOCK_FEE_SECTION__
__BTP_FEE_SECTION__
- **Required parameter for every BUY/SELL:**
  - `"take_profit_pct"`: Float 0.005–2.0. ALWAYS required (fallback for ATR). If using "atr_multiple", set to the estimated ATR-based take-profit percentage.

## Max Hold Time
- Set a `max_hold_time_seconds` for every trade. If the price doesn't hit TP/SL within this time, the position closes automatically.
- **Do NOT set it too short.** Err on the side of longer hold times. Guidelines: 1h candles → 1-3 days; 1d candles → 1-2 months; 1w candles → 3-6 months; 1M candles → 6-12 months.
- **Required parameter for every BUY/SELL:**
  - `"max_hold_time_seconds"`: Positive integer (seconds).

## Trailing Stops
- Use trailing stops to lock in profits when the price moves favourably.
- **Required parameters for every BUY/SELL:**
  - `"trailing_stop"`: Boolean.
  - `"trailing_stop_distance_pct"`: Float 0.001–0.1. Required if `trailing_stop` is true. Must be < `stop_loss_pct`. Set to null if false.
- **Optional parameters:**
  - `"trailing_stop_activation_pct"`: Float 0–1.0. Trailing starts when price moves this % in your favor. Omit for immediate activation.

## Risk Management
- Adjust position size based on confidence, risk level, account drawdown, and portfolio exposure. You decide the fraction that balances profit potential with capital preservation.
- **Risk Appetite Framework:** Adapt dynamically to market and portfolio conditions.
  - **Calculated Risk (Normal):** Breadth > 40%, P&L > -5%, no consecutive losses. Trade at least 1–2 stocks with small positions to probe. Avoid staying idle.
  - **Conservative (Adverse):** P&L < -5%, 2+ consecutive losses, or breadth < 30%. Reduce position sizes, be selective, prioritize capital preservation. You may select 0 stocks to pause.
  - **Probing (Neutral):** Breadth 30-40%, no high-conviction setups. Select 1–2 stocks with small positions (`position_size_fraction` ≤ 0.2) and tight stops. Do not pause completely.
- **Hybrid Capital Allocation:** You may allocate ALL available capital to a single high-conviction trade. Prioritize quality over quantity. Do NOT place small trades that are unprofitable after fees just to fill slots. Stocks, ETFs, and BTPs have equal priority.

## Cooldown & Optional Risk Parameters
- **Cooldown:** Set `cooldown_after_loss_seconds` to **0** (no cooldown) unless you have a very strong reason to avoid a stock. Quick re-entry after a small loss is often profitable.
- **Stock Selection JSON:** You may include `"max_portfolio_exposure_pct"` (0.0-1.0), `"max_portfolio_stop_risk_pct"` (0.0-1.0), `"min_risk_reward_ratio"` (e.g., 1.5), and `"confidence_rejection_threshold"` (0.0-1.0) to define global limits for the cycle.
- **Required parameter for every BUY/SELL:**
  - `"cooldown_after_loss_seconds"`: Non-negative integer. Set 0 to allow immediate re-entry.
- **Optional parameters:**
  - `"position_size_fraction"`: Float 0.01–1.0. Fraction of total available cash balance. Sum across all stocks ≤ 1.0.
  - `"max_risk_per_trade_pct"`: Float 0–1.0. Limits position size so potential loss ≤ this fraction of portfolio value.
  - `"max_portfolio_risk_pct"`: Float 0–1.0. Skips trade if total potential loss of all positions + this trade exceeds this % of portfolio.
  - `"min_profit_per_trade"`: Non-negative number. Skips trade if expected gross profit < this value. Set to 0 to allow tiny profits.
  - `"min_risk_reward_ratio"`: Positive number. Rejects trade unless `take_profit_pct / stop_loss_pct` >= this value.
  - `"position_size_multiplier"`: Float 0.0–1.0. Further multiplies final position size after global risk multiplier.
  - `"confidence_sizing_weight"`: Float 0.0–1.0. Scales position size by `(1.0 - confidence_sizing_weight × (1.0 - confidence))`. 0.0 = disabled, 1.0 = directly proportional to confidence.
  - `"min_confidence"`: Float 0.0–1.0. Skips trade if confidence is below this threshold.
  - `"portfolio_risk_adjustment_factor"`: Float 0.1–1.0. Per-symbol "vote" on portfolio risk. Engine takes the minimum across all symbols as a global multiplier. Use lower values for high volatility/risk, 1.0 for normal conditions.

## Position Sizing — Your Full Responsibility
You MUST decide the exact currency amount to trade by setting `position_size_fraction`. The engine will NOT automatically reduce your position size based on ATR or fixed risk limits. You must calculate the appropriate size yourself considering ALL of the following:
1. **Risk per share**: For ATR-based stops, `risk_per_share = stop_loss_atr_multiple × ATR`. For fixed stops, `risk_per_share = stop_loss_pct × current_price`.
2. **Max risk amount**: `max_risk_amount = total_portfolio_value × max_risk_per_trade_pct` (if you set `max_risk_per_trade_pct`).
3. **Max quantity**: `max_quantity = max_risk_amount / risk_per_share`.
4. **Position size fraction**: `position_size_fraction = (max_quantity × current_price) / total_portfolio_value`.
5. Also consider: transaction costs (fees), your confidence level, backtest results (win rate, drawdown, profit factor), market conditions (volatility, regime, breadth), portfolio exposure, and concentration.
Example: Portfolio €10,000, risk 1% (€100), ATR €0.50, stop = 2×ATR (€1.00 risk/share) → max 100 shares. At €25/share, `position_size_fraction = (100 × 25) / 10000 = 0.25`.
If you prefer not to use risk-based sizing, you may set `position_size_fraction` based on confidence and setup quality. The engine respects your decision as long as it does not exceed available balance or exchange minimums.

## Pause/Resume
- You may include `"pause_trading"` (boolean) in your stock selection JSON to pause/resume trading. Always include a `"pause_reason"` string when setting pause_trading. You may also set `"pause_duration_seconds"` (positive integer) to auto-resume after a delay.
- If you pause because of consecutive losses, drawdown, or lack of high‑confidence setups, you MUST set a longer pause_duration_seconds (at least 1800–7200 seconds). A very short pause will almost certainly result in the same market conditions and an immediate re‑pause.
- Use shorter pauses (e.g., 600–1800s) only when you expect a specific short‑term event to pass.
- If you omit pause_duration_seconds, the engine will default to a 30‑minute pause.

## Learn from Past Trades
- After a losing trade on a stock, avoid it for several evaluation cycles. Use the provided list of recent closed trades to avoid repeating mistakes and reinforce successful patterns.
- Learn from historical performance: avoid stocks and strategies with poor win rates or negative average P&L.
- Calibrate your confidence: if high-confidence trades are losing, lower confidence for similar setups; if low-confidence trades are winning, consider raising confidence.

## News Sentiment
- You will receive news sentiment data for each stock. Use it to gauge market sentiment and catalysts: prefer stocks with positive sentiment; be cautious with negative sentiment.
- If sentiment conflicts with technicals, give more weight to technicals but explain your reasoning.

## Output Format
- Output strict JSON only. The response must start with `{` or `[` and end with `}` or `]`. No markdown fences, no explanations, no extra text.

## Stock & ETF Market Specifics
- **Earnings & Corporate Events:** Stocks can gap significantly due to earnings or major events. Avoid holding through them unless you have very high conviction.
- **ETFs:** Lower volatility and smoother trends than stocks. Beware of decay in leveraged ETFs if held long.
- **BTP Bonds (Italian Sovereign Bonds):** Identified by ISIN (e.g., IT0001234567). Fixed-income, low volatility. Use wider stops, longer hold times, and smaller TP targets. Suitable for capital preservation.
  - **Trailing Stops Not Supported:** Do NOT set `trailing_stop` to true for BTPs. The engine will ignore it.
  - **Yield to Maturity (YTM) Assessment:** `ticker` includes `coupon` (decimal) and `maturity`. Price is % of par (e.g., 101.68 = 101.68% of face value). Calculate approximate YTM:
    - `Annual Coupon = coupon × 100`
    - `Years to Maturity = (maturity_date - current_date).days / 365`
    - `Approximate YTM = (Annual Coupon + (100 - Current Price) / Years to Maturity) / ((100 + Current Price) / 2)`
    - Compare YTM to current Italian yields. If YTM is attractive, it's a good buy. If price > 110 and YTM is low, upside is limited.

## Two-Step Decision Process with Multiple Backtest Variants
You will operate in two steps:
1. **Step 1:** Analyze market data and propose **multiple** backtest variants (min 1, recommended 3–5, max __MAX_BACKTEST_VARIANTS__). Each variant explores a different hypothesis (e.g., tight vs wide stop, short vs long hold). The engine runs a local Python backtest for each. Provide diverse variants to maximize finding a winning strategy.
2. **Step 2:** Receive ALL backtest results and make your final decision (BUY, SELL, HOLD). Compare variants and choose the best-performing one to inform your final parameters.

## Entry Conditions
You must include an `entry_condition` object for every BUY action. The strategy prompt provides full details and examples.

"""

def _compute_stock_fee_text() -> str:
    """Compute stock fee section text from current settings."""
    stock_fee_perc_pct = settings.STOCK_FEE_PERC * 100
    stock_fee_min_eur = settings.STOCK_FEE_MIN
    stock_fee_fixed_eur = settings.STOCK_FEE_FIXED
    tobin_tax_pct = settings.TOBIN_TAX_RATE * 100
    round_trip_perc_pct = (settings.STOCK_FEE_PERC * 2 + settings.TOBIN_TAX_RATE) * 100
    total_fixed_fees = stock_fee_fixed_eur * 2
    small_trade_fixed_cost = stock_fee_min_eur * 2

    trade_1000_buy_fee = max(stock_fee_min_eur, 1000 * settings.STOCK_FEE_PERC) + stock_fee_fixed_eur + (1000 * settings.TOBIN_TAX_RATE)
    trade_1000_sell_fee = max(stock_fee_min_eur, 1000 * settings.STOCK_FEE_PERC) + stock_fee_fixed_eur
    trade_1000_total = trade_1000_buy_fee + trade_1000_sell_fee
    trade_1000_pct = (trade_1000_total / 1000) * 100

    trade_10000_buy_fee = max(stock_fee_min_eur, 10000 * settings.STOCK_FEE_PERC) + stock_fee_fixed_eur + (10000 * settings.TOBIN_TAX_RATE)
    trade_10000_sell_fee = max(stock_fee_min_eur, 10000 * settings.STOCK_FEE_PERC) + stock_fee_fixed_eur
    trade_10000_total = trade_10000_buy_fee + trade_10000_sell_fee
    trade_10000_pct = (trade_10000_total / 10000) * 100

    return f"""- **Transaction Costs (Intesa Sanpaolo Investo):** The simulator applies the following fees per trade:
  - **Bank Commission:** {stock_fee_perc_pct:.2f}% of trade value, with a minimum of €{stock_fee_min_eur:.2f}. Plus a fixed execution fee of €{stock_fee_fixed_eur:.2f} per order.
  - **Tobin Tax (Italian State Tax):** {tobin_tax_pct:.2f}% of trade value, applied ONLY on BUY orders.
  - **Total Round-Trip Cost:** For a BUY followed by a SELL, the total fee is approximately {round_trip_perc_pct:.2f}% of the trade value PLUS €{total_fixed_fees:.2f} in fixed fees (for larger trades > €1,500). For smaller trades, the €{stock_fee_min_eur:.2f} minimum commission applies on both sides, making the total fixed cost €{small_trade_fixed_cost:.2f}.
  - **CRITICAL:** You MUST ensure your `take_profit_pct` is strictly greater than the total round-trip fee percentage. For a €1,000 trade, total fees are ~€{trade_1000_total:.2f} ({trade_1000_pct:.2f}%), so `take_profit_pct` must be > {trade_1000_pct + 0.01:.2f}%. For a €10,000 trade, total fees are ~€{trade_10000_total:.2f} ({trade_10000_pct:.2f}%), so `take_profit_pct` must be > {trade_10000_pct + 0.01:.2f}%. Never set a take-profit target lower than the break-even cost."""


def _compute_btp_fee_text() -> str:
    """Compute BTP fee section text from current settings."""
    if settings.BTP_IS_PRIMARY_ISSUANCE:
        return """- **BTP Bond Transaction Costs:** BTP bonds purchased via primary issuance have zero fees.
  - **Bank Commission:** €0.00 (exempt for primary issuance).
  - **Tobin Tax:** Exempt (sovereign bonds are not subject to Tobin tax).
  - **Total Round-Trip Cost:** €0.00.
  - **CRITICAL:** For BTPs, `take_profit_pct` can be as low as 0.001 (0.1%) since there are no transaction costs."""
    else:
        btp_fee_perc_pct = settings.BTP_FEE_PERC * 100
        btp_min_fee_eur = settings.BTP_MIN_FEE
        round_trip_perc_pct = btp_fee_perc_pct * 2
        small_trade_fixed_cost = btp_min_fee_eur * 2

        trade_1000_fee = max(btp_min_fee_eur, 1000 * settings.BTP_FEE_PERC)
        trade_1000_total = trade_1000_fee * 2
        trade_1000_pct = (trade_1000_total / 1000) * 100

        trade_10000_fee = max(btp_min_fee_eur, 10000 * settings.BTP_FEE_PERC)
        trade_10000_total = trade_10000_fee * 2
        trade_10000_pct = (trade_10000_total / 10000) * 100

        return f"""- **BTP Bond Transaction Costs:** BTP bonds have different fees:
  - **Bank Commission:** {btp_fee_perc_pct:.2f}% of trade value, with a minimum of €{btp_min_fee_eur:.2f}. No fixed execution fee.
  - **Tobin Tax:** Exempt (sovereign bonds are not subject to Tobin tax).
  - **Total Round-Trip Cost:** For a BUY followed by a SELL, the total fee is approximately {round_trip_perc_pct:.2f}% of the trade value (for larger trades). For smaller trades, the €{btp_min_fee_eur:.2f} minimum applies on both sides, making the total fixed cost €{small_trade_fixed_cost:.2f}.
  - **CRITICAL:** For BTPs, ensure your `take_profit_pct` is strictly greater than the total round-trip fee percentage. For a €1,000 BTP trade, total fees are ~€{trade_1000_total:.2f} ({trade_1000_pct:.2f}%), so `take_profit_pct` must be > {trade_1000_pct + 0.01:.2f}%. For a €10,000 BTP trade, total fees are ~€{trade_10000_total:.2f} ({trade_10000_pct:.2f}%), so `take_profit_pct` must be > {trade_10000_pct + 0.01:.2f}%."""


def build_system_prompt() -> str:
    """Build the system prompt with current settings values.

    Called on demand so that settings.reload() picks up new fee values
    and max backtest variants without requiring a module reimport.
    """
    prompt = SYSTEM_PROMPT_TEMPLATE.replace(
        "__MAX_BACKTEST_VARIANTS__", str(settings.MAX_BACKTEST_VARIANTS)
    )
    prompt = prompt.replace("__STOCK_FEE_SECTION__", _compute_stock_fee_text())
    prompt = prompt.replace("__BTP_FEE_SECTION__", _compute_btp_fee_text())
    return prompt

SYSTEM_PROMPT = build_system_prompt()


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
            raw_news = "Recent news for all candidate stocks:\n\n" + "\n\n".join(news_lines)
            # Summarize the combined news section using the weak model to save tokens
            try:
                from src.llm.summarizer import summarize_text
                news_section = summarize_text(raw_news, context="stock selection news", max_length=1000)
            except Exception:
                news_section = raw_news

    available_timeframes = [tf for tf in settings.OHLCV_TIMEFRAMES if tf in TIMEFRAME_MAP]
    prompt = f"""Current base currency: {base_currency}
Your available {base_currency} balance: {base_balance:.2f}
Maximum number of stocks to trade: {max_symbols}
Reference equal-share budget per stock (suggestion only — you decide actual allocations): {per_symbol_budget:.2f} {base_currency}
Available timeframes: {json.dumps(available_timeframes)}
Currently tracked stocks (with assigned timeframes): {json.dumps(current_symbols) if current_symbols else "None"}

**Capital Allocation:** When you select many tickers, you can and should potentially use all of the available balance ({base_balance:.2f} {base_currency}) across your trading decisions. Do not artificially restrict yourself to the equal-share budget if you have high conviction in specific setups."""

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

Select between {settings.MIN_SYMBOLS if settings.MIN_SYMBOLS > 0 else 0} and {max_symbols} assets (stocks, ETFs, or BTP bonds) to trade. The available symbols may include Italian BTP bonds identified by their ISIN (e.g., IT0001234567). The `name` field in the market data contains the bond's description, including maturity and coupon (e.g., 'Btp-1nv26 7,25%' means November 2026 maturity, 7.25% coupon). You can select them alongside stocks. If market conditions are extremely unfavorable (e.g., high losses, poor momentum, negative sentiment), you may select 0 assets to pause trading until the next evaluation. You MUST only select assets where your total available balance ({base_balance:.2f} {base_currency}) is greater than or equal to the asset's min_trade_cost. You may keep some current assets if they are still promising and meet the budget requirement, or replace them. **Prefer to keep assets that have been tracked for a while** – they have more historical data and the bot has already invested in learning their behaviour. Only drop an asset if it shows clear deterioration (e.g., negative momentum on all timeframes, poor win rate, or strongly negative sentiment). For assets already being tracked, re-evaluate their assigned timeframe. If the market regime has changed (e.g., a stock that was trending on 1d is now choppy and better suited to 1w), update the timeframe. If you change the timeframe for an asset with an open position, the bot will switch to managing the position using the new timeframe.
"""
    if settings.MIN_SYMBOLS > 0:
        prompt += (
            f"\n**MANDATORY:** You MUST select at least {settings.MIN_SYMBOLS} symbols. "
            f"Selecting fewer than {settings.MIN_SYMBOLS} is NOT allowed unless you are pausing trading entirely (pause_trading=true). "
            f"If you cannot find {settings.MIN_SYMBOLS} high-conviction setups, select the next best symbols with small position_size_fraction (0.01-0.05) and tight stops. "
            f"Do NOT select fewer than {settings.MIN_SYMBOLS} — the engine will override your selection and add more symbols automatically.\n"
        )
    prompt += f"""

Each symbol can only appear once in your selection. Choose the single best timeframe for each stock based on the multi-timeframe OHLCV data.

**Output ONLY the raw JSON object as specified.**

Return a JSON object with the following fields:
- `"stocks"`: Array of objects with `"symbol"`, `"timeframe"` (one of: {', '.join([repr(tf) for tf in available_timeframes])}), `"sector"`, and optional `"max_tenure_hours"` (float).
- `"max_stocks"`: Integer 0-{max_symbols}. Must equal length of `"stocks"`. Set high (up to {max_symbols}) when opportunities exist.
- `"max_positions_per_sector"`: Integer 1-{max_symbols}. Max open positions per sector.
- `"skip_eval_price_change_atr_mult"`: Float (e.g., 0.5). Min price change (ATR%) to trigger LLM eval.
- `"skip_eval_rsi_change"`: Float (e.g., 5.0). Min RSI change to trigger LLM eval.
- `"skip_eval_rsi_oversold"`: Float (e.g., 30.0). RSI level below which LLM eval is always triggered.
- `"skip_eval_rsi_overbought"`: Float (e.g., 70.0). RSI level above which LLM eval is always triggered.
- `"skip_eval_macd_hist_change"`: Float (e.g., 0.0005). Min MACD histogram change to trigger LLM eval.
- `"regime_adx_strong"`: Float (e.g., 40.0). ADX level for strong trend.
- `"regime_adx_moderate"`: Float (e.g., 25.0). ADX level for moderate trend.
- `"regime_volatility_high_pct"`: Float (e.g., 80.0). ATR percentile for high volatility.
- `"regime_volatility_low_pct"`: Float (e.g., 20.0). ATR percentile for low volatility.
- `"regime_bb_squeeze_width"`: Float (e.g., 0.02). Bollinger Band width for squeeze.
- `"regime_bb_expansion_width"`: Float (e.g., 0.08). Bollinger Band width for expansion.
- `"min_stop_loss_atr_mult"`: Float (e.g., 1.5). Min stop-loss as ATR% multiple.
- `"min_max_hold_time_mult"`: Float (e.g., 2.0). Min max_hold_time as candle timeframe multiple.
- `"max_stop_loss_reviews"`: Integer 1-20. Max LLM reviews on stop-loss trigger.
- `"max_take_profit_reviews"`: Integer 1-20. Max LLM reviews on take-profit trigger.
- `"max_partial_tp_reviews"`: Integer 1-20. Max LLM reviews on partial take-profit trigger.
- `"max_dust_sweep_reviews"`: Integer 1-20. Max LLM reviews on dust sweep trigger.
- `"min_llm_pause_duration_seconds"`: Integer 300-14400. Min pause duration before resume.
- `"pause_max_consecutive_keep"`: Integer 1-10. Max consecutive "keep paused" before force-resume.
- `"pause_force_resume_risk_multiplier"`: Float 0.0-1.0. Risk multiplier on force-resume.
- `"max_portfolio_exposure_pct"`: Float 0.0-1.0. Max portfolio value deployed.
- `"max_portfolio_stop_risk_pct"`: Float 0.0-1.0. Max total stop-loss risk as portfolio %.
- `"min_risk_reward_ratio"`: Positive number (e.g., 1.5). Min reward:risk ratio.
- `"confidence_rejection_threshold"`: Float 0.0-1.0. Min confidence for trade execution.
- `"limit_price_max_distance_pct"`: Optional Float 0.0-1.0. Max limit price distance from bid/ask.
- `"min_viable_trade_amount"`: Optional positive number. Suggested min trade amount in {base_currency}.
- `"reasoning"`: Short string (max 200 chars) explaining your selection.

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
- `"stocks"`: Array of objects with `"symbol"`, `"timeframe"` (one of: {', '.join([repr(tf) for tf in available_timeframes])}), `"sector"`, and optional `"max_tenure_hours"`.
- `"max_stocks"`: Integer 0-{max_symbols}. Must equal length of `"stocks"`.
- `"max_positions_per_sector"`: Integer 1-{max_symbols}. Max open positions per sector.
- `"reasoning"`: Short string (max 200 chars) explaining your final selection.
- `"skip_eval_price_change_atr_mult"`: Float (e.g., 0.5). Min price change (ATR%) to trigger LLM eval.
- `"skip_eval_rsi_change"`: Float (e.g., 5.0). Min RSI change to trigger LLM eval.
- `"skip_eval_rsi_oversold"`: Float (e.g., 30.0). RSI level below which LLM eval is always triggered.
- `"skip_eval_rsi_overbought"`: Float (e.g., 70.0). RSI level above which LLM eval is always triggered.
- `"skip_eval_macd_hist_change"`: Float (e.g., 0.0005). Min MACD histogram change to trigger LLM eval.
- `"regime_adx_strong"`: Float (e.g., 40.0). ADX level for strong trend.
- `"regime_adx_moderate"`: Float (e.g., 25.0). ADX level for moderate trend.
- `"regime_volatility_high_pct"`: Float (e.g., 80.0). ATR percentile for high volatility.
- `"regime_volatility_low_pct"`: Float (e.g., 20.0). ATR percentile for low volatility.
- `"regime_bb_squeeze_width"`: Float (e.g., 0.02). Bollinger Band width for squeeze.
- `"regime_bb_expansion_width"`: Float (e.g., 0.08). Bollinger Band width for expansion.
- `"min_stop_loss_atr_mult"`: Float (e.g., 1.5). Min stop-loss as ATR% multiple.
- `"min_max_hold_time_mult"`: Float (e.g., 2.0). Min max_hold_time as candle timeframe multiple.
- `"max_stop_loss_reviews"`: Integer 1-20. Max LLM reviews on stop-loss trigger.
- `"max_take_profit_reviews"`: Integer 1-20. Max LLM reviews on take-profit trigger.
- `"max_partial_tp_reviews"`: Integer 1-20. Max LLM reviews on partial take-profit trigger.
- `"max_dust_sweep_reviews"`: Integer 1-20. Max LLM reviews on dust sweep trigger.
- `"min_llm_pause_duration_seconds"`: Integer 300-14400. Min pause duration before resume.
- `"pause_max_consecutive_keep"`: Integer 1-10. Max consecutive "keep paused" before force-resume.
- `"pause_force_resume_risk_multiplier"`: Float 0.0-1.0. Risk multiplier on force-resume.
- `"max_portfolio_exposure_pct"`: Float 0.0-1.0. Max portfolio value deployed.
- `"max_portfolio_stop_risk_pct"`: Float 0.0-1.0. Max total stop-loss risk as portfolio %.
- `"min_risk_reward_ratio"`: Positive number (e.g., 1.5). Min reward:risk ratio.
- `"confidence_rejection_threshold"`: Float 0.0-1.0. Min confidence for trade execution.
- `"limit_price_max_distance_pct"`: Optional Float 0.0-1.0. Max limit price distance from bid/ask.
- `"min_viable_trade_amount"`: Optional positive number. Suggested min trade amount in {base_currency}.
- `"stock_revaluation_interval_seconds"`: Optional integer >= 3600.
- `"pause_trading"`: Optional boolean.
- `"pause_reason"`: Optional string.
- `"pause_duration_seconds"`: Optional positive integer.
- `"global_risk_multiplier"`: Optional Float 0.0-1.0.

Set `max_portfolio_exposure_pct` to at least **0.8** and `max_portfolio_stop_risk_pct` to at least **0.1** unless you have a very strong reason to be more conservative.

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
        max_possible_amount = min(per_symbol_budget, remaining_balance)
        prompt += (
            f"Max amount allocatable to this trade: {max_possible_amount:.2f} {base_currency} "
            f"(min of per-symbol budget and remaining balance). If setting `min_profit_per_trade`, "
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
    trade_value = min(per_symbol_budget, remaining_balance if remaining_balance is not None else per_symbol_budget)
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
                f"\n**Transaction Cost Break-Even (BTP):**\n"
                f"  Trade size: ~{trade_value:.2f} {quote_currency}\n"
                f"  Total round-trip fees: {total_fees:.2f} {quote_currency} ({break_even_pct*100:.2f}%)\n"
                f"  `take_profit_pct` MUST be > {break_even_pct*100:.2f}% to be profitable.\n"
            )
        else:
            buy_fee = max(settings.STOCK_FEE_MIN, trade_value * settings.STOCK_FEE_PERC) + settings.STOCK_FEE_FIXED + (trade_value * settings.TOBIN_TAX_RATE)
            sell_fee = max(settings.STOCK_FEE_MIN, trade_value * settings.STOCK_FEE_PERC) + settings.STOCK_FEE_FIXED
            total_fees = buy_fee + sell_fee
            break_even_pct = total_fees / trade_value
            prompt += (
                f"\n**Transaction Cost Break-Even:**\n"
                f"  Trade size: ~{trade_value:.2f} {quote_currency}\n"
                f"  Total round-trip fees: {total_fees:.2f} {quote_currency} ({break_even_pct*100:.2f}%)\n"
                f"  `take_profit_pct` MUST be > {break_even_pct*100:.2f}% to be profitable.\n"
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
                ind_lines.append(f"[{tf}] {json.dumps(ind_compact)}")
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
        "  `stop_loss_pct`, `take_profit_pct`, `position_size_fraction`, `confidence_sizing_weight`, `trailing_stop`, `max_hold_time_seconds`, `cooldown_after_loss_seconds`, `backtest_period_days`, etc.\n"
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
    _is_btp = is_btp_isin(symbol)
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
            buy_fee = max(_settings.STOCK_FEE_MIN, trade_value * _settings.STOCK_FEE_PERC) + _settings.STOCK_FEE_FIXED + (trade_value * _settings.TOBIN_TAX_RATE)
            sell_fee = max(_settings.STOCK_FEE_MIN, trade_value * _settings.STOCK_FEE_PERC) + _settings.STOCK_FEE_FIXED
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
        "  `stop_loss_pct`, `take_profit_pct`, `position_size_fraction`, `confidence_sizing_weight`, `trailing_stop`, `max_hold_time_seconds`, `cooldown_after_loss_seconds`, etc.\n"
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
