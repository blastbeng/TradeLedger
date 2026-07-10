import json
import logging
from typing import List, Dict, Any, Optional
from src.config.settings import settings
from src.database import get_news_for_symbols
from src.exchanges.market_data import TIMEFRAME_MAP
from src.llm.prompt_utils import _format_news_for_prompt, _format_trade_pattern_analysis, _round_floats, _to_toon

logger = logging.getLogger(__name__)


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
    trade_pattern_analysis: Optional[Dict[str, Any]] = None,
    symbol_events: Optional[Dict[str, Dict[str, Any]]] = None,
    symbol_trend_scores: Optional[Dict[str, float]] = None,
    market_breadth: Optional[Dict[str, Any]] = None,
    min_hold_time_mult: float = 1.0,
    min_stop_atr_mult: float = 1.0,
    min_viable_trade_amount: float = 0.0,
) -> str:
    """Build a prompt to ask the LLM which stocks/ETFs to trade."""
    # Trim large lists to prevent context window overflow
    if available_symbols and len(available_symbols) > 100:
        available_symbols = available_symbols[:100]
    if current_symbols and len(current_symbols) > 100:
        current_symbols = current_symbols[:100]
    if tickers:
        # Keep only tickers for the (potentially trimmed) available symbols
        tickers = {k: v for k, v in tickers.items() if k in available_symbols}
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
            _last = t.get("last")
            _vol = t.get("quoteVolume")
            ticker_summary[symbol] = {
                "last": round(_last, 2) if isinstance(_last, (int, float)) else _last,
                "percentage_24h": t.get("percentage"),
                "volume": round(_vol) if isinstance(_vol, (int, float)) else _vol,
                "min_trade_cost": limits.get("min_cost"),
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
Available timeframes: {_to_toon(available_timeframes)}
Currently tracked stocks (with assigned timeframes): {_to_toon(current_symbols) if current_symbols else "None"}

**Capital Allocation:** When you select many tickers, you can and should potentially use all of the available balance ({base_balance:.2f} {base_currency}) across your trading decisions. Do not artificially restrict yourself to the equal-share budget if you have high conviction in specific setups."""

    # --- Open positions summary ---
    if open_positions:
        prompt += "\n**Open positions (these will continue to be managed even if trading is paused):**\n"
        for sym, pos in open_positions.items():
            entry = pos.get("price", "?")
            amount = pos.get("amount", "?")
            sl = pos.get("stop_loss", "?")
            tp = pos.get("take_profit", "?")
            entry_str = f"{entry:.2f}" if isinstance(entry, (int, float)) else entry
            amount_str = f"{amount:.4f}" if isinstance(amount, (int, float)) else amount
            sl_str = f"{sl:.2f}" if isinstance(sl, (int, float)) else sl
            tp_str = f"{tp:.2f}" if isinstance(tp, (int, float)) else tp
            prompt += (
                f"  {sym}: entry={entry_str}, amount={amount_str}, "
                f"stop_loss={sl_str}, take_profit={tp_str}\n"
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

    prompt += f"""Select {settings.MIN_SYMBOLS if settings.MIN_SYMBOLS > 0 else 0}-{max_symbols} assets to trade. BTP bonds (ISIN format, e.g., IT0001234567) included — `name` has maturity/coupon. Select 0 to pause if unfavorable. Balance ({base_balance:.2f} {base_currency}) must be >= min_trade_cost. Keep tracked assets unless clearly deteriorating. Update timeframe if regime changed (bot manages open positions on new TF).
"""
    if settings.MIN_SYMBOLS > 0:
        prompt += (
            f"\n**MANDATORY:** You MUST select at least {settings.MIN_SYMBOLS} symbols. "
            f"Selecting fewer than {settings.MIN_SYMBOLS} is NOT allowed unless you are pausing trading entirely (pause_trading=true). "
            f"If you cannot find {settings.MIN_SYMBOLS} high-conviction setups, select the next best symbols with small position_size_fraction (0.01-0.05) and tight stops. "
            f"Do NOT select fewer than {settings.MIN_SYMBOLS} — the engine will override your selection and add more symbols automatically.\n"
        )
    prompt += f"""
Return JSON:
- "stocks":[{{"symbol","timeframe"({', '.join([repr(tf) for tf in available_timeframes])}),"sector","max_tenure_hours"?}}]
- "max_stocks":int 0-{max_symbols} (=len(stocks))
- "max_positions_per_sector":int 1-{max_symbols}
- "skip_eval_price_change_atr_mult":float
- "skip_eval_rsi_change":float
- "skip_eval_rsi_oversold":float
- "skip_eval_rsi_overbought":float
- "skip_eval_macd_hist_change":float
- "regime_adx_strong":float
- "regime_adx_moderate":float
- "regime_volatility_high_pct":float
- "regime_volatility_low_pct":float
- "regime_bb_squeeze_width":float
- "regime_bb_expansion_width":float
- "min_stop_loss_atr_mult":float
- "min_max_hold_time_mult":float
- "max_stop_loss_reviews":int 1-20
- "max_take_profit_reviews":int 1-20
- "max_partial_tp_reviews":int 1-20
- "max_dust_sweep_reviews":int 1-20
- "min_llm_pause_duration_seconds":int 300-14400
- "pause_max_consecutive_keep":int 1-10
- "pause_force_resume_risk_multiplier":float 0.0-1.0
- "max_portfolio_exposure_pct":float 0.0-1.0
- "max_portfolio_stop_risk_pct":float 0.0-1.0
- "min_risk_reward_ratio":float
- "confidence_rejection_threshold":float 0.0-1.0
- "limit_price_max_distance_pct":float? 0.0-1.0
- "min_viable_trade_amount":float?
- "reasoning":str max 50 chars. Use abbreviations/symbols only. NO full sentences.
- "stock_revaluation_interval_seconds":int? >=3600
- "pause_trading":bool?
- "pause_reason":str?
- "pause_duration_seconds":int?

Example: {{"stocks":[{{"symbol":"ENI.MI/EUR","timeframe":"1Y","sector":"Energy","max_tenure_hours":8760}},{{"symbol":"ENEL.MI/EUR","timeframe":"6M","sector":"Utilities"}}],"max_stocks":2,"max_positions_per_sector":2,"skip_eval_price_change_atr_mult":0.5,"skip_eval_rsi_change":5.0,"skip_eval_rsi_oversold":30.0,"skip_eval_rsi_overbought":70.0,"skip_eval_macd_hist_change":0.0005,"regime_adx_strong":40.0,"regime_adx_moderate":25.0,"regime_volatility_high_pct":80.0,"regime_volatility_low_pct":20.0,"regime_bb_squeeze_width":0.02,"regime_bb_expansion_width":0.08,"min_stop_loss_atr_mult":1.5,"min_max_hold_time_mult":1.5,"max_stop_loss_reviews":3,"max_take_profit_reviews":3,"min_llm_pause_duration_seconds":3600,"pause_max_consecutive_keep":3,"pause_force_resume_risk_multiplier":0.3,"max_partial_tp_reviews":3,"max_dust_sweep_reviews":3,"reasoning":"ENI strong 1Y uptrend; ENEL bullish MACD 6M","stock_revaluation_interval_seconds":3600,"max_portfolio_exposure_pct":0.8,"max_portfolio_stop_risk_pct":0.1,"min_risk_reward_ratio":1.5,"confidence_rejection_threshold":0.4,"limit_price_max_distance_pct":0.05,"pause_trading":false,"pause_reason":"Favorable"}}
"""
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
                prompt += f"  {sym}: {symbol_trend_scores[sym]:.2f}\n"
        prompt += "High trend quality (>0.7) = strong, clean trend suitable for momentum/breakout strategies. Low score (<0.3) = choppy or ranging, better for mean reversion or avoid.\n"
    if ohlcv_summary:
        filtered_ohlcv_summary = {}
        for sym, tfs in ohlcv_summary.items():
            valid_tfs = {}
            for tf, data in tfs.items():
                if data:
                    rounded_data = dict(data)
                    if 'high' in rounded_data and isinstance(rounded_data['high'], (int, float)):
                        rounded_data['high'] = round(rounded_data['high'], 2)
                    if 'low' in rounded_data and isinstance(rounded_data['low'], (int, float)):
                        rounded_data['low'] = round(rounded_data['low'], 2)
                    valid_tfs[tf] = rounded_data
            if valid_tfs:
                filtered_ohlcv_summary[sym] = valid_tfs
        if filtered_ohlcv_summary:
            prompt += f"\nMulti-timeframe OHLCV summary (price change %, high, low, volume):\n{_to_toon(filtered_ohlcv_summary)}\n"
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
            trimmed[sym_a] = {sym_b: round(v, 2) for sym_b, v in row.items() if sym_b in candidate_set}
        if trimmed:
            prompt += (
                "\nPairwise correlation matrix (Pearson correlation of daily returns, range -1 to +1):\n"
                f"{_to_toon(trimmed)}\n"
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
                    tf_lines.append(f"    MACD={ind['macd']:.2f} Signal={ind['macd_signal']:.2f} Hist={ind['macd_hist']:.2f}")
                if ind.get('bb_upper') is not None:
                    tf_lines.append(f"    BB Upper={ind['bb_upper']:.2f} Middle={ind['bb_middle']:.2f} Lower={ind['bb_lower']:.2f}")
                if ind.get('ema_9') is not None:
                    ema9_str = f"EMA9={ind['ema_9']:.2f}"
                    ema21_str = f" EMA21={ind['ema_21']:.2f}" if ind.get('ema_21') is not None else ""
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
                    tf_lines.append(f"    Ichimoku: Tenkan={ich['tenkan_sen']:.2f} Kijun={ich['kijun_sen']:.2f} SpanA={ich['senkou_span_a']:.2f} SpanB={ich['senkou_span_b']:.2f} Cloud={ich['cloud_bottom']:.2f}-{ich['cloud_top']:.2f}")
                if ind.get('donchian_channels') is not None:
                    dc = ind['donchian_channels']
                    tf_lines.append(f"    Donchian: Upper={dc['upper']:.2f} Middle={dc['middle']:.2f} Lower={dc['lower']:.2f}")
                if ind.get('parabolic_sar') is not None:
                    tf_lines.append(f"    SAR={ind['parabolic_sar']:.2f}")
                if ind.get('keltner_channels') is not None:
                    kc = ind['keltner_channels']
                    tf_lines.append(f"    Keltner: Upper={kc['upper']:.2f} Middle={kc['middle']:.2f} Lower={kc['lower']:.2f}")

                if tf_lines:
                    lines.append(f"  [{tf}]")
                    lines.extend(tf_lines)
            if len(lines) > 1:
                prompt += "\n".join(lines) + "\n"
    if market_trend:
        _mt_last = market_trend.get('last')
        _mt_chg = market_trend.get('change_24h')
        _mt_last_str = f"{_mt_last:.2f}" if isinstance(_mt_last, (int, float)) else _mt_last
        _mt_chg_str = f"{_mt_chg:.2f}" if isinstance(_mt_chg, (int, float)) else _mt_chg
        prompt += f"\nOverall market trend ({market_trend['symbol']}): daily change {_mt_chg_str}%, last price {_mt_last_str}\n"
    if session_info:
        prompt += f"\nCurrent UTC hour: {session_info['utc_hour']} ({session_info['session']} session)\n"
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
                prompt += f"  {base}: {delta:+.2f}\n"
    if news_section:
        prompt += f"\n{news_section}\n"
        prompt += (
            "**IMPORTANT:** Do not rely on pre-computed sentiment scores. Read the news headlines and summaries above "
            "and use your own understanding of financial context to assess the sentiment and potential impact for each stock. "
            "Factor this assessment into your stock selection and reasoning.\n"
        )
    if performance:
        perf_lines = ["Historical Performance Data:"]
        equity_curve = _round_floats(performance.get('equity_curve', {}))
        stock_perf = _round_floats(performance.get('stock_performance', {}))
        strategy_perf = _round_floats(performance.get('strategy_performance', {}))

        if equity_curve:
            perf_lines.append(f"Overall equity curve: {_to_toon(equity_curve)}")
        if stock_perf:
            perf_lines.append(f"Per-stock performance (win rate, avg P&L, total trades): {_to_toon(stock_perf)}")
        if strategy_perf:
            perf_lines.append(f"Per-strategy performance: {_to_toon(strategy_perf)}")

        if len(perf_lines) > 1:
            prompt += "\n".join(perf_lines) + "\n"
        if daily_pnl is not None:
            prompt += f"Today's realized P&L: {daily_pnl:.0f} {base_currency}\n"
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
            f"\n**Account P&L**: Today's realized P&L = {daily_pnl:.0f} {base_currency}, "
            f"Total realized P&L = {total_pnl:.0f} {base_currency}.\n"
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
    min_viable_trade_amount: float = 0.0,
    available_timeframes: Optional[List[str]] = None,
    market_limits: Optional[Dict[str, Dict[str, Any]]] = None,
    available_timeframes_by_symbol: Optional[Dict[str, List[str]]] = None,
) -> str:
    """Build a prompt for the final symbol selection from chunk results.

    After evaluating candidates in chunks, this prompt presents the combined
    shortlist from all chunks and asks the LLM to make the final selection.
    """
    # Trim large lists to prevent context window overflow
    if chunk_results and len(chunk_results) > 50:
        chunk_results = chunk_results[:50]
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
Available timeframes: {_to_toon(available_timeframes)}
Currently tracked stocks (with assigned timeframes): {_to_toon(current_symbols) if current_symbols else "None"}

**Combined Shortlist from All Batches (deduplicated):**
{_to_toon(shortlist)}

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
            entry_str = f"{entry:.2f}" if isinstance(entry, (int, float)) else entry
            amount_str = f"{amount:.4f}" if isinstance(amount, (int, float)) else amount
            sl_str = f"{sl:.2f}" if isinstance(sl, (int, float)) else sl
            tp_str = f"{tp:.2f}" if isinstance(tp, (int, float)) else tp
            prompt += f"  {sym}: entry={entry_str}, amount={amount_str}, stop_loss={sl_str}, take_profit={tp_str}\n"
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
        equity_curve = _round_floats(performance.get('equity_curve', {}))
        stock_perf = _round_floats(performance.get('stock_performance', {}))
        strategy_perf = _round_floats(performance.get('strategy_performance', {}))

        if equity_curve:
            perf_lines.append(f"Overall equity curve: {_to_toon(equity_curve)}")
        if stock_perf:
            perf_lines.append(f"Per-stock performance (win rate, avg P&L, total trades): {_to_toon(stock_perf)}")
        if strategy_perf:
            perf_lines.append(f"Per-strategy performance: {_to_toon(strategy_perf)}")

        if len(perf_lines) > 1:
            prompt += "\n".join(perf_lines) + "\n"
        if daily_pnl is not None:
            prompt += f"Today's realized P&L: {daily_pnl:.0f} {base_currency}\n"
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
        _mt_last = market_trend.get('last')
        _mt_chg = market_trend.get('change_24h')
        _mt_last_str = f"{_mt_last:.2f}" if isinstance(_mt_last, (int, float)) else _mt_last
        _mt_chg_str = f"{_mt_chg:.2f}" if isinstance(_mt_chg, (int, float)) else _mt_chg
        prompt += f"\nOverall market trend ({market_trend['symbol']}): daily change {_mt_chg_str}%, last price {_mt_last_str}\n"
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
            f"\n**Account P&L**: Today's realized P&L = {daily_pnl_val:.0f} {base_currency}, "
            f"Total realized P&L = {total_pnl:.0f} {base_currency}.\n"
        )

    # Output format
    prompt += f"""
Return JSON:
- "stocks":[{{"symbol","timeframe"({', '.join([repr(tf) for tf in available_timeframes])}),"sector","max_tenure_hours"?}}]
- "max_stocks":int 0-{max_symbols} (=len(stocks))
- "max_positions_per_sector":int 1-{max_symbols}
- "reasoning":str max 80 chars
- "skip_eval_price_change_atr_mult":float
- "skip_eval_rsi_change":float
- "skip_eval_rsi_oversold":float
- "skip_eval_rsi_overbought":float
- "skip_eval_macd_hist_change":float
- "regime_adx_strong":float
- "regime_adx_moderate":float
- "regime_volatility_high_pct":float
- "regime_volatility_low_pct":float
- "regime_bb_squeeze_width":float
- "regime_bb_expansion_width":float
- "min_stop_loss_atr_mult":float
- "min_max_hold_time_mult":float
- "max_stop_loss_reviews":int 1-20
- "max_take_profit_reviews":int 1-20
- "max_partial_tp_reviews":int 1-20
- "max_dust_sweep_reviews":int 1-20
- "min_llm_pause_duration_seconds":int 300-14400
- "pause_max_consecutive_keep":int 1-10
- "pause_force_resume_risk_multiplier":float 0.0-1.0
- "max_portfolio_exposure_pct":float 0.0-1.0
- "max_portfolio_stop_risk_pct":float 0.0-1.0
- "min_risk_reward_ratio":float
- "confidence_rejection_threshold":float 0.0-1.0
- "limit_price_max_distance_pct":float? 0.0-1.0
- "min_viable_trade_amount":float?
- "stock_revaluation_interval_seconds":int? >=3600
- "pause_trading":bool?
- "pause_reason":str?
- "pause_duration_seconds":int?
- "global_risk_multiplier":float? 0.0-1.0

Output ONLY the raw JSON object."""
    return prompt


def build_stock_selection_messages(
    available_symbols: List[str],
    current_symbols: List[Dict[str, str]],
    max_symbols: int,
    base_currency: str,
    tickers: Dict[str, Any],
    base_balance: float,
    per_symbol_budget: float,
    market_limits: Dict[str, Dict[str, Any]],
    **kwargs
) -> List[Dict[str, str]]:
    """Build a list of messages (system + user) for prompt caching."""
    from src.llm.system_prompt import build_system_prompt
    from src.llm.prompt_utils import compact_prompt
    return [
        {"role": "system", "content": compact_prompt(build_system_prompt(task_type="stock_selection"))},
        {"role": "user", "content": compact_prompt(build_stock_selection_prompt(
            available_symbols=available_symbols,
            current_symbols=current_symbols,
            max_symbols=max_symbols,
            base_currency=base_currency,
            tickers=tickers,
            base_balance=base_balance,
            per_symbol_budget=per_symbol_budget,
            market_limits=market_limits,
            **kwargs
        ))},
    ]


def build_final_selection_messages(
    chunk_results: List[Dict[str, Any]],
    current_symbols: List[Dict[str, str]],
    max_symbols: int,
    base_currency: str,
    base_balance: float,
    per_symbol_budget: float,
    **kwargs
) -> List[Dict[str, str]]:
    """Build a list of messages (system + user) for prompt caching."""
    from src.llm.system_prompt import build_system_prompt
    from src.llm.prompt_utils import compact_prompt
    return [
        {"role": "system", "content": compact_prompt(build_system_prompt(task_type="stock_selection"))},
        {"role": "user", "content": compact_prompt(build_final_selection_prompt(
            chunk_results=chunk_results,
            current_symbols=current_symbols,
            max_symbols=max_symbols,
            base_currency=base_currency,
            base_balance=base_balance,
            per_symbol_budget=per_symbol_budget,
            **kwargs
        ))},
    ]
