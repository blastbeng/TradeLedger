import asyncio
import json
import logging
import re
import toon
from typing import List, Dict, Any, Optional
from src.config.settings import settings
from src.database import get_news_for_symbol
from src.utils.redis_client import get_redis_client
from src.llm.cache import get_cached_llm_response, _llm_executor

logger = logging.getLogger(__name__)


def _round_floats(obj, decimals=2):
    """Recursively round all floats in nested dicts/lists to improve cache stability."""
    if isinstance(obj, float):
        return round(obj, decimals)
    if isinstance(obj, dict):
        return {k: _round_floats(v, decimals) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(item, decimals) for item in obj]
    return obj


def _to_toon(obj: Any) -> str:
    """Serialize a Python object to TOON format using the python-toon library."""
    return toon.encode(obj)


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
        "high": round(high, 2),
        "low": round(low, 2),
        "volume": round(volume),
        "candle_count": len(candles),
        "start_time": candles[0][0],
        "end_time": candles[-1][0],
    }


def _format_trade_pattern_analysis(analysis: Optional[Dict[str, Any]]) -> str:
    """Format trade pattern analysis into a compact string for the LLM prompt."""
    if not analysis:
        return ""

    lines = ["**Trade Pattern Analysis:**"]

    def _fmt_items(label: str, items: list, key: str):
        if items:
            parts = [f"{i[key]}:{i['win_rate']*100:.0f}%WR,{i['trades']}t,{i['avg_pnl']*100:+.2f}%" for i in items[:5]]
            lines.append(f"{label}: {' | '.join(parts)}")

    _fmt_items("BestEntry", analysis.get("best_entry_conditions", []), "condition")
    _fmt_items("BestTF", analysis.get("best_timeframes", []), "timeframe")
    _fmt_items("BestExit", analysis.get("best_exit_reasons", []), "exit_reason")
    _fmt_items("BestConf", analysis.get("best_confidence_ranges", []), "range")
    _fmt_items("BestSym", analysis.get("best_symbols", []), "symbol")
    _fmt_items("WorstSym", analysis.get("worst_symbols", []), "symbol")

    avg_win = analysis.get("avg_hold_time_winning")
    avg_loss = analysis.get("avg_hold_time_losing")
    if avg_win is not None or avg_loss is not None:
        win_str = f"{avg_win/3600:.1f}h" if avg_win is not None else "N/A"
        loss_str = f"{avg_loss/3600:.1f}h" if avg_loss is not None else "N/A"
        lines.append(f"HoldTime: W={win_str},L={loss_str}")

    lines.append("Favor high-WR conditions/timeframes. Avoid low-WR symbols.")
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
        # Cache "No recent news" for a short time to avoid repeated DB queries
        redis_client.set(cache_key, json.dumps(result), ex=60)
        return result
    else:
        try:
            formatted = _format_news_for_prompt(articles)
            prompt = (
                f"Here are recent news headlines and summaries for {symbol}:\n\n"
                f"{formatted}\n\n"
                "Based on these articles, write a single concise sentence (max 20 words) "
                "that summarizes the overall sentiment and the key reason for it. "
                "Do not include any other text."
            )
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": compact_prompt(prompt)},
            ]
            llm_result = get_cached_llm_response("", "", ttl=300, model_type=model_type, symbol=symbol, messages=messages)
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

    # Use short TTL for error/failure results, full TTL for successful summaries
    if result["summary"] in ("Could not generate summary.", "No recent news."):
        ttl = 60
    else:
        ttl = settings.NEWS_CACHE_TTL_SECONDS
    redis_client.set(cache_key, json.dumps(result), ex=ttl)
    return result


async def get_cached_news_summary_async(symbol: str, model_type: str = "weak") -> dict:
    """
    Asynchronous wrapper for get_cached_news_summary.
    Runs the blocking LLM call in a dedicated thread pool to avoid blocking
    the event loop and exhausting the default executor.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _llm_executor,
        get_cached_news_summary,
        symbol,
        model_type,
    )
