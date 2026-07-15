import pytest
from src.llm.prompt_utils import (
    _round_floats,
    _timeframe_to_seconds,
    compact_prompt,
    _summarize_ohlcv,
    _format_news_for_prompt,
    _format_trade_pattern_analysis,
    _to_toon,
)


# ---------- _round_floats ----------

def test_round_floats_simple():
    assert _round_floats(3.14159, 2) == 3.14


def test_round_floats_dict():
    result = _round_floats({"a": 3.14159, "b": 2.71828}, 2)
    assert result == {"a": 3.14, "b": 2.72}


def test_round_floats_nested():
    result = _round_floats({"a": [1.111, 2.222], "b": {"c": 3.333}}, 2)
    assert result == {"a": [1.11, 2.22], "b": {"c": 3.33}}


def test_round_floats_non_float():
    assert _round_floats("hello", 2) == "hello"
    assert _round_floats(42, 2) == 42
    assert _round_floats(None, 2) is None


def test_round_floats_empty_list():
    assert _round_floats([], 2) == []


# ---------- _timeframe_to_seconds ----------

def test_timeframe_to_seconds_minutes():
    assert _timeframe_to_seconds("5m") == 300


def test_timeframe_to_seconds_hours():
    assert _timeframe_to_seconds("1h") == 3600


def test_timeframe_to_seconds_days():
    assert _timeframe_to_seconds("1d") == 86400


def test_timeframe_to_seconds_weeks():
    assert _timeframe_to_seconds("1w") == 604800


def test_timeframe_to_seconds_months():
    assert _timeframe_to_seconds("1M") == 2592000


def test_timeframe_to_seconds_years():
    assert _timeframe_to_seconds("1Y") == 31536000


def test_timeframe_to_seconds_invalid():
    assert _timeframe_to_seconds("invalid") == 3600  # default


def test_timeframe_to_seconds_multi():
    assert _timeframe_to_seconds("3m") == 180
    assert _timeframe_to_seconds("2h") == 7200


# ---------- compact_prompt ----------

def test_compact_prompt_multiple_spaces():
    result = compact_prompt("hello    world")
    assert result == "hello world"


def test_compact_prompt_multiple_newlines():
    result = compact_prompt("hello\n\n\nworld")
    assert result == "hello\nworld"


def test_compact_prompt_tabs():
    result = compact_prompt("hello\t\tworld")
    assert result == "hello world"


def test_compact_prompt_strip():
    result = compact_prompt("  hello world  ")
    assert result == "hello world"


def test_compact_prompt_mixed_whitespace():
    result = compact_prompt("  hello \t\t  world\n\n  ")
    assert result == "hello world"


# ---------- _summarize_ohlcv ----------

def test_summarize_ohlcv_basic():
    candles = [
        [1000, 100.0, 105.0, 95.0, 102.0, 1000.0],
        [2000, 102.0, 108.0, 100.0, 107.0, 2000.0],
    ]
    result = _summarize_ohlcv(candles)
    assert result is not None
    assert result["change_pct"] == 7.0  # (107-100)/100 * 100
    assert result["high"] == 108.0
    assert result["low"] == 95.0
    assert result["volume"] == 3000
    assert result["candle_count"] == 2
    assert result["start_time"] == 1000
    assert result["end_time"] == 2000


def test_summarize_ohlcv_empty():
    result = _summarize_ohlcv([])
    assert result is None


def test_summarize_ohlcv_single():
    candles = [[1000, 100.0, 105.0, 95.0, 100.0, 500.0]]
    result = _summarize_ohlcv(candles)
    assert result is not None
    assert result["change_pct"] == 0.0
    assert result["candle_count"] == 1


# ---------- _format_news_for_prompt ----------

def test_format_news_for_prompt_empty():
    result = _format_news_for_prompt([])
    assert result == "No recent news available."


def test_format_news_for_prompt_with_articles():
    articles = [
        {
            "source": "Reuters",
            "title": "Stock rises",
            "published_at": "2024-01-01",
            "sentiment": {"label": "positive", "compound": 0.8},
            "summary": "Good earnings report",
        },
    ]
    result = _format_news_for_prompt(articles)
    assert "Reuters" in result
    assert "Stock rises" in result
    assert "positive" in result
    assert "0.80" in result
    assert "Good earnings report" in result


def test_format_news_for_prompt_multiple():
    articles = [
        {
            "source": "Bloomberg",
            "title": "Market falls",
            "published_at": "2024-01-02",
            "sentiment": {"label": "negative", "compound": -0.5},
            "summary": "Bad news",
        },
        {
            "source": "CNBC",
            "title": "Tech surges",
            "published_at": "2024-01-03",
            "sentiment": {"label": "positive", "compound": 0.9},
            "summary": "Great results",
        },
    ]
    result = _format_news_for_prompt(articles)
    assert "1." in result
    assert "2." in result
    assert "Bloomberg" in result
    assert "CNBC" in result


# ---------- _format_trade_pattern_analysis ----------

def test_format_trade_pattern_analysis_empty():
    result = _format_trade_pattern_analysis(None)
    assert result == ""


def test_format_trade_pattern_analysis_with_data():
    analysis = {
        "best_entry_conditions": [
            {"condition": "rsi_oversold", "win_rate": 0.65, "trades": 10, "avg_pnl": 0.02},
        ],
        "best_timeframes": [
            {"timeframe": "1d", "win_rate": 0.55, "trades": 20, "avg_pnl": 0.01},
        ],
        "best_exit_reasons": [
            {"exit_reason": "take_profit", "win_rate": 0.70, "trades": 15, "avg_pnl": 0.03},
        ],
        "best_confidence_ranges": [
            {"range": "0.7-0.8", "win_rate": 0.60, "trades": 8, "avg_pnl": 0.015},
        ],
        "best_symbols": [
            {"symbol": "AAPL", "win_rate": 0.58, "trades": 12, "avg_pnl": 0.012},
        ],
        "worst_symbols": [
            {"symbol": "TSLA", "win_rate": 0.30, "trades": 10, "avg_pnl": -0.02},
        ],
        "avg_hold_time_winning": 7200,
        "avg_hold_time_losing": 3600,
    }
    result = _format_trade_pattern_analysis(analysis)
    assert "Trade Pattern Analysis" in result
    assert "rsi_oversold" in result
    assert "1d" in result
    assert "take_profit" in result
    assert "AAPL" in result
    assert "TSLA" in result
    assert "HoldTime" in result
    assert "Favor high-WR" in result


def test_format_trade_pattern_analysis_partial():
    analysis = {
        "best_entry_conditions": [
            {"condition": "macd_cross", "win_rate": 0.50, "trades": 5, "avg_pnl": 0.01},
        ],
    }
    result = _format_trade_pattern_analysis(analysis)
    assert "Trade Pattern Analysis" in result
    assert "macd_cross" in result
    assert "BestEntry" in result


# ---------- _to_toon ----------

def test_to_toon_dict():
    result = _to_toon({"key": "value", "num": 42})
    assert isinstance(result, str)


def test_to_toon_list():
    result = _to_toon([1, 2, 3])
    assert isinstance(result, str)


def test_to_toon_string():
    result = _to_toon("hello")
    assert isinstance(result, str)
