import pytest
from unittest.mock import patch
from datetime import datetime, timezone
from src.llm.cache import estimate_tokens, compute_market_hash, _should_use_primary_model


def test_estimate_tokens():
    assert estimate_tokens("12345678") == 2


def test_compute_market_hash_ignores_volatile_fields():
    data1 = {"price": 100.0, "timestamp": 123456789}
    data2 = {"price": 100.0, "timestamp": 987654321}
    hash1 = compute_market_hash(data1)
    hash2 = compute_market_hash(data2)
    assert hash1 == hash2


# ---------- _normalize_text_for_cache ----------

def test_normalize_text_for_cache_rounds_floats():
    from src.llm.cache import _normalize_text_for_cache
    text = "price: 123.456789, volume: 0.000123456789"
    result = _normalize_text_for_cache(text)
    # 123.456789 → 4 decimal places → 123.4568
    # 0.000123456789 → 4 decimal places → 0.0001
    assert "123.4568" in result
    assert "0.0001" in result


def test_normalize_text_for_cache_empty():
    from src.llm.cache import _normalize_text_for_cache
    assert _normalize_text_for_cache("") == ""
    assert _normalize_text_for_cache(None) is None


def test_normalize_text_for_cache_no_numbers():
    from src.llm.cache import _normalize_text_for_cache
    text = "no numbers here"
    assert _normalize_text_for_cache(text) == text


def test_normalize_text_for_cache_integer():
    from src.llm.cache import _normalize_text_for_cache
    text = "count: 42"
    result = _normalize_text_for_cache(text)
    assert "42" in result


def test_normalize_text_for_cache_scientific_notation():
    from src.llm.cache import _normalize_text_for_cache
    text = "value: 1.234e+2"
    result = _normalize_text_for_cache(text)
    # Should normalize the number
    assert "1.234e+2" not in result or "123.4" in result


# ---------- _compute_fee_fingerprint ----------

def test_compute_fee_fingerprint_consistent():
    from src.llm.cache import _compute_fee_fingerprint
    fp1 = _compute_fee_fingerprint()
    fp2 = _compute_fee_fingerprint()
    assert fp1 == fp2
    assert len(fp1) == 8  # MD5 truncated to 8 chars


# ---------- _normalize_for_hash ----------

def test_normalize_for_hash_excludes_volatile():
    from src.llm.cache import _normalize_for_hash
    data = {"price": 100.0, "timestamp": 123456, "fetched_at": 789, "name": "test"}
    result = _normalize_for_hash(data)
    assert "price" in result
    assert "timestamp" not in result
    assert "fetched_at" not in result
    assert "name" in result


def test_normalize_for_hash_excludes_time_keys():
    from src.llm.cache import _normalize_for_hash
    data = {"price": 100.0, "time": 123456, "created_at": 789, "last_eval": 999}
    result = _normalize_for_hash(data)
    assert "price" in result
    assert "time" not in result
    assert "created_at" not in result
    assert "last_eval" not in result


def test_normalize_for_hash_excludes_datetime_key():
    from src.llm.cache import _normalize_for_hash
    data = {"price": 100.0, "datetime": "2024-01-01", "last_auto_resume": 123}
    result = _normalize_for_hash(data)
    assert "price" in result
    assert "datetime" not in result
    assert "last_auto_resume" not in result


def test_normalize_for_hash_rounds_floats():
    from src.llm.cache import _normalize_for_hash
    data = {"price": 100.123456789}
    result = _normalize_for_hash(data)
    # 5 significant figures: 100.12
    assert result["price"] == pytest.approx(100.12, rel=0.001)


def test_normalize_for_hash_none_value():
    from src.llm.cache import _normalize_for_hash
    data = {"value": None}
    result = _normalize_for_hash(data)
    assert result["value"] == "null"


def test_normalize_for_hash_nested():
    from src.llm.cache import _normalize_for_hash
    data = {"outer": {"inner": 1.23456789, "timestamp": 123}}
    result = _normalize_for_hash(data)
    assert "inner" in result["outer"]
    assert "timestamp" not in result["outer"]


def test_normalize_for_hash_list():
    from src.llm.cache import _normalize_for_hash
    data = [1.23456789, 2.3456789]
    result = _normalize_for_hash(data)
    assert isinstance(result, list)
    assert len(result) == 2


def test_normalize_for_hash_zero_float():
    from src.llm.cache import _normalize_for_hash
    data = {"value": 0.0}
    result = _normalize_for_hash(data)
    assert result["value"] == 0.0


def test_normalize_for_hash_max_depth():
    from src.llm.cache import _normalize_for_hash
    # Create deeply nested structure exceeding max depth of 10
    data = {"a": {"b": {"c": {"d": {"e": {"f": {"g": {"h": {"i": {"j": {"k": 1.0}}}}}}}}}}}
    result = _normalize_for_hash(data)
    # At depth > 10, returns None
    assert result is not None  # top-level dict is fine


# ---------- _strip_ohlcv_timestamps ----------

def test_strip_ohlcv_timestamps_list_of_lists():
    from src.llm.cache import _strip_ohlcv_timestamps
    data = {
        "ohlcv_data": [
            [1000, 100.0, 105.0, 95.0, 102.0, 1000.0],
            [2000, 102.0, 108.0, 100.0, 107.0, 2000.0],
        ],
        "other": "value",
    }
    result = _strip_ohlcv_timestamps(data)
    assert result["ohlcv_data"] == [
        [100.0, 105.0, 95.0, 102.0, 1000.0],
        [102.0, 108.0, 100.0, 107.0, 2000.0],
    ]
    assert result["other"] == "value"


def test_strip_ohlcv_timestamps_dict_of_timeframes():
    from src.llm.cache import _strip_ohlcv_timestamps
    data = {
        "candles": {
            "1d": [[1000, 100.0, 105.0, 95.0, 102.0, 1000.0]],
            "1h": [[2000, 102.0, 108.0, 100.0, 107.0, 2000.0]],
        }
    }
    result = _strip_ohlcv_timestamps(data)
    assert result["candles"]["1d"] == [[100.0, 105.0, 95.0, 102.0, 1000.0]]
    assert result["candles"]["1h"] == [[102.0, 108.0, 100.0, 107.0, 2000.0]]


def test_strip_ohlcv_timestamps_dict_candles():
    from src.llm.cache import _strip_ohlcv_timestamps
    data = {
        "raw_candles": [
            {"timestamp": 1000, "open": 100.0, "high": 105.0, "low": 95.0,
             "close": 102.0, "volume": 1000.0},
        ]
    }
    result = _strip_ohlcv_timestamps(data)
    assert "timestamp" not in result["raw_candles"][0]
    assert result["raw_candles"][0]["open"] == 100.0


def test_strip_ohlcv_timestamps_passthrough_non_ohlcv():
    from src.llm.cache import _strip_ohlcv_timestamps
    data = {"price": 100.0, "name": "AAPL"}
    result = _strip_ohlcv_timestamps(data)
    assert result == data


# ---------- compute_market_hash with OHLCV ----------

def test_compute_market_hash_strips_ohlcv():
    from src.llm.cache import compute_market_hash
    data1 = {
        "ohlcv_data": [[1000, 100.0, 105.0, 95.0, 102.0, 1000.0]],
        "price": 100.0,
    }
    data2 = {
        "ohlcv_data": [[2000, 100.0, 105.0, 95.0, 102.0, 1000.0]],  # different timestamp
        "price": 100.0,
    }
    hash1 = compute_market_hash(data1)
    hash2 = compute_market_hash(data2)
    assert hash1 == hash2


def test_compute_market_hash_different_prices():
    from src.llm.cache import compute_market_hash
    data1 = {"price": 100.0}
    data2 = {"price": 101.0}
    hash1 = compute_market_hash(data1)
    hash2 = compute_market_hash(data2)
    assert hash1 != hash2


# ---------- _is_italian_holiday ----------

def test_is_italian_holiday_new_year():
    from src.llm.cache import _is_italian_holiday
    from datetime import datetime
    assert _is_italian_holiday(datetime(2024, 1, 1)) is True


def test_is_italian_holiday_epiphany():
    from src.llm.cache import _is_italian_holiday
    from datetime import datetime
    assert _is_italian_holiday(datetime(2024, 1, 6)) is True


def test_is_italian_holiday_liberation_day():
    from src.llm.cache import _is_italian_holiday
    from datetime import datetime
    assert _is_italian_holiday(datetime(2024, 4, 25)) is True


def test_is_italian_holiday_labour_day():
    from src.llm.cache import _is_italian_holiday
    from datetime import datetime
    assert _is_italian_holiday(datetime(2024, 5, 1)) is True


def test_is_italian_holiday_republic_day():
    from src.llm.cache import _is_italian_holiday
    from datetime import datetime
    assert _is_italian_holiday(datetime(2024, 6, 2)) is True


def test_is_italian_holiday_assumption():
    from src.llm.cache import _is_italian_holiday
    from datetime import datetime
    assert _is_italian_holiday(datetime(2024, 8, 15)) is True


def test_is_italian_holiday_all_saints():
    from src.llm.cache import _is_italian_holiday
    from datetime import datetime
    assert _is_italian_holiday(datetime(2024, 11, 1)) is True


def test_is_italian_holiday_immaculate_conception():
    from src.llm.cache import _is_italian_holiday
    from datetime import datetime
    assert _is_italian_holiday(datetime(2024, 12, 8)) is True


def test_is_italian_holiday_christmas():
    from src.llm.cache import _is_italian_holiday
    from datetime import datetime
    assert _is_italian_holiday(datetime(2024, 12, 25)) is True


def test_is_italian_holiday_st_stephen():
    from src.llm.cache import _is_italian_holiday
    from datetime import datetime
    assert _is_italian_holiday(datetime(2024, 12, 26)) is True


def test_is_italian_holiday_not_holiday():
    from src.llm.cache import _is_italian_holiday
    from datetime import datetime
    assert _is_italian_holiday(datetime(2024, 7, 15)) is False
    assert _is_italian_holiday(datetime(2024, 3, 10)) is False


def test_is_italian_holiday_easter_monday_2024():
    from src.llm.cache import _is_italian_holiday
    from datetime import datetime
    # Easter 2024 is March 31, Easter Monday is April 1
    assert _is_italian_holiday(datetime(2024, 4, 1)) is True


def test_is_italian_holiday_easter_monday_2023():
    from src.llm.cache import _is_italian_holiday
    from datetime import datetime
    # Easter 2023 is April 9, Easter Monday is April 10
    assert _is_italian_holiday(datetime(2023, 4, 10)) is True


def test_is_italian_holiday_easter_monday_2025():
    from src.llm.cache import _is_italian_holiday
    from datetime import datetime
    # Easter 2025 is April 20, Easter Monday is April 21
    assert _is_italian_holiday(datetime(2025, 4, 21)) is True


# ---------- _should_use_primary_model ----------

@patch("src.llm.cache.datetime")
def test_should_use_primary_model_closed_weekend(mock_datetime):
    from src.llm import cache
    cache._primary_model_cache = None
    cache._primary_model_cache_ts = 0.0
    cache._primary_model_cache_settings = None

    mock_datetime.now.return_value = datetime(2024, 1, 6, 12, 0, 0, tzinfo=timezone.utc)
    mock_datetime.side_effect = lambda *a, **kw: datetime(*a, **kw)
    assert _should_use_primary_model() is False


@patch("src.llm.cache.datetime")
def test_should_use_primary_model_open(mock_datetime):
    from src.llm import cache
    cache._primary_model_cache = None
    cache._primary_model_cache_ts = 0.0
    cache._primary_model_cache_settings = None

    mock_datetime.now.return_value = datetime(2024, 1, 8, 9, 0, 0, tzinfo=timezone.utc)
    mock_datetime.side_effect = lambda *a, **kw: datetime(*a, **kw)
    assert _should_use_primary_model() is True


@patch("src.llm.cache.datetime")
def test_should_use_primary_model_premarket(mock_datetime):
    from src.llm import cache
    cache._primary_model_cache = None
    cache._primary_model_cache_ts = 0.0
    cache._primary_model_cache_settings = None

    mock_datetime.now.return_value = datetime(2024, 1, 8, 7, 30, 0, tzinfo=timezone.utc)
    mock_datetime.side_effect = lambda *a, **kw: datetime(*a, **kw)
    assert _should_use_primary_model() is True


@patch("src.llm.cache.datetime")
def test_should_use_primary_model_after_close(mock_datetime):
    from src.llm import cache
    cache._primary_model_cache = None
    cache._primary_model_cache_ts = 0.0
    cache._primary_model_cache_settings = None

    mock_datetime.now.return_value = datetime(2024, 1, 8, 17, 0, 0, tzinfo=timezone.utc)
    mock_datetime.side_effect = lambda *a, **kw: datetime(*a, **kw)
    assert _should_use_primary_model() is False
