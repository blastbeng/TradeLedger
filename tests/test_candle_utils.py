import pytest
from datetime import datetime, timezone
from src.exchanges.candle_utils import (
    _validate_and_clean_candles,
    _aggregate_candles,
    _merge_candles,
    detect_data_quality_issues,
)


# ---------- _validate_and_clean_candles ----------

def test_validate_and_clean_candles_valid():
    candles = [
        [1000, 100.0, 105.0, 99.0, 103.0, 1000.0],
        [2000, 103.0, 108.0, 102.0, 107.0, 1200.0],
    ]
    result = _validate_and_clean_candles(candles)
    assert len(result) == 2


def test_validate_and_clean_candles_non_positive_open():
    candles = [
        [1000, 0.0, 105.0, 99.0, 103.0, 1000.0],
        [2000, 103.0, 108.0, 102.0, 107.0, 1200.0],
    ]
    result = _validate_and_clean_candles(candles)
    assert len(result) == 1
    assert result[0][0] == 2000


def test_validate_and_clean_candles_non_positive_close():
    candles = [
        [1000, 100.0, 105.0, 99.0, 0.0, 1000.0],
        [2000, 103.0, 108.0, 102.0, 107.0, 1200.0],
    ]
    result = _validate_and_clean_candles(candles)
    assert len(result) == 1


def test_validate_and_clean_candles_negative_volume():
    candles = [
        [1000, 100.0, 105.0, 99.0, 103.0, -100.0],
        [2000, 103.0, 108.0, 102.0, 107.0, 1200.0],
    ]
    result = _validate_and_clean_candles(candles)
    assert len(result) == 1


def test_validate_and_clean_candles_invalid_high():
    # high < open
    candles = [
        [1000, 100.0, 95.0, 99.0, 103.0, 1000.0],
        [2000, 103.0, 108.0, 102.0, 107.0, 1200.0],
    ]
    result = _validate_and_clean_candles(candles)
    assert len(result) == 1


def test_validate_and_clean_candles_invalid_low():
    # low > close
    candles = [
        [1000, 100.0, 105.0, 110.0, 103.0, 1000.0],
        [2000, 103.0, 108.0, 102.0, 107.0, 1200.0],
    ]
    result = _validate_and_clean_candles(candles)
    assert len(result) == 1


def test_validate_and_clean_candles_duplicate_timestamps():
    candles = [
        [1000, 100.0, 105.0, 99.0, 103.0, 1000.0],
        [1000, 104.0, 106.0, 101.0, 105.0, 2000.0],  # duplicate ts
        [2000, 103.0, 108.0, 102.0, 107.0, 1200.0],
    ]
    result = _validate_and_clean_candles(candles)
    assert len(result) == 2
    # Should keep the last occurrence of the duplicate
    assert result[0][4] == 105.0


def test_validate_and_clean_candles_empty():
    result = _validate_and_clean_candles([])
    assert result == []


def test_validate_and_clean_candles_short_entry():
    candles = [[1000, 100.0, 105.0]]
    result = _validate_and_clean_candles(candles)
    assert result == []


def test_validate_and_clean_candles_sorted_output():
    candles = [
        [3000, 103.0, 108.0, 102.0, 107.0, 1200.0],
        [1000, 100.0, 105.0, 99.0, 103.0, 1000.0],
        [2000, 101.0, 106.0, 100.0, 105.0, 1100.0],
    ]
    result = _validate_and_clean_candles(candles)
    assert len(result) == 3
    assert result[0][0] == 1000
    assert result[1][0] == 2000
    assert result[2][0] == 3000


# ---------- _aggregate_candles ----------

def _ts(year, month, day=15):
    return int(datetime(year, month, day, tzinfo=timezone.utc).timestamp() * 1000)


def test_aggregate_candles_6m():
    candles = [
        [_ts(2020, 1), 100.0, 105.0, 95.0, 102.0, 1000.0],
        [_ts(2020, 2), 102.0, 108.0, 100.0, 107.0, 2000.0],
        [_ts(2020, 7), 107.0, 110.0, 105.0, 109.0, 1500.0],
        [_ts(2020, 8), 109.0, 112.0, 108.0, 111.0, 2500.0],
    ]
    result = _aggregate_candles(candles, "6M")
    assert len(result) == 2
    # First half (Jan–Jun)
    assert result[0][1] == 100.0   # open
    assert result[0][2] == 108.0   # high
    assert result[0][3] == 95.0    # low
    assert result[0][4] == 107.0   # close (last in group)
    assert result[0][5] == 3000.0  # volume
    # Second half (Jul–Dec)
    assert result[1][1] == 107.0
    assert result[1][2] == 112.0
    assert result[1][3] == 105.0
    assert result[1][4] == 111.0
    assert result[1][5] == 4000.0


def test_aggregate_candles_1y():
    candles = [
        [_ts(2020, 6), 100.0, 105.0, 95.0, 102.0, 1000.0],
        [_ts(2021, 6), 102.0, 108.0, 100.0, 107.0, 2000.0],
    ]
    result = _aggregate_candles(candles, "1Y")
    assert len(result) == 2


def test_aggregate_candles_3y():
    candles = [
        [_ts(2020, 6), 100.0, 105.0, 95.0, 102.0, 1000.0],
        [_ts(2023, 6), 102.0, 108.0, 100.0, 107.0, 2000.0],
    ]
    result = _aggregate_candles(candles, "3Y")
    assert len(result) == 1


def test_aggregate_candles_5y():
    candles = [
        [_ts(2020, 6), 100.0, 105.0, 95.0, 102.0, 1000.0],
        [_ts(2026, 6), 102.0, 108.0, 100.0, 107.0, 2000.0],
    ]
    result = _aggregate_candles(candles, "5Y")
    assert len(result) == 1


def test_aggregate_candles_invalid_tf_passthrough():
    candles = [
        [1000, 100.0, 105.0, 95.0, 102.0, 1000.0],
    ]
    result = _aggregate_candles(candles, "1d")
    assert result == candles


def test_aggregate_candles_empty():
    result = _aggregate_candles([], "6M")
    assert result == []


# ---------- _merge_candles ----------

def test_merge_candles_both_present():
    borsa = [[1000, 100.0, 105.0, 95.0, 102.0, 1000.0]]
    yf = [
        [1000, 101.0, 106.0, 96.0, 103.0, 2000.0],
        [2000, 103.0, 108.0, 102.0, 107.0, 1200.0],
    ]
    result = _merge_candles(borsa, yf)
    assert len(result) == 2
    # Borsa takes precedence for same timestamp
    assert result[0][1] == 100.0


def test_merge_candles_borsa_only():
    borsa = [[1000, 100.0, 105.0, 95.0, 102.0, 1000.0]]
    result = _merge_candles(borsa, None)
    assert len(result) == 1


def test_merge_candles_yf_only():
    yf = [[1000, 100.0, 105.0, 95.0, 102.0, 1000.0]]
    result = _merge_candles(None, yf)
    assert len(result) == 1


def test_merge_candles_both_empty():
    result = _merge_candles(None, None)
    assert result == []


def test_merge_candles_sorted():
    borsa = [[2000, 103.0, 108.0, 102.0, 107.0, 1200.0]]
    yf = [[1000, 100.0, 105.0, 95.0, 102.0, 1000.0]]
    result = _merge_candles(borsa, yf)
    assert result[0][0] == 1000
    assert result[1][0] == 2000


# ---------- detect_data_quality_issues ----------

def test_detect_data_quality_issues_none():
    candles = [
        [1000, 100.0, 101.0, 99.0, 100.5, 1000.0],
        [2000, 100.5, 101.5, 100.0, 101.0, 1500.0],
    ]
    result = detect_data_quality_issues(candles, "TEST")
    assert result is None


def test_detect_data_quality_issues_price_jump():
    candles = [
        [1000, 100.0, 101.0, 99.0, 100.0, 1000.0],
        [2000, 100.0, 130.0, 99.0, 125.0, 1500.0],  # 25% jump
    ]
    result = detect_data_quality_issues(candles, "TEST")
    assert result is not None
    assert "Large price jump" in result


def test_detect_data_quality_issues_gap():
    candles = [
        [1000, 100.0, 101.0, 99.0, 100.0, 1000.0],
        [2000, 115.0, 116.0, 114.0, 115.0, 1500.0],  # 15% gap
    ]
    result = detect_data_quality_issues(candles, "TEST")
    assert result is not None
    assert "Price gap" in result


def test_detect_data_quality_issues_zero_volume():
    candles = [
        [1000, 100.0, 101.0, 99.0, 100.0, 1000.0],
        [2000, 100.0, 101.0, 99.0, 100.5, 0.0],  # zero volume
    ]
    result = detect_data_quality_issues(candles, "TEST")
    assert result is not None
    assert "Zero volume" in result


def test_detect_data_quality_issues_empty():
    result = detect_data_quality_issues([], "TEST")
    assert result is None


def test_detect_data_quality_issues_single_candle():
    candles = [[1000, 100.0, 101.0, 99.0, 100.0, 1000.0]]
    result = detect_data_quality_issues(candles, "TEST")
    assert result is None
