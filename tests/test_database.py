import pytest
from src.database import _normalize_symbol, compute_btp_ytm, _adapt_sql


# ---------- _normalize_symbol ----------

def test_normalize_symbol_with_slash():
    assert _normalize_symbol("AAPL/USD") == "AAPL"


def test_normalize_symbol_without_slash():
    assert _normalize_symbol("AAPL") == "AAPL"


def test_normalize_symbol_empty():
    assert _normalize_symbol("") == ""


def test_normalize_symbol_btp_with_slash():
    assert _normalize_symbol("IT0001234567/EUR") == "IT0001234567"


# ---------- compute_btp_ytm ----------

def test_compute_btp_ytm_none_inputs():
    assert compute_btp_ytm(None, "2050-01-01", 100.0) is None
    assert compute_btp_ytm(2.5, None, 100.0) is None
    assert compute_btp_ytm(2.5, "2050-01-01", None) is None


def test_compute_btp_ytm_zero_price():
    assert compute_btp_ytm(2.5, "2050-01-01", 0.0) is None


def test_compute_btp_ytm_negative_price():
    assert compute_btp_ytm(2.5, "2050-01-01", -100.0) is None


def test_compute_btp_ytm_past_maturity():
    assert compute_btp_ytm(2.5, "2000-01-01", 100.0) is None


def test_compute_btp_ytm_at_par():
    # When price = par value (100), YTM ≈ coupon rate
    result = compute_btp_ytm(2.5, "2050-01-01", 100.0)
    assert result is not None
    assert abs(result - 2.5) < 0.5


def test_compute_btp_ytm_below_par():
    # When price < par, YTM > coupon rate
    result = compute_btp_ytm(2.5, "2050-01-01", 90.0)
    assert result is not None
    assert result > 2.5


def test_compute_btp_ytm_above_par():
    # When price > par, YTM < coupon rate
    result = compute_btp_ytm(2.5, "2050-01-01", 110.0)
    assert result is not None
    assert result < 2.5


def test_compute_btp_ytm_invalid_maturity_format():
    assert compute_btp_ytm(2.5, "invalid-date", 100.0) is None


# ---------- _adapt_sql ----------

def test_adapt_sql():
    from src.database import _backend
    sql = "SELECT * FROM users WHERE id = %s AND name = %s"
    result = _adapt_sql(sql)
    if _backend == "sqlite":
        assert result == "SELECT * FROM users WHERE id = ? AND name = ?"
    else:
        assert result == sql


def test_adapt_sql_no_placeholders():
    sql = "SELECT * FROM users"
    result = _adapt_sql(sql)
    assert result == sql
