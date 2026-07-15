import pytest
from src.config.settings import Settings


def test_parse_temperature_range_none():
    assert Settings.parse_temperature_range(None) is None


def test_parse_temperature_range_empty():
    assert Settings.parse_temperature_range("") is None
    assert Settings.parse_temperature_range("   ") is None


def test_parse_temperature_range_single_value():
    result = Settings.parse_temperature_range("0.2")
    assert result == (0.2, 0.2)


def test_parse_temperature_range_single_value_zero():
    result = Settings.parse_temperature_range("0.0")
    assert result == (0.0, 0.0)


def test_parse_temperature_range_single_value_max():
    result = Settings.parse_temperature_range("2.0")
    assert result == (2.0, 2.0)


def test_parse_temperature_range_range():
    result = Settings.parse_temperature_range("0.2-0.5")
    assert result == (0.2, 0.5)


def test_parse_temperature_range_range_with_spaces():
    result = Settings.parse_temperature_range(" 0.2 - 0.5 ")
    assert result == (0.2, 0.5)


def test_parse_temperature_range_invalid_reversed():
    with pytest.raises(ValueError):
        Settings.parse_temperature_range("0.5-0.2")


def test_parse_temperature_range_out_of_bounds_high():
    with pytest.raises(ValueError):
        Settings.parse_temperature_range("3.0")


def test_parse_temperature_range_out_of_bounds_negative():
    with pytest.raises(ValueError):
        Settings.parse_temperature_range("-0.5")


def test_parse_temperature_range_invalid_format():
    with pytest.raises(ValueError):
        Settings.parse_temperature_range("abc")


def test_parse_temperature_range_invalid_range_format():
    with pytest.raises(ValueError):
        Settings.parse_temperature_range("0.2-abc")
