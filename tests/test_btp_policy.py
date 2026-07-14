import pytest
from unittest.mock import patch
from src.utils.btp_policy import BTPPolicy


def test_is_btp_valid():
    assert BTPPolicy.is_btp("IT0001234567") is True


def test_is_btp_invalid():
    assert BTPPolicy.is_btp("AAPL") is False


@patch("src.utils.btp_policy.settings")
def test_compute_fees_primary_issuance(mock_settings):
    mock_settings.BTP_IS_PRIMARY_ISSUANCE = True
    result = BTPPolicy.compute_fees("BUY", 10000.0)
    assert result == 0.0


@patch("src.utils.btp_policy.settings")
def test_compute_fees_secondary(mock_settings):
    mock_settings.BTP_IS_PRIMARY_ISSUANCE = False
    mock_settings.BTP_FEE_PERC = 0.001
    mock_settings.BTP_MIN_FEE = 5.0
    result = BTPPolicy.compute_fees("BUY", 10000.0)
    assert result == 10.0
