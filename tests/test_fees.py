import pytest
from unittest.mock import patch
from src.exchanges.fees import calculate_transaction_costs


@patch("src.exchanges.fees.settings")
def test_calculate_stock_buy_fees(mock_settings):
    mock_settings.STOCK_FEE_PERC = 0.001
    mock_settings.STOCK_FEE_MIN = 2.0
    mock_settings.STOCK_FEE_FIXED = 0.5
    mock_settings.TOBIN_TAX_RATE = 0.001
    result = calculate_transaction_costs("BUY", 100.0, 10.0, symbol="AAPL")
    assert result["gross_value"] == 1000.0
    assert result["bank_fee"] == 2.5
    assert result["tobin_tax"] == 1.0
    assert result["total_costs"] == 3.5
    assert result["net_value"] == 1003.5


@patch("src.utils.btp_policy.settings")
@patch("src.exchanges.fees.settings")
def test_calculate_btp_buy_fees(mock_fees_settings, mock_btp_settings):
    mock_fees_settings.STOCK_FEE_PERC = 0.001
    mock_fees_settings.STOCK_FEE_MIN = 2.0
    mock_fees_settings.STOCK_FEE_FIXED = 0.5
    mock_fees_settings.TOBIN_TAX_RATE = 0.001
    
    mock_btp_settings.BTP_IS_PRIMARY_ISSUANCE = False
    mock_btp_settings.BTP_FEE_PERC = 0.001
    mock_btp_settings.BTP_MIN_FEE = 5.0
    
    result = calculate_transaction_costs("BUY", 100.0, 10.0, symbol="IT0001234567")
    assert result["gross_value"] == 1000.0
    assert result["bank_fee"] == 10.0
    assert result["tobin_tax"] == 0.0
    assert result["total_costs"] == 10.0
    assert result["net_value"] == 1010.0
