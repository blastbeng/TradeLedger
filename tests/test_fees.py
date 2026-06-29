"""Unit tests for transaction fee calculations."""
import pytest
from src.exchanges.fees import calculate_transaction_costs


class TestStockFees:
    """Tests for standard stock/ETF fee calculations."""

    def test_buy_above_minimum(self):
        """Bank commission percentage applies when above €3.50 minimum."""
        # Trade value = €10,000 → 0.24% = €24 (above €3.50 min)
        result = calculate_transaction_costs("BUY", 100.0, 100, symbol="ENI.MI")
        assert result["gross_value"] == 10000.0
        assert result["bank_fee"] == pytest.approx(24.0 + 2.50)  # 24 + 2.50
        assert result["tobin_tax"] == pytest.approx(12.0)  # 0.12% of 10000
        assert result["net_value"] == pytest.approx(10000.0 + 26.50 + 12.0)

    def test_buy_below_minimum(self):
        """Minimum bank commission of €3.50 applies for small trades."""
        # Trade value = €100 → 0.24% = €0.24 (below €3.50 min)
        result = calculate_transaction_costs("BUY", 10.0, 10, symbol="ENI.MI")
        assert result["gross_value"] == 100.0
        assert result["bank_fee"] == pytest.approx(3.50 + 2.50)  # min + fixed
        assert result["tobin_tax"] == pytest.approx(0.12)  # 0.12% of 100
        assert result["net_value"] == pytest.approx(100.0 + 6.00 + 0.12)

    def test_sell_no_tobin_tax(self):
        """Tobin tax is not applied on SELL orders."""
        result = calculate_transaction_costs("SELL", 100.0, 100, symbol="ENI.MI")
        assert result["gross_value"] == 10000.0
        assert result["tobin_tax"] == 0.0
        assert result["bank_fee"] == pytest.approx(24.0 + 2.50)
        assert result["net_value"] == pytest.approx(10000.0 - 26.50)

    def test_invalid_operation_type(self):
        """Invalid operation type raises ValueError."""
        with pytest.raises(ValueError, match="operation_type must be 'BUY' or 'SELL'"):
            calculate_transaction_costs("HOLD", 100.0, 10, symbol="ENI.MI")


class TestBTPFees:
    """Tests for BTP bond fee calculations."""

    def test_btp_buy_no_tobin_tax(self):
        """BTP bonds are exempt from Tobin tax."""
        result = calculate_transaction_costs("BUY", 100.0, 100, symbol="IT0001234567")
        assert result["gross_value"] == 10000.0
        assert result["tobin_tax"] == 0.0
        assert result["bank_fee"] == pytest.approx(24.0)  # 0.24%, no fixed fee
        assert result["net_value"] == pytest.approx(10000.0 + 24.0)

    def test_btp_sell(self):
        """BTP sell has no Tobin tax and no fixed execution fee."""
        result = calculate_transaction_costs("SELL", 100.0, 100, symbol="IT0001234567")
        assert result["gross_value"] == 10000.0
        assert result["tobin_tax"] == 0.0
        assert result["bank_fee"] == pytest.approx(24.0)
        assert result["net_value"] == pytest.approx(10000.0 - 24.0)

    def test_btp_below_minimum(self):
        """BTP minimum fee of €3.50 applies for small trades."""
        result = calculate_transaction_costs("BUY", 10.0, 10, symbol="IT0001234567")
        assert result["gross_value"] == 100.0
        assert result["bank_fee"] == pytest.approx(3.50)  # min fee, no fixed
        assert result["tobin_tax"] == 0.0
