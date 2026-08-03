import pytest
from unittest.mock import patch, MagicMock

from src.trading.paper_trader import PaperTrader


@pytest.fixture
def trader():
    with patch("src.trading.paper_trader.load_paper_balances", return_value={"EUR": 10000.0}), \
         patch("src.trading.paper_trader.load_paper_orders", return_value=[]), \
         patch("src.trading.paper_trader.save_paper_balances"), \
         patch("src.trading.paper_trader.save_paper_orders"), \
         patch("src.trading.paper_trader.threading.Thread.start"):
        t = PaperTrader()
        t._balances = {"EUR": 10000.0}
        yield t


def test_market_buy_order(trader):
    with patch("src.trading.paper_trader.get_quotes_cached", return_value={"AAPL": {"last": 100.0}}), \
         patch("src.trading.paper_trader.calculate_transaction_costs", return_value={"net_value": 1010.0, "total_costs": 10.0}), \
         patch.object(trader, "_get_dynamic_slippage", return_value=0.0), \
         patch.object(trader, "_get_max_fillable_volume", return_value=None):
        
        result = trader.create_market_buy_order("AAPL/EUR", 1000.0)
        assert result["status"] == "filled"
        assert trader.get_balance("AAPL") == pytest.approx(1000.0 / 100.0, rel=0.01)
        assert trader.get_balance("EUR") == 10000.0 - 1010.0


def test_market_sell_order(trader):
    trader._balances = {"EUR": 10000.0, "AAPL": 10.0}
    with patch("src.trading.paper_trader.get_quotes_cached", return_value={"AAPL": {"last": 100.0}}), \
         patch("src.trading.paper_trader.calculate_transaction_costs", return_value={"net_value": 990.0, "total_costs": 10.0}), \
         patch.object(trader, "_get_dynamic_slippage", return_value=0.0), \
         patch.object(trader, "_get_max_fillable_volume", return_value=None):
        
        result = trader.create_market_sell_order("AAPL/EUR", 5.0)
        assert result["status"] == "filled"
        assert trader.get_balance("AAPL") == 5.0
        assert trader.get_balance("EUR") == 10000.0 + 990.0


def test_stop_sell_order_trigger(trader):
    trader._balances = {"EUR": 10000.0, "AAPL": 10.0}
    with patch("src.trading.paper_trader.get_quotes_cached", return_value={"AAPL": {"last": 100.0}}), \
         patch.object(trader, "_get_dynamic_slippage", return_value=0.0), \
         patch.object(trader, "_get_max_fillable_volume", return_value=None), \
         patch("src.trading.paper_trader.calculate_transaction_costs", return_value={"net_value": 950.0, "total_costs": 10.0}):
        
        result = trader.create_stop_sell_order("AAPL/EUR", 5.0, 95.0)
        order_id = result["id"]
        assert result["status"] == "open"
        
        # Price drops below stop price
        with patch("src.trading.paper_trader.get_quotes_cached", return_value={"AAPL": {"last": 90.0}}):
            trader.get_order(order_id)
        
        order = trader.get_order(order_id)
        assert order.status == "filled"


def test_trailing_stop_sell_order_trigger(trader):
    trader._balances = {"EUR": 10000.0, "AAPL": 10.0}
    with patch("src.trading.paper_trader.get_quotes_cached", return_value={"AAPL": {"last": 100.0}}), \
         patch.object(trader, "_get_dynamic_slippage", return_value=0.0), \
         patch.object(trader, "_get_max_fillable_volume", return_value=None), \
         patch("src.trading.paper_trader.calculate_transaction_costs", return_value={"net_value": 950.0, "total_costs": 10.0}):
        
        result = trader.create_trailing_stop_sell_order("AAPL/EUR", 5.0, 5.0)
        order_id = result["id"]
        
        # Price rises
        with patch("src.trading.paper_trader.get_quotes_cached", return_value={"AAPL": {"last": 110.0}}):
            trader.get_order(order_id)
        
        # Price drops by trail offset
        with patch("src.trading.paper_trader.get_quotes_cached", return_value={"AAPL": {"last": 104.0}}):
            trader.get_order(order_id)
        
        order = trader.get_order(order_id)
        assert order.status == "filled"


def test_cancel_order(trader):
    trader._balances = {"EUR": 10000.0, "AAPL": 10.0}
    with patch("src.trading.paper_trader.get_quotes_cached", return_value={"AAPL": {"last": 100.0}}):
        result = trader.create_stop_sell_order("AAPL/EUR", 5.0, 95.0)
        order_id = result["id"]
        
        assert trader.cancel_order(order_id) is True
        order = trader.get_order(order_id)
        assert order.status == "canceled"


def test_market_buy_capped_volume(trader):
    with patch("src.trading.paper_trader.get_quotes_cached", return_value={"AAPL": {"last": 100.0}}), \
         patch("src.trading.paper_trader.calculate_transaction_costs", return_value={"net_value": 1010.0, "total_costs": 10.0}), \
         patch.object(trader, "_get_dynamic_slippage", return_value=0.0), \
         patch.object(trader, "_get_max_fillable_volume", return_value=5.0):
        
        result = trader.create_market_buy_order("AAPL/EUR", 1000.0)
        # Since requested is 10 units but max_vol is 5, it should cap the amount to 5 and fill
        assert result["status"] == "filled"
        assert result["amount"] == 5.0


def test_dynamic_slippage_applied(trader):
    with patch("src.trading.paper_trader.get_quotes_cached", return_value={"AAPL": {"last": 100.0}}), \
         patch("src.trading.paper_trader.calculate_transaction_costs", return_value={"net_value": 1050.0, "total_costs": 50.0}), \
         patch.object(trader, "_get_dynamic_slippage", return_value=0.05), \
         patch.object(trader, "_get_max_fillable_volume", return_value=None):
        
        result = trader.create_market_buy_order("AAPL/EUR", 1000.0)
        assert result["status"] == "filled"
        # 100.0 * (1 + 0.05) = 105.0
        assert result["price"] == pytest.approx(105.0, rel=0.01)
