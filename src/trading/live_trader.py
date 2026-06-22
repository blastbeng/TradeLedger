import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class LiveTrader:
    """Dummy trader. Alpaca is no longer used. Will be replaced by PaperTrader in Step 4."""

    def __init__(self, trading_client=None):
        self.trading_client = trading_client

    def get_balance(self, currency: str) -> float:
        return 0.0

    def fetch_balance(self) -> Dict[str, float]:
        return {}

    def create_market_buy_order(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("Live trading is disabled. Use paper mode.")

    def create_market_sell_order(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("Live trading is disabled. Use paper mode.")

    def create_stop_buy_order(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("Live trading is disabled. Use paper mode.")

    def create_stop_sell_order(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("Live trading is disabled. Use paper mode.")

    def create_stop_limit_buy_order(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("Live trading is disabled. Use paper mode.")

    def create_stop_limit_sell_order(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("Live trading is disabled. Use paper mode.")

    def create_trailing_stop_buy_order(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("Live trading is disabled. Use paper mode.")

    def create_trailing_stop_sell_order(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("Live trading is disabled. Use paper mode.")

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        return []

    def cancel_order(self, order_id: str) -> bool:
        return False

    def get_trade_history(self) -> List[Dict[str, Any]]:
        return []


    # ------------------------------------------------------------------
    # Order placement
    # ------------------------------------------------------------------
    @retry_on_rate_limit(max_retries=3, base_delay=1.0)
    def create_market_buy_order(
        self, symbol: str, quote_amount: float, timeout: float = 60.0,
        limit_price: Optional[float] = None, time_in_force: str = "day"
    ) -> Dict[str, Any]:
        base = symbol.split("/")[0]

        asset = self.trading_client.get_asset(base)
        if not asset.fractionable:
            # Non-fractionable asset: must use integer qty
            if limit_price is not None:
                if limit_price <= 0:
                    raise ValueError(f"Invalid limit_price {limit_price} for {symbol}")
                price = limit_price
            else:
                # Fetch latest quote to determine price for integer qty calculation
                quote = self.trading_client.get_latest_quote(base)
                price = float(quote.ask_price)
            
            qty = int(quote_amount / price)
            if qty < 1:
                raise ValueError(f"Insufficient funds to buy 1 whole share of {symbol} (need {price}, have {quote_amount})")
            
            if limit_price is not None:
                tif = TimeInForce.DAY if time_in_force.lower() == "day" else TimeInForce.GTC
                order_data = LimitOrderRequest(
                    symbol=base,
                    qty=qty,
                    limit_price=limit_price,
                    side=OrderSide.BUY,
                    time_in_force=tif,
                )
            else:
                order_data = MarketOrderRequest(
                    symbol=base,
                    qty=qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                )
        else:
            # Fractionable asset: existing logic
            if limit_price is not None:
                if limit_price <= 0:
                    raise ValueError(f"Invalid limit_price {limit_price} for {symbol}")
                qty = quote_amount / limit_price
                tif = TimeInForce.DAY if time_in_force.lower() == "day" else TimeInForce.GTC
                order_data = LimitOrderRequest(
                    symbol=base,
                    qty=qty,
                    limit_price=limit_price,
                    side=OrderSide.BUY,
                    time_in_force=tif,
                )
            else:
                order_data = MarketOrderRequest(
                    symbol=base,
                    notional=quote_amount,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                )
        order = self.trading_client.submit_order(order_data)

        if limit_price is not None:
            # Limit order – check if it is immediately marketable
            try:
                quote = self.trading_client.get_latest_quote(base)
                ask_price = float(quote.ask_price)
                marketable = limit_price >= ask_price
            except Exception as e:
                logger.warning(f"Could not fetch quote for buy limit marketability check: {e}")
                marketable = False

            if marketable:
                # Marketable limit order – wait briefly for fill
                filled_order = self._wait_for_order_fill(order.id, base, 10.0, cancel_on_timeout=False)
                if filled_order is not None:
                    return self._order_to_dict(filled_order, symbol)

            # Not marketable or didn't fill in the short window – return as open
            open_order = self.trading_client.get_order_by_id(order.id)
            return self._order_to_dict(open_order, symbol)
        else:
            # Market order – wait with full timeout; don't cancel so the engine
            # can queue it for monitoring if it hasn't filled yet.
            filled_order = self._wait_for_order_fill(order.id, base, timeout, cancel_on_timeout=False)
            if filled_order is None:
                # Order didn't fill in time but is still open on Alpaca –
                # return it as open so the engine can queue it.
                open_order = self.trading_client.get_order_by_id(order.id)
                return self._order_to_dict(open_order, symbol)
            return self._order_to_dict(filled_order, symbol)

    @retry_on_rate_limit(max_retries=3, base_delay=1.0)
    def create_market_sell_order(
        self, symbol: str, qty: float, timeout: float = 60.0,
        limit_price: Optional[float] = None, time_in_force: str = "day"
    ) -> Dict[str, Any]:
        base = symbol.split("/")[0]

        if limit_price is not None:
            if limit_price <= 0:
                raise ValueError(f"Invalid limit_price {limit_price} for {symbol}")
            tif = TimeInForce.DAY if time_in_force.lower() == "day" else TimeInForce.GTC
            order_data = LimitOrderRequest(
                symbol=base,
                qty=qty,
                limit_price=limit_price,
                side=OrderSide.SELL,
                time_in_force=tif,
            )
        else:
            order_data = MarketOrderRequest(
                symbol=base,
                qty=qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
        order = self.trading_client.submit_order(order_data)

        if limit_price is not None:
            # Limit order – check if it is immediately marketable
            try:
                quote = self.trading_client.get_latest_quote(base)
                bid_price = float(quote.bid_price)
                marketable = limit_price <= bid_price
            except Exception as e:
                logger.warning(f"Could not fetch quote for sell limit marketability check: {e}")
                marketable = False

            if marketable:
                # Marketable limit order – wait briefly for fill
                filled_order = self._wait_for_order_fill(order.id, base, 10.0, cancel_on_timeout=False)
                if filled_order is not None:
                    return self._order_to_dict(filled_order, symbol)

            # Not marketable or didn't fill in the short window – return as open
            open_order = self.trading_client.get_order_by_id(order.id)
            return self._order_to_dict(open_order, symbol)
        else:
            # Market order – wait with full timeout; don't cancel so the engine
            # can queue it for monitoring if it hasn't filled yet.
            filled_order = self._wait_for_order_fill(order.id, base, timeout, cancel_on_timeout=False)
            if filled_order is None:
                # Order didn't fill in time but is still open on Alpaca –
                # return it as open so the engine can queue it.
                open_order = self.trading_client.get_order_by_id(order.id)
                return self._order_to_dict(open_order, symbol)
            return self._order_to_dict(filled_order, symbol)

    # ------------------------------------------------------------------
    # Advanced order placement (stop, stop-limit, trailing-stop)
    # ------------------------------------------------------------------
    @retry_on_rate_limit(max_retries=3, base_delay=1.0)
    def create_stop_buy_order(
        self, symbol: str, quote_amount: float, stop_price: float,
        time_in_force: str = "day", timeout: float = 60.0
    ) -> Dict[str, Any]:
        base = symbol.split("/")[0]
        asset = self.trading_client.get_asset(base)
        tif = TimeInForce.DAY if time_in_force.lower() == "day" else TimeInForce.GTC
        stop_price = self._round_price(stop_price)

        if not asset.fractionable:
            # Need integer qty – use current ask price to compute qty
            quote = self.trading_client.get_latest_quote(base)
            price = float(quote.ask_price)
            qty = int(quote_amount / price)
            if qty < 1:
                raise ValueError(f"Insufficient funds to buy 1 whole share of {symbol} (need {price}, have {quote_amount})")
            order_data = StopOrderRequest(
                symbol=base,
                qty=qty,
                stop_price=stop_price,
                side=OrderSide.BUY,
                time_in_force=tif,
            )
        else:
            # Fractionable – use notional
            order_data = StopOrderRequest(
                symbol=base,
                notional=quote_amount,
                stop_price=stop_price,
                side=OrderSide.BUY,
                time_in_force=tif,
            )
        order = self.trading_client.submit_order(order_data)
        return self._order_to_dict(order, symbol)

    @retry_on_rate_limit(max_retries=3, base_delay=1.0)
    def create_stop_sell_order(
        self, symbol: str, qty: float, stop_price: float,
        time_in_force: str = "day", timeout: float = 60.0
    ) -> Dict[str, Any]:
        base = symbol.split("/")[0]
        tif = TimeInForce.DAY if time_in_force.lower() == "day" else TimeInForce.GTC
        stop_price = self._round_price(stop_price)
        order_data = StopOrderRequest(
            symbol=base,
            qty=qty,
            stop_price=stop_price,
            side=OrderSide.SELL,
            time_in_force=tif,
        )
        order = self.trading_client.submit_order(order_data)
        return self._order_to_dict(order, symbol)

    @retry_on_rate_limit(max_retries=3, base_delay=1.0)
    def create_stop_limit_buy_order(
        self, symbol: str, quote_amount: float, stop_price: float, limit_price: float,
        time_in_force: str = "day", timeout: float = 60.0
    ) -> Dict[str, Any]:
        base = symbol.split("/")[0]
        asset = self.trading_client.get_asset(base)
        tif = TimeInForce.DAY if time_in_force.lower() == "day" else TimeInForce.GTC
        stop_price = self._round_price(stop_price)
        limit_price = self._round_price(limit_price)

        if not asset.fractionable:
            # Use limit_price to compute integer qty
            qty = int(quote_amount / limit_price)
            if qty < 1:
                raise ValueError(f"Insufficient funds to buy 1 whole share of {symbol} (need {limit_price}, have {quote_amount})")
            order_data = StopLimitOrderRequest(
                symbol=base,
                qty=qty,
                stop_price=stop_price,
                limit_price=limit_price,
                side=OrderSide.BUY,
                time_in_force=tif,
            )
        else:
            order_data = StopLimitOrderRequest(
                symbol=base,
                notional=quote_amount,
                stop_price=stop_price,
                limit_price=limit_price,
                side=OrderSide.BUY,
                time_in_force=tif,
            )
        order = self.trading_client.submit_order(order_data)
        return self._order_to_dict(order, symbol)

    @retry_on_rate_limit(max_retries=3, base_delay=1.0)
    def create_stop_limit_sell_order(
        self, symbol: str, qty: float, stop_price: float, limit_price: float,
        time_in_force: str = "day", timeout: float = 60.0
    ) -> Dict[str, Any]:
        base = symbol.split("/")[0]
        tif = TimeInForce.DAY if time_in_force.lower() == "day" else TimeInForce.GTC
        stop_price = self._round_price(stop_price)
        limit_price = self._round_price(limit_price)
        order_data = StopLimitOrderRequest(
            symbol=base,
            qty=qty,
            stop_price=stop_price,
            limit_price=limit_price,
            side=OrderSide.SELL,
            time_in_force=tif,
        )
        order = self.trading_client.submit_order(order_data)
        return self._order_to_dict(order, symbol)

    @retry_on_rate_limit(max_retries=3, base_delay=1.0)
    def create_trailing_stop_buy_order(
        self, symbol: str, quote_amount: float, trail_offset: float,
        time_in_force: str = "day", timeout: float = 60.0
    ) -> Dict[str, Any]:
        base = symbol.split("/")[0]
        asset = self.trading_client.get_asset(base)
        tif = TimeInForce.DAY if time_in_force.lower() == "day" else TimeInForce.GTC
        trail_offset = self._round_price(trail_offset)

        if not asset.fractionable:
            # Need integer qty – use current ask price
            quote = self.trading_client.get_latest_quote(base)
            price = float(quote.ask_price)
            qty = int(quote_amount / price)
            if qty < 1:
                raise ValueError(f"Insufficient funds to buy 1 whole share of {symbol} (need {price}, have {quote_amount})")
            order_data = TrailingStopOrderRequest(
                symbol=base,
                qty=qty,
                trail_offset=trail_offset,
                side=OrderSide.BUY,
                time_in_force=tif,
            )
        else:
            order_data = TrailingStopOrderRequest(
                symbol=base,
                notional=quote_amount,
                trail_offset=trail_offset,
                side=OrderSide.BUY,
                time_in_force=tif,
            )
        order = self.trading_client.submit_order(order_data)
        return self._order_to_dict(order, symbol)

    @retry_on_rate_limit(max_retries=3, base_delay=1.0)
    def create_trailing_stop_sell_order(
        self, symbol: str, qty: float, trail_offset: float,
        time_in_force: str = "day", timeout: float = 60.0
    ) -> Dict[str, Any]:
        base = symbol.split("/")[0]
        tif = TimeInForce.DAY if time_in_force.lower() == "day" else TimeInForce.GTC
        trail_offset = self._round_price(trail_offset)
        order_data = TrailingStopOrderRequest(
            symbol=base,
            qty=qty,
            trail_offset=trail_offset,
            side=OrderSide.SELL,
            time_in_force=tif,
        )
        order = self.trading_client.submit_order(order_data)
        return self._order_to_dict(order, symbol)

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch open orders, optionally filtered by symbol."""
        request = GetOrdersRequest(status="open")
        if symbol:
            base = symbol.split("/")[0]
            request.symbols = [base]
        orders = self.trading_client.get_orders(request)
        return [self._order_to_dict(o, o.symbol) for o in orders]

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order by ID. Returns True if successful."""
        try:
            self.trading_client.cancel_order_by_id(order_id)
            return True
        except Exception:
            return False

    def get_trade_history(self) -> List[Dict[str, Any]]:
        """Fetch recent closed orders."""
        request = GetOrdersRequest(status=OrderStatus.CLOSED, limit=100)
        orders = self.trading_client.get_orders(request)
        return [self._order_to_dict(o, o.symbol) for o in orders]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _round_price(price: float) -> float:
        """Round a price to Alpaca's valid tick size: $0.01 for >=$1, $0.0001 for <$1."""
        if price >= 1.0:
            return round(price, 2)
        else:
            return round(price, 4)

    def _wait_for_order_fill(self, order_id: str, symbol: str, timeout: float, cancel_on_timeout: bool = True) -> Any:
        """Poll Alpaca until the order is filled, rejected, or cancelled.
        If the timeout expires and *cancel_on_timeout* is True, cancel the
        order to avoid orphans.  If *cancel_on_timeout* is False, simply
        return None and leave the order open on Alpaca."""
        start = time.time()
        while time.time() - start < timeout:
            order = self.trading_client.get_order_by_id(order_id)
            if order.status == OrderStatus.FILLED:
                return order
            elif order.status in (OrderStatus.REJECTED, OrderStatus.CANCELED, OrderStatus.EXPIRED):
                raise RuntimeError(f"Order {order_id} {order.status}")
            # PARTIALLY_FILLED → keep waiting
            time.sleep(0.5)

        # Timeout – fetch one last time in case it filled just after the loop
        order = self.trading_client.get_order_by_id(order_id)
        if order.status == OrderStatus.FILLED:
            return order

        if cancel_on_timeout:
            # Cancel to avoid leaving an orphan order on Alpaca
            try:
                self.trading_client.cancel_order_by_id(order_id)
                logger.warning(f"Order {order_id} for {symbol} cancelled after {timeout}s timeout.")
            except Exception as e:
                logger.error(f"Failed to cancel order {order_id}: {e}")

        return None

    def _order_to_dict(self, order, symbol: str) -> Dict[str, Any]:
        """Convert an Alpaca order object to the dict format expected by the engine."""
        qty = float(order.filled_qty) if order.filled_qty else 0.0
        price = float(order.filled_avg_price) if order.filled_avg_price else 0.0
        cost = qty * price
        status = 'closed' if order.status == OrderStatus.FILLED else 'open'
        result = {
            'id': str(order.id),
            'symbol': symbol,          # original pair (e.g., "AAPL/USD")
            'side': 'buy' if order.side == OrderSide.BUY else 'sell',
            'amount': qty,
            'price': price,
            'cost': cost,
            'fee': {'cost': 0.0, 'currency': 'USD'},
            'status': status,
            'timestamp': int(order.created_at.timestamp() * 1000) if order.created_at else int(time.time() * 1000),
        }
        # Include limit price if the order has one
        if hasattr(order, 'limit_price') and order.limit_price is not None:
            result['limit_price'] = float(order.limit_price)
        # Include stop price if present
        if hasattr(order, 'stop_price') and order.stop_price is not None:
            result['stop_price'] = float(order.stop_price)
        # Include trail offset if present
        if hasattr(order, 'trail_offset') and order.trail_offset is not None:
            result['trail_offset'] = float(order.trail_offset)
        return result
