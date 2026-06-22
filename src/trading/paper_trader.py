import logging
import time
import uuid
from typing import Dict, List, Optional, Any

from src.config.settings import settings
from src.database import load_paper_balances, save_paper_balances, load_paper_orders, save_paper_orders
from src.exchanges.market_data import get_quotes

logger = logging.getLogger(__name__)


class PaperOrder:
    """Simple order object for the paper trader."""

    def __init__(
        self,
        order_id: str,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: Optional[float] = None,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        trail_offset: Optional[float] = None,
        time_in_force: str = "day",
        status: str = "open",
        filled_qty: float = 0.0,
        filled_avg_price: float = 0.0,
        timestamp: Optional[int] = None,
    ):
        self.id = order_id
        self.symbol = symbol
        self.side = side
        self.order_type = order_type
        self.amount = amount
        self.price = price
        self.limit_price = limit_price
        self.stop_price = stop_price
        self.trail_offset = trail_offset
        self.time_in_force = time_in_force
        self.status = status
        self.filled_qty = filled_qty
        self.filled_avg_price = filled_avg_price
        self.timestamp = timestamp or int(time.time() * 1000)
        self._highest_price: Optional[float] = None
        self._lowest_price: Optional[float] = None


class PaperTrader:
    """Custom paper trading simulator with configurable fees."""

    def __init__(self, trading_client=None):
        self.fee_pct = settings.PAPER_TRADING_FEE_PCT
        self.base_currency = settings.BASE_CURRENCY
        self._balances: Dict[str, float] = {}
        self._orders: Dict[str, PaperOrder] = {}
        self._load_balances()

    # ------------------------------------------------------------------
    # Balance management
    # ------------------------------------------------------------------

    def _load_balances(self):
        """Load balances and open orders from SQLite, or initialize with defaults."""
        self._balances = load_paper_balances()
        if not self._balances:
            self._balances = {self.base_currency: settings.PAPER_INITIAL_BALANCE}
            self._save_balances()
        if self.base_currency not in self._balances:
            self._balances[self.base_currency] = 0.0

        # Load persisted open orders
        persisted_orders = load_paper_orders()
        for od in persisted_orders:
            if od.get("status") == "open":
                order = self._dict_to_order(od)
                self._orders[order.id] = order
        if self._orders:
            logger.info(f"Loaded {len(self._orders)} persisted open paper orders.")

    def _save_balances(self):
        """Persist balances to SQLite."""
        save_paper_balances(self._balances)

    def _save_orders(self):
        """Persist open orders to SQLite."""
        orders_list = []
        for order in self._orders.values():
            orders_list.append({
                "id": order.id,
                "symbol": order.symbol,
                "side": order.side,
                "order_type": order.order_type,
                "amount": order.amount,
                "price": order.price,
                "limit_price": order.limit_price,
                "stop_price": order.stop_price,
                "trail_offset": order.trail_offset,
                "time_in_force": order.time_in_force,
                "status": order.status,
                "filled_qty": order.filled_qty,
                "filled_avg_price": order.filled_avg_price,
                "timestamp": order.timestamp,
                "_highest_price": order._highest_price,
                "_lowest_price": order._lowest_price,
            })
        save_paper_orders(orders_list)

    @staticmethod
    def _dict_to_order(d: dict) -> PaperOrder:
        """Reconstruct a PaperOrder from a persisted dict."""
        order = PaperOrder(
            order_id=d["id"],
            symbol=d["symbol"],
            side=d["side"],
            order_type=d["order_type"],
            amount=d["amount"],
            price=d.get("price"),
            limit_price=d.get("limit_price"),
            stop_price=d.get("stop_price"),
            trail_offset=d.get("trail_offset"),
            time_in_force=d.get("time_in_force", "gtc"),
            status=d.get("status", "open"),
            filled_qty=d.get("filled_qty", 0.0),
            filled_avg_price=d.get("filled_avg_price", 0.0),
            timestamp=d.get("timestamp"),
        )
        order._highest_price = d.get("_highest_price")
        order._lowest_price = d.get("_lowest_price")
        return order

    def fetch_balance(self) -> Dict[str, float]:
        return dict(self._balances)

    def get_balance(self, currency: str) -> float:
        return self._balances.get(currency, 0.0)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_current_price(self, symbol: str) -> Optional[float]:
        base = symbol.split("/")[0] if "/" in symbol else symbol
        try:
            quotes = get_quotes(None, [base])
            q = quotes.get(base, {})
            return q.get("last")
        except Exception as e:
            logger.warning(f"Failed to fetch price for {symbol}: {e}")
            return None

    @staticmethod
    def _generate_order_id() -> str:
        return str(uuid.uuid4())

    @staticmethod
    def _make_order_dict(
        order: PaperOrder,
        cost: float = 0.0,
        fee_cost: float = 0.0,
        fee_currency: str = "",
    ) -> Dict[str, Any]:
        return {
            "id": order.id,
            "status": order.status,
            "symbol": order.symbol,
            "side": order.side,
            "amount": order.filled_qty,
            "price": order.filled_avg_price,
            "cost": cost,
            "fee": {"cost": fee_cost, "currency": fee_currency},
            "timestamp": order.timestamp,
            "limit_price": order.limit_price,
            "stop_price": order.stop_price,
            "order_type": order.order_type,
        }

    def _fill_order(self, order: PaperOrder, fill_price: float, base: str, quote: str):
        """Fill an open order and update balances."""
        if order.side == "buy":
            # amount is in quote currency
            base_amount = order.amount / fill_price
            cost = base_amount * fill_price
            fee_cost = cost * self.fee_pct
            total_cost = cost + fee_cost

            quote_balance = self._balances.get(quote, 0.0)
            if total_cost > quote_balance:
                order.status = "rejected"
                logger.warning(
                    f"Paper buy rejected for {order.symbol}: insufficient {quote} "
                    f"(need {total_cost:.2f}, have {quote_balance:.2f})"
                )
                return

            self._balances[quote] = quote_balance - total_cost
            self._balances[base] = self._balances.get(base, 0.0) + base_amount
            order.filled_qty = base_amount
        else:
            # amount is in base currency
            base_amount = order.amount
            base_balance = self._balances.get(base, 0.0)
            if base_amount > base_balance:
                order.status = "rejected"
                logger.warning(
                    f"Paper sell rejected for {order.symbol}: insufficient {base} "
                    f"(need {base_amount:.6f}, have {base_balance:.6f})"
                )
                return

            cost = base_amount * fill_price
            fee_cost = cost * self.fee_pct
            net_quote = cost - fee_cost

            self._balances[base] = base_balance - base_amount
            self._balances[quote] = self._balances.get(quote, 0.0) + net_quote
            order.filled_qty = base_amount

        order.filled_avg_price = fill_price
        order.status = "filled"
        self._save_balances()
        self._save_orders()
        logger.info(
            f"Paper order filled: {order.side} {order.symbol} "
            f"qty={order.filled_qty:.6f} @ {fill_price:.4f}"
        )

    # ------------------------------------------------------------------
    # Market / limit orders
    # ------------------------------------------------------------------

    def create_market_buy_order(
        self,
        symbol: str,
        amount: float,
        fill_timeout: float = 60.0,
        limit_price: Optional[float] = None,
        time_in_force: str = "day",
    ) -> Dict[str, Any]:
        """Execute a market buy. amount is in quote currency."""
        price = self._get_current_price(symbol)
        if price is None or price <= 0:
            return {
                "id": "", "status": "rejected", "symbol": symbol, "side": "buy",
                "amount": 0, "price": 0, "cost": 0,
                "fee": {"cost": 0, "currency": self.base_currency},
                "timestamp": int(time.time() * 1000),
            }

        base = symbol.split("/")[0] if "/" in symbol else symbol
        quote = symbol.split("/")[1] if "/" in symbol else self.base_currency

        # If limit_price provided, check marketability
        if limit_price is not None:
            if price > limit_price:
                # Not marketable – create open limit order
                order_id = self._generate_order_id()
                order = PaperOrder(
                    order_id=order_id, symbol=symbol, side="buy",
                    order_type="limit", amount=amount, limit_price=limit_price,
                    time_in_force=time_in_force, status="open",
                )
                self._orders[order_id] = order
                self._save_orders()
                return self._make_order_dict(order)
            fill_price = limit_price
        else:
            fill_price = price

        base_amount = amount / fill_price
        cost = base_amount * fill_price
        fee_cost = cost * self.fee_pct
        fee_currency = quote
        total_cost = cost + fee_cost

        quote_balance = self._balances.get(quote, 0.0)
        if total_cost > quote_balance:
            logger.warning(
                f"Insufficient {quote} for buy: need {total_cost:.2f}, have {quote_balance:.2f}"
            )
            return {
                "id": "", "status": "rejected", "symbol": symbol, "side": "buy",
                "amount": 0, "price": fill_price, "cost": 0,
                "fee": {"cost": 0, "currency": fee_currency},
                "timestamp": int(time.time() * 1000),
            }

        self._balances[quote] = quote_balance - total_cost
        self._balances[base] = self._balances.get(base, 0.0) + base_amount
        self._save_balances()

        order_id = self._generate_order_id()
        order = PaperOrder(
            order_id=order_id, symbol=symbol, side="buy", order_type="market",
            amount=base_amount, price=fill_price, filled_qty=base_amount,
            filled_avg_price=fill_price, status="filled",
        )
        self._orders[order_id] = order

        return {
            "id": order_id, "status": "filled", "symbol": symbol, "side": "buy",
            "amount": base_amount, "price": fill_price, "cost": cost,
            "fee": {"cost": fee_cost, "currency": fee_currency},
            "timestamp": order.timestamp,
        }

    def create_market_sell_order(
        self,
        symbol: str,
        amount: float,
        fill_timeout: float = 60.0,
        limit_price: Optional[float] = None,
        time_in_force: str = "day",
    ) -> Dict[str, Any]:
        """Execute a market sell. amount is in base currency."""
        price = self._get_current_price(symbol)
        if price is None or price <= 0:
            return {
                "id": "", "status": "rejected", "symbol": symbol, "side": "sell",
                "amount": 0, "price": 0, "cost": 0,
                "fee": {"cost": 0, "currency": self.base_currency},
                "timestamp": int(time.time() * 1000),
            }

        base = symbol.split("/")[0] if "/" in symbol else symbol
        quote = symbol.split("/")[1] if "/" in symbol else self.base_currency

        # If limit_price provided, check marketability
        if limit_price is not None:
            if price < limit_price:
                # Not marketable – create open limit order
                order_id = self._generate_order_id()
                order = PaperOrder(
                    order_id=order_id, symbol=symbol, side="sell",
                    order_type="limit", amount=amount, limit_price=limit_price,
                    time_in_force=time_in_force, status="open",
                )
                self._orders[order_id] = order
                self._save_orders()
                return self._make_order_dict(order)
            fill_price = limit_price
        else:
            fill_price = price

        base_balance = self._balances.get(base, 0.0)
        if amount > base_balance:
            logger.warning(
                f"Insufficient {base} for sell: need {amount:.6f}, have {base_balance:.6f}"
            )
            return {
                "id": "", "status": "rejected", "symbol": symbol, "side": "sell",
                "amount": 0, "price": fill_price, "cost": 0,
                "fee": {"cost": 0, "currency": quote},
                "timestamp": int(time.time() * 1000),
            }

        cost = amount * fill_price
        fee_cost = cost * self.fee_pct
        fee_currency = quote
        net_quote = cost - fee_cost

        self._balances[base] = base_balance - amount
        self._balances[quote] = self._balances.get(quote, 0.0) + net_quote
        self._save_balances()

        order_id = self._generate_order_id()
        order = PaperOrder(
            order_id=order_id, symbol=symbol, side="sell", order_type="market",
            amount=amount, price=fill_price, filled_qty=amount,
            filled_avg_price=fill_price, status="filled",
        )
        self._orders[order_id] = order

        return {
            "id": order_id, "status": "filled", "symbol": symbol, "side": "sell",
            "amount": amount, "price": fill_price, "cost": cost,
            "fee": {"cost": fee_cost, "currency": fee_currency},
            "timestamp": order.timestamp,
        }

    # ------------------------------------------------------------------
    # Stop / stop-limit / trailing-stop orders
    # ------------------------------------------------------------------

    def create_stop_buy_order(
        self, symbol: str, amount: float, stop_price: float,
        time_in_force: str = "gtc", timeout: float = 60.0,
    ) -> Dict[str, Any]:
        order_id = self._generate_order_id()
        order = PaperOrder(
            order_id=order_id, symbol=symbol, side="buy", order_type="stop",
            amount=amount, stop_price=stop_price, time_in_force=time_in_force,
            status="open",
        )
        self._orders[order_id] = order
        self._save_orders()
        return self._make_order_dict(order)

    def create_stop_sell_order(
        self, symbol: str, amount: float, stop_price: float,
        time_in_force: str = "gtc", timeout: float = 60.0,
    ) -> Dict[str, Any]:
        order_id = self._generate_order_id()
        order = PaperOrder(
            order_id=order_id, symbol=symbol, side="sell", order_type="stop",
            amount=amount, stop_price=stop_price, time_in_force=time_in_force,
            status="open",
        )
        self._orders[order_id] = order
        self._save_orders()
        return self._make_order_dict(order)

    def create_stop_limit_buy_order(
        self, symbol: str, amount: float, stop_price: float,
        limit_price: float, time_in_force: str = "gtc", timeout: float = 60.0,
    ) -> Dict[str, Any]:
        order_id = self._generate_order_id()
        order = PaperOrder(
            order_id=order_id, symbol=symbol, side="buy", order_type="stop_limit",
            amount=amount, stop_price=stop_price, limit_price=limit_price,
            time_in_force=time_in_force, status="open",
        )
        self._orders[order_id] = order
        self._save_orders()
        return self._make_order_dict(order)

    def create_stop_limit_sell_order(
        self, symbol: str, amount: float, stop_price: float,
        limit_price: float, time_in_force: str = "gtc", timeout: float = 60.0,
    ) -> Dict[str, Any]:
        order_id = self._generate_order_id()
        order = PaperOrder(
            order_id=order_id, symbol=symbol, side="sell", order_type="stop_limit",
            amount=amount, stop_price=stop_price, limit_price=limit_price,
            time_in_force=time_in_force, status="open",
        )
        self._orders[order_id] = order
        self._save_orders()
        return self._make_order_dict(order)

    def create_trailing_stop_buy_order(
        self, symbol: str, amount: float, trail_offset: float,
        time_in_force: str = "gtc", timeout: float = 60.0,
    ) -> Dict[str, Any]:
        order_id = self._generate_order_id()
        order = PaperOrder(
            order_id=order_id, symbol=symbol, side="buy", order_type="trailing_stop",
            amount=amount, trail_offset=trail_offset, time_in_force=time_in_force,
            status="open",
        )
        self._orders[order_id] = order
        self._save_orders()
        return self._make_order_dict(order)

    def create_trailing_stop_sell_order(
        self, symbol: str, amount: float, trail_offset: float,
        time_in_force: str = "gtc", timeout: float = 60.0,
    ) -> Dict[str, Any]:
        order_id = self._generate_order_id()
        order = PaperOrder(
            order_id=order_id, symbol=symbol, side="sell", order_type="trailing_stop",
            amount=amount, trail_offset=trail_offset, time_in_force=time_in_force,
            status="open",
        )
        self._orders[order_id] = order
        self._save_orders()
        return self._make_order_dict(order)

    # ------------------------------------------------------------------
    # Order polling / management
    # ------------------------------------------------------------------

    def get_order(self, order_id: str) -> Optional[PaperOrder]:
        """Get order by ID. Checks and updates open orders against current price."""
        order = self._orders.get(order_id)
        if order is None:
            return None
        if order.status != "open":
            return order

        price = self._get_current_price(order.symbol)
        if price is None:
            return order

        base = order.symbol.split("/")[0] if "/" in order.symbol else order.symbol
        quote = order.symbol.split("/")[1] if "/" in order.symbol else self.base_currency

        # Trailing stop
        if order.order_type == "trailing_stop":
            if order.side == "sell":
                if order._highest_price is None or price > order._highest_price:
                    order._highest_price = price
                if order._highest_price is not None and order.trail_offset is not None:
                    trigger = order._highest_price - order.trail_offset
                    if price <= trigger:
                        self._fill_order(order, price, base, quote)
            elif order.side == "buy":
                if order._lowest_price is None or price < order._lowest_price:
                    order._lowest_price = price
                if order._lowest_price is not None and order.trail_offset is not None:
                    trigger = order._lowest_price + order.trail_offset
                    if price >= trigger:
                        self._fill_order(order, price, base, quote)

        # Stop orders
        elif order.order_type == "stop":
            if order.side == "buy" and order.stop_price is not None:
                if price >= order.stop_price:
                    self._fill_order(order, price, base, quote)
            elif order.side == "sell" and order.stop_price is not None:
                if price <= order.stop_price:
                    self._fill_order(order, price, base, quote)

        # Stop-limit orders
        elif order.order_type == "stop_limit":
            if order.side == "buy" and order.stop_price is not None:
                if price >= order.stop_price:
                    fill = order.limit_price if order.limit_price else price
                    self._fill_order(order, fill, base, quote)
            elif order.side == "sell" and order.stop_price is not None:
                if price <= order.stop_price:
                    fill = order.limit_price if order.limit_price else price
                    self._fill_order(order, fill, base, quote)

        # Limit orders
        elif order.order_type == "limit":
            if order.side == "buy" and order.limit_price is not None:
                if price <= order.limit_price:
                    self._fill_order(order, order.limit_price, base, quote)
            elif order.side == "sell" and order.limit_price is not None:
                if price >= order.limit_price:
                    self._fill_order(order, order.limit_price, base, quote)

        if order.status == "filled":
            self._save_orders()

        return order

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        result = []
        for order in self._orders.values():
            if order.status != "open":
                continue
            if symbol is not None and order.symbol != symbol:
                continue
            result.append({
                "id": order.id,
                "symbol": order.symbol,
                "timestamp": order.timestamp,
            })
        return result

    def cancel_order(self, order_id: str) -> bool:
        order = self._orders.get(order_id)
        if order is None or order.status != "open":
            return False
        order.status = "canceled"
        self._save_orders()
        return True

    def get_trade_history(self) -> List[Dict[str, Any]]:
        return []
