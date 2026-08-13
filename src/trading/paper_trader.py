import logging
import time
import threading
import uuid
from typing import Dict, List, Optional, Any

from src.config.settings import settings
from src.database import load_paper_balances, save_paper_balances, load_paper_orders, save_paper_orders, get_ohlcv
from src.exchanges.market_data import get_quotes, get_quotes_cached
from src.exchanges.fees import calculate_transaction_costs
from src.indicators import compute_atr
from src.utils.btp_policy import BTPPolicy

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
        self.base_currency = settings.BASE_CURRENCY
        self._balances: Dict[str, float] = {}
        self._orders: Dict[str, PaperOrder] = {}
        self.slippage_base_pct = 0.001  # 0.1% base slippage
        self.slippage_max_pct = 0.01    # 1.0% max slippage
        self._balances_dirty = False
        self._slippage_cache: Dict[str, tuple] = {}  # symbol -> (timestamp, slippage)
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._poll_interval_base = 15.0
        self._poll_interval_max = 120.0
        self._poll_interval = self._poll_interval_base
        self._consecutive_idle_polls = 0
        self._poller_thread = threading.Thread(target=self._poll_open_orders, daemon=True)
        self._poller_thread.start()
        self._load_balances()

    # ------------------------------------------------------------------
    # Balance management
    # ------------------------------------------------------------------

    def _load_balances(self):
        """Load balances and open orders from SQLite, or initialize with defaults."""
        self._balances = load_paper_balances()
        if not self._balances:
            self._balances = {self.base_currency: settings.PAPER_INITIAL_BALANCE}
            self._balances_dirty = True
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
        self._balances_dirty = False

    def _save_balances(self):
        """Persist balances to SQLite."""
        if not self._balances_dirty:
            return
        save_paper_balances(self._balances)
        self._balances_dirty = False

    def _save_orders(self):
        """Persist open orders to SQLite."""
        orders_list = []
        for order in self._orders.values():
            if order.status != "open":
                continue
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
        # Clean up old non-open orders to prevent unbounded memory growth
        if len(self._orders) > 100:
            stale_ids = [
                oid for oid, o in self._orders.items()
                if o.status != "open"
            ]
            for oid in stale_ids:
                del self._orders[oid]

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
        with self._lock:
            return dict(self._balances)

    def get_balance(self, currency: str) -> float:
        with self._lock:
            return self._balances.get(currency, 0.0)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_current_price(self, symbol: str) -> Optional[float]:
        base = symbol.split("/")[0] if "/" in symbol else symbol
        try:
            quotes = get_quotes_cached([base])
            q = quotes.get(base, {})
            return q.get("last")
        except Exception as e:
            logger.warning(f"Failed to fetch price for {symbol}: {type(e).__name__}: {e}")
            return None

    def _get_dynamic_slippage(self, symbol: str, price: float) -> float:
        """Compute dynamic slippage based on recent volume and volatility.
        
        Adapts the backtester's dynamic slippage model to live trading by using
        the most recent daily candle and a 20-day average volume.
        """
        btp_slippage = BTPPolicy.get_slippage_pct(symbol)
        if btp_slippage is not None:
            return btp_slippage

        # Check cache first
        cached = self._slippage_cache.get(symbol)
        if cached:
            ts, cached_slippage = cached
            if time.time() - ts < 5:
                return cached_slippage

        base = symbol.split("/")[0] if "/" in symbol else symbol
        try:
            candles = get_ohlcv(base, "1d", limit=21)
            if not candles or len(candles) < 5:
                return self.slippage_base_pct
            
            # Use the most recent (current) candle for current volume
            current_vol = candles[-1]["volume"] if candles else 0.0
            volumes = [c["volume"] for c in candles[:-1]]
            avg_vol = sum(volumes) / len(volumes) if volumes else 0.0
            
            slippage = self.slippage_base_pct
            if avg_vol > 0 and current_vol > 0:
                vol_ratio = avg_vol / current_vol
                if vol_ratio > 1.0:
                    slippage *= min(vol_ratio, 3.0)
            
            # Compute ATR for volatility adjustment
            candle_list = [
                [c["timestamp"], c["open"], c["high"], c["low"], c["close"], c["volume"]]
                for c in candles
            ]
            atr = compute_atr(candle_list, period=14)
            if atr and atr > 0 and price > 0:
                atr_pct = atr / price
                slippage += atr_pct * 0.05
                
            final_slippage = min(slippage, self.slippage_max_pct)
            self._slippage_cache[symbol] = (time.time(), final_slippage)
            return final_slippage
        except Exception as e:
            logger.warning(f"Failed to compute dynamic slippage for {symbol}: {type(e).__name__}: {e}")
            return self.slippage_base_pct

    def _get_max_fillable_volume(self, symbol: str) -> Optional[float]:
        """Estimate max fillable volume based on recent 1m candle volume, with daily fallback."""
        base = symbol.split("/")[0] if "/" in symbol else symbol
        try:
            candles = get_ohlcv(base, "1m", limit=2)
            if candles:
                # Use the most recent completed candle's volume
                vol = candles[-2]["volume"] if len(candles) >= 2 else candles[-1]["volume"]
                return vol * settings.PARTIAL_FILL_VOLUME_CAP_PCT

            # Fallback to daily volume if 1m is unavailable
            logger.info(f"1m volume unavailable for {symbol}, skipping partial fill cap.")
            return None
        except Exception as e:
            logger.warning(f"Failed to fetch volume for partial fill check for {symbol}: {type(e).__name__}: {e}")
            return None

    def _poll_open_orders(self):
        """Background thread to poll open orders and trigger stops promptly."""
        while not self._stop_event.is_set():
            try:
                with self._lock:
                    open_order_ids = [
                        oid for oid, o in self._orders.items() if o.status == "open"
                    ]
                state_changed = False
                for oid in open_order_ids:
                    prev_status = self._orders.get(oid).status if oid in self._orders else None
                    self.get_order(oid)
                    new_status = self._orders.get(oid).status if oid in self._orders else None
                    if prev_status != new_status:
                        state_changed = True
                if state_changed:
                    self._consecutive_idle_polls = 0
                    self._poll_interval = self._poll_interval_base
                else:
                    self._consecutive_idle_polls += 1
                    self._poll_interval = min(
                        self._poll_interval_base * (1.5 ** min(self._consecutive_idle_polls, 4)),
                        self._poll_interval_max
                    )
            except Exception as e:
                logger.warning(f"Error polling open orders: {e}")
            time.sleep(self._poll_interval)

    @staticmethod
    def _generate_order_id() -> str:
        return str(uuid.uuid4())

    def _compute_market_impact_pct(self, base_amount: float, max_vol: Optional[float]) -> float:
        """Compute market impact percentage using a simple square-root model."""
        if max_vol is None or max_vol <= 0:
            return 0.0
        impact_ratio = base_amount / max_vol
        return 0.05 * (impact_ratio ** 0.5)

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
        # Apply slippage to stop-order fills (they execute at market)
        if order.order_type in ("stop", "trailing_stop"):
            slippage_pct = self._get_dynamic_slippage(order.symbol, fill_price)
            if order.side == "buy":
                fill_price = fill_price * (1 + slippage_pct)
            else:
                fill_price = fill_price * (1 - slippage_pct)

        # Fetch max fillable volume outside the lock to avoid blocking during IO
        max_vol = self._get_max_fillable_volume(order.symbol)

        with self._lock:
            if order.side == "buy":
                # amount is in quote currency
                requested_base_amount = order.amount / fill_price
            else:
                # amount is in base currency
                requested_base_amount = order.amount

            # Check for volume-based partial fill
            is_partial = False
            if max_vol is not None and requested_base_amount > max_vol:
                logger.info(f"Partial fill for {order.symbol}: requested {requested_base_amount}, capped to {max_vol}")
                base_amount = max_vol
                is_partial = True
            else:
                base_amount = requested_base_amount

            # Apply market impact
            impact_pct = self._compute_market_impact_pct(base_amount, max_vol)
            if order.side == "buy":
                fill_price = fill_price * (1 + impact_pct)
            else:
                fill_price = fill_price * (1 - impact_pct)

            if order.side == "buy":
                costs = calculate_transaction_costs("BUY", fill_price, base_amount, symbol=order.symbol)
                total_cost = costs["net_value"]
                fee_cost = costs["total_costs"]
                fee_currency = quote

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
            else:
                base_balance = self._balances.get(base, 0.0)
                if base_amount > base_balance:
                    order.status = "rejected"
                    logger.warning(
                        f"Paper sell rejected for {order.symbol}: insufficient {base} "
                        f"(need {base_amount:.6f}, have {base_balance:.6f})"
                    )
                    return

                costs = calculate_transaction_costs("SELL", fill_price, base_amount, symbol=order.symbol)
                net_quote = costs["net_value"]
                fee_cost = costs["total_costs"]
                fee_currency = quote

                self._balances[base] = base_balance - base_amount
                self._balances[quote] = self._balances.get(quote, 0.0) + net_quote

            order.filled_avg_price = fill_price
            if is_partial:
                # Reduce remaining amount; keep order open for next poll
                if order.side == "buy":
                    order.amount -= base_amount * fill_price
                else:
                    order.amount -= base_amount
                order.filled_qty = (order.filled_qty or 0.0) + base_amount
                order.status = "open"
            else:
                order.filled_qty = (order.filled_qty or 0.0) + base_amount
                order.status = "filled"
            self._balances_dirty = True
            self._save_orders()
            self._save_balances()
        fill_status = "partially filled" if is_partial else "filled"
        logger.info(
            f"Paper order {fill_status}: {order.side} {order.symbol} "
            f"qty={base_amount:.6f} @ {fill_price:.4f}"
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
                with self._lock:
                    order_id = self._generate_order_id()
                    order = PaperOrder(
                        order_id=order_id, symbol=symbol, side="buy",
                        order_type="limit", amount=amount, limit_price=limit_price,
                        time_in_force=time_in_force, status="open",
                    )
                    self._orders[order_id] = order
                    self._save_orders()
                return self._make_order_dict(order)
            # Fill at the best available price (current price if lower than limit)
            fill_price = min(price, limit_price)
        else:
            fill_price = price

        # Apply dynamic slippage to market order fills
        slippage_pct = self._get_dynamic_slippage(symbol, fill_price)
        fill_price = fill_price * (1 + slippage_pct)

        max_vol = self._get_max_fillable_volume(symbol)
        
        # Apply market impact
        impact_pct = self._compute_market_impact_pct(amount / fill_price, max_vol)
        fill_price = fill_price * (1 + impact_pct)

        base_amount = amount / fill_price

        # Check for volume-based partial fill
        if max_vol is not None and base_amount > max_vol:
            logger.info(f"Partial fill for {symbol}: requested {base_amount}, capped to {max_vol}")
            base_amount = max_vol

        costs = calculate_transaction_costs("BUY", fill_price, base_amount, symbol=symbol)
        total_cost = costs["net_value"]
        fee_cost = costs["total_costs"]
        fee_currency = quote

        with self._lock:
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

            filled_base_amount = base_amount
            filled_cost = filled_base_amount * fill_price

            self._balances[quote] = quote_balance - (filled_cost + fee_cost)
            self._balances[base] = self._balances.get(base, 0.0) + filled_base_amount
            self._balances_dirty = True

            order_id = self._generate_order_id()
            order = PaperOrder(
                order_id=order_id, symbol=symbol, side="buy", order_type="market",
                amount=base_amount, price=fill_price, filled_qty=filled_base_amount,
                filled_avg_price=fill_price, status="filled",
            )
            self._orders[order_id] = order

            self._save_balances()
        return {
            "id": order_id, "status": "filled", "symbol": symbol, "side": "buy",
            "amount": filled_base_amount, "price": fill_price,
            "cost": filled_base_amount * fill_price,
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
                with self._lock:
                    order_id = self._generate_order_id()
                    order = PaperOrder(
                        order_id=order_id, symbol=symbol, side="sell",
                        order_type="limit", amount=amount, limit_price=limit_price,
                        time_in_force=time_in_force, status="open",
                    )
                    self._orders[order_id] = order
                    self._save_orders()
                return self._make_order_dict(order)
            # Fill at the best available price (current price if higher than limit)
            fill_price = max(price, limit_price)
        else:
            fill_price = price

        # Apply dynamic slippage to market order fills
        slippage_pct = self._get_dynamic_slippage(symbol, fill_price)
        fill_price = fill_price * (1 - slippage_pct)

        max_vol = self._get_max_fillable_volume(symbol)
        
        # Apply market impact
        impact_pct = self._compute_market_impact_pct(amount, max_vol)
        fill_price = fill_price * (1 - impact_pct)

        # Check for volume-based partial fill
        if max_vol is not None and amount > max_vol:
            logger.info(f"Partial fill for {symbol}: requested {amount}, capped to {max_vol}")
            amount = max_vol

        with self._lock:
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

            filled_amount = amount

            costs = calculate_transaction_costs("SELL", fill_price, filled_amount, symbol=symbol)
            net_quote = costs["net_value"]
            fee_cost = costs["total_costs"]
            fee_currency = quote

            self._balances[base] = base_balance - filled_amount
            self._balances[quote] = self._balances.get(quote, 0.0) + net_quote
            self._balances_dirty = True

            order_id = self._generate_order_id()
            order = PaperOrder(
                order_id=order_id, symbol=symbol, side="sell", order_type="market",
                amount=filled_amount, price=fill_price, filled_qty=filled_amount,
                filled_avg_price=fill_price, status="filled",
            )
            self._orders[order_id] = order

            self._save_balances()
        return {
            "id": order_id, "status": "filled", "symbol": symbol, "side": "sell",
            "amount": filled_amount, "price": fill_price,
            "cost": filled_amount * fill_price,
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
        with self._lock:
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
        with self._lock:
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
        with self._lock:
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
        with self._lock:
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
        with self._lock:
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
        with self._lock:
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
        with self._lock:
            order = self._orders.get(order_id)
            if order is None:
                return None
            if order.status != "open":
                return order
            symbol = order.symbol

        price = self._get_current_price(symbol)
        if price is None:
            return order

        base = symbol.split("/")[0] if "/" in symbol else symbol
        quote = symbol.split("/")[1] if "/" in symbol else self.base_currency

        with self._lock:
            # Re-check status in case it changed while fetching the price
            if order.status != "open":
                return order

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
                        self._fill_order(order, order.stop_price, base, quote)
                elif order.side == "sell" and order.stop_price is not None:
                    if price <= order.stop_price:
                        self._fill_order(order, order.stop_price, base, quote)

            # Stop-limit orders
            elif order.order_type == "stop_limit":
                if order.side == "buy" and order.stop_price is not None:
                    if price >= order.stop_price:
                        fill = order.limit_price if order.limit_price else order.stop_price
                        self._fill_order(order, fill, base, quote)
                elif order.side == "sell" and order.stop_price is not None:
                    if price <= order.stop_price:
                        fill = order.limit_price if order.limit_price else order.stop_price
                        self._fill_order(order, fill, base, quote)

            # Limit orders
            elif order.order_type == "limit":
                if order.side == "buy" and order.limit_price is not None:
                    if price <= order.limit_price:
                        # Fill at the best available price (current price if lower than limit)
                        fill_price = min(price, order.limit_price)
                        self._fill_order(order, fill_price, base, quote)
                elif order.side == "sell" and order.limit_price is not None:
                    if price >= order.limit_price:
                        # Fill at the best available price (current price if higher than limit)
                        fill_price = max(price, order.limit_price)
                        self._fill_order(order, fill_price, base, quote)

            if order.status == "filled":
                self._save_orders()

        return order

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock:
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
        with self._lock:
            order = self._orders.get(order_id)
            if order is None or order.status != "open":
                return False
            order.status = "canceled"
            self._save_orders()
        return True

    def get_trade_history(self) -> List[Dict[str, Any]]:
        return []
