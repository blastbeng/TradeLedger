"""Buy order execution component for the TradingEngine.

Handles BUY signal execution, position sizing, stop-loss/take-profit computation,
and position creation after fills.
Extracted from OrderExecutor to reduce class size and improve maintainability.
"""
import asyncio
import logging
import time
from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple

from src.config.settings import settings
from src.database import insert_trade
from src.strategies.base import Signal
from src.utils.symbol_utils import is_btp_isin
from src.trading.components.order_executor import OrderExecutor

logger = logging.getLogger(__name__)


class BuyExecutor:
    """Handles BUY order execution for the TradingEngine."""

    def __init__(self, engine, event_bus):
        self.engine = engine
        self.shared_state = engine.shared_state
        self.event_bus = event_bus
        self._exit_order_manager = None
        self.event_bus.subscribe("execute_buy", self.execute_buy)

    async def execute_buy(
        self,
        symbol: str,
        display_symbol: str,
        signal: Signal,
        timeframe: Optional[str],
        exit_reason: Optional[str],
        atr: Optional[float],
        balance: Dict[str, float],
    ) -> None:
        """Execute a BUY signal."""
        engine = self.engine
        parts = symbol.split("/")
        if len(parts) != 2:
            logger.error(f"Invalid symbol format: {symbol}")
            return
        base, quote = parts
        _exec_is_btp = is_btp_isin(symbol)

        # Safety: never buy when trading is paused
        paused = await asyncio.to_thread(engine.redis.get, "trading:paused")
        if paused:
            logger.info(f"Ignoring BUY {symbol}: trading is paused (safety check).")
            return
        # Hard cap on total open positions
        if symbol not in self.shared_state.positions and len(self.shared_state.positions) >= settings.MAX_OPEN_POSITIONS:
            logger.info(f"Skipping BUY {symbol}: maximum open positions limit ({settings.MAX_OPEN_POSITIONS}) reached.")
            if engine.notifier:
                await engine.notifier.send_notification(
                    f"⚠️ Skipping BUY {display_symbol}: maximum open positions limit ({settings.MAX_OPEN_POSITIONS}) reached.",
                    summary={"symbol": symbol, "action": "SKIP", "reason": "Max open positions limit reached"}
                )
            return
        # Extract known parameters from the LLM's strategy_params (if any)
        params = signal.strategy_params or {}
        fill_timeout = params.get("order_fill_timeout_seconds", settings.ORDER_FILL_TIMEOUT_SECONDS)

        # Fetch current price early for position sizing and stop calculations
        base = symbol.split("/")[0]
        quotes = await engine._market_data_manager._get_quotes_async([base], timeout=45.0)
        ticker = quotes.get(base)
        current_price = ticker['last'] if ticker else None
        if current_price is None or current_price <= 0:
            logger.warning(f"Cannot execute BUY for {symbol}: no valid current price.")
            return

        # --- Stale quote guard: skip BUY if the price is too old ---
        tf = timeframe or (self.shared_state.positions.get(symbol, {}).get("timeframe") if symbol in self.shared_state.positions else None)
        if tf and await engine._is_quote_too_stale(ticker, tf):
            age_seconds = (time.time() * 1000 - ticker.get("last_update", 0)) / 1000
            logger.warning(
                f"Skipping BUY {symbol}: quote is {age_seconds:.0f}s old "
                f"(threshold scaled for timeframe {tf}). "
                f"Stale prices lead to incorrect position sizing and stop-loss calculations."
            )
            if engine.notifier:
                await engine.notifier.send_notification(
                    f"⚠️ Skipping BUY {display_symbol}: quote data is {age_seconds / 60:.0f} min old. "
                    f"Waiting for fresher data.",
                    summary={
                        "symbol": symbol,
                        "action": "SKIP",
                        "reason": "Stale quote data",
                        "age_seconds": round(age_seconds, 1),
                    }
                )
            return

        # --- Compute stop-loss and take-profit parameters from LLM params ---
        _sl_tp_result = await self.compute_sl_tp_params(
            symbol=symbol,
            display_symbol=display_symbol,
            params=params,
            atr=atr,
            current_price=current_price,
            is_btp=_exec_is_btp,
        )
        if _sl_tp_result is None:
            return
        sl_pct, tp_pct, trailing_stop, trailing_stop_distance_pct = _sl_tp_result

        quote_balance = balance.get(quote, 0.0)
        position_fraction = params["position_size_fraction"]

        # Desired amount based on fraction of total available quote balance
        desired_amount = quote_balance * position_fraction

        # Apply confidence-based position sizing (LLM-decided weight)
        confidence_sizing_weight = params.get("confidence_sizing_weight", 0.0)
        if confidence_sizing_weight is not None:
            try:
                confidence_sizing_weight = float(confidence_sizing_weight)
            except (TypeError, ValueError):
                confidence_sizing_weight = 0.0
        if confidence_sizing_weight > 0 and signal.confidence < 1.0:
            confidence_multiplier = 1.0 - confidence_sizing_weight * (1.0 - signal.confidence)
            confidence_multiplier = max(confidence_multiplier, 0.01)  # floor to prevent zeroing
            desired_amount *= confidence_multiplier
            logger.info(
                f"Confidence sizing applied: weight={confidence_sizing_weight}, "
                f"confidence={signal.confidence:.2f}, multiplier={confidence_multiplier:.4f}, "
                f"adjusted amount={desired_amount:.2f}"
            )

        # --- Consolidated position sizing: single hard ceiling from all caps ---
        _sizing_result = await self.compute_position_size(
            symbol=symbol,
            display_symbol=display_symbol,
            quote_balance=quote_balance,
            desired_amount=desired_amount,
            params=params,
            sl_pct=sl_pct,
        )
        if _sizing_result is None:
            return
        amount, desired_amount = _sizing_result

        # --- Minimum profit check and exchange minimum order size adjustment ---
        amount = await self.check_min_profit_and_order_size(
            symbol=symbol,
            display_symbol=display_symbol,
            quote=quote,
            params=params,
            amount=amount,
            desired_amount=desired_amount,
            quote_balance=quote_balance,
            tp_pct=tp_pct,
            current_price=current_price,
        )
        if amount is None:
            return

        # --- Determine limit price for BUY ---
        _limit_result = await self.compute_buy_limit_price(
            symbol=symbol,
            display_symbol=display_symbol,
            params=params,
            ticker=ticker,
            atr=atr,
        )
        if _limit_result is None:
            return
        limit_price, time_in_force, need_limit = _limit_result

        # --- Determine order type ---
        order_type = signal.order_type
        if order_type not in ("market", "limit", "stop", "stop_limit", "trailing_stop"):
            # Fallback to existing behaviour: limit if limit_price provided, else market
            if limit_price is not None:
                order_type = "limit"
            else:
                order_type = "market"

        # --- Reserve cycle budget before placing order to prevent race condition ---
        async with self.shared_state._cycle_spent_lock:
            available = max(0.0, quote_balance - self.shared_state._cycle_spent)
            if amount > available:
                logger.info(
                    f"Skipping BUY {symbol}: cycle budget exhausted "
                    f"(needed {amount:.2f}, available {available:.2f}) due to concurrent order"
                )
                if engine.notifier:
                    await engine.notifier.send_notification(
                        f"⚠️ Skipping BUY {display_symbol}: cycle budget exhausted by concurrent order",
                        summary={"symbol": symbol, "action": "SKIP", "reason": "Cycle budget exhausted (concurrent order)"}
                    )
                return
            self.shared_state._cycle_spent += amount

        try:
            if order_type == "market":
                order = await asyncio.to_thread(
                    engine.trader.create_market_buy_order, symbol, amount, fill_timeout,
                    limit_price=None, time_in_force='day'
                )
            elif order_type == "limit":
                order = await asyncio.to_thread(
                    engine.trader.create_market_buy_order, symbol, amount, fill_timeout,
                    limit_price=limit_price, time_in_force=time_in_force
                )
            elif order_type == "stop":
                stop_price = signal.stop_price
                if stop_price is None or stop_price <= 0:
                    raise ValueError("Missing or invalid stop_price for stop order")
                order = await asyncio.to_thread(
                    engine.trader.create_stop_buy_order, symbol, amount, stop_price,
                    time_in_force=time_in_force, timeout=fill_timeout
                )
            elif order_type == "stop_limit":
                stop_price = signal.stop_price
                limit_price_sl = limit_price  # LLM-provided limit_price for stop_limit
                if stop_price is None or stop_price <= 0:
                    raise ValueError("Missing or invalid stop_price for stop_limit order")
                if limit_price_sl is None or limit_price_sl <= 0:
                    raise ValueError("Missing or invalid limit_price for stop_limit order")
                order = await asyncio.to_thread(
                    engine.trader.create_stop_limit_buy_order, symbol, amount,
                    stop_price, limit_price_sl,
                    time_in_force=time_in_force, timeout=fill_timeout
                )
            elif order_type == "trailing_stop":
                trail_offset = signal.trail_offset
                if trail_offset is None or trail_offset <= 0:
                    raise ValueError("Missing or invalid trail_offset for trailing_stop order")
                order = await asyncio.to_thread(
                    engine.trader.create_trailing_stop_buy_order, symbol, amount,
                    trail_offset,
                    time_in_force=time_in_force, timeout=fill_timeout
                )
            else:
                raise ValueError(f"Unknown order_type: {order_type}")
            if order.get('status') == 'open':
                price_str = f" at {limit_price}" if limit_price is not None else ""
                logger.info(f"BUY {order_type} order for {symbol} queued{price_str}")
                queued_entry = {
                    'symbol': symbol,
                    'side': 'buy',
                    'amount': amount,
                    'original_amount': amount,
                    'limit_price': limit_price if order_type in ("limit", "stop_limit") else None,
                    'stop_price': signal.stop_price if order_type in ("stop", "stop_limit") else None,
                    'trail_offset': signal.trail_offset if order_type == "trailing_stop" else None,
                    'order_type': order_type,
                    'time_in_force': time_in_force,
                    'signal': asdict(signal),
                    'timeframe': timeframe,
                    'atr': atr,
                    'order_id': order['id'],
                    'queued_at': time.time(),
                    'filled_qty': 0,
                    'filled_cost': 0.0,
                }
                async with self.shared_state._queued_orders_lock:
                    self.shared_state.queued_orders.append(queued_entry)
                if engine.notifier:
                    await engine.notifier.send_notification(
                        f"⏳ BUY {order_type} order for {display_symbol} queued{price_str}",
                        summary={"symbol": symbol, "action": "QUEUE", "reason": "Order not yet filled"}
                    )
                return
            if order.get('status') == 'rejected':
                async with self.shared_state._cycle_spent_lock:
                    self.shared_state._cycle_spent = max(0.0, self.shared_state._cycle_spent - amount)
                logger.warning(f"BUY order rejected for {symbol}")
                if engine.notifier:
                    await engine.notifier.send_notification(
                        f"❌ BUY order rejected for {display_symbol}",
                        summary={"symbol": symbol, "action": "REJECT", "reason": "Order rejected by simulator"}
                    )
                return
            logger.info(f"BUY {symbol}: {order}")
            # Queue remaining partial market order for polling
            if order.get("remaining_order_id"):
                queued_entry = {
                    'symbol': symbol,
                    'side': 'buy',
                    'amount': amount - order['cost'],
                    'original_amount': amount - order['cost'],
                    'limit_price': order['price'],
                    'stop_price': None,
                    'trail_offset': None,
                    'order_type': 'limit',
                    'time_in_force': 'day',
                    'signal': asdict(signal),
                    'timeframe': timeframe,
                    'atr': atr,
                    'order_id': order['remaining_order_id'],
                    'queued_at': time.time(),
                    'filled_qty': 0,
                    'filled_cost': 0.0,
                }
                async with self.shared_state._queued_orders_lock:
                    self.shared_state.queued_orders.append(queued_entry)
            await self.update_or_create_buy_position(
                symbol=symbol,
                order=order,
                signal=signal,
                params=params,
                quote=quote,
                base=base,
                sl_pct=sl_pct,
                tp_pct=tp_pct,
                trailing_stop=trailing_stop,
                trailing_stop_distance_pct=trailing_stop_distance_pct,
                order_type=order_type,
                timeframe=timeframe,
            )
            await self.record_buy_fill_and_notify(
                symbol=symbol,
                display_symbol=display_symbol,
                order=order,
                signal=signal,
                timeframe=timeframe,
                atr=atr,
            )
        except (RuntimeError, ValueError, ConnectionError, KeyError) as e:
            async with self.shared_state._cycle_spent_lock:
                self.shared_state._cycle_spent = max(0.0, self.shared_state._cycle_spent - amount)
            logger.error(f"Buy order failed for {symbol}: {e}")
            if engine.notifier:
                await engine.notifier.send_notification(
                    f"❌ Buy order failed for {display_symbol}: {e}",
                    summary={
                        "symbol": symbol,
                        "action": "ERROR",
                        "reason": f"Buy order failed: {e}"[:200],
                    }
                )

    async def compute_position_size(
        self,
        symbol: str,
        display_symbol: str,
        quote_balance: float,
        desired_amount: float,
        params: Dict[str, Any],
        sl_pct: float,
    ) -> Optional[Tuple[float, float]]:
        """Compute the final position size applying all risk caps.

        Applies global risk multiplier, per-symbol multiplier, and a single
        hard ceiling from all risk caps (max_risk_per_trade, max_portfolio_risk,
        max_portfolio_exposure, max_portfolio_stop_risk, and remaining cycle budget).

        Returns (amount, desired_amount, available) or None if the position
        should be skipped (amount <= 0).
        """
        engine = self.engine

        # --- Consolidated position sizing: single hard ceiling from all caps ---
        pos_tickers = await engine._get_cached_position_tickers()

        # Compute current portfolio state once
        total_value = quote_balance
        total_open_exposure = 0.0
        total_open_stop_risk = 0.0
        for sym, pos in self.shared_state.positions.items():
            try:
                t = pos_tickers.get(sym)
                price = t['last'] if t and t.get('last') else 0.0
                pos_value = pos['amount'] * price
                total_open_exposure += pos_value
                total_value += pos_value
                stop_loss = pos.get('stop_loss')
                if stop_loss is not None and price > 0:
                    loss_if_stop = pos_value * (price - stop_loss) / price
                    total_open_stop_risk += max(0, loss_if_stop)
            except (KeyError, TypeError, ValueError):
                pass

        # Apply global risk multiplier to desired amount (scales all positions)
        global_mult = await engine._get_global_risk_multiplier()
        if global_mult is not None and 0.0 < global_mult <= 1.0:
            desired_amount *= global_mult

        # Apply per-symbol position size multiplier to desired amount
        per_symbol_mult = params.get("position_size_multiplier")
        if per_symbol_mult is not None:
            try:
                per_symbol_mult = float(per_symbol_mult)
                if 0.0 < per_symbol_mult <= 1.0:
                    desired_amount *= per_symbol_mult
            except (ValueError, TypeError):
                pass

        # --- Compute individual caps ---
        caps = []

        # Cap 1: max_risk_per_trade_pct (per-trade risk from LLM strategy params)
        max_risk_pct = params.get("max_risk_per_trade_pct")
        if max_risk_pct is not None and max_risk_pct > 0 and sl_pct > 0:
            caps.append(((total_value * max_risk_pct) / sl_pct, f"max_risk_per_trade={max_risk_pct:.2%}"))

        # Cap 2: max_portfolio_risk_pct (portfolio risk from LLM strategy params)
        max_portfolio_risk_pct = params.get("max_portfolio_risk_pct")
        if max_portfolio_risk_pct is not None and max_portfolio_risk_pct > 0 and sl_pct > 0:
            available_risk_budget = max(0.0, (total_value * max_portfolio_risk_pct) - total_open_stop_risk)
            caps.append((available_risk_budget / sl_pct, f"max_portfolio_risk={max_portfolio_risk_pct:.2%}"))

        # Cap 3: max_portfolio_exposure_pct (global LLM setting from stock selection)
        max_port_exp_raw = await engine.config_service.get_config("max_portfolio_exposure_pct")
        max_port_exp = float(max_port_exp_raw) if max_port_exp_raw else None
        if max_port_exp is not None and max_port_exp > 0 and total_value > 0:
            available_exposure = max(0.0, (max_port_exp * total_value) - total_open_exposure)
            caps.append((available_exposure, f"max_exposure={max_port_exp:.2%}"))

        # Cap 4: max_portfolio_stop_risk_pct (global LLM setting from stock selection)
        max_port_risk_raw = await engine.config_service.get_config("max_portfolio_stop_risk_pct")
        max_port_risk = float(max_port_risk_raw) if max_port_risk_raw else None
        if max_port_risk is not None and max_port_risk > 0 and sl_pct > 0 and total_value > 0:
            available_stop_risk_budget = max(0.0, (total_value * max_port_risk) - total_open_stop_risk)
            caps.append((available_stop_risk_budget / sl_pct, f"max_stop_risk={max_port_risk:.2%}"))

        # Cap 5: remaining cycle budget
        async with self.shared_state._cycle_spent_lock:
            available = max(0.0, quote_balance - self.shared_state._cycle_spent)
        caps.append((available, "cycle_budget"))

        # --- Determine the binding cap ---
        hard_max = float('inf')
        binding_reason = None
        for cap_value, cap_reason in caps:
            if cap_value < hard_max:
                hard_max = cap_value
                binding_reason = cap_reason

        if hard_max == float('inf'):
            hard_max = 0.0

        # Final amount: min of LLM's desired amount and the single hard ceiling
        amount = min(desired_amount, hard_max)

        if amount <= 0:
            logger.info(f"Skipping BUY {symbol}: position size reduced to 0 by portfolio constraints")
            if engine.notifier:
                await engine.notifier.send_notification(
                    f"⚠️ Skipping BUY {display_symbol}: portfolio constraints leave no room for new position",
                    summary={
                        "symbol": symbol,
                        "action": "SKIP",
                        "reason": "Portfolio constraints exhausted",
                        "desired_amount": desired_amount,
                        "hard_max": 0.0,
                    }
                )
            return None

        if amount < desired_amount:
            # Single consolidated notification about which cap was binding
            cap_reasons = []
            if binding_reason:
                cap_reasons.append(binding_reason)
            if global_mult is not None and global_mult < 1.0:
                cap_reasons.append(f"global_risk_mult={global_mult:.2f}")
            if per_symbol_mult is not None and per_symbol_mult < 1.0:
                cap_reasons.append(f"position_size_mult={per_symbol_mult:.2f}")
            reason_str = ", ".join(cap_reasons) if cap_reasons else "portfolio constraints"
            logger.info(
                f"Position size capped for {symbol}: {desired_amount:.2f} -> {amount:.2f} "
                f"({reason_str})"
            )
            if engine.notifier:
                await engine.notifier.send_notification(
                    f"⚠️ {display_symbol}: position capped {desired_amount:.2f} → {amount:.2f} ({reason_str})",
                    summary={
                        "symbol": symbol,
                        "action": "INFO",
                        "reason": f"Position size capped: {reason_str}",
                        "desired_amount": desired_amount,
                        "capped_amount": amount,
                    }
                )

        return amount, desired_amount

    async def compute_sl_tp_params(
        self,
        symbol: str,
        display_symbol: str,
        params: Dict[str, Any],
        atr: Optional[float],
        current_price: float,
        is_btp: bool,
    ) -> Optional[Tuple[float, float, bool, Optional[float]]]:
        """Compute stop-loss and take-profit percentages from LLM params and ATR.

        Returns (sl_pct, tp_pct, trailing_stop, trailing_stop_distance_pct)
        or None if the position should be skipped (missing required params).
        """
        engine = self.engine

        # Determine take-profit percentage based on method
        if "take_profit_atr_multiple" in params and atr is not None and atr > 0 and current_price > 0:
            tp_atr_mult = params["take_profit_atr_multiple"]
            tp_pct = (tp_atr_mult * atr) / current_price
            logger.info(f"ATR-based take-profit: ATR={atr}, multiplier={tp_atr_mult}, take_profit_pct={tp_pct:.4%}")
        else:
            if "take_profit_atr_multiple" in params:
                logger.warning(f"ATR unavailable for {symbol}, falling back to fixed take_profit_pct from LLM params.")
            tp_pct = params.get("take_profit_pct")
            if tp_pct is None or tp_pct <= 0:
                logger.warning(f"Cannot execute BUY for {symbol}: take_profit_pct missing/invalid and ATR unavailable.")
                if engine.notifier:
                    await engine.notifier.send_notification(
                        f"⚠️ Skipping BUY {display_symbol}: missing take_profit_pct and ATR unavailable.",
                        summary={"symbol": symbol, "action": "SKIP", "reason": "Missing take_profit_pct and ATR unavailable"}
                    )
                return None

        # --- BTP take-profit cap: enforce smaller targets for bonds ---
        if is_btp and tp_pct is not None and tp_pct > 0:
            if tp_pct > settings.BTP_MAX_TAKE_PROFIT_PCT:
                logger.info(
                    f"BTP take-profit capped for {symbol}: {tp_pct:.4%} -> "
                    f"{settings.BTP_MAX_TAKE_PROFIT_PCT:.4%}"
                )
                tp_pct = settings.BTP_MAX_TAKE_PROFIT_PCT

        trailing_stop = params["trailing_stop"]
        # Force trailing_stop off for BTPs — not supported by Intesa Sanpaolo Investo
        if is_btp and trailing_stop:
            logger.warning(
                f"LLM set trailing_stop=true for BTP {symbol}, but trailing stops are not supported "
                f"for BTPs on Intesa Sanpaolo Investo. Forcing trailing_stop=false."
            )
            trailing_stop = False
        trailing_stop_distance_pct = params.get("trailing_stop_distance_pct")

        # Determine stop-loss percentage based on method
        stop_method = params.get("stop_loss_method", "fixed")
        if stop_method == "atr_multiple" and atr is not None and atr > 0:
            atr_mult = params["stop_loss_atr_multiple"]
            sl_pct = (atr_mult * atr) / current_price
            logger.info(f"ATR-based stop: ATR={atr}, multiplier={atr_mult}, stop_loss_pct={sl_pct:.4%}")
        else:
            if stop_method == "atr_multiple":
                logger.warning(f"ATR unavailable for {symbol}, falling back to fixed stop_loss_pct from LLM params.")
            sl_pct = params.get("stop_loss_pct")
            if sl_pct is None:
                logger.warning(f"Cannot execute BUY for {symbol}: stop_loss_pct missing and ATR method not applicable/available.")
                if engine.notifier:
                    await engine.notifier.send_notification(
                        f"⚠️ Skipping BUY {display_symbol}: missing stop_loss_pct and ATR unavailable.",
                        summary={"symbol": symbol, "action": "SKIP", "reason": "Missing stop_loss_pct and ATR unavailable"}
                    )
                return None

        return sl_pct, tp_pct, trailing_stop, trailing_stop_distance_pct

    async def check_min_profit_and_order_size(
        self,
        symbol: str,
        display_symbol: str,
        quote: str,
        params: Dict[str, Any],
        amount: float,
        desired_amount: float,
        quote_balance: float,
        tp_pct: float,
        current_price: float,
    ) -> Optional[float]:
        """Check minimum profit and adjust order size to meet exchange minimums.

        Returns the final amount (possibly adjusted upward), or None if the
        order should be skipped.
        """
        engine = self.engine

        # --- Minimum absolute profit check (LLM‑defined) ---
        if settings.ENFORCE_MIN_PROFIT_PER_TRADE:
            min_profit = params.get("min_profit_per_trade")
            if min_profit is not None and min_profit > 0:
                expected_gross_profit = amount * tp_pct
                if expected_gross_profit < min_profit:
                    logger.info(
                        f"Skipping BUY {symbol}: expected gross profit {expected_gross_profit:.4f} {quote} "
                        f"below LLM minimum {min_profit:.4f}"
                    )
                    if engine.notifier:
                        await engine.notifier.send_notification(
                            f"⚠️ Skipping BUY {display_symbol}: profit too small ({expected_gross_profit:.4f} {quote})",
                            summary={
                                "symbol": symbol,
                                "action": "SKIP",
                                "reason": "Expected profit below minimum",
                                "expected_profit": expected_gross_profit,
                                "min_profit": min_profit,
                            }
                        )
                    return None

        # Check minimum order size and adjust upward if needed
        try:
            price = current_price
            # Fetch minimum order size from asset info
            try:
                asset = await engine._market_data_manager.get_asset_info(symbol)
                min_amount_limit = float(asset.min_order_size) if asset.min_order_size else None
                if not asset.fractionable and (min_amount_limit is None or min_amount_limit < 1.0):
                    min_amount_limit = 1.0
            except (AttributeError, TypeError, ValueError):
                min_amount_limit = None
            # Compute min cost from min amount and current price
            if min_amount_limit is not None and price:
                min_cost_limit = min_amount_limit * price
            else:
                min_cost_limit = None

            # Determine the required minimum quote amount
            required_quote = amount
            if min_amount_limit is not None:
                min_base = float(min_amount_limit)
                required_quote = max(required_quote, min_base * price)
            if min_cost_limit is not None:
                required_quote = max(required_quote, float(min_cost_limit))

            if required_quote > amount:
                # If the required minimum exceeds the risk-limited desired_amount, skip
                if required_quote > desired_amount:
                    logger.info(
                        f"Skipping BUY {symbol}: exchange minimum {required_quote:.2f} "
                        f"exceeds risk-limited amount {desired_amount:.2f}"
                    )
                    if engine.notifier:
                        await engine.notifier.send_notification(
                            f"⚠️ Skipping BUY {display_symbol}: exchange minimum exceeds risk limit",
                            summary={
                                "symbol": symbol,
                                "action": "SKIP",
                                "reason": "Exchange minimum exceeds risk limit",
                                "required_quote": required_quote,
                                "desired_amount": desired_amount,
                            }
                        )
                    return None
                # Adjust amount upward to meet the minimum
                old_amount = amount
                amount = required_quote
                # Check if the adjusted amount exceeds remaining cycle budget
                async with self.shared_state._cycle_spent_lock:
                    available = max(0.0, quote_balance - self.shared_state._cycle_spent)
                if amount > available:
                    logger.info(
                        f"BUY amount adjusted from {old_amount:.2f} to {amount:.2f} {quote} "
                        f"to meet minimum, but exceeds remaining cycle budget ({available:.2f}). Skipping."
                    )
                    if engine.notifier:
                        await engine.notifier.send_notification(
                            f"⚠️ BUY skipped for {display_symbol}: amount adjusted to {amount:.2f} but insufficient remaining budget",
                            summary={
                                "symbol": symbol,
                                "action": "SKIP",
                                "reason": "Adjusted amount exceeds remaining budget",
                                "adjusted_amount": amount,
                            }
                        )
                    return None
                logger.info(
                    f"BUY amount adjusted from {old_amount:.2f} to {amount:.2f} {quote} "
                    f"to meet exchange minimum"
                )
                if engine.notifier:
                    await engine.notifier.send_notification(
                        f"ℹ️ {display_symbol}: buy amount adjusted to {amount:.2f} {quote} to meet minimum",
                        summary={
                            "symbol": symbol,
                            "action": "INFO",
                            "reason": "Buy amount adjusted to meet minimum",
                            "adjusted_amount": amount,
                        }
                    )
        except (AttributeError, TypeError, ValueError, RuntimeError) as e:
            logger.warning(f"Could not verify/adjust min order size for {symbol}: {e}")

        return amount

    async def compute_buy_limit_price(
        self,
        symbol: str,
        display_symbol: str,
        params: Dict[str, Any],
        ticker: Dict[str, Any],
        atr: Optional[float],
    ) -> Optional[Tuple[Optional[float], str, bool]]:
        """Determine the limit price, time-in-force, and order path for a BUY.

        Returns (limit_price, time_in_force, need_limit) or None if the
        order should be skipped (invalid limit price).
        """
        engine = self.engine
        need_limit = not await engine._is_market_open()
        limit_price = None
        time_in_force = "day"
        # If LLM provided a limit_price, use it even during regular hours
        llm_limit_price = params.get("limit_price")
        if llm_limit_price is not None and llm_limit_price > 0:
            limit_price = llm_limit_price
            time_in_force = params.get("time_in_force", "day")
            need_limit = True  # force limit order path
            # Validate that the limit price is within a reasonable distance from the market
            # Read LLM-controlled limit price max distance (fallback to static setting)
            max_distance = settings.LIMIT_PRICE_MAX_DISTANCE_PCT
            try:
                raw = await engine.config_service.get_config("limit_price_max_distance_pct")
                if raw:
                    max_distance = float(raw)
            except (ValueError, TypeError, ConnectionError, TimeoutError, OSError):
                pass
            if ticker and ticker.get('ask') and max_distance > 0:
                ask = ticker['ask']
                if limit_price < ask * (1 - max_distance):
                    logger.warning(
                        f"LLM limit_price {limit_price} for {symbol} is >{max_distance*100:.0f}% below ask {ask}. "
                        f"Rejecting BUY to avoid indefinite queuing."
                    )
                    if engine.notifier:
                        await engine.notifier.send_notification(
                            f"⚠️ Skipping BUY {display_symbol}: limit price {limit_price} too far below ask {ask}.",
                            summary={"symbol": symbol, "action": "SKIP", "reason": "Limit price too far from market"}
                        )
                    return None
        elif need_limit:
            limit_price = OrderExecutor._default_limit_price(symbol, "BUY", ticker, atr=atr)
            time_in_force = params.get("time_in_force", "day")
            if limit_price is None:
                logger.error(f"Cannot place limit order for {symbol}: no limit price available.")
                return None

        if limit_price is not None:
            # Round to valid tick size ($0.01 for >=$1, $0.0001 for <$1)
            if limit_price >= 1.0:
                limit_price = round(limit_price, 2)
            else:
                limit_price = round(limit_price, 4)

        if limit_price is not None and limit_price <= 0:
            logger.error(f"Invalid limit_price {limit_price} for {symbol}, skipping.")
            if engine.notifier:
                await engine.notifier.send_notification(
                    f"❌ Invalid limit price for {display_symbol}, skipping.",
                    summary={"symbol": symbol, "action": "SKIP", "reason": "Invalid limit price"}
                )
            return None

        return limit_price, time_in_force, need_limit

    def _apply_buy_to_position(
        self,
        symbol: str,
        cost_basis: float,
        net_base: float,
        timestamp: int,
        sl_pct: Optional[float],
        tp_pct: Optional[float],
        trailing_stop: bool,
        trailing_stop_distance_pct: Optional[float],
        params: Dict[str, Any],
        signal_confidence: float,
        signal_reasoning: str,
        signal_strategy_type: Optional[str],
        indicator_config: Optional[Dict[str, Any]],
        order_type: str,
        timeframe: Optional[str],
    ) -> None:
        """Shared helper to update or create a position after a BUY fill."""
        positions = self.shared_state.positions
        if symbol in positions:
            old_cost_basis = positions[symbol].get("cost_basis", positions[symbol]["amount"] * positions[symbol]["price"])
            old_net_base = positions[symbol].get("net_base", positions[symbol]["amount"])
            new_cost_basis = old_cost_basis + cost_basis
            new_net_base = old_net_base + net_base
            new_price = new_cost_basis / new_net_base if new_net_base > 0 else 0.0
            positions[symbol]["amount"] = new_net_base
            positions[symbol]["price"] = new_price
            positions[symbol]["cost_basis"] = new_cost_basis
            positions[symbol]["net_base"] = new_net_base
            positions[symbol]["take_profit_atr_multiple"] = params.get("take_profit_atr_multiple")
            positions[symbol]["trailing_stop"] = trailing_stop
            positions[symbol]["trailing_stop_distance_pct"] = trailing_stop_distance_pct
            positions[symbol]["trailing_stop_atr_multiple"] = params.get("trailing_stop_atr_multiple")
            positions[symbol]["max_hold_time_seconds"] = params.get("max_hold_time_seconds")
            positions[symbol]["trailing_stop_activation_pct"] = params.get("trailing_stop_activation_pct")
            positions[symbol]["trailing_take_profit"] = params.get("trailing_take_profit", False)
            positions[symbol]["trailing_take_profit_distance_pct"] = params.get("trailing_take_profit_distance_pct")
            positions[symbol]["breakeven_activation_pct"] = params.get("breakeven_activation_pct")
            partial_levels = params.get("partial_take_profit_levels")
            if partial_levels:
                positions[symbol]["partial_take_profit_levels"] = partial_levels
                positions[symbol]["partial_tp_levels_triggered"] = []
                positions[symbol]["partial_tp_depth_wait_start"] = {}
                positions[symbol]["partial_take_profit_pct"] = None
                positions[symbol]["partial_take_profit_fraction"] = None
                positions[symbol]["partial_tp_triggered"] = None
            else:
                positions[symbol]["partial_take_profit_pct"] = params.get("partial_take_profit_pct")
                positions[symbol]["partial_take_profit_fraction"] = params.get("partial_take_profit_fraction")
                positions[symbol]["partial_tp_triggered"] = False
            positions[symbol]["cooldown_after_loss_seconds"] = params.get("cooldown_after_loss_seconds", 0)
            positions[symbol]["news_sentiment_exit_threshold"] = params.get("news_sentiment_exit_threshold")
            positions[symbol]["max_unrealized_loss_pct"] = params.get("max_unrealized_loss_pct")
            positions[symbol]["timeframe"] = timeframe
            positions[symbol]["indicator_config"] = indicator_config
            positions[symbol]["entry_order_type"] = order_type
            positions[symbol]["buy_confidence"] = signal_confidence
            positions[symbol]["buy_reasoning"] = (signal_reasoning or "")[:200]
        else:
            entry_price = cost_basis / net_base if net_base > 0 else 0.0
            positions[symbol] = {
                "symbol": symbol,
                "side": "buy",
                "amount": net_base,
                "price": entry_price,
                "timestamp": timestamp,
                "stop_loss": entry_price * (1 - sl_pct) if sl_pct else None,
                "take_profit": entry_price * (1 + tp_pct) if tp_pct else None,
                "take_profit_atr_multiple": params.get("take_profit_atr_multiple"),
                "cost_basis": cost_basis,
                "net_base": net_base,
                "buy_confidence": signal_confidence,
                "buy_reasoning": (signal_reasoning or "")[:200],
                "trailing_stop": trailing_stop,
                "trailing_stop_distance_pct": trailing_stop_distance_pct,
                "trailing_stop_atr_multiple": params.get("trailing_stop_atr_multiple"),
                "max_hold_time_seconds": params.get("max_hold_time_seconds"),
                "trailing_stop_activation_pct": params.get("trailing_stop_activation_pct"),
                "trailing_take_profit": params.get("trailing_take_profit", False),
                "trailing_take_profit_distance_pct": params.get("trailing_take_profit_distance_pct"),
                "breakeven_activation_pct": params.get("breakeven_activation_pct"),
                "partial_take_profit_levels": params.get("partial_take_profit_levels"),
                "partial_tp_levels_triggered": [],
                "partial_tp_depth_wait_start": {},
                "original_amount": net_base,
                "partial_take_profit_pct": params.get("partial_take_profit_pct") if not params.get("partial_take_profit_levels") else None,
                "partial_take_profit_fraction": params.get("partial_take_profit_fraction") if not params.get("partial_take_profit_levels") else None,
                "partial_tp_triggered": False if not params.get("partial_take_profit_levels") else None,
                "cooldown_after_loss_seconds": params.get("cooldown_after_loss_seconds", 0),
                "news_sentiment_exit_threshold": params.get("news_sentiment_exit_threshold"),
                "max_unrealized_loss_pct": params.get("max_unrealized_loss_pct"),
                "timeframe": timeframe,
                "indicator_config": indicator_config,
                "entry_order_type": order_type,
                "strategy_type": signal_strategy_type,
            }

        custom_interval = params.get("strategy_interval_seconds")
        if custom_interval is not None:
            self.shared_state._strategy_intervals[symbol] = custom_interval

    async def update_or_create_buy_position(
        self,
        symbol: str,
        order: Dict[str, Any],
        signal: Signal,
        params: Dict[str, Any],
        quote: str,
        base: str,
        sl_pct: float,
        tp_pct: float,
        trailing_stop: bool,
        trailing_stop_distance_pct: Optional[float],
        order_type: str,
        timeframe: Optional[str],
    ) -> None:
        """Update an existing position or create a new one after a filled BUY order."""
        engine = self.engine
        fee = order.get('fee', {})
        fee_cost = float(fee.get('cost', 0.0) or 0.0)
        fee_currency = fee.get('currency', '')

        cost_basis = order['cost'] + (fee_cost if fee_currency == quote else 0.0)
        net_base = order['amount'] - (fee_cost if fee_currency == base else 0.0)

        self._apply_buy_to_position(
            symbol=symbol,
            cost_basis=cost_basis,
            net_base=net_base,
            timestamp=order["timestamp"],
            sl_pct=sl_pct,
            tp_pct=tp_pct,
            trailing_stop=trailing_stop,
            trailing_stop_distance_pct=trailing_stop_distance_pct,
            params=params,
            signal_confidence=signal.confidence,
            signal_reasoning=signal.reasoning,
            signal_strategy_type=signal.strategy_type,
            indicator_config=signal.indicator_config,
            order_type=order_type,
            timeframe=timeframe,
        )

    async def record_buy_fill_and_notify(
        self,
        symbol: str,
        display_symbol: str,
        order: Dict[str, Any],
        signal: Signal,
        timeframe: Optional[str],
        atr: Optional[float],
    ) -> None:
        """Place exit orders, record the trade, and send BUY notification after a fill."""
        engine = self.engine
        # --- Place native exit orders (OCO) if LLM specified them ---
        current_entry = self.shared_state.positions[symbol]["price"]
        exit_prices = self._exit_order_manager.compute_exit_order_prices(
            entry_price=current_entry,
            signal=signal,
            atr=atr,
        )
        await self.event_bus.request("place_exit_orders", symbol, signal, exit_prices, timeframe)
        order["strategy_type"] = signal.strategy_type
        order["timeframe"] = timeframe
        order["buy_confidence"] = signal.confidence
        order["buy_reasoning"] = (signal.reasoning or "")[:200]
        if hasattr(signal, 'backtest_summary') and signal.backtest_summary:
            order["backtest_summary"] = signal.backtest_summary
        self.shared_state.append_trade(order, settings.MAX_TRADES_IN_MEMORY)
        self.shared_state._balance_cache = None  # force refresh on next fetch
        await asyncio.to_thread(insert_trade, order)
        await self.event_bus.publish("save_state", force=True)
        self.shared_state._portfolio_exposure_cache = None
        if engine.notifier:
            buy_msg = f"🟢 BUY {display_symbol}: {order['amount']:.6f} @ {order['price']:.4f}"
            buy_summary = {
                "symbol": symbol,
                "action": "BUY",
                "price": order["price"],
                "amount": order["amount"],
                "confidence": signal.confidence,
                "reason": signal.reasoning[:200],
                "strategy_type": signal.strategy_type,
            }
            if signal.model_type:
                buy_summary["model_type"] = signal.model_type
            if signal.llm_provider:
                buy_summary["llm_provider"] = signal.llm_provider
            if signal.llm_model:
                buy_summary["llm_model"] = signal.llm_model
            await engine.notifier.send_notification(
                buy_msg,
                summary=buy_summary,
            )
