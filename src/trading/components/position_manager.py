"""Position management component for the TradingEngine.

Handles position-related operations: cost basis computation and portfolio
exposure calculation.
Extracted from TradingEngine to reduce class size and improve maintainability.
"""
import asyncio
import logging
import time
from typing import Any, Dict, Optional

from src.config.settings import settings
from src.utils.symbol_utils import is_btp_isin

logger = logging.getLogger(__name__)


class PositionManager:
    """Handles position management operations for the TradingEngine."""

    def __init__(self, engine):
        self.engine = engine

    def ensure_cost_basis(self):
        """If positions lack cost_basis, compute it from amount and price (backward compat)."""
        for sym, pos in self.engine.positions.items():
            if 'cost_basis' not in pos or 'net_base' not in pos:
                # Assume no fees for old positions; cost_basis = amount * price
                pos['cost_basis'] = pos['amount'] * pos['price']
                pos['net_base'] = pos['amount']

    async def compute_portfolio_exposure_summary(self, base_balance: float) -> Dict[str, float]:
        """Compute portfolio exposure, stop-loss risk, and available capital for the prompt."""
        engine = self.engine
        now = time.time()
        if (
            engine._portfolio_exposure_cache is not None
            and (now - engine._portfolio_exposure_cache_time) < 30
        ):
            # Return cached ticker-dependent values, but recompute available capital
            # from the current cycle_spent (which changes during the cycle).
            # Acquire the lock before copying the cache so the cache read and
            # _cycle_spent read are atomic — prevents a race where another
            # coroutine modifies _cycle_spent between the copy and the lock.
            async with engine._cycle_spent_lock:
                result = dict(engine._portfolio_exposure_cache)
                result["portfolio_available_capital"] = max(0.0, base_balance - engine._cycle_spent)
            return result

        portfolio_total_value = base_balance
        portfolio_exposure = 0.0
        portfolio_stop_risk = 0.0
        pos_tickers = await engine._get_all_position_tickers()
        for sym, pos in engine.positions.items():
            try:
                t = pos_tickers.get(sym)
                price = t['last'] if t and t.get('last') else 0.0
                pos_value = pos['amount'] * price
                portfolio_exposure += pos_value
                portfolio_total_value += pos_value
                stop_loss = pos.get('stop_loss')
                if stop_loss is not None and price > 0:
                    loss_if_stop = pos_value * (price - stop_loss) / price
                    portfolio_stop_risk += max(0, loss_if_stop)
            except Exception:
                pass
        portfolio_exposure_pct = (portfolio_exposure / portfolio_total_value * 100) if portfolio_total_value > 0 else 0.0
        portfolio_stop_risk_pct = (portfolio_stop_risk / portfolio_total_value * 100) if portfolio_total_value > 0 else 0.0
        async with engine._cycle_spent_lock:
            portfolio_available_capital = max(0.0, base_balance - engine._cycle_spent)
        result = {
            "portfolio_total_value": portfolio_total_value,
            "portfolio_exposure": portfolio_exposure,
            "portfolio_stop_risk": portfolio_stop_risk,
            "portfolio_exposure_pct": portfolio_exposure_pct,
            "portfolio_stop_risk_pct": portfolio_stop_risk_pct,
            "portfolio_available_capital": portfolio_available_capital,
        }
        engine._portfolio_exposure_cache = result
        engine._portfolio_exposure_cache_time = now
        return result

    @staticmethod
    def _validate_param_range(
        name: str,
        value: Any,
        min_val: float,
        max_val: float,
        symbol: str,
        allow_none: bool = True,
    ) -> Optional[float]:
        """Validate that a numeric parameter is within [min_val, max_val].
        Returns the float value if valid, None if invalid or missing.
        Logs a warning for invalid values."""
        if value is None:
            return None
        try:
            fval = float(value)
        except (TypeError, ValueError):
            logger.warning(f"Invalid {name}={value!r} for {symbol}: not a number")
            return None
        if fval < min_val or fval > max_val:
            logger.warning(
                f"Invalid {name}={fval} for {symbol}: outside range [{min_val}, {max_val}]"
            )
            return None
        return fval

    async def update_position_params(
        self,
        symbol: str,
        params: Dict[str, Any],
        indicator_config: Optional[Dict[str, Any]],
        timeframe: str,
        current_price: float,
        atr: Optional[float],
    ):
        """Update risk parameters of an open position from LLM strategy_params."""
        engine = self.engine
        async with engine._positions_lock:
            pos = engine.positions.get(symbol)
        if not pos:
            return

        # --- Stop-loss (supports fixed pct and ATR multiple) ---
        stop_method = params.get("stop_loss_method", "fixed")
        if stop_method == "atr_multiple" and atr is not None and atr > 0:
            atr_mult = self._validate_param_range(
                "stop_loss_atr_multiple", params.get("stop_loss_atr_multiple"),
                0.01, 10.0, symbol,
            )
            if atr_mult is not None:
                sl_pct = (atr_mult * atr) / current_price
                pos["stop_loss"] = current_price * (1 - sl_pct)
        elif "stop_loss_pct" in params:
            sl_pct = self._validate_param_range(
                "stop_loss_pct", params["stop_loss_pct"],
                0.0001, 0.95, symbol,
            )
            if sl_pct is not None and current_price > 0:
                pos["stop_loss"] = current_price * (1 - sl_pct)

        # --- Take-profit (supports fixed pct and ATR multiple) ---
        if "take_profit_atr_multiple" in params and atr is not None and atr > 0 and current_price > 0:
            atr_mult = self._validate_param_range(
                "take_profit_atr_multiple", params["take_profit_atr_multiple"],
                0.01, 20.0, symbol,
            )
            if atr_mult is not None:
                tp_pct = (atr_mult * atr) / current_price
                pos["take_profit"] = current_price * (1 + tp_pct)
        elif "take_profit_pct" in params:
            tp_pct = self._validate_param_range(
                "take_profit_pct", params["take_profit_pct"],
                0.0001, 5.0, symbol,
            )
            if tp_pct is not None and current_price > 0:
                pos["take_profit"] = current_price * (1 + tp_pct)

        # --- BTP take-profit cap: enforce smaller targets for bonds ---
        if is_btp_isin(symbol) and pos.get("take_profit") and current_price > 0:
            tp_pct = (pos["take_profit"] - current_price) / current_price
            if tp_pct > settings.BTP_MAX_TAKE_PROFIT_PCT:
                capped_tp = current_price * (1 + settings.BTP_MAX_TAKE_PROFIT_PCT)
                logger.info(
                    f"BTP take-profit capped for {symbol}: {tp_pct:.4%} -> "
                    f"{settings.BTP_MAX_TAKE_PROFIT_PCT:.4%} (price {pos['take_profit']:.4f} -> {capped_tp:.4f})"
                )
                pos["take_profit"] = capped_tp

        # --- Trailing stop ---
        if "trailing_stop" in params:
            _upd_is_btp = is_btp_isin(symbol)
            if _upd_is_btp and params["trailing_stop"]:
                logger.warning(
                    f"LLM set trailing_stop=true for BTP {symbol} in position update, but trailing stops "
                    f"are not supported for BTPs. Forcing trailing_stop=false."
                )
                pos["trailing_stop"] = False
            else:
                pos["trailing_stop"] = params["trailing_stop"]
        if "trailing_stop_distance_pct" in params:
            val = self._validate_param_range(
                "trailing_stop_distance_pct", params["trailing_stop_distance_pct"],
                0.0001, 0.95, symbol,
            )
            if val is not None:
                pos["trailing_stop_distance_pct"] = val
        if "trailing_stop_activation_pct" in params:
            val = self._validate_param_range(
                "trailing_stop_activation_pct", params["trailing_stop_activation_pct"],
                0.0, 0.95, symbol,
            )
            if val is not None:
                pos["trailing_stop_activation_pct"] = val

        # --- Trailing take-profit ---
        if "trailing_take_profit" in params:
            pos["trailing_take_profit"] = params["trailing_take_profit"]
        if "trailing_take_profit_distance_pct" in params:
            val = self._validate_param_range(
                "trailing_take_profit_distance_pct", params["trailing_take_profit_distance_pct"],
                0.0001, 0.95, symbol,
            )
            if val is not None:
                pos["trailing_take_profit_distance_pct"] = val

        # --- Breakeven / lock-profit ---
        if "breakeven_activation_pct" in params:
            val = self._validate_param_range(
                "breakeven_activation_pct", params["breakeven_activation_pct"],
                0.0001, 0.95, symbol,
            )
            if val is not None:
                pos["breakeven_activation_pct"] = val
        # --- Time-based exits ---
        if "max_hold_time_seconds" in params:
            val = self._validate_param_range(
                "max_hold_time_seconds", params["max_hold_time_seconds"],
                1.0, 31_536_000.0, symbol,  # 1 second to ~1 year
            )
            if val is not None:
                pos["max_hold_time_seconds"] = val
                # If the LLM explicitly sets a new hold time, clear any expiry flag
                pos.pop("_max_hold_expired", None)
                pos.pop("_max_hold_expired_count", None)

        # --- Cooldown after loss ---
        if "cooldown_after_loss_seconds" in params:
            val = self._validate_param_range(
                "cooldown_after_loss_seconds", params["cooldown_after_loss_seconds"],
                0.0, 2_592_000.0, symbol,  # 0 to ~30 days
            )
            if val is not None:
                pos["cooldown_after_loss_seconds"] = val

        # --- News sentiment exit ---
        if "news_sentiment_exit_threshold" in params:
            val = self._validate_param_range(
                "news_sentiment_exit_threshold", params["news_sentiment_exit_threshold"],
                -1.0, 0.0, symbol,
            )
            if val is not None:
                pos["news_sentiment_exit_threshold"] = val

        # --- Max unrealized loss ---
        if "max_unrealized_loss_pct" in params:
            val = self._validate_param_range(
                "max_unrealized_loss_pct", params["max_unrealized_loss_pct"],
                0.0001, 0.95, symbol,
            )
            if val is not None:
                pos["max_unrealized_loss_pct"] = val

        # --- Partial take-profit levels ---
        if "partial_take_profit_levels" in params:
            raw_levels = params["partial_take_profit_levels"]
            validated_levels = []
            if isinstance(raw_levels, list):
                for i, level in enumerate(raw_levels):
                    if not isinstance(level, dict):
                        continue
                    lvl_tp = self._validate_param_range(
                        f"partial_take_profit_levels[{i}].take_profit_pct",
                        level.get("take_profit_pct"), 0.0001, 5.0, symbol,
                    )
                    lvl_frac = self._validate_param_range(
                        f"partial_take_profit_levels[{i}].fraction",
                        level.get("fraction"), 0.01, 0.99, symbol,
                    )
                    if lvl_tp is not None and lvl_frac is not None:
                        validated_level = dict(level)
                        validated_level["take_profit_pct"] = lvl_tp
                        validated_level["fraction"] = lvl_frac
                        validated_levels.append(validated_level)
            if validated_levels:
                pos["partial_take_profit_levels"] = validated_levels
                pos["partial_tp_levels_triggered"] = []
                pos["partial_tp_depth_wait_start"] = {}
                # Clear single-level fields to avoid confusion
                pos["partial_take_profit_pct"] = None
                pos["partial_take_profit_fraction"] = None
                pos["partial_tp_triggered"] = None
        else:
            if "partial_take_profit_pct" in params:
                val = self._validate_param_range(
                    "partial_take_profit_pct", params["partial_take_profit_pct"],
                    0.0001, 5.0, symbol,
                )
                if val is not None:
                    pos["partial_take_profit_pct"] = val
            if "partial_take_profit_fraction" in params:
                val = self._validate_param_range(
                    "partial_take_profit_fraction", params["partial_take_profit_fraction"],
                    0.01, 0.99, symbol,
                )
                if val is not None:
                    pos["partial_take_profit_fraction"] = val
            if "partial_tp_triggered" not in pos:
                pos["partial_tp_triggered"] = False

        # --- Strategy interval ---
        if "strategy_interval_seconds" in params:
            val = self._validate_param_range(
                "strategy_interval_seconds", params["strategy_interval_seconds"],
                60.0, 2_592_000.0, symbol,  # 1 minute to ~30 days
            )
            if val is not None:
                engine._strategy_intervals[symbol] = val

        # --- Indicator config ---
        if indicator_config is not None:
            pos["indicator_config"] = indicator_config

        # --- Timeframe (if changed) ---
        if timeframe:
            pos["timeframe"] = timeframe

        logger.info(f"Updated risk parameters for {symbol} from LLM strategy_params")
        engine._portfolio_exposure_cache = None
        engine._state_dirty = True
