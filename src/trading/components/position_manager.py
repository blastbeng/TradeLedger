"""Position management component for the TradingEngine.

Handles position-related operations: cost basis computation and portfolio
exposure calculation.
Extracted from TradingEngine to reduce class size and improve maintainability.
"""
import asyncio
import logging
import time
from typing import Any, Dict, Optional

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
