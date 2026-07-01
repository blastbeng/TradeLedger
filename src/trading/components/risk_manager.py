"""Risk management component for the TradingEngine.

Handles stop-loss, take-profit, trailing stop, partial TP, dust sweep,
and other risk rule checks on open positions.
Extracted from TradingEngine to reduce class size and improve maintainability.
"""
import asyncio
import logging
import time
from typing import Any, Dict

from src.database import insert_position_pnl_snapshot

logger = logging.getLogger(__name__)


class RiskManager:
    """Handles risk management checks for open positions."""

    def __init__(self, engine):
        self.engine = engine

    async def record_position_pnl_snapshots(self):
        """Record P&L snapshots for all open positions to the database."""
        engine = self.engine
        if not engine.positions:
            return
        pos_tickers = await engine._get_all_position_tickers_sync()
        now_ms = int(time.time() * 1000)
        for symbol, pos in engine.positions.items():
            try:
                t = pos_tickers.get(symbol)
                current_price = t['last'] if t and t.get('last') else pos.get('price', 0.0)
                amount = pos.get('amount', 0.0)
                entry_price = pos.get('price', 0.0)
                cost_basis = pos.get('cost_basis', amount * entry_price)
                position_value = amount * current_price
                unrealized_pnl = (current_price - entry_price) * amount
                pnl_pct = (unrealized_pnl / cost_basis) if cost_basis > 0 else 0.0
                # Realized P&L: sum of all closed sell trades for this symbol
                realized_pnl = sum(
                    t.get("realized_pnl", 0.0)
                    for t in engine.trade_history
                    if t.get("symbol") == symbol and t.get("side") == "sell"
                )
                await asyncio.to_thread(
                    insert_position_pnl_snapshot,
                    symbol=symbol,
                    timestamp=now_ms,
                    unrealized_pnl=round(unrealized_pnl, 6),
                    realized_pnl=round(realized_pnl, 6),
                    position_value=round(position_value, 6),
                    cost_basis=round(cost_basis, 6),
                    amount=amount,
                    current_price=current_price,
                    pnl_pct=round(pnl_pct, 6),
                )
            except Exception as e:
                logger.debug(f"Failed to record P&L snapshot for {symbol}: {e}")
