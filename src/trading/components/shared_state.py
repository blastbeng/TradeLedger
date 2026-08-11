"""Shared mutable state container for trading engine components.

Encapsulates state that was previously accessed directly through the engine
(e.g., engine.positions, engine._positions_lock, engine._cycle_spent).
Components receive this container instead of the full engine, reducing
bidirectional coupling.
"""
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from src.config.settings import settings


class SharedState:
    """Holds mutable state shared across trading engine components."""

    def __init__(self):
        # --- Positions ---
        self.positions: Dict[str, Dict[str, Any]] = {}
        self._positions_lock = asyncio.Lock()

        # --- Queued orders ---
        self.queued_orders: List[Dict[str, Any]] = []
        self._queued_orders_lock = asyncio.Lock()

        # --- Cycle spending tracker ---
        self._cycle_spent: float = 0.0
        self._cycle_spent_lock = asyncio.Lock()

        # --- Balance cache ---
        self._balance_cache: Optional[Dict[str, float]] = None
        self._balance_cache_time: float = 0.0

        # --- Portfolio exposure cache ---
        self._portfolio_exposure_cache: Optional[Dict[str, float]] = None

        # --- Position tickers cache ---
        self._position_tickers_cache: Optional[Dict[str, Dict[str, Any]]] = None
        self._position_tickers_cache_time: float = 0.0

        # --- Trade history ---
        self.trade_history: List[Dict[str, Any]] = []
        self._trade_history_lock = threading.Lock()
        self._trade_history_version: int = 0
        self._realized_pnl_offset: float = 0.0

        # --- Current symbols ---
        self.current_symbols: List[Dict[str, str]] = []
        self._current_symbols_lock = asyncio.Lock()

        # --- Recent signals ---
        self.recent_signals: List[Dict[str, Any]] = []

        # --- Loss tracking ---
        self.last_loss_time: Dict[str, float] = {}
        self.cooldown_durations: Dict[str, float] = {}

        # --- Global risk multiplier ---
        self._global_risk_multiplier: Optional[float] = None

        # --- Evaluation state ---
        self._force_eval: Dict[str, bool] = {}
        self._force_eval_time: Dict[str, float] = {}
        self._last_strategy_eval: Dict[str, float] = {}
        self._strategy_intervals: Dict[str, float] = {}
        self._eval_state_lock = asyncio.Lock()

        # --- Entry signal state ---
        self._entry_signal_state: Dict[str, Dict[str, Any]] = {}

        # --- Last decisions ---
        self._last_decisions: Dict[str, Dict[str, Any]] = {}

        # --- Last eval snapshot ---
        self._last_eval_snapshot: Dict[str, Dict[str, float]] = {}

        # --- Pending entries ---
        self._pending_entries: Dict[str, Dict[str, Any]] = {}
        self._pending_entries_lock = asyncio.Lock()

        # --- Symbol first seen ---
        self._symbol_first_seen: Dict[str, float] = {}

        # --- State management ---
        self._state_lock = asyncio.Lock()
        self._state_save_pending: bool = False
        self._state_dirty: bool = False

        # --- Sentiment cache ---
        self._sentiment_cache: Dict[str, tuple] = {}

        # --- Market breadth ---
        self._market_breadth: Optional[Dict[str, Any]] = None

        # --- Delayed entry tasks ---
        self._delayed_entry_tasks: set = set()

        # --- Daily P&L and fee tracking (avoids pruning issues) ---
        self._daily_realized_pnl: Dict[str, float] = {}
        self._daily_buy_fees: Dict[str, float] = {}

    async def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Safely retrieve a single position."""
        async with self._positions_lock:
            return self.positions.get(symbol)

    async def get_all_positions(self) -> Dict[str, Dict[str, Any]]:
        """Safely retrieve a copy of all positions."""
        async with self._positions_lock:
            return self.positions.copy()

    async def set_position(self, symbol: str, position: Dict[str, Any]) -> None:
        """Safely set or update a position."""
        async with self._positions_lock:
            self.positions[symbol] = position

    async def remove_position(self, symbol: str) -> None:
        """Safely remove a position."""
        async with self._positions_lock:
            self.positions.pop(symbol, None)

    # --- Current symbols ---

    async def get_current_symbols(self) -> List[Dict[str, str]]:
        """Safely retrieve a copy of current symbols."""
        async with self._current_symbols_lock:
            return self.current_symbols.copy()

    async def set_current_symbols(self, symbols: List[Dict[str, str]]) -> None:
        """Safely set the current symbols list."""
        async with self._current_symbols_lock:
            self.current_symbols = symbols

    async def update_current_symbol(self, symbol: str, updates: Dict[str, Any]) -> None:
        """Safely update a specific symbol in the current symbols list."""
        async with self._current_symbols_lock:
            for entry in self.current_symbols:
                if entry.get("symbol") == symbol:
                    entry.update(updates)
                    break

    # --- Queued orders ---

    async def get_queued_orders(self) -> List[Dict[str, Any]]:
        """Safely retrieve a copy of all queued orders."""
        async with self._queued_orders_lock:
            return self.queued_orders.copy()

    async def append_queued_order(self, order: Dict[str, Any]) -> None:
        """Safely append an order to the queue."""
        async with self._queued_orders_lock:
            self.queued_orders.append(order)

    async def clear_queued_orders(self) -> None:
        """Safely clear all queued orders."""
        async with self._queued_orders_lock:
            self.queued_orders.clear()

    # --- Cycle spending tracker ---

    async def get_cycle_spent(self) -> float:
        """Safely retrieve the current cycle spend total."""
        async with self._cycle_spent_lock:
            return self._cycle_spent

    async def add_cycle_spent(self, amount: float) -> None:
        """Safely add to the cycle spend total."""
        async with self._cycle_spent_lock:
            self._cycle_spent += amount

    async def reset_cycle_spent(self) -> None:
        """Safely reset the cycle spend total to zero."""
        async with self._cycle_spent_lock:
            self._cycle_spent = 0.0

    # --- Pending entries ---

    async def get_pending_entry(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Safely retrieve a single pending entry."""
        async with self._pending_entries_lock:
            return self._pending_entries.get(symbol)

    async def get_all_pending_entries(self) -> Dict[str, Dict[str, Any]]:
        """Safely retrieve a copy of all pending entries."""
        async with self._pending_entries_lock:
            return self._pending_entries.copy()

    async def set_pending_entry(self, symbol: str, entry: Dict[str, Any]) -> None:
        """Safely set or update a pending entry."""
        async with self._pending_entries_lock:
            self._pending_entries[symbol] = entry

    async def remove_pending_entry(self, symbol: str) -> None:
        """Safely remove a pending entry."""
        async with self._pending_entries_lock:
            self._pending_entries.pop(symbol, None)

    # --- Evaluation state ---

    async def get_force_eval(self, symbol: str) -> bool:
        """Safely retrieve the force-eval flag for a symbol."""
        async with self._eval_state_lock:
            return self._force_eval.get(symbol, False)

    async def set_force_eval(self, symbol: str, value: bool) -> None:
        """Safely set the force-eval flag for a symbol."""
        async with self._eval_state_lock:
            self._force_eval[symbol] = value

    async def get_force_eval_time(self, symbol: str) -> float:
        """Safely retrieve the force-eval timestamp for a symbol."""
        async with self._eval_state_lock:
            return self._force_eval_time.get(symbol, 0.0)

    async def set_force_eval_time(self, symbol: str, timestamp: float) -> None:
        """Safely set the force-eval timestamp for a symbol."""
        async with self._eval_state_lock:
            self._force_eval_time[symbol] = timestamp

    async def get_last_strategy_eval(self, symbol: str) -> float:
        """Safely retrieve the last strategy eval timestamp for a symbol."""
        async with self._eval_state_lock:
            return self._last_strategy_eval.get(symbol, 0.0)

    async def set_last_strategy_eval(self, symbol: str, timestamp: float) -> None:
        """Safely set the last strategy eval timestamp for a symbol."""
        async with self._eval_state_lock:
            self._last_strategy_eval[symbol] = timestamp

    async def get_strategy_interval(self, symbol: str) -> float:
        """Safely retrieve the strategy interval for a symbol."""
        async with self._eval_state_lock:
            return self._strategy_intervals.get(symbol, 0.0)

    async def set_strategy_interval(self, symbol: str, interval: float) -> None:
        """Safely set the strategy interval for a symbol."""
        async with self._eval_state_lock:
            self._strategy_intervals[symbol] = interval

    def append_trade(self, trade: Dict[str, Any], max_trades: int = 500):
        """Append a trade to history and prune old entries."""
        with self._trade_history_lock:
            self._trade_history_version += 1
            self.trade_history.append(trade)

            # Track daily P&L and buy fees to avoid issues with pruned trade history
            ts = trade.get("timestamp", 0)
            if ts:
                tz = ZoneInfo(settings.MARKET_TIMEZONE)
                trade_date = datetime.fromtimestamp(ts / 1000.0, tz=tz).date().isoformat()
                if trade.get("side") == "sell":
                    self._daily_realized_pnl[trade_date] = self._daily_realized_pnl.get(trade_date, 0.0) + trade.get("realized_pnl", 0.0)
                elif trade.get("side") == "buy":
                    fee = trade.get("fee", {})
                    self._daily_buy_fees[trade_date] = self._daily_buy_fees.get(trade_date, 0.0) + float(fee.get("cost", 0.0) or 0.0)

                # Prune entries older than 7 days to prevent unbounded memory growth
                cutoff_date = (datetime.now(tz) - timedelta(days=7)).date().isoformat()
                for d in list(self._daily_realized_pnl.keys()):
                    if d < cutoff_date:
                        del self._daily_realized_pnl[d]
                for d in list(self._daily_buy_fees.keys()):
                    if d < cutoff_date:
                        del self._daily_buy_fees[d]

            if len(self.trade_history) > max_trades:
                pruned = self.trade_history[:-max_trades]
                for t in pruned:
                    if t.get("side") == "sell":
                        self._realized_pnl_offset += t.get("realized_pnl", 0.0)
                self.trade_history = self.trade_history[-max_trades:]

    def get_daily_realized_pnl(self) -> Dict[str, float]:
        """Safely retrieve a copy of daily realized P&L."""
        with self._trade_history_lock:
            return dict(self._daily_realized_pnl)

    def get_daily_buy_fees(self) -> Dict[str, float]:
        """Safely retrieve a copy of daily buy fees."""
        with self._trade_history_lock:
            return dict(self._daily_buy_fees)
