"""Shared mutable state container for trading engine components.

Encapsulates state that was previously accessed directly through the engine
(e.g., engine.positions, engine._positions_lock, engine._cycle_spent).
Components receive this container instead of the full engine, reducing
bidirectional coupling.
"""
import asyncio
import threading
from typing import Any, Dict, List, Optional


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

    def append_trade(self, trade: Dict[str, Any], max_trades: int = 500):
        """Append a trade to history and prune old entries."""
        with self._trade_history_lock:
            self._trade_history_version += 1
            self.trade_history.append(trade)
            if len(self.trade_history) > max_trades:
                pruned = self.trade_history[:-max_trades]
                for t in pruned:
                    if t.get("side") == "sell":
                        self._realized_pnl_offset += t.get("realized_pnl", 0.0)
                self.trade_history = self.trade_history[-max_trades:]
