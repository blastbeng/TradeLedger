import asyncio
import logging
import time
from typing import Any, Dict

from src.config.settings import settings
from src.database import load_trading_state, save_trading_state, reset_paper_trading_data
from src.trading.paper_trader import PaperTrader

logger = logging.getLogger(__name__)

class StateInitializer:
    def __init__(self, engine, event_bus):
        self.engine = engine
        self.event_bus = event_bus

    async def _initialize_clients(self):
        """Initialize clients and load persisted state (non‑blocking)."""
        # Check if PAPER_INITIAL_BALANCE changed since last run
        state = await asyncio.to_thread(load_trading_state)
        persisted_balance = state.get("paper_initial_balance")

        # If paper_initial_balance was never persisted, infer it from paper_balances
        if persisted_balance is None:
            paper_balances = state.get("paper_balances")
            if paper_balances and isinstance(paper_balances, dict):
                persisted_balance = paper_balances.get(self.engine.base_currency)
                if persisted_balance is not None:
                    logger.info(
                        f"paper_initial_balance not found in DB. "
                        f"Inferred {persisted_balance} from paper_balances. "
                        f"Current setting: {settings.PAPER_INITIAL_BALANCE}."
                    )

        if persisted_balance is not None and persisted_balance != settings.PAPER_INITIAL_BALANCE:
            logger.info(
                f"PAPER_INITIAL_BALANCE changed from {persisted_balance} to {settings.PAPER_INITIAL_BALANCE}. "
                "Resetting paper trading state."
            )
            await self.reset_paper_trading_state()
        else:
            self.engine.trader = PaperTrader()
            logger.info(f"PaperTrader initialized for {settings.TRADING_MODE} trading mode.")
            try:
                self.engine._state_persistence.load_state()
            except ValueError as e:
                logger.critical(f"State corruption detected during load: {e}. Resetting paper trading state.")
                await self.reset_paper_trading_state()
            self.engine._position_manager.ensure_cost_basis()
            # Initialize _cycle_spent from any queued buy orders loaded from persisted
            # state so capital is reserved immediately at startup, before the first
            # re-evaluation cycle runs (which would otherwise leave _cycle_spent at 0.0
            # and allow over-allocation of capital already reserved by stale orders).
            queued_buy_total = sum(
                q.get('amount', 0.0) for q in self.engine.shared_state.queued_orders
                if q.get('side') == 'buy'
            )
            async with self.engine.shared_state._cycle_spent_lock:
                self.engine.shared_state._cycle_spent = queued_buy_total
            if queued_buy_total > 0:
                logger.info(f"Initialized _cycle_spent={queued_buy_total:.2f} from {sum(1 for q in self.engine.shared_state.queued_orders if q.get('side') == 'buy')} queued buy orders.")

        # Persist the current PAPER_INITIAL_BALANCE so we can detect changes on next startup
        await asyncio.to_thread(save_trading_state, "paper_initial_balance", settings.PAPER_INITIAL_BALANCE)

    async def reset_paper_trading_state(self):
        """Reset paper trading state."""
        logger.info("Resetting paper trading state...")

        # Clear in-memory state
        self.engine.shared_state.positions.clear()
        self.engine.shared_state.queued_orders.clear()
        self.engine.shared_state.current_symbols.clear()
        self.engine.shared_state._pending_entries.clear()
        async with self.engine.shared_state._eval_state_lock:
            self.engine.shared_state._last_strategy_eval.clear()
            self.engine.shared_state._strategy_intervals.clear()
            self.engine.shared_state._force_eval.clear()
            self.engine.shared_state._force_eval_time.clear()
        self.engine.shared_state._entry_signal_state.clear()
        self.engine.shared_state._last_decisions.clear()
        self.engine.shared_state._last_eval_snapshot.clear()
        async with self.engine.shared_state._cycle_spent_lock:
            self.engine.shared_state._cycle_spent = 0.0
        self.engine.shared_state._balance_cache = None
        self.engine.shared_state._balance_cache_time = 0.0
        self.engine.shared_state._position_tickers_cache = None
        self.engine.shared_state._position_tickers_cache_time = 0.0
        self.engine._perf_cache = None
        self.engine._perf_cache_time = 0.0
        self.engine._perf_cache_trade_count = -1
        self.engine._trade_pattern_cache = None
        self.engine._trade_pattern_cache_trade_count = -1
        self.engine.shared_state._trade_history_version = 0
        self.engine.shared_state._realized_pnl_offset = 0.0
        self.engine.shared_state.trade_history.clear()
        self.engine.shared_state.recent_signals.clear()
        self.engine.shared_state.last_loss_time.clear()
        self.engine.shared_state.cooldown_durations.clear()
        self.engine.shared_state._global_risk_multiplier = None
        self.engine.shared_state._symbol_first_seen.clear()
        self.engine.shared_state._sentiment_cache.clear()
        self.engine.shared_state._market_breadth = None
        self.engine.shared_state._daily_realized_pnl.clear()
        self.engine.shared_state._daily_buy_fees.clear()
        self.engine.initial_balance = settings.PAPER_INITIAL_BALANCE

        # Clear the persisted peak total equity so drawdown starts fresh
        await asyncio.to_thread(self.engine.redis.delete, "trading:peak_total_equity")

        # Reset DB data (unconditionally clear all trade data for both modes)
        await asyncio.to_thread(reset_paper_trading_data, keep_trade_history=False)

        # Re-initialize paper trader with new balance
        self.engine.trader = PaperTrader()

        # Save the fresh state
        self.engine.shared_state._state_dirty = True
        await self.engine._state_persistence.save_state(force=True)

        # Persist the new PAPER_INITIAL_BALANCE so we don't reset again on next restart
        await asyncio.to_thread(save_trading_state, "paper_initial_balance", settings.PAPER_INITIAL_BALANCE)

        # Persist the new initial_balance so profit calculations are correct after restart
        await asyncio.to_thread(save_trading_state, "initial_balance", settings.PAPER_INITIAL_BALANCE)

        if self.engine.notifier:
            await self.engine.notifier.send_notification(
                "♻️ Paper trading state has been reset.",
                summary={"action": "RESET", "reason": "State reset"}
            )
        logger.info("Paper trading state reset complete.")
