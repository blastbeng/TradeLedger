import asyncio
import logging
from src.utils.task_supervisor import TaskSupervisor

logger = logging.getLogger(__name__)

class EngineOrchestrator:
    """Handles background task orchestration for the TradingEngine."""

    def __init__(self, engine):
        self.engine = engine

    async def start_background_tasks(self):
        """Initialize and start all supervised background tasks."""
        self.engine._background_tasks.clear()
        self.engine._supervisors.clear()
        
        background_factories = [
            self.engine._refresh_news_cache,
            self.engine._refresh_current_symbols_news_fast,
            self.engine._download_market_data_loop,
            self.engine._download_all_assets_data_loop,
            self.engine._download_all_news_loop,
            self.engine._risk_management_loop,
            self.engine._periodic_reconcile,
            self.engine._periodic_reevaluate,
            self.engine._periodic_pause_check,
            self.engine._periodic_pause_resume_check,
            self.engine._periodic_full_market_breadth,
            self.engine._periodic_market_condition_check,
            self.engine._periodic_portfolio_rebalance,
            self.engine._check_pending_entries,
            self.engine._cleanup_orphaned_orders,
            self.engine._process_queued_orders,
            self.engine._monitor_entry_signals_loop,
            self.engine._market_clock_monitor,
            self.engine._refresh_all_quotes_loop,
            self.engine._refresh_ticker_discovery_loop,
            self.engine._fetch_dividends_loop,
            self.engine._redis_health_check_loop,
            self.engine._health_check_loop,
            self.engine._evaluate_llm_decisions_loop,
        ]
        
        for factory in background_factories:
            sup = TaskSupervisor(factory, name=factory.__qualname__)
            sup.set_notifier(self.engine.notifier)
            task = asyncio.create_task(sup.run(), name=f"supervisor:{factory.__qualname__}")
            task.add_done_callback(self.engine._log_task_exception)
            self.engine._background_tasks.append(task)
            self.engine._supervisors.append(sup)
