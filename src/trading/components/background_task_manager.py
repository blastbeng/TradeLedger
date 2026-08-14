import asyncio
import json
import logging

logger = logging.getLogger(__name__)


class BackgroundTaskManager:
    def __init__(self, engine, event_bus):
        self.engine = engine
        self.event_bus = event_bus

    async def _periodic_reconcile(self):
        """Run position reconciliation every 5 minutes (medium/long-term)."""
        while self.engine._running:
            if self.engine._reconcile_running:
                logger.warning("Reconcile still running; skipping this cycle.")
                await asyncio.sleep(60)
                continue
            self.engine._reconcile_running = True
            try:
                await self.event_bus.request("reconcile_positions")
            except asyncio.CancelledError:
                raise
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.warning(f"Reconcile network/IO error: {type(e).__name__}: {e}")
            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, json.JSONDecodeError) as e:
                logger.error(f"Reconcile data/logic error: {type(e).__name__}: {e}", exc_info=True)
                await self.engine._record_unexpected_exception("reconcile", e)
            finally:
                self.engine._reconcile_running = False
            await asyncio.sleep(300)
