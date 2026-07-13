import asyncio
import logging
import time
from typing import Callable, Awaitable, Optional, Dict, Any

logger = logging.getLogger(__name__)

class TaskSupervisor:
    """Supervises a background task, restarting it on failure (Erlang supervisor pattern)."""
    def __init__(
        self,
        coro_factory: Callable[[], Awaitable],
        name: str,
        max_restarts: int = 5,
        restart_delay: float = 5.0,
        cooling_off_period: float = 300.0,
    ):
        self.coro_factory = coro_factory
        self.name = name
        self.max_restarts = max_restarts
        self.restart_delay = restart_delay
        self.cooling_off_period = cooling_off_period
        self._task: Optional[asyncio.Task] = None
        self._restart_count = 0
        self._running = True
        self.last_failure_time: Optional[float] = None
        self.last_exception: Optional[str] = None
        self.is_healthy: bool = True
        self._notifier = None

    async def run(self):
        while self._running:
            try:
                self._task = asyncio.create_task(self.coro_factory(), name=self.name)
                await self._task
                # Task completed without exception — reset restart count
                # so that occasional failures over a long uptime don't exhaust the budget.
                self._restart_count = 0
                # If the task exits normally, it might be a graceful shutdown.
                # If the supervisor is still running, we should restart it.
                if not self._running:
                    break
                logger.info(f"Task {self.name} exited normally but supervisor is still running. Restarting in {self.restart_delay}s...")
                # Delay before restarting to avoid tight loops for tasks that
                # exit immediately (e.g., when a feature is disabled).
                await asyncio.sleep(self.restart_delay)
            except asyncio.CancelledError:
                logger.info(f"Supervisor for {self.name} cancelled.")
                if self._task and not self._task.done():
                    self._task.cancel()
                break
            except Exception as e:
                logger.error(f"Task {self.name} failed: {type(e).__name__}: {e}", exc_info=True)
                self._restart_count += 1
                self.last_failure_time = time.time()
                self.last_exception = str(e)
                if self._restart_count > self.max_restarts:
                    logger.critical(
                        f"Task {self.name} exceeded max_restarts ({self.max_restarts}). "
                        f"Entering cooling off period for {self.cooling_off_period}s before retrying."
                    )
                    if self._notifier:
                        try:
                            asyncio.create_task(self._notifier.send_notification(
                                f"⚠️ Background task '{self.name}' has exceeded max restarts ({self.max_restarts}) "
                                f"and is entering a {self.cooling_off_period}s cooling off period. "
                                f"Last error: {str(e)[:200]}",
                                summary={"action": "WARNING", "reason": f"Task {self.name} cooling off after max restarts"}
                            ))
                        except Exception:
                            pass
                    await asyncio.sleep(self.cooling_off_period)
                    self._restart_count = 0
                    logger.info(f"Cooling off period ended for task {self.name}. Retrying.")
                    continue
                logger.info(f"Restarting task {self.name} in {self.restart_delay}s (attempt {self._restart_count}/{self.max_restarts})")
                await asyncio.sleep(self.restart_delay)

    def set_notifier(self, notifier):
        """Attach a notifier for escalation alerts."""
        self._notifier = notifier

    def cancel(self):
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()

    def get_health(self) -> Dict[str, Any]:
        """Returns the current health and status of the supervised task."""
        return {
            "name": self.name,
            "running": self._running,
            "is_healthy": self.is_healthy,
            "restart_count": self._restart_count,
            "max_restarts": self.max_restarts,
            "last_failure_time": self.last_failure_time,
            "last_exception": self.last_exception,
        }
