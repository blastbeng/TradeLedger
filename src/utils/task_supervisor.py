import asyncio
import logging
from typing import Callable, Awaitable, Optional

logger = logging.getLogger(__name__)

class TaskSupervisor:
    """Supervises a background task, restarting it on failure (Erlang supervisor pattern)."""
    def __init__(
        self,
        coro_factory: Callable[[], Awaitable],
        name: str,
        max_restarts: int = 5,
        restart_delay: float = 5.0,
    ):
        self.coro_factory = coro_factory
        self.name = name
        self.max_restarts = max_restarts
        self.restart_delay = restart_delay
        self._task: Optional[asyncio.Task] = None
        self._restart_count = 0
        self._running = True

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
                logger.error(f"Task {self.name} failed: {e}", exc_info=True)
                self._restart_count += 1
                if self._restart_count > self.max_restarts:
                    logger.critical(f"Task {self.name} exceeded max_restarts ({self.max_restarts}). Aborting supervisor.")
                    raise
                logger.info(f"Restarting task {self.name} in {self.restart_delay}s (attempt {self._restart_count}/{self.max_restarts})")
                await asyncio.sleep(self.restart_delay)

    def cancel(self):
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
