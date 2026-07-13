import asyncio
import logging
from typing import Callable, Dict, List

logger = logging.getLogger(__name__)


class EventBus:
    """Asynchronous event bus for decoupling engine components."""

    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}

    def subscribe(self, event_name: str, callback: Callable):
        if event_name not in self._subscribers:
            self._subscribers[event_name] = []
        self._subscribers[event_name].append(callback)

    async def publish(self, event_name: str, *args, **kwargs):
        if event_name not in self._subscribers:
            return
        for callback in self._subscribers[event_name]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(*args, **kwargs)
                else:
                    callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"Event handler error for '{event_name}': {e}", exc_info=True)

    async def request(self, event_name: str, *args, **kwargs):
        """Send a command/query via the event bus and return the result of the first subscriber."""
        if event_name not in self._subscribers or not self._subscribers[event_name]:
            return None
        callback = self._subscribers[event_name][0]
        try:
            if asyncio.iscoroutinefunction(callback):
                return await callback(*args, **kwargs)
            return callback(*args, **kwargs)
        except Exception as e:
            logger.error(f"Event handler error for '{event_name}': {type(e).__name__}: {e}", exc_info=True)
            return None
