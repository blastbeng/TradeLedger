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
        logger.debug(
            f"EventBus subscription: '{event_name}' -> {getattr(callback, '__qualname__', repr(callback))}"
        )

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
                logger.error(f"Event handler error for '{event_name}': {type(e).__name__}: {e}", exc_info=True)

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
            logger.exception(f"Event handler error for '{event_name}': {type(e).__name__}: {e}")
            return None

    def log_subscription_summary(self) -> None:
        """Log a complete registry of all event subscriptions.

        Call this after all components have been initialized to produce
        a central, readable map of every event and its handler(s).
        """
        if not self._subscribers:
            logger.info("EventBus registry: no subscriptions registered.")
            return

        lines = ["EventBus subscription registry:"]
        for event_name in sorted(self._subscribers.keys()):
            callbacks = self._subscribers[event_name]
            for idx, cb in enumerate(callbacks):
                qualname = getattr(cb, '__qualname__', repr(cb))
                is_async = asyncio.iscoroutinefunction(cb)
                lines.append(
                    f"  {event_name} -> {qualname} "
                    f"({'async' if is_async else 'sync'}, handler {idx + 1}/{len(callbacks)})"
                )
        logger.info("\n".join(lines))
