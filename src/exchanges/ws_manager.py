import asyncio
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Dummy WebSocket manager. Alpaca is no longer used; engine uses REST polling."""

    def __init__(self, stream=None, symbols: List[str] = None):
        self.symbols = set(self._plain(s) for s in (symbols or []))
        self.tickers: Dict[str, Dict[str, Any]] = {}
        self._running = False

    @staticmethod
    def _plain(symbol: str) -> str:
        return symbol.split("/")[0] if "/" in symbol else symbol

    async def start(self):
        self._running = True
        logger.info("WebSocket manager started (dummy mode, using REST polling).")

    async def stop(self):
        self._running = False

    async def update_subscriptions(self, symbols: List[str]):
        self.symbols = set(self._plain(s) for s in symbols)

    def get_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        return None

    def get_order_book(self, symbol: str) -> Optional[Dict[str, Any]]:
        return None

    def get_trades(self, symbol: str) -> List[Dict[str, Any]]:
        return []

    async def wait_for_update(self, timeout: float = 5.0) -> Optional[tuple]:
        await asyncio.sleep(timeout)
        return None

    @property
    def healthy(self) -> bool:
        return False
