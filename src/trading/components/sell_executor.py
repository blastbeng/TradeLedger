"""SELL order execution component for the TradingEngine.

Handles SELL order creation, fill processing, partial take-profits, and dust sweeps.
Extracted from TradingEngine to reduce class size and improve maintainability.
"""
import asyncio
import logging
import time
from dataclasses import asdict
from typing import Any, Dict, Optional

from src.config.settings import settings
from src.database import insert_trade
from src.strategies.base import Signal
from src.utils.symbol_utils import is_btp_isin

logger = logging.getLogger(__name__)


class SellExecutor:
    """Handles SELL order execution and fill processing for the TradingEngine."""

    def __init__(self, engine, event_bus):
        self.engine = engine
        self.event_bus = event_bus
        self._exit_order_manager = None
