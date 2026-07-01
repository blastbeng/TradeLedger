"""Risk management component for the TradingEngine.

Handles stop-loss, take-profit, trailing stop, partial TP, dust sweep,
and other risk rule checks on open positions.
Extracted from TradingEngine to reduce class size and improve maintainability.
"""
import asyncio
import logging
import time
from typing import Any, Dict

from src.config.settings import settings
from src.database import insert_position_pnl_snapshot
from src.strategies.base import Signal
from src.utils.symbol_utils import is_btp_isin

logger = logging.getLogger(__name__)


class RiskManager:
    """Handles risk management checks for open positions."""

    def __init__(self, engine):
        self.engine = engine

    async def record_position_pnl_snapshots(self):
        """Record P&L snapshots for all open positions to the database."""
        engine = self.engine
        if not engine.positions:
            return
        pos_tickers = await engine._get_all_position_tickers_sync()
        now_ms = int(time.time() * 1000)
        for symbol, pos in engine.positions.items():
            try:
                t = pos_tickers.get(symbol)
                current_price = t['last'] if t and t.get('last') else pos.get('price', 0.0)
                amount = pos.get('amount', 0.0)
                entry_price = pos.get('price', 0.0)
                cost_basis = pos.get('cost_basis', amount * entry_price)
                position_value = amount * current_price
                unrealized_pnl = (current_price - entry_price) * amount
                pnl_pct = (unrealized_pnl / cost_basis) if cost_basis > 0 else 0.0
                # Realized P&L: sum of all closed sell trades for this symbol
                realized_pnl = sum(
                    t.get("realized_pnl", 0.0)
                    for t in engine.trade_history
                    if t.get("symbol") == symbol and t.get("side") == "sell"
                )
                await asyncio.to_thread(
                    insert_position_pnl_snapshot,
                    symbol=symbol,
                    timestamp=now_ms,
                    unrealized_pnl=round(unrealized_pnl, 6),
                    realized_pnl=round(realized_pnl, 6),
                    position_value=round(position_value, 6),
                    cost_basis=round(cost_basis, 6),
                    amount=amount,
                    current_price=current_price,
                    pnl_pct=round(pnl_pct, 6),
                )
            except Exception as e:
                logger.debug(f"Failed to record P&L snapshot for {symbol}: {e}")

    async def read_review_limits(self) -> Dict[str, int]:
        """Read LLM-decided review limits from Redis, falling back to settings defaults."""
        engine = self.engine
        max_sl_reviews = settings.MAX_STOP_LOSS_REVIEWS
        max_tp_reviews = settings.MAX_TAKE_PROFIT_REVIEWS
        max_partial_tp_reviews = settings.MAX_PARTIAL_TP_REVIEWS
        max_dust_sweep_reviews = settings.MAX_DUST_SWEEP_REVIEWS
        try:
            raw = await asyncio.to_thread(engine.redis.get, "trading:max_stop_loss_reviews")
            if raw:
                max_sl_reviews = int(raw)
            raw = await asyncio.to_thread(engine.redis.get, "trading:max_take_profit_reviews")
            if raw:
                max_tp_reviews = int(raw)
            raw = await asyncio.to_thread(engine.redis.get, "trading:max_partial_tp_reviews")
            if raw:
                max_partial_tp_reviews = int(raw)
            raw = await asyncio.to_thread(engine.redis.get, "trading:max_dust_sweep_reviews")
            if raw:
                max_dust_sweep_reviews = int(raw)
        except Exception:
            pass
        return {
            "max_sl_reviews": max_sl_reviews,
            "max_tp_reviews": max_tp_reviews,
            "max_partial_tp_reviews": max_partial_tp_reviews,
            "max_dust_sweep_reviews": max_dust_sweep_reviews,
        }

    async def check_hard_stop(
        self,
        symbol: str,
        pos: Dict[str, Any],
        current_price: float,
        display_symbol: str,
    ) -> bool:
        """Check if the position has exceeded the hard maximum loss threshold.

        Returns True if the hard stop was triggered (caller should skip to
        the next position), False otherwise.
        """
        engine = self.engine
        entry_price = pos["price"]
        if entry_price <= 0:
            return False
        unrealized_loss_pct = (entry_price - current_price) / entry_price
        _is_btp = is_btp_isin(symbol)
        _hard_max_loss = settings.BTP_HARD_MAX_LOSS_PCT if _is_btp else settings.HARD_MAX_LOSS_PCT
        if unrealized_loss_pct >= _hard_max_loss:
            logger.warning(
                f"Hard max loss threshold reached for {symbol}: "
                f"unrealized loss {unrealized_loss_pct:.2%} >= {_hard_max_loss:.2%}. Forcing SELL."
            )
            if engine.notifier:
                await engine.notifier.send_notification(
                    f"⛔ Hard stop for {display_symbol}: unrealized loss {unrealized_loss_pct:.2%} "
                    f"exceeds maximum {_hard_max_loss:.2%} – force selling.",
                    summary={
                        "symbol": symbol,
                        "action": "SELL",
                        "reason": "Hard maximum loss threshold",
                        "price": current_price,
                        "unrealized_loss_pct": round(unrealized_loss_pct, 4),
                        "exit_reason": "hard_max_loss",
                    }
                )
            await engine._execute_signal(
                symbol,
                Signal(action="SELL", confidence=1.0, reasoning="Hard maximum loss threshold exceeded"),
                exit_reason="hard_max_loss"
            )
            return True
        return False

    async def check_soft_stop(
        self,
        symbol: str,
        pos: Dict[str, Any],
        current_price: float,
        display_symbol: str,
    ) -> bool:
        """Check if the position has exceeded the maximum unrealized loss threshold.

        Returns True if the soft stop was triggered (caller should skip to
        the next position), False otherwise.
        """
        engine = self.engine
        max_ul_pct = pos.get("max_unrealized_loss_pct")
        if max_ul_pct is not None and max_ul_pct > 0:
            entry_price = pos["price"]
            if current_price <= entry_price * (1 - max_ul_pct):
                logger.info(f"Max unrealized loss reached for {symbol} ({max_ul_pct:.2%}). Closing position.")
                if engine.notifier:
                    await engine.notifier.send_notification(
                        f"📉 Soft stop triggered for {display_symbol} at {current_price:.4f} (max loss {max_ul_pct:.2%})",
                        summary={
                            "symbol": symbol,
                            "action": "SELL",
                            "reason": "Max unrealized loss",
                            "price": current_price,
                            "exit_reason": "max_unrealized_loss",
                        }
                    )
                await engine._execute_signal(
                    symbol,
                    Signal(action="SELL", confidence=1.0, reasoning="Max unrealized loss"),
                    exit_reason="max_unrealized_loss"
                )
                return True
        return False

    async def check_news_sentiment_exit(
        self,
        symbol: str,
        pos: Dict[str, Any],
        display_symbol: str,
    ) -> bool:
        """Check if negative news sentiment should trigger an exit.

        Returns True if the sentiment exit was triggered (caller should skip to
        the next position), False otherwise.
        """
        engine = self.engine
        news_threshold = pos.get("news_sentiment_exit_threshold")
        if news_threshold is not None and settings.NEWS_ENABLED:
            pos_tf = pos.get("timeframe")
            if pos_tf and engine._timeframe_to_seconds(pos_tf) >= 604_800:
                logger.debug(
                    f"Skipping news sentiment exit for {symbol}: "
                    f"long-term timeframe ({pos_tf}) ignores short-term sentiment."
                )
            else:
                # Clamp to non-positive: a positive threshold would trigger
                # an exit even when sentiment is mildly positive, which is
                # almost certainly not the LLM's intent.  Only negative
                # compound scores should trigger a sentiment-based exit.
                effective_threshold = min(float(news_threshold), 0.0)
                try:
                    agg = await engine._get_cached_sentiment(symbol)
                    if agg and agg["avg_compound"] < effective_threshold:
                        logger.info(
                            f"News sentiment exit for {symbol}: compound {agg['avg_compound']:.2f} < threshold {effective_threshold}"
                        )
                        if engine.notifier:
                            await engine.notifier.send_notification(
                                f"📰 Negative news exit for {display_symbol} (sentiment {agg['avg_compound']:.2f})",
                                summary={
                                    "symbol": symbol,
                                    "action": "SELL",
                                    "reason": "News sentiment exit",
                                    "sentiment": agg,
                                    "exit_reason": "news_sentiment_exit",
                                }
                            )
                        await engine._execute_signal(
                            symbol,
                            Signal(action="SELL", confidence=1.0, reasoning="News sentiment exit"),
                            exit_reason="news_sentiment_exit"
                        )
                        return True
                except Exception as e:
                    logger.info(f"News sentiment check failed for {symbol}: {e}")
        return False
