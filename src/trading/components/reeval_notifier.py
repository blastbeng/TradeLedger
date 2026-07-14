"""Handles building and sending notifications for symbol re-evaluation."""
import asyncio
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ReevalNotifier:
    """Builds and sends the re-evaluation completion notification."""

    def __init__(self, engine, event_bus):
        self.engine = engine
        self.shared_state = engine.shared_state
        self.event_bus = event_bus

    async def build_and_send_reeval_notification(
        self,
        base_balance: float,
        per_symbol_budget: float,
        pause_trading: Optional[bool],
        pause_reason: str,
        pause_duration: Optional[Any],
        trading_paused_bool: bool,
        force: bool,
        is_user_forced: bool,
        parsed: Dict[str, Any],
        llm_provider: Optional[str],
        llm_model: Optional[str],
    ) -> None:
        """Build and send the re-evaluation completion notification."""
        engine = self.engine

        # Build formatted symbol labels with stock names (parallelized)
        async def _fetch_label(c):
            name = await engine._market_data_manager.get_stock_name(c['symbol'])
            return engine._format_symbol_display(c['symbol'], name, c['timeframe'])
        
        labels_or_exc = await asyncio.gather(
            *[_fetch_label(c) for c in self.shared_state.current_symbols],
            return_exceptions=True
        )
        symbol_labels = []
        for c, res in zip(self.shared_state.current_symbols, labels_or_exc):
            if isinstance(res, Exception):
                logger.error(f"Failed to fetch stock name for {c['symbol']}: {res}", exc_info=res)
                symbol_labels.append(engine._format_symbol_display(c['symbol'], None, c['timeframe']))
            else:
                symbol_labels.append(res)
        logger.info(f"Selected symbols: {symbol_labels}")

        # Build a pause/resume message if the LLM provided a decision
        pause_msg = ""
        if isinstance(pause_trading, bool):
            if pause_trading:
                if trading_paused_bool:
                    pause_msg = "⏸️ LLM decided to keep trading paused"
                else:
                    pause_msg = "⏸️ LLM decided to pause trading"
            else:
                if trading_paused_bool:
                    pause_msg = "▶️ LLM decided to resume trading"
                else:
                    pause_msg = "▶️ LLM decided to keep trading active"
            if pause_reason:
                pause_msg += f" – {pause_reason}"

        # Include pause duration if set
        if pause_duration is not None and isinstance(pause_duration, (int, float)) and pause_duration > 0:
            minutes = pause_duration / 60
            if minutes >= 1:
                duration_str = f"{minutes:.0f} min"
            else:
                duration_str = f"{pause_duration:.0f}s"
            if pause_msg:
                pause_msg += f" (auto‑resume in {duration_str})"
            else:
                pause_msg = f"⏱️ LLM set pause duration: {duration_str}"

        if force:
            market_open = await engine._is_market_open()
            if not market_open:
                status_str = "paused"
                emoji = "⏸️"
            else:
                if trading_paused_bool:
                    if isinstance(pause_trading, bool) and not pause_trading:
                        status_str = "resumed"
                        emoji = "▶️"
                    else:
                        status_str = "paused"
                        emoji = "⏸️"
                else:
                    status_str = "active"
                    emoji = "▶️"
            forced_by = "manually forced" if is_user_forced else "forced by market conditions"
            pause_msg = f"{emoji} Reevaluation has been {forced_by} – Bot is currently {status_str}"
            if pause_reason:
                pause_msg += f" – {pause_reason}"

        if not self.shared_state.current_symbols:
            logger.warning("No symbols selected after evaluation. Bot will idle until next cycle.")
            if engine.notifier:
                msg = f"⚠️ No stocks selected. Bot will idle.\n"
                msg += f"Balance: {base_balance:.2f} {engine.base_currency}, "
                msg += f"Per-symbol budget: {per_symbol_budget:.2f}"
                if pause_msg:
                    msg = pause_msg + "\n" + msg
                await engine.notifier.send_notification(
                    msg,
                    summary={
                        "action": "HOLD",
                        "reason": "No stocks selected",
                        "base_balance": base_balance,
                        "per_symbol_budget": per_symbol_budget,
                        "pause_decision": pause_trading if isinstance(pause_trading, bool) else None,
                        "pause_reason": pause_reason,
                        "model_type": "mind",
                        "llm_provider": llm_provider,
                        "llm_model": llm_model,
                    }
                )
        elif engine.notifier:
            stock_reasoning = parsed.get("reasoning", "") if isinstance(parsed, dict) else ""
            if stock_reasoning:
                msg = f"🔄 Tickers Updated: {', '.join(symbol_labels)}\n💡 {stock_reasoning}"
            else:
                msg = f"🔄 Tickers Updated: {', '.join(symbol_labels)}"
            if pause_msg:
                msg = pause_msg + "\n" + msg
            await engine.notifier.send_notification(
                msg,
                summary={
                    "action": "INFO",
                    "reason": "Symbols updated",
                    "stocks": [c["symbol"] for c in self.shared_state.current_symbols],
                    "stock_reasoning": stock_reasoning,
                    "pause_decision": pause_trading if isinstance(pause_trading, bool) else None,
                    "pause_reason": pause_reason,
                    "model_type": "mind",
                    "llm_provider": llm_provider,
                    "llm_model": llm_model,
                }
            )
