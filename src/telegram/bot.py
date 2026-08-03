import asyncio
import collections
import html
import json
import logging
import os
import re
import threading
import time
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import List
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from src.config.settings import settings
from src.trading.engine import TradingEngine
from src.utils.redis_client import get_redis_client
from src.database import set_telegram_chat_id, get_telegram_chat_id, get_news_for_symbol, get_signals
from src.llm.prompts import _format_news_for_prompt, get_cached_news_summary

logger = logging.getLogger(__name__)

class TelegramBot:
    _log_lock = threading.Lock()
    MAX_LOG_SIZE = 512 * 1024   # 512 KB
    MAX_LOG_BACKUPS = 10

    def __init__(self, engine: TradingEngine):
        self.engine = engine
        self.redis = get_redis_client()
        # Allowed chat ID – bot will only respond to this chat
        self.allowed_chat_id = None
        if settings.TELEGRAM_CHAT_ID:
            try:
                self.allowed_chat_id = int(settings.TELEGRAM_CHAT_ID)
            except ValueError:
                logger.error("TELEGRAM_CHAT_ID must be a valid integer")
        else:
            logger.warning("TELEGRAM_CHAT_ID not set. Bot will not respond to any chat.")
        self.app = (
            Application.builder()
            .token(settings.TELEGRAM_BOT_TOKEN)
            .concurrent_updates(True)
            .build()
        )
        # Reduce long-polling connection churn to avoid asyncio socket warnings
        self.app.updater.poll_interval = 5.0   # default 0.0 (aggressive)
        self.app.updater.poll_timeout = 30.0   # default 10.0
        self._notification_timestamps = collections.deque()
        self._max_notifications_per_minute = 15
        self._notification_queue = asyncio.Queue()
        self._notification_task = None
        self._register_handlers()
        self.keyboard = ReplyKeyboardMarkup(
            [
                [KeyboardButton("📊 Status"), KeyboardButton("📈 Trades")],
                [KeyboardButton("💰 Profit"), KeyboardButton("🚀 Performance")],
                [KeyboardButton("⚠️ Risk"), KeyboardButton("📡 Signals")],
                [KeyboardButton("📰 News"), KeyboardButton("🔄 Re-eval")],
                [KeyboardButton("⏸️ Pause"), KeyboardButton("▶️ Resume")],
                [KeyboardButton("💸 Sell All"), KeyboardButton("⬇️ Backfill")],
            ],
            resize_keyboard=True,
        )

    def _is_authorized(self, update: Update) -> bool:
        """Return True if the update comes from the allowed chat ID."""
        if self.allowed_chat_id is None:
            return False
        return update.effective_chat.id == self.allowed_chat_id

    async def _send_long_reply(self, update: Update, text: str, parse_mode: str = None, reply_markup=None):                                                                                                                                                       
        """Send a message, splitting it into chunks if it exceeds Telegram's 4096 char limit."""                                                                                                                                                                  
        chunks = self._split_text(text)                                                                                                                                                                                                                           
        for i, chunk in enumerate(chunks):                                                                                                                                                                                                                        
            # Only attach reply_markup to the last message                                                                                                                                                                                                        
            markup = reply_markup if i == len(chunks) - 1 else None                                                                                                                                                                                               
            await update.message.reply_text(chunk, parse_mode=parse_mode, reply_markup=markup)

    @staticmethod
    def _split_text(text: str, max_len: int = 4000) -> List[str]:
        """Split text into chunks, trying to avoid breaking HTML tags."""
        if len(text) <= max_len:
            return [text]

        chunks = []
        current_chunk = ""
        lines = text.split('\n')
        for line in lines:
            if len(current_chunk) + len(line) + 1 > max_len:
                if current_chunk:
                    chunks.append(current_chunk)
                while len(line) > max_len:
                    split_at = max_len
                    # Avoid splitting inside an HTML tag
                    if '<' in line[:max_len] and '>' not in line[:max_len]:
                        split_at = line.rfind('<', 0, max_len)
                    chunks.append(line[:split_at])
                    line = line[split_at:]
                current_chunk = line
            else:
                if current_chunk:
                    current_chunk += "\n" + line
                else:
                    current_chunk = line
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    def _register_handlers(self):
        self.app.add_handler(CommandHandler("start", self.cmd_start))
        self.app.add_handler(CommandHandler("menu", self.cmd_menu))
        self.app.add_handler(CommandHandler("pause", self.cmd_pause))
        self.app.add_handler(CommandHandler("resume", self.cmd_resume))
        self.app.add_handler(CommandHandler("status", self.cmd_status))
        self.app.add_handler(CommandHandler("trades", self.cmd_trades))
        self.app.add_handler(CommandHandler("profit", self.cmd_profit))
        self.app.add_handler(CommandHandler("performance", self.cmd_performance))
        self.app.add_handler(CommandHandler("news", self.cmd_news_search))
        self.app.add_handler(CommandHandler("news_status", self.cmd_news_status))
        self.app.add_handler(CommandHandler("risk", self.cmd_risk))
        self.app.add_handler(CommandHandler("market", self.cmd_market))
        self.app.add_handler(CommandHandler("sell", self.cmd_sell))
        self.app.add_handler(CommandHandler("backfill", self.cmd_backfill))
        self.app.add_handler(CommandHandler("signals", self.cmd_signals))
        self.app.add_handler(CommandHandler("reset", self.cmd_reset))
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_button))

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            return
        chat_id = update.effective_chat.id
        try:
            await asyncio.wait_for(asyncio.to_thread(set_telegram_chat_id, chat_id), timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning("set_telegram_chat_id timed out")
        await update.message.reply_text(
            "Bot started! You will receive trade notifications here.\nUse the buttons below or type /menu to see them again.",
            reply_markup=self.keyboard,
        )

    async def cmd_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            return
        await update.message.reply_text("Choose an option:", reply_markup=self.keyboard)

    async def handle_button(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            return
        text = update.message.text
        logger.debug(f"Received button text: {text}")
        if text == "📊 Status":
            await self.cmd_status(update, context)
        elif text == "📈 Trades":
            await self.cmd_trades(update, context)
        elif text == "💰 Profit":
            await self.cmd_profit(update, context)
        elif text == "🚀 Performance":
            await self.cmd_performance(update, context)
        elif text == "⏸️ Pause":
            await self.cmd_pause(update, context)
        elif text == "▶️ Resume":
            await self.cmd_resume(update, context)
        elif text == "📰 News":
            await self.cmd_news(update, context)
        elif text == "⚠️ Risk":
            await self.cmd_risk(update, context)
        elif text == "🔄 Re-eval":
            await self.cmd_force_reeval(update, context)
        elif text == "💸 Sell All":
            await self.cmd_sell(update, context)
        elif text == "⬇️ Backfill":
            await self.cmd_backfill(update, context)
        elif text == "📡 Signals":
            await self.cmd_signals(update, context)
        else:
            # Any other text (e.g., first message "hi") shows the keyboard
            await update.message.reply_text(
                "Use the buttons below to interact with the bot.",
                reply_markup=self.keyboard,
            )

    async def cmd_pause(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            return
        try:
            from src.utils.pause_utils import set_trading_pause
            await asyncio.wait_for(asyncio.to_thread(set_trading_pause, self.redis, "manual", set_pause_start=False), timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning("Redis operation timed out during pause")
            await update.message.reply_text("⚠️ Failed to pause: Redis timed out.", reply_markup=self.keyboard)
            return
        await self.send_notification(
            "⏸️ Trading paused manually.",
            summary={"action": "PAUSE", "reason": "Manual pause"}
        )
        await update.message.reply_text("Trading paused.", reply_markup=self.keyboard)

    async def cmd_resume(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            return
        # Refuse to resume if the market is currently closed
        try:
            is_open = await asyncio.wait_for(
                self.engine._is_market_open(),
                timeout=10.0
            )
        except asyncio.TimeoutError:
            await update.message.reply_text("⚠️ Market status check timed out.", reply_markup=self.keyboard)
            return
        if not is_open:
            await update.message.reply_text(
                "⏸️ Cannot resume: market is currently closed.",
                reply_markup=self.keyboard
            )
            return
        # Delete all pause-related keys
        try:
            from src.utils.pause_utils import clear_trading_pause_keys
            await asyncio.wait_for(asyncio.to_thread(clear_trading_pause_keys, self.redis), timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning("Redis operation timed out during resume")
            await update.message.reply_text("⚠️ Failed to resume: Redis timed out.", reply_markup=self.keyboard)
            return
        self.engine.trigger_symbol_reevaluation()
        await self.send_notification(
            "▶️ Trading resumed manually.",
            summary={"action": "RESUME", "reason": "Manual resume"}
        )
        await update.message.reply_text("Trading resumed.", reply_markup=self.keyboard)

    async def cmd_reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            return
        await update.message.reply_text("♻️ Resetting trading state...", reply_markup=self.keyboard)
        try:
            await asyncio.wait_for(self.engine.reset_paper_trading_state(), timeout=30.0)
        except asyncio.TimeoutError:
            await update.message.reply_text("⚠️ Reset timed out.", reply_markup=self.keyboard)
            return
        await update.message.reply_text("✅ Trading state has been reset.", reply_markup=self.keyboard)

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            return
        try:
            symbols = self.engine.current_symbols
            positions = self.engine.positions
            balance = await asyncio.wait_for(
                asyncio.to_thread(self.engine.trader.fetch_balance),
                timeout=15.0
            )
        except asyncio.TimeoutError:
            logger.warning("fetch_balance timed out for status command")
            await update.message.reply_text("⚠️ Balance fetch timed out.", reply_markup=self.keyboard)
            return
        except Exception as e:
            logger.error(f"Failed to get status: {e}", exc_info=True)
            await update.message.reply_text("⚠️ Could not retrieve status.", reply_markup=self.keyboard)
            return

        pos_symbols = {sym.split("/")[0] for sym in positions.keys()}
        pos_quotes = {}
        if pos_symbols:
            try:
                pos_quotes = await asyncio.wait_for(
                    self.engine._market_data_manager._get_quotes_async(list(pos_symbols), timeout=15.0),
                    timeout=20.0
                )
            except asyncio.TimeoutError:
                logger.warning("Batch quote fetch timed out for status")
            except Exception as e:
                logger.warning(f"Batch quote fetch failed for status: {type(e).__name__}: {e}")

        msg = "<b>📊 Current Status</b>\n\n"
        msg += "<b>📈 Tracked Tickers:</b>\n"
        if symbols:
            try:
                names = await asyncio.wait_for(
                    asyncio.gather(*[self.engine._market_data_manager.get_stock_name(entry["symbol"]) for entry in symbols]),
                    timeout=15.0
                )
            except asyncio.TimeoutError:
                logger.warning("get_stock_name timed out for tracked tickers")
                names = [entry["symbol"] for entry in symbols]
            for i, entry in enumerate(symbols):
                symbol = entry["symbol"]
                tf = entry["timeframe"]
                display = self.engine._format_symbol_display(symbol, names[i], tf)
                msg += f"  • <code>{display}</code>\n"
        else:
            msg += "  None\n"
        msg += "\n"

        if positions:
            msg += "<b>📈 Open Positions:</b>\n"
            try:
                pos_names = await asyncio.wait_for(
                    asyncio.gather(*[self.engine._market_data_manager.get_stock_name(sym) for sym in positions.keys()]),
                    timeout=15.0
                )
            except asyncio.TimeoutError:
                logger.warning("get_stock_name timed out for open positions")
                pos_names = list(positions.keys())
            for i, (sym, pos) in enumerate(positions.items()):
                pos_tf = pos.get("timeframe")
                pos_name = pos_names[i]
                pos_display = self.engine._format_symbol_display(sym, pos_name, pos_tf)
                msg += (
                    f"  • <code>{pos_display}</code>\n"
                    f"    Amount: {pos['amount']:.6f}\n"
                    f"    Entry: {pos['price']:.4f}\n"
                )
                base_sym = sym.split("/")[0]
                ticker = pos_quotes.get(base_sym)
                current_price = ticker.get('last') if ticker else None
                if current_price is not None:
                    pnl = (current_price - pos['price']) * pos['amount']
                    pnl_pct = ((current_price - pos['price']) / pos['price']) * 100 if pos['price'] else 0
                    pnl_sign = "+" if pnl >= 0 else ""
                    pnl_pct_sign = "+" if pnl_pct >= 0 else ""
                    msg += f"    Current: {current_price:.4f}  P&L: {pnl_sign}{pnl:.2f} ({pnl_pct_sign}{pnl_pct:.2f}%)\n"
                msg += f"    SL: {pos['stop_loss']:.4f}  TP: {pos['take_profit']:.4f}\n"
        else:
            msg += "<b>📈 Open Positions:</b> None\n"

        msg += "\n<b>💰 Balances:</b>\n"
        non_zero = {k: v for k, v in balance.items() if v > 0}
        if non_zero:
            for cur, amt in non_zero.items():
                msg += f"  • {cur}: {amt:.6f}\n"
        else:
            msg += "  No balances\n"

        # Trading paused status
        try:
            paused = await asyncio.wait_for(asyncio.to_thread(self.redis.get, "trading:paused"), timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning("Redis get timed out for trading:paused")
            paused = None
        status_text = "⏸️ Paused" if paused else "▶️ Active"
        msg += f"\n<b>⚙️ Trading:</b> {status_text}\n"

        if paused:
            try:
                pause_status = await asyncio.wait_for(
                    self.engine.get_pause_status(),
                    timeout=10.0
                )
            except asyncio.TimeoutError:
                logger.warning("get_pause_status timed out")
                pause_status = {}
            except Exception:
                pause_status = {}
            pause_reason = pause_status.get("reason", "")
            countdown = pause_status.get("countdown_str", "")
            if pause_reason:
                msg += f"<b>⏸️ Reason:</b> {pause_reason}\n"
            if countdown:
                msg += f"<b>⏱️ Resumes in:</b> {countdown}\n"
            market_time = pause_status.get("market_time_str", "")
            if market_time:
                msg += f"<b>🕒 Market Time:</b> {market_time}\n"

        queued_count = len(self.engine.queued_orders)
        if queued_count > 0:
            msg += f"\n<b>⏳ Queued Orders:</b> {queued_count}\n"

        # Market status
        try:
            raw = await asyncio.wait_for(asyncio.to_thread(self.redis.get, "market:status"), timeout=5.0)
            if raw:
                data = json.loads(raw)
                msg += "\n<b>🌐 Market Status</b>\n"
                if data.get("market_breadth"):
                    mb = data["market_breadth"]
                    msg += f"  📊 Breadth (candidates): {mb['positive_pct']}% positive ({mb['positive_count']}/{mb['total_count']})\n"
                if data.get("full_market_breadth"):
                    fmb = data["full_market_breadth"]
                    msg += f"  🌐 Full Breadth: {fmb['positive_pct']}% positive ({fmb['positive_count']}/{fmb['total_count']})\n"
                if data.get("spy_price") is not None:
                    msg += f"  📈 Benchmark: {data['spy_price']:.2f}\n"
        except Exception:
            pass

        await self._send_long_reply(update, msg, parse_mode='HTML', reply_markup=self.keyboard)

    async def cmd_trades(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            return
        try:
            open_trades = await asyncio.wait_for(
                self.engine.event_bus.request("get_open_trades"),
                timeout=15.0
            )
        except asyncio.TimeoutError:
            await update.message.reply_text("⚠️ Fetching open trades timed out.", reply_markup=self.keyboard)
            return
        except Exception as e:
            logger.error(f"Failed to get open trades: {e}", exc_info=True)
            await update.message.reply_text("⚠️ Could not retrieve open trades.", reply_markup=self.keyboard)
            return

        queued_orders = self.engine.queued_orders

        # Batch-fetch current prices for all symbols in open trades and queued orders
        all_price_symbols = set()
        for t in open_trades:
            all_price_symbols.add(t['symbol'].split("/")[0])
        for q in queued_orders:
            all_price_symbols.add(q['symbol'].split("/")[0])
        batch_quotes = {}
        if all_price_symbols:
            try:
                batch_quotes = await asyncio.wait_for(
                    self.engine._market_data_manager._get_quotes_async(list(all_price_symbols), timeout=15.0),
                    timeout=20.0
                )
            except asyncio.TimeoutError:
                logger.warning("Batch quote fetch timed out for trades")
            except Exception as e:
                logger.warning(f"Batch quote fetch failed for trades: {type(e).__name__}: {e}")

        if not open_trades and not queued_orders:
            await update.message.reply_text("📈 No open trades or queued orders.", reply_markup=self.keyboard)
            return

        msg = "<b>📈 Open Trades</b>\n\n" if open_trades else ""
        try:
            trade_names = await asyncio.wait_for(
                asyncio.gather(*[self.engine._market_data_manager.get_stock_name(t['symbol']) for t in open_trades]),
                timeout=15.0
            ) if open_trades else []
        except asyncio.TimeoutError:
            logger.warning("get_stock_name timed out for open trades")
            trade_names = [t['symbol'] for t in open_trades]
        for idx, t in enumerate(open_trades, start=1):
            sym = t['symbol']
            trade_tf = t.get('timeframe')
            trade_name = trade_names[idx-1]
            trade_display = self.engine._format_symbol_display(sym, trade_name, trade_tf)
            amt = t['amount']
            price = t['price']
            fee = t.get('fee', {})
            fee_cost = fee.get('cost', 0) or 0
            fee_currency = fee.get('currency', '')
            fee_str = f"{fee_cost:.6f} {fee_currency}" if fee_cost else "—"

            ts = datetime.fromtimestamp(t['timestamp'] / 1000).strftime('%Y-%m-%d %H:%M:%S')

            # Use batched quote
            base_sym = sym.split("/")[0]
            ticker = batch_quotes.get(base_sym)
            current_price = ticker.get('last') if ticker else None

            # --- Get position for SL/TP and sector ---
            pos = self.engine.positions.get(sym)
            sl = pos.get('stop_loss') if pos else None
            tp = pos.get('take_profit') if pos else None
            sector = None
            for entry in self.engine.current_symbols:
                if entry['symbol'] == sym:
                    sector = entry.get('sector')
                    break

            line = f"<b>#{idx}</b> 🟢 <b>BUY</b> <code>{trade_display}</code>\n"
            if sector:
                line += f"   🏭 Sector: {sector}\n"
            tf = t.get('timeframe')
            if tf:
                line += f"   ⏱️ {tf}\n"
            line += f"   🕒 {ts}\n"
            line += f"   Amount: {amt:.6f}  Entry: {price:.2f}"
            if current_price is not None:
                line += f"  Current: {current_price:.2f}"
            line += "\n"
            line += f"   Fee: {fee_str}\n"
            # Add position value in base currency
            value = amt * (current_price if current_price is not None else price)
            line += f"   Value: {value:,.2f} {self.engine.base_currency}\n"
            # SL/TP
            if sl is not None:
                line += f"   🛑 Stop: {sl:.2f}"
            if tp is not None:
                line += f"  🎯 Target: {tp:.2f}"
            if sl is not None or tp is not None:
                line += "\n"

            pnl = t['unrealized_pnl']
            pnl_pct = t['unrealized_pnl_pct']
            pnl_sign = "+" if pnl >= 0 else ""
            pnl_pct_sign = "+" if pnl_pct >= 0 else ""
            line += f"   Unrealized P&L: {pnl_sign}${pnl:,.2f} ({pnl_pct_sign}{pnl_pct:.2f}%)"
            # Entry order type
            entry_order_type = pos.get('entry_order_type')
            if entry_order_type:
                line += f"\n   📝 Entry Type: {entry_order_type}"
            # Trailing stop details
            if pos.get('trailing_stop'):
                ts_dist = pos.get('trailing_stop_distance_pct')
                ts_act = pos.get('trailing_stop_activation_pct')
                line += f"\n   🚶 Trailing Stop: enabled"
                if ts_dist is not None:
                    line += f" (distance: {ts_dist*100:.2f}%)"
                if ts_act is not None:
                    line += f" [activates at +{ts_act*100:.2f}%]"
            # Max hold time remaining
            max_hold = pos.get('max_hold_time_seconds')
            if max_hold is not None and max_hold > 0:
                entry_ts = pos.get('timestamp', 0) / 1000.0
                elapsed = time.time() - entry_ts if entry_ts > 0 else 0
                remaining = max(0, max_hold - elapsed)
                if remaining > 0:
                    line += f"\n   ⏰ Max Hold: {remaining/60:.0f} min remaining"
                else:
                    line += f"\n   ⏰ Max Hold: EXPIRED"
            # Native stop order info
            sl_order_id = pos.get('stop_loss_order_id')
            if sl_order_id:
                sl_ot = pos.get('stop_loss_order_type', 'stop')
                sl_price = pos.get('stop_loss')
                line += f"\n   🛑 Native Stop: {sl_ot}"
                if sl_price is not None:
                    line += f" @ ${sl_price:.4f}"
            # Native take-profit order info
            tp_order_id = pos.get('take_profit_order_id')
            if tp_order_id:
                tp_price = pos.get('take_profit')
                line += f"\n   🎯 Native TP: limit"
                if tp_price is not None:
                    line += f" @ ${tp_price:.4f}"

            msg += line + "\n\n"

        # --- Queued Orders ---
        if queued_orders:
            msg += "\n<b>⏳ Queued Orders</b>\n\n"
            try:
                q_names = await asyncio.wait_for(
                    asyncio.gather(*[self.engine._market_data_manager.get_stock_name(q['symbol']) for q in queued_orders]),
                    timeout=15.0
                )
            except asyncio.TimeoutError:
                logger.warning("get_stock_name timed out for queued orders")
                q_names = [q['symbol'] for q in queued_orders]
            for idx, q in enumerate(queued_orders, start=1):
                sym = q['symbol']
                side = q['side']
                side_emoji = "🟢" if side == "buy" else "🔴"
                side_label = "BUY" if side == "buy" else "SELL"
                q_tf = q.get('timeframe')
                q_name = q_names[idx-1]
                q_display = self.engine._format_symbol_display(sym, q_name, q_tf)
                original_amount = q.get('original_amount', q['amount'])
                filled_qty = q.get('filled_qty', 0.0)
                limit_price = q.get('limit_price')
                queued_at = q.get('queued_at', 0)
                ts = datetime.fromtimestamp(queued_at).strftime('%Y-%m-%d %H:%M:%S') if queued_at else "?"
                exit_reason = q.get('exit_reason')  # for sells

                # --- Use batched quote ---
                base_sym = sym.split("/")[0]
                ticker = batch_quotes.get(base_sym)
                current_price = ticker.get('last') if ticker else None
                sector = None
                for entry in self.engine.current_symbols:
                    if entry['symbol'] == sym:
                        sector = entry.get('sector')
                        break

                # Status
                if filled_qty > 0 and filled_qty < original_amount:
                    status = f"⏳ Partially filled ({filled_qty:.6f}/{original_amount:.6f})"
                else:
                    status = "⏳ Waiting"

                line = f"<b>#Q{idx}</b> {side_emoji} <b>{side_label}</b> <code>{q_display}</code>\n"
                if sector:
                    line += f"   🏭 Sector: {sector}\n"
                if q_tf:
                    line += f"   ⏱️ {q_tf}\n"
                line += f"   🕒 Queued: {ts}\n"
                line += f"   Amount: {original_amount:.6f}"
                if limit_price is not None:
                    line += f"  Limit: {limit_price:.2f}"
                if current_price is not None:
                    line += f"  Current: {current_price:.2f}"
                    if limit_price is not None and current_price > 0:
                        diff_pct = (current_price - limit_price) / limit_price * 100
                        line += f"  ({diff_pct:+.2f}%)"
                line += "\n"
                if exit_reason:
                    line += f"   Reason: {exit_reason}\n"
                line += f"   Status: {status}\n"
                # Order type
                order_type = q.get('order_type', 'market')
                line += f"   Type: {order_type}\n"
                # Stop price / trail offset
                stop_price = q.get('stop_price')
                if stop_price is not None:
                    line += f"   Stop: ${stop_price:.2f}\n"
                trail_offset = q.get('trail_offset')
                if trail_offset is not None:
                    line += f"   Trail: ${trail_offset:.2f}\n"
                # Exit order label
                if q.get('is_exit_order'):
                    line += f"   🎯 Exit order\n"

                msg += line + "\n"

        await self._send_long_reply(update, msg, parse_mode='HTML', reply_markup=self.keyboard)

    async def cmd_performance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            return
        """Show performance summary grouped by symbol and timeframe."""
        # Check if there are any closed sell trades at all
        closed_sells = [t for t in self.engine.trade_history if t.get("side") == "sell"]
        if not closed_sells:
            await update.message.reply_text(
                "🚀 No closed sell trades yet.", reply_markup=self.keyboard
            )
            return

        try:
            perf = await asyncio.wait_for(
                self.engine.get_performance_summary(),
                timeout=15.0
            )
        except asyncio.TimeoutError:
            await update.message.reply_text("⚠️ Fetching performance summary timed out.", reply_markup=self.keyboard)
            return
        except Exception as e:
            logger.error(f"Failed to get performance summary: {e}", exc_info=True)
            msg = "⚠️ Could not retrieve performance summary. Please try again later."
            await self._send_long_reply(update, msg, parse_mode='HTML', reply_markup=self.keyboard)
            return

        try:
            rows = perf.get("rows", [])
            total = perf.get("total", {})

            if not rows:
                await update.message.reply_text(
                    "🚀 No closed sell trades yet.", reply_markup=self.keyboard
                )
                return

            msg = "<b>🚀 Performance by Symbol</b>\n\n"
            try:
                perf_names = await asyncio.wait_for(
                    asyncio.gather(*[self.engine._market_data_manager.get_stock_name(r["symbol"]) for r in rows]),
                    timeout=15.0
                )
            except asyncio.TimeoutError:
                logger.warning("get_stock_name timed out for performance rows")
                perf_names = [r["symbol"] for r in rows]
            for i, r in enumerate(rows):
                symbol = r["symbol"]
                tf = r.get("timeframe") or "—"
                perf_name = perf_names[i]
                perf_display = self.engine._format_symbol_display(symbol, perf_name, tf)
                trades = r["trade_count"]
                profit = r["profit"]
                profit_pct = r["profit_pct"]
                win_rate = r["win_rate"]

                profit_emoji = "📈" if profit >= 0 else "📉"
                profit_sign = "+" if profit >= 0 else ""
                msg += (
                    f"<b>{perf_display}</b>\n"
                    f"  Trades: {trades}  |  {profit_emoji} {profit_sign}{profit:.4f} ({profit_sign}{profit_pct:.2f}%)\n"
                    f"  Win Rate: {win_rate:.1f}%\n\n"
                )

            if total:
                t = total
                t_profit = t["profit"]
                t_sign = "+" if t_profit >= 0 else ""
                t_emoji = "📈" if t_profit >= 0 else "📉"
                msg += (
                    f"<b>── TOTAL ──</b>\n"
                    f"  Trades: {t['trade_count']}  |  {t_emoji} {t_sign}{t_profit:.4f} ({t_sign}{t['profit_pct']:.2f}%)\n"
                    f"  Win Rate: {t['win_rate']:.1f}%"
                )
        except Exception as e:
            logger.error(f"Failed to get performance summary: {e}", exc_info=True)
            msg = "⚠️ Could not retrieve performance summary. Please try again later."

        await self._send_long_reply(update, msg, parse_mode='HTML', reply_markup=self.keyboard)

    async def cmd_news_search(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            return
        """Show recent news for a specific symbol (e.g., /news AAPL)."""
        if not context.args:
            await update.message.reply_text(
                "Usage: /news <symbol>\nExample: /news AAPL",
                reply_markup=self.keyboard,
            )
            return

        symbol = context.args[0].upper()
        # Remove any trailing "/USD" if user typed a pair
        if "/" in symbol:
            symbol = symbol.split("/")[0]

        try:
            articles = await asyncio.wait_for(
                asyncio.to_thread(get_news_for_symbol, symbol, max_age_seconds=settings.NEWS_CACHE_TTL_SECONDS),
                timeout=15.0
            )
        except asyncio.TimeoutError:
            await update.message.reply_text("⚠️ News fetch timed out.", reply_markup=self.keyboard)
            return
        if not articles:
            await update.message.reply_text(f"No recent news for {symbol}.", reply_markup=self.keyboard)
            return

        try:
            formatted = await asyncio.wait_for(asyncio.to_thread(_format_news_for_prompt, articles), timeout=15.0)
        except asyncio.TimeoutError:
            logger.warning("_format_news_for_prompt timed out")
            await update.message.reply_text("⚠️ News formatting timed out.", reply_markup=self.keyboard)
            return
        msg = f"*{symbol}*\n{formatted}"
        # Send as plain text to avoid Markdown parsing errors
        await self._send_long_reply(update, msg, parse_mode=None, reply_markup=self.keyboard)

    async def cmd_risk(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            return
        try:
            metrics = await asyncio.wait_for(
                self.engine.event_bus.request("get_risk_metrics"),
                timeout=15.0
            )
        except asyncio.TimeoutError:
            await update.message.reply_text("⚠️ Fetching risk metrics timed out.", reply_markup=self.keyboard)
            return
        except Exception as e:
            logger.error(f"Failed to get risk metrics: {e}", exc_info=True)
            await update.message.reply_text("⚠️ Could not retrieve risk metrics.", reply_markup=self.keyboard)
            return

        pf = metrics['profit_factor']
        pf_str = f"{pf:.2f}" if pf != float('inf') else "∞"
        msg = (
            f"<b>⚠️ Risk Metrics</b>\n\n"
            f"<b>Portfolio</b>\n"
            f"💰 Balance: {metrics['current_balance']:.2f} {metrics['base_currency']}\n"
            f"🏦 Initial: {metrics['initial_balance']:.2f} {metrics['base_currency']}\n"
            f"📊 P&L: {metrics['total_pnl']:.2f} ({metrics['total_pnl_pct']:.2f}%)\n"
            f"📉 Max Drawdown: {metrics['max_drawdown_pct']:.2f}%\n\n"
            f"<b>Positions</b>\n"
            f"📈 Open: {metrics['open_positions_count']}\n"
            f"💼 Exposure: {metrics['total_exposure']:.2f} {metrics['base_currency']}\n"
            f"🔝 Largest Position: {metrics['largest_position_exposure_pct']:.1f}% of portfolio\n"
            f"⛔ Total Stop Risk: {metrics['total_stop_loss_risk']:.2f} {metrics['base_currency']}\n\n"
            f"<b>Trade Stats</b>\n"
            f"📋 Total Trades: {metrics['total_trades']}\n"
            f"🏆 Win Rate: {metrics['win_rate']:.1f}%\n"
            f"📊 Profit Factor: {pf_str}\n"
            f"🟢 Avg Win: {metrics['avg_win']:.2f}  🔴 Avg Loss: {metrics['avg_loss']:.2f}"
        )
        await self._send_long_reply(update, msg, parse_mode='HTML', reply_markup=self.keyboard)

    async def cmd_force_reeval(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            return
        self.engine.trigger_symbol_reevaluation(force=True)
        await update.message.reply_text("🔄 Forced symbol re-evaluation triggered.", reply_markup=self.keyboard)

    async def cmd_backfill(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            return
        asyncio.create_task(self.engine.force_download_all_assets())
        await update.message.reply_text("⬇️ Force backfill of all discovered symbols triggered.", reply_markup=self.keyboard)

    async def cmd_signals(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            return
        limit = 5
        if context.args:
            try:
                limit = int(context.args[0])
            except ValueError:
                await update.message.reply_text("Usage: /signals <limit> (e.g., /signals 10)", reply_markup=self.keyboard)
                return

        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(get_signals, limit, 0),
                timeout=15.0
            )
        except asyncio.TimeoutError:
            await update.message.reply_text("⚠️ Signal fetch timed out.", reply_markup=self.keyboard)
            return
        signals = result.get("signals", [])

        if not signals:
            await update.message.reply_text("📡 No recent signals found.", reply_markup=self.keyboard)
            return

        msg = f"<b>📡 Latest {len(signals)} Signals</b>\n\n"
        for s in signals:
            symbol = s.get("symbol", "")
            display_symbol = s.get("display_symbol", symbol)
            action = s.get("action", "")
            confidence = s.get("confidence")
            reasoning = s.get("reasoning", "")
            timestamp = s.get("timestamp")
            ts_str = datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S') if timestamp else "?"

            action_emoji = "🟢" if action == "BUY" else "🔴" if action == "SELL" else "⚪"
            conf_str = f" ({confidence:.0f}%)" if confidence is not None else ""

            msg += f"<b>{action_emoji} {action}</b> <code>{display_symbol}</code>{conf_str}\n"
            msg += f"   🕒 {ts_str}\n"
            if reasoning:
                if len(reasoning) > 200:
                    reasoning = reasoning[:197] + "..."
                msg += f"   📝 {html.escape(reasoning)}\n"
            msg += "\n"

        await self._send_long_reply(update, msg, parse_mode='HTML', reply_markup=self.keyboard)

    async def cmd_market(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            return
        try:
            raw = await asyncio.wait_for(asyncio.to_thread(self.redis.get, "market:status"), timeout=5.0)
            if not raw:
                await update.message.reply_text("Market data not available yet.", reply_markup=self.keyboard)
                return
            data = json.loads(raw)
        except asyncio.TimeoutError:
            logger.warning("Redis get timed out for market:status")
            await update.message.reply_text("⚠️ Market status fetch timed out.", reply_markup=self.keyboard)
            return
        except Exception as e:
            logger.error(f"Failed to get market status: {e}", exc_info=True)
            await update.message.reply_text("⚠️ Could not retrieve market status.", reply_markup=self.keyboard)
            return

        msg = "<b>🌐 Market Status</b>\n\n"
        if data.get("market_breadth"):
            mb = data["market_breadth"]
            msg += f"<b>📊 Market Breadth (candidates):</b> {mb['positive_pct']}% positive ({mb['positive_count']}/{mb['total_count']})\n"
        if data.get("full_market_breadth"):
            fmb = data["full_market_breadth"]
            msg += f"<b>🌐 Full Market Breadth:</b> {fmb['positive_pct']}% positive ({fmb['positive_count']}/{fmb['total_count']})\n"
        if data.get("spy_price") is not None:
            msg += f"<b>📈 Benchmark Price:</b> {data['spy_price']:.2f}\n"
        await self._send_long_reply(update, msg, parse_mode='HTML', reply_markup=self.keyboard)

    async def cmd_news(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            return
        """Show LLM-generated news summaries for all tracked symbols (same as web card)."""
        try:
            symbols = self.engine.current_symbols
            if not symbols:
                await update.message.reply_text("No symbols currently tracked.", reply_markup=self.keyboard)
                return

            await update.message.reply_text("Generating news summaries...", reply_markup=self.keyboard)
            async def _process_news_entry(entry):
                symbol = entry["symbol"]
                news_tf = entry.get("timeframe")
                try:
                    news_name = await asyncio.wait_for(
                        self.engine._market_data_manager.get_stock_name(symbol),
                        timeout=10.0
                    )
                except asyncio.TimeoutError:
                    news_name = symbol
                news_display = self.engine._format_symbol_display(symbol, news_name, news_tf)
                base_symbol = symbol.split("/")[0] if "/" in symbol else symbol
                articles = await asyncio.wait_for(
                    asyncio.to_thread(get_news_for_symbol, base_symbol, max_age_seconds=settings.NEWS_CACHE_TTL_SECONDS),
                    timeout=15.0
                )
                if not articles:
                    return None
                try:
                    news_data = await asyncio.wait_for(
                        asyncio.to_thread(get_cached_news_summary, symbol),
                        timeout=30.0
                    )
                    summary_text = html.escape(news_data["summary"])
                    provider = news_data.get("provider", "")
                    model = news_data.get("model", "")
                except asyncio.TimeoutError:
                    logger.warning(f"get_cached_news_summary timed out for {symbol}")
                    summary_text = ""
                    provider = ""
                    model = ""
                except Exception:
                    summary_text = ""
                    provider = ""
                    model = ""
                if not summary_text or summary_text == "Could not generate summary.":
                    return None
                msg_line = f"<b>{news_display}</b>\n{summary_text}"
                if provider and model:
                    msg_line += f"\n⚡ Generated by {model} ({provider})"
                return msg_line

            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*[_process_news_entry(entry) for entry in symbols]),
                    timeout=60.0
                )
            except asyncio.TimeoutError:
                await update.message.reply_text("⚠️ News generation timed out.", reply_markup=self.keyboard)
                return
            messages = [msg for msg in results if msg]

            if not messages:
                await update.message.reply_text("No news available for tracked symbols.", reply_markup=self.keyboard)
                return

            full_text = "\n\n".join(messages)

            await self._send_long_reply(update, full_text, parse_mode='HTML')
        except Exception as e:
            logger.error(f"Failed to generate news summaries: {e}", exc_info=True)
            await update.message.reply_text("⚠️ Could not retrieve news.", reply_markup=self.keyboard)

    async def cmd_news_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            return
        """Show news article counts for tracked symbols."""
        try:
            symbols = self.engine.current_symbols
            if not symbols:
                await update.message.reply_text("No symbols currently tracked.", reply_markup=self.keyboard)
                return

            msg = "<b>📰 News Article Counts</b>\n\n"
            async def _get_news_count(entry):
                symbol = entry["symbol"]
                ns_tf = entry.get("timeframe")
                try:
                    ns_name = await asyncio.wait_for(
                        self.engine._market_data_manager.get_stock_name(symbol),
                        timeout=10.0
                    )
                except asyncio.TimeoutError:
                    ns_name = symbol
                ns_display = self.engine._format_symbol_display(symbol, ns_name, ns_tf)
                base_symbol = symbol.split("/")[0] if "/" in symbol else symbol
                articles = await asyncio.wait_for(
                    asyncio.to_thread(get_news_for_symbol, base_symbol, max_age_seconds=settings.NEWS_CACHE_TTL_SECONDS),
                    timeout=15.0
                )
                return f"<b>{ns_display}</b>: {len(articles)} articles\n"

            try:
                msg_lines = await asyncio.wait_for(
                    asyncio.gather(*[_get_news_count(entry) for entry in symbols]),
                    timeout=60.0
                )
            except asyncio.TimeoutError:
                await update.message.reply_text("⚠️ News status fetch timed out.", reply_markup=self.keyboard)
                return
            msg += "".join(msg_lines)
            await self._send_long_reply(update, msg, parse_mode='HTML', reply_markup=self.keyboard)
        except Exception as e:
            logger.error(f"Failed to get news status: {e}", exc_info=True)
            await update.message.reply_text("⚠️ Could not retrieve news status.", reply_markup=self.keyboard)

    async def cmd_sell(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            return
        """Sell all open positions, or a specific one by trade ID (e.g., /sell 2)."""
        try:
            is_open = await asyncio.wait_for(
                self.engine._is_market_open(),
                timeout=10.0
            )
        except asyncio.TimeoutError:
            await update.message.reply_text("⚠️ Market status check timed out.", reply_markup=self.keyboard)
            return
        if not is_open:
            await update.message.reply_text(
                "⏸️ Cannot sell: market is currently closed.",
                reply_markup=self.keyboard
            )
            return

        try:
            open_trades = await asyncio.wait_for(
                self.engine.event_bus.request("get_open_trades"),
                timeout=15.0
            )
        except asyncio.TimeoutError:
            await update.message.reply_text("⚠️ Fetching open trades timed out.", reply_markup=self.keyboard)
            return
        except Exception as e:
            logger.error(f"Failed to get open trades: {e}", exc_info=True)
            await update.message.reply_text("⚠️ Could not retrieve open trades.", reply_markup=self.keyboard)
            return

        if not open_trades:
            await update.message.reply_text("📈 No open trades to sell.", reply_markup=self.keyboard)
            return

        if context.args:
            # Sell a specific trade by its symbol (e.g., /sell AAPL)
            symbol_arg = context.args[0].upper()
            if "/" in symbol_arg:
                symbol_arg = symbol_arg.split("/")[0]

            target_trade = next((t for t in open_trades if t['symbol'].split("/")[0] == symbol_arg), None)
            if not target_trade:
                await update.message.reply_text(f"❌ No open trade found for {symbol_arg}. Use /trades to see open positions.", reply_markup=self.keyboard)
                return

            symbol = target_trade['symbol']
            sell_tf = target_trade.get('timeframe')
            try:
                sell_name = await asyncio.wait_for(
                    self.engine._market_data_manager.get_stock_name(symbol),
                    timeout=10.0
                )
            except asyncio.TimeoutError:
                sell_name = symbol
            sell_display = self.engine._format_symbol_display(symbol, sell_name, sell_tf)
            await update.message.reply_text(f"🔄 Selling {sell_display}...", reply_markup=self.keyboard)
            try:
                await asyncio.wait_for(self.engine.sell_position(symbol), timeout=30.0)
            except asyncio.TimeoutError:
                await update.message.reply_text("⚠️ Sell order timed out.", reply_markup=self.keyboard)
                return
            await update.message.reply_text(f"✅ Sell order placed for {sell_display}.", reply_markup=self.keyboard)
        else:
            # Sell all open positions
            count = len(open_trades)
            await update.message.reply_text(f"🔄 Selling all {count} open positions...", reply_markup=self.keyboard)
            try:
                await asyncio.wait_for(self.engine.sell_all_positions(), timeout=60.0)
            except asyncio.TimeoutError:
                await update.message.reply_text("⚠️ Sell all orders timed out.", reply_markup=self.keyboard)
                return
            await update.message.reply_text(f"✅ Sell orders placed for all {count} positions.", reply_markup=self.keyboard)

    async def cmd_profit(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            return
        try:
            summary = await asyncio.wait_for(
                self.engine.event_bus.request("get_profit_summary"),
                timeout=15.0
            )
            base_currency = summary.get('base_currency', '')
            pnl = summary['total_pnl']
            pnl_pct = summary['pnl_percent']
            pnl_emoji = "📈" if pnl >= 0 else "📉"
            pnl_sign = "+" if pnl >= 0 else ""

            msg = "<b>💰 Profit Summary</b>\n\n"
            msg += f"💱 Base Currency: {base_currency}\n\n"
            msg += f"💵 Initial Balance:  {summary['initial_balance']:,.2f}\n"
            msg += f"🏦 Current Balance:  {summary['current_balance']:,.2f}\n"

            # Effective balance (cash not tied up in pending buys)
            eff_bal = summary.get('effective_balance', summary['current_balance'])
            if eff_bal != summary['current_balance']:
                msg += f"💳 Available Cash:   {eff_bal:,.2f}  (balance − pending buys)\n"
            else:
                msg += f"💳 Available Cash:   {eff_bal:,.2f}\n"

            msg += f"📊 Open Positions:   {summary['open_value']:,.2f}\n"

            # Queued orders
            q_buy_cnt = summary.get('queued_buy_count', 0)
            q_sell_cnt = summary.get('queued_sell_count', 0)
            if q_buy_cnt > 0 or q_sell_cnt > 0:
                msg += "\n<b>⏳ Queued Orders</b>\n"
                if q_buy_cnt > 0:
                    q_buy_quote = summary.get('queued_buy_quote_total', 0.0)
                    msg += f"  🟢 Pending Buys: {q_buy_cnt} order(s), {q_buy_quote:,.2f} {base_currency} committed\n"
                if q_sell_cnt > 0:
                    q_sell_base = summary.get('queued_sell_base_total', 0.0)
                    q_sell_val = summary.get('queued_sell_value', 0.0)
                    msg += f"  🔴 Pending Sells: {q_sell_cnt} order(s), {q_sell_base:,.2f} base units"
                    if q_sell_val > 0:
                        msg += f" (~{q_sell_val:,.2f} {base_currency})"
                    msg += "\n"

            total_wallet = summary['current_balance'] + summary['open_value']
            msg += f"💼 Total Wallet:     {total_wallet:,.2f}\n"
            msg += f"🧾 Fees Paid:        {summary['total_fees']:,.2f}\n"
            msg += f"{pnl_emoji} Total P&L:         {pnl_sign}{pnl:,.2f}  ({pnl_sign}{pnl_pct:.2f}%)\n"
            wins = summary.get('wins', 0)
            losses = summary.get('losses', 0)
            win_rate = summary.get('win_rate', 0.0)
            msg += f"\n🏆 Wins: {wins}  💔 Losses: {losses}\n"
            msg += f"📊 Win Rate: {win_rate*100:.1f}%\n"
        except asyncio.TimeoutError:
            await update.message.reply_text("⚠️ Fetching profit summary timed out.", reply_markup=self.keyboard)
            return
        except Exception as e:
            logger.error(f"Failed to get profit summary: {e}", exc_info=True)
            msg = "⚠️ Could not retrieve profit summary. Please try again later."

        await self._send_long_reply(update, msg, parse_mode='HTML', reply_markup=self.keyboard)

    def _write_notification_log(self, log_path: Path, summary: dict):
        """Write a summary dict as a JSON line to log_path, rotating if > MAX_LOG_SIZE."""
        with TelegramBot._log_lock:
            # Rotate if file exists and is too large
            if log_path.exists() and log_path.stat().st_size >= self.MAX_LOG_SIZE:
                # Remove oldest backup if it exists
                oldest = log_path.with_suffix(f".jsonl.{self.MAX_LOG_BACKUPS}")
                if oldest.exists():
                    oldest.unlink()
                # Shift existing backups
                for i in range(self.MAX_LOG_BACKUPS - 1, 0, -1):
                    src = log_path.with_suffix(f".jsonl.{i}")
                    dst = log_path.with_suffix(f".jsonl.{i+1}")
                    if src.exists():
                        src.rename(dst)
                # Rename current log to .1
                log_path.rename(log_path.with_suffix(".jsonl.1"))
            # Write the new entry
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(summary, ensure_ascii=False) + "\n")

    @staticmethod
    def _compact_summary(summary: dict) -> dict:
        """Return a compacted version of the summary dict to keep the notification log small."""
        compact = {}
        for key, value in summary.items():
            # If symbols is a list of dicts, keep only the symbols
            if key == "symbols" and isinstance(value, list):
                if value and isinstance(value[0], dict):
                    value = [c.get("symbol", c) for c in value]
            # Compact sentiment to just the numeric compound value (e.g., 0.05 or -0.05)
            elif key == "sentiment" and isinstance(value, dict):
                value = round(value.get("avg_compound", 0), 2)
            # Compact backtest to a short win/loss summary
            elif key == "backtest" and isinstance(value, dict):
                value = TelegramBot._compact_backtest(value)
            compact[key] = value
        return compact

    @staticmethod
    def _compact_backtest(stats: dict) -> str:
        """Format a short win/loss summary directly from the backtest stats dict."""
        if not isinstance(stats, dict):
            # Fallback for non-dict inputs (e.g., if a string is accidentally passed)
            text = str(stats)
            if len(text) > 50:
                text = text[:47] + "..."
            return text

        total_trades = stats.get("total_trades", 0)
        win_rate = stats.get("win_rate", 0.0)
        wins = stats.get("wins", 0)
        losses = stats.get("losses", 0)
        
        timeframe = stats.get("timeframe", "")
        prefix = f"{timeframe}: " if timeframe else ""
        
        return f"{prefix}{total_trades} trades, {win_rate*100:.0f}% win ({wins}W/{losses}L)"

    async def send_notification(self, message: str, summary: dict = None, disable_notification: bool = True):
        """Send a notification to the stored chat ID and optionally log a summary."""
        # Capture full stacktrace for error notifications if an exception is active.
        # This must be done at the very start, before any try/except blocks in this
        # method could overwrite sys.exc_info().
        if summary and summary.get("action") == "ERROR" and "traceback" not in summary:
            exc_info = sys.exc_info()
            if exc_info[0] is not None:
                summary["traceback"] = "".join(traceback.format_exception(*exc_info))

        try:
            chat_id = await asyncio.wait_for(
                asyncio.to_thread(get_telegram_chat_id),
                timeout=5.0
            )
        except asyncio.TimeoutError:
            logger.warning("get_telegram_chat_id timed out")
            return
        logger.info(f"send_notification called, chat_id={chat_id}, message={message[:50]}...")
        if not chat_id:
            logger.warning("No chat_id stored – cannot send notification. Use /start first.")
            return

        # --- Verbosity filter ---
        verbosity = settings.NOTIFICATION_VERBOSITY
        action = summary.get("action", "") if summary else ""

        if action in ("PAUSE", "RESUME"):
            should_send = True
        elif verbosity == "all":
            should_send = True
        elif verbosity == "none":
            should_send = False
        elif verbosity == "errors_only":
            should_send = (action == "ERROR" or not action)
        elif verbosity == "trades_only":
            should_send = (action in ("BUY", "SELL") or not action)
        else:
            should_send = False

        if should_send:
            # --- Rate limiting ---
            is_critical = action in ("BUY", "SELL", "ERROR")
            now = time.time()
            # Remove timestamps older than 60 seconds
            while self._notification_timestamps and self._notification_timestamps[0] <= now - 60:
                self._notification_timestamps.popleft()

            rate_limited = len(self._notification_timestamps) >= self._max_notifications_per_minute
            if rate_limited:
                if is_critical:
                    logger.critical(f"Telegram rate limit exceeded, dropping critical notification: {action}")
                else:
                    logger.warning(f"Telegram rate limit exceeded, dropping non-critical notification: {action}")
            else:
                self._notification_timestamps.append(now)

                if summary and summary.get("model_type"):
                    model = summary["model_type"]
                    emoji = "🧠" if model == "mind" else "⚡"
                    # Use actual provider/model if provided, otherwise fall back to settings
                    llm_provider = summary.get("llm_provider")
                    llm_model = summary.get("llm_model")
                    if llm_provider and llm_model:
                        provider_name = llm_provider
                        model_name = llm_model
                    else:
                        if model == "mind":
                            provider_name = settings.LLM_MIND_PROVIDER or settings.LLM_PROVIDER
                            if provider_name == "ollama":
                                model_name = settings.OLLAMA_MIND_MODEL
                            elif provider_name == "g4f":
                                from src.llm.g4f_client import _get_g4f_models
                                model_name = _get_g4f_models("mind")
                            else:
                                model_name = settings.OPENAI_MIND_MODEL
                        elif model == "weak":
                            provider_name = settings.LLM_WEAK_PROVIDER or settings.LLM_PROVIDER
                            if provider_name == "ollama":
                                model_name = settings.OLLAMA_WEAK_MODEL
                            elif provider_name == "g4f":
                                from src.llm.g4f_client import _get_g4f_models
                                model_name = _get_g4f_models("weak")
                            else:
                                model_name = settings.OPENAI_WEAK_MODEL
                        else:
                            provider_name = settings.LLM_ACTUATOR_PROVIDER or settings.LLM_PROVIDER
                            if provider_name == "ollama":
                                model_name = settings.OLLAMA_ACTUATOR_MODEL
                            elif provider_name == "g4f":
                                from src.llm.g4f_client import _get_g4f_models
                                model_name = _get_g4f_models("actuator")
                            else:
                                model_name = settings.OPENAI_ACTUATOR_MODEL
                    message += f"\n{emoji} Generated by {model_name} ({provider_name})"
                # Store message for web interface FIRST (always, regardless of Telegram success)
                try:
                    msg_data = json.dumps({
                        "timestamp": time.time(),
                        "message": message
                    })
                    await asyncio.wait_for(asyncio.to_thread(self.redis.lpush, "web:messages", msg_data), timeout=5.0)
                    await asyncio.wait_for(asyncio.to_thread(self.redis.ltrim, "web:messages", 0, 99), timeout=5.0)
                except Exception as e:
                    logger.warning(f"Failed to store message for web interface: {type(e).__name__}: {e}")

                # --- Determine if notification should be silent ---
                # All notifications are silent by default.
                # Only BUY, SELL, and ERROR actions ring the phone, unless
                # explicitly overridden by the caller via disable_notification=False.
                if action in ("BUY", "SELL", "ERROR"):
                    disable_notification = False

                # Enqueue the message to be sent by the background task
                await self._notification_queue.put({
                    "chat_id": int(chat_id),
                    "message": message,
                    "disable_notification": disable_notification,
                    "is_critical": is_critical
                })
        else:
            logger.info("Notification suppressed by verbosity setting.")

        # --- Log summary to JSONL file (always, if enabled) ---
        if summary is not None and settings.NOTIFICATION_LOG_ENABLED:
            data_dir = Path(settings.DATA_DIR)
            data_dir.mkdir(parents=True, exist_ok=True)
            log_path = data_dir / "notifications.jsonl"

            # Ensure a UTC timestamp is present
            if "timestamp" not in summary:
                summary["timestamp"] = datetime.now(timezone.utc).isoformat()

            # Compact the summary to keep the log small
            summary = self._compact_summary(summary)

            try:
                await asyncio.wait_for(asyncio.to_thread(self._write_notification_log, log_path, summary), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("_write_notification_log timed out")
            except Exception as e:
                logger.warning(f"Failed to write notification log: {type(e).__name__}: {e}")

    async def _process_notification_queue(self):
        """Background task to send Telegram notifications without blocking the event loop."""
        while True:
            payload = await self._notification_queue.get()
            try:
                chat_id = payload["chat_id"]
                message = payload["message"]
                disable_notification = payload["disable_notification"]
                is_critical = payload["is_critical"]
                
                max_retries = 3 if is_critical else 1
                retry_delay = 2.0

                for attempt in range(1, max_retries + 1):
                    try:
                        chunks = self._split_text(message)
                        for chunk in chunks:
                            await asyncio.wait_for(
                                self.app.bot.send_message(
                                    chat_id=chat_id,
                                    text=chunk,
                                    disable_notification=disable_notification,
                                ),
                                timeout=15.0
                            )
                        logger.debug(f"Notification sent successfully (silent={disable_notification}).")
                        break
                    except Exception as e:
                        if attempt < max_retries:
                            logger.warning(f"Failed to send Telegram notification (attempt {attempt}/{max_retries}): {e}. Retrying in {retry_delay}s...")
                            await asyncio.sleep(retry_delay)
                        else:
                            logger.critical(f"Failed to send Telegram notification after {max_retries} attempts: {e}", exc_info=True)
            except Exception as e:
                logger.error(f"Error processing notification queue: {e}", exc_info=True)
            finally:
                self._notification_queue.task_done()

    async def start(self):
        """Start the bot (initialize, start polling, start application)."""
        # Guard against supervisor restarts: if the updater/app is already
        # running, don't try to start it again.
        if self.app.updater and self.app.updater.running:
            logger.info("Telegram bot updater is already running, skipping initialization.")
        else:
            await self.app.initialize()
            await self.app.updater.start_polling()
        if not self.app.running:
            await self.app.start()
        logger.info("Telegram bot started and polling.")
        # Notify the user about the trading mode
        mode = settings.TRADING_MODE.upper()
        try:
            await self.send_notification(
                f"🤖 Bot started in {mode} mode.",
                summary={
                    "action": "INFO",
                    "reason": "Bot started",
                    "mode": mode,
                }
            )
        except Exception as e:
            logger.warning(f"Failed to send startup notification: {type(e).__name__}: {e}")
        self._notification_task = asyncio.create_task(self._process_notification_queue())
        # Keep the task alive until cancelled by the supervisor during shutdown.
        # Without this, the coroutine returns immediately (PTB v20 start methods
        # are non-blocking), causing the supervisor to think the task exited
        # normally and restart it — which fails with "Updater is already running".
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            pass

    async def stop(self):
        """Stop the bot gracefully."""
        if self._notification_task:
            self._notification_task.cancel()
            try:
                await self._notification_task
            except asyncio.CancelledError:
                pass
        await self.app.updater.stop()
        await self.app.stop()
        await self.app.shutdown()
