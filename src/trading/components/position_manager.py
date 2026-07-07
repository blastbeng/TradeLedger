"""Position management component for the TradingEngine.

Handles position-related operations: cost basis computation and portfolio
exposure calculation.
Extracted from TradingEngine to reduce class size and improve maintainability.
"""
import asyncio
import logging
import math
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.config.settings import settings
from src.database import insert_trade
from src.strategies.base import Signal
from src.utils.symbol_utils import is_btp_isin

logger = logging.getLogger(__name__)


class PositionManager:
    """Handles position management operations for the TradingEngine."""

    def __init__(self, engine, event_bus):
        self.engine = engine
        self.event_bus = event_bus
        self.event_bus.subscribe("update_position_params", self.update_position_params)
        self.event_bus.subscribe("compute_portfolio_exposure_summary", self.compute_portfolio_exposure_summary)
        self.event_bus.subscribe("compute_equity_and_drawdown", self.compute_equity_and_drawdown)
        self.event_bus.subscribe("compute_performance_metrics", self.compute_performance_metrics)
        self.event_bus.subscribe("compute_trade_pattern_analysis", self.compute_trade_pattern_analysis)
        self.event_bus.subscribe("get_open_trades", self.get_open_trades)
        self.event_bus.subscribe("get_profit_summary", self.get_profit_summary)
        self.event_bus.subscribe("get_risk_metrics", self.get_risk_metrics)
        self.event_bus.subscribe("reconcile_positions", self.reconcile_positions)

    def ensure_cost_basis(self):
        """If positions lack cost_basis, compute it from amount and price (backward compat)."""
        for sym, pos in self.engine.positions.items():
            if 'cost_basis' not in pos or 'net_base' not in pos:
                # Assume no fees for old positions; cost_basis = amount * price
                pos['cost_basis'] = pos['amount'] * pos['price']
                pos['net_base'] = pos['amount']

        # Pre-set the BTP trailing stop warning flag for loaded positions
        # so we don't spam warnings on every risk check cycle.
        for sym, pos in self.engine.positions.items():
            base = sym.split("/")[0]
            if is_btp_isin(base) and pos.get("trailing_stop") and "_ts_btp_warned" not in pos:
                pos["_ts_btp_warned"] = True

    async def compute_portfolio_exposure_summary(self, base_balance: float) -> Dict[str, float]:
        """Compute portfolio exposure, stop-loss risk, and available capital for the prompt."""
        engine = self.engine
        now = time.time()
        if (
            engine._portfolio_exposure_cache is not None
            and (now - engine._portfolio_exposure_cache_time) < 30
        ):
            # Return cached ticker-dependent values, but recompute available capital
            # from the current cycle_spent (which changes during the cycle).
            # Acquire the lock before copying the cache so the cache read and
            # _cycle_spent read are atomic — prevents a race where another
            # coroutine modifies _cycle_spent between the copy and the lock.
            async with engine._cycle_spent_lock:
                result = dict(engine._portfolio_exposure_cache)
                result["portfolio_available_capital"] = max(0.0, base_balance - engine._cycle_spent)
            return result

        portfolio_total_value = base_balance
        portfolio_exposure = 0.0
        portfolio_stop_risk = 0.0
        pos_tickers = await engine._market_data_manager._get_all_position_tickers()
        for sym, pos in engine.positions.items():
            try:
                t = pos_tickers.get(sym)
                price = t['last'] if t and t.get('last') else 0.0
                pos_value = pos['amount'] * price
                portfolio_exposure += pos_value
                portfolio_total_value += pos_value
                stop_loss = pos.get('stop_loss')
                if stop_loss is not None and price > 0:
                    loss_if_stop = pos_value * (price - stop_loss) / price
                    portfolio_stop_risk += max(0, loss_if_stop)
            except Exception:
                pass
        portfolio_exposure_pct = (portfolio_exposure / portfolio_total_value * 100) if portfolio_total_value > 0 else 0.0
        portfolio_stop_risk_pct = (portfolio_stop_risk / portfolio_total_value * 100) if portfolio_total_value > 0 else 0.0
        async with engine._cycle_spent_lock:
            portfolio_available_capital = max(0.0, base_balance - engine._cycle_spent)
        result = {
            "portfolio_total_value": portfolio_total_value,
            "portfolio_exposure": portfolio_exposure,
            "portfolio_stop_risk": portfolio_stop_risk,
            "portfolio_exposure_pct": portfolio_exposure_pct,
            "portfolio_stop_risk_pct": portfolio_stop_risk_pct,
            "portfolio_available_capital": portfolio_available_capital,
        }
        engine._portfolio_exposure_cache = result
        engine._portfolio_exposure_cache_time = now
        return result

    async def get_profit_summary(self) -> Dict[str, Any]:
        """Return profit/loss summary including queued orders."""
        engine = self.engine
        balance = await asyncio.to_thread(engine.trader.fetch_balance)
        current_balance = balance.get(engine.base_currency, 0.0)

        # --- Early exit: no positions and no queued orders → nothing to compute ---
        if not engine.positions and not engine.queued_orders:
            return {
                "initial_balance": engine.initial_balance,
                "current_balance": current_balance,
                "effective_balance": current_balance,
                "open_value": 0.0,
                "total_pnl": current_balance - engine.initial_balance,
                "pnl_percent": ((current_balance - engine.initial_balance) / engine.initial_balance * 100) if engine.initial_balance else 0.0,
                "total_fees": 0.0,
                "total_fees_display": "0.000000",
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "base_currency": engine.base_currency,
                "queued_buy_count": 0,
                "queued_sell_count": 0,
                "queued_buy_quote_total": 0.0,
                "queued_sell_base_total": 0.0,
                "queued_sell_value": 0.0,
            }

        open_value = 0.0
        pos_tickers = await asyncio.to_thread(engine._market_data_manager._get_all_position_tickers_sync)
        for sym, pos in engine.positions.items():
            try:
                t = pos_tickers.get(sym)
                price = t['last'] if t and t.get('last') else 0.0
                open_value += pos['amount'] * price
            except Exception:
                pass

        # --- Queued orders ---
        queued_buy_count = 0
        queued_sell_count = 0
        queued_buy_quote_total = 0.0
        queued_sell_base_total = 0.0
        queued_sell_value = 0.0

        # Collect symbols for queued sells to fetch prices
        queued_sell_symbols = []
        for q in engine.queued_orders:
            if q['side'] == 'buy':
                queued_buy_count += 1
                # 'amount' is the remaining quote to spend
                queued_buy_quote_total += q.get('amount', 0.0)
            elif q['side'] == 'sell':
                queued_sell_count += 1
                queued_sell_base_total += q.get('amount', 0.0)
                queued_sell_symbols.append(q['symbol'])

        if queued_sell_symbols:
            sell_tickers = await asyncio.to_thread(engine._market_data_manager._get_tickers_for_symbols_sync, queued_sell_symbols)
        else:
            sell_tickers = {}
        for q in engine.queued_orders:
            if q['side'] == 'sell':
                sym = q['symbol']
                t = sell_tickers.get(sym) if sell_tickers else None
                price = t['last'] if t and t.get('last') else 0.0
                queued_sell_value += q.get('amount', 0.0) * price

        effective_balance = current_balance - queued_buy_quote_total

        total_fees = 0.0
        for t in engine.trade_history:
            fee = t.get('fee', {})
            fee_cost = float(fee.get('cost', 0) or 0)
            fee_currency = fee.get('currency', '')
            if fee_cost == 0.0:
                continue
            if fee_currency == engine.base_currency:
                total_fees += fee_cost
            else:
                # fee is in the base symbol (e.g., BTC) → convert using trade price
                price = t.get('price', 0.0)
                total_fees += fee_cost * price
        total_value = current_balance + open_value
        pnl = total_value - engine.initial_balance
        pnl_percent = (pnl / engine.initial_balance * 100) if engine.initial_balance else 0.0

        # Win/Loss stats
        wins = 0
        losses = 0
        for t in engine.trade_history:
            if t.get('side') == 'sell' and 'realized_pnl' in t:
                pnl_val = t['realized_pnl']
                if pnl_val > 0:
                    wins += 1
                elif pnl_val < 0:
                    losses += 1
        total_closed = wins + losses
        win_rate = (wins / total_closed) if total_closed > 0 else 0.0

        return {
            "initial_balance": engine.initial_balance,
            "current_balance": current_balance,
            "effective_balance": effective_balance,
            "open_value": open_value,
            "total_pnl": pnl,
            "pnl_percent": pnl_percent,
            "total_fees": round(total_fees, 6),
            "total_fees_display": f"{total_fees:.6f}",
            "wins": wins,
            "losses": losses,
            "win_rate": round(win_rate, 4),
            "base_currency": engine.base_currency,
            "queued_buy_count": queued_buy_count,
            "queued_sell_count": queued_sell_count,
            "queued_buy_quote_total": queued_buy_quote_total,
            "queued_sell_base_total": queued_sell_base_total,
            "queued_sell_value": queued_sell_value,
        }

    async def get_risk_metrics(self) -> Dict[str, Any]:
        """Return current risk/exposure metrics."""
        engine = self.engine
        balance = await asyncio.to_thread(engine.trader.fetch_balance)
        total_balance = balance.get(engine.base_currency, 0.0)

        pnl = total_balance - engine.initial_balance
        pnl_pct = (pnl / engine.initial_balance * 100) if engine.initial_balance else 0.0

        # Open positions exposure and stop‑loss risk
        exposure = 0.0
        position_exposures = []
        total_stop_risk = 0.0
        pos_tickers = await asyncio.to_thread(engine._market_data_manager._get_all_position_tickers_sync)
        for sym, pos in engine.positions.items():
            try:
                t = pos_tickers.get(sym)
                price = t['last'] if t and t.get('last') else 0.0
                pos_value = pos['amount'] * price
                exposure += pos_value
                position_exposures.append(pos_value)
                stop_loss = pos.get('stop_loss')
                if stop_loss is not None and price > 0:
                    loss_if_stop = pos_value * (price - stop_loss) / price
                    total_stop_risk += loss_if_stop
            except Exception:
                pass

        total_portfolio_value = total_balance + exposure
        largest_position_exposure_pct = (
            (max(position_exposures) / total_portfolio_value * 100)
            if position_exposures and total_portfolio_value > 0
            else 0.0
        )

        # Drawdown from performance metrics
        perf = await engine.event_bus.request("compute_performance_metrics")
        max_drawdown_pct = perf.get('equity_curve', {}).get('drawdown_pct', 0.0)

        # Trade statistics
        wins = []
        losses = []
        for t in engine.trade_history:
            if t.get('side') == 'sell' and 'realized_pnl' in t:
                pnl_val = t['realized_pnl']
                if pnl_val > 0:
                    wins.append(pnl_val)
                elif pnl_val < 0:
                    losses.append(abs(pnl_val))
        total_trades = len(wins) + len(losses)
        win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0.0
        gross_profit = sum(wins)
        gross_loss = sum(losses)
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0
        avg_win = (gross_profit / len(wins)) if wins else 0.0
        avg_loss = (gross_loss / len(losses)) if losses else 0.0

        # Sanitize non-finite floats for JSON serialization
        def _sanitize_float(value):
            if isinstance(value, float) and not math.isfinite(value):
                return None
            return value

        profit_factor = _sanitize_float(profit_factor)
        avg_win = _sanitize_float(avg_win)
        avg_loss = _sanitize_float(avg_loss)

        return {
            'current_balance': total_balance,
            'initial_balance': engine.initial_balance,
            'total_pnl': pnl,
            'total_pnl_pct': pnl_pct,
            'open_positions_count': len(engine.positions),
            'total_exposure': exposure,
            'base_currency': engine.base_currency,
            'max_drawdown_pct': max_drawdown_pct,
            'largest_position_exposure_pct': largest_position_exposure_pct,
            'total_stop_loss_risk': total_stop_risk,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_trades': total_trades,
        }

    async def get_open_trades(self) -> List[Dict[str, Any]]:
        """Return current open positions as trade-like dicts with unrealized P&L."""
        engine = self.engine
        open_trades = []
        pos_tickers = await asyncio.to_thread(engine._market_data_manager._get_all_position_tickers_sync)
        for symbol, pos in engine.positions.items():
            # Skip invalid positions (zero amount or zero price)
            if pos.get("amount", 0) <= 0 or pos.get("price", 0) <= 0:
                continue
            try:
                t = pos_tickers.get(symbol)
                current_price = t['last'] if t and t.get('last') else pos['price']
            except Exception:
                current_price = pos['price']  # fallback to entry price

            entry_price = pos['price']
            amount = pos['amount']
            cost_basis = pos.get('cost_basis', amount * entry_price)
            unrealized_pnl = (current_price - entry_price) * amount
            unrealized_pnl_pct = (unrealized_pnl / cost_basis * 100) if cost_basis > 0 else 0.0

            # Try to get fee from the most recent buy trade for this symbol
            fee = {}
            for t in reversed(engine.trade_history):
                if t['symbol'] == symbol and t['side'] == 'buy':
                    fee = t.get('fee', {})
                    break

            open_trades.append({
                'symbol': symbol,
                'timeframe': pos.get('timeframe'),
                'side': 'buy',
                'amount': amount,
                'price': entry_price,
                'timestamp': pos.get('timestamp', 0),
                'fee': fee,
                'unrealized_pnl': unrealized_pnl,
                'unrealized_pnl_pct': unrealized_pnl_pct,
                'cost_basis': cost_basis,
            })
        return open_trades

    def compute_equity_and_drawdown(self, trades_snapshot: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute current equity, peak, and drawdown percentage.
        
        Returns a dict with keys: current_realized_equity, unrealized_pnl,
        current_equity, peak, drawdown_pct.
        """
        engine = self.engine
        equity_series = []
        running_equity = engine.initial_balance + engine._realized_pnl_offset
        for trade in trades_snapshot:
            if trade.get("side") == "sell":
                running_equity += trade.get("realized_pnl", 0.0)
            equity_series.append(running_equity)
        peak = max(equity_series) if equity_series else engine.initial_balance

        current_realized_equity = equity_series[-1] if equity_series else engine.initial_balance
        unrealized_pnl = 0.0
        try:
            pos_tickers = engine._market_data_manager._get_all_position_tickers_sync()
            for sym, pos in engine.positions.items():
                t = pos_tickers.get(sym)
                if t and t.get('last'):
                    unrealized_pnl += (t['last'] - pos['price']) * pos['amount']
        except Exception:
            pass
        current_equity = current_realized_equity + unrealized_pnl
        if current_equity > peak:
            peak = current_equity
        drawdown_pct = ((peak - current_equity) / peak * 100) if peak > 0 else 0.0

        return {
            "current_realized_equity": current_realized_equity,
            "unrealized_pnl": unrealized_pnl,
            "current_equity": current_equity,
            "peak": peak,
            "drawdown_pct": drawdown_pct,
        }

    def compute_performance_metrics(self) -> Dict[str, Any]:
        """Analyze trade history to produce per-symbol and per-strategy performance summaries."""
        engine = self.engine
        now = time.time()
        if (
            engine._trade_history_version == engine._perf_cache_trade_count
            and engine._perf_cache is not None
            and (now - engine._perf_cache_time) < 60
        ):
            return engine._perf_cache

        trades_snapshot = list(engine.trade_history)

        from collections import defaultdict

        symbol_stats = defaultdict(lambda: {"trades": 0, "wins": 0, "total_pnl": 0.0, "last_trade_ts": 0})
        strategy_stats = defaultdict(lambda: {"trades": 0, "wins": 0, "total_pnl": 0.0})
        symbol_stop_losses = defaultdict(int)
        symbol_hold_times = defaultdict(list)

        for trade in trades_snapshot:
            if trade.get("side") != "sell":
                continue
            symbol = trade["symbol"]
            pnl = trade.get("realized_pnl", 0.0)
            strategy = trade.get("strategy_type", "unknown")
            exit_reason = trade.get("exit_reason", "")
            if exit_reason == "stop_loss":
                symbol_stop_losses[symbol] += 1
            hold_time = trade.get("hold_time_seconds")
            if hold_time is not None:
                symbol_hold_times[symbol].append(hold_time)

            symbol_stats[symbol]["trades"] += 1
            symbol_stats[symbol]["total_pnl"] += pnl
            if pnl > 0:
                symbol_stats[symbol]["wins"] += 1
            symbol_stats[symbol]["last_trade_ts"] = max(symbol_stats[symbol]["last_trade_ts"], trade.get("timestamp", 0) / 1000.0)

            strategy_stats[strategy]["trades"] += 1
            strategy_stats[strategy]["total_pnl"] += pnl
            if pnl > 0:
                strategy_stats[strategy]["wins"] += 1

        symbol_perf = {}
        for sym, s in symbol_stats.items():
            win_rate = s["wins"] / s["trades"] if s["trades"] > 0 else 0.0
            avg_pnl = s["total_pnl"] / s["trades"] if s["trades"] > 0 else 0.0
            symbol_perf[sym] = {
                "trades": s["trades"],
                "win_rate": round(win_rate, 3),
                "avg_pnl": round(avg_pnl, 4),
                "total_pnl": round(s["total_pnl"], 4),
                "last_trade_seconds_ago": round(now - s["last_trade_ts"]) if s["last_trade_ts"] else None,
                "stop_loss_hits": symbol_stop_losses.get(sym, 0),
                "avg_hold_time_seconds": round(sum(symbol_hold_times[sym]) / len(symbol_hold_times[sym]), 1) if symbol_hold_times.get(sym) else None,
            }

        strategy_perf = {}
        for st, s in strategy_stats.items():
            win_rate = s["wins"] / s["trades"] if s["trades"] > 0 else 0.0
            avg_pnl = s["total_pnl"] / s["trades"] if s["trades"] > 0 else 0.0
            strategy_perf[st] = {
                "trades": s["trades"],
                "win_rate": round(win_rate, 3),
                "avg_pnl": round(avg_pnl, 4),
                "total_pnl": round(s["total_pnl"], 4),
            }

        recent_sells = [t for t in trades_snapshot if t.get("side") == "sell"][-10:]
        recent_pnl = [t.get("realized_pnl", 0.0) for t in recent_sells]
        total_recent_pnl = sum(recent_pnl)
        trend = "up" if total_recent_pnl > 0 else "down" if total_recent_pnl < 0 else "flat"

        _equity = self.compute_equity_and_drawdown(trades_snapshot)
        current_realized_equity = _equity["current_realized_equity"]
        unrealized_pnl = _equity["unrealized_pnl"]
        current_equity = _equity["current_equity"]
        peak = _equity["peak"]
        drawdown_pct = _equity["drawdown_pct"]

        daily_pnl = engine._daily_realized_pnl()

        consecutive_losses = 0
        for trade in reversed(trades_snapshot):
            if trade.get("side") == "sell":
                pnl = trade.get("realized_pnl", 0.0)
                if pnl < 0:
                    consecutive_losses += 1
                else:
                    break

        result = {
            "stock_performance": symbol_perf,
            "strategy_performance": strategy_perf,
            "equity_curve": {
                "total_pnl": round(engine._realized_pnl_offset + sum(t.get("realized_pnl", 0.0) for t in trades_snapshot if t.get("side") == "sell"), 4),
                "recent_10_trades_pnl": round(total_recent_pnl, 4),
                "trend": trend,
                "drawdown_pct": round(drawdown_pct, 2),
                "daily_pnl": round(daily_pnl, 4),
                "consecutive_losses": consecutive_losses,
            },
        }

        engine._perf_cache = result
        engine._perf_cache_trade_count = engine._trade_history_version
        engine._perf_cache_time = now

        return result

    def compute_trade_pattern_analysis(self) -> Dict[str, Any]:
        """Analyze closed trades to identify which conditions, timeframes, and parameters
        have historically led to wins vs losses. Cached and only recomputed when new trades arrive."""
        engine = self.engine
        if engine._trade_history_version == engine._trade_pattern_cache_trade_count and engine._trade_pattern_cache is not None:
            return engine._trade_pattern_cache

        # Snapshot trade_history to avoid concurrent modification during iteration
        trades_snapshot = list(engine.trade_history)

        sells = [t for t in trades_snapshot if t.get("side") == "sell" and "realized_pnl" in t]
        if not sells:
            result: Dict[str, Any] = {}
            engine._trade_pattern_cache = result
            engine._trade_pattern_cache_trade_count = engine._trade_history_version
            return result

        def _win_rate_stats(trades: list) -> Optional[Dict[str, Any]]:
            if not trades:
                return None
            wins = [t for t in trades if t["realized_pnl"] > 0]
            total_pnl = sum(t["realized_pnl"] for t in trades)
            return {
                "win_rate": round(len(wins) / len(trades), 3),
                "trades": len(trades),
                "avg_pnl": round(total_pnl / len(trades), 6),
            }

        # --- Entry conditions (strategy type + confidence range as proxies) ---
        condition_groups: Dict[str, list] = defaultdict(list)
        for t in sells:
            strategy = t.get("strategy_type", "unknown")
            condition_groups[f"strategy={strategy}"].append(t)

        best_entry_conditions = []
        for cond, trades in condition_groups.items():
            stats = _win_rate_stats(trades)
            if stats and stats["trades"] >= 3:
                best_entry_conditions.append({"condition": cond, **stats})
        best_entry_conditions.sort(key=lambda x: x["win_rate"], reverse=True)
        best_entry_conditions = best_entry_conditions[:5]

        # --- Timeframes ---
        tf_groups: Dict[str, list] = defaultdict(list)
        for t in sells:
            tf = t.get("timeframe", "unknown")
            tf_groups[tf].append(t)
        best_timeframes = []
        for tf, trades in tf_groups.items():
            stats = _win_rate_stats(trades)
            if stats and stats["trades"] >= 3:
                best_timeframes.append({"timeframe": tf, **stats})
        best_timeframes.sort(key=lambda x: x["win_rate"], reverse=True)

        # --- Exit reasons ---
        exit_groups: Dict[str, list] = defaultdict(list)
        for t in sells:
            reason = t.get("exit_reason", "unknown")
            exit_groups[reason].append(t)
        best_exit_reasons = []
        for reason, trades in exit_groups.items():
            stats = _win_rate_stats(trades)
            if stats and stats["trades"] >= 2:
                best_exit_reasons.append({"exit_reason": reason, **stats})
        best_exit_reasons.sort(key=lambda x: x["win_rate"], reverse=True)

        # --- Confidence ranges ---
        conf_groups: Dict[str, list] = defaultdict(list)
        for t in sells:
            conf = t.get("buy_confidence", 0.5)
            if conf >= 0.8:
                conf_groups["0.8-1.0"].append(t)
            elif conf >= 0.5:
                conf_groups["0.5-0.8"].append(t)
            elif conf >= 0.3:
                conf_groups["0.3-0.5"].append(t)
            else:
                conf_groups["0.0-0.3"].append(t)
        best_confidence_ranges = []
        for rng, trades in conf_groups.items():
            stats = _win_rate_stats(trades)
            if stats and stats["trades"] >= 3:
                best_confidence_ranges.append({"range": rng, **stats})
        best_confidence_ranges.sort(key=lambda x: x["win_rate"], reverse=True)

        # --- Per-symbol performance ---
        symbol_groups: Dict[str, list] = defaultdict(list)
        for t in sells:
            symbol_groups[t["symbol"]].append(t)
        best_symbols = []
        worst_symbols = []
        for sym, trades in symbol_groups.items():
            stats = _win_rate_stats(trades)
            if stats and stats["trades"] >= 3:
                if stats["win_rate"] >= 0.5:
                    best_symbols.append({"symbol": sym, **stats})
                else:
                    worst_symbols.append({"symbol": sym, **stats})
        best_symbols.sort(key=lambda x: x["avg_pnl"], reverse=True)
        best_symbols = best_symbols[:5]
        worst_symbols.sort(key=lambda x: x["avg_pnl"])
        worst_symbols = worst_symbols[:5]

        # --- Hold time analysis ---
        winning_holds = [t.get("hold_time_seconds") for t in sells if t["realized_pnl"] > 0 and t.get("hold_time_seconds")]
        losing_holds = [t.get("hold_time_seconds") for t in sells if t["realized_pnl"] < 0 and t.get("hold_time_seconds")]
        avg_hold_winning = round(sum(winning_holds) / len(winning_holds)) if winning_holds else None
        avg_hold_losing = round(sum(losing_holds) / len(losing_holds)) if losing_holds else None

        result = {
            "best_entry_conditions": best_entry_conditions,
            "best_timeframes": best_timeframes,
            "best_exit_reasons": best_exit_reasons,
            "best_confidence_ranges": best_confidence_ranges,
            "best_symbols": best_symbols,
            "worst_symbols": worst_symbols,
            "avg_hold_time_winning": avg_hold_winning,
            "avg_hold_time_losing": avg_hold_losing,
        }

        engine._trade_pattern_cache = result
        engine._trade_pattern_cache_trade_count = engine._trade_history_version
        return result

    @staticmethod
    def _validate_param_range(
        name: str,
        value: Any,
        min_val: float,
        max_val: float,
        symbol: str,
        allow_none: bool = True,
    ) -> Optional[float]:
        """Validate that a numeric parameter is within [min_val, max_val].
        Returns the float value if valid, None if invalid or missing.
        Logs a warning for invalid values."""
        if value is None:
            return None
        try:
            fval = float(value)
        except (TypeError, ValueError):
            logger.warning(f"Invalid {name}={value!r} for {symbol}: not a number")
            return None
        if fval < min_val or fval > max_val:
            logger.warning(
                f"Invalid {name}={fval} for {symbol}: outside range [{min_val}, {max_val}]"
            )
            return None
        return fval

    async def update_position_params(
        self,
        symbol: str,
        params: Dict[str, Any],
        indicator_config: Optional[Dict[str, Any]],
        timeframe: str,
        current_price: float,
        atr: Optional[float],
    ):
        """Update risk parameters of an open position from LLM strategy_params."""
        engine = self.engine
        async with engine._positions_lock:
            pos = engine.positions.get(symbol)
        if not pos:
            return

        # --- Stop-loss (supports fixed pct and ATR multiple) ---
        stop_method = params.get("stop_loss_method", "fixed")
        if stop_method == "atr_multiple" and atr is not None and atr > 0:
            atr_mult = self._validate_param_range(
                "stop_loss_atr_multiple", params.get("stop_loss_atr_multiple"),
                0.01, 10.0, symbol,
            )
            if atr_mult is not None:
                sl_pct = (atr_mult * atr) / current_price
                pos["stop_loss"] = current_price * (1 - sl_pct)
        elif "stop_loss_pct" in params:
            sl_pct = self._validate_param_range(
                "stop_loss_pct", params["stop_loss_pct"],
                0.0001, 0.95, symbol,
            )
            if sl_pct is not None and current_price > 0:
                pos["stop_loss"] = current_price * (1 - sl_pct)

        # --- Take-profit (supports fixed pct and ATR multiple) ---
        if "take_profit_atr_multiple" in params and atr is not None and atr > 0 and current_price > 0:
            atr_mult = self._validate_param_range(
                "take_profit_atr_multiple", params["take_profit_atr_multiple"],
                0.01, 20.0, symbol,
            )
            if atr_mult is not None:
                tp_pct = (atr_mult * atr) / current_price
                pos["take_profit"] = current_price * (1 + tp_pct)
        elif "take_profit_pct" in params:
            tp_pct = self._validate_param_range(
                "take_profit_pct", params["take_profit_pct"],
                0.0001, 5.0, symbol,
            )
            if tp_pct is not None and current_price > 0:
                pos["take_profit"] = current_price * (1 + tp_pct)

        # --- BTP take-profit cap: enforce smaller targets for bonds ---
        if is_btp_isin(symbol) and pos.get("take_profit") and current_price > 0:
            tp_pct = (pos["take_profit"] - current_price) / current_price
            if tp_pct > settings.BTP_MAX_TAKE_PROFIT_PCT:
                capped_tp = current_price * (1 + settings.BTP_MAX_TAKE_PROFIT_PCT)
                logger.info(
                    f"BTP take-profit capped for {symbol}: {tp_pct:.4%} -> "
                    f"{settings.BTP_MAX_TAKE_PROFIT_PCT:.4%} (price {pos['take_profit']:.4f} -> {capped_tp:.4f})"
                )
                pos["take_profit"] = capped_tp

        # --- Trailing stop ---
        if "trailing_stop" in params:
            _upd_is_btp = is_btp_isin(symbol)
            if _upd_is_btp and params["trailing_stop"]:
                logger.warning(
                    f"LLM set trailing_stop=true for BTP {symbol} in position update, but trailing stops "
                    f"are not supported for BTPs. Forcing trailing_stop=false."
                )
                pos["trailing_stop"] = False
            else:
                pos["trailing_stop"] = params["trailing_stop"]
        if "trailing_stop_distance_pct" in params:
            val = self._validate_param_range(
                "trailing_stop_distance_pct", params["trailing_stop_distance_pct"],
                0.0001, 0.95, symbol,
            )
            if val is not None:
                pos["trailing_stop_distance_pct"] = val
        if "trailing_stop_activation_pct" in params:
            val = self._validate_param_range(
                "trailing_stop_activation_pct", params["trailing_stop_activation_pct"],
                0.0, 0.95, symbol,
            )
            if val is not None:
                pos["trailing_stop_activation_pct"] = val

        # --- Trailing take-profit ---
        if "trailing_take_profit" in params:
            pos["trailing_take_profit"] = params["trailing_take_profit"]
        if "trailing_take_profit_distance_pct" in params:
            val = self._validate_param_range(
                "trailing_take_profit_distance_pct", params["trailing_take_profit_distance_pct"],
                0.0001, 0.95, symbol,
            )
            if val is not None:
                pos["trailing_take_profit_distance_pct"] = val

        # --- Breakeven / lock-profit ---
        if "breakeven_activation_pct" in params:
            val = self._validate_param_range(
                "breakeven_activation_pct", params["breakeven_activation_pct"],
                0.0001, 0.95, symbol,
            )
            if val is not None:
                pos["breakeven_activation_pct"] = val
        # --- Time-based exits ---
        if "max_hold_time_seconds" in params:
            val = self._validate_param_range(
                "max_hold_time_seconds", params["max_hold_time_seconds"],
                1.0, 31_536_000.0, symbol,  # 1 second to ~1 year
            )
            if val is not None:
                # Store the original max hold time on first set so the
                # maximum position age safeguard can reference it even
                # if the LLM later extends max_hold_time_seconds.
                if "_original_max_hold_time_seconds" not in pos:
                    pos["_original_max_hold_time_seconds"] = val
                pos["max_hold_time_seconds"] = val
                # If the LLM explicitly sets a new hold time, clear any expiry flag
                pos.pop("_max_hold_expired", None)
                pos.pop("_max_hold_expired_count", None)

        # --- Cooldown after loss ---
        if "cooldown_after_loss_seconds" in params:
            val = self._validate_param_range(
                "cooldown_after_loss_seconds", params["cooldown_after_loss_seconds"],
                0.0, 2_592_000.0, symbol,  # 0 to ~30 days
            )
            if val is not None:
                pos["cooldown_after_loss_seconds"] = val

        # --- News sentiment exit ---
        if "news_sentiment_exit_threshold" in params:
            val = self._validate_param_range(
                "news_sentiment_exit_threshold", params["news_sentiment_exit_threshold"],
                -1.0, 0.0, symbol,
            )
            if val is not None:
                pos["news_sentiment_exit_threshold"] = val

        # --- Max unrealized loss ---
        if "max_unrealized_loss_pct" in params:
            val = self._validate_param_range(
                "max_unrealized_loss_pct", params["max_unrealized_loss_pct"],
                0.0001, 0.95, symbol,
            )
            if val is not None:
                pos["max_unrealized_loss_pct"] = val

        # --- Partial take-profit levels ---
        if "partial_take_profit_levels" in params:
            raw_levels = params["partial_take_profit_levels"]
            validated_levels = []
            if isinstance(raw_levels, list):
                for i, level in enumerate(raw_levels):
                    if not isinstance(level, dict):
                        continue
                    lvl_tp = self._validate_param_range(
                        f"partial_take_profit_levels[{i}].take_profit_pct",
                        level.get("take_profit_pct"), 0.0001, 5.0, symbol,
                    )
                    lvl_frac = self._validate_param_range(
                        f"partial_take_profit_levels[{i}].fraction",
                        level.get("fraction"), 0.01, 0.99, symbol,
                    )
                    if lvl_tp is not None and lvl_frac is not None:
                        validated_level = dict(level)
                        validated_level["take_profit_pct"] = lvl_tp
                        validated_level["fraction"] = lvl_frac
                        validated_levels.append(validated_level)
            if validated_levels:
                pos["partial_take_profit_levels"] = validated_levels
                pos["partial_tp_levels_triggered"] = []
                pos["partial_tp_depth_wait_start"] = {}
                # Clear single-level fields to avoid confusion
                pos["partial_take_profit_pct"] = None
                pos["partial_take_profit_fraction"] = None
                pos["partial_tp_triggered"] = None
        else:
            if "partial_take_profit_pct" in params:
                val = self._validate_param_range(
                    "partial_take_profit_pct", params["partial_take_profit_pct"],
                    0.0001, 5.0, symbol,
                )
                if val is not None:
                    pos["partial_take_profit_pct"] = val
            if "partial_take_profit_fraction" in params:
                val = self._validate_param_range(
                    "partial_take_profit_fraction", params["partial_take_profit_fraction"],
                    0.01, 0.99, symbol,
                )
                if val is not None:
                    pos["partial_take_profit_fraction"] = val
            if "partial_tp_triggered" not in pos:
                pos["partial_tp_triggered"] = False

        # --- Strategy interval ---
        if "strategy_interval_seconds" in params:
            val = self._validate_param_range(
                "strategy_interval_seconds", params["strategy_interval_seconds"],
                60.0, 2_592_000.0, symbol,  # 1 minute to ~30 days
            )
            if val is not None:
                engine._strategy_intervals[symbol] = val

        # --- Indicator config ---
        if indicator_config is not None:
            pos["indicator_config"] = indicator_config

        # --- Timeframe (if changed) ---
        if timeframe:
            pos["timeframe"] = timeframe

        logger.info(f"Updated risk parameters for {symbol} from LLM strategy_params")
        engine._portfolio_exposure_cache = None
        engine._state_dirty = True

    async def _close_btp_at_par(self, symbol: str, entry: dict, pos: dict, exit_reason: str, note: str, log_reason: str):
        """Helper to close a BTP position at par value (100.0) and record the trade."""
        engine = self.engine
        logger.info(f"Closing BTP {symbol} at par value. Reason: {log_reason}")
        engine.current_symbols.remove(entry)
        # Refund reserved cycle capital for any removed buy orders
        async with engine._queued_orders_lock:
            removed_buys = [q for q in engine.queued_orders if q['symbol'] == symbol and q['side'] == 'buy']
            engine.queued_orders = [q for q in engine.queued_orders if q['symbol'] != symbol]
        if removed_buys:
            async with engine._cycle_spent_lock:
                engine._cycle_spent = max(0.0, engine._cycle_spent - sum(q.get('amount', 0.0) for q in removed_buys))

        await self.event_bus.publish("cancel_exit_orders", symbol)
        async with engine._positions_lock:
            engine.positions.pop(symbol, None)

        par_value = 100.0
        cost = pos["amount"] * par_value
        from src.exchanges.fees import calculate_transaction_costs
        costs = calculate_transaction_costs("SELL", par_value, pos["amount"], symbol=symbol)
        fee_cost = costs["total_costs"]
        cost_basis = pos.get("cost_basis", pos["amount"] * pos["price"])
        net_quote = cost - fee_cost
        realized_pnl = net_quote - cost_basis
        trade = {
            "symbol": symbol,
            "side": "sell",
            "amount": pos["amount"],
            "price": par_value,
            "cost": cost,
            "fee": {"cost": fee_cost, "currency": engine.base_currency},
            "timestamp": time.time() * 1000,
            "note": note,
            "exit_reason": exit_reason,
            "realized_pnl": realized_pnl,
            "cost_basis": cost_basis,
        }
        engine._append_trade(trade)
        await asyncio.to_thread(insert_trade, trade)
        logger.info(f"Closed BTP {symbol}: {pos['amount']} at par value {par_value}.")
        if engine.notifier:
            stock_name = await engine._market_data_manager.get_stock_name(symbol)
            display_symbol = engine._format_symbol_display(symbol, stock_name, pos.get("timeframe"))
            await engine.notifier.send_notification(
                f"💰 BTP {display_symbol} closed at par value {par_value}. P&L: {realized_pnl:+.4f}",
                summary={
                    "symbol": symbol,
                    "action": "SELL",
                    "reason": log_reason,
                    "price": par_value,
                    "realized_pnl": realized_pnl,
                    "exit_reason": exit_reason,
                }
            )
        await self.event_bus.publish("remove_symbol_if_paused", symbol)

    async def reconcile_positions(self):
        """Detect and handle external changes: delisted symbols, externally sold positions."""
        engine = self.engine
        # --- Delisted stocks ---
        plain_assets = await engine._market_data_manager.get_tradable_assets()
        available_pairs = [f"{sym}/{engine.base_currency}" for sym in plain_assets]
        # Include BTP bonds and ETFs so they are not removed during reconciliation
        btp_bonds = await engine._market_data_manager.get_btp_bonds()
        available_pairs += [f"{b['isin']}/{engine.base_currency}" for b in btp_bonds]
        etf_symbols = await engine._market_data_manager.get_etf_symbols()
        available_pairs += [f"{sym}/{engine.base_currency}" for sym in etf_symbols]

        # Build BTP maturity map for maturity checking
        btp_maturity_map: Dict[str, Optional[str]] = {}
        for b in btp_bonds:
            isin = b.get("isin")
            maturity = b.get("maturity")
            if isin and maturity:
                btp_maturity_map[isin] = maturity

        # --- Matured BTP bonds: close at par value (100.0) ---
        now_dt = datetime.now(timezone.utc)
        for entry in list(engine.current_symbols):
            symbol = entry["symbol"]
            base = symbol.split("/")[0]
            if not is_btp_isin(base):
                continue
            maturity_str = btp_maturity_map.get(base)
            if maturity_str is None:
                continue
            maturity_dt = None
            maturity_str_clean = maturity_str.strip()
            # Try ISO 8601 formats first (with or without time component)
            try:
                if "T" in maturity_str_clean:
                    maturity_dt = datetime.fromisoformat(maturity_str_clean.replace("Z", "+00:00"))
                else:
                    maturity_dt = datetime.fromisoformat(maturity_str_clean)
                if maturity_dt.tzinfo is None:
                    maturity_dt = maturity_dt.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                pass

            # Try common European date formats
            if maturity_dt is None:
                # Normalize Italian month names/abbreviations to English
                # so strptime with %b/%B works regardless of system locale.
                _italian_months = {
                    # Abbreviations (case-insensitive matching)
                    "gen": "Jan", "feb": "Feb", "mar": "Mar", "apr": "Apr",
                    "mag": "May", "giu": "Jun", "lug": "Jul", "ago": "Aug",
                    "set": "Sep", "ott": "Oct", "nov": "Nov", "dic": "Dec",
                    # Full names
                    "gennaio": "January", "febbraio": "February", "marzo": "March",
                    "aprile": "April", "maggio": "May", "giugno": "June",
                    "luglio": "July", "agosto": "August", "settembre": "September",
                    "ottobre": "October", "novembre": "November", "dicembre": "December",
                }
                _normalized = maturity_str_clean
                for it_month, en_month in _italian_months.items():
                    _normalized = _normalized.replace(it_month, en_month)
                    _normalized = _normalized.replace(it_month.capitalize(), en_month)
                maturity_str_clean = _normalized

                _date_formats = [
                    "%d/%m/%Y",       # 01/10/2025
                    "%d-%m-%Y",       # 01-10-2025
                    "%d.%m.%Y",       # 01.10.2025
                    "%d/%m/%y",       # 01/10/25
                    "%d %b %Y",       # 01 Oct 2025 (also handles Italian after normalization)
                    "%d %B %Y",       # 01 October 2025 (also handles Italian after normalization)
                    "%B %d, %Y",      # October 01, 2025
                    "%Y-%m-%d",       # 2025-10-01 (ISO date only)
                    "%Y/%m/%d",       # 2025/10/01
                ]
                for fmt in _date_formats:
                    try:
                        maturity_dt = datetime.strptime(maturity_str_clean, fmt).replace(tzinfo=timezone.utc)
                        break
                    except (ValueError, TypeError):
                        continue

            if maturity_dt is None:
                logger.warning(f"Could not parse maturity date '{maturity_str}' for BTP {symbol}")
                pos = engine.positions.get(symbol)
                if pos:
                    grace_period = 7 * 24 * 3600  # 7 days
                    unparseable_since = pos.get("_unparseable_maturity_since")
                    if unparseable_since is None:
                        pos["_unparseable_maturity_since"] = time.time()
                        if engine.notifier:
                            stock_name = await engine._market_data_manager.get_stock_name(symbol)
                            display_symbol = engine._format_symbol_display(symbol, stock_name, None)
                            await engine.notifier.send_notification(
                                f"⚠️ Could not parse maturity date '{maturity_str}' for BTP {display_symbol}. Manual check required. Will auto-close at par in 7 days.",
                                summary={
                                    "symbol": symbol,
                                    "action": "WARNING",
                                    "reason": "Unparseable maturity date",
                                }
                            )
                        continue
                    elif time.time() - unparseable_since > grace_period:
                        await self._close_btp_at_par(symbol, entry, pos, "btp_unparseable_maturity", "btp_unparseable_maturity", "Unparseable maturity date grace period expired")
                        continue
                else:
                    if engine.notifier:
                        stock_name = await engine._market_data_manager.get_stock_name(symbol)
                        display_symbol = engine._format_symbol_display(symbol, stock_name, None)
                        await engine.notifier.send_notification(
                            f"⚠️ Could not parse maturity date '{maturity_str}' for BTP {display_symbol}. No open position found.",
                            summary={
                                "symbol": symbol,
                                "action": "WARNING",
                                "reason": "Unparseable maturity date",
                            }
                        )
                continue
            if now_dt < maturity_dt:
                continue
            # BTP has matured – close at par value
            pos = engine.positions.get(symbol)
            if pos:
                await self._close_btp_at_par(symbol, entry, pos, "btp_matured", "btp_matured", "BTP matured")

        for entry in list(engine.current_symbols):
            symbol = entry["symbol"]
            if symbol not in available_pairs:
                logger.warning(f"Stock {symbol} no longer available. Removing from tracking.")
                engine.current_symbols.remove(entry)
                # Remove any queued orders for this delisted symbol and refund reserved capital
                async with engine._queued_orders_lock:
                    removed_buys = [q for q in engine.queued_orders if q['symbol'] == symbol and q['side'] == 'buy']
                    engine.queued_orders = [q for q in engine.queued_orders if q['symbol'] != symbol]
                if removed_buys:
                    async with engine._cycle_spent_lock:
                        engine._cycle_spent = max(0.0, engine._cycle_spent - sum(q.get('amount', 0.0) for q in removed_buys))
                if symbol in engine.positions:
                    await self.event_bus.publish("cancel_exit_orders", symbol)
                    async with engine._positions_lock:
                        pos = engine.positions.pop(symbol)
                    cost_basis = pos.get("cost_basis", pos["amount"] * pos["price"])
                    base = symbol.split("/")[0]
                    is_btp = is_btp_isin(base)
                    if is_btp:
                        close_price = 100.0  # par value for delisted BTPs
                        close_cost = pos["amount"] * close_price
                        from src.exchanges.fees import calculate_transaction_costs
                        costs = calculate_transaction_costs("SELL", close_price, pos["amount"], symbol=symbol)
                        fee_cost = costs["total_costs"]
                        net_quote = close_cost - fee_cost
                        realized_pnl = net_quote - cost_basis
                        note = "btp_delisted"
                        exit_reason = "btp_delisted"
                    else:
                        close_price = 0.0
                        close_cost = 0.0
                        fee_cost = 0.0
                        realized_pnl = -cost_basis
                        note = "delisted"
                        exit_reason = "delisted"
                    trade = {
                        "symbol": symbol,
                        "side": "sell",
                        "amount": pos["amount"],
                        "price": close_price,
                        "cost": close_cost,
                        "fee": {"cost": fee_cost, "currency": engine.base_currency},
                        "timestamp": time.time() * 1000,
                        "note": note,
                        "exit_reason": exit_reason,
                        "realized_pnl": realized_pnl,
                        "cost_basis": cost_basis,
                    }
                    engine._append_trade(trade)
                    await asyncio.to_thread(insert_trade, trade)
                    logger.warning(f"Delisted {symbol}: recorded forced sell of {pos['amount']} at {close_price}.")
                    await self.event_bus.publish("remove_symbol_if_paused", symbol)

        # --- Externally modified balances ---
        # Fetch all balances at once instead of per-position API calls
        try:
            all_balances = await asyncio.to_thread(engine.trader.fetch_balance)
        except Exception as e:
            logger.error(f"Failed to fetch balances for reconciliation: {e}")
            all_balances = {}
        for symbol, pos in list(engine.positions.items()):
            base = symbol.split('/')[0]
            try:
                actual_balance = all_balances.get(base, 0.0)
            except Exception as e:
                logger.error(f"Failed to get balance for {base}: {e}")
                continue

            recorded_amount = pos.get("amount", 0.0)
            if actual_balance < recorded_amount - 1e-8:
                # External sell detected
                sold_amount = recorded_amount - actual_balance
                try:
                    tickers_map = await engine._market_data_manager._get_quotes_async([symbol.split("/")[0]], timeout=45.0)
                    ticker = tickers_map.get(symbol.split("/")[0])
                    current_price = ticker['last'] if ticker else pos.get("price", 0.0)
                except Exception:
                    current_price = pos.get("price", 0.0)  # fallback to entry price
                cost = sold_amount * current_price
                from src.exchanges.fees import calculate_transaction_costs
                costs = calculate_transaction_costs("SELL", current_price, sold_amount, symbol=symbol)
                fee_cost = costs["total_costs"]
                trade = {
                    "symbol": symbol,
                    "side": "sell",
                    "amount": sold_amount,
                    "price": current_price,
                    "cost": cost,
                    "fee": {"cost": fee_cost, "currency": engine.base_currency},
                    "timestamp": time.time() * 1000,
                    "note": "external_sell",
                    "exit_reason": "external_sell"
                }
                # Compute realized P&L for the externally sold portion
                cost_basis = pos.get("cost_basis", pos["amount"] * pos["price"])
                net_base = pos.get("net_base", pos["amount"])
                prorated_cost_basis = cost_basis * (sold_amount / net_base) if net_base > 0 else 0.0
                net_quote = cost - fee_cost
                trade["realized_pnl"] = net_quote - prorated_cost_basis
                trade["cost_basis"] = prorated_cost_basis
                engine._append_trade(trade)
                await asyncio.to_thread(insert_trade, trade)
                logger.warning(
                    f"External sell detected for {symbol}: {sold_amount} sold at ~{current_price}. "
                    f"Updating position from {recorded_amount} to {actual_balance}."
                )
                if actual_balance == 0.0:
                    await engine._cancel_exit_orders(symbol)
                    async with engine._positions_lock:
                        del engine.positions[symbol]
                    await self.event_bus.publish("remove_symbol_if_paused", symbol)
                else:
                    async with engine._positions_lock:
                        engine.positions[symbol]["amount"] = actual_balance
                        engine.positions[symbol]["cost_basis"] = cost_basis - prorated_cost_basis
                        engine.positions[symbol]["net_base"] = net_base - sold_amount
                        new_net_base = engine.positions[symbol]["net_base"]
                        new_cost_basis = engine.positions[symbol]["cost_basis"]
                        engine.positions[symbol]["price"] = new_cost_basis / new_net_base if new_net_base > 0 else 0.0
            elif actual_balance > recorded_amount + 1e-8:
                # External deposit – sync to actual balance
                logger.warning(
                    f"Balance of {base} increased externally from {recorded_amount} to {actual_balance}. "
                    f"Updating position."
                )
                async with engine._positions_lock:
                    engine.positions[symbol]["amount"] = actual_balance
                    engine.positions[symbol]["net_base"] = actual_balance
                    cost_basis = engine.positions[symbol].get("cost_basis", 0.0)
                    engine.positions[symbol]["price"] = cost_basis / actual_balance if actual_balance > 0 else 0.0

        # --- Handle positions that were loaded without LLM risk parameters ---
        for symbol, pos in list(engine.positions.items()):
            if pos.get("_needs_risk_params"):
                # Check if risk parameters have been populated by a re-evaluation
                if pos.get("stop_loss") is not None and pos.get("take_profit") is not None:
                    logger.info(f"Risk parameters obtained for {symbol}; clearing _needs_risk_params flag.")
                    pos.pop("_needs_risk_params", None)
                    pos.pop("_needs_risk_params_attempts", None)
                    continue

                # Risk parameters still missing — increment attempt counter
                attempts = pos.get("_needs_risk_params_attempts", 0) + 1
                pos["_needs_risk_params_attempts"] = attempts

                # Force another re-evaluation so the LLM gets another chance
                engine._force_eval[symbol] = True
                engine._last_strategy_eval.pop(symbol, None)

                max_attempts = 3  # ~15 minutes across 3 reconcile cycles (5 min each)
                if attempts >= max_attempts:
                    logger.warning(
                        f"Force-closing {symbol}: missing LLM risk parameters after {attempts} "
                        f"re-evaluation attempts."
                    )
                    stock_name = await engine._market_data_manager.get_stock_name(symbol)
                    display_symbol = engine._format_symbol_display(symbol, stock_name, pos.get("timeframe"))
                    if engine.notifier:
                        await engine.notifier.send_notification(
                            f"🔻 Closing {display_symbol} – missing LLM risk parameters after {attempts} attempts.",
                            summary={
                                "symbol": symbol,
                                "action": "SELL",
                                "reason": f"Missing LLM risk parameters after {attempts} attempts",
                                "exit_reason": "force_close",
                            }
                        )
                    signal = Signal(action="SELL", confidence=1.0, reasoning="Missing LLM risk parameters after re-evaluation attempts")
                    await self.event_bus.publish("execute_signal", symbol, signal, exit_reason="force_close")
                else:
                    logger.info(
                        f"Position {symbol} still missing risk parameters "
                        f"(attempt {attempts}/{max_attempts}); forcing re-evaluation."
                    )

        # Persist any changes made during reconciliation
        await self.event_bus.publish("save_state", force=True)
        engine._portfolio_exposure_cache = None
