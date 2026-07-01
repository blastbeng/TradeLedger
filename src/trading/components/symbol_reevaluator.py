"""Symbol re-evaluation component for the TradingEngine.

Handles asset discovery, quote fetching, sentiment, correlation, LLM chunking,
final selection, pause/resume, and state cleanup.
Extracted from TradingEngine to reduce class size and improve maintainability.
"""
import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple

from src.config.settings import settings

logger = logging.getLogger(__name__)


class SymbolReevaluator:
    """Handles symbol re-evaluation for the TradingEngine."""

    def __init__(self, engine):
        self.engine = engine

    async def check_cooldown_and_reset(
        self, force: bool
    ) -> Optional[Tuple[bool, bool, float]]:
        """Check re-evaluation cooldown and reset per-cycle spending.

        Resets _cycle_spent from queued buy orders, checks the triggered
        re-evaluation cooldown for market-condition triggers, clears
        pre-market and user-forced flags, and checks the last eval interval.

        Returns None if re-evaluation should be skipped.
        Otherwise returns (is_user_forced, is_market_condition_trigger, now).
        """
        engine = self.engine

        # Reset per-cycle spending tracker, but carry over capital already reserved
        # by queued buy orders from previous cycles so it is not re-allocated.
        async with engine._queued_orders_lock:
            queued_buy_total = sum(
                q.get('amount', 0.0) for q in engine.queued_orders
                if q.get('side') == 'buy'
            )
        async with engine._cycle_spent_lock:
            engine._cycle_spent = queued_buy_total
        logger.info("Re-evaluation step 1/12: Checking cooldown and fetching asset lists...")

        # Respect triggered re-evaluation cooldown for market-condition triggers only.
        # Pre-market re-evaluations are always allowed (they are time-critical).
        # Forced re-evaluations (explicit user or critical condition requests) always bypass
        # the cooldown since they are intentionally requested.
        # Capture whether this is a market-condition trigger before clearing flags
        is_market_condition_trigger = force and not engine._pre_market_reeval and not engine._user_forced_reeval

        if is_market_condition_trigger:
            last_triggered = await asyncio.to_thread(engine.redis.get, "trading:last_triggered_reeval")
            if last_triggered:
                elapsed = time.time() - float(last_triggered)
                if elapsed < settings.TRIGGERED_REEVALUATION_COOLDOWN:
                    logger.info(f"Forced re-evaluation skipped: triggered cooldown active ({settings.TRIGGERED_REEVALUATION_COOLDOWN - elapsed:.0f}s remaining)")
                    return None

        is_user_forced = engine._user_forced_reeval
        # Clear the pre-market flag after reading it
        engine._pre_market_reeval = False
        # Clear the user-forced flag after reading it
        engine._user_forced_reeval = False

        # Only re-evaluate every SYMBOL_REVALUATION_INTERVAL
        last_key = "trading:last_symbol_eval"
        last_eval = await asyncio.to_thread(engine.redis.get, last_key)
        now = time.time()
        if last_eval and (now - float(last_eval)) < engine._symbol_reevaluation_interval and engine.current_symbols and not force:
            logger.info("Skipping symbol re-evaluation: last eval was recent and symbols are already loaded.")
            return None

        return (is_user_forced, is_market_condition_trigger, now)

    def cleanup_stale_state_entries(self):
        """Remove stale entries from engine state dicts and base-symbol caches.

        Called at the end of each re-evaluation cycle to prune entries for
        symbols that are no longer tracked and have no open position.
        """
        engine = self.engine
        active_symbols = {entry["symbol"] for entry in engine.current_symbols}
        active_symbols.update(engine.positions.keys())
        for state_dict in (
            engine._force_eval,
            engine._last_decisions,
            engine._entry_signal_state,
            engine._force_eval_time,
            engine._last_strategy_eval,
            engine._strategy_intervals,
            engine._last_eval_snapshot,
            engine.last_loss_time,
            engine.cooldown_durations,
            engine._pending_entries,
        ):
            stale_keys = [s for s in state_dict if s not in active_symbols]
            for s in stale_keys:
                state_dict.pop(s, None)
            if stale_keys:
                logger.debug(f"Cleaned {len(stale_keys)} stale entries from engine state dicts")

        active_bases = {s.split("/")[0] for s in active_symbols}
        for cache_dict in (
            engine._sentiment_cache,
            engine._asset_cache,
            engine._asset_cache_time,
        ):
            stale_keys = [k for k in cache_dict if k not in active_bases]
            for k in stale_keys:
                cache_dict.pop(k, None)
            if stale_keys:
                logger.debug(f"Cleaned {len(stale_keys)} stale entries from base-symbol caches")

    def compute_correlation_matrix(
        self,
        ohlcv_data: Dict[str, List[List]],
        sorted_by_vol: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """Compute pairwise Pearson correlation matrix from OHLCV close prices.

        Tries timeframes from longest to shortest, requiring a minimum of
        20 candles and 19 returns for statistical significance.
        """
        corr_matrix: Dict[str, Dict[str, float]] = {}
        if ohlcv_data and settings.OHLCV_TIMEFRAMES:
            MIN_CANDLES = 20
            MIN_RETURNS = 19

            returns_series: Dict[str, List[float]] = {}
            used_tf = None
            for tf in settings.OHLCV_TIMEFRAMES:
                close_series: Dict[str, List[float]] = {}
                for sym in sorted_by_vol:
                    if sym in ohlcv_data and tf in ohlcv_data[sym]:
                        candles = ohlcv_data[sym][tf]
                        if len(candles) >= MIN_CANDLES:
                            close_series[sym] = [c[4] for c in candles]
                candidate_returns: Dict[str, List[float]] = {}
                for sym, closes in close_series.items():
                    returns = [(closes[i] - closes[i - 1]) / closes[i - 1]
                               for i in range(1, len(closes)) if closes[i - 1] != 0]
                    if len(returns) >= MIN_RETURNS:
                        candidate_returns[sym] = returns
                if len(candidate_returns) >= 2:
                    returns_series = candidate_returns
                    used_tf = tf
                    break

            if used_tf:
                logger.debug(
                    f"Correlation matrix computed using {used_tf} timeframe "
                    f"({len(returns_series)} symbols)"
                )
            corr_symbols = list(returns_series.keys())
            for sym_a in corr_symbols:
                corr_matrix[sym_a] = {}
                for sym_b in corr_symbols:
                    if sym_a == sym_b:
                        corr_matrix[sym_a][sym_b] = 1.0
                    elif sym_b in corr_matrix and sym_a in corr_matrix[sym_b]:
                        corr_matrix[sym_a][sym_b] = corr_matrix[sym_b][sym_a]
                    else:
                        ret_a = returns_series[sym_a]
                        ret_b = returns_series[sym_b]
                        min_len = min(len(ret_a), len(ret_b))
                        if min_len < 2:
                            continue
                        a = ret_a[-min_len:]
                        b = ret_b[-min_len:]
                        mean_a = sum(a) / min_len
                        mean_b = sum(b) / min_len
                        cov = sum((a[k] - mean_a) * (b[k] - mean_b) for k in range(min_len)) / min_len
                        std_a = (sum((x - mean_a) ** 2 for x in a) / min_len) ** 0.5
                        std_b = (sum((x - mean_b) ** 2 for x in b) / min_len) ** 0.5
                        if std_a > 0 and std_b > 0:
                            corr_matrix[sym_a][sym_b] = round(cov / (std_a * std_b), 3)
        return corr_matrix
