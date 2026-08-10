"""Handles shortlist building and composite score computation for re-evaluation."""
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from src.config.settings import settings

logger = logging.getLogger(__name__)


class ReevalShortlistBuilder:
    """Builds the LLM shortlist and enforces minimum symbol constraints."""

    def __init__(self, engine, event_bus):
        self.engine = engine
        self.shared_state = engine.shared_state
        self.event_bus = event_bus

    def compute_ohlcv_summary(
        self,
        ohlcv_data: Dict[str, Dict[str, List[List]]],
        sample_pairs: List[str],
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Compute a per-symbol OHLCV summary from raw candle data.

        Returns {symbol: {timeframe: {change_pct, high, low, volume}}}.
        """
        ohlcv_summary: Dict[str, Dict[str, Dict[str, Any]]] = {}
        if ohlcv_data:
            for symbol in sample_pairs:
                if symbol in ohlcv_data:
                    tf_data = ohlcv_data[symbol]
                    summary: Dict[str, Dict[str, Any]] = {}
                    for tf, candles in tf_data.items():
                        if not candles:
                            continue
                        open_price = candles[0][1]
                        close_price = candles[-1][4]
                        high = max(c[2] for c in candles)
                        low = min(c[3] for c in candles)
                        volume = sum(c[5] for c in candles)
                        change_pct = ((close_price - open_price) / open_price) * 100 if open_price else 0
                        summary[tf] = {
                            "change_pct": round(change_pct, 2),
                            "high": high,
                            "low": low,
                            "volume": volume,
                        }
                    ohlcv_summary[symbol] = summary
        return ohlcv_summary

    def compute_composite_scores_and_shortlist(
        self,
        sample_pairs: List[str],
        symbol_trend_scores: Dict[str, float],
        news_sentiment: Dict[str, Any],
        trade_pattern_analysis: Dict[str, Any],
        etf_pairs: List[str],
        btp_pairs: List[str],
        incremental_offset: int = 0,
        incremental_batch_size: Optional[int] = None,
    ) -> Tuple[Dict[str, float], List[str]]:
        """Compute composite opportunity scores and build the LLM shortlist.

        Returns (composite_scores, shortlist) where:
        - composite_scores: {symbol: float} (0.0–1.0)
        - shortlist: deduplicated list of symbols sorted by composite score,
          with currently held symbols, historically best symbols, configured
          ETFs, all discovered ETFs, and all BTPs appended.
        """
        engine = self.engine

        # --- Composite opportunity score (trend + sentiment) ---
        composite_scores: Dict[str, float] = {}
        for sym in sample_pairs:
            trend = symbol_trend_scores.get(sym, 0.0)
            base_sym = sym.split("/")[0] if "/" in sym else sym
            sent = news_sentiment.get(base_sym, {}).get("avg_compound", 0.0) if news_sentiment else 0.0
            sentiment_score = (sent + 1.0) / 2.0  # map -1..1 to 0..1
            composite = settings.COMPOSITE_TREND_WEIGHT * trend + settings.COMPOSITE_SENTIMENT_WEIGHT * sentiment_score
            composite_scores[sym] = round(composite, 3)

        # Build a shortlist for the LLM: all symbols sorted by composite score,
        # plus any currently held symbols and historically best symbols.
        sorted_by_composite = sorted(sample_pairs, key=lambda s: composite_scores.get(s, 0), reverse=True)

        if incremental_batch_size is not None and incremental_batch_size > 0 and len(sorted_by_composite) > incremental_batch_size:
            # Incremental mode: take a rotating batch starting at the offset,
            # wrapping around if the offset exceeds the list length.
            total = len(sorted_by_composite)
            offset = incremental_offset % total
            batch = sorted_by_composite[offset:offset + incremental_batch_size]
            if len(batch) < incremental_batch_size:
                batch = batch + sorted_by_composite[:incremental_batch_size - len(batch)]
            logger.info(
                f"Incremental re-evaluation: evaluating batch of {len(batch)} symbols "
                f"(offset={offset}, total universe={total})"
            )
            shortlist = batch
        else:
            # Non-incremental: limit the base shortlist to avoid token explosion
            max_candidates = engine.max_symbols * 3
            if len(sorted_by_composite) > max_candidates:
                sorted_by_composite = sorted_by_composite[:max_candidates]
            shortlist = sorted_by_composite

        # Always include currently held symbols (they must be managed)
        for entry in self.shared_state.current_symbols:
            sym = entry["symbol"]
            if sym in sample_pairs and sym not in shortlist:
                shortlist.append(sym)

        # Always include historically best symbols (from trade pattern analysis)
        if trade_pattern_analysis:
            best_syms = [item["symbol"] for item in trade_pattern_analysis.get("best_symbols", [])]
            for sym in best_syms:
                if sym in sample_pairs and sym not in shortlist:
                    shortlist.append(sym)

        # Always include the configured ETFs
        for etf in settings.ALWAYS_INCLUDE_ETFS:
            pair = f"{etf}/{engine.base_currency}"
            if pair in sample_pairs and pair not in shortlist:
                shortlist.append(pair)

        # Always include ALL discovered ETFs for the LLM to consider
        for sym in etf_pairs:
            if sym not in shortlist:
                shortlist.append(sym)

        # Always include all BTPs for the LLM to consider
        for sym in btp_pairs:
            if sym not in shortlist:
                shortlist.append(sym)

        # Deduplicate shortlist while preserving order
        seen = set()
        shortlist = [s for s in shortlist if not (s in seen or seen.add(s))]

        return composite_scores, shortlist

    def enforce_asset_class_allocation(
        self,
        deduped: List[Dict[str, str]],
        etf_pairs: List[str],
        btp_pairs: List[str],
        ohlcv_data: Dict[str, Dict[str, List[List]]],
        base_balance: float,
        market_limits: Dict[str, Dict[str, float]],
        tickers: Dict[str, Dict[str, Any]],
    ) -> None:
        """Enforce minimum asset class allocation by appending missing ETFs/BTPs.
        
        Appends ETFs and BTPs to the deduped list if they are underrepresented,
        ensuring a balanced allocation across asset classes (20% ETFs, 20% BTPs).
        Modifies deduped in-place.
        """
        if not deduped:
            return

        engine = self.engine

        target_etf_pct = 0.20
        target_btp_pct = 0.20
        
        min_etfs = max(1, int(len(deduped) * target_etf_pct))
        min_btps = max(1, int(len(deduped) * target_btp_pct))
        
        current_etfs = [d for d in deduped if d['symbol'] in etf_pairs]
        current_btps = [d for d in deduped if d['symbol'] in btp_pairs]
        
        existing_syms = {d['symbol'] for d in deduped}
        default_tf = settings.OHLCV_TIMEFRAMES[0] if settings.OHLCV_TIMEFRAMES else "1h"
        
        # Add ETFs if underrepresented
        if len(current_etfs) < min_etfs:
            for etf in etf_pairs:
                if etf not in existing_syms:
                    if engine._is_excluded(etf, default_tf):
                        continue
                    sym_data = ohlcv_data.get(etf, {})
                    available_tfs = [t for t in settings.OHLCV_TIMEFRAMES if sym_data.get(t)]
                    if available_tfs:
                        tf = default_tf if default_tf in available_tfs else available_tfs[0]
                        # Check if the symbol has a valid current quote
                        quote = tickers.get(etf, {})
                        current_price = quote.get('close') or quote.get('last')
                        if not current_price or current_price <= 0:
                            continue
                        # Check if we can afford the minimum trade cost
                        min_cost = market_limits.get(etf, {}).get("min_cost", 0)
                        if base_balance >= min_cost:
                            deduped.append({"symbol": etf, "timeframe": tf})
                            existing_syms.add(etf)
                            if len([d for d in deduped if d['symbol'] in etf_pairs]) >= min_etfs:
                                break
                            
        # Add BTPs if underrepresented
        if len(current_btps) < min_btps:
            for btp in btp_pairs:
                if btp not in existing_syms:
                    if engine._is_excluded(btp, default_tf):
                        continue
                    sym_data = ohlcv_data.get(btp, {})
                    available_tfs = [t for t in settings.OHLCV_TIMEFRAMES if sym_data.get(t)]
                    if available_tfs:
                        tf = default_tf if default_tf in available_tfs else available_tfs[0]
                        # Check if the symbol has a valid current quote
                        quote = tickers.get(btp, {})
                        current_price = quote.get('close') or quote.get('last')
                        if not current_price or current_price <= 0:
                            continue
                        # Check if we can afford the minimum trade cost
                        min_cost = market_limits.get(btp, {}).get("min_cost", 0)
                        if base_balance >= min_cost:
                            deduped.append({"symbol": btp, "timeframe": tf})
                            existing_syms.add(btp)
                            if len([d for d in deduped if d['symbol'] in btp_pairs]) >= min_btps:
                                break

        # Update effective_max_symbols to accommodate newly appended symbols
        if len(deduped) > engine.effective_max_symbols:
            engine.effective_max_symbols = len(deduped)

    def enforce_min_symbols(
        self,
        deduped: List[Dict[str, str]],
        pause_trading: Optional[bool],
        sorted_by_composite: List[str],
        market_limits: Dict[str, Dict[str, float]],
        base_balance: float,
        ohlcv_data: Dict[str, Dict[str, List[List]]],
        tickers: Dict[str, Dict[str, Any]],
    ) -> None:
        """Enforce MIN_SYMBOLS setting, filling remaining slots from composite scores.

        Modifies engine.effective_max_symbols and appends to deduped in-place
        if additional symbols are needed to reach MIN_SYMBOLS.
        """
        engine = self.engine

        # --- Enforce minimum symbols (unless LLM explicitly paused) ---
        if (
            settings.MIN_SYMBOLS > 0
            and pause_trading is not True
            and engine.effective_max_symbols < settings.MIN_SYMBOLS
            and len(deduped) >= settings.MIN_SYMBOLS
        ):
            logger.info(
                f"LLM selected {engine.effective_max_symbols} symbols; "
                f"enforcing MIN_SYMBOLS={settings.MIN_SYMBOLS}"
            )
            engine.effective_max_symbols = settings.MIN_SYMBOLS

        # --- Fallback: fill remaining slots if LLM returned fewer than MIN_SYMBOLS ---
        if (
            settings.MIN_SYMBOLS > 0
            and pause_trading is not True
            and len(deduped) < settings.MIN_SYMBOLS
        ):
            # Try to fill remaining slots from composite-score-sorted sample_pairs
            existing_syms = {e["symbol"] for e in deduped}
            default_tf = settings.OHLCV_TIMEFRAMES[0] if settings.OHLCV_TIMEFRAMES else "1h"
            needed = settings.MIN_SYMBOLS - len(deduped)
            filled = 0
            for sym in sorted_by_composite:
                if filled >= needed:
                    break
                if sym in existing_syms:
                    continue
                if engine._is_excluded(sym, default_tf):
                    continue

                # Check if the symbol has a valid current quote
                quote = tickers.get(sym, {})
                current_price = quote.get('close') or quote.get('last')
                if not current_price or current_price <= 0:
                    continue

                # Check if OHLCV data is available for the symbol
                sym_data = ohlcv_data.get(sym, {})
                available_tfs = [t for t in settings.OHLCV_TIMEFRAMES if sym_data.get(t)]
                if not available_tfs:
                    continue
                tf = default_tf if default_tf in available_tfs else available_tfs[0]
                # Explicitly ensure the chosen timeframe has valid OHLCV data
                if not sym_data.get(tf):
                    continue

                # Check if we can afford the minimum trade cost
                min_cost = market_limits.get(sym, {}).get("min_cost", 0)
                if base_balance >= min_cost:
                    deduped.append({"symbol": sym, "timeframe": tf})
                    existing_syms.add(sym)
                    filled += 1
            if filled > 0:
                logger.info(
                    f"LLM returned only {len(deduped) - filled} symbols; "
                    f"filled {filled} additional slots from composite scores to reach MIN_SYMBOLS={settings.MIN_SYMBOLS}"
                )
                engine.effective_max_symbols = max(engine.effective_max_symbols, len(deduped))

    async def apply_fallback_selection(
        self,
        sample_pairs: List[str],
        composite_scores: Dict[str, float],
        tickers: Dict[str, Dict[str, Any]],
        market_limits: Dict[str, Dict[str, float]],
        base_balance: float,
        old_symbols: List[Dict[str, str]],
        pause_trading: Optional[bool],
        ohlcv_data: Dict[str, Dict[str, List[List]]],
    ) -> None:
        """Apply composite-score-based fallback selection when LLM returns no symbols.

        Picks top affordable symbols by composite score, falling back to
        previously tracked symbols if no suitable candidates are found.
        """
        engine = self.engine
        if self.shared_state.current_symbols or pause_trading is True:
            return

        logger.warning("LLM returned no symbols without pausing – using composite-score-based fallback.")
        if engine.notifier:
            await engine.notifier.send_notification(
                "⚠️ LLM returned no symbols. Using composite-score-based fallback selection.",
                summary={
                    "action": "FALLBACK",
                    "reason": "LLM returned no symbols, using fallback",
                    "model_type": "mind",
                }
            )
        # Sort sample_pairs by composite score (already computed above)
        sorted_pairs = sorted(sample_pairs, key=lambda s: composite_scores.get(s, 0), reverse=True)
        fallback_symbols: List[Dict[str, str]] = []
        default_tf = settings.OHLCV_TIMEFRAMES[0] if settings.OHLCV_TIMEFRAMES else "1h"
        for sym in sorted_pairs:
            if composite_scores.get(sym, 0) < settings.FALLBACK_MIN_COMPOSITE_SCORE:
                continue
            # Apply minimum 24h volume filter if configured
            if settings.FALLBACK_MIN_24H_VOLUME > 0:
                vol = tickers.get(sym, {}).get('quoteVolume', 0) or 0
                if vol < settings.FALLBACK_MIN_24H_VOLUME:
                    continue
            min_cost = market_limits.get(sym, {}).get('min_cost', 0)
            # Use total base_balance, not per_symbol_budget, since the LLM
            # allocates capital dynamically (not equal split)
            if base_balance >= min_cost:
                if engine._is_excluded(sym, default_tf):
                    continue

                # Check if the symbol has a valid current quote
                quote = tickers.get(sym, {})
                current_price = quote.get('close') or quote.get('last')
                if not current_price or current_price <= 0:
                    continue

                # Check if OHLCV data is available for the symbol
                sym_data = ohlcv_data.get(sym, {})
                available_tfs = [t for t in settings.OHLCV_TIMEFRAMES if sym_data.get(t)]
                if not available_tfs:
                    continue
                tf = default_tf if default_tf in available_tfs else available_tfs[0]

                fallback_symbols.append({"symbol": sym, "timeframe": tf})
            if len(fallback_symbols) >= engine.effective_max_symbols:
                break
        if fallback_symbols:
            existing_symbols = {c['symbol']: c for c in self.shared_state.current_symbols}
            for entry in fallback_symbols:
                if entry['symbol'] in existing_symbols and 'entry_time' in existing_symbols[entry['symbol']]:
                    entry['entry_time'] = existing_symbols[entry['symbol']]['entry_time']
                else:
                    entry['entry_time'] = time.time()
            self.shared_state.current_symbols = fallback_symbols
        elif old_symbols:
            logger.warning("Fallback found no symbols. Keeping previously tracked symbols.")
            self.shared_state.current_symbols = old_symbols
            engine.effective_max_symbols = max(len(old_symbols), 1)

    def update_current_symbols(
        self,
        deduped: List[Dict[str, str]],
        old_symbols: List[Dict[str, str]],
    ) -> None:
        """Replace current_symbols with the newly selected symbols.

        Preserves entry_time and max_tenure_hours for existing symbols,
        and updates position timeframes if they changed. If the LLM
        returned no symbols, keeps previously tracked symbols.
        """
        engine = self.engine
        if deduped and engine.effective_max_symbols > 0:
            existing_symbols = {c['symbol']: c for c in self.shared_state.current_symbols}
            for entry in deduped[: engine.effective_max_symbols]:
                sym = entry['symbol']
                new_tf = entry['timeframe']
                if sym in existing_symbols:
                    old_entry = existing_symbols[sym]
                    if 'entry_time' in old_entry:
                        entry['entry_time'] = old_entry['entry_time']
                    else:
                        entry['entry_time'] = time.time()

                    # Preserve max_tenure_hours from existing symbol if LLM didn't specify it
                    if 'max_tenure_hours' not in entry and 'max_tenure_hours' in old_entry:
                        entry['max_tenure_hours'] = old_entry['max_tenure_hours']

                    # Check if timeframe changed for an existing symbol
                    old_tf = old_entry.get('timeframe')
                    if old_tf != new_tf:
                        logger.info(f"Timeframe changed for {sym}: {old_tf} -> {new_tf}")
                        if sym in self.shared_state.positions:
                            self.shared_state.positions[sym]['timeframe'] = new_tf
                            # Clear max hold expired flags since the timeframe context changed
                            self.shared_state.positions[sym].pop("_max_hold_expired", None)
                            self.shared_state.positions[sym].pop("_max_hold_expired_count", None)
                else:
                    entry['entry_time'] = time.time()
            self.shared_state.current_symbols = deduped[: engine.effective_max_symbols]
        else:
            # LLM returned no symbols – keep previously tracked symbols
            if old_symbols:
                logger.info("LLM selected 0 symbols. Keeping previously tracked symbols for signal generation.")
                self.shared_state.current_symbols = old_symbols
                engine.effective_max_symbols = max(len(old_symbols), 1)
            else:
                self.shared_state.current_symbols = []
                engine.effective_max_symbols = 0
                logger.info("LLM selected 0 symbols – pausing trading until next evaluation.")
