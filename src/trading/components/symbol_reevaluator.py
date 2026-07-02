"""Symbol re-evaluation component for the TradingEngine.

Handles asset discovery, quote fetching, sentiment, correlation, LLM chunking,
final selection, pause/resume, and state cleanup.
Extracted from TradingEngine to reduce class size and improve maintainability.
"""
import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from src.config.settings import settings
from src.database import get_ohlcv, get_indicators_for_symbols, get_aggregate_sentiment_for_symbols, get_aggregate_sentiment_from_db
from src.exchanges.market_data import get_quotes_cached
from src.llm.prompts import build_stock_selection_prompt, build_system_prompt, compact_prompt
from src.llm.cache import get_cached_llm_response, compute_market_hash

try:
    from src.news.fetcher import discover_trending_stocks, discover_tickers_from_news, detect_upcoming_events
except ImportError:
    discover_trending_stocks = None
    discover_tickers_from_news = None
    detect_upcoming_events = None

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

    async def fetch_and_filter_candidate_assets(
        self, now: float
    ) -> Optional[Tuple[List[str], List[str], List[str], List[Dict[str, str]], str]]:
        """Fetch tradable assets, BTPs, ETFs, filter by name, and run RSS/news discovery.

        Returns (available_pairs, btp_pairs, etf_pairs, old_symbols, last_key)
        or None if no symbols are available.
        """
        engine = self.engine
        last_key = "trading:last_symbol_eval"

        logger.info("Re-evaluation step 2/12: Fetching tradable assets, BTPs, and ETFs...")
        old_symbols = list(engine.current_symbols)
        plain_assets = await engine._get_tradable_assets()
        stock_pairs = [f"{sym}/{engine.base_currency}" for sym in plain_assets]

        # Fetch BTP bonds
        btp_bonds = await engine._get_btp_bonds()
        btp_pairs = [f"{b['isin']}/{engine.base_currency}" for b in btp_bonds]

        # Fetch ETFs
        etf_symbols = await engine._get_etf_symbols()
        etf_pairs = [f"{sym}/{engine.base_currency}" for sym in etf_symbols]
        available_pairs = stock_pairs + btp_pairs

        # --- Filter: only include symbols that have a name in discovered_symbols ---
        from src.database import get_discovered_symbols_with_names
        symbols_with_names = await asyncio.to_thread(get_discovered_symbols_with_names)
        _suffix = settings.TICKER_SUFFIX

        def _has_name(pair: str) -> bool:
            base = pair.split("/")[0]
            db_base = base
            if _suffix and db_base.endswith(_suffix):
                db_base = db_base[:-len(_suffix)]
            return db_base in symbols_with_names or base in symbols_with_names

        available_pairs = [p for p in available_pairs if _has_name(p)]
        btp_pairs = [p for p in btp_pairs if p.split("/")[0] in symbols_with_names]
        etf_pairs = [p for p in etf_pairs if _has_name(p)]

        if not available_pairs and not btp_pairs:
            logger.warning("No symbols with names in discovered_symbols. Skipping re-evaluation.")
            await asyncio.to_thread(engine.redis.set, last_key, now)
            return None

        logger.info("Re-evaluation step 3/12: RSS and news-driven symbol discovery...")
        # --- RSS-based ticker discovery: scan news feeds for symbols with TICKER_SUFFIX ---
        if settings.NEWS_ENABLED and settings.NEWS_TICKER_DISCOVERY_ENABLED and discover_tickers_from_news is not None:
            try:
                rss_discovered = await asyncio.to_thread(
                    discover_tickers_from_news,
                    existing_pairs=available_pairs,
                    cache_only=True,
                )
                # Convert discovered base symbols to full pairs and add to the front
                for base in rss_discovered:
                    pair = f"{base}/{engine.base_currency}"
                    if pair not in available_pairs:
                        available_pairs.insert(0, pair)
                if rss_discovered:
                    logger.info(f"RSS ticker discovery added {len(rss_discovered)} new symbols: {rss_discovered}")
            except Exception as e:
                logger.warning(f"RSS ticker discovery failed: {e}")

        if not available_pairs:
            logger.warning("No available pairs found.")
            return None

        # --- News-driven symbol discovery: add trending symbols not in the top 50 ---
        if settings.NEWS_ENABLED and settings.NEWS_SYMBOL_DISCOVERY_ENABLED and discover_trending_stocks is not None:
            try:
                discovered = await asyncio.to_thread(
                    discover_trending_stocks,
                    engine.base_currency,
                    available_pairs,
                    max_symbols=settings.NEWS_SYMBOL_DISCOVERY_MAX_SYMBOLS,
                    min_sentiment=settings.NEWS_SYMBOL_DISCOVERY_MIN_SENTIMENT,
                    min_articles=settings.NEWS_SYMBOL_DISCOVERY_MIN_ARTICLES,
                    cache_only=True,
                )
                # Add discovered symbols to the front of the list so they are included in the sample
                for pair in discovered:
                    if pair not in available_pairs:
                        available_pairs.insert(0, pair)
                if discovered:
                    logger.info(f"Added {len(discovered)} news-discovered symbols to candidate pool.")
            except Exception as e:
                logger.warning(f"News stock discovery failed: {e}")

        return available_pairs, btp_pairs, etf_pairs, old_symbols, last_key

    async def fetch_ohlcv_from_db(
        self, sorted_by_vol: List[str]
    ) -> Tuple[Dict[str, Dict[str, List[List]]], Dict[str, List[str]]]:
        """Fetch OHLCV data from the database for all candidate symbols.

        Returns (ohlcv_data, available_timeframes_by_symbol) where:
        - ohlcv_data: {symbol: {timeframe: [[ts, o, h, l, c, v], ...]}}
        - available_timeframes_by_symbol: {symbol: [tf1, tf2, ...]}
        """
        ohlcv_data: Dict[str, Dict[str, List[List]]] = {}
        if settings.OHLCV_TIMEFRAMES:
            async def _fetch_ohlcv(sym: str):
                data = {}
                for tf in settings.OHLCV_TIMEFRAMES:
                    try:
                        db_candles = await asyncio.to_thread(
                            get_ohlcv, sym, tf, limit=50
                        )
                        if db_candles:
                            data[tf] = [
                                [c["timestamp"], c["open"], c["high"], c["low"], c["close"], c["volume"]]
                                for c in db_candles
                            ]
                    except Exception as e:
                        logger.debug(f"DB OHLCV fetch failed for {sym} {tf}: {e}")
                return sym, data
            tasks = [_fetch_ohlcv(sym) for sym in sorted_by_vol]
            results = await asyncio.gather(*tasks)
            ohlcv_data = dict(results)

        available_timeframes_by_symbol: Dict[str, List[str]] = {}
        for sym, tf_data in ohlcv_data.items():
            available_tfs = [tf for tf in settings.OHLCV_TIMEFRAMES if tf in tf_data and tf_data[tf]]
            if available_tfs:
                available_timeframes_by_symbol[sym] = available_tfs

        return ohlcv_data, available_timeframes_by_symbol

    async def fetch_indicators_and_trend_scores(
        self,
        sorted_by_vol: List[str],
        sample_pairs: List[str],
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, float]]:
        """Batch-fetch indicators from DB and compute per-symbol trend scores.

        Returns (symbol_indicators, symbol_trend_scores) where:
        - symbol_indicators: {symbol: {timeframe: {indicator: value}}}
        - symbol_trend_scores: {symbol: float} (0.0–1.0)
        """
        primary_tf = settings.OHLCV_TIMEFRAMES[0] if settings.OHLCV_TIMEFRAMES else "1h"

        batch_indicators = await asyncio.to_thread(
            get_indicators_for_symbols, sorted_by_vol, settings.OHLCV_TIMEFRAMES
        )

        def _compute_trend_score(sym: str, sym_indicators: Dict[str, Dict[str, Any]]) -> float:
            trend_score = 0.0
            try:
                ind = sym_indicators.get(primary_tf, {})
                score = 0.0
                components = 0

                adx_val = ind.get('adx')
                if adx_val is not None:
                    score += min(1.0, adx_val / 50.0)
                    components += 1

                ema_9_val = ind.get('ema_9')
                ema_21_val = ind.get('ema_21')
                if ema_9_val is not None and ema_21_val is not None:
                    score += 1.0 if ema_9_val > ema_21_val else 0.0
                    components += 1

                rsi_val = ind.get('rsi')
                if rsi_val is not None:
                    if 40 <= rsi_val <= 70:
                        score += 1.0
                    elif 30 <= rsi_val <= 80:
                        score += 0.5
                    else:
                        score += 0.0
                    components += 1

                macd_hist_val = ind.get('macd_hist')
                if macd_hist_val is not None:
                    score += 1.0 if macd_hist_val > 0 else 0.0
                    components += 1

                plus_di_val = ind.get('plus_di')
                minus_di_val = ind.get('minus_di')
                if plus_di_val is not None and minus_di_val is not None:
                    score += 1.0 if plus_di_val > minus_di_val else 0.0
                    components += 1

                if components > 0:
                    trend_score = round(score / components, 3)
            except Exception:
                pass
            return trend_score

        symbol_indicators: Dict[str, Dict[str, Any]] = {}
        symbol_trend_scores: Dict[str, float] = {}
        for sym in sorted_by_vol:
            sym_inds = batch_indicators.get(sym, {})
            symbol_indicators[sym] = sym_inds
            symbol_trend_scores[sym] = _compute_trend_score(sym, sym_inds)

        # Ensure all sample_pairs have a trend score even if OHLCV was missing
        for sym in sample_pairs:
            if sym not in symbol_trend_scores:
                symbol_trend_scores[sym] = 0.0

        return symbol_indicators, symbol_trend_scores

    async def fetch_yahoo_fallback_quotes(
        self, sample_pairs: List[str], tickers: Dict[str, Dict[str, Any]]
    ) -> None:
        """Fetch missing quotes (last, bid, ask) from Yahoo Finance for up to 20 symbols.

        Updates the tickers dict in-place.
        """
        if not settings.YAHOO_FINANCE_ENABLED:
            return

        from src.exchanges.yahoo_finance import get_yahoo_quote

        missing_quotes = [
            sym for sym in sample_pairs
            if tickers.get(sym, {}).get('last') is None
            or tickers.get(sym, {}).get('bid') is None
            or tickers.get(sym, {}).get('ask') is None
        ]
        missing_quotes = missing_quotes[:20]

        async def _fetch_yahoo_quote(sym: str):
            base = sym.split("/")[0]
            yahoo = await asyncio.to_thread(get_yahoo_quote, base)
            if yahoo:
                t = tickers.setdefault(sym, {})
                if t.get('last') is None:
                    t['last'] = yahoo.get('last')
                if t.get('bid') is None:
                    t['bid'] = yahoo.get('bid')
                if t.get('ask') is None:
                    t['ask'] = yahoo.get('ask')

        await asyncio.gather(*[_fetch_yahoo_quote(sym) for sym in missing_quotes])

    async def fetch_news_sentiment_and_trends(
        self,
        sample_pairs: List[str],
        tickers: Dict[str, Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], Dict[str, Optional[float]], Optional[Dict[str, Any]]]:
        """Batch-fetch news sentiment, compute sentiment trends, and market trend.

        Returns (news_sentiment, sentiment_trend, market_trend) where:
        - news_sentiment: {base_symbol: aggregate_sentiment_dict}
        - sentiment_trend: {base_symbol: delta_or_None}
        - market_trend: dict with symbol/change_24h/last, or None
        """
        engine = self.engine
        news_sentiment: Dict[str, Any] = {}
        if settings.NEWS_ENABLED:
            batch_sentiment = await asyncio.to_thread(
                get_aggregate_sentiment_for_symbols, sample_pairs, settings.NEWS_CACHE_TTL_SECONDS
            )
            for sym, agg in batch_sentiment.items():
                if agg:
                    base = sym.split("/")[0] if "/" in sym else sym
                    news_sentiment[base] = agg

        # Sentiment trend (delta from previous cycle)
        sentiment_trend: Dict[str, Optional[float]] = {}
        for sym in sample_pairs:
            base_symbol = sym.split("/")[0] if "/" in sym else sym
            current_compound = None
            if base_symbol in news_sentiment:
                current_compound = news_sentiment[base_symbol].get("avg_compound")
            prev_key = f"sentiment:prev:{base_symbol}"
            prev_raw = await asyncio.to_thread(engine.redis.get, prev_key)
            prev_compound = float(prev_raw) if prev_raw else None
            if current_compound is not None:
                await asyncio.to_thread(engine.redis.setex, prev_key, settings.NEWS_CACHE_TTL_SECONDS, str(current_compound))
            if current_compound is not None and prev_compound is not None:
                sentiment_trend[base_symbol] = round(current_compound - prev_compound, 4)
            else:
                sentiment_trend[base_symbol] = None

        # Overall market trend (use configured benchmark, e.g., FTSEMIB.MI)
        market_trend = None
        benchmark_symbol = settings.BENCHMARK_SYMBOL
        if benchmark_symbol in tickers:
            benchmark_ticker = tickers[benchmark_symbol]
            market_trend = {
                "symbol": benchmark_symbol,
                "change_24h": benchmark_ticker.get("percentage"),
                "last": benchmark_ticker.get("last"),
            }
        elif sample_pairs:
            first = sample_pairs[0]
            if first in tickers:
                t = tickers[first]
                market_trend = {
                    "symbol": first,
                    "change_24h": t.get("percentage"),
                    "last": t.get("last"),
                }

        return news_sentiment, sentiment_trend, market_trend

    async def fetch_quotes_and_sort(
        self,
        available_pairs: List[str],
        btp_pairs: List[str],
        etf_pairs: List[str],
        now: float,
        last_key: str,
    ) -> Optional[Tuple[Dict[str, float], float, float, Dict[str, Dict[str, Any]], List[str], List[str]]]:
        """Fetch quotes from cache, apply Yahoo fallback, filter, and sort by volume.

        Returns (balance, base_balance, per_symbol_budget, tickers, sample_pairs, stock_pairs)
        or None if no valid price data is found.
        """
        engine = self.engine

        # Reconstruct stock_pairs (stocks only, excluding BTPs and ETFs)
        _btp_set = set(btp_pairs)
        _etf_set = set(etf_pairs)
        stock_pairs = [p for p in available_pairs if p not in _btp_set and p not in _etf_set]

        logger.info("Re-evaluation step 4/12: Fetching balance and quotes (from %d available pairs)...", len(available_pairs))
        balance = await engine._get_cached_balance()
        base_balance = balance.get(engine.base_currency, 0.0)
        per_symbol_budget = base_balance / engine.max_symbols if engine.max_symbols > 0 else 0.0

        # Apply sentiment filter if configured
        if settings.SYMBOL_SELECTION_MIN_SENTIMENT > -1.0 and settings.NEWS_ENABLED:
            candidate_pairs = available_pairs
            async def _fetch_sentiment_filter(sym):
                try:
                    base_symbol = sym.split("/")[0] if "/" in sym else sym
                    agg = await asyncio.to_thread(get_aggregate_sentiment_from_db, base_symbol, max_age_seconds=settings.NEWS_CACHE_TTL_SECONDS)
                    if agg and agg["avg_compound"] >= settings.SYMBOL_SELECTION_MIN_SENTIMENT:
                        return sym
                    elif not agg:
                        return sym
                    return None
                except Exception:
                    return sym
            sentiment_filter_tasks = [_fetch_sentiment_filter(sym) for sym in candidate_pairs]
            sentiment_filter_results = await asyncio.gather(*sentiment_filter_tasks)
            sample_pairs = [sym for sym in sentiment_filter_results if sym is not None]
        else:
            sample_pairs = available_pairs

        # Ensure BTPs and ETFs are always included in the candidate pool
        for btp in btp_pairs:
            if btp not in sample_pairs:
                sample_pairs.append(btp)
        for etf in etf_pairs:
            if etf not in sample_pairs:
                sample_pairs.append(etf)

        # Remove fully excluded symbols from the candidate pool
        sample_pairs = [
            sym for sym in sample_pairs
            if not any(
                entry.split("/")[0] == sym.split("/")[0] and
                entry.split("/")[1] == sym.split("/")[1] and
                len(entry.split("/")) == 2
                for entry in settings.EXCLUDED_SYMBOLS
            )
        ]

        logger.info(f"Step 4: Fetching quotes for {len(sample_pairs)} symbols from Redis/DB cache")

        # Fetch quotes from Redis/DB cache only — no network calls.
        plain_sample = [s.split("/")[0] for s in sample_pairs]
        raw_quotes = await asyncio.to_thread(get_quotes_cached, plain_sample)
        tickers = {pair: raw_quotes.get(pair.split("/")[0], {}) for pair in sample_pairs}

        # Filter out symbols with no valid last price
        valid_sample_pairs = [
            sym for sym in sample_pairs
            if tickers.get(sym, {}).get('last') is not None and tickers[sym]['last'] > 0
        ]
        if not valid_sample_pairs:
            logger.warning("No symbols with valid price data. Idling until next evaluation.")
            await asyncio.to_thread(engine.redis.set, last_key, now)
            no_price_key = "trading:no_price_data_notify"
            last_notify = await asyncio.to_thread(engine.redis.get, no_price_key)
            should_notify = True
            if last_notify:
                try:
                    if (time.time() - float(last_notify)) < 3600:
                        should_notify = False
                except (ValueError, TypeError):
                    pass
            if should_notify and engine.notifier:
                await engine.notifier.send_notification(
                    "⚠️ No symbols with valid price data. Bot will idle.",
                    summary={"action": "HOLD", "reason": "No valid price data"}
                )
                await asyncio.to_thread(engine.redis.set, no_price_key, str(time.time()))
            return None
        sample_pairs = valid_sample_pairs

        # Yahoo Finance fallback for missing quotes
        logger.info("Re-evaluation step 5/12: Yahoo Finance fallback for missing quotes...")
        await self.fetch_yahoo_fallback_quotes(sample_pairs, tickers)

        # Sort candidate pool by 24h volume (preserve BTPs and ETFs)
        def _volume(sym):
            t = tickers.get(sym, {})
            return t.get('quoteVolume', 0) or 0
        stock_sample_sorted = sorted([s for s in sample_pairs if s in stock_pairs and s not in etf_pairs], key=_volume, reverse=True)
        etf_sample_sorted = [s for s in sample_pairs if s in etf_pairs]
        sample_pairs = stock_sample_sorted + etf_sample_sorted + [s for s in sample_pairs if s in btp_pairs]

        return balance, base_balance, per_symbol_budget, tickers, sample_pairs, stock_pairs

    async def compute_market_limits(
        self,
        sample_pairs: List[str],
        tickers: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, float]]:
        """Compute per-symbol market limits (min order size and min cost).

        Returns a dict: {symbol: {"min_cost": float, "min_amount": float|None}}
        """
        engine = self.engine
        market_limits: Dict[str, Dict[str, float]] = {}
        for symbol in sample_pairs:
            base = symbol.split('/')[0]
            try:
                asset = await engine._get_asset_info(symbol)
                min_amount = float(asset.min_order_size) if asset.min_order_size else None
            except Exception:
                min_amount = None
            ticker = tickers.get(symbol, {})
            last_price = ticker.get('last', 0)
            if min_amount is not None and last_price:
                numeric_min_cost = min_amount * last_price
            else:
                numeric_min_cost = 0.0
            market_limits[symbol] = {
                'min_cost': numeric_min_cost,
                'min_amount': min_amount,
            }
        return market_limits

    def compute_composite_scores_and_shortlist(
        self,
        sample_pairs: List[str],
        symbol_trend_scores: Dict[str, float],
        news_sentiment: Dict[str, Any],
        trade_pattern_analysis: Dict[str, Any],
        etf_pairs: List[str],
        btp_pairs: List[str],
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
            composite = 0.6 * trend + 0.4 * sentiment_score
            composite_scores[sym] = round(composite, 3)

        # Build a shortlist for the LLM: all symbols sorted by composite score,
        # plus any currently held symbols and historically best symbols.
        sorted_by_composite = sorted(sample_pairs, key=lambda s: composite_scores.get(s, 0), reverse=True)
        shortlist = sorted_by_composite

        # Always include currently held symbols (they must be managed)
        for entry in engine.current_symbols:
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

    async def get_or_compute_correlation_matrix(
        self,
        ohlcv_data: Dict[str, Dict[str, List[List]]],
        sorted_by_vol: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """Fetch the correlation matrix from Redis cache, or compute and cache it.

        Uses a dynamic TTL: shorter (10 min) during extreme market breadth,
        longer (30 min) otherwise.
        """
        engine = self.engine
        corr_cache_key = "reeval:correlation_matrix"
        correlation_matrix = None
        try:
            cached_corr = await asyncio.to_thread(engine.redis.get, corr_cache_key)
            if cached_corr:
                correlation_matrix = json.loads(cached_corr)
        except Exception:
            pass

        if correlation_matrix is None:
            correlation_matrix = await asyncio.to_thread(
                self.compute_correlation_matrix, ohlcv_data, sorted_by_vol
            )
            # Dynamic TTL: shorter during high-volatility / extreme market conditions
            corr_ttl = 1800  # default 30 minutes
            _mb = getattr(engine, '_market_breadth', None)
            if _mb:
                pos_pct = _mb.get("positive_pct", 50)
                if pos_pct > 80 or pos_pct < 20:
                    corr_ttl = 600  # 10 minutes during extreme breadth
            _fmb = None
            try:
                _fmb_raw = await asyncio.to_thread(engine.redis.get, "market:breadth:full")
                if _fmb_raw:
                    _fmb = json.loads(_fmb_raw)
            except Exception:
                pass
            if _fmb:
                pos_pct = _fmb.get("positive_pct", 50)
                if pos_pct > 80 or pos_pct < 20:
                    corr_ttl = 600
            try:
                await asyncio.to_thread(
                    engine.redis.setex, corr_cache_key, corr_ttl, json.dumps(correlation_matrix)
                )
            except Exception:
                pass

        return correlation_matrix

    async def fetch_shortlist_context(
        self,
        sample_pairs: List[str],
        tickers: Dict[str, Dict[str, Any]],
        market_trend: Optional[Dict[str, Any]],
    ) -> Tuple[Dict[str, Dict[str, Any]], dict, Dict[str, Any], Optional[Dict[str, Any]], Optional[float]]:
        """Fetch missing tickers, detect events, compute market breadth, and store market status.

        Returns (symbol_events, session_info, market_breadth, full_market_breadth, vix).
        """
        engine = self.engine

        # --- Ensure tickers dict covers all symbols in the final shortlist ---
        missing_tickers = [s for s in sample_pairs if s not in tickers or not tickers.get(s, {}).get('last')]
        if missing_tickers:
            missing_plain = [s.split("/")[0] for s in missing_tickers]
            try:
                extra_raw = await asyncio.to_thread(get_quotes_cached, missing_plain)
                for pair in missing_tickers:
                    base = pair.split("/")[0]
                    if base in extra_raw and extra_raw[base].get('last'):
                        tickers[pair] = extra_raw[base]
            except Exception as e:
                logger.warning(f"Failed to fetch missing tickers for shortlist: {e}")

        # --- Detect upcoming corporate events from news (parallelized) ---
        symbol_events: Dict[str, Dict[str, Any]] = {}
        if settings.NEWS_ENABLED and detect_upcoming_events is not None:
            async def _detect_event(sym: str):
                try:
                    event = await asyncio.to_thread(detect_upcoming_events, sym)
                    if event:
                        return sym, event
                except Exception:
                    pass
                return sym, None

            event_tasks = [_detect_event(sym) for sym in sample_pairs]
            event_results = await asyncio.gather(*event_tasks)
            for sym, event in event_results:
                if event:
                    symbol_events[sym] = event

        session_info = engine._get_session_info()

        # Market breadth: percentage of candidate stocks with positive 24h change
        positive_count = sum(1 for sym in sample_pairs if (tickers.get(sym, {}).get('percentage') or 0) > 0)
        total_count = len(sample_pairs)
        market_breadth = {
            "positive_pct": round(positive_count / total_count * 100, 1) if total_count > 0 else 0.0,
            "positive_count": positive_count,
            "total_count": total_count,
        }
        engine._market_breadth = market_breadth

        # Read full market breadth from Redis (computed by background task)
        full_market_breadth = None
        try:
            full_breadth_raw = await asyncio.to_thread(engine.redis.get, "market:breadth:full")
            if full_breadth_raw:
                full_market_breadth = json.loads(full_breadth_raw)
        except Exception:
            pass

        vix = await engine._fetch_vix()

        # Store market status in Redis for the web dashboard
        market_status = {
            "vix": vix,
            "market_breadth": market_breadth,
            "full_market_breadth": full_market_breadth,
            "spy_price": market_trend["last"] if market_trend else None,
            "timestamp": time.time(),
        }
        await asyncio.to_thread(engine.redis.setex, "market:status", 3600, json.dumps(market_status))

        return symbol_events, session_info, market_breadth, full_market_breadth, vix

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

    async def evaluate_llm_chunks(
        self,
        sample_pairs: List[str],
        tickers: Dict[str, Dict[str, Any]],
        ohlcv_summary: Dict[str, Dict[str, Dict[str, Any]]],
        symbol_indicators: Dict[str, Dict[str, Any]],
        market_limits: Dict[str, Dict[str, float]],
        symbol_events: Dict[str, Dict[str, Any]],
        symbol_trend_scores: Dict[str, float],
        sentiment_trend: Dict[str, Optional[float]],
        correlation_matrix: Dict[str, Dict[str, float]],
        ohlcv_data: Dict[str, Dict[str, List[List]]],
        perf: Dict[str, Any],
        market_trend: Optional[Dict[str, Any]],
        session_info: dict,
        market_breadth: Dict[str, Any],
        trading_paused_bool: bool,
        symbol_tenure: Dict[str, float],
        symbol_max_tenure: Dict[str, Any],
        vix: Optional[float],
        trade_pattern_analysis: Dict[str, Any],
        min_viable_amount: float,
        base_balance: float,
        per_symbol_budget: float,
        auto_resume_note: str,
        effective_temp: float,
    ) -> List[Dict[str, Any]]:
        """Evaluate the shortlist in chunks using the LLM.

        Returns a list of parsed chunk result dicts.
        """
        engine = self.engine
        system_prompt = compact_prompt(build_system_prompt())

        CHUNK_SIZE = settings.LLM_CHUNK_SIZE
        chunk_results: List[Dict[str, Any]] = []
        chunks = [sample_pairs[i:i + CHUNK_SIZE] for i in range(0, len(sample_pairs), CHUNK_SIZE)]
        total_steps = 10 + len(chunks) + 2
        logger.info("Re-evaluation step 11/%d: Evaluating %d chunks of ~%d symbols each...", total_steps, len(chunks), CHUNK_SIZE)

        for chunk_idx, chunk_symbols in enumerate(chunks):
            chunk_set = set(chunk_symbols)

            # Filter per-symbol data to chunk symbols
            chunk_tickers = {s: tickers.get(s, {}) for s in chunk_symbols}
            chunk_ohlcv_summary = {s: ohlcv_summary.get(s, {}) for s in chunk_symbols if s in ohlcv_summary}
            chunk_symbol_indicators = {s: symbol_indicators.get(s, {}) for s in chunk_symbols if s in symbol_indicators}
            chunk_market_limits = {s: market_limits.get(s, {}) for s in chunk_symbols if s in market_limits}
            chunk_symbol_events = {s: symbol_events.get(s, {}) for s in chunk_symbols if s in symbol_events}
            chunk_symbol_trend_scores = {s: symbol_trend_scores.get(s, 0.0) for s in chunk_symbols}
            chunk_sentiment_trend = {s.split("/")[0]: sentiment_trend.get(s.split("/")[0]) for s in chunk_symbols if s.split("/")[0] in sentiment_trend}

            # Filter correlation matrix to chunk symbols
            chunk_corr = {}
            if correlation_matrix:
                for sym_a, row in correlation_matrix.items():
                    if sym_a in chunk_set:
                        chunk_corr[sym_a] = {sym_b: v for sym_b, v in row.items() if sym_b in chunk_set}

            # Build chunk prompt
            chunk_prompt = await asyncio.to_thread(
                build_stock_selection_prompt,
                available_symbols=chunk_symbols,
                current_symbols=engine.current_symbols,
                max_symbols=engine.effective_max_symbols,
                base_currency=engine.base_currency,
                tickers=chunk_tickers,
                base_balance=base_balance,
                per_symbol_budget=per_symbol_budget,
                market_limits=chunk_market_limits,
                performance=perf,
                ohlcv_summary=chunk_ohlcv_summary,
                market_trend=market_trend,
                symbol_indicators=chunk_symbol_indicators,
                daily_pnl=perf["equity_curve"].get("daily_pnl"),
                correlation_matrix=chunk_corr if chunk_corr else None,
                session_info=session_info,
                sentiment_trend=chunk_sentiment_trend,
                trading_paused=trading_paused_bool,
                open_positions=engine.positions,
                symbol_tenure=symbol_tenure,
                symbol_max_tenure=symbol_max_tenure,
                vix=vix,
                trade_pattern_analysis=trade_pattern_analysis,
                symbol_events=chunk_symbol_events,
                symbol_trend_scores=chunk_symbol_trend_scores,
                market_breadth=market_breadth,
                min_viable_trade_amount=min_viable_amount,
            )
            if auto_resume_note:
                chunk_prompt += "\n" + auto_resume_note

            # Build market snapshot for caching
            chunk_market_snapshot = {
                "chunk_idx": chunk_idx,
                "available_pairs": chunk_symbols,
                "tickers": chunk_tickers,
                "ohlcv_data": {s: ohlcv_data.get(s, {}) for s in chunk_symbols},
                "symbol_indicators": chunk_symbol_indicators,
                "performance": perf,
                "session_info": session_info,
                "market_breadth": market_breadth,
                "trading_paused": trading_paused_bool,
                "open_positions": engine.positions,
                "base_balance": base_balance,
                "per_symbol_budget": per_symbol_budget,
                "current_symbols": engine.current_symbols,
            }
            chunk_market_hash = compute_market_hash(chunk_market_snapshot)

            # Call LLM for this chunk
            chunk_response = None
            max_retries = 2
            for attempt in range(max_retries + 1):
                try:
                    result = await asyncio.wait_for(
                        asyncio.to_thread(
                            get_cached_llm_response,
                            compact_prompt(chunk_prompt),
                            system_prompt,
                            300,
                            market_hash=chunk_market_hash,
                            model_type="mind",
                            temperature=effective_temp,
                        ),
                        timeout=settings.LLM_TIMEOUT
                    )
                    chunk_response = result["response"]
                    break
                except asyncio.TimeoutError:
                    if attempt < max_retries:
                        logger.warning(f"Chunk {chunk_idx + 1}/{len(chunks)} LLM timed out (attempt {attempt + 1}). Retrying...")
                        await asyncio.sleep(5 * (attempt + 1))
                    else:
                        logger.warning(f"Chunk {chunk_idx + 1}/{len(chunks)} LLM timed out after all retries. Skipping.")
                except Exception as e:
                    if attempt < max_retries:
                        logger.warning(f"Chunk {chunk_idx + 1}/{len(chunks)} LLM failed: {e}. Retrying...")
                        await asyncio.sleep(5 * (attempt + 1))
                    else:
                        logger.error(f"Chunk {chunk_idx + 1}/{len(chunks)} LLM failed after all retries: {e}")

            if chunk_response:
                try:
                    chunk_parsed = json.loads(chunk_response)
                    chunk_results.append(chunk_parsed)
                    logger.info("Chunk %d/%d: received %d symbol selections", chunk_idx + 1, len(chunks), len(chunk_parsed.get("stocks", [])))
                except json.JSONDecodeError:
                    logger.warning(f"Chunk {chunk_idx + 1}/{len(chunks)}: invalid JSON, retrying with correction.")
                    correction = (
                        "Your previous response was not valid JSON. "
                        "You MUST output ONLY a single JSON object, with no markdown fences, no explanations, no extra text. "
                        "Here is the original request:\n\n" + chunk_prompt
                    )
                    try:
                        correction_result = await asyncio.wait_for(
                            asyncio.to_thread(
                                get_cached_llm_response, compact_prompt(correction), system_prompt, 120,
                                model_type="actuator", temperature=effective_temp,
                            ),
                            timeout=settings.LLM_TIMEOUT
                        )
                        chunk_parsed = json.loads(correction_result["response"])
                        chunk_results.append(chunk_parsed)
                        logger.info("Chunk %d/%d: corrected, received %d selections", chunk_idx + 1, len(chunks), len(chunk_parsed.get("stocks", [])))
                    except Exception as e:
                        logger.warning(f"Chunk {chunk_idx + 1}/{len(chunks)}: correction also failed: {e}")
            else:
                logger.warning(f"Chunk {chunk_idx + 1}/{len(chunks)}: no response, skipping.")

            await asyncio.sleep(1)

        return chunk_results

    async def run_final_selection_llm_call(
        self,
        chunk_results: List[Dict[str, Any]],
        sample_pairs: List[str],
        base_balance: float,
        per_symbol_budget: float,
        perf: Dict[str, Any],
        market_trend: Optional[Dict[str, Any]],
        session_info: dict,
        market_breadth: Dict[str, Any],
        full_market_breadth: Optional[Dict[str, Any]],
        trading_paused_bool: bool,
        symbol_tenure: Dict[str, float],
        symbol_max_tenure: Dict[str, Any],
        trade_pattern_analysis: Dict[str, Any],
        vix: Optional[float],
        min_viable_amount: float,
        market_limits: Dict[str, Dict[str, float]],
        available_timeframes_by_symbol: Dict[str, List[str]],
        auto_resume_note: str,
        effective_temp: float,
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Run the final selection LLM call with retries and fallback merge.

        Returns (response, llm_provider, llm_model).
        If all retries fail and chunk_results exist, merges all chunk
        selections as a fallback.
        """
        from src.llm.prompts import build_final_selection_prompt
        engine = self.engine

        num_chunks = (len(sample_pairs) + settings.LLM_CHUNK_SIZE - 1) // settings.LLM_CHUNK_SIZE
        total_steps = 10 + num_chunks + 2
        logger.info("Re-evaluation step %d/%d: Calling LLM for final selection from %d chunk results...", total_steps - 1, total_steps, len(chunk_results))

        response = None
        llm_provider = None
        llm_model = None

        if not chunk_results:
            logger.warning("All chunk LLM calls failed. Will use fallback selection.")
        else:
            final_prompt = await asyncio.to_thread(
                build_final_selection_prompt,
                chunk_results=chunk_results,
                current_symbols=engine.current_symbols,
                max_symbols=engine.effective_max_symbols,
                base_currency=engine.base_currency,
                base_balance=base_balance,
                per_symbol_budget=per_symbol_budget,
                performance=perf,
                open_positions=engine.positions,
                market_breadth=market_breadth,
                full_market_breadth=full_market_breadth,
                market_trend=market_trend,
                session_info=session_info,
                trading_paused=trading_paused_bool,
                symbol_tenure=symbol_tenure,
                symbol_max_tenure=symbol_max_tenure,
                trade_pattern_analysis=trade_pattern_analysis,
                daily_pnl=perf["equity_curve"].get("daily_pnl"),
                vix=vix,
                min_viable_trade_amount=min_viable_amount,
                available_timeframes=settings.OHLCV_TIMEFRAMES,
                market_limits=market_limits,
                available_timeframes_by_symbol=available_timeframes_by_symbol,
            )
            if auto_resume_note:
                final_prompt += "\n" + auto_resume_note

            max_retries = 2
            for attempt in range(max_retries + 1):
                try:
                    result = await asyncio.wait_for(
                        asyncio.to_thread(
                            get_cached_llm_response,
                            compact_prompt(final_prompt),
                            compact_prompt(build_system_prompt()),
                            300,
                            model_type="mind",
                            temperature=effective_temp,
                        ),
                        timeout=settings.LLM_TIMEOUT
                    )
                    response = result["response"]
                    llm_provider = result["provider"]
                    llm_model = result["model"]
                    break
                except asyncio.TimeoutError:
                    if attempt < max_retries:
                        logger.warning(f"Final selection LLM timed out (attempt {attempt + 1}). Retrying...")
                        await asyncio.sleep(5 * (attempt + 1))
                    else:
                        logger.warning("Final selection LLM timed out after all retries.")
                except Exception as e:
                    if attempt < max_retries:
                        logger.warning(f"Final selection LLM failed: {e}. Retrying...")
                        await asyncio.sleep(5 * (attempt + 1))
                    else:
                        logger.error(f"Final selection LLM failed after all retries: {e}")

            # Fallback: merge all chunk selections if final call failed
            if response is None and chunk_results:
                logger.warning("Final selection LLM call failed. Merging all chunk selections as fallback.")
                merged_stocks = []
                for chunk in chunk_results:
                    for stock in chunk.get("stocks", []):
                        if isinstance(stock, dict) and "symbol" in stock:
                            merged_stocks.append(stock)
                seen = set()
                deduped = []
                for s in merged_stocks:
                    if s["symbol"] not in seen:
                        seen.add(s["symbol"])
                        deduped.append(s)
                response = json.dumps({
                    "stocks": deduped[:engine.effective_max_symbols],
                    "max_stocks": min(len(deduped), engine.effective_max_symbols),
                    "reasoning": "Fallback: merged all chunk selections (final LLM call failed)",
                })
                llm_provider = "fallback"
                llm_model = "merged_chunks"

        return response, llm_provider, llm_model

    async def store_llm_decided_parameters(self, parsed: Dict[str, Any]) -> None:
        """Store LLM-decided parameters from the stock selection response to Redis.

        Handles: max_positions_per_sector, portfolio risk thresholds, confidence
        rejection, limit price distance, min viable trade amount, eval skip
        thresholds, regime thresholds, stop/hold multipliers, review limits,
        pause settings, and re-evaluation interval.
        """
        engine = self.engine

        # Parse max_positions_per_sector from LLM
        max_positions_per_sector = parsed.get("max_positions_per_sector")
        if max_positions_per_sector is not None and isinstance(max_positions_per_sector, int) and max_positions_per_sector > 0:
            await asyncio.to_thread(engine.redis.setex, "trading:max_positions_per_sector", 7 * 24 * 3600, str(max_positions_per_sector))
            logger.info(f"LLM set max positions per sector to {max_positions_per_sector}")
        else:
            await asyncio.to_thread(engine.redis.delete, "trading:max_positions_per_sector")

        # Parse LLM-decided portfolio risk thresholds
        max_port_exp = parsed.get("max_portfolio_exposure_pct")
        if max_port_exp is not None and isinstance(max_port_exp, (int, float)) and 0.0 <= float(max_port_exp) <= 1.0:
            await asyncio.to_thread(engine.redis.setex, "trading:max_portfolio_exposure_pct", 7 * 24 * 3600, str(float(max_port_exp)))
        else:
            await asyncio.to_thread(engine.redis.delete, "trading:max_portfolio_exposure_pct")

        max_port_risk = parsed.get("max_portfolio_stop_risk_pct")
        if max_port_risk is not None and isinstance(max_port_risk, (int, float)) and 0.0 <= float(max_port_risk) <= 1.0:
            await asyncio.to_thread(engine.redis.setex, "trading:max_portfolio_stop_risk_pct", 7 * 24 * 3600, str(float(max_port_risk)))
        else:
            await asyncio.to_thread(engine.redis.delete, "trading:max_portfolio_stop_risk_pct")

        min_rr = parsed.get("min_risk_reward_ratio")
        if min_rr is not None and isinstance(min_rr, (int, float)) and min_rr > 0:
            await asyncio.to_thread(engine.redis.setex, "trading:min_risk_reward_ratio", 7 * 24 * 3600, str(float(min_rr)))
        else:
            await asyncio.to_thread(engine.redis.delete, "trading:min_risk_reward_ratio")

        conf_rejection = parsed.get("confidence_rejection_threshold")
        if conf_rejection is not None and isinstance(conf_rejection, (int, float)) and 0.0 <= float(conf_rejection) <= 1.0:
            await asyncio.to_thread(engine.redis.setex, "trading:confidence_rejection_threshold", 7 * 24 * 3600, str(float(conf_rejection)))
            logger.info(f"LLM set confidence rejection threshold to {float(conf_rejection):.2f}")
        else:
            await asyncio.to_thread(engine.redis.delete, "trading:confidence_rejection_threshold")

        # Parse LLM-controlled limit price max distance
        limit_price_max_dist = parsed.get("limit_price_max_distance_pct")
        if limit_price_max_dist is not None and isinstance(limit_price_max_dist, (int, float)) and 0.0 <= float(limit_price_max_dist) <= 1.0:
            await asyncio.to_thread(engine.redis.setex, "trading:limit_price_max_distance_pct", 7 * 24 * 3600, str(float(limit_price_max_dist)))
        else:
            await asyncio.to_thread(engine.redis.delete, "trading:limit_price_max_distance_pct")

        # Parse LLM-controlled minimum viable trade amount
        min_viable = parsed.get("min_viable_trade_amount")
        if min_viable is not None and isinstance(min_viable, (int, float)) and min_viable > 0:
            await asyncio.to_thread(engine.redis.setex, "trading:min_viable_trade_amount", 7 * 24 * 3600, str(float(min_viable)))
            logger.info(f"LLM set min viable trade amount to {float(min_viable):.2f}")
        else:
            await asyncio.to_thread(engine.redis.delete, "trading:min_viable_trade_amount")

        # Parse LLM evaluation skip thresholds
        skip_price_mult = parsed.get("skip_eval_price_change_atr_mult")
        if skip_price_mult is not None and isinstance(skip_price_mult, (int, float)) and skip_price_mult > 0:
            await asyncio.to_thread(engine.redis.setex, "trading:skip_eval_price_change_atr_mult", 7 * 24 * 3600, str(float(skip_price_mult)))
        else:
            await asyncio.to_thread(engine.redis.delete, "trading:skip_eval_price_change_atr_mult")

        skip_rsi = parsed.get("skip_eval_rsi_change")
        if skip_rsi is not None and isinstance(skip_rsi, (int, float)) and skip_rsi > 0:
            await asyncio.to_thread(engine.redis.setex, "trading:skip_eval_rsi_change", 7 * 24 * 3600, str(float(skip_rsi)))
        else:
            await asyncio.to_thread(engine.redis.delete, "trading:skip_eval_rsi_change")

        skip_rsi_oversold = parsed.get("skip_eval_rsi_oversold")
        if skip_rsi_oversold is not None and isinstance(skip_rsi_oversold, (int, float)) and skip_rsi_oversold > 0:
            await asyncio.to_thread(engine.redis.setex, "trading:skip_eval_rsi_oversold", 7 * 24 * 3600, str(float(skip_rsi_oversold)))
        else:
            await asyncio.to_thread(engine.redis.delete, "trading:skip_eval_rsi_oversold")

        skip_rsi_overbought = parsed.get("skip_eval_rsi_overbought")
        if skip_rsi_overbought is not None and isinstance(skip_rsi_overbought, (int, float)) and skip_rsi_overbought > 0:
            await asyncio.to_thread(engine.redis.setex, "trading:skip_eval_rsi_overbought", 7 * 24 * 3600, str(float(skip_rsi_overbought)))
        else:
            await asyncio.to_thread(engine.redis.delete, "trading:skip_eval_rsi_overbought")

        skip_macd = parsed.get("skip_eval_macd_hist_change")
        if skip_macd is not None and isinstance(skip_macd, (int, float)) and skip_macd > 0:
            await asyncio.to_thread(engine.redis.setex, "trading:skip_eval_macd_hist_change", 7 * 24 * 3600, str(float(skip_macd)))
        else:
            await asyncio.to_thread(engine.redis.delete, "trading:skip_eval_macd_hist_change")

        # Parse LLM-driven market regime thresholds
        regime_adx_strong = parsed.get("regime_adx_strong")
        if regime_adx_strong is not None and isinstance(regime_adx_strong, (int, float)) and regime_adx_strong > 0:
            await asyncio.to_thread(engine.redis.setex, "trading:regime_adx_strong", 7 * 24 * 3600, str(float(regime_adx_strong)))
        else:
            await asyncio.to_thread(engine.redis.delete, "trading:regime_adx_strong")

        regime_adx_moderate = parsed.get("regime_adx_moderate")
        if regime_adx_moderate is not None and isinstance(regime_adx_moderate, (int, float)) and regime_adx_moderate > 0:
            await asyncio.to_thread(engine.redis.setex, "trading:regime_adx_moderate", 7 * 24 * 3600, str(float(regime_adx_moderate)))
        else:
            await asyncio.to_thread(engine.redis.delete, "trading:regime_adx_moderate")

        regime_vol_high = parsed.get("regime_volatility_high_pct")
        if regime_vol_high is not None and isinstance(regime_vol_high, (int, float)) and regime_vol_high > 0:
            await asyncio.to_thread(engine.redis.setex, "trading:regime_volatility_high_pct", 7 * 24 * 3600, str(float(regime_vol_high)))
        else:
            await asyncio.to_thread(engine.redis.delete, "trading:regime_volatility_high_pct")

        regime_vol_low = parsed.get("regime_volatility_low_pct")
        if regime_vol_low is not None and isinstance(regime_vol_low, (int, float)) and regime_vol_low > 0:
            await asyncio.to_thread(engine.redis.setex, "trading:regime_volatility_low_pct", 7 * 24 * 3600, str(float(regime_vol_low)))
        else:
            await asyncio.to_thread(engine.redis.delete, "trading:regime_volatility_low_pct")

        regime_bb_squeeze = parsed.get("regime_bb_squeeze_width")
        if regime_bb_squeeze is not None and isinstance(regime_bb_squeeze, (int, float)) and regime_bb_squeeze > 0:
            await asyncio.to_thread(engine.redis.setex, "trading:regime_bb_squeeze_width", 7 * 24 * 3600, str(float(regime_bb_squeeze)))
        else:
            await asyncio.to_thread(engine.redis.delete, "trading:regime_bb_squeeze_width")

        regime_bb_expansion = parsed.get("regime_bb_expansion_width")
        if regime_bb_expansion is not None and isinstance(regime_bb_expansion, (int, float)) and regime_bb_expansion > 0:
            await asyncio.to_thread(engine.redis.setex, "trading:regime_bb_expansion_width", 7 * 24 * 3600, str(float(regime_bb_expansion)))
        else:
            await asyncio.to_thread(engine.redis.delete, "trading:regime_bb_expansion_width")

        min_stop_atr_mult = parsed.get("min_stop_loss_atr_mult")
        if min_stop_atr_mult is not None and isinstance(min_stop_atr_mult, (int, float)) and min_stop_atr_mult > 0:
            await asyncio.to_thread(engine.redis.setex, "trading:min_stop_loss_atr_mult", 7 * 24 * 3600, str(float(min_stop_atr_mult)))
        else:
            await asyncio.to_thread(engine.redis.delete, "trading:min_stop_loss_atr_mult")

        min_hold_time_mult = parsed.get("min_max_hold_time_mult")
        if min_hold_time_mult is not None and isinstance(min_hold_time_mult, (int, float)) and min_hold_time_mult > 0:
            await asyncio.to_thread(engine.redis.setex, "trading:min_max_hold_time_mult", 7 * 24 * 3600, str(float(min_hold_time_mult)))
        else:
            await asyncio.to_thread(engine.redis.delete, "trading:min_max_hold_time_mult")

        max_sl_reviews = parsed.get("max_stop_loss_reviews")
        if max_sl_reviews is not None and isinstance(max_sl_reviews, int) and 1 <= max_sl_reviews <= 20:
            await asyncio.to_thread(engine.redis.setex, "trading:max_stop_loss_reviews", 7 * 24 * 3600, str(max_sl_reviews))
        else:
            await asyncio.to_thread(engine.redis.delete, "trading:max_stop_loss_reviews")

        max_tp_reviews = parsed.get("max_take_profit_reviews")
        if max_tp_reviews is not None and isinstance(max_tp_reviews, int) and 1 <= max_tp_reviews <= 20:
            await asyncio.to_thread(engine.redis.setex, "trading:max_take_profit_reviews", 7 * 24 * 3600, str(max_tp_reviews))
        else:
            await asyncio.to_thread(engine.redis.delete, "trading:max_take_profit_reviews")

        min_llm_pause = parsed.get("min_llm_pause_duration_seconds")
        if min_llm_pause is not None and isinstance(min_llm_pause, int) and 300 <= min_llm_pause <= 14400:
            await asyncio.to_thread(engine.redis.setex, "trading:min_llm_pause_duration", 7 * 24 * 3600, str(min_llm_pause))
        else:
            await asyncio.to_thread(engine.redis.delete, "trading:min_llm_pause_duration")

        pause_max_keep = parsed.get("pause_max_consecutive_keep")
        if pause_max_keep is not None and isinstance(pause_max_keep, int) and 1 <= pause_max_keep <= 10:
            await asyncio.to_thread(engine.redis.setex, "trading:pause_max_consecutive_keep", 7 * 24 * 3600, str(pause_max_keep))
        else:
            await asyncio.to_thread(engine.redis.delete, "trading:pause_max_consecutive_keep")

        pause_force_mult = parsed.get("pause_force_resume_risk_multiplier")
        if pause_force_mult is not None and isinstance(pause_force_mult, (int, float)) and 0.0 <= float(pause_force_mult) <= 1.0:
            await asyncio.to_thread(engine.redis.setex, "trading:pause_force_resume_risk_multiplier", 7 * 24 * 3600, str(float(pause_force_mult)))
        else:
            await asyncio.to_thread(engine.redis.delete, "trading:pause_force_resume_risk_multiplier")

        max_partial_tp = parsed.get("max_partial_tp_reviews")
        if max_partial_tp is not None and isinstance(max_partial_tp, int) and 1 <= max_partial_tp <= 20:
            await asyncio.to_thread(engine.redis.setex, "trading:max_partial_tp_reviews", 7 * 24 * 3600, str(max_partial_tp))
        else:
            await asyncio.to_thread(engine.redis.delete, "trading:max_partial_tp_reviews")

        max_dust_sweep = parsed.get("max_dust_sweep_reviews")
        if max_dust_sweep is not None and isinstance(max_dust_sweep, int) and 1 <= max_dust_sweep <= 20:
            await asyncio.to_thread(engine.redis.setex, "trading:max_dust_sweep_reviews", 7 * 24 * 3600, str(max_dust_sweep))
        else:
            await asyncio.to_thread(engine.redis.delete, "trading:max_dust_sweep_reviews")

        # Optional: LLM can set the global symbol re-evaluation interval
        new_interval = parsed.get("stock_revaluation_interval_seconds")
        if new_interval is not None:
            if isinstance(new_interval, (int, float)) and new_interval >= 3600:
                clamped = max(new_interval, settings.MIN_SYMBOL_REEVALUATION_INTERVAL)
                engine._symbol_reevaluation_interval = clamped
                logger.info(f"LLM set symbol re-evaluation interval to {clamped}s (requested {new_interval}s)")
            else:
                logger.warning(f"Invalid stock_revaluation_interval_seconds: {new_interval} (must be >= 3600)")

    async def apply_fallback_selection(
        self,
        sample_pairs: List[str],
        composite_scores: Dict[str, float],
        tickers: Dict[str, Dict[str, Any]],
        market_limits: Dict[str, Dict[str, float]],
        base_balance: float,
        old_symbols: List[Dict[str, str]],
        pause_trading: Optional[bool],
    ) -> None:
        """Apply composite-score-based fallback selection when LLM returns no symbols.

        Picks top affordable symbols by composite score, falling back to
        previously tracked symbols if no suitable candidates are found.
        """
        engine = self.engine
        if engine.current_symbols or pause_trading is True:
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
                fallback_symbols.append({"symbol": sym, "timeframe": default_tf})
            if len(fallback_symbols) >= engine.effective_max_symbols:
                break
        if fallback_symbols:
            existing_symbols = {c['symbol']: c for c in engine.current_symbols}
            for entry in fallback_symbols:
                if entry['symbol'] in existing_symbols and 'entry_time' in existing_symbols[entry['symbol']]:
                    entry['entry_time'] = existing_symbols[entry['symbol']]['entry_time']
                else:
                    entry['entry_time'] = time.time()
            engine.current_symbols = fallback_symbols
        elif old_symbols:
            logger.warning("Fallback found no symbols. Keeping previously tracked symbols.")
            engine.current_symbols = old_symbols
            engine.effective_max_symbols = max(len(old_symbols), 1)

    def parse_and_validate_symbols(
        self,
        response: str,
        sample_pairs: List[str],
        ohlcv_data: Dict[str, Dict[str, List[List]]],
    ) -> Optional[List[Dict[str, str]]]:
        """Parse the LLM stock selection response and validate symbols.

        Returns a list of validated symbol entries (dicts with 'symbol' and
        'timeframe' keys), or None if parsing fails.
        """
        engine = self.engine
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError:
            logger.error("Failed to parse symbol selection response.")
            return None

        new_symbols: List[Dict[str, str]] = []

        if isinstance(parsed, dict):
            stocks_list = parsed.get("stocks", [])
            if not isinstance(stocks_list, list):
                logger.error("LLM symbol selection 'stocks' field is not a list.")
                stocks_list = []
            for item in stocks_list:
                if isinstance(item, dict) and "symbol" in item:
                    sym = item["symbol"]
                    normalized = engine._normalize_llm_symbol(sym, sample_pairs)
                    if normalized:
                        sym = normalized
                        tf = item.get("timeframe")
                        if tf not in settings.OHLCV_TIMEFRAMES:
                            tf = settings.OHLCV_TIMEFRAMES[0] if settings.OHLCV_TIMEFRAMES else "1h"
                        entry = {"symbol": sym, "timeframe": tf}
                        sector = item.get("sector")
                        if sector:
                            entry["sector"] = sector
                        mth = item.get("max_tenure_hours")
                        if mth is not None:
                            entry["max_tenure_hours"] = mth
                        new_symbols.append(entry)
                elif isinstance(item, str):
                    normalized = engine._normalize_llm_symbol(item, sample_pairs)
                    if normalized:
                        default_tf = settings.OHLCV_TIMEFRAMES[0] if settings.OHLCV_TIMEFRAMES else "1h"
                        new_symbols.append({"symbol": normalized, "timeframe": default_tf})
        elif isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict) and "symbol" in item:
                    sym = item["symbol"]
                    normalized = engine._normalize_llm_symbol(sym, sample_pairs)
                    if normalized:
                        sym = normalized
                        tf = item.get("timeframe")
                        if tf not in settings.OHLCV_TIMEFRAMES:
                            tf = settings.OHLCV_TIMEFRAMES[0] if settings.OHLCV_TIMEFRAMES else "1h"
                        entry = {"symbol": sym, "timeframe": tf}
                        sector = item.get("sector")
                        if sector:
                            entry["sector"] = sector
                        mth = item.get("max_tenure_hours")
                        if mth is not None:
                            entry["max_tenure_hours"] = mth
                        new_symbols.append(entry)
                elif isinstance(item, str):
                    normalized = engine._normalize_llm_symbol(item, sample_pairs)
                    if normalized:
                        default_tf = settings.OHLCV_TIMEFRAMES[0] if settings.OHLCV_TIMEFRAMES else "1h"
                        new_symbols.append({"symbol": normalized, "timeframe": default_tf})
        else:
            logger.error("LLM symbol selection response is neither a list nor a dict.")

        # Deduplicate by symbol, keeping first occurrence
        seen = set()
        deduped = []
        for entry in new_symbols:
            sym = entry["symbol"]
            if sym not in seen:
                seen.add(sym)
                deduped.append(entry)

        # Remove excluded pairs
        deduped = [
            e for e in deduped
            if not engine._is_excluded(e["symbol"], e["timeframe"])
        ]

        # Validate that each selected symbol/timeframe has OHLCV data;
        # fall back to an available timeframe or skip the symbol entirely
        validated_deduped = []
        for entry in deduped:
            sym = entry["symbol"]
            tf = entry["timeframe"]
            sym_data = ohlcv_data.get(sym, {})
            if tf in sym_data and sym_data[tf]:
                validated_deduped.append(entry)
            else:
                available_tfs = [t for t in settings.OHLCV_TIMEFRAMES if t in sym_data and sym_data[t]]
                if available_tfs:
                    entry["timeframe"] = available_tfs[0]
                    validated_deduped.append(entry)
                    logger.info(f"No OHLCV data for {sym} on {tf}, falling back to {available_tfs[0]}")
                else:
                    logger.warning(f"Skipping {sym}: no OHLCV data available for any timeframe")

        return validated_deduped

    async def retry_json_parsing(
        self,
        response: str,
        effective_temp: float,
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Retry JSON parsing if the first attempt fails.

        Returns (response, llm_provider, llm_model).
        If the retry also fails, returns (None, None, None).
        """
        engine = self.engine
        logger.warning("LLM symbol selection response was not valid JSON. Retrying with correction prompt.")
        correction_prompt = (
            "Your previous response was not valid JSON. "
            "You MUST output ONLY a single JSON object, with no markdown fences, no explanations, no extra text. "
            f"Here is your previous response:\n\n{response}"
        )
        try:
            correction_result = await asyncio.wait_for(
                asyncio.to_thread(
                    get_cached_llm_response, compact_prompt(correction_prompt), compact_prompt(build_system_prompt()), 120,
                    model_type="actuator",
                    temperature=effective_temp,
                ),
                timeout=settings.LLM_TIMEOUT
            )
            response = correction_result["response"]
            llm_provider = correction_result["provider"]
            llm_model = correction_result["model"]
            json.loads(response)  # validate the retry response
            return response, llm_provider, llm_model
        except Exception as e:
            logger.error(f"LLM symbol selection still invalid after retry: {e}")
            return None, None, None
