"""Symbol re-evaluation component for the TradingEngine.

Handles asset discovery, quote fetching, sentiment, correlation, LLM chunking,
final selection, pause/resume, and state cleanup.
Extracted from TradingEngine to reduce class size and improve maintainability.
"""
import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from src.config.settings import settings
from src.database import get_ohlcv, get_indicators_for_symbols, get_aggregate_sentiment_for_symbols, get_aggregate_sentiment_from_db
from src.exchanges.market_data import get_quotes_cached

try:
    from src.news.fetcher import discover_trending_stocks, discover_tickers_from_news
except ImportError:
    discover_trending_stocks = None
    discover_tickers_from_news = None

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
