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
from src.llm.prompts import build_stock_selection_prompt, build_system_prompt, compact_prompt
from src.llm.cache import get_cached_llm_response, compute_market_hash
from src.trading.components.reeval_config_manager import ReevalConfigManager
from src.trading.components.reeval_data_fetcher import ReevalDataFetcher
from src.trading.components.reeval_market_condition_monitor import ReevalMarketConditionMonitor
from src.trading.components.reeval_llm_runner import ReevalLLMRunner
from src.trading.components.reeval_shortlist_builder import ReevalShortlistBuilder
from src.trading.components.reeval_response_processor import ReevalResponseProcessor
from src.trading.components.reeval_pause_resume_manager import ReevalPauseResumeManager

from src.trading.components.reeval_post_selection_manager import ReevalPostSelectionManager
from src.trading.components.reeval_notifier import ReevalNotifier

logger = logging.getLogger(__name__)


class SymbolReevaluator:
    """Handles symbol re-evaluation for the TradingEngine."""

    def __init__(self, engine, event_bus):
        self.engine = engine
        self.shared_state = engine.shared_state
        self.event_bus = event_bus
        self.config_manager = ReevalConfigManager(engine)
        self.data_fetcher = ReevalDataFetcher(engine, event_bus)
        self.market_condition_monitor = ReevalMarketConditionMonitor(engine, event_bus)
        self.llm_runner = ReevalLLMRunner(engine, event_bus)
        self.shortlist_builder = ReevalShortlistBuilder(engine, event_bus)
        self.response_processor = ReevalResponseProcessor(engine, event_bus)
        self.pause_resume_manager = ReevalPauseResumeManager(engine, event_bus)
        self.post_selection_manager = ReevalPostSelectionManager(engine, event_bus)
        self.notifier = ReevalNotifier(engine, event_bus)
        self.event_bus.subscribe("reevaluate_symbols_impl", self.reevaluate_symbols_impl)
        self.event_bus.subscribe("check_market_conditions", self.market_condition_monitor.check_market_conditions)
        self._step = 0
        self._total_steps = 12

    def _log_step(self, message: str, *args):
        """Logs the current re-evaluation step with an auto-incrementing counter."""
        self._step += 1
        logger.info(f"Re-evaluation step {self._step}/{self._total_steps}: {message}", *args)

    async def check_cooldown_and_reset(
        self, force: bool
    ) -> Optional[Tuple[bool, bool, float, bool]]:
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
        async with self.shared_state._queued_orders_lock:
            queued_buy_total = sum(
                q.get('amount', 0.0) for q in self.shared_state.queued_orders
                if q.get('side') == 'buy'
            )
        async with self.shared_state._cycle_spent_lock:
            self.shared_state._cycle_spent = queued_buy_total
        self._log_step("Checking cooldown and fetching asset lists...")

        # Respect triggered re-evaluation cooldown for market-condition triggers only.
        # Pre-market re-evaluations are always allowed (they are time-critical).
        # Forced re-evaluations (explicit user or critical condition requests) always bypass
        # the cooldown since they are intentionally requested.
        # Capture whether this is a market-condition trigger before clearing flags
        is_market_condition_trigger = force and not engine._pre_market_reeval and not engine._user_forced_reeval and not engine._rebalance_reeval

        if is_market_condition_trigger:
            last_triggered = await asyncio.to_thread(engine.redis.get, "trading:last_triggered_reeval")
            if last_triggered:
                elapsed = time.time() - float(last_triggered)
                if elapsed < settings.TRIGGERED_REEVALUATION_COOLDOWN:
                    logger.info(f"Forced re-evaluation skipped: triggered cooldown active ({settings.TRIGGERED_REEVALUATION_COOLDOWN - elapsed:.0f}s remaining)")
                    return None

        is_user_forced = engine._user_forced_reeval
        is_rebalance = engine._rebalance_reeval
        # Clear the pre-market flag after reading it
        engine._pre_market_reeval = False
        # Clear the user-forced flag after reading it
        engine._user_forced_reeval = False
        engine._rebalance_reeval = False

        # Only re-evaluate every SYMBOL_REVALUATION_INTERVAL
        last_key = "trading:last_symbol_eval"
        last_eval = await asyncio.to_thread(engine.redis.get, last_key)
        now = time.time()
        if last_eval and (now - float(last_eval)) < engine._symbol_reevaluation_interval and self.shared_state.current_symbols and not force:
            logger.info("Skipping symbol re-evaluation: last eval was recent and symbols are already loaded.")
            return None

        return (is_user_forced, is_market_condition_trigger, now, is_rebalance)

    async def process_llm_response(
        self,
        response: Optional[str],
        llm_provider: Optional[str],
        llm_model: Optional[str],
        effective_temp: float,
        sample_pairs: List[str],
        ohlcv_data: Dict[str, Dict[str, List[List]]],
        sorted_by_composite: List[str],
        market_limits: Dict[str, Dict[str, float]],
        base_balance: float,
        old_symbols: List[Dict[str, str]],
        trading_paused_bool: bool,
        etf_pairs: List[str],
        btp_pairs: List[str],
        is_rebalance: bool,
        tickers: Dict[str, Dict[str, Any]],
        is_user_forced: bool = False,
        model_type: str = "actuator",
    ) -> Tuple[Dict[str, Any], Optional[bool], str, Optional[Any], List[Dict[str, str]], Optional[str], Optional[str]]:
        """Process the LLM response, parse symbols, and handle pause/resume logic.

        Returns (parsed, pause_trading, pause_reason, pause_duration, deduped, llm_provider, llm_model).
        """
        engine = self.engine

        logger.info("Re-evaluation: LLM response received (%d chars), parsing...", len(response) if response else 0)
        if response:
            # Truncate long responses to avoid flooding logs with HTML error pages
            if len(response) > 500:
                logger.info("LLM stock selection raw response (truncated): %.500s...", response)
            else:
                logger.info("LLM stock selection raw response: %s", response)
            # Warn if the response looks like HTML (common when the LLM endpoint returns an error page)
            if response.lstrip().startswith('<'):
                logger.warning(
                    "LLM stock selection response appears to be HTML (length %d). "
                    "The LLM endpoint may be returning an error page.",
                    len(response)
                )
        else:
            logger.info("LLM stock selection returned empty response")
            if engine.notifier:
                await engine.notifier.send_notification(
                    "⚠️ LLM symbol selection failed after all retries. " +
                    ("Keeping previously tracked symbols." if old_symbols else "Will attempt fallback selection."),
                    summary={
                        "action": "ERROR",
                        "reason": "LLM symbol selection failed after all retries",
                        "model_type": model_type,
                    }
                )

        # Initialize variables that may be used later even if LLM fails
        parsed = {}
        pause_trading = None
        pause_reason = ""
        pause_duration = None
        new_symbols: List[Dict[str, str]] = []
        deduped: List[Dict[str, str]] = []

        # Retry JSON parsing if the first attempt fails
        if response is not None:
            try:
                json.loads(response)  # validate
            except json.JSONDecodeError:
                response, llm_provider, llm_model = await self.response_processor.retry_json_parsing(
                    response=response,
                    effective_temp=effective_temp,
                    is_user_forced=is_user_forced,
                )

        if response is not None:
            try:
                parsed = json.loads(response)
                if not isinstance(parsed, dict):
                    logger.warning(
                        "LLM symbol selection response is not a JSON object (got %s). "
                        "Treating as empty dict.",
                        type(parsed).__name__
                    )
                    parsed = {}
                llm_max_stocks = parsed.get("max_stocks")
                deduped = self.response_processor.parse_and_validate_symbols(
                    response=response,
                    sample_pairs=sample_pairs,
                    ohlcv_data=ohlcv_data,
                )
                if deduped is None:
                    deduped = []

                # --- Extract pause_trading early so MIN_SYMBOLS enforcement can respect it ---
                pause_trading = parsed.get("pause_trading")
                if isinstance(pause_trading, str):
                    low = pause_trading.strip().lower()
                    if low in ("true", "1"):
                        pause_trading = True
                    elif low in ("false", "0"):
                        pause_trading = False
                    else:
                        pause_trading = None

                # Use the LLM's chosen number of symbols to update effective_max_symbols
                if llm_max_stocks is not None and isinstance(llm_max_stocks, int) and 0 <= llm_max_stocks <= engine.max_symbols:
                    # Don't allow 0 unless the LLM explicitly paused trading
                    if llm_max_stocks == 0 and not pause_trading:
                        llm_max_stocks = max(1, settings.MIN_SYMBOLS)
                        logger.info(f"LLM set max_stocks=0 without pausing; clamping to {llm_max_stocks}")
                    engine.effective_max_symbols = llm_max_stocks
                else:
                    # Fallback: use the length of the deduped list, capped at the engine's max
                    engine.effective_max_symbols = min(len(deduped), engine.effective_max_symbols)

                self.shortlist_builder.enforce_min_symbols(
                    deduped=deduped,
                    pause_trading=pause_trading,
                    sorted_by_composite=sorted_by_composite,
                    market_limits=market_limits,
                    base_balance=base_balance,
                    ohlcv_data=ohlcv_data,
                    tickers=tickers,
                )

                if is_rebalance:
                    self.shortlist_builder.enforce_asset_class_allocation(
                        deduped=deduped,
                        etf_pairs=etf_pairs,
                        btp_pairs=btp_pairs,
                        ohlcv_data=ohlcv_data,
                        base_balance=base_balance,
                        market_limits=market_limits,
                        tickers=tickers,
                    )

                # --- Store LLM-decided parameters to Redis ---
                await self.config_manager.store_llm_decided_parameters(parsed)

                pause_trading, pause_reason, pause_duration = await self.pause_resume_manager.handle_pause_resume_and_risk_multiplier(
                    parsed=parsed,
                    pause_trading=pause_trading,
                    trading_paused_bool=trading_paused_bool,
                )

                self.shortlist_builder.update_current_symbols(
                    deduped=deduped,
                    old_symbols=old_symbols,
                )

            except json.JSONDecodeError:
                logger.error("Failed to parse symbol selection response.")

        return parsed, pause_trading, pause_reason, pause_duration, deduped, llm_provider, llm_model

    async def finalize_reevaluation(
        self,
        sample_pairs: List[str],
        composite_scores: Dict[str, float],
        tickers: Dict[str, Dict[str, Any]],
        market_limits: Dict[str, Dict[str, float]],
        base_balance: float,
        old_symbols: List[Dict[str, str]],
        deduped: List[Dict[str, str]],
        pause_trading: Optional[bool],
        pause_reason: str,
        pause_duration: Optional[Any],
        trading_paused_bool: bool,
        force: bool,
        is_user_forced: bool,
        parsed: Dict[str, Any],
        llm_provider: Optional[str],
        llm_model: Optional[str],
        model_type: str = "actuator",
        is_market_condition_trigger: bool,
        per_symbol_budget: float,
        last_key: str,
        now: float,
        ohlcv_data: Dict[str, Dict[str, List[List]]],
    ) -> None:
        """Apply fallback selection, cleanup, send notifications, and finalize state."""
        engine = self.engine

        await self.shortlist_builder.apply_fallback_selection(
            sample_pairs=sample_pairs,
            composite_scores=composite_scores,
            tickers=tickers,
            market_limits=market_limits,
            base_balance=base_balance,
            old_symbols=old_symbols,
            pause_trading=pause_trading,
            ohlcv_data=ohlcv_data,
        )

        await self.post_selection_manager.post_selection_cleanup_and_backfill(
            old_symbols=old_symbols,
            deduped=deduped,
            force=force,
        )

        await self.notifier.build_and_send_reeval_notification(
            base_balance=base_balance,
            per_symbol_budget=per_symbol_budget,
            pause_trading=pause_trading,
            pause_reason=pause_reason,
            pause_duration=pause_duration,
            trading_paused_bool=trading_paused_bool,
            force=force,
            is_user_forced=is_user_forced,
            parsed=parsed,
            llm_provider=llm_provider,
            llm_model=llm_model,
            model_type=model_type,
        )

        # If no symbols were selected, shorten the re‑evaluation interval to retry sooner.
        if not self.shared_state.current_symbols:
            engine._symbol_reevaluation_interval = max(engine._symbol_reevaluation_interval, settings.MIN_SYMBOL_REEVALUATION_INTERVAL)
            logger.info(f"No symbols selected – next re‑evaluation in {engine._symbol_reevaluation_interval}s")
        # else: keep the current interval (may have been set by LLM via
        # stock_revaluation_interval_seconds, or the default SYMBOL_REEVALUATION_INTERVAL)

        # Set the triggered cooldown key AFTER a successful market-condition-triggered
        # re-evaluation to prevent the market condition monitor from firing again too soon.
        # This must be set at the END, not at the trigger point, otherwise the re-evaluation
        # itself would see the cooldown as active and skip itself.
        if is_market_condition_trigger:
            await asyncio.to_thread(engine.redis.set, "trading:last_triggered_reeval", str(time.time()))
            await asyncio.to_thread(engine.redis.expire, "trading:last_triggered_reeval", 7200)

        # --- Cleanup stale entries from engine state dicts and caches ---
        await self.post_selection_manager.cleanup_stale_state_entries()

        self.shared_state._state_dirty = True
        logger.info("Re-evaluation complete: %d symbols selected.", len(self.shared_state.current_symbols))
        await asyncio.to_thread(engine.redis.set, last_key, now)

    async def _fetch_candidate_data(
        self, force: bool
    ) -> Optional[Tuple[bool, bool, float, bool, List[str], List[str], List[str], List[Dict[str, str]], str, float, float, float, Dict[str, Dict[str, Any]], List[str], List[str], Dict[str, float]]]:
        """Phase 1: Check cooldown, fetch candidate assets, quotes, and sort."""
        _cooldown_result = await self.check_cooldown_and_reset(force)
        if _cooldown_result is None:
            return None
        is_user_forced, is_market_condition_trigger, now, is_rebalance = _cooldown_result
        _assets_result = await self.data_fetcher.fetch_and_filter_candidate_assets(now)
        if _assets_result is None:
            return None
        available_pairs, btp_pairs, etf_pairs, old_symbols, last_key = _assets_result
        _quotes_result = await self.data_fetcher.fetch_quotes_and_sort(
            available_pairs, btp_pairs, etf_pairs, now, last_key
        )
        if _quotes_result is None:
            return None
        balance, base_balance, per_symbol_budget, tickers, sample_pairs, stock_pairs, btp_ytm = _quotes_result
        return (is_user_forced, is_market_condition_trigger, now, is_rebalance,
                available_pairs, btp_pairs, etf_pairs, old_symbols, last_key,
                balance, base_balance, per_symbol_budget, tickers, sample_pairs, stock_pairs, btp_ytm)

    async def _fetch_market_data(
        self,
        sample_pairs: List[str],
        tickers: Dict[str, Dict[str, Any]],
        sorted_by_vol: List[str],
    ) -> Tuple[Dict, Any, Any, Dict, Dict, Dict[str, Dict[str, List[List]]], Dict[str, List[str]], Dict[str, Dict[str, float]]]:
        """Phase 2: Fetch news sentiment, OHLCV, indicators, and market limits."""
        self._log_step("Batch-fetching news sentiment for %d symbols...", len(sample_pairs))
        news_sentiment, sentiment_trend, market_trend = await self.data_fetcher.fetch_news_sentiment_and_trends(
            sample_pairs, tickers
        )
        self._log_step("Fetching OHLCV from DB for %d symbols...", len(sorted_by_vol))
        ohlcv_data, available_timeframes_by_symbol = await self.data_fetcher.fetch_ohlcv_from_db(sorted_by_vol)
        self._log_step("Batch-fetching indicators for %d symbols...", len(sorted_by_vol))
        symbol_indicators, symbol_trend_scores = await self.data_fetcher.fetch_indicators_and_trend_scores(
            sorted_by_vol, sample_pairs
        )
        market_limits = await self.data_fetcher.compute_market_limits(sample_pairs, tickers)
        return news_sentiment, sentiment_trend, market_trend, symbol_indicators, symbol_trend_scores, ohlcv_data, available_timeframes_by_symbol, market_limits

    async def _compute_analytics_and_shortlist(
        self,
        sample_pairs: List[str],
        ohlcv_data: Dict[str, Dict[str, List[List]]],
        sorted_by_vol: List[str],
        symbol_trend_scores: Dict,
        news_sentiment: Dict,
        trade_pattern_analysis: Any,
        etf_pairs: List[str],
        btp_pairs: List[str],
    ) -> Tuple[Any, Dict[str, float], List[str], int, Optional[int]]:
        """Phase 3: Compute correlation, performance, incremental offset, and shortlist."""
        self._log_step("Computing correlation matrix and performance metrics...")
        correlation_matrix = await self.data_fetcher.get_or_compute_correlation_matrix(
            ohlcv_data, sorted_by_vol
        )
        incremental_offset = 0
        incremental_batch_size = None
        if settings.INCREMENTAL_REEVALUATION_ENABLED:
            incremental_batch_size = settings.INCREMENTAL_REEVALUATION_BATCH_SIZE
            offset_raw = await asyncio.to_thread(self.engine.redis.get, "reeval:incremental_offset")
            if offset_raw:
                try:
                    incremental_offset = int(offset_raw)
                except (ValueError, TypeError):
                    incremental_offset = 0
        composite_scores, shortlist = self.shortlist_builder.compute_composite_scores_and_shortlist(
            sample_pairs, symbol_trend_scores, news_sentiment, trade_pattern_analysis, etf_pairs, btp_pairs,
            incremental_offset=incremental_offset,
            incremental_batch_size=incremental_batch_size,
        )
        return correlation_matrix, composite_scores, shortlist, incremental_offset, incremental_batch_size

    async def _run_llm_evaluation(
        self,
        sample_pairs: List[str],
        tickers: Dict[str, Dict[str, Any]],
        ohlcv_data: Dict[str, Dict[str, List[List]]],
        symbol_indicators: Dict,
        market_limits: Dict[str, Dict[str, float]],
        symbol_trend_scores: Dict,
        sentiment_trend: Any,
        correlation_matrix: Any,
        perf: Any,
        market_trend: Any,
        trade_pattern_analysis: Any,
        min_viable_amount: float,
        base_balance: float,
        per_symbol_budget: float,
        btp_ytm: Any,
        news_sentiment: Dict,
        is_user_forced: bool,
        is_rebalance: bool,
        now: float,
        available_timeframes_by_symbol: Dict[str, List[str]],
    ) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[float], Optional[bool], str]:
        """Phase 4: Fetch shortlist context, run chunked LLM eval, and final selection."""
        symbol_events, session_info, market_breadth, full_market_breadth, vix = await self.data_fetcher.fetch_shortlist_context(
            sample_pairs, tickers, market_trend
        )
        trading_paused_bool, symbol_tenure, symbol_max_tenure, auto_resume_note, ohlcv_summary, effective_temp, reasoning_effort, model_type = await self.llm_runner.prepare_reeval_prompt_context(
            now=now,
            sample_pairs=sample_pairs,
            ohlcv_data=ohlcv_data,
            sentiment_trend=sentiment_trend,
            market_breadth=market_breadth,
            is_rebalance=is_rebalance,
        )
        chunk_results = await self.llm_runner.evaluate_llm_chunks(
            sample_pairs=sample_pairs,
            tickers=tickers,
            ohlcv_summary=ohlcv_summary,
            symbol_indicators=symbol_indicators,
            market_limits=market_limits,
            symbol_events=symbol_events,
            symbol_trend_scores=symbol_trend_scores,
            sentiment_trend=sentiment_trend,
            correlation_matrix=correlation_matrix,
            ohlcv_data=ohlcv_data,
            perf=perf,
            market_trend=market_trend,
            session_info=session_info,
            market_breadth=market_breadth,
            trading_paused_bool=trading_paused_bool,
            symbol_tenure=symbol_tenure,
            symbol_max_tenure=symbol_max_tenure,
            vix=vix,
            trade_pattern_analysis=trade_pattern_analysis,
            min_viable_amount=min_viable_amount,
            base_balance=base_balance,
            per_symbol_budget=per_symbol_budget,
            auto_resume_note=auto_resume_note,
            effective_temp=effective_temp,
            btp_ytm=btp_ytm,
            news_sentiment=news_sentiment,
            is_user_forced=is_user_forced,
            reasoning_effort=reasoning_effort,
            model_type=model_type,
        )
        response, llm_provider, llm_model = await self.llm_runner.run_final_selection_llm_call(
            chunk_results=chunk_results,
            sample_pairs=sample_pairs,
            base_balance=base_balance,
            per_symbol_budget=per_symbol_budget,
            perf=perf,
            market_trend=market_trend,
            session_info=session_info,
            market_breadth=market_breadth,
            full_market_breadth=full_market_breadth,
            trading_paused_bool=trading_paused_bool,
            symbol_tenure=symbol_tenure,
            symbol_max_tenure=symbol_max_tenure,
            trade_pattern_analysis=trade_pattern_analysis,
            vix=vix,
            min_viable_amount=min_viable_amount,
            market_limits=market_limits,
            available_timeframes_by_symbol=available_timeframes_by_symbol,
            auto_resume_note=auto_resume_note,
            effective_temp=effective_temp,
            news_sentiment=news_sentiment,
            is_user_forced=is_user_forced,
            reasoning_effort=reasoning_effort,
            model_type=model_type,
        )
        return response, llm_provider, llm_model, effective_temp, trading_paused_bool, model_type

    async def reevaluate_symbols_impl(self, force: bool = False):
        """Main re-evaluation orchestration: delegates to phase methods."""
        self._step = 0
        self._total_steps = 12
        engine = self.engine
        try:
            # Phase 1: Cooldown, assets, quotes
            data = await self._fetch_candidate_data(force)
            if data is None:
                return
            is_user_forced, is_market_condition_trigger, now, is_rebalance, \
                available_pairs, btp_pairs, etf_pairs, old_symbols, last_key, \
                balance, base_balance, per_symbol_budget, tickers, sample_pairs, \
                stock_pairs, btp_ytm = data

            sorted_by_vol = sample_pairs

            # Phase 2: News, OHLCV, indicators, market limits
            news_sentiment, sentiment_trend, market_trend, symbol_indicators, \
                symbol_trend_scores, ohlcv_data, available_timeframes_by_symbol, \
                market_limits = await self._fetch_market_data(sample_pairs, tickers, sorted_by_vol)

            # Recompute effective_max_symbols and per_symbol_budget
            engine.effective_max_symbols = engine.max_symbols
            per_symbol_budget = base_balance / engine.effective_max_symbols
            min_viable_amount = settings.MIN_VIABLE_TRADE_AMOUNT

            perf = await engine.event_bus.request("compute_performance_metrics")
            trade_pattern_analysis = await engine.event_bus.request("compute_trade_pattern_analysis")

            # Phase 3: Correlation, composite scores, shortlist
            correlation_matrix, composite_scores, shortlist, incremental_offset, \
                incremental_batch_size = await self._compute_analytics_and_shortlist(
                    sample_pairs, ohlcv_data, sorted_by_vol, symbol_trend_scores,
                    news_sentiment, trade_pattern_analysis, etf_pairs, btp_pairs
                )
            sorted_by_composite = sorted(sample_pairs, key=lambda s: composite_scores.get(s, 0), reverse=True)
            sample_pairs = shortlist
            logger.info(f"LLM candidate list: {len(sample_pairs)} symbols (will be evaluated in chunks)")

            if settings.INCREMENTAL_REEVALUATION_ENABLED:
                new_offset = incremental_offset + settings.INCREMENTAL_REEVALUATION_BATCH_SIZE
                await asyncio.to_thread(engine.redis.set, "reeval:incremental_offset", str(new_offset))

            # Phase 4: LLM evaluation
            response, llm_provider, llm_model, effective_temp, trading_paused_bool, model_type = await self._run_llm_evaluation(
                sample_pairs=sample_pairs,
                tickers=tickers,
                ohlcv_data=ohlcv_data,
                symbol_indicators=symbol_indicators,
                market_limits=market_limits,
                symbol_trend_scores=symbol_trend_scores,
                sentiment_trend=sentiment_trend,
                correlation_matrix=correlation_matrix,
                perf=perf,
                market_trend=market_trend,
                trade_pattern_analysis=trade_pattern_analysis,
                min_viable_amount=min_viable_amount,
                base_balance=base_balance,
                per_symbol_budget=per_symbol_budget,
                btp_ytm=btp_ytm,
                news_sentiment=news_sentiment,
                is_user_forced=is_user_forced,
                is_rebalance=is_rebalance,
                now=now,
                available_timeframes_by_symbol=available_timeframes_by_symbol,
            )

            # Phase 5: Process response and finalize
            parsed, pause_trading, pause_reason, pause_duration, deduped, \
                llm_provider, llm_model = await self.process_llm_response(
                    response=response,
                    llm_provider=llm_provider,
                    llm_model=llm_model,
                    effective_temp=effective_temp,
                    sample_pairs=sample_pairs,
                    ohlcv_data=ohlcv_data,
                    sorted_by_composite=sorted_by_composite,
                    market_limits=market_limits,
                    base_balance=base_balance,
                    old_symbols=old_symbols,
                    trading_paused_bool=trading_paused_bool,
                    etf_pairs=etf_pairs,
                    btp_pairs=btp_pairs,
                    is_rebalance=is_rebalance,
                    tickers=tickers,
                    is_user_forced=is_user_forced,
                    model_type=model_type,
                )

            await self.finalize_reevaluation(
                sample_pairs=sample_pairs,
                composite_scores=composite_scores,
                tickers=tickers,
                market_limits=market_limits,
                base_balance=base_balance,
                old_symbols=old_symbols,
                deduped=deduped,
                pause_trading=pause_trading,
                pause_reason=pause_reason,
                pause_duration=pause_duration,
                trading_paused_bool=trading_paused_bool,
                force=force,
                is_user_forced=is_user_forced,
                parsed=parsed,
                llm_provider=llm_provider,
                llm_model=llm_model,
                is_market_condition_trigger=is_market_condition_trigger,
                per_symbol_budget=per_symbol_budget,
                last_key=last_key,
                now=now,
                ohlcv_data=ohlcv_data,
                model_type=model_type,
            )
        except Exception as e:
            logger.error(f"Re-evaluation failed: {type(e).__name__}: {e}", exc_info=True)
            if engine.notifier:
                await engine.notifier.send_notification(
                    f"⚠️ Re-evaluation failed: {type(e).__name__}: {e}. Keeping previously tracked symbols.",
                    summary={"action": "ERROR", "reason": "Re-evaluation failed"}
                )
