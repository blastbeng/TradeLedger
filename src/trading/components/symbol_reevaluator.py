"""Symbol re-evaluation component for the TradingEngine.

Handles asset discovery, quote fetching, sentiment, correlation, LLM chunking,
final selection, pause/resume, and state cleanup.
Extracted from TradingEngine to reduce class size and improve maintainability.
"""
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.config.settings import settings
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


@dataclass
class ReevalContext:
    """Context object to hold state between re-evaluation phases."""
    force: bool = False
    is_user_forced: bool = False
    is_market_condition_trigger: bool = False
    is_rebalance: bool = False
    now: float = 0.0
    available_pairs: List[str] = field(default_factory=list)
    btp_pairs: List[str] = field(default_factory=list)
    etf_pairs: List[str] = field(default_factory=list)
    old_symbols: List[Dict[str, str]] = field(default_factory=list)
    last_key: str = ""
    balance: float = 0.0
    base_balance: float = 0.0
    per_symbol_budget: float = 0.0
    tickers: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    sample_pairs: List[str] = field(default_factory=list)
    stock_pairs: List[str] = field(default_factory=list)
    btp_ytm: Any = None
    sorted_by_vol: List[str] = field(default_factory=list)
    news_sentiment: Dict = field(default_factory=dict)
    sentiment_trend: Any = None
    market_trend: Any = None
    symbol_indicators: Dict = field(default_factory=dict)
    symbol_trend_scores: Dict = field(default_factory=dict)
    ohlcv_data: Dict[str, Dict[str, List[List]]] = field(default_factory=dict)
    available_timeframes_by_symbol: Dict[str, List[str]] = field(default_factory=dict)
    market_limits: Dict[str, Dict[str, float]] = field(default_factory=dict)
    min_viable_amount: float = 0.0
    perf: Any = None
    trade_pattern_analysis: Any = None
    correlation_matrix: Any = None
    composite_scores: Dict[str, float] = field(default_factory=dict)
    shortlist: List[str] = field(default_factory=list)
    incremental_offset: int = 0
    incremental_batch_size: Optional[int] = None
    sorted_by_composite: List[str] = field(default_factory=list)
    response: Optional[str] = None
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    effective_temp: Optional[float] = None
    trading_paused_bool: Optional[bool] = None
    model_type: str = "actuator"
    parsed: Dict[str, Any] = field(default_factory=dict)
    pause_trading: Optional[bool] = None
    pause_reason: str = ""
    pause_duration: Optional[Any] = None
    deduped: List[Dict[str, str]] = field(default_factory=list)


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

    async def check_cooldown_and_reset(self, ctx: ReevalContext) -> bool:
        """Check re-evaluation cooldown and reset per-cycle spending.

        Resets _cycle_spent from queued buy orders, checks the triggered
        re-evaluation cooldown for market-condition triggers, clears
        pre-market and user-forced flags, and checks the last eval interval.

        Returns False if re-evaluation should be skipped.
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
        ctx.is_market_condition_trigger = ctx.force and not engine._pre_market_reeval and not engine._user_forced_reeval and not engine._rebalance_reeval

        if ctx.is_market_condition_trigger:
            last_triggered = await asyncio.to_thread(engine.redis.get, "trading:last_triggered_reeval")
            if last_triggered:
                elapsed = time.time() - float(last_triggered)
                if elapsed < settings.TRIGGERED_REEVALUATION_COOLDOWN:
                    logger.info(f"Forced re-evaluation skipped: triggered cooldown active ({settings.TRIGGERED_REEVALUATION_COOLDOWN - elapsed:.0f}s remaining)")
                    return False

        ctx.is_user_forced = engine._user_forced_reeval
        ctx.is_rebalance = engine._rebalance_reeval
        # Clear the pre-market flag after reading it
        engine._pre_market_reeval = False
        # Clear the user-forced flag after reading it
        engine._user_forced_reeval = False
        engine._rebalance_reeval = False

        # Only re-evaluate every SYMBOL_REVALUATION_INTERVAL
        last_key = "trading:last_symbol_eval"
        last_eval = await asyncio.to_thread(engine.redis.get, last_key)
        ctx.now = time.time()
        if last_eval and (ctx.now - float(last_eval)) < engine._symbol_reevaluation_interval and self.shared_state.current_symbols and not ctx.force:
            logger.info("Skipping symbol re-evaluation: last eval was recent and symbols are already loaded.")
            return False

        return True

    async def process_llm_response(self, ctx: ReevalContext) -> None:
        """Process the LLM response, parse symbols, and handle pause/resume logic."""
        engine = self.engine
        response = ctx.response

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
                    ("Keeping previously tracked symbols." if ctx.old_symbols else "Will attempt fallback selection."),
                    summary={
                        "action": "ERROR",
                        "reason": "LLM symbol selection failed after all retries",
                        "model_type": ctx.model_type,
                    }
                )

        # Initialize variables that may be used later even if LLM fails
        ctx.parsed = {}
        ctx.pause_trading = None
        ctx.pause_reason = ""
        ctx.pause_duration = None
        deduped: List[Dict[str, str]] = []

        # Retry JSON parsing if the first attempt fails
        if response is not None:
            try:
                json.loads(response)  # validate
            except json.JSONDecodeError:
                ctx.response, ctx.llm_provider, ctx.llm_model = await self.response_processor.retry_json_parsing(
                    response=response,
                    effective_temp=ctx.effective_temp,
                    is_user_forced=ctx.is_user_forced,
                )
                response = ctx.response

        if response is not None:
            try:
                ctx.parsed = json.loads(response)
                if not isinstance(ctx.parsed, dict):
                    logger.warning(
                        "LLM symbol selection response is not a JSON object (got %s). "
                        "Treating as empty dict.",
                        type(ctx.parsed).__name__
                    )
                    ctx.parsed = {}
                llm_max_stocks = ctx.parsed.get("max_stocks")
                deduped = self.response_processor.parse_and_validate_symbols(
                    response=response,
                    sample_pairs=ctx.sample_pairs,
                    ohlcv_data=ctx.ohlcv_data,
                )
                if deduped is None:
                    deduped = []

                # --- Extract pause_trading early so MIN_SYMBOLS enforcement can respect it ---
                ctx.pause_trading = ctx.parsed.get("pause_trading")
                if isinstance(ctx.pause_trading, str):
                    low = ctx.pause_trading.strip().lower()
                    if low in ("true", "1"):
                        ctx.pause_trading = True
                    elif low in ("false", "0"):
                        ctx.pause_trading = False
                    else:
                        ctx.pause_trading = None

                # Use the LLM's chosen number of symbols to update effective_max_symbols
                if llm_max_stocks is not None and isinstance(llm_max_stocks, int) and 0 <= llm_max_stocks <= engine.max_symbols:
                    # Don't allow 0 unless the LLM explicitly paused trading
                    if llm_max_stocks == 0 and not ctx.pause_trading:
                        llm_max_stocks = max(1, settings.MIN_SYMBOLS)
                        logger.info(f"LLM set max_stocks=0 without pausing; clamping to {llm_max_stocks}")
                    engine.effective_max_symbols = llm_max_stocks
                else:
                    # Fallback: use the length of the deduped list, capped at the engine's max
                    engine.effective_max_symbols = min(len(deduped), engine.effective_max_symbols)

                self.shortlist_builder.enforce_min_symbols(
                    deduped=deduped,
                    pause_trading=ctx.pause_trading,
                    sorted_by_composite=ctx.sorted_by_composite,
                    market_limits=ctx.market_limits,
                    base_balance=ctx.base_balance,
                    ohlcv_data=ctx.ohlcv_data,
                    tickers=ctx.tickers,
                )

                if ctx.is_rebalance:
                    self.shortlist_builder.enforce_asset_class_allocation(
                        deduped=deduped,
                        etf_pairs=ctx.etf_pairs,
                        btp_pairs=ctx.btp_pairs,
                        ohlcv_data=ctx.ohlcv_data,
                        base_balance=ctx.base_balance,
                        market_limits=ctx.market_limits,
                        tickers=ctx.tickers,
                    )

                # --- Store LLM-decided parameters to Redis ---
                await self.config_manager.store_llm_decided_parameters(ctx.parsed)

                ctx.pause_trading, ctx.pause_reason, ctx.pause_duration = await self.pause_resume_manager.handle_pause_resume_and_risk_multiplier(
                    parsed=ctx.parsed,
                    pause_trading=ctx.pause_trading,
                    trading_paused_bool=ctx.trading_paused_bool,
                )

                await self.shortlist_builder.update_current_symbols(
                    deduped=deduped,
                    old_symbols=ctx.old_symbols,
                )

            except json.JSONDecodeError:
                logger.error("Failed to parse symbol selection response.")

        ctx.deduped = deduped

    async def finalize_reevaluation(self, ctx: ReevalContext) -> None:
        """Apply fallback selection, cleanup, send notifications, and finalize state."""
        engine = self.engine

        await self.shortlist_builder.apply_fallback_selection(
            sample_pairs=ctx.sample_pairs,
            composite_scores=ctx.composite_scores,
            tickers=ctx.tickers,
            market_limits=ctx.market_limits,
            base_balance=ctx.base_balance,
            old_symbols=ctx.old_symbols,
            pause_trading=ctx.pause_trading,
            ohlcv_data=ctx.ohlcv_data,
        )

        await self.post_selection_manager.post_selection_cleanup_and_backfill(
            old_symbols=ctx.old_symbols,
            deduped=ctx.deduped,
            force=ctx.force,
        )

        await self.notifier.build_and_send_reeval_notification(
            base_balance=ctx.base_balance,
            per_symbol_budget=ctx.per_symbol_budget,
            pause_trading=ctx.pause_trading,
            pause_reason=ctx.pause_reason,
            pause_duration=ctx.pause_duration,
            trading_paused_bool=ctx.trading_paused_bool,
            force=ctx.force,
            is_user_forced=ctx.is_user_forced,
            parsed=ctx.parsed,
            llm_provider=ctx.llm_provider,
            llm_model=ctx.llm_model,
            model_type=ctx.model_type,
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
        if ctx.is_market_condition_trigger:
            await asyncio.to_thread(engine.redis.set, "trading:last_triggered_reeval", str(time.time()))
            await asyncio.to_thread(engine.redis.expire, "trading:last_triggered_reeval", 7200)

        # --- Cleanup stale entries from engine state dicts and caches ---
        await self.post_selection_manager.cleanup_stale_state_entries()

        self.shared_state._state_dirty = True
        logger.info("Re-evaluation complete: %d symbols selected.", len(self.shared_state.current_symbols))
        await asyncio.to_thread(engine.redis.set, ctx.last_key, ctx.now)

    async def _fetch_candidate_data(self, ctx: ReevalContext) -> bool:
        """Phase 1: Check cooldown, fetch candidate assets, quotes, and sort."""
        if not await self.check_cooldown_and_reset(ctx):
            return False
        _assets_result = await self.data_fetcher.fetch_and_filter_candidate_assets(ctx.now)
        if _assets_result is None:
            return False
        ctx.available_pairs, ctx.btp_pairs, ctx.etf_pairs, ctx.old_symbols, ctx.last_key = _assets_result
        _quotes_result = await self.data_fetcher.fetch_quotes_and_sort(
            ctx.available_pairs, ctx.btp_pairs, ctx.etf_pairs, ctx.now, ctx.last_key
        )
        if _quotes_result is None:
            return False
        ctx.balance, ctx.base_balance, ctx.per_symbol_budget, ctx.tickers, ctx.sample_pairs, ctx.stock_pairs, ctx.btp_ytm = _quotes_result
        return True

    async def _fetch_market_data(self, ctx: ReevalContext) -> None:
        """Phase 2: Fetch news sentiment, OHLCV, indicators, and market limits."""
        self._log_step("Batch-fetching news sentiment for %d symbols...", len(ctx.sample_pairs))
        ctx.news_sentiment, ctx.sentiment_trend, ctx.market_trend = await self.data_fetcher.fetch_news_sentiment_and_trends(
            ctx.sample_pairs, ctx.tickers
        )
        self._log_step("Fetching OHLCV from DB for %d symbols...", len(ctx.sorted_by_vol))
        ctx.ohlcv_data, ctx.available_timeframes_by_symbol = await self.data_fetcher.fetch_ohlcv_from_db(ctx.sorted_by_vol)
        self._log_step("Batch-fetching indicators for %d symbols...", len(ctx.sorted_by_vol))
        ctx.symbol_indicators, ctx.symbol_trend_scores = await self.data_fetcher.fetch_indicators_and_trend_scores(
            ctx.sorted_by_vol, ctx.sample_pairs
        )
        ctx.market_limits = await self.data_fetcher.compute_market_limits(ctx.sample_pairs, ctx.tickers)

    async def _compute_analytics_and_shortlist(self, ctx: ReevalContext) -> None:
        """Phase 3: Compute correlation, performance, incremental offset, and shortlist."""
        self._log_step("Computing correlation matrix and performance metrics...")
        ctx.correlation_matrix = await self.data_fetcher.get_or_compute_correlation_matrix(
            ctx.ohlcv_data, ctx.sorted_by_vol
        )
        ctx.incremental_offset = 0
        ctx.incremental_batch_size = None
        if settings.INCREMENTAL_REEVALUATION_ENABLED:
            ctx.incremental_batch_size = settings.INCREMENTAL_REEVALUATION_BATCH_SIZE
            offset_raw = await asyncio.to_thread(self.engine.redis.get, "reeval:incremental_offset")
            if offset_raw:
                try:
                    ctx.incremental_offset = int(offset_raw)
                except (ValueError, TypeError):
                    ctx.incremental_offset = 0
        ctx.composite_scores, ctx.shortlist = self.shortlist_builder.compute_composite_scores_and_shortlist(
            ctx.sample_pairs, ctx.symbol_trend_scores, ctx.news_sentiment, ctx.trade_pattern_analysis, ctx.etf_pairs, ctx.btp_pairs,
            incremental_offset=ctx.incremental_offset,
            incremental_batch_size=ctx.incremental_batch_size,
        )

    async def _run_llm_evaluation(self, ctx: ReevalContext) -> None:
        """Phase 4: Fetch shortlist context, run chunked LLM eval, and final selection."""
        symbol_events, session_info, market_breadth, full_market_breadth, vix = await self.data_fetcher.fetch_shortlist_context(
            ctx.sample_pairs, ctx.tickers, ctx.market_trend
        )
        ctx.trading_paused_bool, symbol_tenure, symbol_max_tenure, auto_resume_note, ohlcv_summary, ctx.effective_temp, reasoning_effort, ctx.model_type = await self.llm_runner.prepare_reeval_prompt_context(
            now=ctx.now,
            sample_pairs=ctx.sample_pairs,
            ohlcv_data=ctx.ohlcv_data,
            sentiment_trend=ctx.sentiment_trend,
            market_breadth=market_breadth,
            is_rebalance=ctx.is_rebalance,
        )
        chunk_results = await self.llm_runner.evaluate_llm_chunks(
            sample_pairs=ctx.sample_pairs,
            tickers=ctx.tickers,
            ohlcv_summary=ohlcv_summary,
            symbol_indicators=ctx.symbol_indicators,
            market_limits=ctx.market_limits,
            symbol_events=symbol_events,
            symbol_trend_scores=ctx.symbol_trend_scores,
            sentiment_trend=ctx.sentiment_trend,
            correlation_matrix=ctx.correlation_matrix,
            ohlcv_data=ctx.ohlcv_data,
            perf=ctx.perf,
            market_trend=ctx.market_trend,
            session_info=session_info,
            market_breadth=market_breadth,
            trading_paused_bool=ctx.trading_paused_bool,
            symbol_tenure=symbol_tenure,
            symbol_max_tenure=symbol_max_tenure,
            vix=vix,
            trade_pattern_analysis=ctx.trade_pattern_analysis,
            min_viable_amount=ctx.min_viable_amount,
            base_balance=ctx.base_balance,
            per_symbol_budget=ctx.per_symbol_budget,
            auto_resume_note=auto_resume_note,
            effective_temp=ctx.effective_temp,
            btp_ytm=ctx.btp_ytm,
            news_sentiment=ctx.news_sentiment,
            is_user_forced=ctx.is_user_forced,
            reasoning_effort=reasoning_effort,
            model_type=ctx.model_type,
        )
        ctx.response, ctx.llm_provider, ctx.llm_model = await self.llm_runner.run_final_selection_llm_call(
            chunk_results=chunk_results,
            sample_pairs=ctx.sample_pairs,
            base_balance=ctx.base_balance,
            per_symbol_budget=ctx.per_symbol_budget,
            perf=ctx.perf,
            market_trend=ctx.market_trend,
            session_info=session_info,
            market_breadth=market_breadth,
            full_market_breadth=full_market_breadth,
            trading_paused_bool=ctx.trading_paused_bool,
            symbol_tenure=symbol_tenure,
            symbol_max_tenure=symbol_max_tenure,
            trade_pattern_analysis=ctx.trade_pattern_analysis,
            vix=vix,
            min_viable_amount=ctx.min_viable_amount,
            market_limits=ctx.market_limits,
            available_timeframes_by_symbol=ctx.available_timeframes_by_symbol,
            auto_resume_note=auto_resume_note,
            effective_temp=ctx.effective_temp,
            news_sentiment=ctx.news_sentiment,
            is_user_forced=ctx.is_user_forced,
            reasoning_effort=reasoning_effort,
            model_type=ctx.model_type,
        )

    async def reevaluate_symbols_impl(self, force: bool = False):
        """Main re-evaluation orchestration: delegates to phase methods."""
        self._step = 0
        self._total_steps = 12
        engine = self.engine
        ctx = ReevalContext(force=force)
        try:
            # Phase 1: Cooldown, assets, quotes
            if not await self._fetch_candidate_data(ctx):
                return

            ctx.sorted_by_vol = ctx.sample_pairs

            # Phase 2: News, OHLCV, indicators, market limits
            await self._fetch_market_data(ctx)

            # Recompute effective_max_symbols and per_symbol_budget
            engine.effective_max_symbols = engine.max_symbols
            ctx.per_symbol_budget = ctx.base_balance / engine.effective_max_symbols
            ctx.min_viable_amount = settings.MIN_VIABLE_TRADE_AMOUNT

            ctx.perf = await engine.event_bus.request("compute_performance_metrics")
            ctx.trade_pattern_analysis = await engine.event_bus.request("compute_trade_pattern_analysis")

            # Phase 3: Correlation, composite scores, shortlist
            await self._compute_analytics_and_shortlist(ctx)
            ctx.sorted_by_composite = sorted(ctx.sample_pairs, key=lambda s: ctx.composite_scores.get(s, 0), reverse=True)
            ctx.sample_pairs = ctx.shortlist
            logger.info(f"LLM candidate list: {len(ctx.sample_pairs)} symbols (will be evaluated in chunks)")

            if settings.INCREMENTAL_REEVALUATION_ENABLED:
                new_offset = ctx.incremental_offset + settings.INCREMENTAL_REEVALUATION_BATCH_SIZE
                await asyncio.to_thread(engine.redis.set, "reeval:incremental_offset", str(new_offset))

            # Phase 4: LLM evaluation
            await self._run_llm_evaluation(ctx)

            # Phase 5: Process response and finalize
            await self.process_llm_response(ctx)

            await self.finalize_reevaluation(ctx)
        except Exception as e:
            logger.error(f"Re-evaluation failed: {type(e).__name__}: {e}", exc_info=True)
            if engine.notifier:
                await engine.notifier.send_notification(
                    f"⚠️ Re-evaluation failed: {type(e).__name__}: {e}. Keeping previously tracked symbols.",
                    summary={"action": "ERROR", "reason": "Re-evaluation failed"}
                )
