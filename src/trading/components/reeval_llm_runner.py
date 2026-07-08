"""Handles LLM chunk evaluation and final selection for symbol re-evaluation."""
import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from src.config.settings import settings
from src.llm.cache import get_cached_llm_response, compute_market_hash
from src.llm.prompts import build_stock_selection_prompt, build_system_prompt, compact_prompt

logger = logging.getLogger(__name__)


class ReevalLLMRunner:
    """Runs LLM chunk evaluations and the final selection call for re-evaluation."""

    def __init__(self, engine, event_bus):
        self.engine = engine
        self.event_bus = event_bus

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

        semaphore = asyncio.Semaphore(5)  # Limit to 5 concurrent chunk evaluations

        async def _evaluate_chunk(chunk_idx: int, chunk_symbols: List[str]) -> Optional[Dict[str, Any]]:
            async with semaphore:
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
                        logger.info("Chunk %d/%d: received %d symbol selections", chunk_idx + 1, len(chunks), len(chunk_parsed.get("stocks", [])))
                        return chunk_parsed
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
                            logger.info("Chunk %d/%d: corrected, received %d selections", chunk_idx + 1, len(chunks), len(chunk_parsed.get("stocks", [])))
                            return chunk_parsed
                        except Exception as e:
                            logger.warning(f"Chunk {chunk_idx + 1}/{len(chunks)}: correction also failed: {e}")
                else:
                    logger.warning(f"Chunk {chunk_idx + 1}/{len(chunks)}: no response, skipping.")

                return None

        tasks = [_evaluate_chunk(idx, chunk) for idx, chunk in enumerate(chunks)]
        results = await asyncio.gather(*tasks)

        # Filter out None results (failed chunks) and maintain order
        chunk_results: List[Dict[str, Any]] = [r for r in results if r is not None]
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
