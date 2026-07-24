"""Handles parsing and validation of LLM responses for symbol re-evaluation."""
import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from src.config.settings import settings
from src.llm.cache import get_cached_llm_response
from src.llm.prompts import build_system_prompt, compact_prompt

logger = logging.getLogger(__name__)


class ReevalResponseProcessor:
    """Parses and validates LLM symbol selection responses."""

    def __init__(self, engine, event_bus):
        self.engine = engine
        self.event_bus = event_bus

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
        is_user_forced: bool = False,
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Retry JSON parsing if the first attempt fails.

        Returns (response, llm_provider, llm_model).
        If the retry also fails, returns (None, None, None).
        """
        logger.warning("LLM symbol selection response was not valid JSON. Retrying with correction prompt.")
        correction_prompt = (
            "Your previous response was not valid JSON. "
            "You MUST output ONLY a single JSON object, with no markdown fences, no explanations, no extra text. "
            f"Here is your previous response:\n\n{response}"
        )
        try:
            correction_result = await asyncio.wait_for(
                asyncio.to_thread(
                    get_cached_llm_response, "", "", 120,
                    model_type="actuator",
                    temperature=effective_temp,
                    messages=[
                        {"role": "system", "content": compact_prompt(build_system_prompt(task_type="stock_selection"))},
                        {"role": "user", "content": compact_prompt(correction_prompt)},
                    ],
                    force_primary_model=is_user_forced,
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
