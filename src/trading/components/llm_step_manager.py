"""LLM Step 1a and 1b call management."""
import asyncio
import json
import logging
import time
from typing import Any, Dict, Optional, Tuple

from src.config.settings import settings
from src.llm.cache import get_cached_llm_response, compute_market_hash
from src.llm.prompts import compact_prompt, build_system_prompt, build_backtest_variants_prompt, BacktestPromptData, build_analysis_messages
from src.llm.backtest_prompts import build_backtest_variants_messages
from src.strategies.base import Signal
from src.strategies.llm_parser import create_strategy_from_llm, LLMStrategy

logger = logging.getLogger(__name__)


class LLMStepManager:
    """Handles LLM Step 1a and 1b calls, including fallbacks and circuit breaker."""

    def __init__(self, signal_processor):
        self.sp = signal_processor
        self.engine = signal_processor.engine
        self.shared_state = self.engine.shared_state
        self.event_bus = signal_processor.event_bus

    def _create_fallback_hold_signal(
        self, symbol: str, reason: str, strategy_model_type: str
    ) -> Signal:
        """Create a fallback HOLD signal when LLM calls fail or return unparseable JSON."""
        signal = Signal(
            action="HOLD",
            confidence=0.0,
            reasoning=reason,
        )
        signal.model_type = strategy_model_type
        signal.llm_provider = "fallback"
        signal.llm_model = "default_hold"
        return signal

    def _update_last_eval_snapshot(self, symbol: str, price: float, rsi: Optional[float], macd_hist: Optional[float], tf_seconds: int):
        self.shared_state._last_eval_snapshot[symbol] = {
            "timestamp": time.time(),
            "price": price,
            "rsi": rsi,
            "macd_hist": macd_hist,
            "timeframe_seconds": tf_seconds,
        }

    async def run_step1a_llm_call(
        self,
        symbol: str,
        display_symbol: str,
        analysis_prompt: str,
        system_prompt: str,
        market_hash: str,
        strategy_model_type: str,
        effective_temp: float,
        current_price: float,
        rsi: Optional[float],
        macd_hist: Optional[float],
        is_critical: bool,
        critical_reason: Optional[str],
        tf_seconds: int,
        reasoning_effort: str = "low",
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str], bool, bool]:
        """Run the Step 1a LLM call and handle timeouts/retries.

        Returns (analysis_result, llm_provider, llm_model, should_return, is_fallback).
        If should_return is True, the caller should return immediately.
        """
        engine = self.engine
        analysis_result = None
        llm_provider = None
        llm_model = None

        try:
            step1a_result = await asyncio.to_thread(
                get_cached_llm_response,
                "", "", None,
                market_hash=market_hash,
                model_type=strategy_model_type,
                temperature=effective_temp,
                symbol=symbol,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": compact_prompt(analysis_prompt)},
                ],
                request_type="trading_decision_step1a",
                reasoning_effort=reasoning_effort,
            )
            step1a_response = step1a_result["response"]
            llm_provider = step1a_result["provider"]
            llm_model = step1a_result["model"]
            is_fallback = step1a_result.get("is_fallback", False)
            logger.info(f"LLM Step 1a (analysis) completed for {symbol} (provider={llm_provider}, model={llm_model})")
            analysis_result = self.sp._parse_analysis_response(step1a_response)
            if analysis_result is None:
                logger.warning(f"Failed to parse Step 1a analysis response for {symbol}. Retrying with correction.")
                correction_prompt = (
                    "Your previous response was not valid JSON. "
                    "You MUST output ONLY a single JSON object with fields: "
                    '"action", "confidence", "reasoning", "strategy_direction". '
                    "No markdown fences, no explanations, no extra text. "
                    "Here is the original request:\n\n" + analysis_prompt
                )
                retry_result = await asyncio.to_thread(
                    get_cached_llm_response,
                    "", "", None,
                    model_type=strategy_model_type,
                    temperature=effective_temp,
                    market_hash=market_hash,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": compact_prompt(correction_prompt)},
                    ],
                    request_type="trading_decision_step1a_retry",
                )
                analysis_result = self.sp._parse_analysis_response(retry_result["response"])
                llm_provider = retry_result["provider"]
                llm_model = retry_result["model"]
                is_fallback = retry_result.get("is_fallback", False)
            if analysis_result is not None:
                logger.info(f"Step 1a result for {symbol}: action={analysis_result.get('action')}, confidence={analysis_result.get('confidence', 0):.2f}")
            # Reset consecutive LLM failure counter on success
            try:
                await asyncio.to_thread(engine.redis.delete, "llm:consecutive_failures")
            except (ConnectionError, TimeoutError, OSError):
                pass
            # Update snapshot after a real LLM call
            self._update_last_eval_snapshot(symbol, current_price, rsi, macd_hist, tf_seconds)
            async with self.shared_state._eval_state_lock:
                self.shared_state._force_eval.pop(symbol, None)
        except (ConnectionError, TimeoutError, OSError) as e:
            logger.error(f"LLM Step 1a network error for {symbol}: {type(e).__name__}: {e}")
            async with self.shared_state._eval_state_lock:
                self.shared_state._force_eval[symbol] = True  # Force retry on next cycle
            # Per-model blacklisting in cache.py handles failures
            # Fall through to fallback HOLD below
        except (ValueError, TypeError, RuntimeError, json.JSONDecodeError) as e:
            logger.error(f"LLM Step 1a parse/logic error for {symbol}: {type(e).__name__}: {e}")
            # Do not force retry or increment failures for parse errors, as the model is reachable.
            # Fall through to fallback HOLD below

        return analysis_result, llm_provider, llm_model, False, is_fallback

    async def handle_step1a_fallback(
        self,
        symbol: str,
        analysis_result: Optional[Dict[str, Any]],
        has_position: bool,
        strategy_model_type: str,
        llm_provider: Optional[str],
        llm_model: Optional[str],
        is_fallback: bool = False,
    ) -> Tuple[Signal, str, Optional[str], Optional[str], bool, bool]:
        """Handle fallback HOLD signal when Step 1a fails or returns HOLD with no position.

        Returns (signal, combined_bt_summary, llm_provider, llm_model, skip_backtest).
        If signal is None, the caller should proceed with Step 1b and Step 2.
        """
        engine = self.engine
        if analysis_result is None:
            logger.warning(f"Step 1a analysis failed for {symbol} after all retries. Using fallback HOLD.")
            # Check if circuit breaker is active; if so, clear _force_eval to break retry loop
            cb_active = False
            try:
                cb_raw = await asyncio.to_thread(engine.redis.get, "llm:circuit_breaker")
                if cb_raw:
                    cb_data = json.loads(cb_raw)
                    if time.time() < cb_data.get("active_until", 0):
                        cb_active = True
            except (ValueError, TypeError, ConnectionError, TimeoutError, OSError, json.JSONDecodeError):
                pass
            if cb_active:
                async with self.shared_state._eval_state_lock:
                    self.shared_state._force_eval.pop(symbol, None)
            # Otherwise, keep _force_eval set to retry on the next cycle.
            # Create a fallback HOLD signal so the bot continues functioning
            preliminary_signal = self._create_fallback_hold_signal(
                symbol, "LLM Step 1a analysis failed after retries", strategy_model_type
            )
            signal = preliminary_signal
            llm_provider = "fallback"
            llm_model = "default_hold"
            combined_bt_summary = ""
            _skip_backtest = True
            is_fallback = True
        # If analysis says HOLD with no position, only skip backtesting if confidence is very high
        elif analysis_result.get("action") == "HOLD" and not has_position:
            hold_confidence = analysis_result.get("confidence", 0.0)
            if hold_confidence >= 0.85:
                logger.info(f"Step 1a analysis returned HOLD with high confidence ({hold_confidence:.2f}) and no position for {symbol}. Skipping Step 1b.")
                # Create a minimal preliminary signal for the notification flow
                preliminary_signal = Signal(
                    action="HOLD",
                    confidence=hold_confidence,
                    reasoning=analysis_result.get("reasoning", ""),
                )
                preliminary_signal.model_type = strategy_model_type
                preliminary_signal.llm_provider = llm_provider or "fallback"
                preliminary_signal.llm_model = llm_model or "default_hold"
                # Skip backtests and Step 2 — go directly to notification
                signal = preliminary_signal
                combined_bt_summary = ""
                _skip_backtest = True
            else:
                logger.info(f"Step 1a returned HOLD with low confidence ({hold_confidence:.2f}) for {symbol}. Proceeding to Step 1b/Step 2 with backtests.")
                signal = None
                combined_bt_summary = ""
                _skip_backtest = False
        else:
            signal = None
            combined_bt_summary = ""
            _skip_backtest = False

        return signal, combined_bt_summary, llm_provider, llm_model, _skip_backtest, is_fallback

    async def run_step1b_llm_call(
        self,
        symbol: str,
        analysis_result: Dict[str, Any],
        ticker: Dict[str, Any],
        current_price: float,
        atr: Optional[float],
        assigned_tf: str,
        base_balance: float,
        per_symbol_budget: float,
        min_order_amount: Optional[float],
        min_order_cost: Optional[float],
        remaining: float,
        portfolio_total_value: float,
        portfolio_exposure_pct: float,
        portfolio_stop_risk_pct: float,
        portfolio_available_capital: float,
        max_port_exp: Optional[float],
        max_port_risk: Optional[float],
        global_risk_mult: Optional[float],
        min_stop_atr_mult: float,
        min_hold_time_mult: float,
        trading_paused: bool,
        has_position: bool,
        strategy_model_type: str,
        effective_temp: float,
        market_snapshot: Dict[str, Any],
        historical_backtest_results: Optional[list],
        is_critical: bool = False,
        reasoning_effort: str = "low",
    ) -> Tuple[Signal, Optional[str], Optional[str], bool]:
        """Run the Step 1b LLM call for backtest variants and parameters.

        Returns (preliminary_signal, llm_provider, llm_model, is_fallback).
        """
        engine = self.engine
        llm_provider = None
        llm_model = None
        is_fallback = False

        # --- Build variants prompt ---
        prompt_data = BacktestPromptData(
            symbol=symbol,
            analysis=analysis_result,
            ticker=ticker,
            current_price=current_price,
            atr=atr,
            assigned_timeframe=assigned_tf,
            base_currency=engine.base_currency,
            base_balance=base_balance,
            per_symbol_budget=per_symbol_budget,
            min_order_amount=min_order_amount,
            min_order_cost=min_order_cost,
            remaining_balance=remaining,
            portfolio_total_value=portfolio_total_value,
            portfolio_exposure_pct=portfolio_exposure_pct,
            portfolio_stop_risk_pct=portfolio_stop_risk_pct,
            portfolio_available_capital=portfolio_available_capital,
            max_portfolio_exposure_pct=max_port_exp,
            max_portfolio_stop_risk_pct=max_port_risk,
            global_risk_multiplier=global_risk_mult,
            min_stop_atr_mult=min_stop_atr_mult,
            min_hold_time_mult=min_hold_time_mult,
            trading_paused=trading_paused,
            has_position=has_position,
            historical_backtest_results=historical_backtest_results,
        )
        variants_prompt = await asyncio.to_thread(
            build_backtest_variants_prompt,
            prompt_data
        )
        logger.info(f"LLM Step 1b variants prompt for {symbol}: {len(variants_prompt)} chars")

        # Build messages for prompt caching (system + user)
        variants_messages = await asyncio.to_thread(
            build_backtest_variants_messages,
            prompt_data
        )

        # Use a different market hash for Step 1b (include analysis to differentiate)
        variants_market_hash = compute_market_hash({
            **market_snapshot,
            "step": "1b",
            "analysis": analysis_result,
        })

        try:
            step1b_result = await asyncio.to_thread(
                get_cached_llm_response,
                "", "", None,
                market_hash=variants_market_hash,
                model_type=strategy_model_type,
                temperature=effective_temp,
                symbol=symbol,
                messages=variants_messages,
                request_type="trading_decision_step1b",
                reasoning_effort=reasoning_effort,
            )
            step1b_response = step1b_result["response"]
            llm_provider = step1b_result["provider"]
            llm_model = step1b_result["model"]
            is_fallback = step1b_result.get("is_fallback", False)
            logger.info(f"LLM Step 1b (variants) completed for {symbol} (provider={llm_provider}, model={llm_model})")
        except (ConnectionError, TimeoutError, OSError, ValueError, TypeError, RuntimeError, json.JSONDecodeError) as e:
            logger.error(f"LLM Step 1b failed for {symbol}: {type(e).__name__}: {e}. Using Step 1a analysis as fallback.")
            step1b_response = json.dumps({
                "action": analysis_result.get("action", "HOLD"),
                "confidence": analysis_result.get("confidence", 0.0),
                "reasoning": analysis_result.get("reasoning", ""),
                "strategy": {
                    "type": "fallback",
                    "parameters": {},
                },
            })

        # --- Parse Step 1b response ---
        try:
            preliminary_strategy = create_strategy_from_llm(step1b_response)
        except ValueError as e:
            logger.warning(f"LLM Step 1b response parse failed for {symbol}: {e}. Retrying with correction prompt.")
            correction_prompt = (
                "Your previous response was not valid JSON. "
                "You MUST output ONLY a single JSON object, with no markdown fences, no explanations, no extra text. "
                "Here is the original request:\n\n" + variants_prompt
            )
            try:
                response2 = await asyncio.to_thread(
                    get_cached_llm_response, "", "", None,
                    model_type=strategy_model_type,
                    temperature=effective_temp,
                    market_hash=variants_market_hash,
                    messages=[
                        {"role": "system", "content": compact_prompt(build_system_prompt())},
                        {"role": "user", "content": compact_prompt(correction_prompt)},
                    ],
                    request_type="trading_decision_step1b_retry",
                )
                preliminary_strategy = create_strategy_from_llm(response2["response"])
                llm_provider = response2["provider"]
                llm_model = response2["model"]
                is_fallback = response2.get("is_fallback", False)
            except (ConnectionError, TimeoutError, OSError, ValueError, TypeError, RuntimeError, json.JSONDecodeError) as e2:
                logger.error(f"LLM Step 1b response still invalid after retry for {symbol}: {type(e2).__name__}: {e2}")
                preliminary_strategy = LLMStrategy(self._create_fallback_hold_signal(
                    symbol, "Failed to parse LLM Step 1b response after retry", strategy_model_type
                ))

        preliminary_signal = preliminary_strategy.generate_signal({})
        preliminary_signal.model_type = strategy_model_type
        preliminary_signal.llm_provider = llm_provider
        preliminary_signal.llm_model = llm_model

        return preliminary_signal, llm_provider, llm_model, is_fallback
