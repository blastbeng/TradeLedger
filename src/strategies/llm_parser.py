import json
import re
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, ValidationError
from src.config.settings import settings
from .base import Signal, LLMStrategy


def _extract_first_json(text: str) -> dict:
    """Extracts the first valid JSON object from a string by tracking brace depth."""
    start = text.find('{')
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            char = text[i]
            if escape:
                escape = False
                continue
            if char == '\\':
                escape = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    json_str = text[start:i+1]
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        break  # Break inner loop to find next '{'
        start = text.find('{', start + 1)
    raise ValueError("No valid JSON object found in LLM response")


def _validate_semantic_quality(action: str, params: dict, reasoning: str) -> tuple[str, str]:
    """
    Validates semantic quality of LLM parameters to prevent bad trades.
    Returns potentially modified (action, reasoning).
    """
    issues = []
    
    if action == "BUY":
        stop_loss = params.get("stop_loss_pct")
        take_profit = params.get("take_profit_pct")
        stop_loss_atr = params.get("stop_loss_atr_multiple")
        take_profit_atr = params.get("take_profit_atr_multiple")
        position_size = params.get("position_size_fraction")
        trailing_stop = params.get("trailing_stop", False)
        trailing_stop_distance = params.get("trailing_stop_distance_pct")
        trailing_stop_atr = params.get("trailing_stop_atr_multiple")
        trailing_stop_activation = params.get("trailing_stop_activation_pct")
        breakeven_activation = params.get("breakeven_activation_pct")
        max_unrealized_loss = params.get("max_unrealized_loss_pct")
        max_hold_time = params.get("max_hold_time_seconds")
        cooldown = params.get("cooldown_after_loss_seconds")
        
        if stop_loss is not None:
            if stop_loss <= 0 or stop_loss > 0.5:
                issues.append(f"unreasonable stop_loss_pct ({stop_loss})")
        
        if take_profit is not None:
            if take_profit <= 0 or take_profit > 5.0:
                issues.append(f"unreasonable take_profit_pct ({take_profit})")
                
        if stop_loss_atr is not None:
            if stop_loss_atr <= 0 or stop_loss_atr > 10.0:
                issues.append(f"unreasonable stop_loss_atr_multiple ({stop_loss_atr})")
                
        if take_profit_atr is not None:
            if take_profit_atr <= 0 or take_profit_atr > 20.0:
                issues.append(f"unreasonable take_profit_atr_multiple ({take_profit_atr})")
                
        if position_size is not None:
            if position_size <= 0:
                issues.append(f"non-positive position_size_fraction ({position_size})")
                
        if trailing_stop:
            if trailing_stop_distance is None and trailing_stop_atr is None:
                issues.append("trailing_stop enabled but no distance/atr_multiple provided")
            if trailing_stop_distance is not None and (trailing_stop_distance <= 0 or trailing_stop_distance > 0.5):
                issues.append(f"unreasonable trailing_stop_distance_pct ({trailing_stop_distance})")
            if trailing_stop_atr is not None and (trailing_stop_atr <= 0 or trailing_stop_atr > 10.0):
                issues.append(f"unreasonable trailing_stop_atr_multiple ({trailing_stop_atr})")
                
        if trailing_stop_activation is not None:
            if trailing_stop_activation <= 0 or trailing_stop_activation > 5.0:
                issues.append(f"unreasonable trailing_stop_activation_pct ({trailing_stop_activation})")
                
        if breakeven_activation is not None:
            if breakeven_activation <= 0 or breakeven_activation > 5.0:
                issues.append(f"unreasonable breakeven_activation_pct ({breakeven_activation})")
                
        if max_unrealized_loss is not None:
            if max_unrealized_loss <= 0 or max_unrealized_loss > 0.5:
                issues.append(f"unreasonable max_unrealized_loss_pct ({max_unrealized_loss})")
                
        if max_hold_time is not None:
            if max_hold_time <= 0 or max_hold_time > 157_680_000:
                issues.append(f"unreasonable max_hold_time_seconds ({max_hold_time})")
                
        if cooldown is not None and cooldown < 0:
            issues.append(f"negative cooldown_after_loss_seconds ({cooldown})")

        max_risk_per_trade = params.get("max_risk_per_trade_pct")
        max_portfolio_risk_pct = params.get("max_portfolio_risk_pct")
        min_profit_per_trade = params.get("min_profit_per_trade")
        min_risk_reward_ratio = params.get("min_risk_reward_ratio")
        min_confidence = params.get("min_confidence")
        news_sentiment_exit_threshold = params.get("news_sentiment_exit_threshold")
        strategy_interval_seconds = params.get("strategy_interval_seconds")
        backtest_period_days = params.get("backtest_period_days")
        trailing_take_profit_distance = params.get("trailing_take_profit_distance_pct")
        partial_take_profit_pct = params.get("partial_take_profit_pct")
        partial_take_profit_fraction = params.get("partial_take_profit_fraction")
        position_size_multiplier = params.get("position_size_multiplier")
        confidence_sizing_weight = params.get("confidence_sizing_weight")

        if max_risk_per_trade is not None and (max_risk_per_trade <= 0 or max_risk_per_trade > 0.1):
            issues.append(f"unreasonable max_risk_per_trade_pct ({max_risk_per_trade})")
        if max_portfolio_risk_pct is not None and (max_portfolio_risk_pct <= 0 or max_portfolio_risk_pct > 1.0):
            issues.append(f"unreasonable max_portfolio_risk_pct ({max_portfolio_risk_pct})")
        if min_profit_per_trade is not None and (min_profit_per_trade < 0 or min_profit_per_trade > 1.0):
            issues.append(f"unreasonable min_profit_per_trade ({min_profit_per_trade})")
        if min_risk_reward_ratio is not None and (min_risk_reward_ratio < 0 or min_risk_reward_ratio > 10.0):
            issues.append(f"unreasonable min_risk_reward_ratio ({min_risk_reward_ratio})")
        if min_confidence is not None and (min_confidence < 0 or min_confidence > 1.0):
            issues.append(f"unreasonable min_confidence ({min_confidence})")
        if news_sentiment_exit_threshold is not None and (news_sentiment_exit_threshold < -1.0 or news_sentiment_exit_threshold > 0.0):
            issues.append(f"unreasonable news_sentiment_exit_threshold ({news_sentiment_exit_threshold})")
        if strategy_interval_seconds is not None and (strategy_interval_seconds <= 0 or strategy_interval_seconds > 30 * 24 * 3600):
            issues.append(f"unreasonable strategy_interval_seconds ({strategy_interval_seconds})")
        if backtest_period_days is not None and (backtest_period_days < 30 or backtest_period_days > 365 * 10):
            issues.append(f"unreasonable backtest_period_days ({backtest_period_days})")
        order_fill_timeout = params.get("order_fill_timeout_seconds")
        if order_fill_timeout is not None and (order_fill_timeout <= 0 or order_fill_timeout > 3600):
            issues.append(f"unreasonable order_fill_timeout_seconds ({order_fill_timeout})")
        if trailing_take_profit_distance is not None and (trailing_take_profit_distance <= 0 or trailing_take_profit_distance > 0.5):
            issues.append(f"unreasonable trailing_take_profit_distance_pct ({trailing_take_profit_distance})")
        if partial_take_profit_pct is not None and (partial_take_profit_pct <= 0 or partial_take_profit_pct > 5.0):
            issues.append(f"unreasonable partial_take_profit_pct ({partial_take_profit_pct})")
        if partial_take_profit_fraction is not None and (partial_take_profit_fraction <= 0 or partial_take_profit_fraction > 1.0):
            issues.append(f"unreasonable partial_take_profit_fraction ({partial_take_profit_fraction})")
        if position_size_multiplier is not None and (position_size_multiplier <= 0 or position_size_multiplier > 5.0):
            issues.append(f"unreasonable position_size_multiplier ({position_size_multiplier})")
        if confidence_sizing_weight is not None and (confidence_sizing_weight < 0 or confidence_sizing_weight > 1.0):
            issues.append(f"unreasonable confidence_sizing_weight ({confidence_sizing_weight})")

        # Validate complex nested objects and enums
        ptpl = params.get("partial_take_profit_levels")
        if ptpl is not None:
            if not isinstance(ptpl, list):
                issues.append("partial_take_profit_levels is not a list")
            else:
                for level in ptpl:
                    if not isinstance(level, dict):
                        issues.append("partial_take_profit_levels contains non-dict item")
                    else:
                        lvl_pct = level.get("take_profit_pct")
                        lvl_frac = level.get("fraction")
                        if not isinstance(lvl_pct, (int, float)) or lvl_pct <= 0 or lvl_pct > 5.0:
                            issues.append(f"unreasonable take_profit_pct in partial_take_profit_levels ({lvl_pct})")
                        if not isinstance(lvl_frac, (int, float)) or lvl_frac <= 0 or lvl_frac > 1.0:
                            issues.append(f"unreasonable fraction in partial_take_profit_levels ({lvl_frac})")

        bec = params.get("backtest_entry_config")
        if bec is not None:
            if not isinstance(bec, dict):
                issues.append("backtest_entry_config is not a dict")
            else:
                ema_period = bec.get("ema_period")
                if ema_period is not None and (not isinstance(ema_period, (int, float)) or ema_period < 0 or ema_period > 500):
                    issues.append(f"unreasonable ema_period in backtest_entry_config ({ema_period})")
                min_adx = bec.get("min_adx")
                if min_adx is not None and (not isinstance(min_adx, (int, float)) or min_adx < 0 or min_adx > 100):
                    issues.append(f"unreasonable min_adx in backtest_entry_config ({min_adx})")
                max_rsi = bec.get("max_rsi")
                if max_rsi is not None and (not isinstance(max_rsi, (int, float)) or max_rsi < 0 or max_rsi > 100):
                    issues.append(f"unreasonable max_rsi in backtest_entry_config ({max_rsi})")
                min_rsi = bec.get("min_rsi")
                if min_rsi is not None and (not isinstance(min_rsi, (int, float)) or min_rsi < 0 or min_rsi > 100):
                    issues.append(f"unreasonable min_rsi in backtest_entry_config ({min_rsi})")

        time_in_force = params.get("time_in_force")
        if time_in_force is not None and time_in_force not in ("day", "gtc", "ioc", "fok"):
            issues.append(f"invalid time_in_force ({time_in_force})")
        
        order_type = params.get("order_type")
        if order_type is not None and order_type not in ("market", "limit", "stop", "stop_limit", "trailing_stop"):
            issues.append(f"invalid order_type ({order_type})")

        stop_loss_order_type = params.get("stop_loss_order_type")
        if stop_loss_order_type is not None and stop_loss_order_type not in ("market", "limit", "stop", "stop_limit", "trailing_stop"):
            issues.append(f"invalid stop_loss_order_type ({stop_loss_order_type})")

        take_profit_order_type = params.get("take_profit_order_type")
        if take_profit_order_type is not None and take_profit_order_type not in ("market", "limit", "stop", "stop_limit", "trailing_stop"):
            issues.append(f"invalid take_profit_order_type ({take_profit_order_type})")

        # Validate remaining strategy and backtest config parameters
        max_portfolio_exposure = params.get("max_portfolio_exposure_pct")
        if max_portfolio_exposure is not None and (not isinstance(max_portfolio_exposure, (int, float)) or max_portfolio_exposure < 0 or max_portfolio_exposure > 1.0):
            issues.append(f"unreasonable max_portfolio_exposure_pct ({max_portfolio_exposure})")

        max_portfolio_stop_risk = params.get("max_portfolio_stop_risk_pct")
        if max_portfolio_stop_risk is not None and (not isinstance(max_portfolio_stop_risk, (int, float)) or max_portfolio_stop_risk < 0 or max_portfolio_stop_risk > 1.0):
            issues.append(f"unreasonable max_portfolio_stop_risk_pct ({max_portfolio_stop_risk})")

        direction = params.get("direction")
        if direction is not None and direction not in ("long", "short", "both"):
            issues.append(f"invalid direction ({direction})")

        fee_model = params.get("fee_model")
        if fee_model is not None and fee_model not in ("flat", "intesa"):
            issues.append(f"invalid fee_model ({fee_model})")

        slippage_model = params.get("slippage_model")
        if slippage_model is not None and slippage_model not in ("fixed", "dynamic"):
            issues.append(f"invalid slippage_model ({slippage_model})")

        slippage_pct = params.get("slippage_pct")
        if slippage_pct is not None and (not isinstance(slippage_pct, (int, float)) or slippage_pct < 0):
            issues.append(f"unreasonable slippage_pct ({slippage_pct})")

        slippage_base_pct = params.get("slippage_base_pct")
        if slippage_base_pct is not None and (not isinstance(slippage_base_pct, (int, float)) or slippage_base_pct <= 0):
            issues.append(f"unreasonable slippage_base_pct ({slippage_base_pct})")

        slippage_max_pct = params.get("slippage_max_pct")
        if slippage_max_pct is not None and (not isinstance(slippage_max_pct, (int, float)) or slippage_max_pct <= 0):
            issues.append(f"unreasonable slippage_max_pct ({slippage_max_pct})")

        simulate_position_sizing = params.get("simulate_position_sizing")
        if simulate_position_sizing is not None and not isinstance(simulate_position_sizing, bool):
            issues.append("simulate_position_sizing is not a boolean")

        global_risk_multiplier = params.get("global_risk_multiplier")
        if global_risk_multiplier is not None and (not isinstance(global_risk_multiplier, (int, float)) or global_risk_multiplier < 0):
            issues.append(f"unreasonable global_risk_multiplier ({global_risk_multiplier})")

        gap_tolerance_mult = params.get("gap_tolerance_mult")
        if gap_tolerance_mult is not None and (not isinstance(gap_tolerance_mult, (int, float)) or gap_tolerance_mult <= 0):
            issues.append(f"unreasonable gap_tolerance_mult ({gap_tolerance_mult})")

        on_gaps = params.get("on_gaps")
        if on_gaps is not None and on_gaps not in ("warn", "skip"):
            issues.append(f"invalid on_gaps ({on_gaps})")

        fee_rate = params.get("fee_rate")
        if fee_rate is not None and (not isinstance(fee_rate, (int, float)) or fee_rate < 0):
            issues.append(f"unreasonable fee_rate ({fee_rate})")

        max_trades = params.get("max_trades")
        if max_trades is not None and (not isinstance(max_trades, (int, float)) or max_trades <= 0):
            issues.append(f"unreasonable max_trades ({max_trades})")

        initial_balance = params.get("initial_balance")
        if initial_balance is not None and (not isinstance(initial_balance, (int, float)) or initial_balance <= 0):
            issues.append(f"unreasonable initial_balance ({initial_balance})")

        trade_value = params.get("trade_value")
        if trade_value is not None and (not isinstance(trade_value, (int, float)) or trade_value <= 0):
            issues.append(f"unreasonable trade_value ({trade_value})")

        # Ensure absolute prices and offsets are positive
        for price_key in ["limit_price", "stop_price", "trail_offset", "stop_loss_stop_price", "stop_loss_limit_price", "stop_loss_trail_offset", "take_profit_limit_price"]:
            val = params.get(price_key)
            if val is not None and (not isinstance(val, (int, float)) or val <= 0):
                issues.append(f"unreasonable {price_key} ({val})")

        # Validate remaining enums and booleans
        stop_loss_method = params.get("stop_loss_method")
        if stop_loss_method is not None and stop_loss_method not in ("fixed", "atr_multiple"):
            issues.append(f"invalid stop_loss_method ({stop_loss_method})")

        trailing_take_profit = params.get("trailing_take_profit")
        if trailing_take_profit is not None and not isinstance(trailing_take_profit, bool):
            issues.append("trailing_take_profit is not a boolean")

    if issues:
        new_reasoning = f"{reasoning} [Semantic validation failed: {'; '.join(issues)}. Downgraded to HOLD.]"
        return "HOLD", new_reasoning
        
    return action, reasoning


def _clamp_parameter_ranges(params: dict) -> dict:
    """Clamps LLM-provided parameters to safe, reasonable ranges to prevent hallucinations."""
    limits = {
        "stop_loss_pct": (0.01, 0.5),
        "take_profit_pct": (0.01, 5.0),
        "position_size_fraction": (0.01, 1.0),
        "stop_loss_atr_multiple": (0.1, 10.0),
        "take_profit_atr_multiple": (0.1, 20.0),
        "trailing_stop_distance_pct": (0.01, 0.5),
        "trailing_stop_atr_multiple": (0.1, 10.0),
        "trailing_stop_activation_pct": (0.01, 5.0),
        "breakeven_activation_pct": (0.01, 5.0),
        "max_unrealized_loss_pct": (0.01, 0.5),
        "max_hold_time_seconds": (60, 157_680_000),  # 60s to ~5 years (matches validator cap)
        "cooldown_after_loss_seconds": (0, 30 * 24 * 3600),
        "max_risk_per_trade_pct": (0.001, 0.1),
        "max_portfolio_risk_pct": (0.01, 1.0),
        "min_profit_per_trade": (0.0, 1.0),
        "min_risk_reward_ratio": (0.0, 10.0),
        "min_confidence": (0.0, 1.0),
        "news_sentiment_exit_threshold": (-1.0, 0.0),
        "strategy_interval_seconds": (60, 30 * 24 * 3600),
        "backtest_period_days": (30, 365 * 10),
        "order_fill_timeout_seconds": (10, 3600),
        "trailing_take_profit_distance_pct": (0.01, 0.5),
        "partial_take_profit_pct": (0.01, 5.0),
        "partial_take_profit_fraction": (0.01, 1.0),
        "position_size_multiplier": (0.1, 5.0),
        "confidence_sizing_weight": (0.0, 1.0),
    }
    for key, (min_val, max_val) in limits.items():
        if key in params and params[key] is not None:
            try:
                val = float(params[key])
                params[key] = max(min_val, min(max_val, val))
            except (ValueError, TypeError):
                pass
    return params


def _score_reasoning_quality(reasoning: str) -> float:
    """Scores the quality of LLM reasoning from 0.0 to 1.0 based on heuristics."""
    if not reasoning:
        return 0.0
    
    score = 0.0
    
    # Length check
    if len(reasoning) < 50:
        score += 0.1
    elif len(reasoning) < 200:
        score += 0.3
    else:
        score += 0.5
        
    # Keyword check
    keywords = [
        "rsi", "macd", "support", "resistance", "earnings", "trend", 
        "volume", "atr", "volatility", "risk", "reward", "dividend", 
        "yield", "spread", "momentum", "reversion", "breakout", "moving average",
        "bollinger", "stochastic", "fibonacci", "candlestick", "price action",
        "market cap", "p/e", "sentiment", "news", "macro", "fed", "interest rate"
    ]
    keyword_count = sum(1 for kw in keywords if kw in reasoning.lower())
    score += min(keyword_count * 0.1, 0.5)
    
    # Vague phrase penalty
    vague_phrases = [
        "goes up", "will go down", "because it's good", "because it's bad", 
        "just a feeling", "gut feeling", "will be higher", "will be lower"
    ]
    if any(phrase in reasoning.lower() for phrase in vague_phrases):
        score -= 0.3
        
    return max(0.0, min(1.0, score))


class StrategyModel(BaseModel):
    type: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None

class EntryConditionModel(BaseModel):
    type: str
    price: Optional[float] = None
    timeout_seconds: Optional[int] = None
    rsi_below: Optional[float] = None
    delay_seconds: Optional[int] = None
    conditions: Optional[List[Dict[str, Any]]] = None

class LLMResponseModel(BaseModel):
    action: str = "HOLD"
    confidence: float = 0.0
    reasoning: str = ""
    strategy: Optional[StrategyModel] = None
    risk_level: Optional[str] = "medium"
    indicator_config: Optional[Dict[str, Any]] = None
    backtest_summary: Optional[str] = None
    reason: Optional[str] = None
    entry_condition: Optional[EntryConditionModel] = None
    backtest_variants: Optional[List[Dict[str, Any]]] = None
    
    # Known root-level parameters
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    take_profit_atr_multiple: Optional[float] = None
    position_size_fraction: Optional[float] = None
    trailing_stop: Optional[bool] = False
    max_hold_time_seconds: Optional[int] = None
    stop_loss_method: Optional[str] = None
    stop_loss_atr_multiple: Optional[float] = None
    trailing_stop_distance_pct: Optional[float] = None
    trailing_stop_atr_multiple: Optional[float] = None
    trailing_stop_activation_pct: Optional[float] = None
    cooldown_after_loss_seconds: Optional[int] = 0
    portfolio_risk_adjustment_factor: Optional[float] = None
    max_risk_per_trade_pct: Optional[float] = None
    max_portfolio_risk_pct: Optional[float] = None
    min_profit_per_trade: Optional[float] = None
    min_risk_reward_ratio: Optional[float] = None
    min_confidence: Optional[float] = None
    news_sentiment_exit_threshold: Optional[float] = None
    strategy_interval_seconds: Optional[int] = None
    limit_price: Optional[float] = None
    time_in_force: Optional[str] = None
    backtest_period_days: Optional[int] = None
    order_fill_timeout_seconds: Optional[int] = None
    trailing_take_profit: Optional[bool] = None
    trailing_take_profit_distance_pct: Optional[float] = None
    breakeven_activation_pct: Optional[float] = None
    partial_take_profit_levels: Optional[List[Dict[str, Any]]] = None
    partial_take_profit_pct: Optional[float] = None
    partial_take_profit_fraction: Optional[float] = None
    max_unrealized_loss_pct: Optional[float] = None
    position_size_multiplier: Optional[float] = None
    confidence_sizing_weight: Optional[float] = None
    order_type: Optional[str] = None
    stop_price: Optional[float] = None
    trail_offset: Optional[float] = None
    stop_loss_order_type: Optional[str] = None
    stop_loss_stop_price: Optional[float] = None
    stop_loss_limit_price: Optional[float] = None
    stop_loss_trail_offset: Optional[float] = None
    take_profit_order_type: Optional[str] = None
    take_profit_limit_price: Optional[float] = None
    backtest_entry_config: Optional[Dict[str, Any]] = None


def parse_llm_response(response_text: str) -> Signal:
    """
    Parse the LLM's JSON response into a Signal.
    Supports JSON wrapped in ```json ... ``` code blocks or raw JSON.
    Raises ValueError if the response cannot be parsed as valid JSON.
    """
    try:
        # Try to extract JSON from a markdown code block first
        json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', response_text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(1))
        else:
            try:
                data = json.loads(response_text)
            except json.JSONDecodeError:
                # Fallback: try to extract the first valid JSON object from the text
                data = _extract_first_json(response_text)

        if isinstance(data, list):
            if not data:
                raise ValueError("LLM returned an empty JSON array")
            data = data[0]

        # Validate against Pydantic schema
        try:
            model = LLMResponseModel.model_validate(data)
        except ValidationError as e:
            raise ValueError(f"LLM response failed schema validation: {e}") from e

        action = model.action.upper()
        if action not in ("BUY", "SELL", "HOLD"):
            action = "HOLD"
        confidence = max(0.0, min(1.0, model.confidence))
        reasoning = model.reasoning

        strategy_type = model.strategy.type if model.strategy else None
        strategy_params = model.strategy.parameters if model.strategy else None

        risk_level = model.risk_level
        if risk_level not in ("low", "medium", "high"):
            risk_level = "medium"

        indicator_config = model.indicator_config
        backtest_summary = model.backtest_summary
        reason = model.reason or ""

        # --- dynamic trading parameters ---
        params = {}
        known_params = [
            "stop_loss_pct", "take_profit_pct", "take_profit_atr_multiple",
            "position_size_fraction", "trailing_stop", "max_hold_time_seconds",
            "stop_loss_method", "stop_loss_atr_multiple",
            "trailing_stop_distance_pct", "trailing_stop_atr_multiple",
            "trailing_stop_activation_pct", "cooldown_after_loss_seconds",
            "portfolio_risk_adjustment_factor",
            "max_risk_per_trade_pct", "max_portfolio_risk_pct",
            "min_profit_per_trade", "min_risk_reward_ratio",
            "min_confidence", "news_sentiment_exit_threshold",
            "strategy_interval_seconds", "limit_price", "time_in_force",
            "backtest_period_days", "order_fill_timeout_seconds",
            "trailing_take_profit", "trailing_take_profit_distance_pct",
            "breakeven_activation_pct",
            "partial_take_profit_levels", "partial_take_profit_pct",
            "partial_take_profit_fraction",
            "max_unrealized_loss_pct",
            "position_size_multiplier",
            "confidence_sizing_weight",
            "order_type", "stop_price", "trail_offset",
            "stop_loss_order_type", "stop_loss_stop_price",
            "stop_loss_limit_price", "stop_loss_trail_offset",
            "take_profit_order_type", "take_profit_limit_price",
            "backtest_entry_config",
        ]
        for k in known_params:
            val = getattr(model, k, None)
            if val is not None:
                params[k] = val
        if isinstance(strategy_params, dict):
            params.update(strategy_params)
        
        stop_loss = params.get("stop_loss_pct")
        take_profit = params.get("take_profit_pct")
        take_profit_atr_multiple = params.get("take_profit_atr_multiple")
        
        position_size = params.get("position_size_fraction")
        if position_size is not None:
            position_size = max(0.0, min(1.0, position_size))
            
        trailing_stop = bool(params.get("trailing_stop", False))
        max_hold_time_seconds = params.get("max_hold_time_seconds")
        
        stop_loss_method = params.get("stop_loss_method")
        stop_loss_atr_multiple = params.get("stop_loss_atr_multiple")
        trailing_stop_distance_pct = params.get("trailing_stop_distance_pct")
        trailing_stop_atr_multiple = params.get("trailing_stop_atr_multiple")
        trailing_stop_activation_pct = params.get("trailing_stop_activation_pct")
        cooldown_after_loss_seconds = params.get("cooldown_after_loss_seconds", 0) or 0

        # Write the safely casted values back into params
        if position_size is not None:
            params["position_size_fraction"] = position_size
            
        confidence_sizing_weight = params.get("confidence_sizing_weight")
        if confidence_sizing_weight is not None:
            confidence_sizing_weight = max(0.0, min(1.0, confidence_sizing_weight))
            params["confidence_sizing_weight"] = confidence_sizing_weight

        portfolio_risk_adjustment_factor = params.get("portfolio_risk_adjustment_factor")
        if portfolio_risk_adjustment_factor is not None:
            try:
                portfolio_risk_adjustment_factor = max(0.1, min(1.0, float(portfolio_risk_adjustment_factor)))
            except (TypeError, ValueError):
                portfolio_risk_adjustment_factor = None

        # --- reasoning quality scoring ---
        reasoning_quality_score = _score_reasoning_quality(reasoning)
        params["reasoning_quality_score"] = reasoning_quality_score

        # --- entry condition ---
        entry_condition = None
        if model.entry_condition:
            etype = model.entry_condition.type
            valid_types = ("limit_price", "rsi_threshold", "delay", "indicator_combo")
            if etype in valid_types:
                ec_dict = model.entry_condition.model_dump(exclude_none=True)
                if etype == "limit_price" and "price" in ec_dict and "timeout_seconds" in ec_dict:
                    entry_condition = ec_dict
                elif etype == "rsi_threshold" and "rsi_below" in ec_dict and "timeout_seconds" in ec_dict:
                    entry_condition = ec_dict
                elif etype == "delay" and "delay_seconds" in ec_dict:
                    entry_condition = ec_dict
                elif etype == "indicator_combo" and isinstance(ec_dict.get("conditions"), list) and len(ec_dict["conditions"]) > 0 and "timeout_seconds" in ec_dict:
                    entry_condition = ec_dict

        # --- order execution parameters ---
        order_type = params.get("order_type")
        stop_price = params.get("stop_price")
        limit_price = params.get("limit_price")
        trail_offset = params.get("trail_offset")
        stop_loss_order_type = params.get("stop_loss_order_type")
        stop_loss_stop_price = params.get("stop_loss_stop_price")
        stop_loss_limit_price = params.get("stop_loss_limit_price")
        stop_loss_trail_offset = params.get("stop_loss_trail_offset")
        take_profit_order_type = params.get("take_profit_order_type")
        take_profit_limit_price = params.get("take_profit_limit_price")

        backtest_period_days = params.get("backtest_period_days")
        if backtest_period_days is not None:
            backtest_period_days = max(30, min(backtest_period_days, settings.OHLCV_RETENTION_DAYS))
            params["backtest_period_days"] = backtest_period_days

        # --- Extract backtest_variants array (Step 1 LLM output) ---
        backtest_variants = model.backtest_variants
        if backtest_variants is not None:
            # Validate each variant is a dict; drop invalid entries
            backtest_variants = [v for v in backtest_variants if isinstance(v, dict)]
            # Clamp each variant's parameters to safe ranges
            backtest_variants = [_clamp_parameter_ranges(v) for v in backtest_variants]
            if not backtest_variants:
                backtest_variants = None

        # Fallback: if backtest_entry_config is missing from params but present
        # in the first backtest variant, copy it so the validator doesn't reject BUY signals.
        if "backtest_entry_config" not in params and backtest_variants and isinstance(backtest_variants[0], dict):
            bec = backtest_variants[0].get("backtest_entry_config")
            if isinstance(bec, dict):
                params["backtest_entry_config"] = bec

        # --- Clamp parameter ranges to prevent hallucinations (BEFORE semantic validation) ---
        params = _clamp_parameter_ranges(params)

        # Re-read clamped values so the Signal reflects the safe ranges
        stop_loss = params.get("stop_loss_pct")
        take_profit = params.get("take_profit_pct")
        take_profit_atr_multiple = params.get("take_profit_atr_multiple")
        position_size = params.get("position_size_fraction")
        if position_size is not None:
            position_size = max(0.0, min(1.0, position_size))

        # --- Semantic quality validation (after clamping, so clamped values don't trigger false positives) ---
        action, reasoning = _validate_semantic_quality(action, params, reasoning)

        return Signal(
            action=action,
            confidence=confidence,
            reasoning=reasoning,
            strategy_type=strategy_type,
            strategy_params=params,
            risk_level=risk_level,
            indicator_config=indicator_config,
            backtest_summary=backtest_summary,
            stop_loss=stop_loss,
            take_profit=take_profit,
            take_profit_atr_multiple=take_profit_atr_multiple,
            position_size=position_size,
            confidence_sizing_weight=confidence_sizing_weight,
            trailing_stop=trailing_stop,
            reason=reason,
            entry_condition=entry_condition,
            stop_loss_method=stop_loss_method,
            stop_loss_atr_multiple=stop_loss_atr_multiple,
            trailing_stop_distance_pct=trailing_stop_distance_pct,
            trailing_stop_atr_multiple=trailing_stop_atr_multiple,
            trailing_stop_activation_pct=trailing_stop_activation_pct,
            max_hold_time_seconds=max_hold_time_seconds,
            cooldown_after_loss_seconds=cooldown_after_loss_seconds,
            portfolio_risk_adjustment_factor=portfolio_risk_adjustment_factor,
            order_type=order_type,
            stop_price=stop_price,
            limit_price=limit_price,
            trail_offset=trail_offset,
            stop_loss_order_type=stop_loss_order_type,
            stop_loss_stop_price=stop_loss_stop_price,
            stop_loss_limit_price=stop_loss_limit_price,
            stop_loss_trail_offset=stop_loss_trail_offset,
            take_profit_order_type=take_profit_order_type,
            take_profit_limit_price=take_profit_limit_price,
            backtest_period_days=backtest_period_days,
            backtest_variants=backtest_variants,
        )
    except (json.JSONDecodeError, ValueError, TypeError, ValidationError) as e:
        raise ValueError(f"Failed to parse LLM response as valid JSON: {e}") from e


def create_strategy_from_llm(response_text: str) -> LLMStrategy:
    """
    Parse the LLM response and return an LLMStrategy instance.
    """
    signal = parse_llm_response(response_text)
    return LLMStrategy(signal)
