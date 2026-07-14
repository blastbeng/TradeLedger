import json
import re
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
    
    stop_loss = params.get("stop_loss_pct")
    take_profit = params.get("take_profit_pct")
    
    if action == "BUY":
        if stop_loss is not None:
            if stop_loss <= 0 or stop_loss > 0.5:
                issues.append(f"unreasonable stop_loss_pct ({stop_loss})")
        
        if take_profit is not None:
            if take_profit <= 0 or take_profit > 5.0:
                issues.append(f"unreasonable take_profit_pct ({take_profit})")
                
        if stop_loss is not None and take_profit is not None:
            if take_profit < stop_loss:
                issues.append(f"take_profit_pct ({take_profit}) < stop_loss_pct ({stop_loss})")
                
    if issues:
        new_reasoning = f"{reasoning} [Semantic validation failed: {'; '.join(issues)}. Downgraded to HOLD.]"
        return "HOLD", new_reasoning
        
    return action, reasoning


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

        action = data.get("action", "HOLD").upper()
        if action not in ("BUY", "SELL", "HOLD"):
            action = "HOLD"
        confidence = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
        reasoning = data.get("reasoning", "")

        strategy = data.get("strategy")
        strategy_type = None
        strategy_params = None
        if isinstance(strategy, dict):
            strategy_type = strategy.get("type")
            strategy_params = strategy.get("parameters")

        risk_level = data.get("risk_level")
        if risk_level not in ("low", "medium", "high"):
            risk_level = "medium"

        indicator_config = data.get("indicator_config")
        if not isinstance(indicator_config, dict):
            indicator_config = None

        backtest_summary = data.get("backtest_summary")
        if not isinstance(backtest_summary, str):
            backtest_summary = None

        # --- dynamic trading parameters ---
        # The LLM puts these inside strategy.parameters, but may also put them at the root level.
        # We merge root-level parameters with strategy.parameters, preferring strategy.parameters.
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
            if k in data:
                params[k] = data[k]
        if isinstance(strategy_params, dict):
            params.update(strategy_params)
        
        def _safe_float(val):
            if val is None:
                return None
            try:
                return float(val)
            except (ValueError, TypeError):
                return None

        def _safe_int(val):
            if val is None:
                return None
            try:
                return int(float(val))
            except (ValueError, TypeError):
                return None

        stop_loss = _safe_float(params.get("stop_loss_pct"))
        take_profit = _safe_float(params.get("take_profit_pct"))
        take_profit_atr_multiple = _safe_float(params.get("take_profit_atr_multiple"))
        if take_profit_atr_multiple is not None:
            params["take_profit_atr_multiple"] = take_profit_atr_multiple
        position_size = _safe_float(params.get("position_size_fraction"))
        if position_size is not None:
            position_size = max(0.0, min(1.0, position_size))
        trailing_stop = bool(params.get("trailing_stop", False))
        
        max_hold_time_seconds = _safe_int(params.get("max_hold_time_seconds"))
        
        stop_loss_method = params.get("stop_loss_method")
        stop_loss_atr_multiple = _safe_float(params.get("stop_loss_atr_multiple"))
        trailing_stop_distance_pct = _safe_float(params.get("trailing_stop_distance_pct"))
        trailing_stop_atr_multiple = _safe_float(params.get("trailing_stop_atr_multiple"))
        if trailing_stop_atr_multiple is not None:
            params["trailing_stop_atr_multiple"] = trailing_stop_atr_multiple
        trailing_stop_activation_pct = _safe_float(params.get("trailing_stop_activation_pct"))
        cooldown_after_loss_seconds = _safe_int(params.get("cooldown_after_loss_seconds", 0)) or 0

        # Write the safely casted values back into params so that the validator
        # and engine receive correct numeric types instead of raw strings.
        if stop_loss is not None:
            params["stop_loss_pct"] = stop_loss
        if take_profit is not None:
            params["take_profit_pct"] = take_profit
        if position_size is not None:
            params["position_size_fraction"] = position_size
        confidence_sizing_weight = _safe_float(params.get("confidence_sizing_weight"))
        if confidence_sizing_weight is not None:
            confidence_sizing_weight = max(0.0, min(1.0, confidence_sizing_weight))
            params["confidence_sizing_weight"] = confidence_sizing_weight
        params["trailing_stop"] = trailing_stop
        if max_hold_time_seconds is not None:
            params["max_hold_time_seconds"] = max_hold_time_seconds
        if stop_loss_atr_multiple is not None:
            params["stop_loss_atr_multiple"] = stop_loss_atr_multiple
        if trailing_stop_distance_pct is not None:
            params["trailing_stop_distance_pct"] = trailing_stop_distance_pct
        if trailing_stop_activation_pct is not None:
            params["trailing_stop_activation_pct"] = trailing_stop_activation_pct
        params["cooldown_after_loss_seconds"] = cooldown_after_loss_seconds

        portfolio_risk_adjustment_factor = params.get("portfolio_risk_adjustment_factor")
        if portfolio_risk_adjustment_factor is not None:
            try:
                portfolio_risk_adjustment_factor = max(0.1, min(1.0, float(portfolio_risk_adjustment_factor)))
            except (TypeError, ValueError):
                portfolio_risk_adjustment_factor = None

        reason = data.get("reason", "")

        # --- entry condition ---
        entry_condition_raw = data.get("entry_condition")
        entry_condition = None
        if isinstance(entry_condition_raw, dict):
            etype = entry_condition_raw.get("type")
            valid_types = ("limit_price", "rsi_threshold", "delay", "indicator_combo")
            if etype in valid_types:
                if etype == "limit_price" and "price" in entry_condition_raw and "timeout_seconds" in entry_condition_raw:
                    entry_condition = entry_condition_raw
                elif etype == "rsi_threshold" and "rsi_below" in entry_condition_raw and "timeout_seconds" in entry_condition_raw:
                    entry_condition = entry_condition_raw
                elif etype == "delay" and "delay_seconds" in entry_condition_raw:
                    entry_condition = entry_condition_raw
                elif etype == "indicator_combo" and isinstance(entry_condition_raw.get("conditions"), list) and len(entry_condition_raw["conditions"]) > 0 and "timeout_seconds" in entry_condition_raw:
                    entry_condition = entry_condition_raw

        # --- order execution parameters ---
        order_type = params.get("order_type")

        stop_price = _safe_float(params.get("stop_price"))
        if stop_price is not None:
            params["stop_price"] = stop_price

        # limit_price is already in params, but we also expose it on the Signal
        limit_price = _safe_float(params.get("limit_price"))
        if limit_price is not None:
            params["limit_price"] = limit_price

        trail_offset = _safe_float(params.get("trail_offset"))
        if trail_offset is not None:
            params["trail_offset"] = trail_offset

        stop_loss_order_type = params.get("stop_loss_order_type")

        stop_loss_stop_price = _safe_float(params.get("stop_loss_stop_price"))
        if stop_loss_stop_price is not None:
            params["stop_loss_stop_price"] = stop_loss_stop_price

        stop_loss_limit_price = _safe_float(params.get("stop_loss_limit_price"))
        if stop_loss_limit_price is not None:
            params["stop_loss_limit_price"] = stop_loss_limit_price

        stop_loss_trail_offset = _safe_float(params.get("stop_loss_trail_offset"))
        if stop_loss_trail_offset is not None:
            params["stop_loss_trail_offset"] = stop_loss_trail_offset

        take_profit_order_type = params.get("take_profit_order_type")

        take_profit_limit_price = _safe_float(params.get("take_profit_limit_price"))
        if take_profit_limit_price is not None:
            params["take_profit_limit_price"] = take_profit_limit_price

        backtest_period_days = _safe_int(params.get("backtest_period_days"))
        if backtest_period_days is not None:
            backtest_period_days = max(30, min(backtest_period_days, settings.OHLCV_RETENTION_DAYS))
            params["backtest_period_days"] = backtest_period_days

        # --- Extract backtest_variants array (Step 1 LLM output) ---
        backtest_variants = data.get("backtest_variants")
        if not isinstance(backtest_variants, list):
            backtest_variants = None
        else:
            # Validate each variant is a dict; drop invalid entries
            backtest_variants = [v for v in backtest_variants if isinstance(v, dict)]
            if not backtest_variants:
                backtest_variants = None

        # Fallback: if backtest_entry_config is missing from params but present
        # in the first backtest variant, copy it so the validator doesn't reject BUY signals.
        if "backtest_entry_config" not in params and backtest_variants and isinstance(backtest_variants[0], dict):
            bec = backtest_variants[0].get("backtest_entry_config")
            if isinstance(bec, dict):
                params["backtest_entry_config"] = bec

        # --- Semantic quality validation ---
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
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        raise ValueError(f"Failed to parse LLM response as valid JSON: {e}") from e


def create_strategy_from_llm(response_text: str) -> LLMStrategy:
    """
    Parse the LLM response and return an LLMStrategy instance.
    """
    signal = parse_llm_response(response_text)
    return LLMStrategy(signal)
