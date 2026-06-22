import json
import re
from .base import Signal, LLMStrategy


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
                # Fallback: try to extract the first JSON object from the text
                start = response_text.find('{')
                end = response_text.rfind('}')
                if start != -1 and end != -1 and end > start:
                    json_str = response_text[start:end+1]
                    data = json.loads(json_str)
                else:
                    raise ValueError("No JSON object found in LLM response")

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
            "stop_loss_pct", "take_profit_pct", "position_size_fraction", "trailing_stop",
            "max_hold_time_seconds", "stop_loss_method", "stop_loss_atr_multiple",
            "trailing_stop_distance_pct", "trailing_stop_activation_pct", "cooldown_after_loss_seconds",
            "portfolio_risk_adjustment_factor",
            "max_risk_per_trade_pct", "min_profit_per_trade", "min_risk_reward_ratio",
            "min_confidence", "news_sentiment_exit_threshold",
            "strategy_interval_seconds", "limit_price", "time_in_force",
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
        position_size = _safe_float(params.get("position_size_fraction"))
        if position_size is not None:
            position_size = max(0.0, min(1.0, position_size))
        trailing_stop = bool(params.get("trailing_stop", False))
        
        max_hold_time_seconds = _safe_int(params.get("max_hold_time_seconds"))
        # Keep max_hold_minutes for backwards compatibility with the engine
        max_hold_minutes = int(max_hold_time_seconds // 60) if max_hold_time_seconds is not None else None
        
        stop_loss_method = params.get("stop_loss_method")
        stop_loss_atr_multiple = _safe_float(params.get("stop_loss_atr_multiple"))
        trailing_stop_distance_pct = _safe_float(params.get("trailing_stop_distance_pct"))
        trailing_stop_activation_pct = _safe_float(params.get("trailing_stop_activation_pct"))
        cooldown_after_loss_seconds = _safe_int(params.get("cooldown_after_loss_seconds", 0)) or 0

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
            position_size=position_size,
            trailing_stop=trailing_stop,
            max_hold_minutes=max_hold_minutes,
            reason=reason,
            entry_condition=entry_condition,
            stop_loss_method=stop_loss_method,
            stop_loss_atr_multiple=stop_loss_atr_multiple,
            trailing_stop_distance_pct=trailing_stop_distance_pct,
            trailing_stop_activation_pct=trailing_stop_activation_pct,
            max_hold_time_seconds=max_hold_time_seconds,
            cooldown_after_loss_seconds=cooldown_after_loss_seconds,
            portfolio_risk_adjustment_factor=portfolio_risk_adjustment_factor,
        )
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        raise ValueError(f"Failed to parse LLM response as valid JSON: {e}") from e


def create_strategy_from_llm(response_text: str) -> LLMStrategy:
    """
    Parse the LLM response and return an LLMStrategy instance.
    """
    signal = parse_llm_response(response_text)
    return LLMStrategy(signal)
