"""Unit tests for LLM response parsing."""
import pytest
import json
from src.strategies.llm_parser import parse_llm_response, create_strategy_from_llm
from src.strategies.base import Signal


class TestParseValidResponse:
    def test_valid_buy_signal(self):
        response = json.dumps({
            "action": "BUY",
            "confidence": 0.85,
            "reasoning": "Strong uptrend with bullish MACD crossover",
            "strategy": {
                "type": "momentum",
                "parameters": {
                    "stop_loss_pct": 0.05,
                    "take_profit_pct": 0.10,
                    "position_size_fraction": 0.25,
                    "trailing_stop": False,
                    "max_hold_time_seconds": 86400,
                    "cooldown_after_loss_seconds": 3600,
                }
            }
        })
        signal = parse_llm_response(response)
        assert signal.action == "BUY"
        assert signal.confidence == 0.85
        assert signal.strategy_type == "momentum"
        assert signal.strategy_params["stop_loss_pct"] == 0.05
        assert signal.strategy_params["take_profit_pct"] == 0.10

    def test_valid_sell_signal(self):
        response = json.dumps({
            "action": "SELL",
            "confidence": 0.9,
            "reasoning": "Stop-loss triggered",
            "strategy": {
                "type": "mean_reversion",
                "parameters": {
                    "stop_loss_pct": 0.03,
                    "take_profit_pct": 0.06,
                    "position_size_fraction": 0.5,
                    "trailing_stop": True,
                    "trailing_stop_distance_pct": 0.02,
                    "max_hold_time_seconds": 7200,
                    "cooldown_after_loss_seconds": 0,
                }
            }
        })
        signal = parse_llm_response(response)
        assert signal.action == "SELL"
        assert signal.confidence == 0.9
        assert signal.trailing_stop is True

    def test_hold_signal(self):
        response = json.dumps({
            "action": "HOLD",
            "confidence": 0.3,
            "reasoning": "No clear signal",
        })
        signal = parse_llm_response(response)
        assert signal.action == "HOLD"
        assert signal.confidence == 0.3

    def test_root_level_params_merged(self):
        """Parameters at root level should be merged into strategy_params."""
        response = json.dumps({
            "action": "BUY",
            "confidence": 0.7,
            "reasoning": "Test",
            "stop_loss_pct": 0.04,
            "take_profit_pct": 0.08,
            "position_size_fraction": 0.3,
            "trailing_stop": False,
            "max_hold_time_seconds": 3600,
            "cooldown_after_loss_seconds": 1800,
            "strategy": {
                "type": "breakout",
                "parameters": {
                    "stop_loss_pct": 0.05,  # strategy.parameters takes precedence
                }
            }
        })
        signal = parse_llm_response(response)
        # strategy.parameters should override root-level
        assert signal.strategy_params["stop_loss_pct"] == 0.05
        # But root-level take_profit_pct should be merged
        assert signal.strategy_params["take_profit_pct"] == 0.08


class TestParseMarkdownWrapped:
    def test_json_in_code_block(self):
        response = "```json\n" + json.dumps({
            "action": "BUY",
            "confidence": 0.8,
            "reasoning": "Test",
        }) + "\n```"
        signal = parse_llm_response(response)
        assert signal.action == "BUY"

    def test_json_in_plain_code_block(self):
        response = "```\n" + json.dumps({
            "action": "HOLD",
            "confidence": 0.5,
            "reasoning": "Test",
        }) + "\n```"
        signal = parse_llm_response(response)
        assert signal.action == "HOLD"


class TestParseEdgeCases:
    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="Failed to parse"):
            parse_llm_response("not json at all")

    def test_empty_array_raises(self):
        with pytest.raises(ValueError, match="empty JSON array"):
            parse_llm_response("[]")

    def test_array_takes_first_element(self):
        response = json.dumps([
            {"action": "BUY", "confidence": 0.8, "reasoning": "First"},
            {"action": "SELL", "confidence": 0.9, "reasoning": "Second"},
        ])
        signal = parse_llm_response(response)
        assert signal.action == "BUY"
        assert signal.reasoning == "First"

    def test_invalid_action_defaults_to_hold(self):
        response = json.dumps({
            "action": "INVALID",
            "confidence": 0.5,
            "reasoning": "Test",
        })
        signal = parse_llm_response(response)
        assert signal.action == "HOLD"

    def test_confidence_clamped_to_range(self):
        response = json.dumps({
            "action": "BUY",
            "confidence": 1.5,  # > 1.0
            "reasoning": "Test",
        })
        signal = parse_llm_response(response)
        assert signal.confidence == 1.0

    def test_negative_confidence_clamped(self):
        response = json.dumps({
            "action": "BUY",
            "confidence": -0.5,
            "reasoning": "Test",
        })
        signal = parse_llm_response(response)
        assert signal.confidence == 0.0

    def test_json_embedded_in_text(self):
        """JSON object embedded in surrounding text should be extracted."""
        response = "Here is my decision:\n" + json.dumps({
            "action": "BUY",
            "confidence": 0.7,
            "reasoning": "Test",
        }) + "\nThat's all."
        signal = parse_llm_response(response)
        assert signal.action == "BUY"


class TestEntryCondition:
    def test_limit_price_entry(self):
        response = json.dumps({
            "action": "BUY",
            "confidence": 0.8,
            "reasoning": "Test",
            "entry_condition": {
                "type": "limit_price",
                "price": 95.0,
                "timeout_seconds": 3600,
            }
        })
        signal = parse_llm_response(response)
        assert signal.entry_condition is not None
        assert signal.entry_condition["type"] == "limit_price"
        assert signal.entry_condition["price"] == 95.0

    def test_rsi_threshold_entry(self):
        response = json.dumps({
            "action": "BUY",
            "confidence": 0.8,
            "reasoning": "Test",
            "entry_condition": {
                "type": "rsi_threshold",
                "rsi_below": 30,
                "timeout_seconds": 7200,
            }
        })
        signal = parse_llm_response(response)
        assert signal.entry_condition is not None
        assert signal.entry_condition["type"] == "rsi_threshold"

    def test_delay_entry(self):
        response = json.dumps({
            "action": "BUY",
            "confidence": 0.8,
            "reasoning": "Test",
            "entry_condition": {
                "type": "delay",
                "delay_seconds": 3600,
            }
        })
        signal = parse_llm_response(response)
        assert signal.entry_condition is not None
        assert signal.entry_condition["type"] == "delay"

    def test_invalid_entry_condition_type(self):
        response = json.dumps({
            "action": "BUY",
            "confidence": 0.8,
            "reasoning": "Test",
            "entry_condition": {
                "type": "invalid_type",
                "timeout_seconds": 3600,
            }
        })
        signal = parse_llm_response(response)
        assert signal.entry_condition is None

    def test_missing_required_fields(self):
        """Entry condition with missing required fields should be None."""
        response = json.dumps({
            "action": "BUY",
            "confidence": 0.8,
            "reasoning": "Test",
            "entry_condition": {
                "type": "limit_price",
                # Missing "price" and "timeout_seconds"
            }
        })
        signal = parse_llm_response(response)
        assert signal.entry_condition is None


class TestBacktestVariants:
    def test_backtest_variants_extracted(self):
        response = json.dumps({
            "action": "BUY",
            "confidence": 0.8,
            "reasoning": "Test",
            "backtest_variants": [
                {"stop_loss_pct": 0.03, "take_profit_pct": 0.06},
                {"stop_loss_pct": 0.05, "take_profit_pct": 0.10},
            ]
        })
        signal = parse_llm_response(response)
        assert signal.backtest_variants is not None
        assert len(signal.backtest_variants) == 2

    def test_backtest_variants_none_when_missing(self):
        response = json.dumps({
            "action": "BUY",
            "confidence": 0.8,
            "reasoning": "Test",
        })
        signal = parse_llm_response(response)
        assert signal.backtest_variants is None

    def test_backtest_variants_filtered_invalid(self):
        """Non-dict entries in backtest_variants should be dropped."""
        response = json.dumps({
            "action": "BUY",
            "confidence": 0.8,
            "reasoning": "Test",
            "backtest_variants": [
                {"stop_loss_pct": 0.03},
                "invalid_string",
                {"stop_loss_pct": 0.05},
            ]
        })
        signal = parse_llm_response(response)
        assert signal.backtest_variants is not None
        assert len(signal.backtest_variants) == 2  # only valid dicts


class TestCreateStrategy:
    def test_create_strategy_returns_llm_strategy(self):
        response = json.dumps({
            "action": "BUY",
            "confidence": 0.8,
            "reasoning": "Test",
        })
        strategy = create_strategy_from_llm(response)
        signal = strategy.generate_signal({})
        assert signal.action == "BUY"
        assert isinstance(signal, Signal)
