import pytest
from src.strategies.llm_parser import parse_llm_response
from src.strategies.validator import validate_signal
from src.strategies.base import LLMStrategy, Signal


def test_signal_pipeline_valid_buy():
    """Test the full pipeline from LLM response to validated strategy signal."""
    # 1. Parse LLM response
    llm_response = '{"action": "BUY", "confidence": 0.8, "reasoning": "Strong uptrend", "stop_loss_pct": 0.05, "take_profit_pct": 0.10}'
    signal = parse_llm_response(llm_response)
    assert signal.action == "BUY"
    
    # 2. Validate signal
    validated_signal = validate_signal(signal, price=100.0, atr=2.0, timeframe_seconds=3600)
    assert validated_signal.action == "BUY"
    assert validated_signal.sl is not None
    assert validated_signal.tp is not None
    
    # 3. Create strategy from LLM
    strategy = LLMStrategy(signal)
    generated_signal = strategy.generate_signal({})
    assert generated_signal.action == "BUY"
    assert generated_signal.confidence == 0.8


def test_signal_pipeline_invalid_action():
    """Test pipeline handling of an invalid action from the LLM."""
    llm_response = '{"action": "INVALID", "confidence": 0.5, "reasoning": "Test"}'
    signal = parse_llm_response(llm_response)
    assert signal.action == "HOLD"
    
    validated_signal = validate_signal(signal)
    assert validated_signal.action == "HOLD"
    
    strategy = LLMStrategy(signal)
    generated_signal = strategy.generate_signal({})
    assert generated_signal.action == "HOLD"


def test_signal_pipeline_semantic_validation_failure():
    """Test pipeline handling of a semantically invalid signal (e.g., stop loss too wide)."""
    llm_response = '{"action": "BUY", "confidence": 0.8, "reasoning": "Test", "stop_loss_pct": 0.8}'
    signal = parse_llm_response(llm_response)
    # Semantic validation should fail and change action to HOLD
    assert signal.action == "HOLD"
    
    validated_signal = validate_signal(signal)
    assert validated_signal.action == "HOLD"
