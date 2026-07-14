import pytest
from src.strategies.llm_parser import parse_llm_response, _validate_semantic_quality


def test_parse_valid_json():
    signal = parse_llm_response('{"action": "BUY", "confidence": 0.8, "reasoning": "Strong uptrend"}')
    assert signal.action == "BUY"
    assert signal.confidence == 0.8
    assert signal.reasoning == "Strong uptrend"


def test_parse_markdown_json():
    signal = parse_llm_response('```json\n{"action": "SELL", "confidence": 0.9, "reasoning": "Bearish"}\n```')
    assert signal.action == "SELL"
    assert signal.confidence == 0.9


def test_parse_invalid_json():
    with pytest.raises(ValueError):
        parse_llm_response('This is not JSON')


def test_validate_semantic_quality_bad_stop_loss():
    action, reasoning = _validate_semantic_quality("BUY", {"stop_loss_pct": 0.8}, "Test")
    assert action == "HOLD"
    assert "Semantic validation failed" in reasoning
