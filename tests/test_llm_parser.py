import pytest
from src.strategies.llm_parser import parse_llm_response, _validate_semantic_quality, _extract_first_json, _score_reasoning_quality


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


# ---------- parse_llm_response (array & invalid action) ----------

def test_parse_array_json():
    signal = parse_llm_response('[{"action": "BUY", "confidence": 0.7, "reasoning": "Test"}]')
    assert signal.action == "BUY"


def test_parse_invalid_action_defaults_to_hold():
    signal = parse_llm_response('{"action": "INVALID", "confidence": 0.5, "reasoning": "Test"}')
    assert signal.action == "HOLD"


# ---------- _extract_first_json ----------

def test_extract_first_json_simple():
    result = _extract_first_json('prefix {"a": 1} suffix')
    assert result == {"a": 1}


def test_extract_first_json_nested():
    result = _extract_first_json('{"a": {"b": 2}, "c": 3}')
    assert result == {"a": {"b": 2}, "c": 3}


def test_extract_first_json_no_json():
    with pytest.raises(ValueError):
        _extract_first_json('no json here')


# ---------- _score_reasoning_quality ----------

def test_score_reasoning_quality_empty():
    assert _score_reasoning_quality("") == 0.0


def test_score_reasoning_quality_short():
    assert _score_reasoning_quality("short") == 0.1


def test_score_reasoning_quality_medium():
    assert _score_reasoning_quality("a" * 100) == 0.3


def test_score_reasoning_quality_with_keywords():
    result = _score_reasoning_quality("The RSI and MACD indicators show strong momentum with good volume.")
    assert result > 0.3


def test_score_reasoning_quality_vague_penalty():
    result = _score_reasoning_quality("The price will go up because it's good.")
    assert result < 0.5
