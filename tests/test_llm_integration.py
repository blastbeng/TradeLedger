import pytest
from unittest.mock import patch, MagicMock
from src.llm.cache import get_cached_llm_response


@pytest.fixture
def mock_redis():
    with patch("src.llm.cache.get_redis_client") as mock_get_redis:
        mock_client = MagicMock()
        mock_get_redis.return_value = mock_client
        yield mock_client


def test_llm_cache_miss_and_set(mock_redis):
    with patch("src.llm.cache._get_cached_response", return_value=None), \
         patch("src.llm.cache._execute_primary_call", return_value=("LLM response", {}, "openai", "gpt-4", False)), \
         patch("src.llm.cache._should_use_primary_model", return_value=True), \
         patch("src.llm.cache._get_primary_provider_config", return_value=("openai", ["gpt-4"], "http://localhost", "key")):
        
        result = get_cached_llm_response(
            prompt="Test prompt",
            system_prompt="System",
            model_type="actuator"
        )
        
        assert result is not None
        assert result["response"] == "LLM response"
        assert result["provider"] == "openai"
        assert result["model"] == "gpt-4"
        # Verify it was saved to Redis
        mock_redis.set.assert_called_once()


def test_llm_cache_hit(mock_redis):
    cached_data = {
        "response": "Cached response",
        "provider": "openai",
        "model": "gpt-4",
        "is_fallback": False
    }
    with patch("src.llm.cache._get_cached_response", return_value=cached_data), \
         patch("src.llm.cache._execute_primary_call") as mock_exec, \
         patch("src.llm.cache._get_primary_provider_config", return_value=("openai", ["gpt-4"], "http://localhost", "key")):
        
        result = get_cached_llm_response(
            prompt="Test prompt",
            system_prompt="System",
            model_type="actuator"
        )
        
        assert result["response"] == "Cached response"
        # Should not execute primary call
        mock_exec.assert_not_called()
        # Should not save to Redis
        mock_redis.set.assert_not_called()


def test_llm_fallback_on_primary_failure(mock_redis):
    with patch("src.llm.cache._get_cached_response", return_value=None), \
         patch("src.llm.cache._execute_primary_call", side_effect=RuntimeError("Primary failed")), \
         patch("src.llm.cache._execute_fallback_call", return_value=("Fallback response", {}, "openai", "gpt-3.5", True)), \
         patch("src.llm.cache._should_use_primary_model", return_value=True), \
         patch("src.llm.cache._get_primary_provider_config", return_value=("openai", ["gpt-4"], "http://localhost", "key")):
        
        result = get_cached_llm_response(
            prompt="Test prompt",
            system_prompt="System",
            model_type="actuator"
        )
        
        assert result["response"] == "Fallback response"
        assert result["is_fallback"] is True
        mock_redis.set.assert_called_once()


def test_llm_all_providers_fail(mock_redis):
    with patch("src.llm.cache._get_cached_response", return_value=None), \
         patch("src.llm.cache._execute_primary_call", side_effect=RuntimeError("Primary failed")), \
         patch("src.llm.cache._execute_fallback_call", side_effect=RuntimeError("Fallback failed")), \
         patch("src.llm.cache._try_aol_model", return_value=None), \
         patch("src.llm.cache._should_use_primary_model", return_value=True), \
         patch("src.llm.cache._get_primary_provider_config", return_value=("openai", ["gpt-4"], "http://localhost", "key")):
        
        with pytest.raises(RuntimeError, match="Fallback failed"):
            get_cached_llm_response(
                prompt="Test prompt",
                system_prompt="System",
                model_type="actuator"
            )
        
        # Should increment consecutive failures
        mock_redis.incr.assert_called_with("llm:consecutive_failures")
