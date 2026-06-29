import hashlib
import json
import logging
from typing import Optional
from src.config.settings import settings
from src.utils.redis_client import get_redis_client

logger = logging.getLogger(__name__)

def get_cached_llm_response(
    prompt: str,
    system_prompt: str = "",
    ttl: int = 300,
    market_hash: str = None,
    model_type: str = "actuator",
    temperature: Optional[float] = None,
) -> Optional[dict]:
    """
    Get an LLM response, using Redis cache to avoid duplicate calls.
    Returns a dict with keys: "response" (str), "provider" (str), "model" (str).
    If market_hash is provided, the cache key is based on that hash
    (representing the market snapshot). Otherwise, the key is based on
    the prompt and system prompt.
    ttl: time-to-live in seconds (default 5 minutes).
    model_type: "mind" for complex reasoning, "actuator" for fast time‑critical decisions.

    When the primary provider is "ollama" and the call fails, automatically
    falls back to the OpenAI-compatible endpoint (which can be OpenRouter)
    using the per-role settings, but only if the OpenAI API key or base URL
    for that role is configured.
    """
    redis_client = get_redis_client()

    # Determine effective provider and model for the primary choice
    if model_type == "mind":
        provider = settings.LLM_MIND_PROVIDER or settings.LLM_PROVIDER
    else:
        provider = settings.LLM_ACTUATOR_PROVIDER or settings.LLM_PROVIDER

    if provider == "openai":
        model = (settings.OPENAI_MIND_MODEL or settings.OPENAI_MODEL) if model_type == "mind" else (settings.OPENAI_ACTUATOR_MODEL or settings.OPENAI_MODEL)
        base_url = (settings.OPENAI_MIND_BASE_URL or settings.OPENAI_BASE_URL) if model_type == "mind" else (settings.OPENAI_ACTUATOR_BASE_URL or settings.OPENAI_BASE_URL)
        api_key = (settings.OPENAI_MIND_API_KEY or settings.OPENAI_API_KEY) if model_type == "mind" else (settings.OPENAI_ACTUATOR_API_KEY or settings.OPENAI_API_KEY)
    else:  # ollama
        model = (settings.OLLAMA_MIND_MODEL or settings.OLLAMA_MODEL) if model_type == "mind" else (settings.OLLAMA_ACTUATOR_MODEL or settings.OLLAMA_MODEL)
        base_url = (settings.OLLAMA_MIND_BASE_URL or settings.OLLAMA_BASE_URL) if model_type == "mind" else (settings.OLLAMA_ACTUATOR_BASE_URL or settings.OLLAMA_BASE_URL)
        api_key = (settings.OLLAMA_MIND_API_KEY or settings.OLLAMA_API_KEY) if model_type == "mind" else (settings.OLLAMA_ACTUATOR_API_KEY or settings.OLLAMA_API_KEY)

    # Round temperature to 1 decimal place for cache key to improve cache hit rate
    # when temperature is dynamically computed based on complexity.
    cache_temp = round(temperature, 1) if temperature is not None else None

    # Build cache key
    if market_hash:
        cache_key = f"llm:{provider}:{model}:{model_type}:market:{market_hash}:t{cache_temp if cache_temp is not None else 'def'}"
    else:
        key_data = json.dumps(
            {"prompt": prompt, "system": system_prompt, "model_type": model_type,
             "provider": provider, "model": model,
             "temperature": cache_temp if cache_temp is not None else settings.LLM_TEMPERATURE},
            sort_keys=True
        )
        cache_key = f"llm:{hashlib.sha256(key_data.encode()).hexdigest()}"

    # Try cache
    try:
        cached = redis_client.get(cache_key)
        if cached:
            try:
                data = json.loads(cached)
                if isinstance(data, dict) and "response" in data:
                    logger.info("LLM cache hit: key=%.32s, model_type=%s", cache_key, model_type)
                    return data
            except (json.JSONDecodeError, TypeError):
                pass  # fall through to re-fetch
    except Exception as e:
        logger.warning(f"Redis cache get failed: {e}. Proceeding without cache.")

    logger.debug("LLM cache miss: model_type=%s, system_prompt=%.200s..., prompt=%.500s...", model_type, system_prompt, prompt)
    # --- Primary call ---
    response_text = None
    used_provider = provider
    used_model = model

    try:
        if provider == "openai":
            from src.llm.llm_client import _get_openai_response
            response_text = _get_openai_response(prompt, system_prompt, model=model, base_url=base_url, api_key=api_key, temperature=temperature)
        else:
            from src.llm.llm_client import _get_ollama_response
            response_text = _get_ollama_response(prompt, system_prompt, model=model, base_url=base_url, api_key=api_key, temperature=temperature)
        
        if not response_text or not response_text.strip():
            raise RuntimeError("LLM returned an empty response")
    except Exception as e:
        logger.error("LLM primary call failed (provider=%s, model=%s, model_type=%s): %s", provider, model, model_type, e, exc_info=True)
        if not settings.LLM_FALLBACK_ENABLED:
            logger.warning(
                "LLM primary call failed and fallback is disabled (LLM_FALLBACK_ENABLED=False). "
                "Original error: %s", e
            )
            raise

        if provider == "ollama":
            # --- Fallback to OpenAI-compatible provider ---
            fallback_model = settings.OPENAI_MIND_MODEL if model_type == "mind" else settings.OPENAI_ACTUATOR_MODEL
            fallback_base_url = (settings.OPENAI_MIND_BASE_URL or settings.OPENAI_BASE_URL) if model_type == "mind" else (settings.OPENAI_ACTUATOR_BASE_URL or settings.OPENAI_BASE_URL)
            fallback_api_key = (settings.OPENAI_MIND_API_KEY or settings.OPENAI_API_KEY) if model_type == "mind" else (settings.OPENAI_ACTUATOR_API_KEY or settings.OPENAI_API_KEY)

            if fallback_api_key or fallback_base_url:
                logger.warning(
                    "Ollama call failed (%s). Falling back to OpenAI-compatible provider "
                    "for %s role (model=%s).", e, model_type, fallback_model
                )
                try:
                    from src.llm.llm_client import _get_openai_response
                    response_text = _get_openai_response(
                        prompt, system_prompt,
                        model=fallback_model,
                        base_url=fallback_base_url,
                        api_key=fallback_api_key,
                        temperature=temperature,
                    )
                    used_provider = "openai"
                    used_model = fallback_model
                except Exception as fallback_e:
                    logger.error("OpenAI fallback also failed: %s", fallback_e, exc_info=True)
                    raise
            else:
                logger.warning(
                    "Ollama call failed and no OpenAI fallback credentials configured. "
                    "Original error: %s", e
                )
                raise
        elif provider == "openai":
            # --- Fallback to Ollama provider ---
            fallback_model = settings.OLLAMA_MIND_MODEL if model_type == "mind" else settings.OLLAMA_ACTUATOR_MODEL
            fallback_base_url = (settings.OLLAMA_MIND_BASE_URL or settings.OLLAMA_BASE_URL) if model_type == "mind" else (settings.OLLAMA_ACTUATOR_BASE_URL or settings.OLLAMA_BASE_URL)
            fallback_api_key = (settings.OLLAMA_MIND_API_KEY or settings.OLLAMA_API_KEY) if model_type == "mind" else (settings.OLLAMA_ACTUATOR_API_KEY or settings.OLLAMA_API_KEY)

            if fallback_base_url:
                logger.warning(
                    "OpenAI call failed (%s). Falling back to Ollama provider "
                    "for %s role (model=%s).", e, model_type, fallback_model
                )
                try:
                    from src.llm.llm_client import _get_ollama_response
                    response_text = _get_ollama_response(
                        prompt, system_prompt,
                        model=fallback_model,
                        base_url=fallback_base_url,
                        api_key=fallback_api_key,
                        temperature=temperature,
                    )
                    used_provider = "ollama"
                    used_model = fallback_model
                except Exception as fallback_e:
                    logger.error("Ollama fallback also failed: %s", fallback_e, exc_info=True)
                    raise
            else:
                logger.warning(
                    "OpenAI call failed and no Ollama fallback base URL configured. "
                    "Original error: %s", e
                )
                raise
        else:
            raise

    if response_text is None:
        logger.warning("LLM returned None response; not caching.")
        return None

    logger.debug("LLM response cached: %.500s...", response_text)
    # Store in cache as JSON
    cache_data = json.dumps({
        "response": response_text,
        "provider": used_provider,
        "model": used_model,
    })
    try:
        redis_client.set(cache_key, cache_data, ex=ttl)
        logger.debug("LLM cache miss – stored response for key %s (provider=%s, model=%s)", cache_key[:32], used_provider, used_model)
    except Exception as e:
        logger.warning(f"Redis cache setex failed: {e}. Response will not be cached.")
    return {
        "response": response_text,
        "provider": used_provider,
        "model": used_model,
    }


def _stringify_keys(obj):
    """Recursively convert all dict keys to strings for JSON-safe sorting."""
    if isinstance(obj, dict):
        return {str(k): _stringify_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_stringify_keys(item) for item in obj]
    return obj


def _normalize_for_hash(obj, depth=0):
    """Recursively normalize data for stable hashing.
    
    - Rounds floats to 12 decimal places to reduce noise from tiny price changes while preserving precision for very small indicator values (e.g., MACD histogram).
    - Excludes keys containing 'timestamp', 'time', 'fetched_at', 'created_at',
      'published_at', 'last_eval', 'last_auto_resume' (volatile fields that
      change every cycle but don't affect trading decisions).
    - Converts None values to a string "null" for consistent serialization.
    """
    _VOLATILE_KEY_FRAGMENTS = ("timestamp", "time", "fetched_at", "created_at",
                                "published_at", "last_eval", "last_auto_resume",
                                "_last_state_save")
    if depth > 10:
        return None
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            key_str = str(k).lower()
            if any(frag in key_str for frag in _VOLATILE_KEY_FRAGMENTS):
                continue
            result[k] = _normalize_for_hash(v, depth + 1)
        return result
    if isinstance(obj, list):
        return [_normalize_for_hash(item, depth + 1) for item in obj]
    if isinstance(obj, float):
        # Round to 12 decimal places — enough precision for very small indicator values (e.g., MACD histogram),
        # while filtering out floating-point noise that changes every cycle.
        return round(obj, 12)
    if obj is None:
        return "null"
    return obj


def compute_market_hash(data: dict) -> str:
    """Return a SHA-256 hex digest of the JSON-serialised market data.
    
    Volatile fields (timestamps, etc.) are excluded and floats are rounded
    so that essentially-identical market states produce the same hash,
    enabling LLM response caching.
    """
    normalized = _normalize_for_hash(data)
    safe_data = _stringify_keys(normalized)
    serialized = json.dumps(safe_data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()
