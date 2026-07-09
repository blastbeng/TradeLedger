import hashlib
import json
import logging
import time
from typing import Optional, List, Dict
from src.config.settings import settings
from src.utils.redis_client import get_redis_client
from src.database import save_llm_metrics

logger = logging.getLogger(__name__)

def estimate_tokens(text: str) -> int:
    """Rough estimate of token count (1 token ~ 4 chars)."""
    return len(text) // 4

def get_cached_llm_response(
    prompt: str,
    system_prompt: str = "",
    ttl: Optional[int] = None,
    market_hash: str = None,
    model_type: str = "actuator",
    temperature: Optional[float] = None,
    symbol: Optional[str] = None,
    messages: Optional[List[Dict[str, str]]] = None,
) -> Optional[dict]:
    """
    Get an LLM response, using Redis cache to avoid duplicate calls.
    Returns a dict with keys: "response" (str), "provider" (str), "model" (str).
    If market_hash is provided, the cache key is based on that hash
    (representing the market snapshot). Otherwise, the key is based on
    the prompt and system prompt.
    ttl: time-to-live in seconds (default 30 minutes).
    model_type: "mind" for complex reasoning, "actuator" for fast time‑critical decisions.

    When the primary provider is "ollama" and the call fails, automatically
    falls back to the OpenAI-compatible endpoint (which can be OpenRouter)
    using the per-role settings, but only if the OpenAI API key or base URL
    for that role is configured.
    """
    redis_client = get_redis_client()

    if ttl is None:
        ttl = settings.LLM_CACHE_TTL


    # Determine effective provider and model for the primary choice
    if model_type == "mind":
        provider = settings.LLM_MIND_PROVIDER or settings.LLM_PROVIDER
    elif model_type == "weak":
        provider = settings.LLM_WEAK_PROVIDER or settings.LLM_PROVIDER
    else:
        provider = settings.LLM_ACTUATOR_PROVIDER or settings.LLM_PROVIDER

    if provider == "openai":
        if model_type == "mind":
            model = settings.OPENAI_MIND_MODEL or settings.OPENAI_MODEL
            base_url = settings.OPENAI_MIND_BASE_URL or settings.OPENAI_BASE_URL
            api_key = settings.OPENAI_MIND_API_KEY or settings.OPENAI_API_KEY
        elif model_type == "weak":
            model = settings.OPENAI_WEAK_MODEL or settings.OPENAI_MODEL
            base_url = settings.OPENAI_WEAK_BASE_URL or settings.OPENAI_BASE_URL
            api_key = settings.OPENAI_WEAK_API_KEY or settings.OPENAI_API_KEY
        else:
            model = settings.OPENAI_ACTUATOR_MODEL or settings.OPENAI_MODEL
            base_url = settings.OPENAI_ACTUATOR_BASE_URL or settings.OPENAI_BASE_URL
            api_key = settings.OPENAI_ACTUATOR_API_KEY or settings.OPENAI_API_KEY
    else:  # ollama
        if model_type == "mind":
            model = settings.OLLAMA_MIND_MODEL or settings.OLLAMA_MODEL
            base_url = settings.OLLAMA_MIND_BASE_URL or settings.OLLAMA_BASE_URL
            api_key = settings.OLLAMA_MIND_API_KEY or settings.OLLAMA_API_KEY
        elif model_type == "weak":
            model = settings.OLLAMA_WEAK_MODEL or settings.OLLAMA_MODEL
            base_url = settings.OLLAMA_WEAK_BASE_URL or settings.OLLAMA_BASE_URL
            api_key = settings.OLLAMA_WEAK_API_KEY or settings.OLLAMA_API_KEY
        else:
            model = settings.OLLAMA_ACTUATOR_MODEL or settings.OLLAMA_MODEL
            base_url = settings.OLLAMA_ACTUATOR_BASE_URL or settings.OLLAMA_BASE_URL
            api_key = settings.OLLAMA_ACTUATOR_API_KEY or settings.OLLAMA_API_KEY

    # Resolve effective temperature: use explicit temperature if provided,
    # otherwise fall back to per-role temperature, then global temperature.
    if temperature is None:
        if model_type == "mind" and settings.LLM_MIND_TEMPERATURE:
            temp_range = settings.parse_temperature_range(settings.LLM_MIND_TEMPERATURE)
            temperature = temp_range[0] if temp_range else settings.LLM_TEMPERATURE
        elif model_type == "weak" and settings.LLM_WEAK_TEMPERATURE:
            temp_range = settings.parse_temperature_range(settings.LLM_WEAK_TEMPERATURE)
            temperature = temp_range[0] if temp_range else settings.LLM_TEMPERATURE
        elif model_type == "actuator" and settings.LLM_ACTUATOR_TEMPERATURE:
            temp_range = settings.parse_temperature_range(settings.LLM_ACTUATOR_TEMPERATURE)
            temperature = temp_range[0] if temp_range else settings.LLM_TEMPERATURE
        else:
            temperature = settings.LLM_TEMPERATURE

    # Round temperature to the nearest 0.5 for cache key to improve cache hit rate
    # when temperature is dynamically computed based on complexity.
    cache_temp = round(temperature * 2) / 2 if temperature is not None else None

    # Determine effective timeout: use shorter timeout for actuator calls
    if model_type == "actuator":
        effective_timeout = settings.LLM_ACTUATOR_TIMEOUT
    else:
        effective_timeout = settings.LLM_TIMEOUT

    # Determine whether to add the cache_control header (only for providers that support it)
    add_cache_control = (
        settings.LLM_PROMPT_CACHING_ENABLED
        and provider in settings.LLM_PROMPT_CACHING_CONTROL_PROVIDERS
        and messages is not None
    )

    # Build cache key
    if messages is not None:
        # Build a deterministic key from the message list + system prompt
        key_data = json.dumps(
            {"messages": messages, "system": system_prompt, "model_type": model_type,
             "provider": provider, "model": model,
             "temperature": cache_temp if cache_temp is not None else settings.LLM_TEMPERATURE,
             "cache_version": settings.LLM_CACHE_VERSION},
            sort_keys=True
        )
        cache_key = f"llm:{hashlib.sha256(key_data.encode()).hexdigest()}"
    elif market_hash:
        sys_hash = hashlib.sha256(system_prompt.encode()).hexdigest()[:16] if system_prompt else "none"
        cache_key = f"llm:{settings.LLM_CACHE_VERSION}:{provider}:{model}:{model_type}:market:{market_hash}:sys:{sys_hash}:t{cache_temp if cache_temp is not None else 'def'}"
    else:
        key_data = json.dumps(
            {"prompt": prompt, "system": system_prompt, "model_type": model_type,
             "provider": provider, "model": model,
             "temperature": cache_temp if cache_temp is not None else settings.LLM_TEMPERATURE,
             "cache_version": settings.LLM_CACHE_VERSION},
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
                    # Record cache hit metric
                    try:
                        save_llm_metrics({
                            "timestamp": time.time(),
                            "provider": data.get("provider", provider),
                            "model": data.get("model", model),
                            "model_type": model_type,
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                            "cache_hit": 1,
                            "latency_ms": 0,
                            "error": None,
                        })
                    except Exception as metric_err:
                        logger.warning("Failed to save cache hit metric: %s", metric_err)
                    return data
            except (json.JSONDecodeError, TypeError):
                pass  # fall through to re-fetch
    except Exception as e:
        logger.warning(f"Redis cache get failed: {e}. Proceeding without cache.")

    logger.debug("LLM cache miss: model_type=%s, system_prompt=%.200s..., prompt=%.500s...", model_type, system_prompt, prompt)
    # Context window management: hard limit at 1,000,000 tokens
    MAX_TOKENS = 1_000_000
    prompt_tokens = estimate_tokens(prompt) + estimate_tokens(system_prompt)
    if prompt_tokens > MAX_TOKENS:
        logger.warning(
            "Prompt size (~%d tokens) exceeds context window limit (%d). Truncating...",
            prompt_tokens, MAX_TOKENS
        )
        keep_start = 2000
        keep_end = 4000
        if len(prompt) > keep_start + keep_end:
            prompt = (
                prompt[:keep_start] +
                "\n... [TRUNCATED DUE TO CONTEXT WINDOW LIMIT] ...\n" +
                prompt[-keep_end:]
            )

    if messages is not None:
        api_messages = []
        if system_prompt:
            api_messages.append({"role": "system", "content": system_prompt})
        # Use the provided message dicts directly (they already have role/content)
        api_messages.extend(messages)
    else:
        api_messages = None  # will be built inside _get_*_response from prompt/system_prompt

    # --- Primary call ---
    response_text = None
    used_provider = provider
    used_model = model
    usage = None

    start_time = time.time()
    try:
        if provider == "openai":
            from src.llm.llm_client import _get_openai_response
            result = _get_openai_response(
                prompt=prompt if messages is None else "",
                system_prompt=system_prompt if messages is None else "",
                model=model, base_url=base_url, api_key=api_key,
                temperature=temperature, timeout=effective_timeout,
                messages=api_messages,
                add_cache_control=add_cache_control,
            )
        else:
            from src.llm.llm_client import _get_ollama_response
            result = _get_ollama_response(
                prompt=prompt if messages is None else "",
                system_prompt=system_prompt if messages is None else "",
                model=model, base_url=base_url, api_key=api_key,
                temperature=temperature, timeout=effective_timeout,
                messages=api_messages,
                add_cache_control=add_cache_control,
            )

        response_text = result["content"]
        usage = result.get("usage", {})
        
        if not response_text or not response_text.strip():
            raise RuntimeError("LLM returned an empty response")
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        # Record error metric for primary call
        try:
            save_llm_metrics({
                "timestamp": time.time(),
                "provider": provider,
                "model": model,
                "model_type": model_type,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cache_hit": 0,
                "latency_ms": latency_ms,
                "error": str(e)[:500],
            })
        except Exception as metric_err:
            logger.warning("Failed to save primary error metric: %s", metric_err)

        logger.error("LLM primary call failed (provider=%s, model=%s, model_type=%s): %s", provider, model, model_type, e, exc_info=True)
        if not settings.LLM_FALLBACK_ENABLED:
            logger.warning(
                "LLM primary call failed and fallback is disabled (LLM_FALLBACK_ENABLED=False). "
                "Original error: %s", e
            )
            raise

        if provider == "ollama":
            # --- Fallback to OpenAI-compatible provider ---
            if model_type == "mind":
                fallback_model = settings.OPENAI_MIND_MODEL
                fallback_base_url = settings.OPENAI_MIND_BASE_URL or settings.OPENAI_BASE_URL
                fallback_api_key = settings.OPENAI_MIND_API_KEY or settings.OPENAI_API_KEY
            elif model_type == "weak":
                fallback_model = settings.OPENAI_WEAK_MODEL
                fallback_base_url = settings.OPENAI_WEAK_BASE_URL or settings.OPENAI_BASE_URL
                fallback_api_key = settings.OPENAI_WEAK_API_KEY or settings.OPENAI_API_KEY
            else:
                fallback_model = settings.OPENAI_ACTUATOR_MODEL
                fallback_base_url = settings.OPENAI_ACTUATOR_BASE_URL or settings.OPENAI_BASE_URL
                fallback_api_key = settings.OPENAI_ACTUATOR_API_KEY or settings.OPENAI_API_KEY

            if fallback_api_key or fallback_base_url:
                logger.warning(
                    "Ollama call failed (%s). Falling back to OpenAI-compatible provider "
                    "for %s role (model=%s).", e, model_type, fallback_model
                )
                fallback_start = time.time()
                try:
                    from src.llm.llm_client import _get_openai_response
                    result = _get_openai_response(
                        prompt=prompt if messages is None else "",
                        system_prompt=system_prompt if messages is None else "",
                        model=fallback_model,
                        base_url=fallback_base_url,
                        api_key=fallback_api_key,
                        temperature=temperature,
                        timeout=effective_timeout,
                        messages=api_messages,
                        add_cache_control=add_cache_control,
                    )
                    response_text = result["content"]
                    usage = result.get("usage", {})
                    used_provider = "openai"
                    used_model = fallback_model
                except Exception as fallback_e:
                    fallback_latency = (time.time() - fallback_start) * 1000
                    try:
                        save_llm_metrics({
                            "timestamp": time.time(),
                            "provider": "openai",
                            "model": fallback_model,
                            "model_type": model_type,
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                            "cache_hit": 0,
                            "latency_ms": fallback_latency,
                            "error": str(fallback_e)[:500],
                        })
                    except Exception as metric_err:
                        logger.warning("Failed to save fallback error metric: %s", metric_err)
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
            if model_type == "mind":
                fallback_model = settings.OLLAMA_MIND_MODEL
                fallback_base_url = settings.OLLAMA_MIND_BASE_URL or settings.OLLAMA_BASE_URL
                fallback_api_key = settings.OLLAMA_MIND_API_KEY or settings.OLLAMA_API_KEY
            elif model_type == "weak":
                fallback_model = settings.OLLAMA_WEAK_MODEL
                fallback_base_url = settings.OLLAMA_WEAK_BASE_URL or settings.OLLAMA_BASE_URL
                fallback_api_key = settings.OLLAMA_WEAK_API_KEY or settings.OLLAMA_API_KEY
            else:
                fallback_model = settings.OLLAMA_ACTUATOR_MODEL
                fallback_base_url = settings.OLLAMA_ACTUATOR_BASE_URL or settings.OLLAMA_BASE_URL
                fallback_api_key = settings.OLLAMA_ACTUATOR_API_KEY or settings.OLLAMA_API_KEY

            if fallback_base_url:
                logger.warning(
                    "OpenAI call failed (%s). Falling back to Ollama provider "
                    "for %s role (model=%s).", e, model_type, fallback_model
                )
                fallback_start = time.time()
                try:
                    from src.llm.llm_client import _get_ollama_response
                    result = _get_ollama_response(
                        prompt=prompt if messages is None else "",
                        system_prompt=system_prompt if messages is None else "",
                        model=fallback_model,
                        base_url=fallback_base_url,
                        api_key=fallback_api_key,
                        temperature=temperature,
                        timeout=effective_timeout,
                        messages=api_messages,
                        add_cache_control=add_cache_control,
                    )
                    response_text = result["content"]
                    usage = result.get("usage", {})
                    used_provider = "ollama"
                    used_model = fallback_model
                except Exception as fallback_e:
                    fallback_latency = (time.time() - fallback_start) * 1000
                    try:
                        save_llm_metrics({
                            "timestamp": time.time(),
                            "provider": "ollama",
                            "model": fallback_model,
                            "model_type": model_type,
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                            "cache_hit": 0,
                            "latency_ms": fallback_latency,
                            "error": str(fallback_e)[:500],
                        })
                    except Exception as metric_err:
                        logger.warning("Failed to save fallback error metric: %s", metric_err)
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

    # Record success metric (primary or fallback)
    latency_ms = (time.time() - start_time) * 1000
    try:
        save_llm_metrics({
            "timestamp": time.time(),
            "provider": used_provider,
            "model": used_model,
            "model_type": model_type,
            "prompt_tokens": usage.get("prompt_tokens", 0) if usage else 0,
            "completion_tokens": usage.get("completion_tokens", 0) if usage else 0,
            "total_tokens": usage.get("total_tokens", 0) if usage else 0,
            "cache_hit": 0,
            "latency_ms": latency_ms,
            "error": None,
        })
    except Exception as metric_err:
        logger.warning("Failed to save success metric: %s", metric_err)

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
