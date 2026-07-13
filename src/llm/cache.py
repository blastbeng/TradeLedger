import hashlib
import json
import logging
import re
import math
import time
from datetime import datetime, timezone
from typing import Optional, List, Dict
from src.config.settings import settings
from src.utils.redis_client import get_redis_client
from src.database import save_llm_metrics

logger = logging.getLogger(__name__)

def estimate_tokens(text: str) -> int:
    """Rough estimate of token count (1 token ~ 4 chars)."""
    return len(text) // 4

def _get_max_input_tokens(provider: str, model_type: str, is_fallback: bool) -> int:
    """Return the max input tokens for the given provider, model type, and fallback status."""
    if provider == "openai":
        if is_fallback:
            if model_type == "mind":
                return settings.OPENAI_MIND_FALLBACK_MAX_INPUT_TOKENS
            elif model_type == "weak":
                return settings.OPENAI_WEAK_FALLBACK_MAX_INPUT_TOKENS
            else:
                return settings.OPENAI_ACTUATOR_FALLBACK_MAX_INPUT_TOKENS
        else:
            if model_type == "mind":
                return settings.OPENAI_MIND_MAX_INPUT_TOKENS
            elif model_type == "weak":
                return settings.OPENAI_WEAK_MAX_INPUT_TOKENS
            else:
                return settings.OPENAI_ACTUATOR_MAX_INPUT_TOKENS
    else:  # ollama
        if is_fallback:
            if model_type == "mind":
                return settings.OLLAMA_MIND_FALLBACK_MAX_INPUT_TOKENS
            elif model_type == "weak":
                return settings.OLLAMA_WEAK_FALLBACK_MAX_INPUT_TOKENS
            else:
                return settings.OLLAMA_ACTUATOR_FALLBACK_MAX_INPUT_TOKENS
        else:
            if model_type == "mind":
                return settings.OLLAMA_MIND_MAX_INPUT_TOKENS
            elif model_type == "weak":
                return settings.OLLAMA_WEAK_MAX_INPUT_TOKENS
            else:
                return settings.OLLAMA_ACTUATOR_MAX_INPUT_TOKENS

def _get_weak_model_config() -> tuple:
    """Return (provider, model, base_url, api_key) for the weak model."""
    provider = settings.LLM_WEAK_PROVIDER or settings.LLM_PROVIDER
    if provider == "openai":
        model = settings.OPENAI_WEAK_MODEL or settings.OPENAI_MODEL
        base_url = settings.OPENAI_WEAK_BASE_URL or settings.OPENAI_BASE_URL
        api_key = settings.OPENAI_WEAK_API_KEY or settings.OPENAI_API_KEY
    else:  # ollama
        model = settings.OLLAMA_WEAK_MODEL or settings.OLLAMA_MODEL
        base_url = settings.OLLAMA_WEAK_BASE_URL or settings.OLLAMA_BASE_URL
        api_key = settings.OLLAMA_WEAK_API_KEY or settings.OLLAMA_API_KEY
    return (provider, model, base_url, api_key)

def _split_and_merge_prompt(
    prompt: str,
    system_prompt: str,
    model_type: str,
    provider: str,
    model: str,
    base_url: str,
    api_key: str,
    temperature: Optional[float],
    timeout: float,
    max_input_tokens: int,
    depth: int = 0,
) -> str:
    """
    Splits an oversized prompt into chunks, summarizes each chunk using the weak LLM,
    and merges the summaries into a single prompt that fits within the context window.
    """
    if depth >= 3:
        logger.warning("Max split/merge depth reached. Truncating prompt.")
        return prompt[:max_input_tokens * 4]  # 4 chars per token approx

    logger.warning(
        "Prompt size exceeds context window limit (%d tokens). Splitting and merging (depth=%d)...",
        max_input_tokens, depth
    )

    # Use the weak model for summarization to save time and tokens
    weak_provider, weak_model, weak_base_url, weak_api_key = _get_weak_model_config()
    weak_max_tokens = _get_max_input_tokens(weak_provider, "weak", False)

    # The chunk limit must fit within the weak model's context window
    chunk_limit = int(weak_max_tokens * 0.7)

    def _split_text(text: str, limit: int) -> List[str]:
        """Recursively split text by paragraphs, lines, and words to fit within limit."""
        if estimate_tokens(text) <= limit:
            return [text]
        
        # Try splitting by double newlines (paragraphs)
        parts = text.split('\n\n')
        if len(parts) > 1:
            return _aggregate_chunks(parts, limit)
        
        # Try splitting by single newlines (lines)
        parts = text.split('\n')
        if len(parts) > 1:
            return _aggregate_chunks(parts, limit)
        
        # Try splitting by words
        words = text.split(' ')
        if len(words) > 1:
            return _aggregate_chunks(words, limit, separator=' ')
        
        # If it's a single huge word, just truncate it
        return [text[:limit * 4]]  # 4 chars per token approx

    def _aggregate_chunks(parts: List[str], limit: int, separator: str = '\n\n') -> List[str]:
        """Aggregate parts into chunks that fit within the limit."""
        chunks = []
        current_chunk = ""
        for part in parts:
            candidate = current_chunk + separator + part if current_chunk else part
            if estimate_tokens(candidate) > limit and current_chunk:
                chunks.append(current_chunk)
                # If the part itself is too large, split it recursively
                if estimate_tokens(part) > limit:
                    chunks.extend(_split_text(part, limit))
                    current_chunk = ""
                else:
                    current_chunk = part
            else:
                current_chunk = candidate
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    # Initial split by paragraphs
    paragraphs = prompt.split('\n\n')
    chunks = _aggregate_chunks(paragraphs, chunk_limit)

    summaries = []
    for i, chunk in enumerate(chunks):
        logger.info("Summarizing chunk %d/%d using weak model...", i + 1, len(chunks))
        summary_prompt = (
            f"You are processing part {i+1} of {len(chunks)} of a large market analysis prompt. "
            f"Summarize the key data points, indicators, and insights from this chunk. "
            f"Preserve all important numbers, dates, and entity names.\n\n"
            f"Chunk:\n{chunk}"
        )
        
        try:
            # Use a fixed low temperature and default timeout for summarization
            summary_temperature = 0.1
            summary_timeout = settings.LLM_TIMEOUT
            
            # Call the LLM to summarize the chunk
            if weak_provider == "openai":
                from src.llm.llm_client import _get_openai_response
                result = _get_openai_response(
                    prompt=summary_prompt,
                    system_prompt="You are an expert summarizer for a stock trading bot.",
                    model=weak_model,
                    base_url=weak_base_url,
                    api_key=weak_api_key,
                    temperature=summary_temperature,
                    timeout=summary_timeout,
                    messages=None,
                    add_cache_control=False,
                    thinking_enabled=False,
                )
            else:
                from src.llm.llm_client import _get_ollama_response
                result = _get_ollama_response(
                    prompt=summary_prompt,
                    system_prompt="You are an expert summarizer for a stock trading bot.",
                    model=weak_model,
                    base_url=weak_base_url,
                    api_key=weak_api_key,
                    temperature=summary_temperature,
                    timeout=summary_timeout,
                    messages=None,
                    add_cache_control=False,
                    thinking_enabled=False,
                )
            summaries.append(result["content"])
        except Exception as e:
            logger.error("Failed to summarize chunk %d: %s. Truncating instead.", i + 1, e)
            # If summarization fails, truncate the chunk to fit the original model's limit
            summaries.append(chunk[:max_input_tokens * 4])
    
    # Combine summaries into a new prompt
    merged_prompt = (
        "The following is a merged summary of a large market analysis prompt. "
        "Use this information to make your trading decision.\n\n"
        + "\n\n".join(summaries)
    )
    
    # If the merged prompt is still too large, split and merge again
    if estimate_tokens(merged_prompt) > max_input_tokens:
        logger.warning("Merged prompt still exceeds limit (%d tokens). Splitting again...", estimate_tokens(merged_prompt))
        return _split_and_merge_prompt(
            prompt=merged_prompt,
            system_prompt=system_prompt,
            model_type=model_type,
            provider=provider,
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            timeout=timeout,
            max_input_tokens=max_input_tokens,
            depth=depth + 1,
        )
    
    return merged_prompt


def _normalize_text_for_cache(text: str) -> str:
    """Round all decimal numbers in text to 5 significant figures for stable cache keys.

    This normalizes the cache key so that tiny changes in floating-point values
    (e.g., 1.23456789 vs 1.23456788) don't cause cache misses. The actual prompt
    text sent to the LLM is not affected — only the cache key is normalized.
    Rounding to 5 significant figures preserves enough precision for prices and
    indicators while still normalizing floating-point noise, consistent with
    _normalize_for_hash.
    """
    if not text:
        return text
    def _round_num(match):
        try:
            val = float(match.group(0))
            if val == 0 or math.isnan(val) or math.isinf(val):
                return match.group(0)
            # Percentage-based rounding: round to 5 significant figures
            decimals = 4 - int(math.floor(math.log10(abs(val))))
            return f"{round(val, decimals)}"
        except (ValueError, OverflowError):
            return match.group(0)
    return re.sub(r'-?\d+\.\d+', _round_num, text)

def get_cached_llm_response(
    prompt: str,
    system_prompt: str = "",
    ttl: Optional[int] = None,
    market_hash: str = None,
    model_type: str = "actuator",
    temperature: Optional[float] = None,
    symbol: Optional[str] = None,
    messages: Optional[List[Dict[str, str]]] = None,
    request_type: Optional[str] = None,
) -> Optional[dict]:
    """
    Get an LLM response, using Redis cache to avoid duplicate calls.
    Returns a dict with keys: "response" (str), "provider" (str), "model" (str).
    If market_hash is provided, the cache key is based on that hash
    (representing the market snapshot). Otherwise, the key is based on
    the prompt and system prompt.
    ttl: time-to-live in seconds (default 30 minutes).
    model_type: "mind" for complex reasoning, "actuator" for fast time‑critical decisions.

    When the primary provider call fails, automatically falls back to the
    configured fallback provider (if any) using the per-role fallback settings
    (e.g., LLM_MIND_FALLBACK_PROVIDER, OPENAI_MIND_FALLBACK_MODEL, etc.).
    If no fallback provider is configured for the role, the global
    LLM_FALLBACK_PROVIDER is used. If no fallback is configured at all,
    the original error is re-raised.
    """
    redis_client = get_redis_client()

    if ttl is None:
        if model_type == "mind":
            ttl = settings.LLM_MIND_CACHE_TTL
        else:
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

    # Determine thinking mode based on model_type
    if model_type == "mind":
        thinking_enabled = settings.LLM_MIND_THINKING_ENABLED
    elif model_type == "weak":
        thinking_enabled = settings.LLM_WEAK_THINKING_ENABLED
    else:
        thinking_enabled = settings.LLM_ACTUATOR_THINKING_ENABLED

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

    # When market is closed (not in pre-market), use fallback models only to save tokens
    is_fallback = False
    if not _should_use_primary_model():
        fb_provider, fb_model, fb_base_url, fb_api_key = _get_fallback_provider_config(model_type)
        if fb_provider and fb_model:
            is_fallback = True
            logger.info("Market is closed - using fallback model only (provider=%s, model=%s, model_type=%s)", fb_provider, fb_model, model_type)
            provider = fb_provider
            model = fb_model
            base_url = fb_base_url
            api_key = fb_api_key

    # Build cache key
    if messages is not None:
        # Normalize message content for cache key to improve cache hit rate.
        # Numbers are rounded to 5 significant figures in the key only;
        # the actual messages sent to the LLM retain full precision.
        normalized_messages = [
            {**msg, "content": _normalize_text_for_cache(msg.get("content", ""))}
            for msg in messages
        ]
        key_data = json.dumps(
            {"messages": normalized_messages, "system": _normalize_text_for_cache(system_prompt), "model_type": model_type,
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
            {"prompt": _normalize_text_for_cache(prompt), "system": _normalize_text_for_cache(system_prompt), "model_type": model_type,
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
                            "request_type": request_type,
                            "is_fallback": data.get("is_fallback", False),
                        })
                    except Exception as metric_err:
                        logger.warning("Failed to save cache hit metric: %s", metric_err)
                    return data
            except (json.JSONDecodeError, TypeError):
                pass  # fall through to re-fetch
    except Exception as e:
        logger.warning(f"Redis cache get failed: {type(e).__name__}: {e}. Proceeding without cache.")

    logger.debug("LLM cache miss: model_type=%s, system_prompt=%.200s..., prompt=%.500s...", model_type, system_prompt, prompt)
    # Context window management
    max_input_tokens = _get_max_input_tokens(provider, model_type, is_fallback)
    # Reserve some tokens for the system prompt and completion
    effective_limit = int(max_input_tokens * 0.8)

    if messages is not None:
        total_tokens = sum(estimate_tokens(msg.get("content", "")) for msg in messages) + estimate_tokens(system_prompt)
        if total_tokens > effective_limit:
            logger.warning(
                "Messages size (~%d tokens) exceeds context window limit (%d). Splitting and merging...",
                total_tokens, effective_limit
            )
            # Copy messages to avoid mutating the caller's list
            messages = [dict(msg) for msg in messages]
            # Assume the last message is the user prompt that needs splitting
            if messages and messages[-1]["role"] == "user":
                user_content = messages[-1]["content"]
                merged_content = _split_and_merge_prompt(
                    prompt=user_content,
                    system_prompt=system_prompt,
                    model_type=model_type,
                    provider=provider,
                    model=model,
                    base_url=base_url,
                    api_key=api_key,
                    temperature=temperature,
                    timeout=effective_timeout,
                    max_input_tokens=effective_limit,
                )
                messages[-1]["content"] = merged_content
    else:
        prompt_tokens = estimate_tokens(prompt) + estimate_tokens(system_prompt)
        if prompt_tokens > effective_limit:
            prompt = _split_and_merge_prompt(
                prompt=prompt,
                system_prompt=system_prompt,
                model_type=model_type,
                provider=provider,
                model=model,
                base_url=base_url,
                api_key=api_key,
                temperature=temperature,
                timeout=effective_timeout,
                max_input_tokens=effective_limit,
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
                thinking_enabled=thinking_enabled,
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
                thinking_enabled=thinking_enabled,
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
                "request_type": request_type,
                "is_fallback": is_fallback,
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

        # Determine fallback provider
        if model_type == "mind":
            fallback_provider = settings.LLM_MIND_FALLBACK_PROVIDER or settings.LLM_FALLBACK_PROVIDER
        elif model_type == "weak":
            fallback_provider = settings.LLM_WEAK_FALLBACK_PROVIDER or settings.LLM_FALLBACK_PROVIDER
        else:
            fallback_provider = settings.LLM_ACTUATOR_FALLBACK_PROVIDER or settings.LLM_FALLBACK_PROVIDER

        if not fallback_provider:
            logger.warning(
                "LLM primary call failed and no fallback provider configured. "
                "Original error: %s", e
            )
            raise

        if fallback_provider == "openai":
            if model_type == "mind":
                fallback_model = settings.OPENAI_MIND_FALLBACK_MODEL or settings.OPENAI_FALLBACK_MODEL
                fallback_base_url = settings.OPENAI_MIND_FALLBACK_BASE_URL or settings.OPENAI_FALLBACK_BASE_URL or settings.OPENAI_BASE_URL
                fallback_api_key = settings.OPENAI_MIND_FALLBACK_API_KEY or settings.OPENAI_FALLBACK_API_KEY or settings.OPENAI_API_KEY
            elif model_type == "weak":
                fallback_model = settings.OPENAI_WEAK_FALLBACK_MODEL or settings.OPENAI_FALLBACK_MODEL
                fallback_base_url = settings.OPENAI_WEAK_FALLBACK_BASE_URL or settings.OPENAI_FALLBACK_BASE_URL or settings.OPENAI_BASE_URL
                fallback_api_key = settings.OPENAI_WEAK_FALLBACK_API_KEY or settings.OPENAI_FALLBACK_API_KEY or settings.OPENAI_API_KEY
            else:
                fallback_model = settings.OPENAI_ACTUATOR_FALLBACK_MODEL or settings.OPENAI_FALLBACK_MODEL
                fallback_base_url = settings.OPENAI_ACTUATOR_FALLBACK_BASE_URL or settings.OPENAI_FALLBACK_BASE_URL or settings.OPENAI_BASE_URL
                fallback_api_key = settings.OPENAI_ACTUATOR_FALLBACK_API_KEY or settings.OPENAI_FALLBACK_API_KEY or settings.OPENAI_API_KEY

            if fallback_api_key or fallback_base_url:
                logger.warning(
                    "Primary LLM call failed (%s). Falling back to OpenAI-compatible provider "
                    "for %s role (model=%s).", e, model_type, fallback_model
                )
                # Check fallback model context window
                fb_max_input_tokens = _get_max_input_tokens("openai", model_type, True)
                fb_effective_limit = int(fb_max_input_tokens * 0.8)
                if messages is not None:
                    fb_total_tokens = sum(estimate_tokens(msg.get("content", "")) for msg in messages) + estimate_tokens(system_prompt)
                    if fb_total_tokens > fb_effective_limit:
                        logger.warning("Fallback messages size (~%d tokens) exceeds limit (%d). Splitting...", fb_total_tokens, fb_effective_limit)
                        messages = [dict(msg) for msg in messages]
                        if messages and messages[-1]["role"] == "user":
                            messages[-1]["content"] = _split_and_merge_prompt(
                                prompt=messages[-1]["content"],
                                system_prompt=system_prompt,
                                model_type=model_type,
                                provider="openai",
                                model=fallback_model,
                                base_url=fallback_base_url,
                                api_key=fallback_api_key,
                                temperature=temperature,
                                timeout=effective_timeout,
                                max_input_tokens=fb_effective_limit,
                            )
                            api_messages = []
                            if system_prompt:
                                api_messages.append({"role": "system", "content": system_prompt})
                            api_messages.extend(messages)
                else:
                    fb_prompt_tokens = estimate_tokens(prompt) + estimate_tokens(system_prompt)
                    if fb_prompt_tokens > fb_effective_limit:
                        prompt = _split_and_merge_prompt(
                            prompt=prompt,
                            system_prompt=system_prompt,
                            model_type=model_type,
                            provider="openai",
                            model=fallback_model,
                            base_url=fallback_base_url,
                            api_key=fallback_api_key,
                            temperature=temperature,
                            timeout=effective_timeout,
                            max_input_tokens=fb_effective_limit,
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
                        thinking_enabled=thinking_enabled,
                    )
                    response_text = result["content"]
                    usage = result.get("usage", {})
                    used_provider = "openai"
                    used_model = fallback_model
                    is_fallback = True
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
                            "request_type": request_type,
                            "is_fallback": True,
                        })
                    except Exception as metric_err:
                        logger.warning("Failed to save fallback error metric: %s", metric_err)
                    logger.error("OpenAI fallback also failed: %s", fallback_e, exc_info=True)
                    raise
            else:
                logger.warning(
                    "Fallback provider is openai but no API key or base URL configured. "
                    "Original error: %s", e
                )
                raise
        elif fallback_provider == "ollama":
            if model_type == "mind":
                fallback_model = settings.OLLAMA_MIND_FALLBACK_MODEL or settings.OLLAMA_FALLBACK_MODEL
                fallback_base_url = settings.OLLAMA_MIND_FALLBACK_BASE_URL or settings.OLLAMA_FALLBACK_BASE_URL or settings.OLLAMA_BASE_URL
                fallback_api_key = settings.OLLAMA_MIND_FALLBACK_API_KEY or settings.OLLAMA_FALLBACK_API_KEY or settings.OLLAMA_API_KEY
            elif model_type == "weak":
                fallback_model = settings.OLLAMA_WEAK_FALLBACK_MODEL or settings.OLLAMA_FALLBACK_MODEL
                fallback_base_url = settings.OLLAMA_WEAK_FALLBACK_BASE_URL or settings.OLLAMA_FALLBACK_BASE_URL or settings.OLLAMA_BASE_URL
                fallback_api_key = settings.OLLAMA_WEAK_FALLBACK_API_KEY or settings.OLLAMA_FALLBACK_API_KEY or settings.OLLAMA_API_KEY
            else:
                fallback_model = settings.OLLAMA_ACTUATOR_FALLBACK_MODEL or settings.OLLAMA_FALLBACK_MODEL
                fallback_base_url = settings.OLLAMA_ACTUATOR_FALLBACK_BASE_URL or settings.OLLAMA_FALLBACK_BASE_URL or settings.OLLAMA_BASE_URL
                fallback_api_key = settings.OLLAMA_ACTUATOR_FALLBACK_API_KEY or settings.OLLAMA_FALLBACK_API_KEY or settings.OLLAMA_API_KEY

            if fallback_base_url:
                logger.warning(
                    "Primary LLM call failed (%s). Falling back to Ollama provider "
                    "for %s role (model=%s).", e, model_type, fallback_model
                )
                # Check fallback model context window
                fb_max_input_tokens = _get_max_input_tokens("ollama", model_type, True)
                fb_effective_limit = int(fb_max_input_tokens * 0.8)
                if messages is not None:
                    fb_total_tokens = sum(estimate_tokens(msg.get("content", "")) for msg in messages) + estimate_tokens(system_prompt)
                    if fb_total_tokens > fb_effective_limit:
                        logger.warning("Fallback messages size (~%d tokens) exceeds limit (%d). Splitting...", fb_total_tokens, fb_effective_limit)
                        messages = [dict(msg) for msg in messages]
                        if messages and messages[-1]["role"] == "user":
                            messages[-1]["content"] = _split_and_merge_prompt(
                                prompt=messages[-1]["content"],
                                system_prompt=system_prompt,
                                model_type=model_type,
                                provider="ollama",
                                model=fallback_model,
                                base_url=fallback_base_url,
                                api_key=fallback_api_key,
                                temperature=temperature,
                                timeout=effective_timeout,
                                max_input_tokens=fb_effective_limit,
                            )
                            api_messages = []
                            if system_prompt:
                                api_messages.append({"role": "system", "content": system_prompt})
                            api_messages.extend(messages)
                else:
                    fb_prompt_tokens = estimate_tokens(prompt) + estimate_tokens(system_prompt)
                    if fb_prompt_tokens > fb_effective_limit:
                        prompt = _split_and_merge_prompt(
                            prompt=prompt,
                            system_prompt=system_prompt,
                            model_type=model_type,
                            provider="ollama",
                            model=fallback_model,
                            base_url=fallback_base_url,
                            api_key=fallback_api_key,
                            temperature=temperature,
                            timeout=effective_timeout,
                            max_input_tokens=fb_effective_limit,
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
                        thinking_enabled=thinking_enabled,
                    )
                    response_text = result["content"]
                    usage = result.get("usage", {})
                    used_provider = "ollama"
                    used_model = fallback_model
                    is_fallback = True
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
                            "request_type": request_type,
                            "is_fallback": True,
                        })
                    except Exception as metric_err:
                        logger.warning("Failed to save fallback error metric: %s", metric_err)
                    logger.error("Ollama fallback also failed: %s", fallback_e, exc_info=True)
                    raise
            else:
                logger.warning(
                    "Fallback provider is ollama but no base URL configured. "
                    "Original error: %s", e
                )
                raise
        else:
            raise

    if response_text is None:
        logger.warning("LLM returned None response; not caching.")
        try:
            save_llm_metrics({
                "timestamp": time.time(),
                "provider": used_provider,
                "model": used_model,
                "model_type": model_type,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cache_hit": 0,
                "latency_ms": (time.time() - start_time) * 1000,
                "error": "LLM returned None response",
                "request_type": request_type,
                "is_fallback": is_fallback,
            })
        except Exception as metric_err:
            logger.warning("Failed to save None response metric: %s", metric_err)
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
            "request_type": request_type,
            "is_fallback": is_fallback,
        })
    except Exception as metric_err:
        logger.warning("Failed to save success metric: %s", metric_err)

    logger.debug("LLM response cached: %.500s...", response_text)
    # Store in cache as JSON
    cache_data = json.dumps({
        "response": response_text,
        "provider": used_provider,
        "model": used_model,
        "is_fallback": is_fallback,
    })
    try:
        redis_client.set(cache_key, cache_data, ex=ttl)
        logger.debug("LLM cache miss – stored response for key %s (provider=%s, model=%s)", cache_key[:32], used_provider, used_model)
    except Exception as e:
        logger.warning(f"Redis cache setex failed: {type(e).__name__}: {e}. Response will not be cached.")
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
    
    - Rounds floats to 5 significant figures (percentage-based rounding) to treat small absolute changes on high-priced assets as insignificant while preserving precision for low-priced assets.
    - Excludes keys containing 'timestamp', 'time', 'fetched_at', 'created_at',
      'published_at', 'last_eval', 'last_auto_resume' (volatile fields that
      change every cycle but don't affect trading decisions).
    - Converts None values to a string "null" for consistent serialization.
    """
    _VOLATILE_KEY_FRAGMENTS = ("timestamp", "time", "fetched_at", "created_at",
                                "published_at", "last_eval", "last_auto_resume",
                                "_last_state_save", "ohlcv_data", "raw_candles", "candles")
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
        if obj == 0:
            return 0.0
        if math.isnan(obj) or math.isinf(obj):
            return obj
        # Percentage-based rounding: round to 5 significant figures
        # to treat tiny floating-point noise as insignificant while
        # preserving meaningful price changes (e.g., 100.12 vs 100.14).
        decimals = 4 - int(math.floor(math.log10(abs(obj))))
        return round(obj, decimals)
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


def _should_use_primary_model() -> bool:
    """Check if primary models should be used based on market status.

    Returns True if market is open or in pre-market session (within 60 mins of open).
    Returns False if market is closed (use fallback models only to save tokens).
    Defaults to True if market status cannot be determined.
    """
    try:
        redis_client = get_redis_client()
        market_closed = redis_client.get("trading:market_closed")
        if not market_closed:
            return True  # Market is open

        # Market is closed - check if we're in pre-market (within 60 mins of open)
        next_open_raw = redis_client.get("trading:market_next_open")
        if next_open_raw:
            next_open_str = next_open_raw.decode() if isinstance(next_open_raw, bytes) else next_open_raw
            next_open_dt = datetime.fromisoformat(next_open_str)
            now = datetime.now(timezone.utc)
            time_to_open = (next_open_dt - now).total_seconds()
            if 0 < time_to_open <= 3600:  # within 60 minutes of open
                return True  # pre-market - use primary models

        # Market is closed - check if there are open positions that need management
        # (stop-loss reviews, max-hold decisions, etc. require primary model quality)
        open_positions_raw = redis_client.get("trading:open_positions_count")
        if open_positions_raw:
            try:
                if int(open_positions_raw) > 0:
                    return True  # Has open positions - use primary models for management
            except (ValueError, TypeError):
                pass

        return False  # market closed, no open positions - use fallback only
    except Exception:
        return True  # Default to primary if we can't determine market status


def _get_fallback_provider_config(model_type: str):
    """Return (provider, model, base_url, api_key) for the fallback configuration.

    Returns (None, None, None, None) if no fallback is configured.
    """
    if model_type == "mind":
        fallback_provider = settings.LLM_MIND_FALLBACK_PROVIDER or settings.LLM_FALLBACK_PROVIDER
    elif model_type == "weak":
        fallback_provider = settings.LLM_WEAK_FALLBACK_PROVIDER or settings.LLM_FALLBACK_PROVIDER
    else:
        fallback_provider = settings.LLM_ACTUATOR_FALLBACK_PROVIDER or settings.LLM_FALLBACK_PROVIDER

    if not fallback_provider:
        return (None, None, None, None)

    if fallback_provider == "openai":
        if model_type == "mind":
            fb_model = settings.OPENAI_MIND_FALLBACK_MODEL or settings.OPENAI_FALLBACK_MODEL
            fb_base_url = settings.OPENAI_MIND_FALLBACK_BASE_URL or settings.OPENAI_FALLBACK_BASE_URL or settings.OPENAI_BASE_URL
            fb_api_key = settings.OPENAI_MIND_FALLBACK_API_KEY or settings.OPENAI_FALLBACK_API_KEY or settings.OPENAI_API_KEY
        elif model_type == "weak":
            fb_model = settings.OPENAI_WEAK_FALLBACK_MODEL or settings.OPENAI_FALLBACK_MODEL
            fb_base_url = settings.OPENAI_WEAK_FALLBACK_BASE_URL or settings.OPENAI_FALLBACK_BASE_URL or settings.OPENAI_BASE_URL
            fb_api_key = settings.OPENAI_WEAK_FALLBACK_API_KEY or settings.OPENAI_FALLBACK_API_KEY or settings.OPENAI_API_KEY
        else:
            fb_model = settings.OPENAI_ACTUATOR_FALLBACK_MODEL or settings.OPENAI_FALLBACK_MODEL
            fb_base_url = settings.OPENAI_ACTUATOR_FALLBACK_BASE_URL or settings.OPENAI_FALLBACK_BASE_URL or settings.OPENAI_BASE_URL
            fb_api_key = settings.OPENAI_ACTUATOR_FALLBACK_API_KEY or settings.OPENAI_FALLBACK_API_KEY or settings.OPENAI_API_KEY
    else:  # ollama
        if model_type == "mind":
            fb_model = settings.OLLAMA_MIND_FALLBACK_MODEL or settings.OLLAMA_FALLBACK_MODEL
            fb_base_url = settings.OLLAMA_MIND_FALLBACK_BASE_URL or settings.OLLAMA_FALLBACK_BASE_URL or settings.OLLAMA_BASE_URL
            fb_api_key = settings.OLLAMA_MIND_FALLBACK_API_KEY or settings.OLLAMA_FALLBACK_API_KEY or settings.OLLAMA_API_KEY
        elif model_type == "weak":
            fb_model = settings.OLLAMA_WEAK_FALLBACK_MODEL or settings.OLLAMA_FALLBACK_MODEL
            fb_base_url = settings.OLLAMA_WEAK_FALLBACK_BASE_URL or settings.OLLAMA_FALLBACK_BASE_URL or settings.OLLAMA_BASE_URL
            fb_api_key = settings.OLLAMA_WEAK_FALLBACK_API_KEY or settings.OLLAMA_FALLBACK_API_KEY or settings.OLLAMA_API_KEY
        else:
            fb_model = settings.OLLAMA_ACTUATOR_FALLBACK_MODEL or settings.OLLAMA_FALLBACK_MODEL
            fb_base_url = settings.OLLAMA_ACTUATOR_FALLBACK_BASE_URL or settings.OLLAMA_FALLBACK_BASE_URL or settings.OLLAMA_BASE_URL
            fb_api_key = settings.OLLAMA_ACTUATOR_FALLBACK_API_KEY or settings.OLLAMA_FALLBACK_API_KEY or settings.OLLAMA_API_KEY

    return (fallback_provider, fb_model, fb_base_url, fb_api_key)
