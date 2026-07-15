import asyncio
import atexit
import hashlib
import json
import logging
import re
import math
import random
import time
import concurrent.futures
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Tuple, Any
from src.config.settings import settings
from src.utils.redis_client import get_redis_client
from src.database import save_llm_metrics, add_model_to_blacklist, get_active_blacklisted_models, remove_model_from_blacklist

logger = logging.getLogger(__name__)

# Shared thread pool for chunk summarization to avoid creating/destroying
# a pool on every split/merge operation.
_split_merge_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=5, thread_name_prefix="split-merge"
)
atexit.register(lambda: _split_merge_executor.shutdown(wait=False))

# Dedicated thread pool for LLM calls to prevent exhausting the default asyncio
# executor, which would block the web server and Telegram bot.
_llm_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=20, thread_name_prefix="llm-call"
)
atexit.register(lambda: _llm_executor.shutdown(wait=False))

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
        models = settings.OPENAI_WEAK_MODEL or settings.OPENAI_MODEL
        base_url = settings.OPENAI_WEAK_BASE_URL or settings.OPENAI_BASE_URL
        api_key = settings.OPENAI_WEAK_API_KEY or settings.OPENAI_API_KEY
    else:  # ollama
        models = settings.OLLAMA_WEAK_MODEL or settings.OLLAMA_MODEL
        base_url = settings.OLLAMA_WEAK_BASE_URL or settings.OLLAMA_BASE_URL
        api_key = settings.OLLAMA_WEAK_API_KEY or settings.OLLAMA_API_KEY
    model = models[0] if models else None
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

    # Extract output format instructions from the end of the prompt
    # to preserve them verbatim in the merged result.  Without this,
    # summarization condenses the exact JSON schema into a generic
    # description, and the LLM may return free-form text instead of
    # valid JSON.
    output_format_section = ""
    json_marker_idx = prompt.rfind("Return JSON:")
    if json_marker_idx != -1:
        output_format_section = prompt[json_marker_idx:]
        prompt = prompt[:json_marker_idx].rstrip()

    logger.warning(
        "Prompt size exceeds context window limit (%d tokens). Splitting and merging (depth=%d)...",
        max_input_tokens, depth
    )

    # Use the weak model for summarization to save time and tokens
    weak_provider, weak_model, weak_base_url, weak_api_key = _get_weak_model_config()
    weak_max_tokens = _get_max_input_tokens(weak_provider, "weak", False)

    # The chunk limit must fit within the weak model's context window, leaving room for instructions.
    # Decrease the chunk limit with each recursion depth to ensure the merged prompt shrinks.
    chunk_limit = int(weak_max_tokens * 0.6 / (depth + 1))

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
            if estimate_tokens(candidate) > limit:
                # Flush the current chunk if it has content
                if current_chunk:
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

    def _summarize_chunk(i: int, chunk: str) -> str:
        logger.info("Summarizing chunk %d/%d using weak model...", i + 1, len(chunks))
        summary_prompt = (
            f"You are processing part {i+1} of {len(chunks)} of a large market analysis prompt. "
            f"Summarize the key data points, indicators, and insights from this chunk. "
            f"Preserve all important numbers, dates, and entity names.\n\n"
            f"Chunk:\n{chunk}"
        )
        
        try:
            # Use a fixed low temperature and capped timeout for summarization
            summary_temperature = 0.1
            summary_timeout = min(settings.LLM_TIMEOUT, 60.0)
            
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
            return result["content"]
        except Exception as e:
            logger.error("Failed to summarize chunk %d: %s. Truncating instead.", i + 1, e)
            # If summarization fails, truncate the chunk to fit the weak model's limit
            return chunk[:chunk_limit * 4]

    # Summarize chunks in parallel using the shared module-level thread pool
    summaries = []
    future_to_chunk = {
        _split_merge_executor.submit(_summarize_chunk, i, chunk): i
        for i, chunk in enumerate(chunks)
    }
    # Collect results in order
    results = [None] * len(chunks)
    for future in concurrent.futures.as_completed(future_to_chunk):
        idx = future_to_chunk[future]
        try:
            results[idx] = future.result()
        except Exception as e:
            logger.error("Chunk %d summarization failed unexpectedly: %s", idx, e)
            results[idx] = chunks[idx][:chunk_limit * 4]
    summaries = [r for r in results if r is not None]
    
    # Combine summaries into a new prompt
    merged_prompt = (
        "The following is a merged summary of a large market analysis prompt. "
        "Use this information to make your trading decision.\n\n"
        + "\n\n".join(summaries)
    )

    # Append the preserved output format section verbatim so the LLM
    # always receives the exact JSON schema instructions.
    if output_format_section:
        merged_prompt = merged_prompt + "\n\n" + output_format_section

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
    """Round all decimal numbers in text to 6 significant figures for stable cache keys.

    This normalizes the cache key so that tiny changes in floating-point values
    (e.g., 1.23456789 vs 1.23456788) don't cause cache misses. The actual prompt
    text sent to the LLM is not affected — only the cache key is normalized.
    Rounding to 6 significant figures preserves enough precision for prices and
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
            # Percentage-based rounding: round to 6 significant figures
            decimals = 5 - int(math.floor(math.log10(abs(val))))
            return f"{round(val, decimals)}"
        except (ValueError, OverflowError):
            return match.group(0)
    return re.sub(r'-?\d+(?:\.\d+)?[eE][+-]?\d+|-?\d+\.\d+', _round_num, text)

def _compute_fee_fingerprint() -> str:
    """Compute a short fingerprint of fee-related settings for cache key inclusion.

    When fee parameters change via settings.reload(), this fingerprint changes,
    invalidating cached LLM responses that contain fee break-even calculations.
    """
    raw = (
        f"{settings.STOCK_FEE_PERC}:{settings.STOCK_FEE_MIN}:{settings.STOCK_FEE_FIXED}:"
        f"{settings.TOBIN_TAX_RATE}:{settings.BTP_FEE_PERC}:{settings.BTP_MIN_FEE}:"
        f"{settings.BTP_IS_PRIMARY_ISSUANCE}"
    )
    return hashlib.md5(raw.encode()).hexdigest()[:8]


def _save_metric(metric_data: dict) -> None:
    """Helper to save LLM metrics and log warnings on failure."""
    try:
        save_llm_metrics(metric_data)
    except Exception as metric_err:
        logger.warning("Failed to save LLM metric: %s", metric_err)


def _sync_blacklist_from_db():
    """Load active blacklisted models from DB into Redis on startup."""
    redis_client = get_redis_client()
    try:
        active = get_active_blacklisted_models()
        for item in active:
            model = item["model"]
            expires_at = item["expires_at"]
            if isinstance(expires_at, str):
                expires_at = datetime.fromisoformat(expires_at.replace('Z', '+00:00'))
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
            
            remaining_ttl = (expires_at - datetime.now(timezone.utc)).total_seconds()
            if remaining_ttl > 0:
                redis_client.setex(f"llm:blacklist:{model}", int(remaining_ttl), "1")
            else:
                remove_model_from_blacklist(model)
    except Exception as e:
        logger.warning(f"Failed to sync blacklist from DB: {e}")

def _is_model_blacklisted(redis_client, model: str) -> bool:
    """Check if a model is currently blacklisted in Redis."""
    try:
        return bool(redis_client.exists(f"llm:blacklist:{model}"))
    except Exception:
        return False

def _record_model_success(redis_client, model: str):
    """Reset failure counters on successful call."""
    try:
        redis_client.delete(f"llm:fail_count:{model}")
        redis_client.delete(f"llm:blacklist_level:{model}")
        remove_model_from_blacklist(model)
    except Exception:
        pass

def _record_model_failure(redis_client, model: str, provider: str, error: str):
    """Track failures and blacklist model if threshold reached."""
    try:
        fail_count = redis_client.incr(f"llm:fail_count:{model}")
        redis_client.expire(f"llm:fail_count:{model}", 3600)  # 1 hour window
        
        if fail_count >= 3:
            level = int(redis_client.get(f"llm:blacklist_level:{model}") or 1)
            ttl = min(3600 * level, 86400)  # 1h * level, max 24h
            redis_client.setex(f"llm:blacklist:{model}", ttl, "1")
            redis_client.incr(f"llm:blacklist_level:{model}")
            redis_client.expire(f"llm:blacklist_level:{model}", 86400 * 7)  # keep level for 7 days
            redis_client.delete(f"llm:fail_count:{model}")
            
            expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl)
            add_model_to_blacklist(model, provider, error[:500], expires_at)
            logger.warning(f"Model {model} blacklisted for {ttl}s due to repeated failures.")
    except Exception as e:
        logger.warning(f"Failed to record model failure: {e}")


def get_model_failure_stats() -> List[Dict[str, Any]]:
    """Get current failure counts and blacklist levels from Redis."""
    redis_client = get_redis_client()
    stats = {}
    try:
        # Get all fail count keys
        fail_keys = redis_client.keys("llm:fail_count:*")
        for key in fail_keys:
            if isinstance(key, bytes):
                key = key.decode('utf-8')
            model = key.split(":", 2)[2]
            count = int(redis_client.get(key) or 0)
            stats[model] = {"model": model, "fail_count": count, "blacklist_level": 0, "blacklisted": False, "ttl_remaining": 0}
        
        # Get all blacklist level keys
        level_keys = redis_client.keys("llm:blacklist_level:*")
        for key in level_keys:
            if isinstance(key, bytes):
                key = key.decode('utf-8')
            model = key.split(":", 2)[2]
            if model not in stats:
                stats[model] = {"model": model, "fail_count": 0, "blacklist_level": 0, "blacklisted": False, "ttl_remaining": 0}
            stats[model]["blacklist_level"] = int(redis_client.get(key) or 1)
        
        # Get all active blacklist keys
        blacklist_keys = redis_client.keys("llm:blacklist:*")
        for key in blacklist_keys:
            if isinstance(key, bytes):
                key = key.decode('utf-8')
            model = key.split(":", 2)[2]
            if model not in stats:
                stats[model] = {"model": model, "fail_count": 0, "blacklist_level": 0, "blacklisted": False, "ttl_remaining": 0}
            stats[model]["blacklisted"] = True
            ttl = redis_client.ttl(key)
            stats[model]["ttl_remaining"] = ttl if ttl > 0 else 0
            
    except Exception as e:
        logger.warning(f"Failed to get model failure stats: {e}")
    
    return list(stats.values())


def _build_cache_key(
    messages: Optional[List[Dict[str, str]]],
    system_prompt: str,
    model_type: str,
    provider: str,
    model: str,
    cache_temp: Optional[float],
    market_hash: Optional[str],
    prompt: str,
) -> str:
    """Build the Redis cache key based on prompt/messages and model config."""
    fee_fp = _compute_fee_fingerprint()
    if messages is not None:
        normalized_messages = [
            {**msg, "content": _normalize_text_for_cache(msg.get("content", ""))}
            for msg in messages
        ]
        key_data = json.dumps(
            {"messages": normalized_messages, "system": _normalize_text_for_cache(system_prompt), "model_type": model_type,
             "provider": provider, "model": model,
             "temperature": cache_temp if cache_temp is not None else settings.LLM_TEMPERATURE,
             "cache_version": settings.LLM_CACHE_VERSION,
             "fee_fp": fee_fp},
            sort_keys=True
        )
        return f"llm:{hashlib.sha256(key_data.encode()).hexdigest()}"
    elif market_hash:
        sys_hash = hashlib.sha256(system_prompt.encode()).hexdigest()[:16] if system_prompt else "none"
        return f"llm:{settings.LLM_CACHE_VERSION}:{fee_fp}:{provider}:{model}:{model_type}:market:{market_hash}:sys:{sys_hash}:t{cache_temp if cache_temp is not None else 'def'}"
    else:
        key_data = json.dumps(
            {"prompt": _normalize_text_for_cache(prompt), "system": _normalize_text_for_cache(system_prompt), "model_type": model_type,
             "provider": provider, "model": model,
             "temperature": cache_temp if cache_temp is not None else settings.LLM_TEMPERATURE,
             "cache_version": settings.LLM_CACHE_VERSION,
             "fee_fp": fee_fp},
            sort_keys=True
        )
        return f"llm:{hashlib.sha256(key_data.encode()).hexdigest()}"


def _get_cached_response(
    redis_client,
    cache_key: str,
    model_type: str,
    provider: str,
    model: str,
    request_type: Optional[str],
) -> Optional[dict]:
    """Attempt to fetch a cached LLM response from Redis."""
    try:
        cached = redis_client.get(cache_key)
        if cached:
            try:
                data = json.loads(cached)
                if isinstance(data, dict) and "response" in data:
                    logger.info("LLM cache hit: key=%.32s, model_type=%s", cache_key, model_type)
                    _save_metric({
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
                    return data
            except (json.JSONDecodeError, TypeError):
                pass
    except Exception as e:
        logger.warning(f"Redis cache get failed: {type(e).__name__}: {e}. Proceeding without cache.")
    return None


def _manage_context_window(
    messages: Optional[List[Dict[str, str]]],
    prompt: str,
    system_prompt: str,
    model_type: str,
    provider: str,
    model: str,
    base_url: str,
    api_key: str,
    temperature: Optional[float],
    effective_timeout: float,
    max_input_tokens: int,
    effective_limit: int,
) -> Tuple[Optional[List[Dict[str, str]]], str]:
    """Manage context window limits by splitting and merging oversized prompts."""
    if messages is not None:
        total_tokens = sum(estimate_tokens(msg.get("content", "")) for msg in messages) + estimate_tokens(system_prompt)
        if total_tokens > effective_limit:
            logger.warning(
                "Messages size (~%d tokens) exceeds context window limit (%d). Splitting and merging...",
                total_tokens, effective_limit
            )
            messages = [dict(msg) for msg in messages]
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
                    max_input_tokens=effective_limit - estimate_tokens(system_prompt),
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
                max_input_tokens=effective_limit - estimate_tokens(system_prompt),
            )
    return messages, prompt


def _execute_primary_call(
    provider: str,
    models: List[str],
    base_url: str,
    api_key: str,
    temperature: Optional[float],
    effective_timeout: float,
    messages: Optional[List[Dict[str, str]]],
    api_messages: Optional[List[Dict[str, str]]],
    prompt: str,
    system_prompt: str,
    add_cache_control: bool,
    thinking_enabled: bool,
    model_type: str,
    request_type: Optional[str],
    is_fallback: bool,
) -> Tuple[str, dict, str, str, bool]:
    """Execute the primary LLM call and return response, usage, and model info."""
    redis_client = get_redis_client()
    available_models = [m for m in models if not _is_model_blacklisted(redis_client, m)]
    
    if not available_models:
        raise RuntimeError("All primary models are blacklisted or unavailable")
        
    shuffled_models = random.sample(available_models, len(available_models))
    last_e = None
    for model in shuffled_models:
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
                    max_retries=1,
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
                    max_retries=1,
                )

            response_text = result["content"]
            usage = result.get("usage", {})

            if not response_text or not response_text.strip():
                raise RuntimeError("LLM returned an empty response")
            
            _record_model_success(redis_client, model)
            return response_text, usage, provider, model, is_fallback
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            _save_metric({
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
            logger.error("LLM primary call failed (provider=%s, model=%s, model_type=%s): %s", provider, model, model_type, e, exc_info=True)
            _record_model_failure(redis_client, model, provider, str(e))
            last_e = e
    if last_e:
        raise last_e
    raise RuntimeError("No primary models configured")


def _execute_fallback_call(
    e: Exception,
    model_type: str,
    messages: Optional[List[Dict[str, str]]],
    prompt: str,
    system_prompt: str,
    temperature: Optional[float],
    effective_timeout: float,
    api_messages: Optional[List[Dict[str, str]]],
    add_cache_control: bool,
    thinking_enabled: bool,
    request_type: Optional[str],
) -> Tuple[str, dict, str, str, bool]:
    """Execute fallback LLM call if primary fails."""
    if not settings.LLM_FALLBACK_ENABLED:
        logger.warning(
            "LLM primary call failed and fallback is disabled (LLM_FALLBACK_ENABLED=False). "
            "Original error: %s", e
        )
        raise

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

    # Re-evaluate cache control for the fallback provider
    fallback_add_cache_control = (
        settings.LLM_PROMPT_CACHING_ENABLED
        and fallback_provider in settings.LLM_PROMPT_CACHING_CONTROL_PROVIDERS
        and messages is not None
    )

    if fallback_provider == "openai":
        if model_type == "mind":
            fallback_models = settings.OPENAI_MIND_FALLBACK_MODEL or settings.OPENAI_FALLBACK_MODEL
            fallback_base_url = settings.OPENAI_MIND_FALLBACK_BASE_URL or settings.OPENAI_FALLBACK_BASE_URL or settings.OPENAI_BASE_URL
            fallback_api_key = settings.OPENAI_MIND_FALLBACK_API_KEY or settings.OPENAI_FALLBACK_API_KEY or settings.OPENAI_API_KEY
        elif model_type == "weak":
            fallback_models = settings.OPENAI_WEAK_FALLBACK_MODEL or settings.OPENAI_FALLBACK_MODEL
            fallback_base_url = settings.OPENAI_WEAK_FALLBACK_BASE_URL or settings.OPENAI_FALLBACK_BASE_URL or settings.OPENAI_BASE_URL
            fallback_api_key = settings.OPENAI_WEAK_FALLBACK_API_KEY or settings.OPENAI_FALLBACK_API_KEY or settings.OPENAI_API_KEY
        else:
            fallback_models = settings.OPENAI_ACTUATOR_FALLBACK_MODEL or settings.OPENAI_FALLBACK_MODEL
            fallback_base_url = settings.OPENAI_ACTUATOR_FALLBACK_BASE_URL or settings.OPENAI_FALLBACK_BASE_URL or settings.OPENAI_BASE_URL
            fallback_api_key = settings.OPENAI_ACTUATOR_FALLBACK_API_KEY or settings.OPENAI_FALLBACK_API_KEY or settings.OPENAI_API_KEY

        if not fallback_models:
            logger.warning(
                "Fallback provider is openai but no fallback models configured. "
                "Original error: %s", e
            )
            raise

        # Context window management for fallback (done once)
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
                        model=fallback_models[0],
                        base_url=fallback_base_url,
                        api_key=fallback_api_key,
                        temperature=temperature,
                        timeout=effective_timeout,
                        max_input_tokens=fb_effective_limit - estimate_tokens(system_prompt),
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
                    model=fallback_models[0],
                    base_url=fallback_base_url,
                    api_key=fallback_api_key,
                    temperature=temperature,
                    timeout=effective_timeout,
                    max_input_tokens=fb_effective_limit - estimate_tokens(system_prompt),
                )

        if fallback_api_key or fallback_base_url:
            # Shuffle fallback models to try them in random order without repeating
            redis_client = get_redis_client()
            available_fallback_models = [m for m in fallback_models if not _is_model_blacklisted(redis_client, m)]
            
            if not available_fallback_models:
                logger.warning("All fallback models are blacklisted or unavailable.")
                raise RuntimeError("All fallback models are blacklisted") from e
                
            shuffled_fallback_models = random.sample(available_fallback_models, len(available_fallback_models))
            last_fallback_e = None
            for fallback_model in shuffled_fallback_models:
                logger.warning(
                    "Primary LLM call failed (%s). Falling back to OpenAI-compatible provider "
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
                        timeout=settings.LLM_FALLBACK_TIMEOUT,
                        messages=api_messages,
                        add_cache_control=fallback_add_cache_control,
                        thinking_enabled=thinking_enabled,
                        max_retries=1,
                    )
                    response_text = result["content"]
                    usage = result.get("usage", {})
                    _record_model_success(redis_client, fallback_model)
                    return response_text, usage, "openai", fallback_model, True
                except Exception as fallback_e:
                    fallback_latency = (time.time() - fallback_start) * 1000
                    _save_metric({
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
                    logger.error("OpenAI fallback model %s failed: %s", fallback_model, fallback_e, exc_info=True)
                    _record_model_failure(redis_client, fallback_model, "openai", str(fallback_e))
                    last_fallback_e = fallback_e
            if last_fallback_e:
                raise last_fallback_e
            raise
        else:
            logger.warning(
                "Fallback provider is openai but no API key or base URL configured. "
                "Original error: %s", e
            )
            raise
    elif fallback_provider == "ollama":
        if model_type == "mind":
            fallback_models = settings.OLLAMA_MIND_FALLBACK_MODEL or settings.OLLAMA_FALLBACK_MODEL
            fallback_base_url = settings.OLLAMA_MIND_FALLBACK_BASE_URL or settings.OLLAMA_FALLBACK_BASE_URL or settings.OLLAMA_BASE_URL
            fallback_api_key = settings.OLLAMA_MIND_FALLBACK_API_KEY or settings.OLLAMA_FALLBACK_API_KEY or settings.OLLAMA_API_KEY
        elif model_type == "weak":
            fallback_models = settings.OLLAMA_WEAK_FALLBACK_MODEL or settings.OLLAMA_FALLBACK_MODEL
            fallback_base_url = settings.OLLAMA_WEAK_FALLBACK_BASE_URL or settings.OLLAMA_FALLBACK_BASE_URL or settings.OLLAMA_BASE_URL
            fallback_api_key = settings.OLLAMA_WEAK_FALLBACK_API_KEY or settings.OLLAMA_FALLBACK_API_KEY or settings.OLLAMA_API_KEY
        else:
            fallback_models = settings.OLLAMA_ACTUATOR_FALLBACK_MODEL or settings.OLLAMA_FALLBACK_MODEL
            fallback_base_url = settings.OLLAMA_ACTUATOR_FALLBACK_BASE_URL or settings.OLLAMA_FALLBACK_BASE_URL or settings.OLLAMA_BASE_URL
            fallback_api_key = settings.OLLAMA_ACTUATOR_FALLBACK_API_KEY or settings.OLLAMA_FALLBACK_API_KEY or settings.OLLAMA_API_KEY

        if not fallback_models:
            logger.warning(
                "Fallback provider is ollama but no fallback models configured. "
                "Original error: %s", e
            )
            raise

        # Context window management for fallback (done once)
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
                        model=fallback_models[0],
                        base_url=fallback_base_url,
                        api_key=fallback_api_key,
                        temperature=temperature,
                        timeout=effective_timeout,
                        max_input_tokens=fb_effective_limit - estimate_tokens(system_prompt),
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
                    model=fallback_models[0],
                    base_url=fallback_base_url,
                    api_key=fallback_api_key,
                    temperature=temperature,
                    timeout=effective_timeout,
                    max_input_tokens=fb_effective_limit - estimate_tokens(system_prompt),
                )

        if fallback_base_url:
            # Shuffle fallback models to try them in random order without repeating
            redis_client = get_redis_client()
            available_fallback_models = [m for m in fallback_models if not _is_model_blacklisted(redis_client, m)]
            
            if not available_fallback_models:
                logger.warning("All fallback models are blacklisted or unavailable.")
                raise RuntimeError("All fallback models are blacklisted") from e
                
            shuffled_fallback_models = random.sample(available_fallback_models, len(available_fallback_models))
            last_fallback_e = None
            for fallback_model in shuffled_fallback_models:
                logger.warning(
                    "Primary LLM call failed (%s). Falling back to Ollama provider "
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
                        timeout=settings.LLM_FALLBACK_TIMEOUT,
                        messages=api_messages,
                        add_cache_control=fallback_add_cache_control,
                        thinking_enabled=thinking_enabled,
                        max_retries=1,
                    )
                    response_text = result["content"]
                    usage = result.get("usage", {})
                    _record_model_success(redis_client, fallback_model)
                    return response_text, usage, "ollama", fallback_model, True
                except Exception as fallback_e:
                    fallback_latency = (time.time() - fallback_start) * 1000
                    _save_metric({
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
                    logger.error("Ollama fallback model %s failed: %s", fallback_model, fallback_e, exc_info=True)
                    _record_model_failure(redis_client, fallback_model, "ollama", str(fallback_e))
                    last_fallback_e = fallback_e
            if last_fallback_e:
                raise last_fallback_e
            raise
        else:
            logger.warning(
                "Fallback provider is ollama but no base URL configured. "
                "Original error: %s", e
            )
            raise
    else:
        raise


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
            models = settings.OPENAI_MIND_MODEL or settings.OPENAI_MODEL
            base_url = settings.OPENAI_MIND_BASE_URL or settings.OPENAI_BASE_URL
            api_key = settings.OPENAI_MIND_API_KEY or settings.OPENAI_API_KEY
        elif model_type == "weak":
            models = settings.OPENAI_WEAK_MODEL or settings.OPENAI_MODEL
            base_url = settings.OPENAI_WEAK_BASE_URL or settings.OPENAI_BASE_URL
            api_key = settings.OPENAI_WEAK_API_KEY or settings.OPENAI_API_KEY
        else:
            models = settings.OPENAI_ACTUATOR_MODEL or settings.OPENAI_MODEL
            base_url = settings.OPENAI_ACTUATOR_BASE_URL or settings.OPENAI_BASE_URL
            api_key = settings.OPENAI_ACTUATOR_API_KEY or settings.OPENAI_API_KEY
    else:  # ollama
        if model_type == "mind":
            models = settings.OLLAMA_MIND_MODEL or settings.OLLAMA_MODEL
            base_url = settings.OLLAMA_MIND_BASE_URL or settings.OLLAMA_BASE_URL
            api_key = settings.OLLAMA_MIND_API_KEY or settings.OLLAMA_API_KEY
        elif model_type == "weak":
            models = settings.OLLAMA_WEAK_MODEL or settings.OLLAMA_MODEL
            base_url = settings.OLLAMA_WEAK_BASE_URL or settings.OLLAMA_BASE_URL
            api_key = settings.OLLAMA_WEAK_API_KEY or settings.OLLAMA_API_KEY
        else:
            models = settings.OLLAMA_ACTUATOR_MODEL or settings.OLLAMA_MODEL
            base_url = settings.OLLAMA_ACTUATOR_BASE_URL or settings.OLLAMA_BASE_URL
            api_key = settings.OLLAMA_ACTUATOR_API_KEY or settings.OLLAMA_API_KEY

    if not models:
        logger.error("No LLM models configured for provider=%s, model_type=%s", provider, model_type)
        raise RuntimeError(f"No LLM models configured for provider={provider}, model_type={model_type}")

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
            models = [fb_model]
            base_url = fb_base_url
            api_key = fb_api_key
        else:
            logger.warning("Market is closed but no fallback model is configured for model_type=%s. Using primary model to avoid downtime.", model_type)

    cache_key_model = models[0] if models else "unknown"
    cache_key = _build_cache_key(messages, system_prompt, model_type, provider, cache_key_model, cache_temp, market_hash, prompt)

    cached_data = _get_cached_response(redis_client, cache_key, model_type, provider, cache_key_model, request_type)
    if cached_data:
        return cached_data

    logger.debug("LLM cache miss: model_type=%s, system_prompt=%.200s..., prompt=%.500s...", model_type, system_prompt, prompt)
    
    max_input_tokens = _get_max_input_tokens(provider, model_type, is_fallback)
    effective_limit = int(max_input_tokens * 0.8)

    messages, prompt = _manage_context_window(
        messages, prompt, system_prompt, model_type, provider, cache_key_model, base_url, api_key, temperature, effective_timeout, max_input_tokens, effective_limit
    )

    if messages is not None:
        api_messages = []
        if system_prompt:
            api_messages.append({"role": "system", "content": system_prompt})
        # Use the provided message dicts directly (they already have role/content)
        api_messages.extend(messages)
    else:
        api_messages = None  # will be built inside _get_*_response from prompt/system_prompt

    start_time = time.time()
    try:
        response_text, usage, used_provider, used_model, is_fallback = _execute_primary_call(
            provider, models, base_url, api_key, temperature, effective_timeout, messages, api_messages, prompt, system_prompt, add_cache_control, thinking_enabled, model_type, request_type, is_fallback
        )
    except Exception as e:
        response_text, usage, used_provider, used_model, is_fallback = _execute_fallback_call(
            e, model_type, messages, prompt, system_prompt, temperature, effective_timeout, api_messages, add_cache_control, thinking_enabled, request_type
        )
    if response_text is None:
        logger.warning("LLM returned None response; not caching.")
        _save_metric({
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
        return None

    latency_ms = (time.time() - start_time) * 1000
    _save_metric({
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

    logger.debug("LLM response cached: %.500s...", response_text)
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
    
    - Rounds floats to 6 significant figures (percentage-based rounding) to treat small absolute changes on high-priced assets as insignificant while preserving precision for low-priced assets.
    - Excludes keys containing 'timestamp', 'time', 'fetched_at', 'created_at',
      'published_at', 'last_eval', 'last_auto_resume' (volatile fields that
      change every cycle but don't affect trading decisions).
    - Converts None values to a string "null" for consistent serialization.
    """
    _VOLATILE_KEY_FRAGMENTS = ("timestamp", "fetched_at", "created_at",
                                "published_at", "last_eval", "last_auto_resume",
                                "_last_state_save", "datetime")
    if depth > 10:
        return None
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            key_str = str(k).lower()
            if any(frag in key_str for frag in _VOLATILE_KEY_FRAGMENTS):
                continue
            if key_str == "time" or key_str.endswith("_time") or key_str.startswith("time_"):
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
        # Percentage-based rounding: round to 6 significant figures
        # to treat tiny floating-point noise as insignificant while
        # preserving meaningful price changes (e.g., 100.12 vs 100.14).
        decimals = 5 - int(math.floor(math.log10(abs(obj))))
        return round(obj, decimals)
    if obj is None:
        return "null"
    return obj


def _strip_ohlcv_timestamps(obj):
    """Recursively remove timestamp values from OHLCV data for stable hashing.

    OHLCV candle lists are often a list of [timestamp, open, high, low, close, volume]
    or a list of dicts with a 'timestamp' key.  This function strips the timestamp
    from each candle so the hash changes only when price/volume data changes,
    not when the same candle is fetched at a different time.
    """
    _OHLCV_KEY_FRAGMENTS = ("ohlcv_data", "raw_candles", "candles")

    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            key_str = str(k).lower()
            if any(frag in key_str for frag in _OHLCV_KEY_FRAGMENTS):
                result[k] = _strip_timestamps_from_candle_list(v)
            else:
                result[k] = _strip_ohlcv_timestamps(v)
        return result
    if isinstance(obj, list):
        return [_strip_ohlcv_timestamps(item) for item in obj]
    return obj


def _strip_timestamps_from_candle_list(data):
    """Remove timestamps from OHLCV data, handling dicts of timeframes."""
    if isinstance(data, dict):
        # Handle dict of timeframes, e.g., {"1d": [[ts, o, h, l, c, v], ...]}
        return {k: _strip_timestamps_from_candle_list(v) for k, v in data.items()}
    if not isinstance(data, list):
        return data
    
    # Check if it's a list of candles (list of lists) or a list of dicts
    if data and isinstance(data[0], (list, tuple, dict)):
        result = []
        for candle in data:
            if isinstance(candle, (list, tuple)):
                # Assume first element is timestamp; keep the rest
                result.append(list(candle[1:]))
            elif isinstance(candle, dict):
                result.append({k: v for k, v in candle.items() if "time" not in str(k).lower()})
            else:
                result.append(candle)
        return result
    return data


def compute_market_hash(data: dict) -> str:
    """Return a SHA-256 hex digest of the JSON-serialised market data.
    
    Volatile fields (timestamps, etc.) are excluded and floats are rounded
    so that essentially-identical market states produce the same hash,
    enabling LLM response caching.
    """
    data = _strip_ohlcv_timestamps(data)
    normalized = _normalize_for_hash(data)
    safe_data = _stringify_keys(normalized)
    serialized = json.dumps(safe_data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()


def _is_italian_holiday(dt) -> bool:
    """Check if a given date is an Italian public holiday."""
    # Fixed-date holidays
    fixed_holidays = {
        (1, 1), (1, 6), (4, 25), (5, 1), (6, 2),
        (8, 15), (11, 1), (12, 8), (12, 25), (12, 26)
    }
    if (dt.month, dt.day) in fixed_holidays:
        return True

    # Easter Monday (depends on Easter Sunday)
    # Simple Computus algorithm (Meeus/Jones/Butcher)
    y = dt.year
    a = y % 19
    b = y // 100
    c = y % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    
    # Easter Monday is the day after Easter Sunday
    easter_monday = datetime(y, month, day) + timedelta(days=1)
    if dt.month == easter_monday.month and dt.day == easter_monday.day:
        return True

    return False


def _should_use_primary_model() -> bool:
    """Check if primary models should be used based on market status.

    Returns True if market is open or in pre-market session (within 60 mins of open).
    Returns False if market is closed (use fallback models only to save tokens).
    Computes market status locally to avoid dependency on Redis background tasks.
    """
    from zoneinfo import ZoneInfo
    try:
        now_rome = datetime.now(timezone.utc).astimezone(ZoneInfo(settings.MARKET_TIMEZONE))
        weekday = now_rome.weekday()
        if weekday >= 5:  # Saturday or Sunday
            return False

        if _is_italian_holiday(now_rome):
            return False

        rome_minutes = now_rome.hour * 60 + now_rome.minute
        open_minutes = settings.MARKET_OPEN_HOUR * 60 + settings.MARKET_OPEN_MINUTE
        close_minutes = settings.MARKET_CLOSE_HOUR * 60 + settings.MARKET_CLOSE_MINUTE

        if open_minutes <= rome_minutes < close_minutes:
            return True  # Market is open

        # Check pre-market (within 60 mins of open)
        if open_minutes - 60 <= rome_minutes < open_minutes:
            return True  # Pre-market

        return False  # Market is closed
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
            fb_models = settings.OPENAI_MIND_FALLBACK_MODEL or settings.OPENAI_FALLBACK_MODEL
            fb_base_url = settings.OPENAI_MIND_FALLBACK_BASE_URL or settings.OPENAI_FALLBACK_BASE_URL or settings.OPENAI_BASE_URL
            fb_api_key = settings.OPENAI_MIND_FALLBACK_API_KEY or settings.OPENAI_FALLBACK_API_KEY or settings.OPENAI_API_KEY
        elif model_type == "weak":
            fb_models = settings.OPENAI_WEAK_FALLBACK_MODEL or settings.OPENAI_FALLBACK_MODEL
            fb_base_url = settings.OPENAI_WEAK_FALLBACK_BASE_URL or settings.OPENAI_FALLBACK_BASE_URL or settings.OPENAI_BASE_URL
            fb_api_key = settings.OPENAI_WEAK_FALLBACK_API_KEY or settings.OPENAI_FALLBACK_API_KEY or settings.OPENAI_API_KEY
        else:
            fb_models = settings.OPENAI_ACTUATOR_FALLBACK_MODEL or settings.OPENAI_FALLBACK_MODEL
            fb_base_url = settings.OPENAI_ACTUATOR_FALLBACK_BASE_URL or settings.OPENAI_FALLBACK_BASE_URL or settings.OPENAI_BASE_URL
            fb_api_key = settings.OPENAI_ACTUATOR_FALLBACK_API_KEY or settings.OPENAI_FALLBACK_API_KEY or settings.OPENAI_API_KEY
    else:  # ollama
        if model_type == "mind":
            fb_models = settings.OLLAMA_MIND_FALLBACK_MODEL or settings.OLLAMA_FALLBACK_MODEL
            fb_base_url = settings.OLLAMA_MIND_FALLBACK_BASE_URL or settings.OLLAMA_FALLBACK_BASE_URL or settings.OLLAMA_BASE_URL
            fb_api_key = settings.OLLAMA_MIND_FALLBACK_API_KEY or settings.OLLAMA_FALLBACK_API_KEY or settings.OLLAMA_API_KEY
        elif model_type == "weak":
            fb_models = settings.OLLAMA_WEAK_FALLBACK_MODEL or settings.OLLAMA_FALLBACK_MODEL
            fb_base_url = settings.OLLAMA_WEAK_FALLBACK_BASE_URL or settings.OLLAMA_FALLBACK_BASE_URL or settings.OLLAMA_BASE_URL
            fb_api_key = settings.OLLAMA_WEAK_FALLBACK_API_KEY or settings.OLLAMA_FALLBACK_API_KEY or settings.OLLAMA_API_KEY
        else:
            fb_models = settings.OLLAMA_ACTUATOR_FALLBACK_MODEL or settings.OLLAMA_FALLBACK_MODEL
            fb_base_url = settings.OLLAMA_ACTUATOR_FALLBACK_BASE_URL or settings.OLLAMA_FALLBACK_BASE_URL or settings.OLLAMA_BASE_URL
            fb_api_key = settings.OLLAMA_ACTUATOR_FALLBACK_API_KEY or settings.OLLAMA_FALLBACK_API_KEY or settings.OLLAMA_API_KEY

    if not fb_models:
        return (None, None, None, None)

    fb_model = random.choice(fb_models)
    return (fallback_provider, fb_model, fb_base_url, fb_api_key)


async def get_cached_llm_response_async(
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
    Asynchronous wrapper for get_cached_llm_response.
    Runs the blocking LLM call in a dedicated thread pool to avoid blocking
    the event loop and exhausting the default executor.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _llm_executor,
        get_cached_llm_response,
        prompt,
        system_prompt,
        ttl,
        market_hash,
        model_type,
        temperature,
        symbol,
        messages,
        request_type,
    )
