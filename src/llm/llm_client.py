import asyncio
import json
import logging
import time
from typing import Optional, List, Dict, Callable

import httpx

from src.config.settings import settings

logger = logging.getLogger(__name__)


def _execute_llm_request(
    provider: str,
    url: str,
    headers: dict,
    payload: dict,
    timeout: Optional[float],
    system_prompt: str,
    prompt: str,
    parse_response_fn: Callable[[dict], dict],
    max_retries: int = 3,
) -> dict:
    """Executes the LLM HTTP request with retries and standard error handling."""
    logger.info("LLM request (%s): model=%s, system_prompt=%.200s..., prompt=%.500s...", provider, payload.get("model"), system_prompt, prompt)
    for attempt in range(max_retries):
        try:
            httpx_timeout = httpx.Timeout(
                connect=10.0,
                read=timeout if timeout is not None else settings.LLM_TIMEOUT,
                write=10.0,
                pool=5.0,
            )
            def _do_request():
                with httpx.Client(timeout=httpx_timeout) as client:
                    response = client.post(url, json=payload, headers=headers)
                    response.raise_for_status()
                    return response.json()

            data = _do_request()
            
            result = parse_response_fn(data)
            logger.info("LLM response (%s): %.500s...", provider, result["content"])
            return result
        except httpx.HTTPStatusError as e:
            if e.response.status_code not in (429, 500, 502, 503, 504):
                logger.error(
                    "%s request failed with HTTP %d (non-retryable). "
                    "URL: %s\nResponse body: %s\nRequest payload model: %s, messages count: %d",
                    provider.capitalize(),
                    e.response.status_code,
                    str(e.request.url),
                    e.response.text[:2000],
                    payload.get("model"),
                    len(payload.get("messages", [])),
                    exc_info=True,
                )
                raise RuntimeError(f"{provider.capitalize()} request failed with HTTP {e.response.status_code}: {e.response.text[:500]}") from e
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                if e.response.status_code == 429:
                    retry_after = e.response.headers.get("Retry-After")
                    if retry_after:
                        try:
                            # Retry-After can be seconds or an HTTP date. 
                            # We only handle the seconds case for simplicity, capping at 60s.
                            wait_time = min(int(retry_after), 60)
                        except ValueError:
                            # It's an HTTP date or invalid format; fall back to exponential backoff
                            pass
                logger.warning(f"{provider.capitalize()} request failed with HTTP {e.response.status_code}. Response: {e.response.text[:500]}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
            raise RuntimeError(f"{provider.capitalize()} request failed: {e.response.status_code} - {e.response.text[:500]}") from e
        except httpx.HTTPError as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.warning(f"{provider.capitalize()} request failed with network error: {e}. Retrying in {wait_time}s...", exc_info=True)
                time.sleep(wait_time)
                continue
            logger.error("%s request failed with network error after all retries: %s", provider.capitalize(), e, exc_info=True)
            raise RuntimeError(f"{provider.capitalize()} request failed: {e}") from e
    raise RuntimeError(f"{provider.capitalize()} request failed after all retries")


def _get_ollama_response(prompt: str = "", system_prompt: str = "", model: str = None,
                         base_url: str = None, api_key: str = None,
                         temperature: Optional[float] = None,
                         timeout: Optional[float] = None,
                        messages: Optional[List[Dict[str, str]]] = None,
                        add_cache_control: bool = False,
                        thinking_enabled: bool = True,
                        reasoning_effort: str = "low",
                       max_retries: int = 3,
) -> dict:
    """Send a prompt to the configured Ollama model and return a dict with 'content' and 'usage'."""
    url = f"{(base_url or settings.OLLAMA_BASE_URL).rstrip('/')}/api/chat"
    headers = {"Content-Type": "application/json"}
    effective_api_key = api_key or settings.OLLAMA_API_KEY
    if effective_api_key:
        headers["Authorization"] = f"Bearer {effective_api_key}"

    if messages is not None:
        # Use the provided message list directly (copy to avoid mutation)
        api_messages = [dict(msg) for msg in messages]
    else:
        api_messages = []
        if system_prompt:
            api_messages.append({"role": "system", "content": system_prompt})
        api_messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model or (settings.OLLAMA_MODEL[0] if settings.OLLAMA_MODEL else None),
        "messages": api_messages,
        "stream": False,
        "temperature": temperature if temperature is not None else settings.LLM_TEMPERATURE,
    }

    # Always send reasoning_effort: "low" when thinking is disabled,
    # or the computed value when thinking is enabled.
    payload["reasoning_effort"] = "low" if not thinking_enabled else reasoning_effort

    def _parse_ollama(data: dict) -> dict:
        if "message" not in data or "content" not in data["message"]:
            logger.error(
                "Ollama response missing 'message.content' key. Full response: %s",
                json.dumps(data)[:2000]
            )
            raise RuntimeError(f"Ollama API returned unexpected format: missing 'message.content'. Response: {str(data)[:500]}")
        
        content = data["message"]["content"]
        prompt_eval_count = data.get("prompt_eval_count")
        eval_count = data.get("eval_count")
        if prompt_eval_count is not None and eval_count is not None:
            prompt_tokens = prompt_eval_count
            completion_tokens = eval_count
        else:
            from src.llm.cache import estimate_tokens
            prompt_tokens = estimate_tokens(prompt) + estimate_tokens(system_prompt)
            completion_tokens = estimate_tokens(content)
        total_tokens = prompt_tokens + completion_tokens
        
        return {
            "content": content,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }
        }

    return _execute_llm_request("ollama", url, headers, payload, timeout, system_prompt, prompt, _parse_ollama, max_retries=max_retries)


def _get_openai_response(prompt: str = "", system_prompt: str = "", model: str = None,
                         base_url: str = None, api_key: str = None,
                         temperature: Optional[float] = None,
                         timeout: Optional[float] = None,
                        messages: Optional[List[Dict[str, str]]] = None,
                        add_cache_control: bool = False,
                        thinking_enabled: bool = True,
                        reasoning_effort: str = "low",
                      max_retries: int = 3,
) -> dict:
    """Send a prompt to the configured OpenAI-compatible API and return a dict with 'content' and 'usage'."""
    url = f"{(base_url or settings.OPENAI_BASE_URL).rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    effective_api_key = api_key or settings.OPENAI_API_KEY
    if effective_api_key:
        headers["Authorization"] = f"Bearer {effective_api_key}"

    if messages is not None:
        # Use the provided message list directly (copy dicts so add_cache_control
        # doesn't mutate the caller's message objects, important for fallback reuse)
        api_messages = [dict(msg) for msg in messages]
    else:
        api_messages = []
        if system_prompt:
            api_messages.append({"role": "system", "content": system_prompt})
        api_messages.append({"role": "user", "content": prompt})

    # Add cache_control to system message and first user message when supported
    if add_cache_control:
        for msg in api_messages:
            if msg["role"] == "system":
                msg["cache_control"] = {"type": "ephemeral"}
                break
        for msg in api_messages:
            if msg["role"] == "user":
                msg["cache_control"] = {"type": "ephemeral"}
                break

    payload = {
        "model": model or (settings.OPENAI_MODEL[0] if settings.OPENAI_MODEL else None),
        "messages": api_messages,
        "temperature": temperature if temperature is not None else settings.LLM_TEMPERATURE,
    }

    # Always send reasoning_effort: "low" when thinking is disabled,
    # or the computed value when thinking is enabled.
    payload["reasoning_effort"] = "low" if not thinking_enabled else reasoning_effort

    def _parse_openai(data: dict) -> dict:
        if "choices" not in data or not data["choices"]:
            logger.error(
                "OpenAI response missing 'choices' key. Full response: %s",
                json.dumps(data)[:2000]
            )
            raise RuntimeError(f"OpenAI API returned unexpected format: missing 'choices'. Response: {str(data)[:500]}")
            
        content = data["choices"][0]["message"]["content"]
        usage_data = data.get("usage", {})
        prompt_tokens = usage_data.get("prompt_tokens", 0)
        completion_tokens = usage_data.get("completion_tokens", 0)
        total_tokens = usage_data.get("total_tokens", 0)
        
        return {
            "content": content,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }
        }

    return _execute_llm_request("openai", url, headers, payload, timeout, system_prompt, prompt, _parse_openai, max_retries=max_retries)


def get_llm_response(prompt: str, system_prompt: str = "", model_type: str = "actuator", symbol: Optional[str] = None, messages: Optional[List[Dict[str, str]]] = None, request_type: Optional[str] = None, force_primary_model: bool = False) -> str:
    """Send a prompt to the configured LLM provider and return the response text.

    Uses Redis caching with a 5-minute TTL (keyed by prompt + system prompt).
    model_type: "mind" for complex reasoning, "actuator" for fast time‑critical decisions.
    symbol: Optional symbol/ticker for semantic cache generalization.
    messages: Optional list of message dicts (role/content) for multi-turn conversations.
    When provided, the cache key is based on the messages list instead of the prompt.

    Note: get_cached_llm_response now returns a dict with "response", "provider", "model".
    This function returns only the response text for backward compatibility.
    """
    from src.llm.cache import get_cached_llm_response  # local import to avoid circular dependency at module level

    result = get_cached_llm_response(prompt, system_prompt, ttl=300, model_type=model_type, symbol=symbol, messages=messages, request_type=request_type, force_primary_model=force_primary_model)
    if result is None:
        # This should not happen because the underlying raw call raises on failure,
        # but guard against unexpected None.
        raise RuntimeError("LLM returned an empty response")
    return result["response"]


def check_llm_health() -> dict:
    """Check if the configured LLM provider is reachable.

    Returns a dict with keys:
        - "mind": {"status": "connected"|"disconnected", "provider": str, "model": str, "error": str|None}
        - "actuator": {"status": "connected"|"disconnected", "provider": str, "model": str, "error": str|None}
    """
    results = {}

    for role in ("mind", "actuator", "weak", "aol"):
        if role == "aol":
            provider = settings.AOL_LLM_PROVIDER
            models = settings.AOL_LLM_MODEL
            base_url = settings.AOL_BASE_URL
            api_key = settings.AOL_LLM_API_KEY
            if not provider or not models:
                results[role] = {
                    "status": "disconnected",
                    "provider": provider or "none",
                    "model": "not configured",
                    "error": "AOL provider not configured",
                }
                continue
        else:
            if role == "mind":
                provider = settings.LLM_MIND_PROVIDER or settings.LLM_PROVIDER
            elif role == "weak":
                provider = settings.LLM_WEAK_PROVIDER or settings.LLM_PROVIDER
            else:
                provider = settings.LLM_ACTUATOR_PROVIDER or settings.LLM_PROVIDER

            if provider == "openai":
                if role == "mind":
                    models = settings.OPENAI_MIND_MODEL or settings.OPENAI_MODEL
                    base_url = settings.OPENAI_MIND_BASE_URL or settings.OPENAI_BASE_URL
                    api_key = settings.OPENAI_MIND_API_KEY or settings.OPENAI_API_KEY
                elif role == "weak":
                    models = settings.OPENAI_WEAK_MODEL or settings.OPENAI_MODEL
                    base_url = settings.OPENAI_WEAK_BASE_URL or settings.OPENAI_BASE_URL
                    api_key = settings.OPENAI_WEAK_API_KEY or settings.OPENAI_API_KEY
                else:
                    models = settings.OPENAI_ACTUATOR_MODEL or settings.OPENAI_MODEL
                    base_url = settings.OPENAI_ACTUATOR_BASE_URL or settings.OPENAI_BASE_URL
                    api_key = settings.OPENAI_ACTUATOR_API_KEY or settings.OPENAI_API_KEY
            elif provider == "g4f":
                from src.llm.g4f_client import _get_g4f_models
                models = _get_g4f_models(role)
                base_url = None
                api_key = None
            else:
                if role == "mind":
                    models = settings.OLLAMA_MIND_MODEL or settings.OLLAMA_MODEL
                    base_url = settings.OLLAMA_MIND_BASE_URL or settings.OLLAMA_BASE_URL
                    api_key = settings.OLLAMA_MIND_API_KEY or settings.OLLAMA_API_KEY
                elif role == "weak":
                    models = settings.OLLAMA_WEAK_MODEL or settings.OLLAMA_MODEL
                    base_url = settings.OLLAMA_WEAK_BASE_URL or settings.OLLAMA_BASE_URL
                    api_key = settings.OLLAMA_WEAK_API_KEY or settings.OLLAMA_API_KEY
                else:
                    models = settings.OLLAMA_ACTUATOR_MODEL or settings.OLLAMA_MODEL
                    base_url = settings.OLLAMA_ACTUATOR_BASE_URL or settings.OLLAMA_BASE_URL
                    api_key = settings.OLLAMA_ACTUATOR_API_KEY or settings.OLLAMA_API_KEY

        model = models[0] if models else "unknown"

        if not base_url:
            if provider == "g4f":
                results[role] = {
                    "status": "connected",
                    "provider": provider,
                    "model": models[0] if models else "dynamic",
                    "error": None,
                }
            else:
                results[role] = {
                    "status": "disconnected",
                    "provider": provider,
                    "model": model or "unknown",
                    "error": "No base URL configured",
                }
            continue

        try:
            if provider == "g4f":
                # g4f has no base URL to check, assume connected if configured
                results[role] = {
                    "status": "connected",
                    "provider": provider,
                    "model": models[0] if models else "dynamic",
                    "error": None,
                }
                continue

            headers = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            if provider == "ollama":
                url = f"{base_url.rstrip('/')}/api/tags"
            else:
                url = f"{base_url.rstrip('/')}/models"

            def _do_health_check():
                with httpx.Client(timeout=10.0) as client:
                    response = client.get(url, headers=headers)
                    response.raise_for_status()

            _do_health_check()

            results[role] = {
                "status": "connected",
                "provider": provider,
                "model": model or "unknown",
                "error": None,
            }
        except (httpx.HTTPError, ConnectionError, TimeoutError, OSError, ValueError, TypeError) as e:
            results[role] = {
                "status": "disconnected",
                "provider": provider,
                "model": model or "unknown",
                "error": str(e)[:200],
            }

    return results


async def get_llm_response_async(prompt: str, system_prompt: str = "", model_type: str = "actuator", symbol: Optional[str] = None, messages: Optional[List[Dict[str, str]]] = None, request_type: Optional[str] = None) -> str:
    """
    Asynchronous wrapper for get_llm_response.
    Runs the blocking LLM call in a dedicated thread pool to avoid blocking
    the event loop and exhausting the default executor.
    """
    from src.llm.cache import _llm_executor
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _llm_executor,
        get_llm_response,
        prompt,
        system_prompt,
        model_type,
        symbol,
        messages,
        request_type,
    )
