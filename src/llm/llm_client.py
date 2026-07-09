import json
import logging
import time
from typing import Optional, List, Dict

import httpx

from src.config.settings import settings

logger = logging.getLogger(__name__)


def _get_ollama_response(prompt: str = "", system_prompt: str = "", model: str = None,
                         base_url: str = None, api_key: str = None,
                         temperature: Optional[float] = None,
                         timeout: Optional[float] = None,
                        messages: Optional[List[Dict[str, str]]] = None,
                        add_cache_control: bool = False,
) -> str:
    """Send a prompt to the configured Ollama model and return the response text."""
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
        "model": model or settings.OLLAMA_MODEL,
        "messages": api_messages,
        "stream": False,
        "temperature": temperature if temperature is not None else settings.LLM_TEMPERATURE,
    }

    logger.info("LLM request (ollama): model=%s, system_prompt=%.200s..., prompt=%.500s...", model, system_prompt, prompt)
    max_retries = 3
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
                                                                                                                                                                                                                                                                                   
            # Validate response structure                                                                                                                                                                                                                                          
            if "message" not in data or "content" not in data["message"]:                                                                                                                                                                                                          
                logger.error(                                                                                                                                                                                                                                                      
                    "Ollama response missing 'message.content' key. Full response: %s",                                                                                                                                                                                            
                    json.dumps(data)[:2000]                                                                                                                                                                                                                                        
                )                                                                                                                                                                                                                                                                  
                raise RuntimeError(f"Ollama API returned unexpected format: missing 'message.content'. Response: {str(data)[:500]}")                                                                                                                                               
                                                                                                                                                                                                                                                                                   
            content = data["message"]["content"]                                                                                                                                                                                                                                   
            logger.info("LLM response (ollama): %.500s...", content)                                                                                                                                                                                                               
            return content
        except httpx.HTTPStatusError as e:
            if e.response.status_code not in (429, 500, 502, 503, 504):
                logger.error(
                    "Ollama request failed with HTTP %d (non-retryable). "
                    "URL: %s\nResponse body: %s\nRequest payload model: %s, messages count: %d",
                    e.response.status_code,
                    str(e.request.url),
                    e.response.text[:2000],
                    payload.get("model"),
                    len(payload.get("messages", [])),
                    exc_info=True,
                )
                raise RuntimeError(f"Ollama request failed with HTTP {e.response.status_code}: {e.response.text[:500]}") from e
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.warning(f"Ollama request failed with HTTP {e.response.status_code}. Response: {e.response.text[:500]}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
            raise RuntimeError(f"Ollama request failed: {e.response.status_code} - {e.response.text[:500]}") from e
        except httpx.HTTPError as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.warning(f"Ollama request failed with network error: {e}. Retrying in {wait_time}s...", exc_info=True)
                time.sleep(wait_time)
                continue
            logger.error("Ollama request failed with network error after all retries: %s", e, exc_info=True)
            raise RuntimeError(f"Ollama request failed: {e}") from e
    raise RuntimeError("Ollama request failed after all retries")


def _get_openai_response(prompt: str = "", system_prompt: str = "", model: str = None,
                         base_url: str = None, api_key: str = None,
                         temperature: Optional[float] = None,
                         timeout: Optional[float] = None,
                        messages: Optional[List[Dict[str, str]]] = None,
                        add_cache_control: bool = False,
) -> str:
    """Send a prompt to the configured OpenAI-compatible API and return the response text."""
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
        "model": model or settings.OPENAI_MODEL,
        "messages": api_messages,
        "temperature": temperature if temperature is not None else settings.LLM_TEMPERATURE,
    }

    logger.info("LLM request (openai): model=%s, system_prompt=%.200s..., prompt=%.500s...", model, system_prompt, prompt)
    max_retries = 3
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


                                                                                                                                                                                                                                                                                   
            # Validate response structure                                                                                                                                                                                                                                          
            if "choices" not in data or not data["choices"]:                                                                                                                                                                                                                       
                logger.error(                                                                                                                                                                                                                                                      
                    "OpenAI response missing 'choices' key. Full response: %s",                                                                                                                                                                                                    
                    json.dumps(data)[:2000]                                                                                                                                                                                                                                        
                )                                                                                                                                                                                                                                                                  
                raise RuntimeError(f"OpenAI API returned unexpected format: missing 'choices'. Response: {str(data)[:500]}")                                                                                                                                                       
                                                                                                                                                                                                                                                                                   


            content = data["choices"][0]["message"]["content"]                                                                                                                                                                                                                     
            logger.info("LLM response (openai): %.500s...", content)                                                                                                                                                                                                               
            return content
        except httpx.HTTPStatusError as e:
            # Log the full response body for non-retryable errors (especially 400)
            if e.response.status_code not in (429, 500, 502, 503, 504):
                logger.error(
                    "OpenAI request failed with HTTP %d (non-retryable). "
                    "URL: %s\nResponse body: %s\nRequest payload model: %s, messages count: %d",
                    e.response.status_code,
                    str(e.request.url),
                    e.response.text[:2000],
                    payload.get("model"),
                    len(payload.get("messages", [])),
                    exc_info=True,
                )
                raise RuntimeError(f"OpenAI request failed with HTTP {e.response.status_code}: {e.response.text[:500]}") from e
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.warning(f"OpenAI request failed with HTTP {e.response.status_code}. Response: {e.response.text[:500]}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
            raise RuntimeError(f"OpenAI request failed: {e.response.status_code} - {e.response.text[:500]}") from e
        except httpx.HTTPError as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.warning(f"OpenAI request failed with network error: {e}. Retrying in {wait_time}s...", exc_info=True)
                time.sleep(wait_time)
                continue
            logger.error("OpenAI request failed with network error after all retries: %s", e, exc_info=True)
            raise RuntimeError(f"OpenAI request failed: {e}") from e
    raise RuntimeError("OpenAI request failed after all retries")


def get_llm_response(prompt: str, system_prompt: str = "", model_type: str = "actuator", symbol: Optional[str] = None) -> str:
    """Send a prompt to the configured LLM provider and return the response text.

    Uses Redis caching with a 5-minute TTL (keyed by prompt + system prompt).
    model_type: "mind" for complex reasoning, "actuator" for fast time‑critical decisions.
    symbol: Optional symbol/ticker for semantic cache generalization.

    Note: get_cached_llm_response now returns a dict with "response", "provider", "model".
    This function returns only the response text for backward compatibility.
    """
    from src.llm.cache import get_cached_llm_response  # local import to avoid circular dependency at module level

    result = get_cached_llm_response(prompt, system_prompt, ttl=300, model_type=model_type, symbol=symbol)
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

    for role in ("mind", "actuator", "weak"):
        if role == "mind":
            provider = settings.LLM_MIND_PROVIDER or settings.LLM_PROVIDER
        elif role == "weak":
            provider = settings.LLM_WEAK_PROVIDER or settings.LLM_PROVIDER
        else:
            provider = settings.LLM_ACTUATOR_PROVIDER or settings.LLM_PROVIDER

        if provider == "openai":
            if role == "mind":
                model = settings.OPENAI_MIND_MODEL or settings.OPENAI_MODEL
                base_url = settings.OPENAI_MIND_BASE_URL or settings.OPENAI_BASE_URL
                api_key = settings.OPENAI_MIND_API_KEY or settings.OPENAI_API_KEY
            elif role == "weak":
                model = settings.OPENAI_WEAK_MODEL or settings.OPENAI_MODEL
                base_url = settings.OPENAI_WEAK_BASE_URL or settings.OPENAI_BASE_URL
                api_key = settings.OPENAI_WEAK_API_KEY or settings.OPENAI_API_KEY
            else:
                model = settings.OPENAI_ACTUATOR_MODEL or settings.OPENAI_MODEL
                base_url = settings.OPENAI_ACTUATOR_BASE_URL or settings.OPENAI_BASE_URL
                api_key = settings.OPENAI_ACTUATOR_API_KEY or settings.OPENAI_API_KEY
        else:
            if role == "mind":
                model = settings.OLLAMA_MIND_MODEL or settings.OLLAMA_MODEL
                base_url = settings.OLLAMA_MIND_BASE_URL or settings.OLLAMA_BASE_URL
                api_key = settings.OLLAMA_MIND_API_KEY or settings.OLLAMA_API_KEY
            elif role == "weak":
                model = settings.OLLAMA_WEAK_MODEL or settings.OLLAMA_MODEL
                base_url = settings.OLLAMA_WEAK_BASE_URL or settings.OLLAMA_BASE_URL
                api_key = settings.OLLAMA_WEAK_API_KEY or settings.OLLAMA_API_KEY
            else:
                model = settings.OLLAMA_ACTUATOR_MODEL or settings.OLLAMA_MODEL
                base_url = settings.OLLAMA_ACTUATOR_BASE_URL or settings.OLLAMA_BASE_URL
                api_key = settings.OLLAMA_ACTUATOR_API_KEY or settings.OLLAMA_API_KEY

        if not base_url:
            results[role] = {
                "status": "disconnected",
                "provider": provider,
                "model": model or "unknown",
                "error": "No base URL configured",
            }
            continue

        try:
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
        except Exception as e:
            results[role] = {
                "status": "disconnected",
                "provider": provider,
                "model": model or "unknown",
                "error": str(e)[:200],
            }

    return results
