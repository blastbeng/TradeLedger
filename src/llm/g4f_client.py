import logging
import time
import asyncio
import random
import re
import threading
from typing import Optional, List, Dict, Callable

from src.config.settings import settings
from src.exchanges.proxy_utils import _get_proxies

logger = logging.getLogger(__name__)

# Cache for dynamically discovered models: {model_type: (timestamp, [models])}
_g4f_models_cache: dict = {}
_g4f_models_cache_lock = threading.Lock()
_G4F_CACHE_TTL = 3600  # 1 hour

# Keyword heuristics for categorizing models into tiers.
# Checked in order: mind first, then weak, then actuator (fallback).
_MIND_KEYWORDS = (
    "o1", "o3", "o1-pro", "o3-pro", "o4-mini",
    "gpt-4o", "gpt-4.1", "gpt-4.5", "gpt-5", "gpt-4-turbo",
    "claude-3.5-sonnet", "claude-3.7-sonnet", "claude-3-opus",
    "claude-sonnet-4", "claude-opus-4", "claude-opus-4.6", "claude-opus-4.7", "claude-opus-4-8",
    "deepseek-r1", "deepseek-v3", "deepseek-v4-pro", "deepseek-v3-pro", "deepseek-r1-pro", "deepseek-pro",
    "gemini-2", "gemini-2.5-pro", "gemini-pro", "gemini-3.1-pro", "gemini-3-pro",
    "qwen-max", "qwen-3", "qwen-2.5-72b", "qwen-2-72b", "qwen-3-235b", "qwen-3.5-397b", "qwen-3-coder-480b",
    "llama-3.3-70b", "llama-3.1-405b", "llama-4",
    "mistral-large", "mixtral-8x22b", "mistral-large-3",
    "grok-2", "grok-3", "grok-4",
    "command-r-plus", "command-a",
    "kimi-k2", "minimax-m3", "glm-5", "glm-4.6", "glm-4.7",
    "nemotron-253b", "nemotron-3-ultra", "cogito-v2.1-671b",
)
_ACTUATOR_KEYWORDS = (
    "mixtral-8x7b", "gpt-3.5", 
    "claude-3", "gemini-1.5", "llama-3.1", "llama-3.2", "llama-3.3", 
    "mistral", "qwen", "deepseek-v2", "deepseek-v4-flash", "deepseek-coder", "grok", "command-r", "mixtral",
    "yi-34b", "zephyr", "starling", "openhermes", "dolphin", "vicuna", "orca", "solar-pro", "code-llama",
    "kimi", "minimax", "glm",
)
_WEAK_KEYWORDS = (
    "3.5-turbo", "small", "nano", "tiny", "gpt-3", 
    "mistral-7b", "qwen-turbo", "deepseek-chat", "gemma", "phi", "solar", "tinyllama", "falcon-7b", "falcon-1b",
    "stablelm", "redpajama", "qwen-1.5b", "qwen-1.8b", "qwen-7b", "openchat", "wizardlm", "haiku-4-5"
)

# Models that contain mini/haiku/flash but should bypass the actuator override
_MIND_OVERRIDES = ("o4-mini",)
_WEAK_OVERRIDES = ("haiku-4-5",)

def _categorize_model(model_name: str) -> Optional[str]:
    """Categorize a model name into 'mind', 'actuator', or 'weak' using keyword heuristics."""
    name_lower = model_name.lower()
    
    # 0. Explicit exceptions that bypass the mini/haiku/flash override below
    if any(kw in name_lower for kw in _MIND_OVERRIDES):
        return "mind"
    if any(kw in name_lower for kw in _WEAK_OVERRIDES):
        return "weak"

    # 1. Check for mini/flash/haiku first to prevent e.g. "gpt-4o-mini" matching "gpt-4o"
    if any(kw in name_lower for kw in ("mini", "haiku", "flash")):
        return "actuator"
        
    # 2. Check weak keywords (non-sizes)
    if any(kw in name_lower for kw in _WEAK_KEYWORDS):
        return "weak"
        
    # 3. Check mind keywords
    if any(kw in name_lower for kw in _MIND_KEYWORDS):
        return "mind"
        
    # 4. Check actuator keywords (non-sizes)
    if any(kw in name_lower for kw in _ACTUATOR_KEYWORDS):
        return "actuator"
        
    # 5. Check sizes using exact word boundaries to avoid false positives (e.g. "27b" matching "7b")
    # Weak sizes
    if re.search(r'\b(1b|2b|3b|6b|7b|8b|9b|11b|13b|14b)\b', name_lower):
        return "weak"
    # Actuator sizes
    if re.search(r'\b(20b|22b|27b|30b|32b|34b|40b|49b|65b|70b|72b|104b)\b', name_lower):
        return "actuator"
    # Mind sizes (huge models)
    if re.search(r'\b(120b|235b|253b|397b|405b|480b|550b|671b|675b)\b', name_lower):
        return "mind"
    
    # Default to actuator for unknown models, but log it for visibility
    logger.debug(f"Model '{model_name}' did not match any specific keywords, defaulting to 'actuator'.")
    return "actuator"

def _discover_g4f_models() -> dict:
    """Dynamically discover and categorize available g4f models.
    
    Returns a dict: {"mind": [...], "actuator": [...], "weak": [...]}
    """
    categorized = {"mind": [], "actuator": [], "weak": []}
    try:
        from g4f.providers import any_model_map
        
        # Get all models from the model_map keys
        all_models = list(any_model_map.model_map.keys())
        
        # Exclude non-text models (audio, image, video).
        # Vision models are intentionally kept as they provide text completions.
        exclude_models = set(any_model_map.audio_models) | set(any_model_map.image_models) | set(any_model_map.video_models)
        # Also exclude generic/default entries
        exclude_models.update(["default", "custom", "video", "auto"])
        
        text_models = [m for m in all_models if m not in exclude_models]
        
        for model_name in text_models:
            tier = _categorize_model(model_name)
            if tier and model_name not in categorized[tier]:
                categorized[tier].append(model_name)
                
    except Exception as e:
        logger.warning("Failed to dynamically discover g4f models: %s. Using fallback list.", e)
        # Absolute fallback
        categorized = {
            "mind": ["gpt-4o", "deepseek-r1", "claude-3.5-sonnet"],
            "actuator": ["gpt-4o-mini", "claude-3-haiku", "gemini-flash", "llama-3.1-70b"],
            "weak": ["llama-3-8b", "gpt-3.5-turbo", "mistral-7b"],
        }
    return categorized

def _get_g4f_models(model_type: str) -> List[str]:
    """Return the list of g4f models for the given model type, dynamically discovered."""
    now = time.time()
    with _g4f_models_cache_lock:
        if model_type in _g4f_models_cache:
            ts, models = _g4f_models_cache[model_type]
            if now - ts < _G4F_CACHE_TTL and models:
                return models
    
    # Discover all categories
    categorized = _discover_g4f_models()
    
    with _g4f_models_cache_lock:
        for mt, models in categorized.items():
            _g4f_models_cache[mt] = (now, models)
    
    return categorized.get(model_type, categorized["actuator"])

def _get_g4f_response(
    prompt: str = "",
    system_prompt: str = "",
    model: str = None,
    base_url: str = None,
    api_key: str = None,
    temperature: Optional[float] = None,
    timeout: Optional[float] = None,
    messages: Optional[List[Dict[str, str]]] = None,
    add_cache_control: bool = False,
    thinking_enabled: bool = True,
    reasoning_effort: str = "low",
    max_retries: int = 3,
    max_tokens: Optional[int] = None,
) -> dict:
    """Send a prompt to the configured g4f model and return a dict with 'content' and 'usage'."""
    from g4f.client import ClientFactory

    # Use the local g4f API server if configured, otherwise let g4f use its defaults
    client_kwargs = {}
    if settings.G4F_BASE_URL:
        client_kwargs["base_url"] = settings.G4F_BASE_URL
    if settings.G4F_API_KEY:
        client_kwargs["api_key"] = settings.G4F_API_KEY

    # Pass a random proxy if enabled and available
    proxy = _get_proxies()
    if proxy:
        client_kwargs["proxies"] = proxy

    # Pass timeout to g4f Client constructor if supported
    if timeout is not None:
        client_kwargs["timeout"] = timeout

    client = ClientFactory.create_async_client(**client_kwargs)

    if messages is not None:
        api_messages = [dict(msg) for msg in messages]
    else:
        api_messages = []
        if system_prompt:
            api_messages.append({"role": "system", "content": system_prompt})
        api_messages.append({"role": "user", "content": prompt})

    if not model:
        model = "gpt-4o-mini"

    payload = {
        "model": model,
        "messages": api_messages,
        "temperature": temperature if temperature is not None else settings.LLM_TEMPERATURE,
    }

    # Always send reasoning_effort: "low" when thinking is disabled,
    # or the computed value when thinking is enabled.
    payload["reasoning_effort"] = "low" if not thinking_enabled else reasoning_effort

    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    for attempt in range(max_retries):
        try:
            async def _make_request():
                return await client.chat.completions.create(**payload)

            try:
                # If there's no running loop, use asyncio.run
                asyncio.get_running_loop()
                # If we are in a running loop, this will raise RuntimeError, handled below.
                # Fallback for running loop (though ideally this function should be async if called from one)
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, _make_request())
                    response = future.result()
            except RuntimeError:
                response = asyncio.run(_make_request())

            content = response.choices[0].message.content
            
            # g4f doesn't always return usage, so we estimate it
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
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.warning(f"g4f request failed with error: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
            logger.error("g4f request failed with error after all retries: %s", e, exc_info=True)
            raise RuntimeError(f"g4f request failed: {e}") from e
    raise RuntimeError("g4f request failed after all retries")
