import logging
import time
import random
import threading
from typing import Optional, List, Dict, Callable

from src.config.settings import settings

logger = logging.getLogger(__name__)

# Cache for dynamically discovered models: {model_type: (timestamp, [models])}
_g4f_models_cache: dict = {}
_g4f_models_cache_lock = threading.Lock()
_G4F_CACHE_TTL = 3600  # 1 hour

# Keyword heuristics for categorizing models into tiers.
# Checked in order: mind first, then weak, then actuator (fallback).
_MIND_KEYWORDS = (
    "o1", "o3", "o1-pro", "o3-pro",
    "gpt-4o", "gpt-4.1", "gpt-4.5", "gpt-5",
    "claude-3.5-sonnet", "claude-3.7-sonnet", "claude-3-opus",
    "deepseek-r1", "deepseek-v3", "deepseek-v4-pro", "deepseek-v3-pro", "deepseek-r1-pro", "deepseek-pro",
    "gemini-2", "gemini-2.5-pro", "gemini-pro",
    "qwen-max", "qwen-3",
    "llama-3.3-70b", "llama-3.1-405b", "llama-4",
    "mistral-large", "mixtral-8x22b",
    "grok-2", "grok-3",
    "command-r-plus", "dbrx",
)
_ACTUATOR_KEYWORDS = (
    "mini", "haiku", "flash", "70b", "72b", "34b", "32b", "mixtral-8x7b", "gpt-4", "gpt-3.5", 
    "claude-3", "gemini-1.5", "llama-3.1", "llama-3.2", "llama-3.3", 
    "mistral", "qwen", "deepseek-v2", "deepseek-v4-flash", "deepseek-coder", "grok", "command-r", "mixtral",
    "yi-34b", "zephyr", "starling", "openhermes", "dolphin", "vicuna", "orca", "solar-pro", "code-llama"
)
_WEAK_KEYWORDS = (
    "1b", "2b", "3b", "6b", "7b", "8b", "9b", "11b", "13b", "14b", "3.5-turbo", "small", "nano", "tiny", "gpt-3", 
    "mistral-7b", "qwen-turbo", "deepseek-chat", "gemma", "phi", "solar", "tinyllama", "falcon-7b", "falcon-1b",
    "stablelm", "redpajama", "qwen-1.5b", "qwen-1.8b", "qwen-7b", "openchat", "wizardlm"
)

def _categorize_model(model_name: str) -> Optional[str]:
    """Categorize a model name into 'mind', 'actuator', or 'weak' using keyword heuristics."""
    name_lower = model_name.lower()
    
    # 1. Check weak first for smaller models (e.g., 7b, 8b, gpt-3.5-turbo)
    if any(kw in name_lower for kw in _WEAK_KEYWORDS):
        return "weak"
    
    # 2. Check actuator second for big but flash models (e.g., mini, haiku, flash, 70b)
    # This prevents "gpt-4o-mini" from being caught by the "gpt-4o" mind keyword.
    if any(kw in name_lower for kw in _ACTUATOR_KEYWORDS):
        return "actuator"
    
    # 3. Check mind third for huge reasoning models
    if any(kw in name_lower for kw in _MIND_KEYWORDS):
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
        from g4f.providers.any_provider import AnyProvider
        
        # Ensure the model map is populated
        if not AnyProvider.models:
            AnyProvider.update_model_map()
            
        all_models = AnyProvider.models
        
        # Filter out non-text models (images, videos, audio, vision)
        exclude_models = set(AnyProvider.image_models) | set(AnyProvider.video_models) | set(AnyProvider.audio_models)
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
    max_retries: int = 3,
) -> dict:
    """Send a prompt to the configured g4f model and return a dict with 'content' and 'usage'."""
    from g4f.client import Client
    from g4f.providers.any_provider import AnyProvider

    # Use AnyProvider to automatically route to the correct working provider for the model
    client_kwargs = {
        "provider": AnyProvider
    }

    # Pass timeout to g4f Client constructor if supported
    if timeout is not None:
        client_kwargs["timeout"] = timeout

    client = Client(**client_kwargs)

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

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(**payload)
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
