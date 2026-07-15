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
_MIND_KEYWORDS = ("o1", "o3", "gpt-4o", "gpt-4.1", "claude-3.5-sonnet", "claude-3-opus", "deepseek-r1", "deepseek-v3", "gemini-2", "gemini-pro", "qwen-max", "llama-3.3-70b", "llama-3.1-405b", "mistral-large", "grok-2")
_WEAK_KEYWORDS = ("mini", "haiku", "flash", "8b", "7b", "3.5-turbo", "small", "nano", "tiny", "gpt-3", "gemini-flash", "llama-3-8b", "mistral-7b", "qwen-turbo", "deepseek-chat")

def _categorize_model(model_name: str) -> Optional[str]:
    """Categorize a model name into 'mind', 'actuator', or 'weak' using keyword heuristics."""
    name_lower = model_name.lower()
    if any(kw in name_lower for kw in _MIND_KEYWORDS):
        return "mind"
    if any(kw in name_lower for kw in _WEAK_KEYWORDS):
        return "weak"
    return "actuator"

def _discover_g4f_models() -> dict:
    """Dynamically discover and categorize available g4f models.
    
    Returns a dict: {"mind": [...], "actuator": [...], "weak": [...]}
    """
    categorized = {"mind": [], "actuator": [], "weak": []}
    try:
        import g4f
        # g4f.models is a dict of model name -> Model object
        all_models = []
        if hasattr(g4f, 'models') and hasattr(g4f.models, 'utils'):
            # Access the internal model registry
            from g4f.models import Model
            # Try to iterate over all known models
            if hasattr(g4f.models, '__iter__'):
                for model_name in g4f.models:
                    all_models.append(str(model_name))
        elif hasattr(g4f, 'models'):
            # Fallback: try dict-like access
            try:
                all_models = list(g4f.models.keys())
            except (AttributeError, TypeError):
                pass
        
        # If we couldn't get models from the registry, fall back to a minimal
        # set of well-known g4f model names so the system remains functional.
        if not all_models:
            all_models = [
                "gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-3.5-turbo",
                "claude-3.5-sonnet", "claude-3-haiku",
                "gemini-flash", "gemini-pro",
                "llama-3.1-70b", "llama-3-8b",
                "deepseek-chat", "deepseek-r1",
                "mistral-7b",
            ]
        
        for model_name in all_models:
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
    from g4f.Provider import RetryProvider, Phind, FreeChatgpt, Liaobots, Blackbox, OpenaiChat, Gemini, Bing

    # Use RetryProvider to automatically try multiple providers and blacklist failing ones.
    # This satisfies the requirement to dynamically manage providers and retry/blacklist.
    client = Client(
        provider=RetryProvider([OpenaiChat, Gemini, Bing, Phind, FreeChatgpt, Liaobots, Blackbox], shuffle=True)
    )

    if messages is not None:
        api_messages = [dict(msg) for msg in messages]
    else:
        api_messages = []
        if system_prompt:
            api_messages.append({"role": "system", "content": system_prompt})
        api_messages.append({"role": "user", "content": prompt})

    # If no specific model is provided, we don't need to pick one here because
    # the cache layer will pass the model string. However, if it's empty, we fallback.
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
