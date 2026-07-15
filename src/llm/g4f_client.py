import logging
import time
import random
from typing import Optional, List, Dict, Callable

from src.config.settings import settings

logger = logging.getLogger(__name__)

# Mapping of model types to g4f model names.
# HUGE REASONING MODELS FOR MIND/FALLBACK_MIND CALLS
# FASTER MODELS FOR ACTUATOR/FALLBACK_ACTUATOR CALLS
# SMALLER/WEAK MODELS FOR WEAK/FALLBACK_WEAK CALLS
G4F_MODEL_MAPPING = {
    "mind": ["gpt-4o", "o1", "claude-3.5-sonnet", "deepseek-r1"],
    "actuator": ["gpt-4o-mini", "claude-3-haiku", "gemini-flash", "llama-3.1-70b"],
    "weak": ["llama-3-8b", "gemini-flash", "gpt-3.5-turbo", "mistral-7b"],
}

def _get_g4f_models(model_type: str) -> List[str]:
    """Return the list of g4f models for the given model type."""
    return G4F_MODEL_MAPPING.get(model_type, G4F_MODEL_MAPPING["actuator"])

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
