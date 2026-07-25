import asyncio
import logging
from src.llm.llm_client import get_llm_response
from src.llm.cache import get_cached_llm_response, _llm_executor

logger = logging.getLogger(__name__)

def summarize_text(text: str, context: str = "general", max_length: int = 500, force_primary_model: bool = False) -> str:
    """
    Summarize the given text using the weak LLM model to save tokens.

    Args:
        text: The text to summarize.
        context: The context of the text (e.g., "news", "prompt") to guide summarization.
        max_length: The maximum desired length of the summary in characters.

    Returns:
        The summarized text, or the original text if summarization fails or text is too short.
    """
    if not text or len(text) <= max_length:
        return text

    system_prompt = (
        "You are an expert summarizer. Your task is to condense the provided text "
        "while retaining all critical information, key metrics, and actionable insights. "
        "Be extremely concise."
    )
    prompt = (
        f"Context: {context}\n"
        f"Summarize the following text in less than {max_length} characters. "
        f"Preserve all important numbers, dates, and entity names.\n\n"
        f"Text to summarize:\n{text}"
    )

    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        llm_result = get_cached_llm_response(
            "",
            "",
            ttl=86400,  # Cache summaries for 24 hours to avoid repeated LLM calls
            model_type="weak",
            messages=messages,
            request_type="summarization"
        )
        summary = llm_result.get("response", "")
        if summary and summary.strip():
            return summary.strip()
        return text
    except Exception as e:
        logger.error(f"Failed to summarize text using weak model: {type(e).__name__}: {e}")
        return text


async def summarize_text_async(text: str, context: str = "general", max_length: int = 500, force_primary_model: bool = False) -> str:
    """
    Asynchronous wrapper for summarize_text.
    Runs the blocking summarization call in a dedicated thread pool to avoid
    blocking the event loop and exhausting the default executor.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _llm_executor,
        summarize_text,
        text,
        context,
        max_length,
        force_primary_model,
    )
