import logging
from src.llm.llm_client import get_llm_response

logger = logging.getLogger(__name__)

def summarize_text(text: str, context: str = "general", max_length: int = 500) -> str:
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
        summary = get_llm_response(prompt, system_prompt, model_type="weak")
        if summary and summary.strip():
            return summary.strip()
        return text
    except Exception as e:
        logger.error(f"Failed to summarize text using weak model: {e}")
        return text
