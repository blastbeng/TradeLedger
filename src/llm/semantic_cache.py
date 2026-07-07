import re
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

# Regex to find tickers ending with .MI (or other configured suffixes)
TICKER_REGEX = re.compile(r'\b([A-Z0-9]+\.MI)\b')

def generalize_prompt(prompt: str) -> Tuple[str, Optional[str]]:
    """
    Replaces specific stock tickers in the prompt with a generic [TICKER] placeholder.

    Returns:
        A tuple containing (generalized_prompt, extracted_ticker).
        If no ticker is found, extracted_ticker will be None.
    """
    match = TICKER_REGEX.search(prompt)
    if not match:
        return prompt, None

    ticker = match.group(1)
    generalized_prompt = prompt.replace(ticker, "[TICKER]")
    return generalized_prompt, ticker

def reconstruct_response(cached_response: str, current_ticker: str, original_prompt: str) -> str:
    """
    Reconstructs the cached response by injecting the current ticker context
    back into the template.
    """
    if not current_ticker:
        return cached_response

    # Replace the placeholder with the current ticker
    return cached_response.replace("[TICKER]", current_ticker)
