import json
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

def get_isin_from_duckduckgo(symbol: str, name: Optional[str] = None) -> Optional[str]:
    """Fetch ISIN for a symbol using DuckDuckGo AI Chat API as a fallback."""
    try:
        from ddgs import DDGS
    except ImportError:
        logger.warning("ddgs not installed. Skipping DuckDuckGo ISIN lookup.")
        return None

    query = f"{name or symbol} ISIN code. Respond ONLY with a JSON object like {{\"ISIN\": \"value\"}}."
    
    try:
        ddgs = DDGS()
        # Use a fast model available on DuckDuckGo
        response = ddgs.chat(query, model="gpt-4o-mini")
        
        # Extract JSON from the response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            isin = data.get("ISIN")
            if isin:
                isin = isin.strip()
                # Validate ISIN format (2 letters, 9 alphanumeric, 1 digit)
                if re.match(r"^[A-Z]{2}[A-Z0-9]{9}\d$", isin):
                    logger.info(f"DuckDuckGo provided ISIN {isin} for {symbol}")
                    return isin
                else:
                    logger.debug(f"DuckDuckGo returned invalid ISIN format for {symbol}: {isin}")
    except Exception as e:
        logger.debug(f"DuckDuckGo ISIN lookup failed for {symbol}: {type(e).__name__}: {e}")
    
    return None
