import logging
import re
from typing import Optional

from src.exchanges.proxy_utils import _get_proxies

logger = logging.getLogger(__name__)

def get_isin_from_duckduckgo(symbol: str, name: Optional[str] = None) -> Optional[str]:
    """Fetch ISIN for a symbol using DuckDuckGo text search as a fallback."""
    try:
        from ddgs import DDGS
    except ImportError:
        logger.warning("ddgs not installed. Skipping DuckDuckGo ISIN lookup.")
        return None

    query = f"{name or symbol} ISIN"
    
    try:
        proxy = _get_proxies()
        ddgs = DDGS(proxy=proxy, timeout=10) if proxy else DDGS(timeout=10)
        # Use text search to find web pages mentioning the ISIN
        results = ddgs.text(query, max_results=5)
        
        # Search through the results for an ISIN pattern (2 letters, 9 alphanumeric, 1 digit)
        isin_pattern = re.compile(r"\b[A-Z]{2}[A-Z0-9]{9}\d\b")
        for r in results:
            title = r.get("title", "")
            body = r.get("body", "")
            href = r.get("href", "")
            
            # Check all text fields for an ISIN
            for text in [title, body, href]:
                match = isin_pattern.search(text)
                if match:
                    isin = match.group(0)
                    logger.info(f"DuckDuckGo search provided ISIN {isin} for {symbol}")
                    return isin
    except Exception as e:
        logger.debug(f"DuckDuckGo ISIN lookup failed for {symbol}: {type(e).__name__}: {e}")
    
    return None
