import logging
import re
from typing import Optional

from src.exchanges.proxy_utils import _get_proxies

logger = logging.getLogger(__name__)

# Suppress noisy INFO/DEBUG logs from the ddgs and primp libraries
logging.getLogger("ddgs").setLevel(logging.WARNING)
logging.getLogger("primp").setLevel(logging.WARNING)

def get_isin_from_duckduckgo(symbol: str, name: Optional[str] = None) -> Optional[str]:
    """Fetch ISIN for a symbol using DuckDuckGo text search as a fallback."""
    # If the symbol is already a valid ISIN, return it immediately
    if re.match(r'^[A-Z]{2}[A-Z0-9]{9}\d$', symbol):
        logger.debug(f"Symbol {symbol} is already a valid ISIN")
        return symbol

    def _search_isin(search_query: str) -> Optional[str]:
        try:
            proxy = _get_proxies()
            ddgs = DDGS(proxy=proxy, timeout=10) if proxy else DDGS(timeout=10)
            results = ddgs.text(search_query, max_results=5)
            
            isin_pattern = re.compile(r"\b[A-Z]{2}[A-Z0-9]{9}\d\b")
            for r in results:
                title = r.get("title", "")
                body = r.get("body", "")
                href = r.get("href", "")
                
                for text in [title, body, href]:
                    match = isin_pattern.search(text)
                    if match:
                        isin = match.group(0)
                        logger.info(f"DuckDuckGo search provided ISIN {isin} for {symbol}")
                        return isin
        except Exception as e:
            logger.warning(f"DuckDuckGo ISIN lookup failed for {symbol}: {type(e).__name__}: {e}")
        return None

    try:
        from ddgs import DDGS
    except ImportError:
        logger.debug("ddgs not installed. Skipping DuckDuckGo ISIN lookup.")
        return None

    # First attempt: explicitly ask for an Italian ISIN
    query = f"{name or symbol} Italian ISIN"
    isin = _search_isin(query)
    
    if isin and isin.startswith("IT"):
        return isin
        
    # If a non-Italian ISIN is found, redo the search to be sure we have the correct one
    if isin and not isin.startswith("IT"):
        logger.info(f"Found non-Italian ISIN {isin} for {symbol}, redoing search to confirm.")
        retry_query = f"{name or symbol} ISIN Italia"
        retry_isin = _search_isin(retry_query)
        if retry_isin:
            return retry_isin
        return isin
        
    # If the first attempt found nothing, do a generic search
    if not isin:
        generic_query = f"{name or symbol} ISIN"
        return _search_isin(generic_query)
        
    return None
