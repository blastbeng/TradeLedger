import logging
import re
from typing import Optional

from src.exchanges.proxy_utils import _get_proxies
from src.config.settings import settings
from src.database import save_discovered_symbol

logger = logging.getLogger(__name__)

# Map of common country names to their 2-letter ISIN country codes
_COUNTRY_ISIN_PREFIX = {
    "italy": "IT",
    "france": "FR",
    "germany": "DE",
    "spain": "ES",
    "netherlands": "NL",
    "united states": "US",
    "usa": "US",
    "united kingdom": "GB",
    "uk": "GB",
}

# Suppress noisy INFO/DEBUG logs from the ddgs and primp libraries
logging.getLogger("ddgs").setLevel(logging.WARNING)
logging.getLogger("primp").setLevel(logging.WARNING)

def get_isin_from_duckduckgo(symbol: str, name: Optional[str] = None, asset_type: str = "stock") -> Optional[str]:
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
            found_isins = []
            for r in results:
                title = r.get("title", "")
                body = r.get("body", "")
                href = r.get("href", "")
                
                for text in [title, body, href]:
                    matches = isin_pattern.findall(text)
                    found_isins.extend(matches)
            
            if not found_isins:
                return None
            
            # Prefer ISINs with the target country prefix
            if isin_prefix:
                for isin in found_isins:
                    if isin.startswith(isin_prefix):
                        logger.info(f"DuckDuckGo search provided target ISIN {isin} for {symbol}")
                        return isin
            
            # Return the first found ISIN if no target prefix match
            isin = found_isins[0]
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

    country_name = settings.TARGET_COUNTRY.capitalize()
    isin_prefix = _COUNTRY_ISIN_PREFIX.get(settings.TARGET_COUNTRY, "")

    def _is_valid(isin_str: Optional[str]) -> bool:
        return bool(isin_str and (not isin_prefix or isin_str.startswith(isin_prefix)))

    def _save_non_target_isin(non_target_isin: str) -> None:
        """Save a non-target ISIN to the DB so re-evaluation skips it, then return None."""
        logger.info(f"Found non-{country_name} ISIN {non_target_isin} for {symbol}. Saving to DB to skip in future re-evaluations.")
        try:
            save_discovered_symbol(symbol, non_target_isin, asset_type, name)
        except Exception as e:
            logger.warning(f"Failed to save non-target ISIN {non_target_isin} for {symbol} to DB: {e}")

    # First attempt: explicitly ask for an ISIN from the target country
    asset_keyword = "ETF" if asset_type.lower() == "etf" else "stock"
    search_terms = f"{name} {symbol}" if name else symbol
    query = f"{search_terms} {country_name} {asset_keyword} ISIN"
    isin = _search_isin(query)
    if _is_valid(isin):
        return isin
        
    # If a non-target ISIN is found, save it to the DB and return None
    if isin and isin_prefix and not isin.startswith(isin_prefix):
        _save_non_target_isin(isin)
        return None
            
    # If the first attempt found nothing, do a generic search
    if not isin:
        generic_query = f"{search_terms} {asset_keyword} ISIN"
        generic_isin = _search_isin(generic_query)
        if _is_valid(generic_isin):
            return generic_isin
        
        # If generic search finds a non-target ISIN, save it and return None
        if generic_isin and isin_prefix and not generic_isin.startswith(isin_prefix):
            _save_non_target_isin(generic_isin)
            return None
        
    return None
