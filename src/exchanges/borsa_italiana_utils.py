import logging
import re
import threading
import time
from typing import Optional, Dict

import httpx
from bs4 import BeautifulSoup

from src.config.settings import settings
from src.exchanges.proxy_utils import _get_proxies

logger = logging.getLogger(__name__)

# --- Borsa Italiana Circuit Breaker ---
_bi_error_count = 0
_bi_last_error_time = 0.0
_bi_circuit_open_until = 0.0
_bi_lock = threading.Lock()

BI_MAX_ERRORS = 20
BI_CIRCUIT_COOLDOWN = 300  # 5 minutes

def _check_bi_circuit() -> bool:
    """Return True if the Borsa Italiana circuit is open (calls should be skipped)."""
    with _bi_lock:
        return time.time() < _bi_circuit_open_until

def _record_bi_error(exc: Optional[Exception] = None):
    """Record a Borsa Italiana error and potentially trip the circuit breaker."""
    global _bi_error_count, _bi_last_error_time, _bi_circuit_open_until
    with _bi_lock:
        now = time.time()
        if now - _bi_last_error_time > 300:
            _bi_error_count = 0
        _bi_error_count += 1
        _bi_last_error_time = now
        if _bi_error_count >= BI_MAX_ERRORS:
            if _bi_circuit_open_until < now:
                exc_msg = f" Last error: {exc}" if exc else ""
                logger.error(f"Borsa Italiana circuit breaker tripped due to {_bi_error_count} errors. Blocking BI calls for {BI_CIRCUIT_COOLDOWN}s.{exc_msg}")
            _bi_circuit_open_until = now + BI_CIRCUIT_COOLDOWN

def _reset_bi_circuit():
    """Reset the Borsa Italiana circuit breaker after a successful call."""
    global _bi_error_count
    with _bi_lock:
        _bi_error_count = 0


# --- Borsa Italiana token cache ---
_borsa_token_cache: Dict[str, tuple] = {}  # {cache_key: (timestamp, token)}
_borsa_token_cache_lock = threading.Lock()
_BORSA_TOKEN_CACHE_TTL = 300  # 5 minutes


def _get_borsa_italiana_token(isin: str, market_code: str) -> Optional[str]:
    """Dynamically fetch the bearer token from the Borsa Italiana summary chart page, with caching."""
    if _check_bi_circuit():
        return None

    cache_key = f"{isin}-{market_code}"
    now = time.time()

    # Check cache first
    with _borsa_token_cache_lock:
        cached = _borsa_token_cache.get(cache_key)
        if cached and (now - cached[0]) < _BORSA_TOKEN_CACHE_TTL:
            return cached[1]

    url = f"https://grafici.borsaitaliana.it/summary-chart/{isin}-{market_code}?lang=it"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        with httpx.Client(proxy=_get_proxies(), timeout=15.0, follow_redirects=True) as client:
            response = client.get(url, headers=headers)
            response.raise_for_status()
            # Extract token from <chart-allinone ... token="..." ...>
            # Use BeautifulSoup for robust parsing, with regex as a fallback
            soup = BeautifulSoup(response.text, "html.parser")
            chart_tag = soup.find("chart-allinone")
            token = chart_tag.get("token") if chart_tag else None


            if not token:
                # Fallback to regex if BeautifulSoup fails to find the tag
                match = re.search(r'<chart-allinone[^>]*token="([^"]+)"', response.text)
                if match:
                    token = match.group(1)


            if token:
                # Cache the token
                with _borsa_token_cache_lock:
                    _borsa_token_cache[cache_key] = (now, token)
                _reset_bi_circuit()
                return token


            logger.warning(f"Could not find Borsa Italiana token for {isin}-{market_code}")
            return None
    except (httpx.RequestError, httpx.HTTPStatusError, ValueError, AttributeError, OSError) as e:
        _record_bi_error(e)
        logger.warning(f"Failed to fetch Borsa Italiana token for {isin}-{market_code}: {e}")
        return None


def _invalidate_borsa_token_cache(isin: str, market_code: str) -> None:
    """Remove a cached Borsa Italiana token so it is re-fetched on next use."""
    cache_key = f"{isin}-{market_code}"
    with _borsa_token_cache_lock:
        _borsa_token_cache.pop(cache_key, None)
