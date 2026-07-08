import asyncio
import logging
import random
import time
from typing import Optional

import httpx
import requests
from bs4 import BeautifulSoup

from src.config.settings import settings

logger = logging.getLogger(__name__)


class DynamicProxyRotator:
    def __init__(self):
        self.proxy_source_url = "https://free-proxy-list.net"
        self.test_url = "http://httpbin.org"
        self.valid_proxies = []
        self._last_refresh = 0.0

    def fetch_raw_proxies(self):
        """Scrapes the latest free proxies from the web."""
        try:
            response = requests.get(self.proxy_source_url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            proxies = []

            table = soup.find('div', class_='table-responsive')
            if not table:
                return []

            for row in table.find('tbody').find_all('tr'):
                cols = row.find_all('td')
                if cols:
                    ip = cols[0].text.strip()
                    port = cols[1].text.strip()
                    proxies.append(f"http://{ip}:{port}")
            return proxies
        except (requests.RequestException, ValueError, OSError) as e:
            logger.warning(f"Error fetching raw proxies: {e}")
            return []

    async def _validate_single_proxy(self, client, proxy):
        """Asynchronously tests a single proxy's viability."""
        try:
            response = await client.get(self.test_url, proxy=proxy, timeout=3.0)
            if response.status_code == 200:
                return proxy
        except (httpx.RequestError, httpx.HTTPStatusError, OSError):
            pass
        return None

    async def refresh_proxy_pool(self):
        """Fetches and tests all proxies, rebuilding the valid pool."""
        raw_proxies = self.fetch_raw_proxies()
        logger.info(f"Fetched {len(raw_proxies)} raw proxies. Validating speed and uptime...")

        async with httpx.AsyncClient() as client:
            tasks = [self._validate_single_proxy(client, proxy) for proxy in raw_proxies]
            results = await asyncio.gather(*tasks)

        self.valid_proxies = [p for p in results if p is not None]
        self._last_refresh = time.time()
        logger.info(f"Pool updated! {len(self.valid_proxies)} proxies are ready to use.")

    def get_proxy(self):
        """Returns a random valid proxy from the active pool."""
        if not self.valid_proxies:
            return None
        return random.choice(self.valid_proxies)

_dynamic_rotator = DynamicProxyRotator()


def _get_proxies() -> Optional[str]:
    """Return a random proxy string for httpx if enabled, else None."""
    if not settings.HTTP_PROXY_ENABLED:
        return None

    # Trigger background refresh if pool is empty or stale (every 30 mins)
    if not _dynamic_rotator.valid_proxies or (time.time() - _dynamic_rotator._last_refresh > 1800):
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_dynamic_rotator.refresh_proxy_pool())
        except RuntimeError:
            pass  # No event loop running

    # Randomly decide to use YF_PROXIES or dynamic rotator
    use_dynamic = False
    if _dynamic_rotator.valid_proxies:
        if settings.HTTP_PROXIES:
            use_dynamic = random.choice([True, False])
        else:
            use_dynamic = True
    elif not settings.HTTP_PROXIES:
        return None

    if use_dynamic:
        proxy = _dynamic_rotator.get_proxy()
        if proxy:
            logger.debug(f"Using dynamic proxy: {proxy}")
            return proxy

    if settings.HTTP_PROXIES:
        proxy = random.choice(settings.HTTP_PROXIES)
        logger.debug(f"Using static proxy: {proxy}")
        return proxy

    return None
