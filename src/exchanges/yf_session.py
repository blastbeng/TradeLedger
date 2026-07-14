import atexit
import collections
import concurrent.futures
import logging
import random
import threading
import time

import yfinance as yf

from src.config.settings import settings

logger = logging.getLogger(__name__)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# Dedicated thread pool for yf.download with a hard timeout wrapper.
# This prevents yf.download from hanging threads indefinitely when
# curl_cffi's own timeout doesn't fire.
_yf_download_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=2, thread_name_prefix="yf-download-timeout"
)

# Ensure the executor is shut down cleanly on program exit
atexit.register(_yf_download_executor.shutdown, wait=False)


def _yf_download_with_timeout(symbols, **kwargs):
    """Run yf.download with a hard 30-second timeout to prevent indefinite hangs.

    If the timeout fires, the underlying thread is abandoned (it will eventually
    die on its own due to the curl_cffi HTTP timeout). Returns None on timeout.
    """
    future = _yf_download_executor.submit(yf.download, symbols, **kwargs)
    try:
        return future.result(timeout=30)
    except concurrent.futures.TimeoutError:
        count = len(symbols) if isinstance(symbols, list) else 1
        logger.warning(f"yf.download timed out after 30s for {count} symbols")
        return None


# --- yfinance Circuit Breaker ---
_yf_error_timestamps: collections.deque = collections.deque()
_yf_circuit_open_until = 0.0
_yf_lock = threading.Lock()

YF_MAX_ERRORS = 20
YF_CIRCUIT_COOLDOWN = 300  # 5 minutes
YF_ERROR_WINDOW_SECONDS = 300  # 5 minutes

_yf_session_cache = None
_yf_session_lock = threading.Lock()
_yf_session_created = False  # True only after successful creation or confirmed ImportError

def _invalidate_yf_session():
    """Invalidate the cached yfinance session so it is recreated on next use."""
    global _yf_session_cache, _yf_session_created
    with _yf_session_lock:
        if _yf_session_created and _yf_session_cache is not None:
            logger.info("Invalidating yfinance session due to repeated errors.")
            _yf_session_cache = None
            _yf_session_created = False

def _check_yf_circuit() -> bool:
    """Return True if the circuit is open (yfinance should be skipped)."""
    with _yf_lock:
        return time.time() < _yf_circuit_open_until

def _record_yf_error():
    """Record a yfinance error and potentially trip the circuit breaker."""
    global _yf_circuit_open_until
    with _yf_lock:
        now = time.time()
        # Remove timestamps outside the window
        while _yf_error_timestamps and _yf_error_timestamps[0] <= now - YF_ERROR_WINDOW_SECONDS:
            _yf_error_timestamps.popleft()

        _yf_error_timestamps.append(now)
        error_count = len(_yf_error_timestamps)

        # Recreate session after a few errors to try to recover from invalid session issues
        if error_count == 5:
            _invalidate_yf_session()

        if error_count >= YF_MAX_ERRORS:
            if _yf_circuit_open_until < now:
                logger.error(f"yfinance circuit breaker tripped due to {error_count} errors in the last {YF_ERROR_WINDOW_SECONDS}s. Blocking yfinance calls for {YF_CIRCUIT_COOLDOWN}s.")
            _yf_circuit_open_until = now + YF_CIRCUIT_COOLDOWN
            _invalidate_yf_session()

def _reset_yf_circuit():
    """Remove the oldest error from the sliding window after a successful call."""
    with _yf_lock:
        if _yf_error_timestamps:
            _yf_error_timestamps.popleft()


class YFinanceRateLimiter:
    """Sliding window rate limiter for yfinance requests."""
    def __init__(self, max_requests: int, window_seconds: int, use_yf_settings: bool = True):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.use_yf_settings = use_yf_settings
        self._timestamps: collections.deque = collections.deque()
        self._lock = threading.Lock()

    def acquire(self):
        if self.use_yf_settings:
            # Read live settings to allow runtime reload without restart
            if not settings.YF_RATE_LIMIT_ENABLED or settings.YF_RATE_LIMIT_MAX_REQUESTS <= 0:
                return
            window = settings.YF_RATE_LIMIT_WINDOW_SECONDS
            max_req = settings.YF_RATE_LIMIT_MAX_REQUESTS
        else:
            if self.max_requests <= 0:
                return
            window = self.window_seconds
            max_req = self.max_requests

        with self._lock:
            now = time.time()
            # Remove timestamps outside the window
            while self._timestamps and self._timestamps[0] <= now - window:
                self._timestamps.popleft()
            if len(self._timestamps) >= max_req:
                # Rate limit exceeded — fail fast instead of blocking.
                # Sleeping would hold the thread pool worker hostage and cause
                # asyncio.wait_for timeouts in the engine.  Raising lets the
                # caller fall back to the database immediately.
                raise ConnectionError("rate limit exceeded")
            self._timestamps.append(time.time())


_yf_rate_limiter = YFinanceRateLimiter(
    max_requests=settings.YF_RATE_LIMIT_MAX_REQUESTS,
    window_seconds=settings.YF_RATE_LIMIT_WINDOW_SECONDS,
)

class YFinance401Filter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        if "HTTP Error 401" in msg or "Unauthorized" in msg:
            _record_yf_error()
            if _check_yf_circuit():
                return False  # suppress log spam when circuit is open
        return True

# Attach filter to yfinance logger
logging.getLogger("yfinance").addFilter(YFinance401Filter())

def _get_yf_session():
    """Return a cached curl_cffi session that impersonates Chrome for yfinance requests.

    Yahoo Finance increasingly blocks requests that don't look like a real
    browser.  curl_cffi can impersonate Chrome's TLS fingerprint, which
    avoids 401/429 responses.  The session is created once and reused for
    all subsequent calls to avoid the overhead of repeated session creation.
    """
    global _yf_session_cache, _yf_session_created

    # Fast path: return cached session without acquiring lock
    if _yf_session_created:
        return _yf_session_cache

    with _yf_session_lock:
        # Double-check after acquiring lock
        if _yf_session_created:
            return _yf_session_cache

        try:
            from curl_cffi import requests as curl_requests

            proxies = None
            if settings.HTTP_PROXY_ENABLED and settings.HTTP_PROXIES:
                proxy = random.choice(settings.HTTP_PROXIES)
                proxies = {"http": proxy, "https": proxy}
                logger.debug(f"Using proxy for yfinance: {proxy}")

            class YFinanceSessionWrapper(curl_requests.Session):
                def request(self, *args, **kwargs):
                    if _check_yf_circuit():
                        raise ConnectionError("yfinance circuit breaker is open")
                    _yf_rate_limiter.acquire()
                    # Enforce a timeout to prevent indefinite hangs
                    kwargs.setdefault('timeout', 15.0)
                    try:
                        response = super().request(*args, **kwargs)
                    except Exception:
                        # Record error to potentially invalidate session or trip circuit
                        _record_yf_error()
                        raise
                    if response.status_code in (401, 403, 429):
                        _record_yf_error()
                    else:
                        _reset_yf_circuit()
                    return response

            _yf_session_cache = YFinanceSessionWrapper(impersonate="chrome", proxies=proxies)
            _yf_session_created = True
            return _yf_session_cache
        except ImportError:
            logger.warning("curl_cffi not installed – yfinance requests may be blocked.")
            _yf_session_cache = None
            _yf_session_created = True
            return None
        except (ImportError, RuntimeError, OSError, AttributeError) as e:
            logger.warning(f"Failed to create curl_cffi session: {e}")
            # Don't set _yf_session_created so we retry on the next call
            return None
