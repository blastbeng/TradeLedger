import re
import logging
import requests
import uuid
import hashlib
import threading
import time
import queue
from concurrent.futures import Future
from typing import Tuple, Optional, List
from src.config.settings import settings
from src.utils.symbol_utils import is_btp_isin

logger = logging.getLogger(__name__)

class LlamaCppExecutor:
    """Serializes and prioritizes requests to a single-threaded llama.cpp server."""
    def __init__(self):
        self._queue = queue.PriorityQueue()
        self._seq = 0
        self._seq_lock = threading.Lock()
        self._worker = threading.Thread(target=self._run, daemon=True, name="llamacpp-worker")
        self._worker.start()

    def _run(self):
        while True:
            task = self._queue.get()
            if task is None:
                break
            priority, seq, future, func, args, kwargs = task
            try:
                result = func(*args, **kwargs)
                if future:
                    future.set_result(result)
            except Exception as e:
                if future:
                    future.set_exception(e)
            finally:
                self._queue.task_done()

    def submit(self, func, *args, priority=10, **kwargs):
        """Submit a task. Lower priority numbers run first."""
        with self._seq_lock:
            seq = self._seq
            self._seq += 1
        future = Future()
        self._queue.put((priority, seq, future, func, args, kwargs))
        return future

llamacpp_executor = LlamaCppExecutor()

# Regex to find tickers ending with .MI (or other configured suffixes)
TICKER_REGEX = re.compile(r'\b((?:[A-Z0-9]{1,6}(?:\.[A-Z]{1,3})?)|(?:IT[A-Z0-9]{10}))\b')

# Common financial, technical, and indicator terms to exclude from ticker matching
EXCLUDED_TERMS = {
    "MACD", "RSI", "ADX", "EMA", "OBV", "MFI", "CCI", "SAR", "SMA", "WMA", "VWAP", "ATR", "ROC",
    "BUY", "SELL", "HOLD", "LONG", "SHORT", "STOP", "LIMIT", "MARKET",
    "EUR", "USD", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD",
    "BTP", "ETF", "BOT", "CTD", "YTD", "Q1", "Q2", "Q3", "Q4", "H1", "H2",
    "PE", "EPS", "ROI", "ROE", "ROA", "GDP", "CPI", "FED", "ECB", "IPO",
    "API", "JSON", "XML", "HTTP", "URL", "ID", "UUID", "TICKER"
}

def generalize_prompt(prompt: str, symbol: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """
    Replaces specific stock tickers in the prompt with a generic [TICKER] placeholder.

    Returns:
        A tuple containing (generalized_prompt, extracted_ticker).
        If no ticker is found, or if it's a BTP/multi-ticker context, extracted_ticker will be None.
    """
    # Never generalize BTP ISINs, as each has unique coupon/maturity/yield
    if symbol and is_btp_isin(symbol):
        return prompt, None

    # Prioritize the explicitly provided symbol if it exists in the prompt
    if symbol and symbol in prompt:
        generalized_prompt = prompt.replace(symbol, "[TICKER]")
        return generalized_prompt, symbol

    # Fallback to regex if no symbol provided or not found
    matches = TICKER_REGEX.findall(prompt)

    # Filter out common indicator abbreviations, currencies, and other non-ticker terms
    valid_tickers = [m for m in matches if m not in EXCLUDED_TERMS]
    unique_tickers = set(valid_tickers)

    # Do not generalize if there are multiple different tickers (e.g., "ENI vs ENEL")
    if len(unique_tickers) != 1:
        return prompt, None

    ticker = valid_tickers[0]
    if is_btp_isin(ticker):
        return prompt, None

    generalized_prompt = prompt.replace(ticker, "[TICKER]")
    return generalized_prompt, ticker

def reconstruct_response(cached_response: str, current_ticker: str) -> str:
    """
    Reconstructs the cached response by injecting the current ticker context
    back into the template.
    """
    if not current_ticker:
        return cached_response

    # Defense in depth: never reconstruct for BTP ISINs
    if is_btp_isin(current_ticker):
        return cached_response

    # Replace the placeholder with the current ticker
    return cached_response.replace("[TICKER]", current_ticker)

class SemanticCacheClient:
    """Handles ChromaDB and embedding server interactions for semantic caching."""
    def __init__(self):
        self.enabled = settings.SEMANTIC_CACHE_ENABLED
        self.embedding_url = settings.EMBEDDING_MODEL_BASE_URL
        self.embedding_model = settings.EMBEDDING_MODEL_NAME
        self.chromadb_host = settings.CHROMADB_HOST
        self.collection_name = settings.CHROMADB_COLLECTION_NAME
        self.distance_threshold = settings.SEMANTIC_CACHE_DISTANCE_THRESHOLD
        self.distance_thresholds = {
            "pause_resume": getattr(settings, 'SEMANTIC_CACHE_THRESHOLD_PAUSE_RESUME', 0.10),
            "strategy": getattr(settings, 'SEMANTIC_CACHE_THRESHOLD_STRATEGY', 0.15),
            "stock_selection": getattr(settings, 'SEMANTIC_CACHE_THRESHOLD_STOCK_SELECTION', 0.20),
            "default": self.distance_threshold
        }
        self.max_entries = getattr(settings, 'SEMANTIC_CACHE_MAX_ENTRIES', 10000)
        self.collection_id = None
        self._initialized = False
        self._init_failed = False

        # Start background cleanup thread
        self._cleanup_thread = threading.Thread(target=self._periodic_cleanup, daemon=True, name="semantic-cache-cleanup")
        self._cleanup_thread.start()

    def _ensure_initialized(self):
        """Lazily initialize the ChromaDB collection on first query/add."""
        if self._initialized and not self._init_failed:
            return
        self._initialized = True
        if self.enabled:
            self._init_collection()

    def _init_collection(self):
        """Initializes or retrieves the ChromaDB collection. Retries on failure."""
        base_url = f"{self.chromadb_host}/api/v2/tenants/default_tenant/databases/default_database/collections"
        collection_url = f"{base_url}/{self.collection_name}"
        logger.info(f"Semantic Cache: Initializing ChromaDB collection at {collection_url}")
        try:
            # Check if collection exists
            resp = requests.get(collection_url, timeout=5)
            if resp.status_code == 200:
                self.collection_id = resp.json().get("id")
                logger.info(f"Semantic Cache: Found existing collection with ID: {self.collection_id}")
            elif resp.status_code == 404:
                # Create collection with cosine distance
                logger.info(f"Semantic Cache: Collection not found, creating new one at {base_url}")
                resp = requests.post(
                    base_url,
                    json={"name": self.collection_name, "metadata": {"hnsw:space": "cosine"}},
                    timeout=5
                )
                resp.raise_for_status()
                self.collection_id = resp.json().get("id")
                logger.info(f"Semantic Cache: Created new collection with ID: {self.collection_id}")
            else:
                resp.raise_for_status()
            self._init_failed = False
        except Exception as e:
            logger.warning(f"Semantic Cache: Failed to initialize ChromaDB collection: {e}. Will retry on next call.", exc_info=True)
            self._init_failed = True
            self._initialized = False  # allow retry

    def _periodic_cleanup(self):
        """Periodically cleans up expired and excess cache entries."""
        while True:
            # Wait 1 hour between cleanup cycles
            time.sleep(3600)
            try:
                self._ensure_initialized()
                if not self.enabled or not self.collection_id:
                    continue
                self.cleanup_expired()
            except Exception as e:
                logger.warning(f"Semantic Cache: Periodic cleanup failed: {e}", exc_info=True)

    def get_embedding(self, text: str, timeout: int = 300, priority: int = 10) -> Optional[List[float]]:
        """Generates an embedding for the given text using the llama.cpp server."""
        embedding_url = f"{self.embedding_url}/embeddings"

        # Truncate to the first 500 characters to preserve the core semantic intent
        # of the prompt (usually the instruction and symbol context) without diluting
        # the signal by averaging embeddings of diverse sections (e.g., market data, news).
        text_to_embed = text[:500]

        logger.info(f"Semantic Cache: get_embedding called for text (len={len(text_to_embed)})")

        if not text_to_embed:
            return None

        logger.info(f"Semantic Cache: Generating embedding at {embedding_url}...")
        start_time = time.time()
        resp = None
        try:
            payload = {"model": self.embedding_model, "input": [text_to_embed]}
            if not self.embedding_model:
                logger.warning("Semantic Cache: EMBEDDING_MODEL_NAME is not set. Sending request without model name.")
                payload.pop("model", None)
            logger.info(f"Semantic Cache: Embedding payload: {payload}")

            def _do_request():
                return requests.post(
                    embedding_url,
                    json=payload,
                    timeout=timeout
                )

            future = llamacpp_executor.submit(_do_request, priority=priority)
            resp = future.result(timeout=timeout + 60)  # Wait slightly longer than HTTP timeout

            logger.info(f"Semantic Cache: Embedding response status: {resp.status_code}, text: {resp.text[:500]}")
            resp.raise_for_status()
            logger.info(f"Semantic Cache: Text embedded successfully in {time.time() - start_time:.2f}s.")
            return resp.json()["data"][0]["embedding"]
        except requests.exceptions.Timeout as e:
            logger.error(f"Semantic Cache: Embedding request timed out after {timeout}s: {e}", exc_info=True)
            return None
        except Exception as e:
            resp_text = getattr(resp, 'text', 'N/A')
            logger.warning(f"Semantic Cache: Failed to get embedding: {e}. Response body: {resp_text}", exc_info=True)
            return None

    def query(self, prompt: str, symbol: Optional[str] = None, model_type: str = "actuator", cache_version: Optional[str] = None, prompt_category: str = "default", market_hash: Optional[str] = None) -> Optional[str]:
        """Queries the semantic cache for a matching prompt."""
        self._ensure_initialized()
        if not self.enabled or not self.collection_id:
            logger.debug("Semantic Cache: Skipping query (disabled or no collection).")
            return None

        # Never query cache for BTP ISINs, as each has unique characteristics
        if symbol and is_btp_isin(symbol):
            logger.debug("Semantic Cache: Skipping query for BTP ISIN.")
            return None

        generalized_prompt, ticker = generalize_prompt(prompt, symbol)
        logger.debug(f"Semantic Cache: Querying cache for prompt (ticker={ticker}, category={prompt_category})...")
        embedding = self.get_embedding(generalized_prompt, timeout=120, priority=1)
        if not embedding:
            logger.debug("Semantic Cache: No embedding generated, skipping query.")
            return None

        try:
            query_url = f"{self.chromadb_host}/api/v2/tenants/default_tenant/databases/default_database/collections/{self.collection_id}/query"
            logger.debug(f"Semantic Cache: Querying ChromaDB at {query_url}...")

            where_clause = {
                "$and": [
                    {"model_type": model_type},
                    {"cache_version": cache_version or ""},
                    {"prompt_category": prompt_category},
                    {"market_hash": market_hash or ""}
                ]
            }

            resp = requests.post(
                query_url,
                json={
                    "query_embeddings": [embedding], 
                    "n_results": 1,
                    "where": where_clause
                },
                timeout=10
            )
            logger.debug(f"Semantic Cache: ChromaDB query response status: {resp.status_code}")
            resp.raise_for_status()
            data = resp.json()

            if data.get("distances") and data["distances"][0]:
                distance = data["distances"][0][0]
                threshold = self.distance_thresholds.get(prompt_category, self.distance_thresholds["default"])
                logger.debug(f"Semantic Cache: Query distance={distance:.4f}, threshold={threshold}")
                if distance <= threshold:
                    logger.info(f"Semantic Cache: Hit! Distance={distance:.4f} <= {threshold}")
                    cached_response = data["documents"][0][0]
                    metadata = data["metadatas"][0][0]
                    # Reconstruct response with current ticker if needed
                    if ticker and metadata.get("ticker"):
                        return reconstruct_response(cached_response, ticker)
                    return cached_response
                else:
                    logger.debug(f"Semantic Cache: Miss. Distance={distance:.4f} > {threshold}")
            else:
                logger.debug("Semantic Cache: Query returned no distances.")
        except Exception as e:
            logger.warning(f"Semantic Cache: Failed to query ChromaDB: {e}.", exc_info=True)

        return None

    def add(self, prompt: str, response: str, symbol: Optional[str] = None, model_type: str = "actuator", cache_version: Optional[str] = None, prompt_category: str = "default", market_hash: Optional[str] = None):
        """Adds a prompt and its response to the semantic cache."""
        self._ensure_initialized()
        if not self.enabled or not self.collection_id:
            logger.debug("Semantic Cache: Skipping add (disabled or no collection).")
            return

        # Never cache BTP ISINs, as each has unique characteristics
        if symbol and is_btp_isin(symbol):
            logger.debug("Semantic Cache: Skipping add for BTP ISIN.")
            return

        generalized_prompt, ticker = generalize_prompt(prompt, symbol)
        logger.debug(f"Semantic Cache: Adding to cache (ticker={ticker}, category={prompt_category})...")
        embedding = self.get_embedding(generalized_prompt, timeout=300)
        if not embedding:
            return

        try:
            # Generalize the response by replacing the ticker with [TICKER]
            generalized_response = response.replace(ticker, "[TICKER]") if ticker else response

            metadata = {
                "ticker": ticker or "",
                "original_prompt": prompt,
                "cached_at": str(time.time()),
                "model_type": model_type,
                "cache_version": cache_version or "",
                "prompt_category": prompt_category,
                "market_hash": market_hash or ""
            }

            # Use a deterministic ID to prevent duplicate entries for the same prompt
            id_source = f"{generalized_prompt}:{model_type}:{cache_version or ''}"
            item_id = hashlib.sha256(id_source.encode()).hexdigest()

            # Use upsert to add or update the entry if it already exists
            add_url = f"{self.chromadb_host}/api/v2/tenants/default_tenant/databases/default_database/collections/{self.collection_id}/upsert"
            logger.debug(f"Semantic Cache: Upserting to ChromaDB at {add_url}. Prompt len={len(generalized_prompt)}, Response len={len(generalized_response)}")
            resp = requests.post(
                add_url,
                json={
                    "embeddings": [embedding],
                    "documents": [generalized_response],  # Store the generalized response
                    "metadatas": [metadata],
                    "ids": [item_id]
                },
                timeout=10
            )
            logger.debug(f"Semantic Cache: ChromaDB add response status: {resp.status_code}")
            resp.raise_for_status()
            logger.info(f"Semantic Cache: Successfully added to ChromaDB with ID {item_id}.")
        except Exception as e:
            logger.warning(f"Semantic Cache: Failed to add to ChromaDB: {e}.", exc_info=True)

    def cleanup_expired(self, max_age_seconds: int = 86400):
        """Remove cache entries older than max_age_seconds and enforce max size limit."""
        if not self.enabled or not self.collection_id:
            return
        try:
            # ChromaDB v2 API: get all entries with metadata, filter by age
            cutoff = time.time() - max_age_seconds
            resp = requests.post(
                f"{self.chromadb_host}/api/v2/tenants/default_tenant/databases/default_database/collections/{self.collection_id}/get",
                json={"include": ["metadatas", "ids"]},
                timeout=10
            )
            resp.raise_for_status()
            data = resp.json()
            ids_to_delete = []
            metadatas = data.get("metadatas", [])
            ids = data.get("ids", [])
            logger.debug(f"Semantic Cache: Retrieved {len(metadatas)} entries for cleanup check.")

            # 1. Find expired entries
            for i, meta in enumerate(metadatas):
                ts = meta.get("cached_at")
                if ts and float(ts) < cutoff:
                    ids_to_delete.append(ids[i])

            # 2. Enforce max size limit by finding oldest entries if over limit
            if len(metadatas) > self.max_entries:
                logger.info(f"Semantic Cache: Collection size {len(metadatas)} exceeds max {self.max_entries}. Pruning oldest entries.")
                # Create list of (timestamp, id) for entries not already marked for deletion
                remaining_entries = []
                for i, meta in enumerate(metadatas):
                    if ids[i] not in ids_to_delete:
                        ts = meta.get("cached_at")
                        if ts:
                            remaining_entries.append((float(ts), ids[i]))

                # Sort by timestamp ascending (oldest first)
                remaining_entries.sort(key=lambda x: x[0])

                # Calculate how many to delete to get under the limit
                num_to_prune = len(remaining_entries) - self.max_entries
                if num_to_prune > 0:
                    for i in range(num_to_prune):
                        ids_to_delete.append(remaining_entries[i][1])

            if ids_to_delete:
                # Deduplicate ids
                ids_to_delete = list(set(ids_to_delete))
                resp = requests.post(
                    f"{self.chromadb_host}/api/v2/tenants/default_tenant/databases/default_database/collections/{self.collection_id}/delete",
                    json={"ids": ids_to_delete},
                    timeout=10
                )
                resp.raise_for_status()
                logger.info(f"Semantic Cache: Cleaned up {len(ids_to_delete)} expired/excess entries.")
        except Exception as e:
            logger.warning(f"Semantic Cache: Cleanup failed: {e}", exc_info=True)

_semantic_cache_client: Optional[SemanticCacheClient] = None

def get_semantic_cache_client() -> SemanticCacheClient:
    """Return the singleton SemanticCacheClient, initializing lazily on first use."""
    global _semantic_cache_client
    if _semantic_cache_client is None:
        _semantic_cache_client = SemanticCacheClient()
    return _semantic_cache_client
