import re
import logging
import requests
import uuid
import threading
import time
from typing import Tuple, Optional, List
from src.config.settings import settings

logger = logging.getLogger(__name__)

# Regex to find tickers ending with .MI (or other configured suffixes)
TICKER_REGEX = re.compile(r'\b((?:[A-Z0-9]{1,6}(?:\.[A-Z]{1,3})?)|(?:IT[A-Z0-9]{10}))\b')

def generalize_prompt(prompt: str, symbol: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """
    Replaces specific stock tickers in the prompt with a generic [TICKER] placeholder.

    Returns:
        A tuple containing (generalized_prompt, extracted_ticker).
        If no ticker is found, extracted_ticker will be None.
    """
    # Prioritize the explicitly provided symbol if it exists in the prompt
    if symbol and symbol in prompt:
        generalized_prompt = prompt.replace(symbol, "[TICKER]")
        return generalized_prompt, symbol

    # Fallback to regex if no symbol provided or not found
    match = TICKER_REGEX.search(prompt)
    if not match:
        return prompt, None

    ticker = match.group(1)
    generalized_prompt = prompt.replace(ticker, "[TICKER]")
    return generalized_prompt, ticker

def reconstruct_response(cached_response: str, current_ticker: str) -> str:
    """
    Reconstructs the cached response by injecting the current ticker context
    back into the template.
    """
    if not current_ticker:
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
        self.collection_id = None
        self._embedding_lock = threading.Lock()  # Serialize embedding requests
        self._initialized = False
        self._init_failed = False

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
        try:
            # Check if collection exists
            resp = requests.get(collection_url, timeout=5)
            if resp.status_code == 200:
                self.collection_id = resp.json().get("id")
            elif resp.status_code == 404:
                # Create collection with cosine distance
                resp = requests.post(
                    base_url,
                    json={"name": self.collection_name, "metadata": {"hnsw:space": "cosine"}},
                    timeout=5
                )
                resp.raise_for_status()
                self.collection_id = resp.json().get("id")
            else:
                resp.raise_for_status()
            self._init_failed = False
        except Exception as e:
            logger.warning(f"Semantic Cache: Failed to initialize ChromaDB collection: {e}. Will retry on next call.")
            self._init_failed = True
            self._initialized = False  # allow retry

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Generates an embedding for the given text using the llama.cpp server."""
        logger.debug(f"Semantic Cache: Generating embedding for text (len={len(text)})...")
        # Use a lock to ensure only one embedding request is processed at a time
        # (RPi5 has limited resources and concurrent requests may cause timeouts)
        with self._embedding_lock:
            try:
                resp = requests.post(
                    f"{self.embedding_url}/embeddings",
                    json={"model": self.embedding_model, "input": text},
                    timeout=120  # Increased timeout for RPi5
                )
                resp.raise_for_status()
                logger.debug("Semantic Cache: Embedding generated successfully.")
                return resp.json()["data"][0]["embedding"]
            except Exception as e:
                logger.warning(f"Semantic Cache: Failed to get embedding: {e}.")
                return None

    def query(self, prompt: str, symbol: Optional[str] = None) -> Optional[str]:
        """Queries the semantic cache for a matching prompt."""
        self._ensure_initialized()
        if not self.enabled or not self.collection_id:
            logger.debug("Semantic Cache: Skipping query (disabled or no collection).")
            return None

        generalized_prompt, ticker = generalize_prompt(prompt, symbol)
        logger.debug(f"Semantic Cache: Querying cache for prompt (ticker={ticker})...")
        embedding = self.get_embedding(generalized_prompt)
        if not embedding:
            logger.debug("Semantic Cache: No embedding generated, skipping query.")
            return None

        try:
            logger.debug("Semantic Cache: Querying ChromaDB...")
            resp = requests.post(
                f"{self.chromadb_host}/api/v2/tenants/default_tenant/databases/default_database/collections/{self.collection_id}/query",
                json={"query_embeddings": [embedding], "n_results": 1},
                timeout=10
            )
            resp.raise_for_status()
            data = resp.json()

            if data.get("distances") and data["distances"][0]:
                distance = data["distances"][0][0]
                if distance <= self.distance_threshold:
                    logger.info(f"Semantic Cache: Hit! Distance={distance:.4f} <= {self.distance_threshold}")
                    cached_response = data["documents"][0][0]
                    metadata = data["metadatas"][0][0]
                    # Reconstruct response with current ticker if needed
                    if ticker and metadata.get("ticker"):
                        return reconstruct_response(cached_response, ticker)
                    return cached_response
                else:
                    logger.debug(f"Semantic Cache: Miss. Distance={distance:.4f} > {self.distance_threshold}")
        except Exception as e:
            logger.warning(f"Semantic Cache: Failed to query ChromaDB: {e}.")

        return None

    def add(self, prompt: str, response: str, symbol: Optional[str] = None):
        """Adds a prompt and its response to the semantic cache."""
        self._ensure_initialized()
        if not self.enabled or not self.collection_id:
            logger.debug("Semantic Cache: Skipping add (disabled or no collection).")
            return

        generalized_prompt, ticker = generalize_prompt(prompt, symbol)
        logger.debug(f"Semantic Cache: Adding to cache (ticker={ticker})...")
        embedding = self.get_embedding(generalized_prompt)
        if not embedding:
            return

        try:
            # Generalize the response by replacing the ticker with [TICKER]
            generalized_response = response.replace(ticker, "[TICKER]") if ticker else response

            metadata = {
                "ticker": ticker or "",
                "original_prompt": prompt,
                "cached_at": str(time.time())
            }
            item_id = str(uuid.uuid4())
            resp = requests.post(
                f"{self.chromadb_host}/api/v2/tenants/default_tenant/databases/default_database/collections/{self.collection_id}/add",
                json={
                    "embeddings": [embedding],
                    "documents": [generalized_response],  # Store the generalized response
                    "metadatas": [metadata],
                    "ids": [item_id]
                },
                timeout=10
            )
            resp.raise_for_status()
            logger.debug("Semantic Cache: Successfully added to ChromaDB.")
        except Exception as e:
            logger.warning(f"Semantic Cache: Failed to add to ChromaDB: {e}.")

    def cleanup_expired(self, max_age_seconds: int = 86400):
        """Remove cache entries older than max_age_seconds."""
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
            for i, meta in enumerate(data.get("metadatas", [])):
                ts = meta.get("cached_at")
                if ts and float(ts) < cutoff:
                    ids_to_delete.append(data["ids"][i])
            if ids_to_delete:
                resp = requests.post(
                    f"{self.chromadb_host}/api/v2/tenants/default_tenant/databases/default_database/collections/{self.collection_id}/delete",
                    json={"ids": ids_to_delete},
                    timeout=10
                )
                resp.raise_for_status()
                logger.info(f"Semantic Cache: Cleaned up {len(ids_to_delete)} expired entries.")
        except Exception as e:
            logger.warning(f"Semantic Cache: Cleanup failed: {e}")

_semantic_cache_client: Optional[SemanticCacheClient] = None

def get_semantic_cache_client() -> SemanticCacheClient:
    """Return the singleton SemanticCacheClient, initializing lazily on first use."""
    global _semantic_cache_client
    if _semantic_cache_client is None:
        _semantic_cache_client = SemanticCacheClient()
    return _semantic_cache_client
