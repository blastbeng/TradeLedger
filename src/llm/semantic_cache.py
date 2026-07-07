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

import requests
import uuid
from typing import Optional, List
from src.config.settings import settings

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

        if self.enabled:
            self._init_collection()

    def _init_collection(self):
        """Initializes or retrieves the ChromaDB collection."""
        try:
            # Check if collection exists
            resp = requests.get(f"{self.chromadb_host}/api/v1/collections/{self.collection_name}", timeout=5)
            if resp.status_code == 200:
                self.collection_id = resp.json().get("id")
            elif resp.status_code == 404:
                # Create collection with cosine distance
                resp = requests.post(
                    f"{self.chromadb_host}/api/v1/collections",
                    json={"name": self.collection_name, "metadata": {"hnsw:space": "cosine"}},
                    timeout=5
                )
                resp.raise_for_status()
                self.collection_id = resp.json().get("id")
            else:
                resp.raise_for_status()
        except Exception as e:
            logger.warning(f"Semantic Cache: Failed to initialize ChromaDB collection: {e}. Disabling semantic cache.")
            self.enabled = False

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Generates an embedding for the given text using the llama.cpp server."""
        try:
            resp = requests.post(
                f"{self.embedding_url}/embeddings",
                json={"model": self.embedding_model, "input": text},
                timeout=10
            )
            resp.raise_for_status()
            return resp.json()["data"][0]["embedding"]
        except Exception as e:
            logger.warning(f"Semantic Cache: Failed to get embedding: {e}.")
            return None

    def query(self, prompt: str) -> Optional[str]:
        """Queries the semantic cache for a matching prompt."""
        if not self.enabled or not self.collection_id:
            return None

        generalized_prompt, ticker = generalize_prompt(prompt)
        embedding = self.get_embedding(generalized_prompt)
        if not embedding:
            return None

        try:
            resp = requests.post(
                f"{self.chromadb_host}/api/v1/collections/{self.collection_id}/query",
                json={"query_embeddings": [embedding], "n_results": 1},
                timeout=10
            )
            resp.raise_for_status()
            data = resp.json()

            if data.get("distances") and data["distances"][0]:
                distance = data["distances"][0][0]
                if distance <= self.distance_threshold:
                    cached_response = data["documents"][0][0]
                    metadata = data["metadatas"][0][0]
                    # Reconstruct response with current ticker if needed
                    if ticker and metadata.get("ticker"):
                        return reconstruct_response(cached_response, ticker, prompt)
                    return cached_response
        except Exception as e:
            logger.warning(f"Semantic Cache: Failed to query ChromaDB: {e}.")

        return None

    def add(self, prompt: str, response: str):
        """Adds a prompt and its response to the semantic cache."""
        if not self.enabled or not self.collection_id:
            return

        generalized_prompt, ticker = generalize_prompt(prompt)
        embedding = self.get_embedding(generalized_prompt)
        if not embedding:
            return

        try:
            metadata = {
                "ticker": ticker or "",
                "original_prompt": prompt,
                "cached_response": response
            }
            item_id = str(uuid.uuid4())
            resp = requests.post(
                f"{self.chromadb_host}/api/v1/collections/{self.collection_id}/add",
                json={
                    "embeddings": [embedding],
                    "documents": [response],
                    "metadatas": [metadata],
                    "ids": [item_id]
                },
                timeout=10
            )
            resp.raise_for_status()
        except Exception as e:
            logger.warning(f"Semantic Cache: Failed to add to ChromaDB: {e}.")

# Singleton instance
semantic_cache_client = SemanticCacheClient()
