"""Snapshot-hash semantic decision cache for Step-2 LLM reviews.

Token-saving optimization: before calling the Step-2 LLM, a deterministic
snapshot hash is computed over the exact LLM prompt payload inputs (symbol,
ticker snapshot, preliminary decision, backtest results, position state).
If it matches the stored hash for the symbol, the prior LLM-reviewed decision
is reused instead of making a new (expensive) LLM call.

The cached decision was originally reviewed by the LLM (it is only stored on
genuine Step-2 success paths), so it keeps ``step2_reviewed=True`` and the
real ``llm_provider``/``llm_model`` — the centralized provenance gate in
``PostDecisionManager.process_post_llm_decision`` still passes. It is marked
with ``decision_source="cache"`` for observability.

Redis unavailability never breaks the decision path: all operations are
best-effort and fall back to the normal LLM flow.
"""
import asyncio
import json
import logging
from dataclasses import asdict
from typing import Any, Dict, Optional

from src.config.settings import settings
from src.llm.cache import compute_market_hash, estimate_tokens
from src.strategies.base import Signal

logger = logging.getLogger(__name__)

_DECISION_CACHE_KEY_PREFIX = "llm:dec_cache:"

# Exceptions treated as "Redis unavailable / data problem" — never fatal.
_CACHE_ERRORS = (ConnectionError, TimeoutError, OSError, ValueError, TypeError, KeyError, AttributeError)


def build_decision_snapshot_hash(snapshot: Dict[str, Any]) -> str:
    """Compute a deterministic SHA-256 hash of the Step-2 prompt payload inputs.

    Delegates to ``compute_market_hash`` which strips volatile fields
    (timestamps, fetched_at, ...) and rounds floats to 6 significant figures,
    so essentially-identical market states produce the same hash.
    """
    return compute_market_hash(snapshot)


def _cache_key(symbol: str) -> str:
    return f"{_DECISION_CACHE_KEY_PREFIX}{symbol}"


def get_cached_decision(redis_client, symbol: str) -> Optional[Dict[str, Any]]:
    """Return the stored cache entry {snapshot_hash, signal, ...} or None.

    Best-effort: returns None on any Redis error (decision path then falls
    back to the normal LLM call).
    """
    try:
        raw = redis_client.get(_cache_key(symbol))
    except _CACHE_ERRORS as e:
        logger.warning(
            f"Decision cache read failed for {symbol}: {type(e).__name__}: {e}",
            extra={"event": "decision_cache_read_error", "symbol": symbol, "error_type": type(e).__name__},
        )
        return None
    if not raw:
        return None
    try:
        entry = json.loads(raw)
        if not isinstance(entry, dict) or "snapshot_hash" not in entry or "signal" not in entry:
            logger.warning(f"Malformed decision cache entry for {symbol}; ignoring.", extra={"event": "decision_cache_malformed", "symbol": symbol})
            return None
        return entry
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        logger.warning(f"Failed to decode decision cache entry for {symbol}: {type(e).__name__}: {e}", extra={"event": "decision_cache_decode_error", "symbol": symbol})
        return None


def estimate_snapshot_tokens(snapshot: Dict[str, Any]) -> int:
    """Estimate the token footprint of the Step-2 prompt payload snapshot.

    The snapshot is the deterministic JSON of ticker, preliminary decision,
    backtest results, combined summary, and position state — exactly the
    inputs the snapshot hash is computed over. Roughly doubles the prompt-token
    cost of a Step-2 call (JSON keys + stringified values), so it is an
    approximate guide for what a cache hit would have saved.
    """
    if not snapshot:
        return 0
    return estimate_tokens(json.dumps(snapshot, default=str))


async def record_cache_event(
    symbol: str,
    outcome: str,
    reason: Optional[str] = None,
    action: Optional[str] = None,
    model_type: Optional[str] = None,
    est_saved_tokens: float = 0.0,
) -> None:
    """Record a decision-cache hit or miss event.

    Best-effort: on any write error the decision path is unaffected. `outcome`
    is hit or miss; `reason` classifies the miss (unchanged | changed | cold |
    rebuild_failed); `action` records the cached action; `model_type` the model
    tier; `est_saved_tokens` the estimated tokens a hit would have saved.

    The insert is a blocking DB write, so it runs in a worker thread and never
    stalls the event loop.
    """
    try:
        from src.database import record_decision_cache_event
        await asyncio.to_thread(
            record_decision_cache_event,
            symbol=symbol,
            outcome=outcome,
            reason=reason,
            action=action,
            model_type=model_type,
            est_saved_tokens=est_saved_tokens,
        )
    except Exception as e:
        logger.debug(
            f"Decision cache metric write failed for {symbol}: {type(e).__name__}: {e}",
            extra={"event": "decision_cache_metric_error", "symbol": symbol, "error_type": type(e).__name__},
        )


def store_cached_decision(
    redis_client,
    symbol: str,
    snapshot_hash: str,
    signal: Signal,
    llm_provider: Optional[str],
    llm_model: Optional[str],
    ttl_seconds: Optional[int] = None,
    est_prompt_tokens: Optional[int] = None,
) -> bool:
    """Store the final LLM-reviewed decision + snapshot hash for a symbol.

    Best-effort: returns False on any Redis error (decision path unaffected).
    """
    if ttl_seconds is None:
        ttl_seconds = settings.LLM_DECISION_CACHE_TTL_SECONDS
    try:
        entry = {
            "snapshot_hash": snapshot_hash,
            "signal": asdict(signal),
            "llm_provider": llm_provider,
            "llm_model": llm_model,
            "est_prompt_tokens": est_prompt_tokens,
        }
        redis_client.setex(_cache_key(symbol), int(ttl_seconds), json.dumps(entry, default=str))
        return True
    except _CACHE_ERRORS as e:
        logger.warning(
            f"Decision cache write failed for {symbol}: {type(e).__name__}: {e}",
            extra={"event": "decision_cache_write_error", "symbol": symbol, "error_type": type(e).__name__},
        )
        return False


def invalidate_decision_cache(redis_client, symbol: str) -> bool:
    """Delete the cached decision for a symbol (position/executed-decision change).

    Best-effort: returns False on any Redis error.
    """
    try:
        redis_client.delete(_cache_key(symbol))
        return True
    except _CACHE_ERRORS as e:
        logger.warning(
            f"Decision cache invalidation failed for {symbol}: {type(e).__name__}: {e}",
            extra={"event": "decision_cache_invalidate_error", "symbol": symbol, "error_type": type(e).__name__},
        )
        return False


def cached_decision_to_signal(entry: Dict[str, Any]) -> Optional[Signal]:
    """Rebuild a reviewed Signal from a stored cache entry.

    Preserves LLM provenance: the entry was only stored on a genuine Step-2
    success path, so step2_reviewed stays True and the original real
    provider/model are restored. The signal is marked decision_source="cache".
    """
    try:
        signal = Signal.from_dict(entry["signal"])
        signal.step2_reviewed = True
        signal.decision_source = "cache"
        signal.llm_provider = entry.get("llm_provider") or signal.llm_provider
        signal.llm_model = entry.get("llm_model") or signal.llm_model
        if signal.reasoning:
            signal.reasoning = f"{signal.reasoning} (reused cached LLM decision — unchanged inputs)"
        else:
            signal.reasoning = "reused cached LLM decision — unchanged inputs"
        return signal
    except (ValueError, TypeError, KeyError, AttributeError) as e:
        logger.warning(f"Failed to rebuild cached decision signal: {type(e).__name__}: {e}", extra={"event": "decision_cache_rebuild_error"})
        return None