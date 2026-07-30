"""
In-memory caching — cachetools instead of Redis.

Design decision: at portfolio/single-user scale, an external Redis instance
is operational overhead with no real benefit — cachetools' TTLCache gives
the same "expire after N seconds, evict least-recently-used when full"
behavior in-process, with zero extra infrastructure to run or deploy. If
this ever needs to scale to multiple app instances sharing one cache,
that's the point to introduce Redis — not before.
"""

from __future__ import annotations

import hashlib
from typing import Any, Callable, Optional

from cachetools import TTLCache

_query_cache: Optional[TTLCache] = None


def get_cache(maxsize: int = 256, ttl_seconds: int = 3600) -> TTLCache:
    """Process-wide singleton cache — reused across Streamlit reruns via st.cache_resource by the caller."""
    global _query_cache
    if _query_cache is None:
        _query_cache = TTLCache(maxsize=maxsize, ttl=ttl_seconds)
    return _query_cache


def cache_key(*parts: Any) -> str:
    """Stable hash key from arbitrary string/number parts (company_id, query text, top_k, etc)."""
    raw = "||".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def cached_call(cache: TTLCache, key: str, compute_fn: Callable[[], Any]) -> tuple[Any, bool]:
    """
    Returns (value, was_cache_hit). Caller supplies compute_fn to only pay
    the cost of the expensive call (embedding, retrieval, LLM generation)
    on a miss.
    """
    if key in cache:
        return cache[key], True
    value = compute_fn()
    cache[key] = value
    return value, False
