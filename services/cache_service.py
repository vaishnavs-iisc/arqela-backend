"""
Cache service — wraps Redis operations for semantic caching and STM.
This is the only module that talks to Redis outside of the graph nodes.
"""
import json
import logging
from typing import Optional

import redis

from config import config

logger = logging.getLogger("CacheService")

# Shared Redis client singleton
_client: Optional[redis.Redis] = None


def get_client() -> redis.Redis:
    """Return (and lazily create) the shared Redis client."""
    global _client
    if _client is None:
        if config.REDIS_URL:
            _client = redis.Redis.from_url(
                config.REDIS_URL,
                decode_responses=True,
            )
        else:
            _client = redis.Redis(
                host=config.REDIS_HOST,
                port=config.REDIS_PORT,
                db=0,
                decode_responses=True,
            )
    return _client


# ---------------------------------------------------------------------------
# Semantic cache (research agent)
# ---------------------------------------------------------------------------

def cache_set(query: str, report: str, embedding: list, ttl: int = 86400) -> None:
    """Store a query→report pair with its embedding vector in Redis."""
    client = get_client()
    cache_key = f"cache:{hash(query)}"
    payload = {"query": query, "report": report, "embedding": embedding}
    client.set(cache_key, json.dumps(payload), ex=ttl)
    logger.debug(f"Cache SET for query hash {hash(query)}")


def cache_get_all() -> list[dict]:
    """Return all cached entries as a list of dicts (for linear similarity scan)."""
    client = get_client()
    entries = []
    for key in client.keys("cache:*"):
        try:
            entries.append(json.loads(client.get(key)))
        except Exception as e:
            logger.warning(f"Failed to deserialise cache key {key}: {e}")
    return entries


# ---------------------------------------------------------------------------
# Short-Term Memory / conversation history
# ---------------------------------------------------------------------------

def stm_append(session_id: str, role: str, content: str, ttl: int = 3600) -> None:
    """Append a message to the session's conversation history in Redis."""
    client = get_client()
    session_key = f"chat:{session_id}"
    client.rpush(session_key, json.dumps({"role": role, "content": content}))
    client.expire(session_key, ttl)


def stm_get(session_id: str) -> list[dict]:
    """Return the full conversation history for a session."""
    client = get_client()
    session_key = f"chat:{session_id}"
    raw = client.lrange(session_key, 0, -1)
    return [json.loads(msg) for msg in raw]


def stm_delete(session_id: str) -> None:
    """Delete a session's conversation history."""
    client = get_client()
    client.delete(f"chat:{session_id}")
