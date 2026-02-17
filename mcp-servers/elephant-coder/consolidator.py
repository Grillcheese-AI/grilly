"""
Memory lifecycle manager for elephant-coder.

Handles staleness detection, relevance decay, and capacity-based eviction.
Analogous to the hippocampal consolidation cycle:
- Staleness detection = checking if a memory's source has changed (like stability decay)
- Eviction = circular buffer overwrite from nn/memory.py MemoryWrite
- Relevance recomputation = consolidation_priority update from CognitiveFeatures
"""

import logging
import os

from memory_store import MemoryStore
from retriever import compute_relevance

logger = logging.getLogger("elephant-coder.consolidator")


def detect_stale(store: MemoryStore) -> int:
    """Mark memories as stale if their source file has been modified.

    Checks file_mtime against actual disk mtime. Returns count of
    newly stale memories. Invalidates Redis cache for stale files.
    """
    # Get all distinct file paths
    rows = store._conn.execute(
        "SELECT DISTINCT file_path FROM memories WHERE is_stale = 0"
    ).fetchall()

    stale_count = 0
    stale_files: list[str] = []
    for row in rows:
        fp = row["file_path"]
        try:
            actual_mtime = os.path.getmtime(fp)
        except OSError:
            # File deleted — mark all its memories stale
            actual_mtime = float("inf")

        # Find memories whose file_mtime is older than the actual file
        outdated = store._conn.execute(
            "SELECT memory_id FROM memories WHERE file_path = ? AND file_mtime < ? AND is_stale = 0",
            (fp, actual_mtime),
        ).fetchall()

        if outdated:
            ids = [r["memory_id"] for r in outdated]
            placeholders = ",".join("?" * len(ids))
            store._conn.execute(
                f"UPDATE memories SET is_stale = 1 WHERE memory_id IN ({placeholders})",
                ids,
            )
            stale_count += len(ids)
            stale_files.append(fp)

    if stale_count:
        store._conn.commit()
        logger.info("Marked %d memories as stale", stale_count)
        # Invalidate Redis cache for stale files
        invalidate_redis_cache(store, stale_files)

    return stale_count


def recompute_relevance(store: MemoryStore) -> int:
    """Recompute relevance_score for all memories based on current time."""
    rows = store._conn.execute(
        "SELECT memory_id, access_count, freshness, created FROM memories"
    ).fetchall()

    updates = []
    for r in rows:
        new_score = compute_relevance(r["access_count"], r["freshness"], r["created"])
        updates.append((new_score, r["memory_id"]))

    store._conn.executemany("UPDATE memories SET relevance_score = ? WHERE memory_id = ?", updates)
    store._conn.commit()
    return len(updates)


def consolidate(store: MemoryStore) -> dict:
    """Run full consolidation cycle.

    1. Detect stale memories
    2. Recompute all relevance scores
    3. Evict if over capacity (bottom 10%)
    4. Flush cached FTS results (may reference evicted entries)

    Returns stats about what happened.
    """
    stats = {
        "stale_detected": detect_stale(store),
        "relevance_updated": recompute_relevance(store),
        "evicted": 0,
    }

    total = store.count()
    if total > store.max_memories:
        to_evict = max(1, total - store.max_memories + store.max_memories // 10)
        stats["evicted"] = store.evict_lowest(to_evict)
        logger.info(
            "Evicted %d memories (was %d, max %d)", stats["evicted"], total, store.max_memories
        )

    # Flush FTS cache after consolidation (results may reference evicted/stale entries)
    store.cache.flush_fts()

    return stats


def should_consolidate(store: MemoryStore) -> bool:
    """Check if consolidation should run (>90% capacity)."""
    return store.count() > store.max_memories * 0.9


def invalidate_redis_cache(store: MemoryStore, file_paths: list[str]) -> None:
    """Invalidate Redis cache entries for the given file paths."""
    for fp in file_paths:
        store.cache.invalidate_file(fp)
    # Also flush FTS cache since results may include entries from these files
    store.cache.flush_fts()
