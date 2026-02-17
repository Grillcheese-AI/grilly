"""
Retriever for elephant-coder — pattern completion via FTS5 search.

Analogous to CA3 pattern completion in nn/hippocampal.py: given a partial
cue (query), retrieve full memory entries using BM25-ranked search, then
strengthen accessed memories (Hebbian learning via access_count increment).

Relevance scoring mirrors CognitiveFeatures.consolidation_priority:
    relevance = recency_weight * recency + frequency_weight * log(1 + access_count)
"""

import logging
import math
import time

from memory_store import MemoryEntry, MemoryStore

logger = logging.getLogger("elephant-coder.retriever")


def compute_relevance(access_count: int, last_access: float, created: float) -> float:
    """Compute relevance score combining recency and frequency.

    Higher score = more relevant. Decays over hours without access.
    """
    now = time.time()
    hours_since_access = max((now - last_access) / 3600.0, 0.001)
    recency = 1.0 / (1.0 + hours_since_access)
    frequency = math.log1p(access_count)
    return round(recency * 0.6 + frequency * 0.4, 4)


def recall(
    store: MemoryStore,
    query: str,
    limit: int = 5,
    kind: str | None = None,
) -> list[MemoryEntry]:
    """Search memories and return ranked results.

    Performs FTS5 search (with Redis cache), optionally filters by kind,
    batch-updates access stats (Hebbian strengthening), and uses lightweight
    relevance updates instead of full upserts.
    """
    results = store.search_fts(query, limit=limit * 3)

    if kind:
        results = [r for r in results if r.kind == kind]

    results = results[:limit]

    # Batch Hebbian strengthening
    now = time.time()
    ids_to_touch = [entry.memory_id for entry in results]
    store.touch_batch(ids_to_touch)

    for entry in results:
        entry.access_count += 1
        entry.freshness = now
        entry.relevance_score = compute_relevance(
            entry.access_count, entry.freshness, entry.created
        )
        # Lightweight relevance update (avoids FTS5 delete+insert)
        store.update_relevance(entry.memory_id, entry.relevance_score)

    return results


def recall_file(store: MemoryStore, file_path: str) -> list[MemoryEntry]:
    """Retrieve all memories for a specific file, touching each."""
    results = store.search_by_file(file_path)
    ids_to_touch = [entry.memory_id for entry in results]
    store.touch_batch(ids_to_touch)
    return results


def format_results(entries: list[MemoryEntry]) -> str:
    """Format memory entries as plain text for minimal token usage."""
    if not entries:
        return "No memories found."

    parts = []
    for i, e in enumerate(entries, 1):
        header = f"[{i}] {e.kind}: {e.symbol_name}"
        if e.file_path:
            header += f"  ({e.file_path})"
        lines = [header, e.summary]
        if e.is_stale:
            lines.append("  [STALE — source file has changed]")
        meta = f"  accessed: {e.access_count}x | relevance: {e.relevance_score:.3f}"
        lines.append(meta)
        parts.append("\n".join(lines))

    return "\n\n".join(parts)
