"""
SQLite-backed memory store for elephant-coder with optional Redis cache.

Inspired by CapsuleMemory (backend/capsule_transformer.py:91-155) and the
hippocampal circular buffer pattern from nn/memory.py MemoryWrite.

Each memory is a compressed "capsule" of code context with cognitive metadata
for relevance-based retrieval and lifecycle management.
"""

import hashlib
import json
import logging
import os
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger("elephant-coder.store")


@dataclass
class MemoryEntry:
    """A compressed code context capsule.

    Mirrors CapsuleMemory's structure: identity + content + cognitive metadata.
    The 'summary' field is the capsule encoding (full source -> compact text),
    'keywords' is the DG sparse expansion (discriminative tokens for search).
    """

    memory_id: str
    file_path: str
    symbol_name: str
    kind: str  # "function" | "class" | "module" | "file_summary" | "note"

    # Capsule content
    summary: str
    keywords: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)

    # Cognitive metadata
    access_count: int = 0
    relevance_score: float = 0.0
    freshness: float = 0.0
    file_mtime: float = 0.0
    created: float = 0.0
    compression_level: int = 0
    is_stale: bool = False


def make_memory_id(file_path: str, symbol_name: str, kind: str) -> str:
    """Deterministic ID from file + symbol + kind."""
    raw = f"{file_path}:{symbol_name}:{kind}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _db_dir(project_root: str) -> Path:
    """Return per-project database directory under ~/.elephant-coder/."""
    project_hash = hashlib.sha256(project_root.encode()).hexdigest()[:12]
    base = Path.home() / ".elephant-coder" / project_hash
    base.mkdir(parents=True, exist_ok=True)
    return base


def _project_hash(project_root: str) -> str:
    return hashlib.sha256(project_root.encode()).hexdigest()[:12]


def _entry_to_dict(entry: MemoryEntry) -> dict:
    """Serialize a MemoryEntry to a JSON-safe dict."""
    d = asdict(entry)
    return d


def _dict_to_entry(d: dict) -> MemoryEntry:
    """Deserialize a dict to a MemoryEntry."""
    return MemoryEntry(**d)


# ------------------------------------------------------------------
# Redis Cache
# ------------------------------------------------------------------


class RedisCache:
    """Write-through Redis cache for MemoryStore.

    Key schema:
        ec:{project_hash}:mem:{memory_id}     — individual entry (JSON)
        ec:{project_hash}:file:{file_path_hash} — set of memory_ids for a file
        ec:{project_hash}:fts:{query_hash}     — cached FTS result (JSON list)
    """

    def __init__(self, redis_url: str, project_hash: str, ttl: int = 86400):
        self._available = False
        self._warned = False
        self._prefix = f"ec:{project_hash}"
        self._ttl = ttl
        self._fts_ttl = 300  # 5 minutes for FTS results
        try:
            import redis as redis_lib
            self._r = redis_lib.from_url(redis_url, decode_responses=True)
            self._r.ping()
            self._available = True
            logger.info("Redis cache connected: %s", redis_url)
        except Exception as exc:
            if not self._warned:
                logger.warning("Redis unavailable, falling back to SQLite-only: %s", exc)
                self._warned = True
            self._r = None

    @property
    def available(self) -> bool:
        return self._available

    def _key(self, kind: str, id_part: str) -> str:
        return f"{self._prefix}:{kind}:{id_part}"

    def _file_hash(self, file_path: str) -> str:
        return hashlib.sha256(file_path.encode()).hexdigest()[:16]

    def _query_hash(self, query: str) -> str:
        return hashlib.sha256(query.encode()).hexdigest()[:16]

    # --- Entry operations ---

    def put_entry(self, entry: MemoryEntry) -> None:
        if not self._available:
            return
        try:
            key = self._key("mem", entry.memory_id)
            self._r.setex(key, self._ttl, json.dumps(_entry_to_dict(entry)))
            # Add to file set
            fkey = self._key("file", self._file_hash(entry.file_path))
            self._r.sadd(fkey, entry.memory_id)
            self._r.expire(fkey, self._ttl)
        except Exception:
            pass

    def get_entry(self, memory_id: str) -> MemoryEntry | None:
        if not self._available:
            return None
        try:
            data = self._r.get(self._key("mem", memory_id))
            if data:
                return _dict_to_entry(json.loads(data))
        except Exception:
            pass
        return None

    def delete_entry(self, memory_id: str, file_path: str | None = None) -> None:
        if not self._available:
            return
        try:
            self._r.delete(self._key("mem", memory_id))
            if file_path:
                fkey = self._key("file", self._file_hash(file_path))
                self._r.srem(fkey, memory_id)
        except Exception:
            pass

    # --- File set operations ---

    def get_file_entries(self, file_path: str) -> list[str] | None:
        """Return memory_ids for a file, or None if not cached."""
        if not self._available:
            return None
        try:
            fkey = self._key("file", self._file_hash(file_path))
            members = self._r.smembers(fkey)
            return list(members) if members else None
        except Exception:
            return None

    def invalidate_file(self, file_path: str) -> None:
        """Remove all cached entries for a file."""
        if not self._available:
            return
        try:
            fkey = self._key("file", self._file_hash(file_path))
            mem_ids = self._r.smembers(fkey)
            if mem_ids:
                pipe = self._r.pipeline()
                for mid in mem_ids:
                    pipe.delete(self._key("mem", mid))
                pipe.delete(fkey)
                pipe.execute()
        except Exception:
            pass

    # --- FTS cache ---

    def put_fts(self, query: str, entries: list[MemoryEntry]) -> None:
        if not self._available:
            return
        try:
            key = self._key("fts", self._query_hash(query))
            data = json.dumps([_entry_to_dict(e) for e in entries])
            self._r.setex(key, self._fts_ttl, data)
        except Exception:
            pass

    def get_fts(self, query: str) -> list[MemoryEntry] | None:
        if not self._available:
            return None
        try:
            data = self._r.get(self._key("fts", self._query_hash(query)))
            if data:
                return [_dict_to_entry(d) for d in json.loads(data)]
        except Exception:
            pass
        return None

    def flush_fts(self) -> None:
        """Invalidate all cached FTS results."""
        if not self._available:
            return
        try:
            pattern = self._key("fts", "*")
            cursor = 0
            while True:
                cursor, keys = self._r.scan(cursor, match=pattern, count=100)
                if keys:
                    self._r.delete(*keys)
                if cursor == 0:
                    break
        except Exception:
            pass


class MemoryStore:
    """SQLite storage with FTS5 full-text search and optional Redis cache.

    Capacity-limited circular buffer: when count exceeds max_memories,
    lowest-relevance entries are evicted (like MemoryWrite overwrite mode).
    """

    def __init__(self, project_root: str, max_memories: int = 10_000, redis_url: str | None = None):
        self.project_root = project_root
        self.max_memories = max_memories
        db_path = _db_dir(project_root) / "memories.db"
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()

        # Redis cache (optional)
        r_url = redis_url or os.environ.get("ELEPHANT_CODER_REDIS_URL", "redis://localhost:6380")
        ttl = int(os.environ.get("ELEPHANT_CODER_REDIS_TTL", "86400"))
        self._cache = RedisCache(r_url, _project_hash(project_root), ttl=ttl)

    @property
    def cache(self) -> RedisCache:
        return self._cache

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                memory_id TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                symbol_name TEXT NOT NULL,
                kind TEXT NOT NULL,
                summary TEXT NOT NULL,
                keywords TEXT NOT NULL DEFAULT '[]',
                dependencies TEXT NOT NULL DEFAULT '[]',
                access_count INTEGER DEFAULT 0,
                relevance_score REAL DEFAULT 0.0,
                freshness REAL NOT NULL,
                file_mtime REAL DEFAULT 0.0,
                created REAL NOT NULL,
                compression_level INTEGER DEFAULT 0,
                is_stale INTEGER DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_memories_file_path
                ON memories(file_path);
            CREATE INDEX IF NOT EXISTS idx_memories_kind
                ON memories(kind);
            CREATE INDEX IF NOT EXISTS idx_memories_symbol_name
                ON memories(symbol_name);
            CREATE INDEX IF NOT EXISTS idx_memories_relevance
                ON memories(relevance_score DESC);
        """)

        # Standalone FTS5 table (not external content — avoids rowid sync issues)
        cur.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                memory_id UNINDEXED,
                symbol_name,
                summary,
                keywords
            )
        """)
        self._conn.commit()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def upsert(self, entry: MemoryEntry) -> None:
        """Insert or replace a memory entry and update FTS index."""
        now = time.time()
        if entry.created == 0.0:
            entry.created = now
        if entry.freshness == 0.0:
            entry.freshness = now

        cur = self._conn.cursor()

        # Delete old FTS row if exists
        cur.execute("DELETE FROM memories_fts WHERE memory_id = ?", (entry.memory_id,))
        # Upsert main table
        cur.execute(
            """
            INSERT OR REPLACE INTO memories
                (memory_id, file_path, symbol_name, kind, summary, keywords,
                 dependencies, access_count, relevance_score, freshness,
                 file_mtime, created, compression_level, is_stale)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.memory_id,
                entry.file_path,
                entry.symbol_name,
                entry.kind,
                entry.summary,
                json.dumps(entry.keywords),
                json.dumps(entry.dependencies),
                entry.access_count,
                entry.relevance_score,
                entry.freshness,
                entry.file_mtime,
                entry.created,
                entry.compression_level,
                int(entry.is_stale),
            ),
        )
        # Insert FTS row
        cur.execute(
            """
            INSERT INTO memories_fts (memory_id, symbol_name, summary, keywords)
            VALUES (?, ?, ?, ?)
            """,
            (
                entry.memory_id,
                entry.symbol_name,
                entry.summary,
                " ".join(entry.keywords),
            ),
        )
        self._conn.commit()

        # Write-through to Redis
        self._cache.put_entry(entry)

    def get(self, memory_id: str) -> MemoryEntry | None:
        """Fetch a single memory by ID."""
        # Check Redis first
        cached = self._cache.get_entry(memory_id)
        if cached is not None:
            return cached

        row = self._conn.execute(
            "SELECT * FROM memories WHERE memory_id = ?", (memory_id,)
        ).fetchone()
        if row is None:
            return None
        entry = self._row_to_entry(row)
        # Backfill cache
        self._cache.put_entry(entry)
        return entry

    def search_fts(self, query: str, limit: int = 10) -> list[MemoryEntry]:
        """Full-text search with BM25 ranking. Checks Redis cache first."""
        # Check Redis FTS cache
        cached = self._cache.get_fts(query)
        if cached is not None:
            return cached[:limit]

        # Escape special FTS5 characters
        safe_query = query.replace('"', '""')
        try:
            rows = self._conn.execute(
                """
                SELECT m.* FROM memories m
                JOIN memories_fts fts ON m.memory_id = fts.memory_id
                WHERE memories_fts MATCH ?
                ORDER BY bm25(memories_fts) ASC
                LIMIT ?
                """,
                (f'"{safe_query}" OR {safe_query}', limit),
            ).fetchall()
        except sqlite3.OperationalError:
            # Fallback: simple LIKE search if FTS query fails
            like = f"%{query}%"
            rows = self._conn.execute(
                """
                SELECT * FROM memories
                WHERE summary LIKE ? OR symbol_name LIKE ? OR keywords LIKE ?
                ORDER BY relevance_score DESC
                LIMIT ?
                """,
                (like, like, like, limit),
            ).fetchall()
        results = [self._row_to_entry(r) for r in rows]

        # Cache results in Redis
        self._cache.put_fts(query, results)
        return results

    def search_by_file(self, file_path: str) -> list[MemoryEntry]:
        """Get all memories for a specific file."""
        # Check Redis file set for cached entries
        cached_ids = self._cache.get_file_entries(file_path)
        if cached_ids:
            entries = []
            for mid in cached_ids:
                e = self._cache.get_entry(mid)
                if e is not None:
                    entries.append(e)
            if entries:
                entries.sort(key=lambda e: (e.kind, e.symbol_name))
                return entries

        rows = self._conn.execute(
            "SELECT * FROM memories WHERE file_path = ? ORDER BY kind, symbol_name",
            (file_path,),
        ).fetchall()
        results = [self._row_to_entry(r) for r in rows]
        # Backfill cache
        for e in results:
            self._cache.put_entry(e)
        return results

    def search_by_kind(self, kind: str, limit: int = 50) -> list[MemoryEntry]:
        """Get memories filtered by kind."""
        rows = self._conn.execute(
            "SELECT * FROM memories WHERE kind = ? ORDER BY relevance_score DESC LIMIT ?",
            (kind, limit),
        ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def search_by_symbol(self, name: str, kind: str | None = None) -> list[MemoryEntry]:
        """Direct symbol lookup by name (exact or prefix match)."""
        if kind:
            rows = self._conn.execute(
                "SELECT * FROM memories WHERE symbol_name = ? AND kind = ? ORDER BY relevance_score DESC",
                (name, kind),
            ).fetchall()
            if not rows:
                rows = self._conn.execute(
                    "SELECT * FROM memories WHERE symbol_name LIKE ? AND kind = ? ORDER BY relevance_score DESC LIMIT 20",
                    (f"{name}%", kind),
                ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM memories WHERE symbol_name = ? ORDER BY relevance_score DESC",
                (name,),
            ).fetchall()
            if not rows:
                rows = self._conn.execute(
                    "SELECT * FROM memories WHERE symbol_name LIKE ? ORDER BY relevance_score DESC LIMIT 20",
                    (f"{name}%",),
                ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def get_dependencies(self, file_path: str) -> dict:
        """Get what a file imports and what imports it."""
        # What this file imports
        row = self._conn.execute(
            "SELECT dependencies FROM memories WHERE file_path = ? AND kind = 'module'",
            (file_path,),
        ).fetchone()
        imports = json.loads(row["dependencies"]) if row else []

        # What imports this file (search dependencies that mention this file's module name)
        module_name = Path(file_path).stem
        rows = self._conn.execute(
            "SELECT DISTINCT file_path FROM memories WHERE kind = 'module' AND dependencies LIKE ? AND file_path != ?",
            (f"%{module_name}%", file_path),
        ).fetchall()
        imported_by = [r["file_path"] for r in rows]

        return {"imports": imports, "imported_by": imported_by}

    def touch(self, memory_id: str) -> None:
        """Increment access_count and update freshness on retrieval (Hebbian)."""
        now = time.time()
        self._conn.execute(
            """
            UPDATE memories
            SET access_count = access_count + 1, freshness = ?
            WHERE memory_id = ?
            """,
            (now, memory_id),
        )
        self._conn.commit()

    def touch_batch(self, memory_ids: list[str]) -> None:
        """Batch touch: increment access_count and update freshness for multiple entries."""
        if not memory_ids:
            return
        now = time.time()
        self._conn.executemany(
            "UPDATE memories SET access_count = access_count + 1, freshness = ? WHERE memory_id = ?",
            [(now, mid) for mid in memory_ids],
        )
        self._conn.commit()

    def update_relevance(self, memory_id: str, relevance_score: float) -> None:
        """Lightweight relevance update without full upsert (avoids FTS5 delete+insert)."""
        self._conn.execute(
            "UPDATE memories SET relevance_score = ? WHERE memory_id = ?",
            (relevance_score, memory_id),
        )
        self._conn.commit()

    def delete(self, memory_id: str) -> bool:
        """Delete a memory and its FTS entry."""
        # Get file_path for cache invalidation
        row = self._conn.execute(
            "SELECT file_path FROM memories WHERE memory_id = ?", (memory_id,)
        ).fetchone()
        file_path = row["file_path"] if row else None

        cur = self._conn.cursor()
        cur.execute("DELETE FROM memories_fts WHERE memory_id = ?", (memory_id,))
        cur.execute("DELETE FROM memories WHERE memory_id = ?", (memory_id,))
        self._conn.commit()

        self._cache.delete_entry(memory_id, file_path)
        return cur.rowcount > 0

    def delete_by_file(self, file_path: str) -> int:
        """Delete all memories for a file."""
        ids = [
            r["memory_id"]
            for r in self._conn.execute(
                "SELECT memory_id FROM memories WHERE file_path = ?", (file_path,)
            ).fetchall()
        ]
        for mid in ids:
            self.delete(mid)
        self._cache.invalidate_file(file_path)
        return len(ids)

    def delete_stale(self) -> int:
        """Delete all stale memories."""
        ids = [
            r["memory_id"]
            for r in self._conn.execute(
                "SELECT memory_id FROM memories WHERE is_stale = 1"
            ).fetchall()
        ]
        for mid in ids:
            self.delete(mid)
        return len(ids)

    def evict_lowest(self, count: int) -> int:
        """Evict lowest-relevance memories, skipping recently accessed ones."""
        one_hour_ago = time.time() - 3600
        rows = self._conn.execute(
            """
            SELECT memory_id FROM memories
            WHERE freshness < ?
            ORDER BY relevance_score ASC
            LIMIT ?
            """,
            (one_hour_ago, count),
        ).fetchall()
        evicted = 0
        for r in rows:
            if self.delete(r["memory_id"]):
                evicted += 1
        return evicted

    def count(self) -> int:
        """Total number of memories."""
        row = self._conn.execute("SELECT COUNT(*) AS c FROM memories").fetchone()
        return row["c"]

    def stats(self) -> dict:
        """Aggregate statistics about the memory store."""
        total = self.count()
        kinds = {}
        for r in self._conn.execute(
            "SELECT kind, COUNT(*) AS c FROM memories GROUP BY kind"
        ).fetchall():
            kinds[r["kind"]] = r["c"]

        stale_count = self._conn.execute(
            "SELECT COUNT(*) AS c FROM memories WHERE is_stale = 1"
        ).fetchone()["c"]

        top_accessed = self._conn.execute(
            "SELECT symbol_name, file_path, access_count FROM memories "
            "ORDER BY access_count DESC LIMIT 5"
        ).fetchall()

        result = {
            "total": total,
            "max_capacity": self.max_memories,
            "utilization_pct": round(total / self.max_memories * 100, 1)
            if self.max_memories
            else 0,
            "by_kind": kinds,
            "stale": stale_count,
            "top_accessed": [
                {"symbol": r["symbol_name"], "file": r["file_path"], "count": r["access_count"]}
                for r in top_accessed
            ],
            "redis_connected": self._cache.available,
        }
        return result

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> MemoryEntry:
        return MemoryEntry(
            memory_id=row["memory_id"],
            file_path=row["file_path"],
            symbol_name=row["symbol_name"],
            kind=row["kind"],
            summary=row["summary"],
            keywords=json.loads(row["keywords"]),
            dependencies=json.loads(row["dependencies"]),
            access_count=row["access_count"],
            relevance_score=row["relevance_score"],
            freshness=row["freshness"],
            file_mtime=row["file_mtime"],
            created=row["created"],
            compression_level=row["compression_level"],
            is_stale=bool(row["is_stale"]),
        )
