"""
SQLite-backed memory store for elephant-coder.

Inspired by CapsuleMemory (backend/capsule_transformer.py:91-155) and the
hippocampal circular buffer pattern from nn/memory.py MemoryWrite.

Each memory is a compressed "capsule" of code context with cognitive metadata
for relevance-based retrieval and lifecycle management.
"""

import hashlib
import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
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


class MemoryStore:
    """SQLite storage with FTS5 full-text search.

    Capacity-limited circular buffer: when count exceeds max_memories,
    lowest-relevance entries are evicted (like MemoryWrite overwrite mode).
    """

    def __init__(self, project_root: str, max_memories: int = 10_000):
        self.project_root = project_root
        self.max_memories = max_memories
        db_path = _db_dir(project_root) / "memories.db"
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()

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

    def get(self, memory_id: str) -> MemoryEntry | None:
        """Fetch a single memory by ID."""
        row = self._conn.execute(
            "SELECT * FROM memories WHERE memory_id = ?", (memory_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_entry(row)

    def search_fts(self, query: str, limit: int = 10) -> list[MemoryEntry]:
        """Full-text search with BM25 ranking."""
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
        return [self._row_to_entry(r) for r in rows]

    def search_by_file(self, file_path: str) -> list[MemoryEntry]:
        """Get all memories for a specific file."""
        rows = self._conn.execute(
            "SELECT * FROM memories WHERE file_path = ? ORDER BY kind, symbol_name",
            (file_path,),
        ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def search_by_kind(self, kind: str, limit: int = 50) -> list[MemoryEntry]:
        """Get memories filtered by kind."""
        rows = self._conn.execute(
            "SELECT * FROM memories WHERE kind = ? ORDER BY relevance_score DESC LIMIT ?",
            (kind, limit),
        ).fetchall()
        return [self._row_to_entry(r) for r in rows]

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

    def delete(self, memory_id: str) -> bool:
        """Delete a memory and its FTS entry."""
        cur = self._conn.cursor()
        cur.execute("DELETE FROM memories_fts WHERE memory_id = ?", (memory_id,))
        cur.execute("DELETE FROM memories WHERE memory_id = ?", (memory_id,))
        self._conn.commit()
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

        return {
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
        }

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
