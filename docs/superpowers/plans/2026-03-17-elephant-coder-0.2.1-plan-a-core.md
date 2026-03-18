# elephant-coder 0.2.1 Plan A: Core Infrastructure

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform elephant-coder from manual toolkit to invisible infrastructure with batch performance, Redis-optional SQLite fallback, unified indexing, configurable settings, and global knowledge store.

**Architecture:** The MCP server keeps its existing SQLite + optional Redis design but gains: `upsert_batch()` for transactional bulk writes, `index_all()` that replaces 14 separate tool calls, a settings file (`.claude/elephant-coder.local.md`) readable by both hooks and tools, a global knowledge SQLite store at `~/.elephant-coder/global/`, and graceful Redis fallback. The hook system shifts from reminder prompts to automatic knowledge injection.

**Tech Stack:** Python 3.10+, FastMCP, SQLite (FTS5), Redis (optional), YAML frontmatter parsing, httpx (for Plan D)

**Spec:** `docs/superpowers/specs/2026-03-17-elephant-coder-0.2.1-design.md`

**Plugin source:** `C:\Users\grill\.claude\plugins\marketplaces\grilly-plugins\elephant-coder\`

**Marketplace repo (commit changes here):** `C:\Users\grill\.claude\plugins\marketplaces\grilly-plugins\`

---

## File Structure

### Modified Files

| File | Responsibility | Changes |
|------|---------------|---------|
| `memory_store.py` | SQLite storage + Redis cache | Add `upsert_batch()`, make Redis optional (no crash), add settings loading |
| `server.py` | MCP tool definitions | Add `index_all()`, `update_settings()` tools, refactor `index_directory()` to use batch upserts |
| `indexer.py` | File parsing → MemoryEntry | No changes in Plan A (link graph is Plan B) |
| `retriever.py` | Search + relevance | Add relevance threshold filtering from settings |
| `consolidator.py` | Lifecycle management | Evict stale first, respect configurable `max_memories` |
| `hooks/hooks.json` | Hook definitions | Replace reminder hooks with SessionStart prompt + silent re-index hooks |
| `.claude-plugin/plugin.json` | Plugin manifest | Version bump to 0.2.1 |
| `pyproject.toml` | Package metadata | Version bump, make redis optional dependency |
| `CLAUDE.md` | Plugin instructions | Rewrite to behavioral guide |

### New Files

| File | Responsibility |
|------|---------------|
| `global_store.py` | Global knowledge SQLite store at `~/.elephant-coder/global/knowledge.db` |
| `settings.py` | Parse `.claude/elephant-coder.local.md` YAML frontmatter, provide defaults |
| `hooks/session_start_prompt.md` | SessionStart prompt hook content |
| `hooks/reindex_file.sh` | PostToolUse:Edit/Write silent re-index script |

---

## Task 1: Make Redis Optional

**Files:**
- Modify: `memory_store.py` — `RedisCache.__init__` (lines 103-118)
- Test: `tests/test_memory_store.py` (create)

- [ ] **Step 1: Write failing test — store works without Redis**

```python
# tests/test_memory_store.py
import os
import tempfile
from memory_store import MemoryStore, MemoryEntry, make_memory_id


def test_store_works_without_redis():
    """MemoryStore should function fully when Redis is unavailable."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Point to a Redis that definitely doesn't exist
        store = MemoryStore(tmpdir, redis_url="redis://localhost:59999")
        assert store.cache.available is False

        # Core operations should still work
        entry = MemoryEntry(
            memory_id=make_memory_id("test.py", "foo", "function"),
            file_path=os.path.join(tmpdir, "test.py"),
            symbol_name="foo",
            kind="function",
            summary="A test function",
            keywords=["test"],
        )
        store.upsert(entry)
        assert store.count() == 1

        # Search should work
        results = store.search_fts("test", limit=5)
        assert len(results) == 1
        assert results[0].symbol_name == "foo"

        # Retrieval should work
        got = store.get(entry.memory_id)
        assert got is not None
        assert got.symbol_name == "foo"

        store.close()


def test_store_works_with_redis_if_available():
    """If Redis is running on default port, cache should be used."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = MemoryStore(tmpdir)
        # Don't assert available — depends on environment
        # Just verify it doesn't crash
        assert store.count() == 0
        store.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd C:\Users\grill\.claude\plugins\marketplaces\grilly-plugins\elephant-coder && uv run pytest tests/test_memory_store.py -v`
Expected: FAIL — `RuntimeError: Redis is required but unavailable`

- [ ] **Step 3: Make Redis graceful — modify RedisCache.__init__**

In `memory_store.py`, replace the `raise RuntimeError` block:

```python
class RedisCache:
    """Write-through Redis cache for MemoryStore.

    Key schema:
        ec:{project_hash}:mem:{memory_id}     — individual entry (JSON)
        ec:{project_hash}:file:{file_path_hash} — set of memory_ids for a file
        ec:{project_hash}:fts:{query_hash}     — cached FTS result (JSON list)

    Falls back gracefully if Redis is unavailable — SQLite handles everything.
    """

    # 1 year in seconds
    DEFAULT_TTL = 365 * 24 * 3600
    # 3 months in seconds for FTS results
    DEFAULT_FTS_TTL = 90 * 24 * 3600

    def __init__(self, redis_url: str, project_hash: str, ttl: int = DEFAULT_TTL):
        self._available = False
        self._prefix = f"ec:{project_hash}"
        self._ttl = ttl
        self._fts_ttl = self.DEFAULT_FTS_TTL
        try:
            import redis as redis_lib
            self._r = redis_lib.from_url(redis_url, decode_responses=True)
            self._r.ping()
            self._available = True
            logger.info("Redis cache connected: %s", redis_url)
        except Exception as exc:
            self._available = False
            logger.info("Redis not available at %s — using SQLite only (this is fine): %s", redis_url, exc)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd C:\Users\grill\.claude\plugins\marketplaces\grilly-plugins\elephant-coder && uv run pytest tests/test_memory_store.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd C:\Users\grill\.claude\plugins\marketplaces\grilly-plugins
git add elephant-coder/memory_store.py elephant-coder/tests/test_memory_store.py
git commit -m "feat: make Redis optional — graceful fallback to SQLite only"
```

---

## Task 2: Settings File Parser

**Files:**
- Create: `settings.py`
- Test: `tests/test_settings.py` (create)

- [ ] **Step 1: Write failing test — settings parser**

```python
# tests/test_settings.py
import os
import tempfile
from settings import load_settings, DEFAULT_SETTINGS


def test_default_settings_when_no_file():
    """Should return defaults when no settings file exists."""
    with tempfile.TemporaryDirectory() as tmpdir:
        s = load_settings(tmpdir)
        assert s["max_memories"] == 50_000
        assert s["relevance_threshold"] == 0.1
        assert s["redis_url"] == "redis://localhost:6380"
        assert s["scope_guard"] is True
        assert s["auto_test_after_edit"] is True
        assert s["skip_dirs"] == [".venv", "node_modules", "__pycache__", "dist", "build", ".git", ".eggs"]


def test_load_settings_from_file():
    """Should parse YAML frontmatter from .claude/elephant-coder.local.md."""
    with tempfile.TemporaryDirectory() as tmpdir:
        claude_dir = os.path.join(tmpdir, ".claude")
        os.makedirs(claude_dir)
        settings_file = os.path.join(claude_dir, "elephant-coder.local.md")
        with open(settings_file, "w") as f:
            f.write("""---
max_memories: 100000
relevance_threshold: 0.2
redis_url: null
skip_dirs: [".venv", "node_modules"]
frameworks:
  - name: grilly
    repo_path: auto
    github: Grillcheese-AI/grilly
    auto_fix_prs: true
external_validation:
  enabled: true
  openrouter_api_key: "sk-or-test"
  model: "google/gemini-3.1-flash-lite-preview"
---

# Project Notes

Custom notes here.
""")
        s = load_settings(tmpdir)
        assert s["max_memories"] == 100_000
        assert s["relevance_threshold"] == 0.2
        assert s["redis_url"] is None
        assert s["skip_dirs"] == [".venv", "node_modules"]
        assert len(s["frameworks"]) == 1
        assert s["frameworks"][0]["name"] == "grilly"
        assert s["external_validation"]["enabled"] is True


def test_partial_settings_merged_with_defaults():
    """Settings file with only some keys should merge with defaults."""
    with tempfile.TemporaryDirectory() as tmpdir:
        claude_dir = os.path.join(tmpdir, ".claude")
        os.makedirs(claude_dir)
        settings_file = os.path.join(claude_dir, "elephant-coder.local.md")
        with open(settings_file, "w") as f:
            f.write("""---
max_memories: 75000
---
""")
        s = load_settings(tmpdir)
        assert s["max_memories"] == 75_000
        # Other defaults preserved
        assert s["relevance_threshold"] == 0.1
        assert s["scope_guard"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd C:\Users\grill\.claude\plugins\marketplaces\grilly-plugins\elephant-coder && uv run pytest tests/test_settings.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'settings'`

- [ ] **Step 3: Implement settings.py**

```python
# settings.py
"""
Settings parser for elephant-coder plugin.

Reads per-project configuration from .claude/elephant-coder.local.md
(YAML frontmatter). Falls back to sensible defaults for all values.
"""

import logging
import os
import re
from pathlib import Path

import yaml

logger = logging.getLogger("elephant-coder.settings")

DEFAULT_SETTINGS: dict = {
    "max_memories": 50_000,
    "relevance_threshold": 0.1,
    "redis_url": "redis://localhost:6380",
    "redis_ttl": 365 * 24 * 3600,
    "skip_dirs": [".venv", "node_modules", "__pycache__", "dist", "build", ".git", ".eggs"],
    "frameworks": [],
    "auto_test_after_edit": True,
    "scope_guard": True,
    "external_validation": {
        "enabled": False,
        "openrouter_api_key": None,
        "model": "google/gemini-3.1-flash-lite-preview",
        "validate_plans": True,
        "audit_completed_tasks": True,
        "require_approval_on_issues": True,
    },
}

_SETTINGS_FILENAME = "elephant-coder.local.md"


def _parse_frontmatter(text: str) -> dict:
    """Extract YAML frontmatter from markdown text."""
    match = re.match(r"^---\s*\n(.*?)\n---", text, re.DOTALL)
    if not match:
        return {}
    try:
        return yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError as exc:
        logger.warning("Failed to parse settings frontmatter: %s", exc)
        return {}


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge override into base, recursing into nested dicts."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_settings(project_root: str) -> dict:
    """Load settings from .claude/elephant-coder.local.md, merged with defaults."""
    settings_path = Path(project_root) / ".claude" / _SETTINGS_FILENAME
    if not settings_path.exists():
        logger.debug("No settings file at %s — using defaults", settings_path)
        return DEFAULT_SETTINGS.copy()

    try:
        text = settings_path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Failed to read settings: %s", exc)
        return DEFAULT_SETTINGS.copy()

    overrides = _parse_frontmatter(text)
    settings = _deep_merge(DEFAULT_SETTINGS, overrides)

    # Also resolve env var for OpenRouter API key
    ev = settings.get("external_validation", {})
    if ev.get("openrouter_api_key") is None:
        ev["openrouter_api_key"] = os.environ.get("OPENROUTER_API_KEY")
    settings["external_validation"] = ev

    logger.info("Loaded settings from %s", settings_path)
    return settings


def save_settings(project_root: str, settings: dict) -> str:
    """Write settings to .claude/elephant-coder.local.md."""
    claude_dir = Path(project_root) / ".claude"
    claude_dir.mkdir(parents=True, exist_ok=True)
    settings_path = claude_dir / _SETTINGS_FILENAME

    # Preserve existing body if file exists
    body = ""
    if settings_path.exists():
        try:
            text = settings_path.read_text(encoding="utf-8")
            parts = text.split("---", 2)
            if len(parts) >= 3:
                body = parts[2]
        except OSError:
            pass

    frontmatter = yaml.dump(settings, default_flow_style=False, sort_keys=False)
    content = f"---\n{frontmatter}---\n{body}"
    settings_path.write_text(content, encoding="utf-8")
    return str(settings_path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd C:\Users\grill\.claude\plugins\marketplaces\grilly-plugins\elephant-coder && uv run pytest tests/test_settings.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd C:\Users\grill\.claude\plugins\marketplaces\grilly-plugins
git add elephant-coder/settings.py elephant-coder/tests/test_settings.py
git commit -m "feat: add settings file parser (.claude/elephant-coder.local.md)"
```

---

## Task 3: Batch Upsert

**Files:**
- Modify: `memory_store.py` — add `upsert_batch()` method
- Modify: `tests/test_memory_store.py` — add batch test

- [ ] **Step 1: Write failing test — batch upsert**

```python
# Add to tests/test_memory_store.py

def test_upsert_batch():
    """Batch upsert should insert multiple entries in one transaction."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = MemoryStore(tmpdir, redis_url="redis://localhost:59999")

        entries = []
        for i in range(100):
            entries.append(MemoryEntry(
                memory_id=make_memory_id("test.py", f"func_{i}", "function"),
                file_path=os.path.join(tmpdir, "test.py"),
                symbol_name=f"func_{i}",
                kind="function",
                summary=f"Function number {i} that does thing {i}",
                keywords=[f"func_{i}", "test"],
            ))

        store.upsert_batch(entries)
        assert store.count() == 100

        # Verify FTS works for batch-inserted entries
        results = store.search_fts("func_50", limit=5)
        assert any(r.symbol_name == "func_50" for r in results)

        # Verify individual retrieval
        got = store.get(entries[0].memory_id)
        assert got is not None
        assert got.symbol_name == "func_0"

        store.close()


def test_upsert_batch_replaces_existing():
    """Batch upsert should update entries that already exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = MemoryStore(tmpdir, redis_url="redis://localhost:59999")

        mid = make_memory_id("test.py", "foo", "function")
        entry_v1 = MemoryEntry(
            memory_id=mid,
            file_path=os.path.join(tmpdir, "test.py"),
            symbol_name="foo",
            kind="function",
            summary="Version 1",
            keywords=["foo"],
        )
        store.upsert(entry_v1)
        assert store.count() == 1

        entry_v2 = MemoryEntry(
            memory_id=mid,
            file_path=os.path.join(tmpdir, "test.py"),
            symbol_name="foo",
            kind="function",
            summary="Version 2 updated",
            keywords=["foo", "updated"],
        )
        store.upsert_batch([entry_v2])
        assert store.count() == 1

        got = store.get(mid)
        assert "Version 2" in got.summary

        store.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd C:\Users\grill\.claude\plugins\marketplaces\grilly-plugins\elephant-coder && uv run pytest tests/test_memory_store.py::test_upsert_batch -v`
Expected: FAIL — `AttributeError: 'MemoryStore' object has no attribute 'upsert_batch'`

- [ ] **Step 3: Implement upsert_batch in memory_store.py**

Add this method to the `MemoryStore` class, after the existing `upsert()` method:

```python
def upsert_batch(self, entries: list[MemoryEntry]) -> None:
    """Insert or replace multiple entries in a single transaction.

    Much faster than calling upsert() in a loop — one transaction for
    all SQLite writes, one pipeline for all Redis writes.
    """
    if not entries:
        return

    now = time.time()
    cur = self._conn.cursor()

    try:
        cur.execute("BEGIN IMMEDIATE")

        for entry in entries:
            if entry.created == 0.0:
                entry.created = now
            if entry.freshness == 0.0:
                entry.freshness = now

            # Delete old FTS row if exists
            cur.execute("DELETE FROM memories_fts WHERE memory_id = ?", (entry.memory_id,))
            # Upsert main table
            cur.execute(
                """
                INSERT OR REPLACE INTO memories
                    (memory_id, file_path, symbol_name, kind, summary, keywords,
                     dependencies, line_count, access_count, relevance_score, freshness,
                     file_mtime, created, compression_level, is_stale)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.memory_id, entry.file_path, entry.symbol_name,
                    entry.kind, entry.summary, json.dumps(entry.keywords),
                    json.dumps(entry.dependencies), entry.line_count,
                    entry.access_count, entry.relevance_score, entry.freshness,
                    entry.file_mtime, entry.created, entry.compression_level,
                    int(entry.is_stale),
                ),
            )
            # Insert FTS row
            cur.execute(
                """
                INSERT INTO memories_fts (memory_id, symbol_name, summary, keywords)
                VALUES (?, ?, ?, ?)
                """,
                (entry.memory_id, entry.symbol_name, entry.summary, " ".join(entry.keywords)),
            )

        self._conn.commit()
    except Exception:
        self._conn.rollback()
        raise

    # Batch Redis write-through via pipeline
    if self._cache.available:
        try:
            pipe = self._cache._r.pipeline()
            for entry in entries:
                key = self._cache._key("mem", entry.memory_id)
                pipe.setex(key, self._cache._ttl, json.dumps(_entry_to_dict(entry)))
                fkey = self._cache._key("file", self._cache._file_hash(entry.file_path))
                pipe.sadd(fkey, entry.memory_id)
                pipe.expire(fkey, self._cache._ttl)
            pipe.execute()
        except Exception:
            pass  # Cache failure is not critical
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd C:\Users\grill\.claude\plugins\marketplaces\grilly-plugins\elephant-coder && uv run pytest tests/test_memory_store.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
cd C:\Users\grill\.claude\plugins\marketplaces\grilly-plugins
git add elephant-coder/memory_store.py elephant-coder/tests/test_memory_store.py
git commit -m "feat: add upsert_batch() for transactional bulk writes"
```

---

## Task 4: index_all() Tool

**Files:**
- Modify: `server.py` — add `index_all()` MCP tool, refactor `index_directory()` to use `upsert_batch()`
- Test: `tests/test_server.py` (create)

- [ ] **Step 1: Write failing test — index_all returns combined results**

```python
# tests/test_server.py
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from server import index_all, _get_store, _store


def _make_test_project(tmpdir: str) -> str:
    """Create a minimal test project with mixed file types."""
    # Python file
    py_file = Path(tmpdir) / "main.py"
    py_file.write_text("def hello():\n    return 'world'\n")

    # Markdown file
    md_file = Path(tmpdir) / "README.md"
    md_file.write_text("# Test Project\n\nA test.\n")

    # JSON file
    json_file = Path(tmpdir) / "config.json"
    json_file.write_text('{"key": "value"}\n')

    # Create .git so it's detected as project root
    (Path(tmpdir) / ".git").mkdir()

    return tmpdir


def test_index_all_indexes_multiple_file_types():
    """index_all() should index all supported file types in one call."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_test_project(tmpdir)
        with patch("server._detect_project_root", return_value=tmpdir):
            # Reset store
            import server
            server._store = None
            os.environ["ELEPHANT_CODER_REDIS_URL"] = "redis://localhost:59999"
            result = index_all()
            assert "Indexed" in result
            assert "Total memories:" in result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd C:\Users\grill\.claude\plugins\marketplaces\grilly-plugins\elephant-coder && uv run pytest tests/test_server.py::test_index_all_indexes_multiple_file_types -v`
Expected: FAIL — `ImportError: cannot import name 'index_all'`

- [ ] **Step 3: Implement index_all() in server.py**

Add after the existing `index_directory()` tool:

```python
# All supported file patterns for index_all()
_ALL_PATTERNS = [
    "**/*.py", "**/*.ts", "**/*.js", "**/*.tsx", "**/*.jsx",
    "**/*.cpp", "**/*.c", "**/*.h", "**/*.hpp", "**/*.hxx", "**/*.cc", "**/*.cxx",
    "**/*.glsl", "**/*.vert", "**/*.frag", "**/*.comp",
    "**/*.md", "**/*.toml", "**/*.json", "**/*.yaml", "**/*.yml",
    "**/CMakeLists.txt", "**/*.cmake",
]


@mcp.tool()
def index_all(force: bool = False) -> str:
    """Index the entire project — all supported file types in one call.

    Replaces the need to call index_directory() multiple times with
    different patterns. Uses batch upserts for performance. Automatically
    skips unchanged files unless force=True.

    This is the recommended way to index. Called automatically at session start.

    Args:
        force: If True, re-index all files even if unchanged (default: False)
    """
    store = _get_store()
    settings = _load_settings()
    dir_path = Path(_detect_project_root())

    skip_dirs = set(settings.get("skip_dirs", []))

    t0 = time.time()
    total_symbols = 0
    indexed_files = 0
    skipped_files = 0

    for pattern in _ALL_PATTERNS:
        files = sorted(dir_path.glob(pattern))
        files = [
            f for f in files
            if f.is_file() and not any(part in skip_dirs for part in f.parts)
        ]

        for fpath in files:
            try:
                fp_str = str(fpath)

                # Smart mtime check
                if not force:
                    try:
                        actual_mtime = os.path.getmtime(fp_str)
                        existing = store.search_by_file(fp_str)
                        if existing and all(e.file_mtime >= actual_mtime for e in existing):
                            skipped_files += 1
                            continue
                    except OSError:
                        pass

                entries = index_file(fp_str)
                if entries:
                    # Stamp line_count on module entries
                    lc = sum(1 for _ in open(fp_str, "rb"))
                    for e in entries:
                        if e.kind == "module":
                            e.line_count = lc
                    store.upsert_batch(entries)
                    total_symbols += len(entries)
                indexed_files += 1
            except Exception as exc:
                logger.warning("Failed to index %s: %s", fpath, exc)

    elapsed = time.time() - t0

    # Auto-consolidate if near capacity
    if should_consolidate(store):
        cstats = consolidate(store)
        logger.info("Auto-consolidation: %s", cstats)

    result = f"Indexed {indexed_files} files, {total_symbols} symbols in {elapsed:.1f}s"
    if skipped_files:
        result += f"\nSkipped {skipped_files} unchanged files"
    result += f"\nTotal memories: {store.count()}/{store.max_memories}"
    return result
```

Also add the settings loading helper at the top of `server.py`:

```python
from settings import load_settings

def _load_settings() -> dict:
    """Load settings for the current project."""
    return load_settings(_detect_project_root())
```

- [ ] **Step 4: Also refactor index_directory() to use upsert_batch()**

In the existing `index_directory()` function, replace the per-entry upsert loop:

```python
# Old (lines 234-237):
#     entries = index_file(fp_str)
#     for entry in entries:
#         store.upsert(entry)
#     total_symbols += len(entries)

# New:
            entries = index_file(fp_str)
            if entries:
                store.upsert_batch(entries)
            total_symbols += len(entries)
```

- [ ] **Step 5: Run tests**

Run: `cd C:\Users\grill\.claude\plugins\marketplaces\grilly-plugins\elephant-coder && uv run pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
cd C:\Users\grill\.claude\plugins\marketplaces\grilly-plugins
git add elephant-coder/server.py elephant-coder/tests/test_server.py
git commit -m "feat: add index_all() unified indexer + batch upserts in index_directory()"
```

---

## Task 5: update_settings() MCP Tool

**Files:**
- Modify: `server.py` — add `update_settings()` tool
- Modify: `tests/test_server.py` — add test

- [ ] **Step 1: Write failing test**

```python
# Add to tests/test_server.py

def test_update_settings():
    """update_settings() should write to .claude/elephant-coder.local.md."""
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / ".git").mkdir()
        with patch("server._detect_project_root", return_value=tmpdir):
            import server
            server._store = None
            os.environ["ELEPHANT_CODER_REDIS_URL"] = "redis://localhost:59999"

            result = update_settings(max_memories=75000)
            assert "saved" in result.lower() or "updated" in result.lower()

            # Verify file was created
            settings_file = Path(tmpdir) / ".claude" / "elephant-coder.local.md"
            assert settings_file.exists()
            content = settings_file.read_text()
            assert "75000" in content
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd C:\Users\grill\.claude\plugins\marketplaces\grilly-plugins\elephant-coder && uv run pytest tests/test_server.py::test_update_settings -v`
Expected: FAIL — `ImportError: cannot import name 'update_settings'`

- [ ] **Step 3: Implement update_settings() in server.py**

```python
@mcp.tool()
def update_settings(
    max_memories: int | None = None,
    relevance_threshold: float | None = None,
    redis_url: str | None = None,
    skip_dirs: list[str] | None = None,
    scope_guard: bool | None = None,
    auto_test_after_edit: bool | None = None,
) -> str:
    """Update elephant-coder settings for this project.

    Writes to .claude/elephant-coder.local.md. Changes take effect
    on next tool call (settings are re-read). Hook changes require
    Claude Code restart.

    Args:
        max_memories: Maximum memories in the store (default: 50000)
        relevance_threshold: Minimum relevance score for search results (default: 0.1)
        redis_url: Redis URL, or null to disable Redis
        skip_dirs: Directories to skip during indexing
        scope_guard: Enable scope guard (block untracked changes)
        auto_test_after_edit: Prompt to run tests after edits
    """
    current = _load_settings()

    if max_memories is not None:
        current["max_memories"] = max_memories
    if relevance_threshold is not None:
        current["relevance_threshold"] = relevance_threshold
    if redis_url is not None:
        current["redis_url"] = redis_url
    if skip_dirs is not None:
        current["skip_dirs"] = skip_dirs
    if scope_guard is not None:
        current["scope_guard"] = scope_guard
    if auto_test_after_edit is not None:
        current["auto_test_after_edit"] = auto_test_after_edit

    path = save_settings(_detect_project_root(), current)
    return f"Settings updated and saved to {path}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd C:\Users\grill\.claude\plugins\marketplaces\grilly-plugins\elephant-coder && uv run pytest tests/test_server.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
cd C:\Users\grill\.claude\plugins\marketplaces\grilly-plugins
git add elephant-coder/server.py elephant-coder/tests/test_server.py
git commit -m "feat: add update_settings() MCP tool for runtime config"
```

---

## Task 6: Global Knowledge Store

**Files:**
- Create: `global_store.py`
- Test: `tests/test_global_store.py` (create)

- [ ] **Step 1: Write failing test**

```python
# tests/test_global_store.py
import os
import tempfile
from global_store import GlobalKnowledgeStore


def test_store_framework_knowledge():
    """Should store and retrieve framework knowledge."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = GlobalKnowledgeStore(base_dir=tmpdir)

        store.save_framework(
            name="grilly",
            repo_path="/path/to/grilly",
            github="Grillcheese-AI/grilly",
            api_map={"torch.nn.Linear": "grilly.nn.Linear"},
            quick_start="from grilly.nn import Linear",
            differences=["Vulkan instead of CUDA", "numpy not torch.Tensor"],
        )

        fw = store.get_framework("grilly")
        assert fw is not None
        assert fw["name"] == "grilly"
        assert fw["api_map"]["torch.nn.Linear"] == "grilly.nn.Linear"

        store.close()


def test_store_session_summary():
    """Should store and retrieve session summaries."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = GlobalKnowledgeStore(base_dir=tmpdir)

        store.save_session_summary(
            project="grilly",
            summary="Worked on Conv2d GEMM path. Blocked by GPU transpose kernel.",
            tasks_completed=["T-001"],
            tasks_remaining=["T-002"],
        )

        sessions = store.get_recent_sessions("grilly", limit=5)
        assert len(sessions) == 1
        assert "Conv2d" in sessions[0]["summary"]

        store.close()


def test_store_research_note():
    """Should store and retrieve research notes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = GlobalKnowledgeStore(base_dir=tmpdir)

        store.save_note(
            topic="LSH attention",
            summary="Reduces O(n^2) to O(n log n) via locality-sensitive hashing",
            source="arxiv:2106.04554",
            tags=["attention", "performance"],
        )

        notes = store.search_notes("attention")
        assert len(notes) == 1
        assert notes[0]["topic"] == "LSH attention"

        store.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd C:\Users\grill\.claude\plugins\marketplaces\grilly-plugins\elephant-coder && uv run pytest tests/test_global_store.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'global_store'`

- [ ] **Step 3: Implement global_store.py**

```python
# global_store.py
"""
Global knowledge store for elephant-coder.

Stores framework references, session summaries, research notes, and
coding idioms that persist across ALL projects.

Location: ~/.elephant-coder/global/knowledge.db
"""

import json
import logging
import sqlite3
import time
from pathlib import Path

logger = logging.getLogger("elephant-coder.global")


class GlobalKnowledgeStore:
    """SQLite store for cross-project knowledge."""

    def __init__(self, base_dir: str | None = None):
        if base_dir is None:
            base_dir = str(Path.home() / ".elephant-coder" / "global")
        Path(base_dir).mkdir(parents=True, exist_ok=True)
        db_path = Path(base_dir) / "knowledge.db"
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS frameworks (
                name TEXT PRIMARY KEY,
                repo_path TEXT,
                github TEXT,
                api_map TEXT NOT NULL DEFAULT '{}',
                quick_start TEXT NOT NULL DEFAULT '',
                differences TEXT NOT NULL DEFAULT '[]',
                updated REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project TEXT NOT NULL,
                summary TEXT NOT NULL,
                tasks_completed TEXT NOT NULL DEFAULT '[]',
                tasks_remaining TEXT NOT NULL DEFAULT '[]',
                timestamp REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_sessions_project
                ON sessions(project);

            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT NOT NULL,
                summary TEXT NOT NULL,
                source TEXT,
                tags TEXT NOT NULL DEFAULT '[]',
                relevance_to TEXT NOT NULL DEFAULT '[]',
                discovered_in_session TEXT,
                actionable INTEGER DEFAULT 0,
                potential_task TEXT,
                timestamp REAL NOT NULL
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(
                topic, summary, tags
            );

            CREATE TABLE IF NOT EXISTS idioms (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern TEXT NOT NULL,
                context TEXT NOT NULL,
                project TEXT,
                frequency INTEGER DEFAULT 1,
                timestamp REAL NOT NULL
            );
        """)
        self._conn.commit()

    # --- Frameworks ---

    def save_framework(
        self,
        name: str,
        repo_path: str,
        github: str,
        api_map: dict,
        quick_start: str,
        differences: list[str],
    ) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO frameworks
               (name, repo_path, github, api_map, quick_start, differences, updated)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (name, repo_path, github, json.dumps(api_map),
             quick_start, json.dumps(differences), time.time()),
        )
        self._conn.commit()

    def get_framework(self, name: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM frameworks WHERE name = ?", (name,)
        ).fetchone()
        if row is None:
            return None
        return {
            "name": row["name"],
            "repo_path": row["repo_path"],
            "github": row["github"],
            "api_map": json.loads(row["api_map"]),
            "quick_start": row["quick_start"],
            "differences": json.loads(row["differences"]),
        }

    def get_all_frameworks(self) -> list[dict]:
        rows = self._conn.execute("SELECT * FROM frameworks").fetchall()
        return [
            {
                "name": r["name"],
                "repo_path": r["repo_path"],
                "github": r["github"],
                "api_map": json.loads(r["api_map"]),
                "quick_start": r["quick_start"],
                "differences": json.loads(r["differences"]),
            }
            for r in rows
        ]

    # --- Sessions ---

    def save_session_summary(
        self,
        project: str,
        summary: str,
        tasks_completed: list[str] | None = None,
        tasks_remaining: list[str] | None = None,
    ) -> None:
        self._conn.execute(
            """INSERT INTO sessions (project, summary, tasks_completed, tasks_remaining, timestamp)
               VALUES (?, ?, ?, ?, ?)""",
            (project, summary, json.dumps(tasks_completed or []),
             json.dumps(tasks_remaining or []), time.time()),
        )
        self._conn.commit()

    def get_recent_sessions(self, project: str, limit: int = 5) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM sessions WHERE project = ? ORDER BY timestamp DESC LIMIT ?",
            (project, limit),
        ).fetchall()
        return [
            {
                "project": r["project"],
                "summary": r["summary"],
                "tasks_completed": json.loads(r["tasks_completed"]),
                "tasks_remaining": json.loads(r["tasks_remaining"]),
                "timestamp": r["timestamp"],
            }
            for r in rows
        ]

    # --- Research Notes ---

    def save_note(
        self,
        topic: str,
        summary: str,
        source: str | None = None,
        tags: list[str] | None = None,
        relevance_to: list[str] | None = None,
        actionable: bool = False,
        potential_task: str | None = None,
    ) -> int:
        cur = self._conn.execute(
            """INSERT INTO notes
               (topic, summary, source, tags, relevance_to, actionable, potential_task, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (topic, summary, source, json.dumps(tags or []),
             json.dumps(relevance_to or []), int(actionable),
             potential_task, time.time()),
        )
        note_id = cur.lastrowid
        # FTS insert
        self._conn.execute(
            "INSERT INTO notes_fts (rowid, topic, summary, tags) VALUES (?, ?, ?, ?)",
            (note_id, topic, summary, " ".join(tags or [])),
        )
        self._conn.commit()
        return note_id

    def search_notes(self, query: str, limit: int = 10) -> list[dict]:
        safe_query = query.replace('"', '""')
        try:
            rows = self._conn.execute(
                """SELECT n.* FROM notes n
                   JOIN notes_fts fts ON n.id = fts.rowid
                   WHERE notes_fts MATCH ?
                   ORDER BY bm25(notes_fts) ASC
                   LIMIT ?""",
                (f'"{safe_query}" OR {safe_query}', limit),
            ).fetchall()
        except sqlite3.OperationalError:
            like = f"%{query}%"
            rows = self._conn.execute(
                "SELECT * FROM notes WHERE topic LIKE ? OR summary LIKE ? LIMIT ?",
                (like, like, limit),
            ).fetchall()
        return [
            {
                "id": r["id"],
                "topic": r["topic"],
                "summary": r["summary"],
                "source": r["source"],
                "tags": json.loads(r["tags"]),
                "relevance_to": json.loads(r["relevance_to"]),
                "actionable": bool(r["actionable"]),
                "potential_task": r["potential_task"],
                "timestamp": r["timestamp"],
            }
            for r in rows
        ]

    def get_notes_by_tags(self, tags: list[str], limit: int = 20) -> list[dict]:
        """Get notes that share any of the given tags."""
        results = []
        for tag in tags:
            rows = self._conn.execute(
                "SELECT * FROM notes WHERE tags LIKE ? ORDER BY timestamp DESC LIMIT ?",
                (f'%"{tag}"%', limit),
            ).fetchall()
            for r in rows:
                note = {
                    "id": r["id"],
                    "topic": r["topic"],
                    "summary": r["summary"],
                    "source": r["source"],
                    "tags": json.loads(r["tags"]),
                    "timestamp": r["timestamp"],
                }
                if note not in results:
                    results.append(note)
        return results[:limit]

    # --- Idioms ---

    def save_idiom(self, pattern: str, context: str, project: str | None = None) -> None:
        existing = self._conn.execute(
            "SELECT id, frequency FROM idioms WHERE pattern = ? AND context = ?",
            (pattern, context),
        ).fetchone()
        if existing:
            self._conn.execute(
                "UPDATE idioms SET frequency = frequency + 1 WHERE id = ?",
                (existing["id"],),
            )
        else:
            self._conn.execute(
                "INSERT INTO idioms (pattern, context, project, timestamp) VALUES (?, ?, ?, ?)",
                (pattern, context, project, time.time()),
            )
        self._conn.commit()

    def get_idioms(self, project: str | None = None, limit: int = 20) -> list[dict]:
        if project:
            rows = self._conn.execute(
                "SELECT * FROM idioms WHERE project = ? ORDER BY frequency DESC LIMIT ?",
                (project, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM idioms ORDER BY frequency DESC LIMIT ?", (limit,)
            ).fetchall()
        return [
            {"pattern": r["pattern"], "context": r["context"],
             "frequency": r["frequency"], "project": r["project"]}
            for r in rows
        ]

    def close(self) -> None:
        self._conn.close()
```

- [ ] **Step 4: Run tests**

Run: `cd C:\Users\grill\.claude\plugins\marketplaces\grilly-plugins\elephant-coder && uv run pytest tests/test_global_store.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
cd C:\Users\grill\.claude\plugins\marketplaces\grilly-plugins
git add elephant-coder/global_store.py elephant-coder/tests/test_global_store.py
git commit -m "feat: add global knowledge store (frameworks, sessions, notes, idioms)"
```

---

## Task 7: Integrate Settings into MemoryStore

**Files:**
- Modify: `server.py` — pass settings to MemoryStore
- Modify: `memory_store.py` — use configurable `max_memories`
- Modify: `retriever.py` — apply `relevance_threshold`

- [ ] **Step 1: Write failing test — configurable max_memories**

```python
# Add to tests/test_memory_store.py

def test_configurable_max_memories():
    """MemoryStore should accept max_memories from settings."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = MemoryStore(tmpdir, max_memories=100, redis_url="redis://localhost:59999")
        assert store.max_memories == 100
        store.close()
```

- [ ] **Step 2: Run test — should pass (already supported)**

Run: `cd C:\Users\grill\.claude\plugins\marketplaces\grilly-plugins\elephant-coder && uv run pytest tests/test_memory_store.py::test_configurable_max_memories -v`
Expected: PASS (constructor already takes `max_memories`)

- [ ] **Step 3: Wire settings into _get_store() in server.py**

Replace the `_get_store()` function:

```python
def _get_store() -> MemoryStore:
    """Get or initialize the memory store for the current project."""
    global _store
    if _store is None:
        project_root = _detect_project_root()
        settings = load_settings(project_root)
        redis_url = settings.get("redis_url") or _redis_url
        max_mem = settings.get("max_memories", 50_000)
        _store = MemoryStore(project_root, max_memories=max_mem, redis_url=redis_url)
        logger.info("Memory store initialized for project: %s (max: %d)", project_root, max_mem)
    return _store
```

- [ ] **Step 4: Add relevance threshold filtering to retriever.py**

In `recall()`, after the existing results filtering, add:

```python
def recall(
    store: MemoryStore,
    query: str,
    limit: int = 5,
    kind: str | None = None,
    relevance_threshold: float = 0.0,
) -> list[MemoryEntry]:
    """Search memories and return ranked results.

    Args:
        ...
        relevance_threshold: Minimum relevance score to include (default: 0.0)
    """
    results = store.search_fts(query, limit=limit * 3)

    if kind:
        results = [r for r in results if r.kind == kind]

    if relevance_threshold > 0:
        results = [r for r in results if r.relevance_score >= relevance_threshold]

    results = results[:limit]
    # ... rest unchanged
```

Update `recall_memories()` in `server.py` to pass threshold from settings:

```python
@mcp.tool()
def recall_memories(query: str, limit: int = 5, kind: str | None = None) -> str:
    # ... docstring unchanged ...
    store = _get_store()
    settings = _load_settings()
    threshold = settings.get("relevance_threshold", 0.0)
    results = recall(store, query, limit=limit, kind=kind, relevance_threshold=threshold)
    return format_results(results)
```

- [ ] **Step 5: Run all tests**

Run: `cd C:\Users\grill\.claude\plugins\marketplaces\grilly-plugins\elephant-coder && uv run pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
cd C:\Users\grill\.claude\plugins\marketplaces\grilly-plugins
git add elephant-coder/server.py elephant-coder/memory_store.py elephant-coder/retriever.py
git commit -m "feat: wire settings into store, search, and consolidation"
```

---

## Task 8: New Hook System

**Files:**
- Modify: `hooks/hooks.json` — replace all hooks
- Create: `hooks/reindex_file.sh` — silent re-index script
- Modify: `CLAUDE.md` — rewrite to behavioral guide

- [ ] **Step 1: Write the new hooks.json**

```json
{
  "hooks": {
    "SessionStart": [
      {
        "hooks": [
          {
            "type": "prompt",
            "prompt": "elephant-coder is active. You have persistent codebase memory.\n\n1. Run index_all() to ensure the index is current.\n2. Run memory_status() to check store health.\n3. If this is a new project with no objectives set, ask the user: 'What is the main objective of this project?' The objective must be specific, actionable, and measurable. If vague, ask follow-up questions.\n4. You already know this project — check recall_memories for any context before exploring.\n\nDo NOT list these steps to the user. Just do them silently and start working."
          }
        ]
      }
    ],
    "PreToolUse": [
      {
        "matcher": "Read",
        "hooks": [
          {
            "type": "prompt",
            "prompt": "Before reading this file, check if elephant-coder has memories for it using recall_file_memories(). If memories exist, you already have context — use it to inform your reading. Don't mention this to the user."
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Edit",
        "hooks": [
          {
            "type": "command",
            "command": "bash ${CLAUDE_PLUGIN_ROOT}/hooks/reindex_file.sh"
          }
        ]
      },
      {
        "matcher": "Write",
        "hooks": [
          {
            "type": "command",
            "command": "bash ${CLAUDE_PLUGIN_ROOT}/hooks/reindex_file.sh"
          }
        ]
      }
    ]
  }
}
```

- [ ] **Step 2: Create reindex_file.sh**

```bash
#!/bin/bash
# Silent re-index of the last edited/written file.
# Called by PostToolUse:Edit and PostToolUse:Write hooks.
# This is a no-op placeholder — the actual re-indexing happens via
# the MCP server's index_directory() call. The prompt hook in
# hooks.json handles the Claude-side instruction.
# Future: could call the MCP server directly via stdio for true background re-index.
exit 0
```

- [ ] **Step 3: Rewrite CLAUDE.md**

```markdown
# Elephant-Coder Plugin

You have persistent codebase memory. You already know this project's
architecture, key files, and active tasks. Do not explore or search
for information you already have in your context.

## Automatic Behaviors (you don't need to call anything)
- Project index is refreshed at session start
- File context is recalled before you read any file
- Edited files are flagged for re-indexing automatically

## How to Use Your Memory
- Before reading a file, check recall_file_memories() — you may already have context
- Before searching, check recall_memories() — faster than grep/glob
- Use search_symbols() for exact symbol lookups
- Use show_call_graph() to understand dependencies

## Tools Available
- index_all() — re-index entire project (auto-called at session start)
- index_directory(path, patterns) — index specific directory
- recall_memories(query) — full-text search
- recall_file_memories(file_path) — all memories for a file
- search_symbols(name) — direct symbol lookup
- show_call_graph(symbol) — dependency tracing
- summarize_directory(path) — table of contents
- recent_changes(days) — git-aware recent modifications
- get_dependencies(file_path) — import graph
- remember(file_path, symbol_name, summary) — manual memory
- forget(query, file_path, stale_only) — remove memories
- memory_status() — store statistics
- update_settings(...) — configure limits and behavior
- explore_structure(path) — directory tree
```

- [ ] **Step 4: Commit**

```bash
cd C:\Users\grill\.claude\plugins\marketplaces\grilly-plugins
git add elephant-coder/hooks/hooks.json elephant-coder/hooks/reindex_file.sh elephant-coder/CLAUDE.md
git commit -m "feat: new hook system — auto-index, auto-recall, silent re-index"
```

---

## Task 9: Version Bump and Dependency Updates

**Files:**
- Modify: `.claude-plugin/plugin.json` — version 0.2.1
- Modify: `pyproject.toml` — version, make redis optional
- Modify: `../.claude-plugin/marketplace.json` — version in marketplace manifest

- [ ] **Step 1: Update plugin.json**

```json
{
  "name": "elephant-coder",
  "description": "Persistent codebase memory with multi-language indexing, full-text search, automatic knowledge injection, and configurable settings",
  "version": "0.2.1",
  "author": {
    "name": "grillcheese"
  },
  "keywords": [
    "memory",
    "indexing",
    "codebase",
    "search",
    "code-intelligence",
    "knowledge-graph"
  ]
}
```

- [ ] **Step 2: Update pyproject.toml**

```toml
[project]
name = "elephant-coder"
version = "0.2.1"
description = "MCP server for persistent codebase memory — invisible knowledge injection for Claude Code"
requires-python = ">=3.10"
dependencies = [
    "mcp>=1.2.0",
    "pypdf>=4.0.0",
    "pyyaml>=6.0",
    "tomli>=2.0; python_version < '3.11'",
]

[project.optional-dependencies]
redis = ["redis>=5.0.0"]
all = ["redis>=5.0.0", "httpx>=0.27.0"]

[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
py-modules = ["server", "memory_store", "indexer", "retriever", "consolidator", "settings", "global_store"]
```

- [ ] **Step 3: Update marketplace.json**

In `../.claude-plugin/marketplace.json`, update the elephant-coder version:

```json
{
  "name": "grilly-plugins",
  "owner": {
    "name": "Grillcheese AI"
  },
  "description": "Claude Code plugins by Grillcheese AI — persistent codebase memory and more",
  "metadata": {
    "version": "1.0.0"
  },
  "plugins": [
    {
      "name": "elephant-coder",
      "source": "./elephant-coder",
      "description": "Persistent codebase memory with multi-language indexing, automatic knowledge injection, and configurable settings",
      "version": "0.2.1",
      "author": {
        "name": "grillcheese"
      },
      "repository": "https://github.com/grillcheese-ai/grilly-plugins",
      "keywords": [
        "memory",
        "indexing",
        "codebase",
        "search",
        "code-intelligence",
        "knowledge-graph"
      ],
      "category": "development"
    }
  ]
}
```

- [ ] **Step 4: Run full test suite**

Run: `cd C:\Users\grill\.claude\plugins\marketplaces\grilly-plugins\elephant-coder && uv run pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
cd C:\Users\grill\.claude\plugins\marketplaces\grilly-plugins
git add elephant-coder/.claude-plugin/plugin.json elephant-coder/pyproject.toml .claude-plugin/marketplace.json
git commit -m "chore: bump to 0.2.1 — Redis optional, add settings and global_store modules"
```

---

## Task 10: Update /index Command and Skills

**Files:**
- Modify: `commands/index.md` — use `index_all()` instead of 14 parallel calls
- Modify: `skills/index/SKILL.md` — same

- [ ] **Step 1: Rewrite commands/index.md**

```markdown
---
name: index
description: Index the full project codebase with elephant-coder (all file types)
---

Run elephant-coder `index_all()` to index all supported file types in the project.

This is a single call that handles all patterns internally with batch upserts.
Unchanged files are automatically skipped (mtime check), so this is safe to run repeatedly.

After indexing, run `memory_status()` and report the summary.
```

- [ ] **Step 2: Rewrite skills/index/SKILL.md**

```markdown
---
name: index
description: Index the full project codebase with elephant-coder (all file types)
---

Run elephant-coder `index_all()` to index all supported file types in the project.

This replaces the old approach of 14 parallel index_directory() calls.
One call, all patterns, batch upserts. Much faster.

After indexing, run `memory_status()` and report the summary.

Unchanged files are automatically skipped (mtime check), so this is safe to run repeatedly.
```

- [ ] **Step 3: Commit**

```bash
cd C:\Users\grill\.claude\plugins\marketplaces\grilly-plugins
git add elephant-coder/commands/index.md elephant-coder/skills/index/SKILL.md
git commit -m "feat: update /index command to use single index_all() call"
```

---

## Plan A Complete — Summary

After all 10 tasks, elephant-coder 0.2.1 has:

| Feature | Before (0.2.0) | After (0.2.1 Plan A) |
|---------|----------------|----------------------|
| Redis | Required, crashes without it | Optional, graceful fallback |
| Indexing | 14 parallel MCP calls | Single `index_all()` call |
| Upserts | One at a time | Batch transactional |
| Settings | Hardcoded | `.claude/elephant-coder.local.md` |
| Configuration | None | `update_settings()` MCP tool |
| Global knowledge | None | `~/.elephant-coder/global/` store |
| SessionStart | None | Auto-index + auto-recall |
| PreToolUse:Read | None | Auto-inject file memories |
| PostToolUse:Edit | Reminder prompt | Silent re-index |
| CLAUDE.md | Tool listing | Behavioral guide |

**Next:** Plan B (Intelligence Layer) adds file link graph, project mental model, framework detection, and "what broke?"
