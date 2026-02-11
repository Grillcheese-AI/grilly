"""
Elephant-Coder MCP Server — persistent codebase memory for Claude Code.

Inspired by the hippocampal memory circuit in grilly/nn/hippocampal.py:
- Capsule encoding: compress source files into compact AST summaries
- DG pattern separation: extract discriminative keywords for FTS5 indexing
- CA3 pattern completion: retrieve full context from partial queries
- Cognitive metadata: track access frequency, relevance, freshness
- Consolidation: evict stale/low-relevance memories, keep capacity bounded

Register with Claude Code:
    claude mcp add --transport stdio elephant-coder -- python server.py
"""

import logging
import os
import sys
import time
from pathlib import Path

from consolidator import consolidate, detect_stale, should_consolidate
from indexer import index_file
from mcp.server.fastmcp import FastMCP
from memory_store import MemoryEntry, MemoryStore, make_memory_id
from retriever import format_results, recall, recall_file

# Logging to stderr only (stdout reserved for MCP stdio transport)
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("elephant-coder")

# ------------------------------------------------------------------
# Server setup
# ------------------------------------------------------------------

mcp = FastMCP("elephant-coder")

# Lazy-initialized store (needs project root at first tool call)
_store: MemoryStore | None = None


def _get_store() -> MemoryStore:
    """Get or initialize the memory store for the current project."""
    global _store
    if _store is None:
        project_root = _detect_project_root()
        _store = MemoryStore(project_root)
        logger.info("Memory store initialized for project: %s", project_root)
    return _store


def _detect_project_root() -> str:
    """Walk up from cwd to find a project root (has .git or pyproject.toml)."""
    cwd = Path.cwd()
    for parent in [cwd, *cwd.parents]:
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            return str(parent)
    return str(cwd)


def _normalize_path(path: str) -> str:
    """Resolve a path relative to project root if not absolute."""
    p = Path(path)
    if not p.is_absolute():
        root = _detect_project_root()
        p = Path(root) / p
    return str(p.resolve())


# ------------------------------------------------------------------
# MCP Tools
# ------------------------------------------------------------------


@mcp.tool()
def remember(
    file_path: str,
    symbol_name: str,
    summary: str,
    kind: str = "note",
    keywords: list[str] | None = None,
) -> str:
    """Store a memory about code you have explored or understood.

    Call this after reading and understanding a file or function to save
    a compressed summary for future retrieval. This avoids re-reading
    the same code in future conversations.

    Args:
        file_path: Path to the source file this memory relates to
        symbol_name: Name of the function, class, or concept
        summary: Compressed summary of what this code does
        kind: One of "function", "class", "module", "file_summary", "note"
        keywords: Optional list of search keywords
    """
    store = _get_store()
    fp = _normalize_path(file_path)

    try:
        mtime = os.path.getmtime(fp)
    except OSError:
        mtime = 0.0

    entry = MemoryEntry(
        memory_id=make_memory_id(fp, symbol_name, kind),
        file_path=fp,
        symbol_name=symbol_name,
        kind=kind,
        summary=summary,
        keywords=keywords or [],
        file_mtime=mtime,
    )
    store.upsert(entry)
    return f"Remembered: {kind} '{symbol_name}' in {file_path} (id: {entry.memory_id})"


@mcp.tool()
def recall_memories(query: str, limit: int = 5, kind: str | None = None) -> str:
    """Retrieve relevant memories about the codebase.

    Search stored memories using full-text search. Returns compressed
    summaries so you don't need to re-read the full source files.
    Call this before reading files to check if you already have context.

    Args:
        query: Search query (keywords, function names, concepts)
        limit: Maximum number of results to return (default 5)
        kind: Optional filter by kind ("function", "class", "module", etc.)
    """
    store = _get_store()
    results = recall(store, query, limit=limit, kind=kind)
    return format_results(results)


@mcp.tool()
def recall_file_memories(file_path: str) -> str:
    """Retrieve all memories about a specific file.

    Returns all stored summaries for functions, classes, and the module
    summary for the given file. Useful to get an overview before diving
    into the file.

    Args:
        file_path: Path to the file to recall memories about
    """
    store = _get_store()
    fp = _normalize_path(file_path)
    results = recall_file(store, fp)
    if not results:
        return f"No memories stored for {file_path}. Use index_directory or remember to add some."
    return format_results(results)


@mcp.tool()
def index_directory(
    path: str = ".",
    patterns: str = "**/*.py",
    max_files: int = 200,
) -> str:
    """Index Python files in a directory, extracting function/class summaries.

    Walks the directory, parses each file's AST, and stores compressed
    summaries of every function, class, and module. Use this when starting
    work on a new codebase or directory.

    Args:
        path: Directory path to index (default: current directory)
        patterns: Glob pattern for files to include (default: **/*.py)
        max_files: Maximum number of files to index in one call
    """
    store = _get_store()
    dir_path = Path(_normalize_path(path))

    if not dir_path.is_dir():
        return f"Error: {path} is not a directory."

    # Collect matching files
    files = sorted(dir_path.glob(patterns))
    # Skip common non-source directories
    skip_dirs = {
        "__pycache__",
        ".git",
        ".venv",
        "venv",
        "node_modules",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
        "dist",
        "build",
        ".eggs",
        "*.egg-info",
    }
    files = [f for f in files if f.is_file() and not any(part in skip_dirs for part in f.parts)]
    files = files[:max_files]

    t0 = time.time()
    total_symbols = 0
    indexed_files = 0

    for fpath in files:
        try:
            entries = index_file(str(fpath))
            for entry in entries:
                store.upsert(entry)
            total_symbols += len(entries)
            indexed_files += 1
        except Exception as exc:
            logger.warning("Failed to index %s: %s", fpath, exc)

    elapsed = time.time() - t0

    # Auto-consolidate if near capacity
    if should_consolidate(store):
        cstats = consolidate(store)
        logger.info("Auto-consolidation: %s", cstats)

    return (
        f"Indexed {indexed_files} files, {total_symbols} symbols in {elapsed:.1f}s\n"
        f"Total memories: {store.count()}/{store.max_memories}"
    )


@mcp.tool()
def explore_structure(path: str = ".", max_depth: int = 3) -> str:
    """Explore and summarize the directory structure of a codebase.

    Walks the directory tree and returns a structured overview including
    file counts per directory and identified patterns (tests, configs, etc.).

    Args:
        path: Root directory to explore (default: current directory)
        max_depth: Maximum depth to traverse (default: 3)
    """
    dir_path = Path(_normalize_path(path))
    if not dir_path.is_dir():
        return f"Error: {path} is not a directory."

    skip_dirs = {
        "__pycache__",
        ".git",
        ".venv",
        "venv",
        "node_modules",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
        "dist",
        "build",
        ".eggs",
    }

    lines = [f"Project structure: {dir_path.name}/"]
    _walk_tree(dir_path, dir_path, lines, skip_dirs, max_depth, depth=0)
    return "\n".join(lines)


def _walk_tree(
    root: Path,
    current: Path,
    lines: list[str],
    skip_dirs: set[str],
    max_depth: int,
    depth: int,
) -> None:
    """Recursively build directory tree with file counts."""
    if depth > max_depth:
        return

    indent = "  " * depth
    try:
        entries = sorted(current.iterdir(), key=lambda e: (not e.is_dir(), e.name))
    except PermissionError:
        return

    dirs = [e for e in entries if e.is_dir() and e.name not in skip_dirs]
    files = [e for e in entries if e.is_file()]

    # Summarize files by extension
    ext_counts: dict[str, int] = {}
    for f in files:
        ext = f.suffix or "(no ext)"
        ext_counts[ext] = ext_counts.get(ext, 0) + 1

    if ext_counts and depth > 0:
        summary = ", ".join(f"{c} {ext}" for ext, c in sorted(ext_counts.items()))
        lines.append(f"{indent}{current.name}/  [{summary}]")
    elif depth > 0:
        lines.append(f"{indent}{current.name}/")

    for d in dirs:
        _walk_tree(root, d, lines, skip_dirs, max_depth, depth + 1)


@mcp.tool()
def forget(
    query: str | None = None,
    file_path: str | None = None,
    stale_only: bool = False,
) -> str:
    """Remove memories. Use to clear stale or irrelevant context.

    Args:
        query: Remove memories matching this search query
        file_path: Remove all memories for this file path
        stale_only: If True, only remove memories whose files have changed on disk
    """
    store = _get_store()

    if stale_only:
        detect_stale(store)
        count = store.delete_stale()
        return f"Removed {count} stale memories."

    if file_path:
        fp = _normalize_path(file_path)
        count = store.delete_by_file(fp)
        return f"Removed {count} memories for {file_path}."

    if query:
        results = store.search_fts(query, limit=100)
        count = 0
        for entry in results:
            if store.delete(entry.memory_id):
                count += 1
        return f"Removed {count} memories matching '{query}'."

    return "Specify query, file_path, or stale_only=True."


@mcp.tool()
def memory_status() -> str:
    """Get statistics about the memory store.

    Returns total memories, breakdown by kind, staleness status,
    most accessed memories, and storage utilization.
    """
    store = _get_store()

    # Detect staleness before reporting
    detect_stale(store)

    s = store.stats()
    lines = [
        "Memory Store Status",
        f"  Total: {s['total']}/{s['max_capacity']} ({s['utilization_pct']}% utilized)",
        f"  Stale: {s['stale']}",
        "",
        "  By kind:",
    ]
    for kind, count in sorted(s["by_kind"].items()):
        lines.append(f"    {kind}: {count}")

    if s["top_accessed"]:
        lines.append("")
        lines.append("  Most accessed:")
        for item in s["top_accessed"]:
            lines.append(f"    {item['symbol']} ({item['file']}) — {item['count']}x")

    return "\n".join(lines)


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="stdio")
