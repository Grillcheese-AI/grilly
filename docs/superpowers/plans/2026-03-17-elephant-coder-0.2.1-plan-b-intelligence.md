# elephant-coder 0.2.1 Plan B: Intelligence Layer

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add resolved file link graph, project mental model generation, framework detection with global knowledge export, and "what broke?" semantic diff tool.

**Architecture:** A new `file_links` SQLite table tracks resolved import/include/shader relationships between files. The `mental_model.py` module reads indexed memories + the link graph to generate a project overview. `framework_detector.py` auto-detects frameworks (grilly, etc.) and exports API maps to the global knowledge store. All powered by the existing indexer infrastructure from Plan A.

**Tech Stack:** Python 3.10+, SQLite, existing indexer/memory_store from Plan A

**Spec:** `docs/superpowers/specs/2026-03-17-elephant-coder-0.2.1-design.md` (Sections 2, 3, 6, Additional Features)

**Plugin source:** `C:\Users\grill\grilly-plugins\elephant-coder\`

**Depends on:** Plan A (complete)

---

## File Structure

### New Files

| File | Responsibility |
|------|---------------|
| `link_graph.py` | file_links table schema, population during indexing, query methods (hub detection, impact analysis) |
| `mental_model.py` | Project overview generation from memories + link graph + git |
| `framework_detector.py` | Auto-detect installed frameworks, generate API maps, export to global store |
| `tests/test_link_graph.py` | Link graph tests |
| `tests/test_mental_model.py` | Mental model tests |
| `tests/test_framework_detector.py` | Framework detector tests |

### Modified Files

| File | Changes |
|------|---------|
| `memory_store.py` | Add `file_links` table to schema, add link CRUD methods |
| `indexer.py` | Extract imports/includes during indexing, return link data |
| `server.py` | Add `project_overview()`, `what_broke()` MCP tools, integrate framework detection into `index_all()` |

---

## Task 1: file_links Table in MemoryStore

**Files:**
- Modify: `C:\Users\grill\grilly-plugins\elephant-coder\memory_store.py`
- Create: `C:\Users\grill\grilly-plugins\elephant-coder\tests\test_link_graph.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_link_graph.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import tempfile
from memory_store import MemoryStore


def test_add_and_query_file_links():
    """Should store and query file-to-file links."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = MemoryStore(tmpdir, redis_url="redis://localhost:59999")
        store.add_file_link("/src/main.py", "/src/utils.py", "import", "utils")
        store.add_file_link("/src/main.py", "/src/config.py", "import", "config")
        store.add_file_link("/src/app.py", "/src/utils.py", "import", "utils")

        # What does main.py import?
        imports = store.get_outbound_links("/src/main.py")
        assert len(imports) == 2
        targets = {link["target_path"] for link in imports}
        assert "/src/utils.py" in targets
        assert "/src/config.py" in targets

        # What imports utils.py?
        importers = store.get_inbound_links("/src/utils.py")
        assert len(importers) == 2

        store.close()


def test_hub_detection():
    """Files with most inbound links should be detected as hubs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = MemoryStore(tmpdir, redis_url="redis://localhost:59999")
        # utils.py imported by 5 files
        for i in range(5):
            store.add_file_link(f"/src/file_{i}.py", "/src/utils.py", "import")
        # config.py imported by 2 files
        for i in range(2):
            store.add_file_link(f"/src/file_{i}.py", "/src/config.py", "import")

        hubs = store.get_hub_files(limit=5)
        assert len(hubs) >= 2
        assert hubs[0]["file_path"] == "/src/utils.py"
        assert hubs[0]["inbound_count"] == 5

        store.close()


def test_clear_file_links():
    """Should clear all links for a source file (used before re-indexing)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = MemoryStore(tmpdir, redis_url="redis://localhost:59999")
        store.add_file_link("/src/main.py", "/src/a.py", "import")
        store.add_file_link("/src/main.py", "/src/b.py", "import")
        assert len(store.get_outbound_links("/src/main.py")) == 2

        store.clear_file_links("/src/main.py")
        assert len(store.get_outbound_links("/src/main.py")) == 0

        store.close()


def test_shader_dispatch_link():
    """Should support shader_dispatch link type."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = MemoryStore(tmpdir, redis_url="redis://localhost:59999")
        store.add_file_link("/backend/conv.py", "/shaders/conv2d_gemm.glsl", "shader_dispatch", "conv2d_forward")

        links = store.get_outbound_links("/backend/conv.py")
        assert len(links) == 1
        assert links[0]["link_type"] == "shader_dispatch"
        assert links[0]["symbol_name"] == "conv2d_forward"

        store.close()
```

- [ ] **Step 2: Run test — should fail (no add_file_link method)**

Run: `cd C:\Users\grill\grilly-plugins\elephant-coder && uv run pytest tests/test_link_graph.py -v`

- [ ] **Step 3: Add file_links table and methods to memory_store.py**

In `_init_schema()`, add after the FTS table creation:

```python
# File link graph
cur.execute("""
    CREATE TABLE IF NOT EXISTS file_links (
        source_path TEXT NOT NULL,
        target_path TEXT NOT NULL,
        link_type TEXT NOT NULL,
        symbol_name TEXT,
        PRIMARY KEY (source_path, target_path, link_type)
    )
""")
cur.execute("CREATE INDEX IF NOT EXISTS idx_links_target ON file_links(target_path)")
cur.execute("CREATE INDEX IF NOT EXISTS idx_links_source ON file_links(source_path)")
```

Add these methods to `MemoryStore`:

```python
# ------------------------------------------------------------------
# File Link Graph
# ------------------------------------------------------------------

def add_file_link(self, source_path: str, target_path: str, link_type: str, symbol_name: str | None = None) -> None:
    """Add a directed link between two files."""
    self._conn.execute(
        "INSERT OR REPLACE INTO file_links (source_path, target_path, link_type, symbol_name) VALUES (?, ?, ?, ?)",
        (source_path, target_path, link_type, symbol_name),
    )
    self._conn.commit()

def add_file_links_batch(self, links: list[tuple[str, str, str, str | None]]) -> None:
    """Batch add file links. Each tuple: (source, target, link_type, symbol_name)."""
    if not links:
        return
    self._conn.executemany(
        "INSERT OR REPLACE INTO file_links (source_path, target_path, link_type, symbol_name) VALUES (?, ?, ?, ?)",
        links,
    )
    self._conn.commit()

def get_outbound_links(self, source_path: str) -> list[dict]:
    """Get all files that source_path imports/includes."""
    rows = self._conn.execute(
        "SELECT * FROM file_links WHERE source_path = ? ORDER BY target_path",
        (source_path,),
    ).fetchall()
    return [{"source_path": r["source_path"], "target_path": r["target_path"],
             "link_type": r["link_type"], "symbol_name": r["symbol_name"]} for r in rows]

def get_inbound_links(self, target_path: str) -> list[dict]:
    """Get all files that import/include target_path."""
    rows = self._conn.execute(
        "SELECT * FROM file_links WHERE target_path = ? ORDER BY source_path",
        (target_path,),
    ).fetchall()
    return [{"source_path": r["source_path"], "target_path": r["target_path"],
             "link_type": r["link_type"], "symbol_name": r["symbol_name"]} for r in rows]

def get_hub_files(self, limit: int = 10) -> list[dict]:
    """Get files with the most inbound links (architectural pillars)."""
    rows = self._conn.execute(
        "SELECT target_path, COUNT(*) as inbound_count FROM file_links GROUP BY target_path ORDER BY inbound_count DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [{"file_path": r["target_path"], "inbound_count": r["inbound_count"]} for r in rows]

def clear_file_links(self, source_path: str) -> None:
    """Clear all outbound links for a file (before re-indexing)."""
    self._conn.execute("DELETE FROM file_links WHERE source_path = ?", (source_path,))
    self._conn.commit()
```

- [ ] **Step 4: Run tests**

Run: `cd C:\Users\grill\grilly-plugins\elephant-coder && uv run pytest tests/ -v`

- [ ] **Step 5: Commit**

```bash
cd C:\Users\grill\grilly-plugins
git add elephant-coder/memory_store.py elephant-coder/tests/test_link_graph.py
git commit -m "feat: add file_links table for import/include/shader dependency graph"
```

---

## Task 2: Import Resolution in Indexer

**Files:**
- Modify: `C:\Users\grill\grilly-plugins\elephant-coder\indexer.py` — add import extraction functions
- Create: `C:\Users\grill\grilly-plugins\elephant-coder\link_graph.py` — import resolution logic

- [ ] **Step 1: Write failing test**

```python
# tests/test_link_graph.py (append)

from link_graph import resolve_python_imports, resolve_cpp_includes, detect_shader_dispatches


def test_resolve_python_imports():
    """Should extract import targets from Python source."""
    source = '''
import os
import sys
from pathlib import Path
from grilly.nn import Linear
from grilly.backend.core import VulkanDevice
from .utils import helper
import numpy as np
'''
    imports = resolve_python_imports(source)
    # Should contain module names (not stdlib filtering — that's optional)
    assert "os" in imports
    assert "grilly.nn" in imports
    assert "grilly.backend.core" in imports
    assert ".utils" in imports
    assert "numpy" in imports


def test_resolve_cpp_includes():
    """Should extract #include targets."""
    source = '''
#include <vulkan/vulkan.h>
#include "core.h"
#include "backend/pipelines.h"
#include <cstdlib>
'''
    includes = resolve_cpp_includes(source)
    assert "vulkan/vulkan.h" in includes
    assert "core.h" in includes
    assert "backend/pipelines.h" in includes


def test_detect_shader_dispatches():
    """Should detect shader loading patterns in Python code."""
    source = '''
shader = self._load_shader("conv2d_gemm")
pipeline = self.create_pipeline("flash_attention2")
self._compile_shader("matmul_tiled")
'''
    dispatches = detect_shader_dispatches(source)
    assert "conv2d_gemm" in dispatches
    assert "flash_attention2" in dispatches
    assert "matmul_tiled" in dispatches
```

- [ ] **Step 2: Run test — should fail**

- [ ] **Step 3: Create link_graph.py**

```python
# link_graph.py
"""
Import/include resolution for the file link graph.

Extracts import relationships from source code to build a directed
dependency graph between files.
"""

import ast
import logging
import re

logger = logging.getLogger("elephant-coder.link_graph")


def resolve_python_imports(source: str) -> list[str]:
    """Extract import module names from Python source code."""
    imports = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return _fallback_python_imports(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            prefix = "." * (node.level or 0)
            imports.append(f"{prefix}{module}" if prefix else module)

    return imports


def _fallback_python_imports(source: str) -> list[str]:
    """Regex fallback for unparseable Python files."""
    imports = []
    for m in re.finditer(r"^\s*import\s+([\w.]+)", source, re.MULTILINE):
        imports.append(m.group(1))
    for m in re.finditer(r"^\s*from\s+([\w.]+)\s+import", source, re.MULTILINE):
        imports.append(m.group(1))
    return imports


_CPP_INCLUDE_PATTERN = re.compile(r'^\s*#include\s+[<"]([^>"]+)[>"]', re.MULTILINE)


def resolve_cpp_includes(source: str) -> list[str]:
    """Extract #include targets from C/C++ source."""
    return [m.group(1) for m in _CPP_INCLUDE_PATTERN.finditer(source)]


_SHADER_LOAD_PATTERN = re.compile(
    r'(?:load_shader|create_pipeline|compile_shader|_load_shader|_compile_shader)\s*\(\s*["\'](\w+)["\']',
)


def detect_shader_dispatches(source: str) -> list[str]:
    """Detect shader loading patterns in Python code."""
    return [m.group(1) for m in _SHADER_LOAD_PATTERN.finditer(source)]


def resolve_module_to_path(module_name: str, project_root: str, source_file: str) -> str | None:
    """Try to resolve a Python module name to a file path within the project.

    Returns the resolved path or None if not found in the project.
    """
    from pathlib import Path

    root = Path(project_root)

    # Handle relative imports
    if module_name.startswith("."):
        source_dir = Path(source_file).parent
        dots = len(module_name) - len(module_name.lstrip("."))
        rel_module = module_name.lstrip(".")
        base = source_dir
        for _ in range(dots - 1):
            base = base.parent
        parts = rel_module.split(".") if rel_module else []
        candidate = base / "/".join(parts)
    else:
        parts = module_name.split(".")
        candidate = root / "/".join(parts)

    # Try: module.py, module/__init__.py
    for suffix in [".py", "/__init__.py"]:
        path = Path(str(candidate) + suffix)
        if path.exists():
            return str(path.resolve())

    return None
```

- [ ] **Step 4: Run tests**

- [ ] **Step 5: Commit**

```bash
cd C:\Users\grill\grilly-plugins
git add elephant-coder/link_graph.py elephant-coder/tests/test_link_graph.py
git commit -m "feat: add link_graph module for import/include/shader resolution"
```

---

## Task 3: Populate Links During Indexing

**Files:**
- Modify: `C:\Users\grill\grilly-plugins\elephant-coder\server.py` — after indexing a file, extract and store its links

- [ ] **Step 1: Update index_all() in server.py**

After the `store.upsert_batch(entries)` call inside `index_all()`, add link extraction:

```python
from link_graph import resolve_python_imports, resolve_cpp_includes, detect_shader_dispatches, resolve_module_to_path
```

In the indexing loop, after `store.upsert_batch(entries)`:

```python
# Extract and store file links
_extract_and_store_links(store, fpath, dir_path)
```

Add helper function:

```python
def _extract_and_store_links(store: MemoryStore, fpath: Path, project_root: Path) -> None:
    """Extract imports/includes from a file and store as links."""
    suffix = fpath.suffix.lower()
    fp_str = str(fpath)

    try:
        source = fpath.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return

    store.clear_file_links(fp_str)
    links: list[tuple[str, str, str, str | None]] = []

    if suffix == ".py":
        for module in resolve_python_imports(source):
            resolved = resolve_module_to_path(module, str(project_root), fp_str)
            if resolved:
                links.append((fp_str, resolved, "import", module))
        for shader_name in detect_shader_dispatches(source):
            # Try to find the shader file
            for shader_ext in [".glsl", ".comp", ".vert", ".frag"]:
                shader_path = project_root / "shaders" / f"{shader_name}{shader_ext}"
                if shader_path.exists():
                    links.append((fp_str, str(shader_path.resolve()), "shader_dispatch", shader_name))
                    break
    elif suffix in (".c", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".hxx"):
        for include in resolve_cpp_includes(source):
            # Try relative to file, then project root
            for base in [fpath.parent, project_root]:
                candidate = base / include
                if candidate.exists():
                    links.append((fp_str, str(candidate.resolve()), "include", include))
                    break

    if links:
        store.add_file_links_batch(links)
```

- [ ] **Step 2: Run all tests**

- [ ] **Step 3: Commit**

```bash
cd C:\Users\grill\grilly-plugins
git add elephant-coder/server.py
git commit -m "feat: populate file link graph during indexing"
```

---

## Task 4: Mental Model Generator

**Files:**
- Create: `C:\Users\grill\grilly-plugins\elephant-coder\mental_model.py`
- Create: `C:\Users\grill\grilly-plugins\elephant-coder\tests\test_mental_model.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_mental_model.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import tempfile
from memory_store import MemoryStore, MemoryEntry, make_memory_id
from mental_model import generate_mental_model


def test_generate_mental_model_basic():
    """Should generate a text overview from indexed memories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = MemoryStore(tmpdir, redis_url="redis://localhost:59999")

        # Add some module entries
        store.upsert(MemoryEntry(
            memory_id=make_memory_id(os.path.join(tmpdir, "main.py"), "main", "module"),
            file_path=os.path.join(tmpdir, "main.py"),
            symbol_name="main",
            kind="module",
            summary="Entry point. Imports: utils, config",
            keywords=["main", "entry"],
            line_count=50,
        ))
        store.upsert(MemoryEntry(
            memory_id=make_memory_id(os.path.join(tmpdir, "utils.py"), "utils", "module"),
            file_path=os.path.join(tmpdir, "utils.py"),
            symbol_name="utils",
            kind="module",
            summary="Utility functions for data processing",
            keywords=["utils", "data"],
            line_count=200,
        ))

        # Add a link
        store.add_file_link(
            os.path.join(tmpdir, "main.py"),
            os.path.join(tmpdir, "utils.py"),
            "import", "utils"
        )

        model = generate_mental_model(store, tmpdir)
        assert "Project Mental Model" in model
        assert "utils" in model.lower()
        store.close()


def test_mental_model_shows_hubs():
    """Hub files should appear in the mental model."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = MemoryStore(tmpdir, redis_url="redis://localhost:59999")
        hub_path = os.path.join(tmpdir, "core.py")

        store.upsert(MemoryEntry(
            memory_id=make_memory_id(hub_path, "core", "module"),
            file_path=hub_path, symbol_name="core", kind="module",
            summary="Core module", keywords=["core"], line_count=500,
        ))
        for i in range(5):
            store.add_file_link(f"/src/file_{i}.py", hub_path, "import")

        model = generate_mental_model(store, tmpdir)
        assert "core" in model.lower()
        store.close()
```

- [ ] **Step 2: Run test — should fail**

- [ ] **Step 3: Create mental_model.py**

```python
# mental_model.py
"""
Project mental model generator for elephant-coder.

Generates a human-readable project overview from indexed memories
and the file link graph. This is injected into Claude's context
at session start.
"""

import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger("elephant-coder.mental_model")


def generate_mental_model(store, project_root: str) -> str:
    """Generate a project mental model from indexed memories + link graph."""
    lines = ["## Project Mental Model (auto-generated by elephant-coder)", ""]

    # 1. Project identity — from module entries
    _add_project_identity(store, project_root, lines)

    # 2. Architecture — hub files and directory structure
    _add_architecture(store, project_root, lines)

    # 3. Recently changed files
    _add_recent_changes(project_root, lines)

    # 4. Memory stats
    _add_stats(store, lines)

    return "\n".join(lines)


def _add_project_identity(store, project_root: str, lines: list[str]) -> None:
    """Add project name and description from top-level modules."""
    root_name = Path(project_root).name
    lines.append(f"### {root_name}")
    lines.append("")

    # Get top-level module entries
    rows = store._conn.execute(
        "SELECT symbol_name, summary, line_count FROM memories WHERE kind = 'module' AND file_path LIKE ? ORDER BY line_count DESC LIMIT 10",
        (f"{project_root}%",),
    ).fetchall()

    if rows:
        lines.append("**Key modules:**")
        for r in rows:
            lc = f" [{r['line_count']}L]" if r['line_count'] else ""
            summary = r['summary'].split('\n')[0][:100]
            lines.append(f"- {r['symbol_name']}{lc}: {summary}")
        lines.append("")


def _add_architecture(store, project_root: str, lines: list[str]) -> None:
    """Add hub files and dependency structure."""
    hubs = store.get_hub_files(limit=10)
    if hubs:
        lines.append("### Most Connected Files (architectural pillars)")
        lines.append("")
        for hub in hubs:
            rel = os.path.relpath(hub["file_path"], project_root) if hub["file_path"].startswith(project_root) else hub["file_path"]
            lines.append(f"- {rel} (imported by {hub['inbound_count']} files)")
        lines.append("")


def _add_recent_changes(project_root: str, lines: list[str]) -> None:
    """Add recently changed files from git."""
    try:
        result = subprocess.run(
            ["git", "log", "--since=7 days ago", "--name-only", "--pretty=format:", "--diff-filter=ACMR"],
            capture_output=True, text=True, cwd=project_root, timeout=5,
        )
        if result.returncode == 0:
            files = list(dict.fromkeys(f.strip() for f in result.stdout.strip().split('\n') if f.strip()))
            if files:
                lines.append("### Recently Changed (last 7 days)")
                lines.append("")
                for f in files[:10]:
                    lines.append(f"- {f}")
                lines.append("")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass


def _add_stats(store, lines: list[str]) -> None:
    """Add memory store stats."""
    total = store.count()
    lines.append(f"### Memory: {total}/{store.max_memories} entries indexed")
    lines.append("")
```

- [ ] **Step 4: Run tests**

- [ ] **Step 5: Commit**

```bash
cd C:\Users\grill\grilly-plugins
git add elephant-coder/mental_model.py elephant-coder/tests/test_mental_model.py
git commit -m "feat: add mental model generator for project overview"
```

---

## Task 5: Framework Detector

**Files:**
- Create: `C:\Users\grill\grilly-plugins\elephant-coder\framework_detector.py`
- Create: `C:\Users\grill\grilly-plugins\elephant-coder\tests\test_framework_detector.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_framework_detector.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import tempfile
from framework_detector import detect_frameworks, is_grilly_project, generate_api_map


def test_detect_grilly_in_requirements():
    """Should detect grilly from requirements.txt."""
    with tempfile.TemporaryDirectory() as tmpdir:
        req = Path(tmpdir) / "requirements.txt"
        req.write_text("numpy>=1.24\ngrilly>=0.1.0\nredis>=5.0\n")
        frameworks = detect_frameworks(tmpdir)
        assert any(f["name"] == "grilly" for f in frameworks)


def test_detect_grilly_in_pyproject():
    """Should detect grilly from pyproject.toml dependencies."""
    with tempfile.TemporaryDirectory() as tmpdir:
        pyproj = Path(tmpdir) / "pyproject.toml"
        pyproj.write_text('[project]\nname = "my-app"\ndependencies = ["grilly>=0.1.0", "numpy"]\n')
        frameworks = detect_frameworks(tmpdir)
        assert any(f["name"] == "grilly" for f in frameworks)


def test_is_grilly_project():
    """Should detect when we're inside the grilly project itself."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create grilly project markers
        (Path(tmpdir) / "backend").mkdir()
        (Path(tmpdir) / "backend" / "compute.py").write_text("class VulkanCompute: pass")
        (Path(tmpdir) / "shaders").mkdir()
        (Path(tmpdir) / "shaders" / "test.glsl").write_text("void main() {}")
        assert is_grilly_project(tmpdir) is True


def test_not_grilly_project():
    """Regular projects should not be detected as grilly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "main.py").write_text("print('hello')")
        assert is_grilly_project(tmpdir) is False


def test_generate_api_map():
    """Should generate torch->grilly API mapping."""
    api_map = generate_api_map()
    assert "torch.nn.Linear" in api_map
    assert api_map["torch.nn.Linear"] == "grilly.nn.Linear"
    assert "torch.optim.Adam" in api_map
```

- [ ] **Step 2: Run test — should fail**

- [ ] **Step 3: Create framework_detector.py**

```python
# framework_detector.py
"""
Framework detection for elephant-coder.

Auto-detects installed frameworks (e.g., grilly) in the current project
and generates API maps for global knowledge export.
"""

import logging
import re
from pathlib import Path

logger = logging.getLogger("elephant-coder.framework_detector")

# Known frameworks and their detection patterns
_KNOWN_FRAMEWORKS = {
    "grilly": {
        "pip_name": "grilly",
        "project_markers": ["backend/compute.py", "shaders/"],
        "github": "Grillcheese-AI/grilly",
    },
}


def detect_frameworks(project_root: str) -> list[dict]:
    """Detect frameworks used in or constituting this project."""
    root = Path(project_root)
    found = []

    # Check if project IS a known framework
    for name, info in _KNOWN_FRAMEWORKS.items():
        if _is_framework_project(root, info["project_markers"]):
            found.append({
                "name": name,
                "detected_as": "source_project",
                "github": info.get("github"),
                "repo_path": str(root),
            })

    # Check dependencies (requirements.txt, pyproject.toml)
    deps = _read_dependencies(root)
    for name, info in _KNOWN_FRAMEWORKS.items():
        if info["pip_name"] in deps and not any(f["name"] == name for f in found):
            found.append({
                "name": name,
                "detected_as": "dependency",
                "github": info.get("github"),
                "repo_path": "auto",
            })

    return found


def is_grilly_project(project_root: str) -> bool:
    """Check if this project IS grilly (not just uses it)."""
    root = Path(project_root)
    return (
        (root / "backend" / "compute.py").exists()
        and (root / "shaders").is_dir()
    )


def _is_framework_project(root: Path, markers: list[str]) -> bool:
    """Check if project root contains all framework markers."""
    for marker in markers:
        path = root / marker
        if not (path.exists() or path.is_dir()):
            return False
    return True


def _read_dependencies(root: Path) -> set[str]:
    """Extract dependency names from requirements.txt and pyproject.toml."""
    deps: set[str] = set()

    # requirements.txt
    req_file = root / "requirements.txt"
    if req_file.exists():
        try:
            for line in req_file.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    name = re.split(r"[>=<!\[]", line)[0].strip()
                    if name:
                        deps.add(name.lower())
        except OSError:
            pass

    # pyproject.toml
    pyproj = root / "pyproject.toml"
    if pyproj.exists():
        try:
            text = pyproj.read_text()
            # Simple regex to find dependencies list
            for m in re.finditer(r'"([\w][\w.-]*?)(?:[>=<!\[]|")', text):
                deps.add(m.group(1).lower())
        except OSError:
            pass

    return deps


def generate_api_map() -> dict[str, str]:
    """Generate torch -> grilly API mapping."""
    return {
        "torch.nn.Linear": "grilly.nn.Linear",
        "torch.nn.Conv2d": "grilly.nn.Conv2d",
        "torch.nn.Module": "grilly.nn.Module",
        "torch.nn.functional.relu": "grilly.functional.relu",
        "torch.nn.functional.softmax": "grilly.functional.softmax",
        "torch.nn.functional.cross_entropy": "grilly.functional.cross_entropy",
        "torch.optim.Adam": "grilly.optim.Adam",
        "torch.optim.AdamW": "grilly.optim.AdamW",
        "torch.optim.SGD": "grilly.optim.SGD",
        "torch.Tensor": "numpy.ndarray (float32)",
        "torch.device('cuda')": "grilly.Compute()",
        "torch.no_grad()": "# not needed — grilly uses explicit GradientTape",
        "torch.save()": "grilly.utils.save_checkpoint()",
        "torch.load()": "grilly.utils.load_checkpoint()",
    }


def generate_quick_start() -> str:
    """Generate grilly quick start snippet."""
    return '''import numpy as np
from grilly import Compute
from grilly.nn import Linear, Module
from grilly.optim import Adam

backend = Compute()
model = Linear(784, 10)
optimizer = Adam(model.parameters(), lr=0.001)'''


def generate_differences() -> list[str]:
    """Generate key differences from PyTorch."""
    return [
        "No CUDA dependency — Vulkan compute shaders on any GPU (AMD, NVIDIA, Intel)",
        "Data is always np.float32 numpy arrays, not torch.Tensor",
        "grilly.Compute() replaces torch.device — single entry point for GPU ops",
        "Shaders are GLSL -> SPIR-V, not CUDA kernels",
        "Explicit GradientTape instead of autograd context managers",
    ]
```

- [ ] **Step 4: Run tests**

- [ ] **Step 5: Commit**

```bash
cd C:\Users\grill\grilly-plugins
git add elephant-coder/framework_detector.py elephant-coder/tests/test_framework_detector.py
git commit -m "feat: add framework detector with grilly detection and API mapping"
```

---

## Task 6: project_overview() and what_broke() MCP Tools

**Files:**
- Modify: `C:\Users\grill\grilly-plugins\elephant-coder\server.py` — add two new tools

- [ ] **Step 1: Add imports to server.py**

```python
from mental_model import generate_mental_model
from framework_detector import detect_frameworks, generate_api_map, generate_quick_start, generate_differences
from link_graph import resolve_python_imports, resolve_cpp_includes, detect_shader_dispatches, resolve_module_to_path
```

- [ ] **Step 2: Add project_overview() tool**

```python
@mcp.tool()
def project_overview() -> str:
    """Generate a comprehensive project mental model.

    Returns the project's architecture, key files, hub nodes (most-imported files),
    recent changes, and framework detection. Called automatically at session start
    to give Claude immediate project context.
    """
    store = _get_store()
    project_root = _detect_project_root()
    model = generate_mental_model(store, project_root)

    # Framework detection
    frameworks = detect_frameworks(project_root)
    if frameworks:
        model += "\n### Detected Frameworks\n"
        for fw in frameworks:
            model += f"\n- **{fw['name']}** ({fw['detected_as']})"
            if fw.get("github"):
                model += f" — {fw['github']}"

    return model
```

- [ ] **Step 3: Add what_broke() tool**

```python
@mcp.tool()
def what_broke(since: str = "1 day ago") -> str:
    """Show what changed semantically since the last session.

    Compares current file mtimes against indexed memories to find
    files that changed. For each changed file, shows what symbols
    were affected and which other files depend on them.

    Args:
        since: Git time expression (default: "1 day ago")
    """
    store = _get_store()
    project_root = _detect_project_root()

    try:
        result = subprocess.run(
            ["git", "diff", f"--since={since}", "--name-only", "--diff-filter=ACMR", "HEAD"],
            capture_output=True, text=True, cwd=project_root, timeout=10,
        )
        if result.returncode != 0:
            # Fallback: use git log
            result = subprocess.run(
                ["git", "log", f"--since={since}", "--name-only", "--pretty=format:", "--diff-filter=ACMR"],
                capture_output=True, text=True, cwd=project_root, timeout=10,
            )
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        return f"Could not run git: {exc}"

    changed_files = list(dict.fromkeys(f.strip() for f in result.stdout.strip().split('\n') if f.strip()))
    if not changed_files:
        return f"No files changed since {since}."

    lines = [f"## What Changed (since {since})", ""]

    for rel_path in changed_files[:20]:
        abs_path = str(Path(project_root) / rel_path)
        file_entries = store.search_by_file(abs_path)
        inbound = store.get_inbound_links(abs_path)

        symbols = [e.symbol_name for e in file_entries if e.kind != "module"][:5]
        stale = any(e.is_stale for e in file_entries)

        line = f"- **{rel_path}**"
        if stale:
            line += " [STALE]"
        if symbols:
            line += f": {', '.join(symbols)}"
        lines.append(line)

        if inbound:
            dependents = [os.path.relpath(l["source_path"], project_root) for l in inbound[:5]]
            lines.append(f"  Impact: {len(inbound)} files depend on this ({', '.join(dependents)})")

    return "\n".join(lines)
```

- [ ] **Step 4: Run all tests**

- [ ] **Step 5: Commit**

```bash
cd C:\Users\grill\grilly-plugins
git add elephant-coder/server.py
git commit -m "feat: add project_overview() and what_broke() MCP tools"
```

---

## Task 7: Update pyproject.toml and SessionStart Hook

**Files:**
- Modify: `C:\Users\grill\grilly-plugins\elephant-coder\pyproject.toml` — add new modules
- Modify: `C:\Users\grill\grilly-plugins\elephant-coder\hooks\hooks.json` — enhance SessionStart

- [ ] **Step 1: Update pyproject.toml py-modules**

Add `link_graph`, `mental_model`, `framework_detector` to py-modules list.

- [ ] **Step 2: Update SessionStart hook to include project_overview**

Update the SessionStart prompt to also call `project_overview()` after `index_all()`.

- [ ] **Step 3: Run all tests**

- [ ] **Step 4: Commit**

```bash
cd C:\Users\grill\grilly-plugins
git add elephant-coder/pyproject.toml elephant-coder/hooks/hooks.json
git commit -m "feat: register new modules and enhance SessionStart with project_overview"
```

---

## Plan B Complete — Summary

After all 7 tasks:

| Feature | Description |
|---------|-------------|
| file_links table | Directed graph of imports/includes/shader dispatches between files |
| Import resolution | Python AST-based, C/C++ regex, shader dispatch pattern detection |
| Auto-population | Links extracted and stored during index_all() |
| Mental model | Auto-generated project overview from memories + link graph |
| Hub detection | Files with most inbound links identified as architectural pillars |
| Framework detection | Auto-detect grilly (from deps or project structure) |
| API map | torch -> grilly migration mapping in global knowledge |
| project_overview() | MCP tool for on-demand project mental model |
| what_broke() | Semantic diff showing changed symbols + dependency impact |

**Next:** Plan C (Discipline System) adds persistent task list, scope guard, objective onboarding, and change requests.
