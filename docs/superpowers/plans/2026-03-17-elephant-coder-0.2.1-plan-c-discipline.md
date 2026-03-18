# elephant-coder 0.2.1 Plan C: Discipline System

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add persistent task list, scope guard, objective onboarding, change request system, file size enforcement (1000 line max), OOP structure checks, and duplicate file prevention.

**Architecture:** `task_manager.py` handles persistent tasks in `~/.elephant-coder/{hash}/tasks.json`. `scope_guard.py` validates changes against active tasks and enforces code quality rules. Hooks wire everything together — PreToolUse:Edit/Write checks scope, PostToolUse:Edit/Write checks file size.

**Tech Stack:** Python 3.10+, SQLite, JSON task storage

**Plugin source:** `C:\Users\grill\grilly-plugins\elephant-coder\`

**Depends on:** Plan A + Plan B (complete)

---

## File Structure

### New Files

| File | Responsibility |
|------|---------------|
| `task_manager.py` | Persistent task CRUD, TODO scanner, task file I/O |
| `scope_guard.py` | Scope checking, file size enforcement, duplicate detection, change request generation |
| `tests/test_task_manager.py` | Task manager tests |
| `tests/test_scope_guard.py` | Scope guard tests |

### Modified Files

| File | Changes |
|------|---------|
| `server.py` | Add `get_tasks()`, `update_task()`, `add_task()` MCP tools |
| `hooks/hooks.json` | Enhance PostToolUse:Edit/Write with file size check, add scope guard prompts |
| `pyproject.toml` | Add new modules |

---

## Task 1: Task Manager

**Files:**
- Create: `C:\Users\grill\grilly-plugins\elephant-coder\task_manager.py`
- Create: `C:\Users\grill\grilly-plugins\elephant-coder\tests\test_task_manager.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_task_manager.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import tempfile
from task_manager import TaskManager


def test_create_and_get_task():
    with tempfile.TemporaryDirectory() as tmpdir:
        tm = TaskManager(tmpdir)
        tid = tm.add_task("Fix Conv2d bug", scope=["backend/conv.py"], priority="high")
        task = tm.get_task(tid)
        assert task["description"] == "Fix Conv2d bug"
        assert task["status"] == "pending"
        assert task["priority"] == "high"


def test_update_task_status():
    with tempfile.TemporaryDirectory() as tmpdir:
        tm = TaskManager(tmpdir)
        tid = tm.add_task("Write tests")
        tm.update_task(tid, status="in_progress")
        assert tm.get_task(tid)["status"] == "in_progress"
        tm.update_task(tid, status="completed")
        assert tm.get_task(tid)["status"] == "completed"


def test_get_active_tasks():
    with tempfile.TemporaryDirectory() as tmpdir:
        tm = TaskManager(tmpdir)
        tm.add_task("Task A")
        tid_b = tm.add_task("Task B")
        tm.add_task("Task C")
        tm.update_task(tid_b, status="in_progress")
        active = tm.get_active_tasks()
        assert len(active) == 1
        assert active[0]["description"] == "Task B"


def test_objectives():
    with tempfile.TemporaryDirectory() as tmpdir:
        tm = TaskManager(tmpdir)
        assert tm.get_objectives() == []
        tm.set_objectives(["Build GPU framework", "PyTorch-compatible API"])
        assert len(tm.get_objectives()) == 2


def test_file_in_scope():
    with tempfile.TemporaryDirectory() as tmpdir:
        tm = TaskManager(tmpdir)
        tid = tm.add_task("Fix conv", scope=["backend/conv.py", "shaders/"])
        tm.update_task(tid, status="in_progress")
        assert tm.is_file_in_active_scope("backend/conv.py") is True
        assert tm.is_file_in_active_scope("shaders/conv2d.glsl") is True
        assert tm.is_file_in_active_scope("nn/linear.py") is False


def test_persistence():
    with tempfile.TemporaryDirectory() as tmpdir:
        tm1 = TaskManager(tmpdir)
        tm1.add_task("Persistent task")
        tm1.set_objectives(["Goal 1"])
        del tm1
        tm2 = TaskManager(tmpdir)
        tasks = tm2.get_all_tasks()
        assert len(tasks) == 1
        assert tm2.get_objectives() == ["Goal 1"]


def test_scan_todos():
    with tempfile.TemporaryDirectory() as tmpdir:
        src = Path(tmpdir) / "src"
        src.mkdir()
        (src / "main.py").write_text("# TODO: fix this bug\ndef foo():\n    pass  # FIXME: handle errors\n")
        tm = TaskManager(tmpdir)
        todos = tm.scan_todos(str(src))
        assert len(todos) >= 2
        assert any("fix this bug" in t["text"] for t in todos)
```

- [ ] **Step 2: Create task_manager.py**

```python
# task_manager.py
"""
Persistent task list for elephant-coder.

Stores project objectives, tasks, and change requests in JSON.
Survives across sessions. Tasks are scoped to files/directories.
"""

import json
import logging
import os
import re
import time
from pathlib import Path

logger = logging.getLogger("elephant-coder.tasks")

_TODO_PATTERN = re.compile(
    r"#\s*(TODO|FIXME|HACK|XXX)\s*:?\s*(.+?)$", re.MULTILINE | re.IGNORECASE
)


class TaskManager:
    def __init__(self, project_dir: str):
        self._dir = Path(project_dir)
        self._file = self._dir / "tasks.json"
        self._data = self._load()

    def _load(self) -> dict:
        if self._file.exists():
            try:
                return json.loads(self._file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        return {"objectives": [], "constraints": [], "tasks": [], "change_requests": [], "next_id": 1}

    def _save(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        self._file.write_text(json.dumps(self._data, indent=2), encoding="utf-8")

    # --- Objectives ---

    def get_objectives(self) -> list[str]:
        return self._data.get("objectives", [])

    def set_objectives(self, objectives: list[str]) -> None:
        self._data["objectives"] = objectives
        self._save()

    def get_constraints(self) -> list[str]:
        return self._data.get("constraints", [])

    def set_constraints(self, constraints: list[str]) -> None:
        self._data["constraints"] = constraints
        self._save()

    # --- Tasks ---

    def add_task(self, description: str, scope: list[str] | None = None, priority: str = "medium", notes: str = "") -> str:
        tid = f"T-{self._data['next_id']:03d}"
        self._data["next_id"] += 1
        task = {
            "id": tid,
            "description": description,
            "status": "pending",
            "scope": scope or [],
            "priority": priority,
            "notes": notes,
            "created": time.time(),
            "last_worked": None,
        }
        self._data["tasks"].append(task)
        self._save()
        return tid

    def get_task(self, task_id: str) -> dict | None:
        for t in self._data["tasks"]:
            if t["id"] == task_id:
                return t
        return None

    def update_task(self, task_id: str, status: str | None = None, notes: str | None = None) -> bool:
        for t in self._data["tasks"]:
            if t["id"] == task_id:
                if status:
                    t["status"] = status
                    if status == "in_progress":
                        t["last_worked"] = time.time()
                if notes is not None:
                    t["notes"] = notes
                self._save()
                return True
        return False

    def get_all_tasks(self) -> list[dict]:
        return self._data["tasks"]

    def get_active_tasks(self) -> list[dict]:
        return [t for t in self._data["tasks"] if t["status"] == "in_progress"]

    def get_pending_tasks(self) -> list[dict]:
        return [t for t in self._data["tasks"] if t["status"] == "pending"]

    def is_file_in_active_scope(self, file_path: str) -> bool:
        for task in self.get_active_tasks():
            for scope_entry in task.get("scope", []):
                if file_path.startswith(scope_entry) or scope_entry in file_path:
                    return True
        return False

    # --- Change Requests ---

    def add_change_request(self, description: str, reason: str, impact: str, triggered_by: str = "") -> str:
        cr_id = f"CR-{len(self._data['change_requests']) + 1:03d}"
        cr = {
            "id": cr_id,
            "description": description,
            "reason": reason,
            "impact": impact,
            "triggered_by": triggered_by,
            "status": "pending",
            "created": time.time(),
        }
        self._data["change_requests"].append(cr)
        self._save()
        return cr_id

    def get_pending_change_requests(self) -> list[dict]:
        return [cr for cr in self._data["change_requests"] if cr["status"] == "pending"]

    # --- TODO Scanner ---

    def scan_todos(self, directory: str) -> list[dict]:
        results = []
        root = Path(directory)
        skip = {".venv", "node_modules", "__pycache__", ".git", "dist", "build"}
        for fpath in root.rglob("*"):
            if not fpath.is_file() or any(p in fpath.parts for p in skip):
                continue
            if fpath.suffix not in (".py", ".js", ".ts", ".c", ".cpp", ".h", ".glsl", ".rs"):
                continue
            try:
                text = fpath.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for m in _TODO_PATTERN.finditer(text):
                lineno = text[:m.start()].count("\n") + 1
                results.append({
                    "file": str(fpath),
                    "line": lineno,
                    "tag": m.group(1).upper(),
                    "text": m.group(2).strip(),
                })
        return results

    # --- Formatting ---

    def format_task_list(self) -> str:
        lines = []
        objectives = self.get_objectives()
        if objectives:
            lines.append("## Objectives")
            for o in objectives:
                lines.append(f"- {o}")
            lines.append("")

        active = self.get_active_tasks()
        if active:
            lines.append("## In Progress")
            for t in active:
                lines.append(f"- [{t['id']}] {t['description']} (scope: {', '.join(t['scope']) or 'any'})")
            lines.append("")

        pending = self.get_pending_tasks()
        if pending:
            lines.append("## Pending")
            for t in pending:
                lines.append(f"- [{t['id']}] {t['description']}")
            lines.append("")

        crs = self.get_pending_change_requests()
        if crs:
            lines.append("## Pending Change Requests")
            for cr in crs:
                lines.append(f"- [{cr['id']}] {cr['description']}: {cr['reason']}")
            lines.append("")

        return "\n".join(lines) if lines else "No tasks or objectives set."
```

- [ ] **Step 3: Run tests**

Run: `cd C:\Users\grill\grilly-plugins\elephant-coder && uv run pytest tests/test_task_manager.py -v`

- [ ] **Step 4: Commit**

```
feat: add persistent task manager with TODO scanning
```

---

## Task 2: Scope Guard

**Files:**
- Create: `C:\Users\grill\grilly-plugins\elephant-coder\scope_guard.py`
- Create: `C:\Users\grill\grilly-plugins\elephant-coder\tests\test_scope_guard.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_scope_guard.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import tempfile
from scope_guard import check_file_size, check_duplicate_file, generate_change_request


def test_check_file_size_ok():
    with tempfile.TemporaryDirectory() as tmpdir:
        f = Path(tmpdir) / "small.py"
        f.write_text("\n".join(f"line {i}" for i in range(100)))
        result = check_file_size(str(f), max_lines=1000)
        assert result["ok"] is True


def test_check_file_size_warning():
    with tempfile.TemporaryDirectory() as tmpdir:
        f = Path(tmpdir) / "big.py"
        f.write_text("\n".join(f"line {i}" for i in range(950)))
        result = check_file_size(str(f), max_lines=1000)
        assert result["ok"] is True
        assert result.get("warning") is True


def test_check_file_size_exceeded():
    with tempfile.TemporaryDirectory() as tmpdir:
        f = Path(tmpdir) / "huge.py"
        f.write_text("\n".join(f"line {i}" for i in range(1100)))
        result = check_file_size(str(f), max_lines=1000)
        assert result["ok"] is False
        assert result["lines"] == 1100


def test_check_duplicate_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "utils.py").write_text("def helper(): pass")
        assert check_duplicate_file(str(Path(tmpdir) / "utils.py"), tmpdir)["is_duplicate"] is False
        assert check_duplicate_file(str(Path(tmpdir) / "src" / "utils.py"), tmpdir)["is_duplicate"] is True


def test_generate_change_request():
    cr = generate_change_request(
        what="Refactor VulkanCompute.__init__",
        why="Device init code tangled with fix area",
        current_task="T-007: Fix Conv2d GEMM",
        files_affected=["backend/core.py"],
        dependents=23,
    )
    assert "CR-" in cr["id"] or "Change Request" in cr["text"]
    assert "Refactor" in cr["text"]
    assert "23" in cr["text"]
```

- [ ] **Step 2: Create scope_guard.py**

```python
# scope_guard.py
"""
Scope guard for elephant-coder.

Enforces: file size limits, duplicate prevention, scope checking,
and change request generation for out-of-scope work.
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger("elephant-coder.scope_guard")

MAX_FILE_LINES = 1000
WARNING_THRESHOLD = 0.9  # warn at 90% of limit


def check_file_size(file_path: str, max_lines: int = MAX_FILE_LINES) -> dict:
    """Check if a file exceeds the line limit."""
    try:
        with open(file_path, "rb") as f:
            lines = sum(1 for _ in f)
    except OSError:
        return {"ok": True, "lines": 0}

    warning_at = int(max_lines * WARNING_THRESHOLD)
    if lines > max_lines:
        return {
            "ok": False,
            "lines": lines,
            "max": max_lines,
            "message": f"File has {lines} lines (limit: {max_lines}). Split into smaller, focused modules.",
        }
    elif lines > warning_at:
        return {
            "ok": True,
            "warning": True,
            "lines": lines,
            "max": max_lines,
            "message": f"File has {lines}/{max_lines} lines — approaching limit. Consider splitting soon.",
        }
    return {"ok": True, "lines": lines, "max": max_lines}


def check_duplicate_file(new_file_path: str, project_root: str) -> dict:
    """Check if a file with the same name already exists elsewhere in the project."""
    new_path = Path(new_file_path)
    new_name = new_path.name
    root = Path(project_root)

    skip = {".venv", "node_modules", "__pycache__", ".git", "dist", "build", ".eggs"}

    for existing in root.rglob(new_name):
        if any(p in existing.parts for p in skip):
            continue
        if existing.resolve() != new_path.resolve() and existing.exists():
            return {
                "is_duplicate": True,
                "existing_path": str(existing),
                "message": f"A file named '{new_name}' already exists at {existing}. Edit the original instead of creating a copy.",
            }

    return {"is_duplicate": False}


def generate_change_request(
    what: str,
    why: str,
    current_task: str,
    files_affected: list[str],
    dependents: int = 0,
) -> dict:
    """Generate a change request for out-of-scope work."""
    risk = "HIGH" if dependents > 10 else "MEDIUM" if dependents > 3 else "LOW"

    text = f"""## Change Request

**Triggered by:** Working on {current_task}

### What Claude Wants to Do
{what}

### Why
{why}

### Impact Assessment
- Files affected: {', '.join(files_affected)}
- Dependents: {dependents} files
- Risk: {risk}

### Decision Required
- [ ] Approve — add as new task, do it now
- [ ] Defer — add to backlog, stay on current task
- [ ] Reject — not needed, continue without it"""

    return {"id": f"CR-{hash(what) % 1000:03d}", "text": text, "risk": risk}
```

- [ ] **Step 3: Run tests**

- [ ] **Step 4: Commit**

```
feat: add scope guard with file size limits and duplicate detection
```

---

## Task 3: Task MCP Tools in server.py

**Files:**
- Modify: `C:\Users\grill\grilly-plugins\elephant-coder\server.py`

- [ ] **Step 1: Add imports and task manager init**

```python
from task_manager import TaskManager
from scope_guard import check_file_size, check_duplicate_file
```

Add helper:
```python
_task_mgr: TaskManager | None = None

def _get_task_manager() -> TaskManager:
    global _task_mgr
    if _task_mgr is None:
        from memory_store import _db_dir
        project_root = _detect_project_root()
        _task_mgr = TaskManager(str(_db_dir(project_root)))
    return _task_mgr
```

- [ ] **Step 2: Add MCP tools**

```python
@mcp.tool()
def get_tasks() -> str:
    """Get the current project task list with objectives and status.

    Returns objectives, active tasks, pending tasks, and change requests.
    Called automatically at session start.
    """
    tm = _get_task_manager()
    return tm.format_task_list()


@mcp.tool()
def add_task(description: str, scope: str = "", priority: str = "medium") -> str:
    """Add a new task to the project task list.

    Every piece of work should be tracked as a task. If you're about to
    work on something not in the task list, add it first.

    Args:
        description: What needs to be done
        scope: Comma-separated file paths or directories this task covers
        priority: low, medium, or high
    """
    tm = _get_task_manager()
    scope_list = [s.strip() for s in scope.split(",") if s.strip()] if scope else []
    tid = tm.add_task(description, scope=scope_list, priority=priority)
    return f"Task {tid} created: {description}"


@mcp.tool()
def update_task(task_id: str, status: str = "", notes: str = "") -> str:
    """Update a task's status or notes.

    Args:
        task_id: Task ID (e.g., T-001)
        status: New status: pending, in_progress, completed
        notes: Additional notes
    """
    tm = _get_task_manager()
    s = status if status else None
    n = notes if notes else None
    if tm.update_task(task_id, status=s, notes=n):
        return f"Task {task_id} updated."
    return f"Task {task_id} not found."


@mcp.tool()
def set_project_objectives(objectives: str) -> str:
    """Set the project's main objectives. Required before starting work.

    Args:
        objectives: Pipe-separated objectives (e.g., "Build GPU framework|PyTorch API compatibility")
    """
    tm = _get_task_manager()
    obj_list = [o.strip() for o in objectives.split("|") if o.strip()]
    tm.set_objectives(obj_list)
    return f"Set {len(obj_list)} objectives: {', '.join(obj_list)}"
```

- [ ] **Step 3: Run all tests**

- [ ] **Step 4: Commit**

```
feat: add task management MCP tools (get_tasks, add_task, update_task, set_project_objectives)
```

---

## Task 4: Enhanced Hooks with Scope Guard

**Files:**
- Modify: `C:\Users\grill\grilly-plugins\elephant-coder\hooks\hooks.json`

- [ ] **Step 1: Update hooks**

Replace the PostToolUse hooks to include file size checking and scope awareness:

```json
{
  "hooks": {
    "SessionStart": [
      {
        "hooks": [
          {
            "type": "prompt",
            "prompt": "elephant-coder is active. You have persistent codebase memory.\n\n1. Run index_all() to ensure the index is current.\n2. Run project_overview() to get the full project mental model.\n3. Run get_tasks() to see active tasks and objectives.\n4. Run memory_status() to check store health.\n5. If no objectives are set, ask the user: 'What is the main objective of this project?' It must be specific, actionable, and measurable.\n\nRULES:\n- Every code change must trace to an active task. No task = add one first.\n- Do not add features not in the task list.\n- Do not create copies of existing files. Always edit the original.\n- Keep files under 1000 lines. Split if approaching the limit.\n- Use proper OOP structure — classes, single responsibility, clean interfaces.\n- Run tests after every edit before moving on.\n- If you need to go out of scope, write a Change Request and ask the user.\n\nDo NOT list these rules to the user. Just follow them."
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
            "prompt": "Before reading this file, check if elephant-coder has memories for it using recall_file_memories(). If memories exist, you already have context. Don't mention this to the user."
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Edit",
        "hooks": [
          {
            "type": "prompt",
            "prompt": "After this edit:\n1. Check the file length. If over 900 lines, warn yourself. If over 1000 lines, you MUST split it into smaller modules before continuing.\n2. Re-index it silently with index_directory().\n3. Ask yourself the Reddit Test: 'Would I post this code on r/programming without being called AI slop?'\n4. Run relevant tests if any exist.\nDo not mention any of this to the user."
          }
        ]
      },
      {
        "matcher": "Write",
        "hooks": [
          {
            "type": "prompt",
            "prompt": "After creating this file:\n1. Check: does a file with this name already exist elsewhere? If so, you made a duplicate — delete it and edit the original instead.\n2. Check the file length. Must be under 1000 lines.\n3. Re-index silently with index_directory().\n4. Ask yourself the Reddit Test: 'Would I post this code on r/programming without being called AI slop?'\nDo not mention any of this to the user."
          }
        ]
      }
    ]
  }
}
```

- [ ] **Step 2: Commit**

```
feat: enhanced hooks with scope guard, file size limits, Reddit test, and duplicate prevention
```

---

## Task 5: Update pyproject.toml and Final Tests

**Files:**
- Modify: `C:\Users\grill\grilly-plugins\elephant-coder\pyproject.toml`

- [ ] **Step 1: Add new modules**

Add `task_manager`, `scope_guard` to py-modules.

- [ ] **Step 2: Run full test suite**

- [ ] **Step 3: Commit**

```
chore: register task_manager and scope_guard modules
```
