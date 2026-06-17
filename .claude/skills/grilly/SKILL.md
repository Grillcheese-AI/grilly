```markdown
# grilly Development Patterns

> Auto-generated skill from repository analysis

## Overview
This skill teaches the core development patterns, coding conventions, and common workflows used in the `grilly` codebase. `grilly` is a Python-centric project (with C++ and shader components) focused on autograd operations, GPU acceleration, and integration between Python and C++. The repository emphasizes modularity, clear commit practices, and a workflow-driven approach to adding features and documenting progress.

## Coding Conventions

- **File Naming:**  
  Use `snake_case` for all Python and script files.  
  *Example:*  
  ```
  test_backward_rmsnorm.py
  train_linear_ce.py
  ```

- **Import Style:**  
  Prefer **relative imports** within Python modules.  
  *Example:*  
  ```python
  from .core import Tensor
  from .autograd import backward_rmsnorm
  ```

- **Export Style:**  
  Use **named exports** for functions and classes.  
  *Example:*  
  ```python
  def backward_rmsnorm(...):
      ...
  ```

- **Commit Messages:**  
  - Freeform, but often start with prefixes like `autograd`, `docs`, `integration`, `todo`.
  - Average commit message length: ~69 characters.

## Workflows

### Add New Autograd Operation
**Trigger:** When adding support for a new operation to the autograd engine (e.g., `backward_rmsnorm`, `forward_linear`).  
**Command:** `/new-autograd-op`

1. Implement or add new GLSL shader(s) for the operation if needed.  
   *Example:* `shaders/rms-norm-backward.glsl`
2. Compile GLSL to SPIR-V if required.  
   *Example:* `shaders/spv/rms-norm-backward.spv`
3. Update C++ autograd headers and source to add the handler.  
   - `cpp/include/grilly/autograd/autograd.h`
   - `cpp/src/autograd.cpp`
4. Wire the operation into Python bindings.  
   - `cpp/python/bindings_autograd.cpp`
5. Add or update a test for the new op.  
   *Example:* `test_backward_rmsnorm.py`

---

### Document Autograd State or Milestone
**Trigger:** When completing a significant feature, bugfix, or integration step and wanting to record the project state.  
**Command:** `/update-state-doc`

1. Edit `AUTOGRAD_STATE.md` to describe new features, bugfixes, or integration steps.
2. Optionally update `TODO.md` or related docs.
3. Commit the documentation.

---

### Integration Test or Training Script Addition
**Trigger:** When verifying that a new feature or workflow works in a real training or integration scenario.  
**Command:** `/add-integration-test`

1. Add or update a script in `experimental/resident_train/`.  
   *Example:* `train_linear_ce.py`
2. Run the script to verify integration.
3. Optionally update documentation with results.

---

### Add or Update Python-C++ Binding
**Trigger:** When adding a new C++ feature that needs to be exposed to Python.  
**Command:** `/add-python-binding`

1. Implement or update binding in `cpp/python/`.  
   *Example:* `bindings_autograd.cpp`, `bindings_core.cpp`
2. Update `CMakeLists.txt` if new binding files are added.
3. Optionally add a Python test or usage example.

---

### Add or Update Shader and SPIR-V
**Trigger:** When implementing a new GPU operation or updating an existing one.  
**Command:** `/add-shader`

1. Write or modify a GLSL shader in `shaders/*.glsl`.
2. Compile to SPIR-V and add/update `shaders/spv/*.spv`.
3. Wire up usage in C++ and/or Python if needed.

---

### Add or Update TODO or Workboard
**Trigger:** When recording new tasks, marking tasks as done, or updating project planning.  
**Command:** `/update-todo`

1. Edit `TODO.md` to add, update, or mark tasks as done.
2. Commit `TODO.md`.

---

## Testing Patterns

- **Framework:** Unknown (no standard Python test framework detected).
- **File Pattern:** Python test files use the pattern `test_*.py`.
- **Integration Tests:** Often placed in `experimental/resident_train/`.
- **Other Patterns:** Some references to `*.test.ts` (TypeScript), but main tests appear to be Python scripts.

*Example test file:*
```python
# test_backward_rmsnorm.py
from .autograd import backward_rmsnorm

def test_backward_rmsnorm_basic():
    # Setup input tensors
    ...
    # Call the function
    result = backward_rmsnorm(...)
    # Assert correctness
    assert ...
```

## Commands

| Command              | Purpose                                                        |
|----------------------|----------------------------------------------------------------|
| /new-autograd-op     | Add a new autograd operation (forward/backward)                |
| /update-state-doc    | Update project state documentation (AUTOGRAD_STATE.md, TODO.md)|
| /add-integration-test| Add or update an integration test or training script           |
| /add-python-binding  | Add or update Python-C++ bindings                              |
| /add-shader          | Add or update a GLSL shader and its SPIR-V binary              |
| /update-todo         | Update TODO.md or project workboard                            |
```
