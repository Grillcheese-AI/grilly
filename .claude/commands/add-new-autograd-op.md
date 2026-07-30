---
name: add-new-autograd-op
description: Workflow command scaffold for add-new-autograd-op in grilly.
allowed_tools: ["Bash", "Read", "Write", "Grep", "Glob"]
---

# /add-new-autograd-op

Use this workflow when working on **add-new-autograd-op** in `grilly`.

## Goal

Implements a new autograd operation (forward/backward) or wires an existing shader into the autograd engine, including Python bindings and tests.

## Common Files

- `cpp/include/grilly/autograd/autograd.h`
- `cpp/src/autograd.cpp`
- `cpp/python/bindings_autograd.cpp`
- `shaders/*.glsl`
- `shaders/spv/*.spv`
- `test_*.py`

## Suggested Sequence

1. Understand the current state and failure mode before editing.
2. Make the smallest coherent change that satisfies the workflow goal.
3. Run the most relevant verification for touched files.
4. Summarize what changed and what still needs review.

## Typical Commit Signals

- Implement or add new GLSL shader(s) for the operation if needed (e.g., shaders/rms-norm-backward.glsl, shaders/spv/rms-norm-backward.spv)
- Update C++ autograd headers and source to add the handler (cpp/include/grilly/autograd/autograd.h, cpp/src/autograd.cpp)
- Wire the operation into Python bindings (cpp/python/bindings_autograd.cpp)
- Add or update a test for the new op (e.g., test_backward_rmsnorm.py, test_backward_swiglu.py, etc.)

## Notes

- Treat this as a scaffold, not a hard-coded script.
- Update the command if the workflow evolves materially.