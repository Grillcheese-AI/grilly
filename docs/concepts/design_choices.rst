Design Choices
==============

This page captures core design decisions in Grilly and their tradeoffs.

1. Vulkan-first backend
-----------------------

Choice:
  Use Vulkan compute shaders as the primary acceleration layer.

Why:
  Vulkan enables cross-vendor GPU support (AMD, NVIDIA, Intel) and avoids
  hard coupling to a single vendor-specific runtime.

Tradeoff:
  Kernel development and debugging are lower-level than CUDA-only frameworks.

2. PyTorch-like UX with explicit internals
------------------------------------------

Choice:
  Expose familiar module/functional/optimizer APIs while keeping backend
  controls visible.

Why:
  Lowers migration cost for users and keeps performance-critical internals
  accessible for research and profiling.

Tradeoff:
  Some flows are more explicit (for example, manual output gradients in certain
  training paths) compared with end-to-end autograd frameworks.

3. CPU fallback paths
---------------------

Choice:
  Implement CPU fallbacks when shaders or Vulkan features are unavailable.

Why:
  Improves portability, simplifies development, and keeps tests runnable in
  constrained environments.

Tradeoff:
  Behavior can be slower or numerically slightly different depending on path.

4. NumPy as primary tensor interchange
--------------------------------------

Choice:
  Standardize external API boundaries around NumPy arrays.

Why:
  NumPy is ubiquitous, simple for integration, and predictable for serialization
  and experiment tooling.

Tradeoff:
  Interop with other tensor runtimes sometimes requires conversion steps.

5. Specialized subsystems in one framework
------------------------------------------

Choice:
  Include SNN, cognitive, VSA, multimodal, and retrieval subsystems together.

Why:
  Enables hybrid research workflows without glue code across many libraries.

Tradeoff:
  Broader surface area increases documentation and maintenance complexity.

6. Determinism tools for experimental pipelines
-----------------------------------------------

Choice:
  Add stable hashing and compressed ingestion checkpoints in `utils`.

Why:
  Reproducibility and resumability are critical for long-running language and
  cognition experiments.

Tradeoff:
  Added format/version management responsibilities.
