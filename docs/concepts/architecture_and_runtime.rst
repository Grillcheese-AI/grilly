Architecture and Runtime
========================

What Grilly is
--------------

Grilly is a Vulkan-first ML framework with a PyTorch-like interface. The stack
has two layers:

1. Runtime layer: Vulkan core, shader pipelines, descriptor sets, buffer ops.
2. User layer: `grilly.nn`, `grilly.functional`, `grilly.optim`, and
   experimental systems.

Core runtime objects
--------------------

- `grilly.backend.core.VulkanCore`
  Initializes Vulkan instance/device, loads SPIR-V shaders, manages raw buffers.
- `grilly.backend.pipelines.VulkanPipelines`
  Creates compute pipelines and descriptor set layouts and caches them.
- `grilly.backend.compute.VulkanCompute`
  Composes operation modules (`fnn`, `attention`, `snn`, `faiss`, etc.) into a
  single backend object.

Execution flow
--------------

A typical operation follows this sequence:

1. Validate and normalize input arrays (`np.float32` in most kernels).
2. Create/acquire GPU buffers.
3. Upload host data.
4. Build or fetch pipeline and descriptor set.
5. Dispatch compute shader with workgroup sizes and push constants.
6. Download output.
7. Release buffers/descriptor sets.

CPU fallback model
------------------

Many operators include a CPU fallback when:

- Vulkan is unavailable.
- A specific shader is not present.
- A GPU dispatch fails and the code path handles fallback.

This allows development and tests to run across more environments while still
using GPU acceleration when available.

Resource lifecycle
------------------

Recommended usage:

.. code-block:: python

   import grilly

   backend = grilly.Compute()
   # run workload...
   backend.cleanup()

`cleanup()` tears down pipelines and device resources deterministically. You
should call it in services and benchmarks instead of relying on object
destructors.

How the high-level APIs fit
---------------------------

- `grilly.Compute()` gives direct backend access.
- `grilly.nn` modules call into backend operators internally.
- `grilly.functional` exposes stateless helpers over the same backend kernels.
- `grilly.optim` updates parameter arrays from gradients produced by module
  backward functions.
