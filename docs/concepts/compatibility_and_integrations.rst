Compatibility and Integrations
==============================

PyTorch-style compatibility layer
---------------------------------

Grilly includes compatibility utilities to ease migration from PyTorch-style
code:

- tensor-like helpers in `grilly.utils.pytorch_compat`
- operation wrappers in `grilly.utils.pytorch_ops`
- module and functional APIs aligned with familiar naming patterns

Device management
-----------------

`grilly.utils.device_manager` and related helpers provide utilities for backend
selection and interop in mixed environments.

Tensor conversion helpers
-------------------------

`grilly.utils.tensor_conversion` utilities can normalize external tensor inputs
for Vulkan-friendly execution paths.

HuggingFace bridge
------------------

Grilly includes bridge utilities aimed at loading or adapting common model
workflows from the HuggingFace ecosystem for Vulkan-backed execution paths.

When to use integrations
------------------------

Use integration utilities when:

1. Porting an existing PyTorch codebase incrementally.
2. Running mixed CPU/GPU or mixed framework experiments.
3. Reusing external model/tokenizer infrastructure while moving heavy compute
   into Grilly kernels.
