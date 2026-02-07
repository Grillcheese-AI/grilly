Specialized Operations
======================

Beyond core deep learning blocks, Grilly includes specialized GPU operation
families for research and hybrid AI workflows.

Signal and spectral ops
-----------------------

FFT-related operations are exposed through backend/functional paths and support:

- FFT/IFFT transforms
- magnitude and power spectrum
- signal-oriented processing pipelines

Neuroscience-inspired cells
---------------------------

Cell modules and kernels include:

- place-cell encodings
- time-cell encodings
- theta-gamma style temporal representations

These are useful for navigation, sequence memory, and cognitive modeling tasks.

Bridge and conversion ops
-------------------------

Grilly includes continuous-to-spike and spike-to-continuous conversions for
hybrid SNN/ANN pipelines.

Domain routing and affective ops
--------------------------------

The framework also contains:

- domain routing and expert-combination kernels
- affective feature processing paths
- capsule-oriented representations in cognitive flows

Embedding and lookup operations
-------------------------------

Embedding workflows include lookup, positional handling, normalization, and
attention-friendly transforms, with Vulkan acceleration where available.

Design choices
--------------

Specialized operations are kept as modular kernels instead of hidden internals:

1. Researchers can compose niche operators without forking core modules.
2. Domain-specific kernels can evolve independently from standard ANN layers.
3. Experimental operators can keep CPU fallbacks while GPU kernels mature.

Practical recommendation
------------------------

Treat these components as composable accelerators around your core model. Start
with one specialized operation at a time and validate numerical behavior before
stacking multiple experimental subsystems together.
