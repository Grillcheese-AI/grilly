Module System and Functional API
================================

Two ways to build models
------------------------

Grilly supports both:

1. Object-oriented modules (`grilly.nn`)
2. Stateless functional ops (`grilly.functional`)

Use modules for model composition and parameter tracking. Use functional ops for
local math where parameter objects are not needed.

`grilly.nn` overview
--------------------

`grilly.nn.Module` provides:

- `forward(...)` contract
- `__call__` dispatch to `forward`
- parameter registration and iteration
- training/eval flags
- `zero_grad()` support

Common module families:

- Feedforward: `Linear`, `LayerNorm`, activations, `Sequential`
- Convolution stack: `Conv1d`, `Conv2d`, `BatchNorm*`, pooling layers
- Recurrent: `LSTM`, `GRU` variants
- Attention: `MultiheadAttention`, `FlashAttention2`
- SNN modules: `LIFNeuron`, `SNNLayer`, STDP/Hebbian layers
- Multimodal and VLM modules

`grilly.functional` overview
----------------------------

Functional namespace mirrors PyTorch-style patterns:

- activations (`relu`, `gelu`, `silu`, `softmax`)
- linear and normalization helpers
- attention helpers
- memory and retrieval helpers
- FFT and signal utilities

When to choose each API
-----------------------

Choose `nn` when:

- You need trainable parameters.
- You need module composition and reusable blocks.
- You want optimizer-driven updates.

Choose `functional` when:

- You want quick stateless transformations.
- You are prototyping kernels around existing arrays.
- You are writing utility code outside model classes.

Minimal composition example
---------------------------

.. code-block:: python

   import numpy as np
   import grilly.nn as nn

   model = nn.Sequential(
       nn.Linear(256, 512),
       nn.GELU(),
       nn.Linear(512, 128),
   )

   x = np.random.randn(32, 256).astype(np.float32)
   y = model(x)

Functional example
------------------

.. code-block:: python

   import grilly.functional as F

   z = F.relu(y)
