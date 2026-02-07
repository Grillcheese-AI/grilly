Attention, Transformers, and Decoding
=====================================

Attention stack
---------------

Grilly provides both module-level and backend-level attention:

- Module API: `nn.MultiheadAttention`, `nn.FlashAttention2`
- Backend API: `backend.attention.*` and `backend.attention.flash_attention2(...)`

Use module attention when building end-to-end model graphs. Use backend attention
when running direct kernel workflows and benchmarks.

Why Flash Attention 2 matters
-----------------------------

Flash-style attention reduces memory pressure for long sequences by computing
attention in tiled blocks rather than materializing full intermediate matrices.

In Grilly this can improve throughput and reduce OOM risk on practical sequence
lengths, especially on consumer GPUs.

Transformer-facing modules
--------------------------

`grilly.nn` also includes transformer-oriented components:

- `TransformerEncoderLayer`
- `TransformerDecoderLayer`
- `RoPE`
- `ProsodyModulatedAttention`
- decoding modules (`GreedyDecoder`, `SampleDecoder`)

Basic attention example
-----------------------

.. code-block:: python

   import numpy as np
   import grilly.nn as nn

   attn = nn.MultiheadAttention(embed_dim=256, num_heads=8)
   x = np.random.randn(4, 32, 256).astype(np.float32)

   out, weights = attn(query=x, key=x, value=x)
   print(out.shape, weights.shape)

Backend flash attention example
-------------------------------

.. code-block:: python

   import numpy as np
   import grilly

   backend = grilly.Compute()
   q = np.random.randn(2, 8, 64, 64).astype(np.float32)
   k = np.random.randn(2, 8, 64, 64).astype(np.float32)
   v = np.random.randn(2, 8, 64, 64).astype(np.float32)

   y = backend.attention.flash_attention2(q, k, v)
   print(y.shape)

Decoding usage
--------------

Decoder modules are used to convert logits to token decisions:

- greedy decoding for deterministic paths
- sampled decoding for stochastic generation

They fit naturally after transformer output projection heads.
