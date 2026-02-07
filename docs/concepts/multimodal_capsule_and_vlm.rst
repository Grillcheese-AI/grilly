Multimodal, Capsule, and VLM Systems
====================================

Multimodal module families
--------------------------

`grilly.nn.multimodal` includes several fusion strategies:

- `BottleneckFusion`
- `PerceiverIO`
- `CrossModalAttentionFusion`
- `ImageBindFusion`
- `PerceiverResampler`
- `FlamingoFusion`
- `VisionLanguageModel` and `VLMLayer`

These modules allow cross-modal reasoning across text, vision, and other
feature streams.

Capsule-related components
--------------------------

Grilly also includes capsule-inspired modules and cognitive encoders, including
capsule projection and semantic encoding paths used in experimental cognitive
systems.

When to use which approach
--------------------------

1. Start with `BottleneckFusion` when you need efficient two-stream fusion.
2. Use `PerceiverIO` for variable-length multimodal input handling.
3. Use `VisionLanguageModel` stack when building end-to-end VLM-style systems.

Design choices
--------------

The multimodal subsystem is intentionally pluralistic:

1. Multiple fusion architectures are provided because no single strategy wins
   across all modality/sequence regimes.
2. Bottleneck and resampler options target memory/compute efficiency.
3. VLM-oriented layers are exposed alongside lower-level fusion blocks so teams
   can choose between rapid assembly and custom architecture work.

Example: simple multimodal fusion
---------------------------------

.. code-block:: python

   import numpy as np
   import grilly.nn as nn

   fusion = nn.BottleneckFusion(d_model=256, num_bottlenecks=32, num_heads=8)

   vision = np.random.randn(2, 64, 256).astype(np.float32)
   text = np.random.randn(2, 32, 256).astype(np.float32)

   fused = fusion(vision, text)
   print(fused.shape)

Design considerations
---------------------

- Keep modality embeddings in compatible dimensions before fusion.
- Monitor sequence lengths because attention cost still scales with token counts.
- Use pooling or resampling to reduce large modality streams early.
