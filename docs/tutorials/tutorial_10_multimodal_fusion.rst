Tutorial 10: Multimodal Fusion
==============================

Goal: fuse vision and text features with Grilly multimodal modules.

Step 1: Prepare modality tensors
--------------------------------

.. code-block:: python

   import numpy as np

   batch = 2
   vision_tokens = 64
   text_tokens = 32
   d_model = 256

   vision = np.random.randn(batch, vision_tokens, d_model).astype(np.float32)
   text = np.random.randn(batch, text_tokens, d_model).astype(np.float32)

Step 2: Bottleneck fusion
-------------------------

.. code-block:: python

   import grilly.nn as nn

   fusion = nn.BottleneckFusion(d_model=d_model, num_bottlenecks=32, num_heads=8)
   fused = fusion(vision, text)
   print("bottleneck fused:", fused.shape)

Step 3: Cross-modal attention fusion
------------------------------------

.. code-block:: python

   cross = nn.CrossModalAttentionFusion(d_model=d_model, num_heads=8, num_encoder_layers=2)
   cross_out = cross(vision, text)
   print("cross fused:", cross_out.shape)

Step 4: Perceiver IO on variable input
--------------------------------------

.. code-block:: python

   perceiver = nn.PerceiverIO(input_dim=d_model, latent_dim=512, num_latents=128, num_heads=8)
   latent = perceiver(vision)
   print("latent shape:", latent.shape)

Step 5: Expand to full VLM stack
--------------------------------

For larger projects, move from these fusion blocks into:

- `nn.VisionLanguageModel`
- `nn.VLMLayer`
- `nn.FlamingoFusion`
