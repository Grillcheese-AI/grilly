Tutorial 09: LoRA Adaptation Workflow
=====================================

Goal: create and manage a LoRA adapter for efficient fine-tuning.

Step 1: Create a LoRA config
----------------------------

.. code-block:: python

   from grilly.nn.lora import LoRAConfig

   config = LoRAConfig(
       rank=8,
       alpha=16.0,
       target_modules=["q_proj", "v_proj"],
       dropout=0.0,
   )

Step 2: Build a LoRA layer
--------------------------

.. code-block:: python

   import numpy as np
   from grilly.nn.lora import LoRALinear
   from grilly.nn.autograd import Variable

   base_weight = np.random.randn(768, 768).astype(np.float32) * 0.02
   layer = LoRALinear(
       in_features=768,
       out_features=768,
       rank=config.rank,
       alpha=config.alpha,
       base_weights=base_weight,
   )

Step 3: Run forward
-------------------

.. code-block:: python

   x = Variable(np.random.randn(2, 768).astype(np.float32))
   y = layer(x)
   print("output shape:", y.shape)
   print("trainable params:", layer.num_trainable_params())

Step 4: Merge for inference
---------------------------

.. code-block:: python

   layer.merge_weights()
   # run inference path...
   layer.unmerge_weights()

Step 5: Manage multi-layer adapters
-----------------------------------

.. code-block:: python

   from grilly.nn.lora import LoRAModel

   lora_model = LoRAModel(config)
   lora_model.add_lora_layer("encoder.attn.q_proj", 768, 768, base_weights=base_weight)
   print("total trainable:", lora_model.num_trainable_params())
