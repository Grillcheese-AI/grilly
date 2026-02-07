LoRA and Adapter Fine-Tuning
============================

Why LoRA
--------

Low-Rank Adaptation (LoRA) fine-tunes large models by training only small
adapter matrices instead of full weight tensors. This reduces trainable
parameters and optimizer memory.

Core classes
------------

Grilly provides:

- `nn.LoRAConfig`
- `nn.LoRALinear`
- `nn.LoRAEmbedding`
- `nn.LoRAAttention`
- `nn.LoRAModel`

You can also use utility functions:

- `nn.apply_lora_to_linear(...)`
- `nn.calculate_lora_params(...)`

Basic `LoRALinear` flow
-----------------------

.. code-block:: python

   import numpy as np
   from grilly.nn.lora import LoRALinear
   from grilly.nn.autograd import Variable

   lora = LoRALinear(in_features=768, out_features=768, rank=8, alpha=16)
   x = Variable(np.random.randn(4, 768).astype(np.float32))
   y = lora(x)

   print("trainable params:", lora.num_trainable_params())

Managing inference overhead
---------------------------

For deployment, merge adapters into base weights:

.. code-block:: python

   lora.merge_weights()
   # inference path
   lora.unmerge_weights()

Checkpointing adapters
----------------------

`LoRAModel` supports saving/loading adapter checkpoints with configuration and
metadata, enabling portable fine-tuning artifacts.
