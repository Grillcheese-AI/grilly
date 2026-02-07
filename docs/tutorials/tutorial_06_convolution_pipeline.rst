Tutorial 06: Convolution Pipeline
=================================

Goal: build a vision-style feature pipeline with convolution, normalization, and
pooling.

Step 1: Create modules
----------------------

.. code-block:: python

   import grilly.nn as nn

   conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
   bn1 = nn.BatchNorm2d(16)
   pool = nn.MaxPool2d(kernel_size=2, stride=2)
   act = nn.ReLU()

Step 2: Prepare image batch
---------------------------

.. code-block:: python

   import numpy as np

   x = np.random.randn(8, 3, 64, 64).astype(np.float32)

Step 3: Run feature extraction block
------------------------------------

.. code-block:: python

   y = conv1(x)
   y = bn1(y)
   y = act(y)
   y = pool(y)

   print("feature shape:", y.shape)

Step 4: Attach a classifier head
--------------------------------

.. code-block:: python

   head = nn.Linear(16 * 32 * 32, 10)
   logits = head(y.reshape(y.shape[0], -1))
   print("logits shape:", logits.shape)

Step 5: Extend to training
--------------------------

You can now attach this block to the training loop pattern from tutorial 03:

1. run forward
2. compute output gradient
3. backward through modules
4. optimizer step
