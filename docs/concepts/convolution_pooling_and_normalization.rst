Convolution, Pooling, and Normalization
=======================================

What is covered
---------------

Grilly includes core vision-style building blocks:

- `nn.Conv1d`, `nn.Conv2d`
- `nn.MaxPool2d`, `nn.AvgPool2d`, adaptive pooling variants
- `nn.BatchNorm1d`, `nn.BatchNorm2d`
- `nn.LayerNorm` for feature-space normalization

Data layout
-----------

Convolution and pooling APIs follow NCHW layout:

- input: `(batch, channels, height, width)`
- output: `(batch, out_channels, out_height, out_width)`

`BatchNorm2d` also expects NCHW and normalizes per channel.

Backward support
----------------

Conv and batchnorm modules include backward paths and parameter gradient
accumulation. This supports full training loops for CNN-style architectures.

Simple CNN block
----------------

.. code-block:: python

   import numpy as np
   import grilly.nn as nn

   conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
   bn = nn.BatchNorm2d(16)
   pool = nn.MaxPool2d(kernel_size=2, stride=2)

   x = np.random.randn(8, 3, 64, 64).astype(np.float32)

   y = conv(x)
   y = bn(y)
   y = pool(y)
   print(y.shape)

Design notes
------------

1. Prefer explicit shape checks near your model entry points.
2. Keep channels and spatial dimensions consistent across blocks.
3. For first debugging pass, verify forward shape flow before enabling training.
