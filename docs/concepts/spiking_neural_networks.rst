Spiking Neural Networks
=======================

SNN support in Grilly
---------------------

Grilly includes both low-level and high-level spiking workflows:

- Backend kernels: LIF update, Hebbian learning, STDP updates.
- Module API: `nn.LIFNeuron`, `nn.SNNLayer`, `nn.HebbianLayer`, `nn.STDPLayer`, `nn.GIFNeuron`.
- High-level wrapper: `grilly.SNNCompute`.

LIF neuron model
----------------

LIF neurons track membrane potential over time and emit spikes when threshold
is crossed. Typical state variables:

- membrane potential
- refractory timer
- optional adaptation state (for GIF variants)

High-level SNN pipeline
-----------------------

`SNNCompute.process(...)` converts an embedding into spike activity metrics:

- spike count
- spike pattern
- firing rate

This is useful for rapid prototyping and visualization.

Example: high-level usage
-------------------------

.. code-block:: python

   import numpy as np
   import grilly

   snn = grilly.SNNCompute(n_neurons=512)
   embedding = np.random.randn(512).astype(np.float32)
   result = snn.process(embedding)

   print(result["spike_activity"], result["firing_rate"])

Example: module usage
---------------------

.. code-block:: python

   import numpy as np
   import grilly.nn as nn

   lif = nn.LIFNeuron(n_neurons=256, dt=0.001, tau_mem=20.0, v_thresh=1.0)
   input_current = np.random.randn(256).astype(np.float32)
   spikes = lif(input_current)
   print(spikes.shape)

Practical guidance
------------------

1. Tune threshold and time constants together.
2. Start with short simulation windows for debugging.
3. Log firing-rate distribution to catch dead or saturated neurons.
