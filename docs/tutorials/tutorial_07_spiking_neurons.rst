Tutorial 07: Spiking Neurons and Activity
=========================================

Goal: run a spiking workflow with both high-level and module-level APIs.

Step 1: High-level SNNCompute path
----------------------------------

.. code-block:: python

   import numpy as np
   import grilly

   snn = grilly.SNNCompute(n_neurons=512)
   embedding = np.random.randn(512).astype(np.float32)
   summary = snn.process(embedding)

   print("spike_activity:", summary["spike_activity"])
   print("firing_rate:", summary["firing_rate"])

Step 2: Module-level LIF neuron
-------------------------------

.. code-block:: python

   import grilly.nn as nn

   lif = nn.LIFNeuron(n_neurons=256, dt=0.001, tau_mem=20.0, v_thresh=1.0)
   current = np.random.randn(256).astype(np.float32)
   spikes = lif(current)
   print("spikes shape:", spikes.shape)

Step 3: Run a small time simulation
-----------------------------------

.. code-block:: python

   rates = []
   lif.reset()
   for _ in range(50):
       s = lif(current)
       rates.append(float(s.mean()))
   print("mean firing:", sum(rates) / len(rates))

Step 4: Add synaptic learning layers
------------------------------------

The same `grilly.nn` namespace includes:

- `HebbianLayer`
- `STDPLayer`
- `GIFNeuron`

These can be integrated after you validate baseline LIF dynamics.
