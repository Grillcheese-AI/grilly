# Spiking Neural Networks

Grilly provides a complete SNN framework with neuron models, surrogate gradients, temporal containers, ANN-to-SNN conversion, and specialized normalization layers.

---

## Neuron Models

### IFNode (Integrate-and-Fire)

```python
from grilly import nn

node = nn.IFNode(v_threshold=1.0, surrogate_function=nn.ATan())
output = node(x)  # x: input current, output: binary spikes
```

Simplest spiking neuron. Integrates input, fires when threshold is reached, then resets.

### LIFNode (Leaky Integrate-and-Fire)

```python
node = nn.LIFNode(tau=2.0, v_threshold=1.0, surrogate_function=nn.ATan())
output = node(x)
```

Adds exponential membrane potential decay (leak). The `tau` parameter controls the time constant.

### ParametricLIFNode

```python
node = nn.ParametricLIFNode(init_tau=2.0, surrogate_function=nn.ATan())
output = node(x)
```

Like LIFNode but with a learnable time constant. The network learns the optimal decay rate.

---

## Surrogate Gradients

Spiking neurons use a step function (non-differentiable). Surrogate gradient functions provide smooth approximations for backpropagation:

```python
atan = nn.ATan(alpha=2.0)         # Arctangent surrogate
sig = nn.Sigmoid(alpha=4.0)       # Sigmoid surrogate
fast = nn.FastSigmoid(alpha=2.0)  # Piecewise linear approximation
```

All neuron nodes accept a `surrogate_function` parameter.

---

## Temporal Containers

### MultiStepContainer

Runs a module over multiple time steps:

```python
container = nn.MultiStepContainer(nn.LIFNode(tau=2.0))
# input: (T, batch, ...) -> output: (T, batch, ...)
output = container(x_temporal)
```

### SeqToANNContainer

Wraps an ANN layer (e.g., Conv2d) to process temporal data by merging the time and batch dimensions:

```python
container = nn.SeqToANNContainer(nn.Conv2d(3, 64, 3, padding=1))
# Reshapes (T, B, C, H, W) -> (T*B, C, H, W), applies conv, reshapes back
output = container(x_temporal)
```

---

## SNN Normalization

Specialized batch normalization layers that account for temporal dynamics:

| Layer | Description |
|-------|-------------|
| `BatchNormThroughTime1d/2d` | BN computed across the time dimension |
| `TemporalEffectiveBatchNorm1d/2d` | Time-weighted batch normalization |
| `ThresholdDependentBatchNorm1d/2d` | BN scaled by firing threshold |
| `NeuNorm` | Neuron-level normalization |

---

## Synapse Models

```python
from grilly import nn

# Basic synapse with delay
synapse = nn.SynapseFilter(tau_syn=5.0)

# Short-term plasticity
stp = nn.STPSynapse(tau_d=200.0, tau_f=20.0, U=0.2)

# Dual timescale (fast + slow)
dual = nn.DualTimescaleSynapse(tau_fast=5.0, tau_slow=50.0)
```

---

## SNN Attention

```python
# Temporal-wise attention for SNN
twa = nn.TemporalWiseAttention(T=10, reduction=4)

# Spiking self-attention
ssa = nn.SpikingSelfAttention(embed_dim=256, num_heads=4)

# QK attention variants
qk = nn.QKAttention(embed_dim=256, num_heads=4)
```

---

## ANN-to-SNN Conversion

Convert a trained ANN to an SNN:

```python
from grilly.nn import Converter, VoltageScaler

# Convert ReLU model to spiking equivalent
converter = Converter(mode="max_norm")
snn_model = converter(ann_model)

# Scale voltage thresholds
scaler = VoltageScaler(snn_model)
scaler.scale_thresholds(calibration_data)
```

---

## Legacy SNN Layers

The original flat-array SNN layers are still available:

```python
lif = nn.LIFNeuron(threshold=1.0, decay=0.9)
snn_layer = nn.SNNLayer(in_features=256, out_features=128)
hebbian = nn.HebbianLayer(in_features=256, out_features=128)
stdp = nn.STDPLayer(in_features=256, out_features=128)
```

These use the 19 SNN compute shaders (`snn-*.glsl`) via the `VulkanCompute.snn` namespace.

---

## Functional API

```python
import grilly.functional as F

# Single LIF step
spike, v = F.lif_step(x, v_prev, tau=2.0, v_threshold=1.0)

# Multi-step forward
output = F.multi_step_forward(model, x_temporal, T=10)

# Reset all neuron states
F.reset_net(model)
```
