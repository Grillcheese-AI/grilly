# Auto Hypergradient Descent for Spiking Neural Networks on Vulkan

## Abstract

We integrate OSGM-style (Online Scaling with Gradient Methods) automatic hypergradient descent into a Vulkan GPU-accelerated spiking neural network framework (Grilly). The optimizer automatically adapts the learning rate -- and optionally the momentum coefficient -- during training using AdaGrad-stabilized hypergradient updates with gradient-norm normalization. This eliminates the need for manual learning rate scheduling, which is particularly beneficial for SNN training where surrogate gradients are noisy and the optimal learning rate shifts as neurons transition between firing and silent phases.

## 1. Background

### 1.1 Spiking Neural Networks and Surrogate Gradients

SNNs process information through discrete binary spikes. The non-differentiable spike generation function (Heaviside step) prevents standard backpropagation. Surrogate gradient methods replace the Heaviside derivative with a smooth approximation during the backward pass:

- **Forward**: S[t] = H(V[t] - V_th) (binary spike)
- **Backward**: dS/dV = surrogate_fn'(V[t] - V_th) (smooth derivative)

Common surrogate functions implemented:
- **ATan**: g(x) = alpha / (2 * (1 + (pi * alpha * x / 2)^2))
- **Sigmoid**: g(x) = alpha * sigmoid(alpha*x) * (1 - sigmoid(alpha*x))
- **FastSigmoid**: g(x) = alpha / (2 * (1 + alpha * |x|)^2)

The surrogate gradient introduces noise into the optimization landscape, making fixed learning rates suboptimal -- the gradient signal quality varies throughout training as neurons change their firing patterns.

### 1.2 Hypergradient Descent

Baydin et al. (ICLR 2018) proposed treating the learning rate as a learnable parameter, updating it each step using the "hypergradient" (gradient of loss w.r.t. the learning rate):

    alpha_{t+1} = alpha_t + beta_hyper * g_t . d_{t-1}

where d_{t-1} is the previous update direction and g_t is the current gradient. This requires a fixed meta-learning rate `beta_hyper`, which itself needs tuning.

### 1.3 OSGM / HDM: Self-Tuning Hypergradients

The OSGM (Online Scaling with Gradient Methods) framework (arXiv:2502.11229, 2505.23081, 2509.11007) eliminates the need to tune the meta-learning rate through two innovations:

1. **Gradient-norm normalization**: h = -g_k . d_{k-1} / ||g_{k-1}||^2, making the hypergradient scale-invariant
2. **AdaGrad accumulator**: G += h^2; delta = eta * h / sqrt(G), where the accumulated squared hypergradients automatically decay the effective meta-learning rate

The algorithm jointly adapts:
- **Step size P** (learning rate) via: P -= eta_P * h_P / sqrt(G_P)
- **Momentum beta** via: beta -= eta_beta * h_beta / sqrt(G_beta)

## 2. Method: AutoHypergradientAdamW

We adapt the OSGM principle to work on top of AdamW rather than vanilla gradient descent. AdamW already provides per-parameter adaptive step sizes via its first and second moment estimates. Our auto hypergradient layer adapts the *global* learning rate and optionally the momentum coefficient (beta1).

### 2.1 Algorithm

Given AdamW with current state (moments m, v, parameters theta):

```
At step k:
  1. Collect gradients g_k for all parameters
  2. If k > warmup_steps and ||g_{k-1}||^2 > 0:
     a. Compute update directions: d_{k-1} = m_hat / (sqrt(v_hat) + eps)
     b. LR hypergradient: h_lr = -sum(g_k * d_{k-1}) / ||g_{k-1}||^2
     c. AdaGrad accumulator: G_lr += h_lr^2
     d. LR update: lr -= eta_lr * h_lr / (sqrt(G_lr) + eps)
     e. Clamp: lr = clip(lr, lr_min, lr_max)
  3. (Optional) momentum hypergradient:
     h_beta = sum(g_k * m_{k-1}) / ||g_{k-1}||^2
     G_beta += h_beta^2
     beta1 -= eta_beta * h_beta / (sqrt(G_beta) + eps)
     beta1 = clip(beta1, beta_min, beta_max)
  4. Run standard AdamW step with adapted lr (and beta1)
  5. Store d_k, ||g_k||^2, m_k for next step
```

### 2.2 Design Decisions

**Why adapt global LR, not per-parameter?** AdamW already provides per-parameter adaptation through its second moment estimates (v_hat). Adding per-parameter hypergradient adaptation would double the memory overhead and interfere with AdamW's own adaptive scaling. The global LR controls the overall step scale, which is what needs dynamic adjustment.

**Why AdaGrad for the meta-update?** AdaGrad's sum-of-squared-gradients accumulator naturally provides decreasing step sizes. For hypergradient descent, this means aggressive adaptation early in training (when the LR may be far from optimal) and conservative updates later (when the LR has stabilized). This is exactly the behavior we want.

**Why normalize by ||g||^2?** Without normalization, the hypergradient magnitude depends on the loss scale and gradient magnitude. Normalizing makes the algorithm invariant to these factors -- a model with loss = 100 and one with loss = 0.01 receive hypergradient updates of similar magnitude.

**Why warmup?** Adam's moment estimates are heavily biased in the first few steps (before the bias correction term 1/(1-beta^t) stabilizes). Adapting LR based on these unreliable estimates causes erratic behavior. A short warmup (10-20 steps) lets the moments initialize.

### 2.3 Relationship to OSGM Reference

The reference OSGM implementation (algorithms/hdm.py) operates on vanilla gradient descent with Polyak heavy-ball momentum:

```
x_{k+1} = x_k - P * g_k + beta * (x_k - x_{k-1})
```

Our adaptation replaces the base GD step with AdamW, which introduces:
- Exponential moving average momentum (vs. Polyak momentum)
- Per-parameter second moment scaling
- Decoupled weight decay
- Bias correction

The hypergradient formulas are analogous:
- OSGM: h_P = -g_{k+1} . g_k / ||g_k||^2 (evaluate gradient at trial point)
- Ours: h_lr = -g_k . d_{k-1} / ||g_{k-1}||^2 (use previous direction, online setting)

The online approximation is necessary because in mini-batch training we only compute one gradient per step (cannot evaluate at a trial point without an extra forward/backward pass).

## 3. Implementation

### 3.1 Optimizer Classes

Two classes in `optim/hypergradient.py`:

1. **HypergradientAdamW**: Basic Baydin et al. approach with fixed beta_hyper. Simple, requires tuning.
2. **AutoHypergradientAdamW**: OSGM-style self-tuning. AdaGrad accumulator eliminates beta_hyper tuning.

### 3.2 Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| lr | 1e-3 | Initial learning rate |
| hyper_lr | 0.01 | Meta-learning rate (auto-modulated by AdaGrad) |
| warmup_steps | 10 | Steps before LR adaptation begins |
| lr_min, lr_max | 1e-6, 1.0 | LR clamp bounds |
| adapt_momentum | False | Also adapt beta1 |
| hyper_lr_beta | 1.0 | Meta-LR for beta1 adaptation |
| beta_min, beta_max | 0.5, 0.9995 | beta1 clamp bounds |

### 3.3 GPU Acceleration

The optimizer inherits AdamW's GPU shader support (`adamw-update.glsl`). The hypergradient computation itself is CPU-side (scalar operations on aggregated dot products), adding negligible overhead.

## 4. SNN Framework Architecture

### 4.1 Neuron Models

| Neuron | Charge Equation | Learnable Parameters |
|--------|-----------------|---------------------|
| IFNode | H = V + X | None |
| LIFNode | H = V*(1-1/tau) + X | tau (fixed) |
| ParametricLIFNode | H = V*(1-1/tau) + X | tau (learnable Parameter) |

### 4.2 BPTT with Surrogate Gradients

Multi-step backward through time:
```
for t = T-1 to 0:
  sg = surrogate_fn'(H[t] - V_th)
  dL/dH[t] = dL/dS[t] * sg + dL/dV[t] * (1 - S[t])
  dL/dX[t] = dL/dH[t] * dH/dX
  dL/dV[t-1] = dL/dH[t] * dH/dV_prev
```

### 4.3 GPU Shaders

- `snn-node-forward.glsl`: Batched charge-fire-reset for IF/LIF/PLIF neurons
- `snn-node-backward.glsl`: Surrogate gradient backward pass (ATan/Sigmoid/FastSigmoid)
- `snn-synapse-filter.glsl`: Exponential decay synaptic filtering
- `snn-spiking-attention.glsl`: Spiking self-attention (QKV without softmax)

### 4.4 Convolutional SNN Architecture (CSNN)

```
Input: (N, 1, 28, 28) grayscale image
  |-- Repeat T times --> (T, N, 1, 28, 28)
  |
  |-- SeqToANNContainer(Conv2d(1, ch, 3, pad=1) + BatchNorm2d(ch))
  |-- IFNode/LIFNode (step_mode='m')
  |-- MaxPool2d(2, 2)                    # 28 -> 14
  |
  |-- SeqToANNContainer(Conv2d(ch, ch, 3, pad=1) + BatchNorm2d(ch))
  |-- IFNode/LIFNode (step_mode='m')
  |-- MaxPool2d(2, 2)                    # 14 -> 7
  |
  |-- Flatten
  |-- SeqToANNContainer(Linear(ch*7*7, ch*4*4))
  |-- IFNode/LIFNode (step_mode='m')
  |-- SeqToANNContainer(Linear(ch*4*4, 10))
  |-- IFNode/LIFNode (step_mode='m')
  |
  |-- Mean over T --> (N, 10) firing rate output
```

## 5. Experiments

### 5.1 Setup

- **Dataset**: Fashion-MNIST (10 classes, 28x28 grayscale)
- **Training subset**: 10,000 samples (speed), Test: 2,000 samples
- **Timesteps**: T=4
- **Channels**: 32
- **Batch size**: 64
- **Epochs**: 5
- **GPU**: AMD Radeon RX 6750 XT (Vulkan compute)
- **Surrogate function**: ATan (alpha=2.0)

### 5.2 Baselines

| Model | Optimizer | LR | Result |
|-------|-----------|-----|--------|
| CSNN-IF | Manual SGD | 0.005 | 36.00% test acc |
| CSNN-LIF | Manual SGD | 0.005 | 48.15% test acc |
| CSNN-LIF | AutoHypergradientAdamW | auto | TBD |

### 5.3 Results

*To be filled after benchmark run.*

### 5.4 LR Trajectory Analysis

*To be filled with lr_history plots showing how the learning rate adapts during SNN training.*

## 6. Discussion

### 6.1 Why Auto Hypergradient is Particularly Suited for SNNs

1. **Noisy surrogate gradients**: The surrogate gradient approximation introduces systematic bias. The gradient signal quality varies as neurons change firing patterns -- some timesteps produce informative gradients while others are near-zero (dead neurons) or saturated. Auto LR adaptation naturally handles this.

2. **Phase transitions**: During training, neurons can transition between firing and silent phases. When many neurons go silent, gradients vanish and a fixed LR makes no progress. The hypergradient detects this (consecutive gradients disagree with update direction) and increases the LR.

3. **No scheduling needed**: Standard SNN training uses cosine annealing or step decay schedules. These require tuning the schedule hyperparameters (T_max, milestones, gamma). Auto hypergradient eliminates this entirely.

4. **Temporal dynamics**: The T-dimensional computation creates a complex loss landscape with temporal correlations. The optimal LR at the start of training (random weights, sparse firing) is different from mid-training (structured firing patterns). Auto adaptation handles this transition.

### 6.2 Connection to Curvature Estimation

The OSGM paper includes dynamic curvature estimation (Lipschitz constant L) to set the initial learning rate. We skip this because AdamW's second moment already provides a form of curvature estimation at the per-parameter level. The global LR adaptation handles the remaining global scale factor.

## 7. References

1. Baydin et al. "Online Learning Rate Adaptation with Hypergradient Descent" (ICLR 2018)
2. "Provable and Practical Online Learning Rate Adaptation with Hypergradient Descent" (arXiv:2502.11229)
3. "Gradient Methods with Online Scaling Part I" (arXiv:2505.23081)
4. "Gradient Methods with Online Scaling Part II" (arXiv:2509.11007)
5. Chandra et al. "Gradient Descent: The Ultimate Optimizer" (NeurIPS 2022)
6. Loshchilov & Hutter "Decoupled Weight Decay Regularization" (ICLR 2019)
7. Fang et al. "Incorporating Learnable Membrane Time Constants to Enhance Learning of Spiking Neural Networks" (ICCV 2021)
8. Zhou et al. "Spikformer: When Spiking Neural Network Meets Transformer" (ICLR 2023)

## Appendix A: File Manifest

| File | Description |
|------|-------------|
| `optim/hypergradient.py` | HypergradientAdamW + AutoHypergradientAdamW |
| `optim/__init__.py` | Exports |
| `tests/test_hypergradient.py` | 20 unit tests |
| `tests/benchmark_snn_fashion_mnist.py` | Fashion-MNIST benchmark (SGD vs Auto) |
| `nn/snn_base.py` | BaseNode with BPTT and GPU dispatch |
| `nn/snn_neurons.py` | IF, LIF, ParametricLIF neuron nodes |
| `nn/snn_surrogate.py` | ATan, Sigmoid, FastSigmoid surrogates |
| `nn/snn_containers.py` | SeqToANNContainer, MultiStepContainer |
| `shaders/snn-node-forward.glsl` | GPU neuron forward shader |
| `shaders/snn-node-backward.glsl` | GPU surrogate gradient backward |
| `docs/auto_hypergradient_snn.md` | This document |

## Appendix B: Optimizer Comparison

### B.1 HypergradientAdamW (Basic)

```python
# Fixed meta-learning rate -- requires tuning beta_hyper
h = sum(g_k * d_{k-1})           # raw hypergradient (not normalized)
lr += beta_hyper * h              # direct update
lr = clip(lr, lr_min, lr_max)
```

Pros: Simple, fast.
Cons: beta_hyper is highly sensitive -- too high causes instability, too low makes adaptation too slow.

### B.2 AutoHypergradientAdamW (OSGM-style)

```python
# Self-tuning -- AdaGrad handles meta-LR automatically
h = -sum(g_k * d_{k-1}) / ||g_{k-1}||^2   # normalized hypergradient
G += h^2                                     # AdaGrad accumulator
lr -= hyper_lr * h / (sqrt(G) + eps)         # stabilized update
lr = clip(lr, lr_min, lr_max)
```

Pros: Robust, scale-invariant, self-stabilizing. hyper_lr is much less sensitive than beta_hyper.
Cons: Slightly more memory (stores previous directions, grad norm, AdaGrad accumulators).

### B.3 Memory Overhead

Per parameter tensor of size N:
- HypergradientAdamW: +N floats (previous direction d_{k-1})
- AutoHypergradientAdamW: +N floats (d_{k-1}) + 2 scalars (G_lr, prev_grad_norm_sq)
- With adapt_momentum: +N floats (m_{k-1}) + 1 scalar (G_beta)

Compared to AdamW's base 2N (m, v), the overhead is +N (50% increase) which is comparable to AMSGrad's overhead.
