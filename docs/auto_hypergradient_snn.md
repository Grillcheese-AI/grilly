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
  3. (Optional) Compute surprise signal for input gain:
     g_ema = gamma * g_ema + (1-gamma) * g_k
     var_ema = gamma * var_ema + (1-gamma) * ||g_k||^2
     surprise = tanh(||g_k - g_ema||^2 / (var_ema + eps))
     (Exposed as current_surprise for model forward pass)
  4. (Optional) momentum hypergradient:
     h_beta = sum(g_k * m_{k-1}) / ||g_{k-1}||^2
     G_beta += h_beta^2
     beta1 -= eta_beta * h_beta / (sqrt(G_beta) + eps)
     beta1 = clip(beta1, beta_min, beta_max)
  5. Run standard AdamW step with adapted lr (and beta1)
  6. Store d_k, ||g_k||^2, m_k for next step
```

### 2.2 Design Decisions

**Why adapt global LR, not per-parameter?** AdamW already provides per-parameter adaptation through its second moment estimates (v_hat). Adding per-parameter hypergradient adaptation would double the memory overhead and interfere with AdamW's own adaptive scaling. The global LR controls the overall step scale, which is what needs dynamic adjustment.

**Why AdaGrad for the meta-update?** AdaGrad's sum-of-squared-gradients accumulator naturally provides decreasing step sizes. For hypergradient descent, this means aggressive adaptation early in training (when the LR may be far from optimal) and conservative updates later (when the LR has stabilized). This is exactly the behavior we want.

**Why normalize by ||g||^2?** Without normalization, the hypergradient magnitude depends on the loss scale and gradient magnitude. Normalizing makes the algorithm invariant to these factors -- a model with loss = 100 and one with loss = 0.01 receive hypergradient updates of similar magnitude.

**Why warmup?** Adam's moment estimates are heavily biased in the first few steps (before the bias correction term 1/(1-beta^t) stabilizes). Adapting LR based on these unreliable estimates causes erratic behavior. A short warmup (10-20 steps) lets the moments initialize.

### 2.3 Surprise Signal (Input-Level Gain)

We introduce a novel extension: **input-level surprise modulation**. The optimizer computes a gradient prediction error as a surprise signal, but instead of using it to modify internal optimizer state (e.g., beta1/momentum), it exposes the signal for the model to use during the forward pass as an input gain factor.

**Surprise signal**: We track an exponential moving average (EMA) of gradients and compute surprise as the normalized prediction error:

```
g_ema = gamma * g_ema + (1-gamma) * g_k          # gradient EMA
var_ema = gamma * var_ema + (1-gamma) * ||g_k||^2  # variance EMA
raw_surprise = ||g_k - g_ema||^2 / (var_ema + eps) # normalized prediction error
S_instant = tanh(raw_surprise)                      # squashed to [0, 1]
```

**Accumulated surprise (S_bar)**: Rather than using the instant surprise directly, we maintain an EMA accumulator that tracks sustained surprise over time:

```
S_bar = alpha * S_instant + (1-alpha) * S_bar_prev
```

where `alpha` (default 0.1) controls the accumulation rate. This mirrors biological stress hormones that build up and decay with a time constant — a single surprising event produces a brief S_bar bump, while sustained novelty (e.g., a distribution shift) causes S_bar to rise steadily.

**Inverted-U gain (Yerkes-Dodson / trauma protection)**: The gain signal is not S_bar directly, but an inverted-U function of it:

```
gain = S_bar * exp(-S_bar / trauma_threshold)
```

This curve peaks at `S_bar = trauma_threshold` (default 0.5) and suppresses above it:

```
S_bar:  0.0  0.1  0.2  0.3  0.5  0.7  1.0  1.5  2.0
gain:   0.00 0.08 0.13 0.16 0.18 0.17 0.14 0.07 0.04
                                  ^peak
```

The optimizer exposes this value as the `current_surprise_gain` property. The model uses it to scale inputs in the forward pass:

```
x_eff = x * (1 + scale * optimizer.current_surprise_gain)
```

where `scale` is a model-side hyperparameter controlling the maximum amplification.

**Biological motivation**: This three-stage design mirrors biological stress response systems:

1. **Norepinephrine (instant surprise)**: The gradient prediction error mirrors the locus coeruleus detecting unexpected stimuli and releasing norepinephrine, which modulates neural gain — the sensitivity of neurons to inputs — rather than synaptic plasticity rates.

2. **HPA axis (S_bar accumulation)**: The accumulated surprise tracks the cumulative stress response. Just as cortisol builds up under sustained stress, S_bar integrates surprise over time with an exponential decay.

3. **Yerkes-Dodson / trauma protection (inverted-U gain)**: Moderate arousal enhances performance (peak gain at `S_bar = trauma_threshold`), but extreme or chronic stress impairs it. If surprise stays high for many consecutive steps (trauma), the gain *suppresses* instead of amplifying, preventing the model from fixating on a single extreme event. This implements the biological observation that acute stress enhances memory encoding while chronic stress impairs plasticity.

**Why input-level, not backprop-level?** Modulating beta1/momentum at the optimizer level conflates two concerns: detecting novelty and deciding how to respond. Exposing surprise as an input gain keeps these separate. The optimizer detects novelty (gradient prediction error); the model decides how to use it (input scaling). This is cleaner architecturally and avoids destabilizing the optimizer's internal dynamics. An earlier version modulated beta1 directly, which created a positive feedback loop: overshoot → gradient reversal → high surprise → more momentum → more overshoot.

**SNN benefits**: In spiking networks, silent neurons that are below threshold fail to contribute to the forward pass. During phase transitions (e.g., when the loss landscape shifts), many neurons may go silent simultaneously. Input-level surprise amplifies the presynaptic current flowing into these neurons, helping them cross the firing threshold and resume contributing gradients. This is more direct than momentum-based approaches, which can only affect neurons that are already producing non-zero gradients.

The tanh squashing on S_instant prevents runaway feedback: without it, surprise-driven gain increase could cause loss spikes, which produce large gradient prediction errors, which further increase surprise. The tanh saturates extreme values while preserving moderate surprise signals. The inverted-U curve provides a second layer of protection: even if S_instant saturates at 1.0 for many steps, the accumulated S_bar will eventually exceed the trauma threshold and the gain will drop.

### 2.4 Relationship to OSGM Reference

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
2. **AutoHypergradientAdamW**: OSGM-style self-tuning. AdaGrad accumulator eliminates beta_hyper tuning. Optional input-level surprise signal for non-stationary landscapes.

### 3.2 Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| lr | 1e-3 | Initial learning rate |
| hyper_lr | 0.01 | Meta-learning rate (auto-modulated by AdaGrad) |
| warmup_steps | 10 | Steps before LR adaptation begins |
| lr_min, lr_max | 1e-6, 1.0 | LR clamp bounds |
| adapt_momentum | False | Also adapt beta1 via hypergradient |
| hyper_lr_beta | 1.0 | Meta-LR for beta1 adaptation |
| track_surprise | False | Compute and expose surprise signal for input gain |
| surprise_gamma | 0.9 | EMA decay for gradient tracking |
| surprise_alpha | 0.1 | EMA decay for S_bar accumulation (lower = longer memory) |
| trauma_threshold | 0.5 | S_bar level where inverted-U gain peaks before suppression |
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

### 5.2 Results

| Model | Optimizer | LR | Train Acc | Test Acc | Fwd ms/batch |
|-------|-----------|-----|-----------|----------|--------------|
| CSNN-IF | Manual SGD | 0.005 | 66.06% | 66.85% | 401.2ms |
| CSNN-LIF | Manual SGD | 0.005 | 46.51% | 45.10% | 417.3ms |
| CSNN-LIF | AutoHypergradientAdamW | auto | 73.93% | **73.60%** | 429.1ms |
| CSNN-LIF | Auto + Surprise (gain=0.5) | auto | TBD | TBD | TBD |

Key observations:
- LIF with SGD (45.10%) performs much worse than IF with SGD (66.85%), showing that LIF's leaky dynamics are harder to optimize with a fixed learning rate
- AutoHypergradientAdamW transforms LIF from the worst to the best model (73.60%), a **28.5pp improvement** over LIF+SGD
- The throughput overhead of the auto optimizer is minimal (~3% slower than SGD, due to AdamW moment computation, not the hypergradient itself)
- Surprise + inverted-U gain results are pending benchmark run

### 5.3 LR Trajectory Analysis

The auto optimizer's learning rate trajectory over 776 steps:

```
Start: 0.001000
End:   0.002407
Min:   0.000010
Max:   0.014307
```

The trajectory reveals three phases:
1. **Exploration phase** (steps 0-100): LR increases from 0.001 to ~0.014 as the optimizer discovers that the initial LR is too conservative for LIF neurons that are mostly silent
2. **Correction phase** (steps 100-300): LR drops sharply to ~0.00001 when aggressive updates cause loss spikes, demonstrating the AdaGrad stabilization working correctly
3. **Convergence phase** (steps 300-776): LR settles around 0.002-0.003, having found the regime where LIF neurons fire reliably and gradients are informative

This adaptive behavior is impossible to replicate with fixed schedules without extensive hyperparameter search.

## 6. Discussion

### 6.1 Why Auto Hypergradient is Particularly Suited for SNNs

1. **Noisy surrogate gradients**: The surrogate gradient approximation introduces systematic bias. The gradient signal quality varies as neurons change firing patterns -- some timesteps produce informative gradients while others are near-zero (dead neurons) or saturated. Auto LR adaptation naturally handles this.

2. **Phase transitions**: During training, neurons can transition between firing and silent phases. When many neurons go silent, gradients vanish and a fixed LR makes no progress. The hypergradient detects this (consecutive gradients disagree with update direction) and increases the LR.

3. **No scheduling needed**: Standard SNN training uses cosine annealing or step decay schedules. These require tuning the schedule hyperparameters (T_max, milestones, gamma). Auto hypergradient eliminates this entirely.

4. **Temporal dynamics**: The T-dimensional computation creates a complex loss landscape with temporal correlations. The optimal LR at the start of training (random weights, sparse firing) is different from mid-training (structured firing patterns). Auto adaptation handles this transition.

5. **Biologically motivated input-level surprise with trauma protection**: The optional surprise signal operates at the input level (modulating neural gain) rather than the optimizer level (modulating momentum). This mirrors the norepinephrine/HPA-axis stress response: instant surprise (norepinephrine) is accumulated into S_bar (cortisol-like), and an inverted-U gain function (Yerkes-Dodson) ensures moderate surprise enhances learning while chronic high surprise (trauma) suppresses gain to prevent fixation. For SNNs, this helps silent neurons cross their firing threshold during transient phase transitions while protecting against catastrophic overreaction to sustained distribution shifts.

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
| `tests/test_hypergradient.py` | 31 unit tests |
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

Note: When `track_surprise=True`, the optimizer tracks three levels of surprise:

1. **`current_surprise`**: Instant gradient prediction error, tanh-squashed to [0, 1]
2. **`accumulated_surprise`** (S_bar): EMA of instant surprise, controlled by `surprise_alpha` (default 0.1). Tracks sustained novelty.
3. **`current_surprise_gain`**: Inverted-U function of S_bar: `gain = S_bar * exp(-S_bar / trauma_threshold)`. Peaks at moderate S_bar, suppresses at high S_bar (trauma protection).

The model reads `current_surprise_gain` for input-level modulation: `x_eff = x * (1 + scale * optimizer.current_surprise_gain)`. Unlike earlier designs that used surprise to modulate beta1/momentum internally, the surprise signal is purely informational from the optimizer's perspective — the model decides how to apply it.

### B.3 Memory Overhead

Per parameter tensor of size N:
- HypergradientAdamW: +N floats (previous direction d_{k-1})
- AutoHypergradientAdamW: +N floats (d_{k-1}) + 2 scalars (G_lr, prev_grad_norm_sq)
- With adapt_momentum: +N floats (m_{k-1}) + 1 scalar (G_beta)

Compared to AdamW's base 2N (m, v), the overhead is +N (50% increase) which is comparable to AMSGrad's overhead.
