# Differentiable VSA I-RAVEN Pipeline — Complete Architecture

**Date:** 2026-03-19
**Status:** Design complete, GPU shader compiled, C++ FFT ops implemented, autograd nodes designed

## Overview

End-to-end differentiable neuro-symbolic pipeline for solving I-RAVEN visual reasoning problems. A CNN perceives panel images, outputs are projected onto block-code probability simplexes via block-wise softmax, then VSA algebra (bind/unbind/bundle) extracts transformation rules and predicts missing panels. Cross-entropy loss backpropagates through the entire graph — the backward of bind IS unbind (same GPU shader, flip flag).

## Architecture

```
Raw Images → CNN Backbone → Per-Attribute Linear Heads → Block Softmax
  → role-filler Bind → Bundle entities → Panel Vectors
  → Unbind(P2, P1) = Rule T → Bind(P2, T) = Predicted P3
  → Similarity(Predicted, Candidates) → cosineTopmf → Cross-Entropy Loss
  → .backward() flows through: unbind ← bind ← bundle ← blockSoftmax ← CNN
```

## 1. Perception Bridge (CNN → VSA)

### Block-Wise Softmax
Each of k blocks independently sums to 1.0, creating valid probability simplexes. This is the CNN-to-VSA bridge.

```cpp
std::vector<float> blockSoftmax(const float* logits, uint32_t k, uint32_t l);
```

For d = k * l (e.g., 2048 with k=16, l=128), the neural network outputs a vector where each of the k blocks independently sums to 1.0.

### Neural Architecture
- Conv2D backbone: (1,80,80) → (16,38,38) → (32,18,18) → (64,8,8)
- Per-attribute linear heads: shape, color, size, position → each outputs (k*l) logits
- Block softmax per head → valid block-code probability simplex

## 2. VSA Reasoning (RavenReasoner)

### Panel Encoding
```cpp
// For each entity in the panel:
bound_shape = blockCodeBind(shape_vec, role_shape);
bound_color = blockCodeBind(color_vec, role_color);
bound_size  = blockCodeBind(size_vec, role_size);
entity_vec  = blockCodeBundle({bound_shape, bound_color, bound_size}, normalize=true);
positioned  = blockCodeBind(entity_vec, position_role);
panel_vec   = blockCodeBundle(all_positioned_entities, normalize=true);
```

### Rule Extraction & Prediction
```cpp
// Extract transformation from completed rows
T01 = blockCodeUnbind(panel1, panel0);  // Row 0 rule
T12 = blockCodeUnbind(panel2, panel1);
T34 = blockCodeUnbind(panel4, panel3);  // Row 1 rule
T45 = blockCodeUnbind(panel5, panel4);

// Average rules (should converge if CNN encodes consistently)
avg_rule = blockCodeBundle({T01, T12, T34, T45}, normalize=true);

// Predict missing panel
predicted = blockCodeBind(panel7, avg_rule);
```

### Candidate Scoring
```cpp
similarities = blockCodeSimilarityBatch(predicted, candidates, 8, k, l);
probs = blockCodeCosineTopmf(similarities, 8, temperature=40.0);
loss = -log(probs[correct_idx]);  // Cross-entropy
```

## 3. GPU Shader (fft-bind.glsl)

Radix-2 Cooley-Tukey FFT in shared memory. One workgroup per block, one thread per element.

- **Bind** (is_unbind=0): C = IFFT(FFT(A) * FFT(B)) — circular convolution
- **Unbind** (is_unbind=1): C = IFFT(FFT(A) * conj(FFT(B))) — circular correlation
- Complexity: O(k * l * log(l)) vs O(k * l^2) for direct

Push constants: `{uint k, uint l, uint is_unbind}`
Shader path: `shaders/fft-bind.glsl` → `shaders/spv/fft-bind.spv`

## 4. Autograd — The Self-Learning Mechanism

### Bind Backward (the key insight)
```
If C = bind(A, B), then:
  dL/dA = unbind(grad_C, B)    ← same shader with is_unbind=1
  dL/dB = unbind(grad_C, A)    ← same shader with is_unbind=1
```

### Bundle Backward (with L1 normalization)
```
If z = normalize(sum(inputs)), S = L1_norm(sum), then:
  dL/dy_i = (1/S) * (g_i - sign(z_i) * sum(g_m * z_m))
  dL/dx_j = dL/dy  (broadcast to all inputs)
```

### BlockSoftmax Backward
Standard per-block softmax Jacobian applied independently to each block.

### Similarity Backward
```
dL/d_predicted = candidate / (k * l)  (scaled by the matched candidate)
```

## 5. Training Loop

```cpp
// For each I-RAVEN problem:
// 1. Encode all 16 panels through PerceptionBridge (CNN → blockSoftmax)
// 2. Multi-row rule extraction: unbind consecutive panels, average 4 rules
// 3. Predict missing panel: bind(panel7, avg_rule)
// 4. Score 8 candidates: similarity → cosineTopmf → probs
// 5. Loss: -log(probs[correct])
// 6. Backward: gradients flow through unbind → bind → bundle → blockSoftmax → CNN
// The VSA forces the CNN to learn disentangled attribute representations
```

## 6. Key Parameters (proven in eval_iraven_full.py)

- K=8, L=32 (d_vsa=256) — smaller works better for I-RAVEN
- Temperature=40.0 for scoring softmax
- N_STATES=16 for HMM fallback codebook
- 20 EM epochs for HMM pre-training
- Attributes: Type, Size, Color (Angle optional)
- Max entities per panel: 9 (distribute_nine config)

## 7. Implementation Status

### Done
- [x] `block_ops.h`: FFT bind/unbind declarations, blockSoftmax, FFTBindParams
- [x] `block_ops.cpp`: CPU FFT implementations using tensor_ops::fft_1d/ifft_1d
- [x] `fft-bind.glsl`: Full GPU shader, compiled to SPIR-V
- [x] Memory notes saved for next session continuity

### TODO
- [ ] GPU dispatch function (`blockCodeBindGPU`) using CommandBatch/BufferPool/PipelineCache
- [ ] `BlockCodeBindFunction` autograd node (forward=bind, backward=unbind)
- [ ] `BlockCodeBundleFunction` autograd node (L1 normalization Jacobian)
- [ ] pybind11 bindings for blockCodeBindFFT, blockCodeUnbindFFT, blockSoftmax
- [ ] Python prototype: validate on real RAVEN data before full C++ port
- [ ] Wire into CubeMind benchmark with PerceptionBridge class
- [ ] Full training loop with Adam optimizer

## 8. Reference Files

- Old working solver: `C:\Users\grill\empirical_grilly_next\scripts\eval_iraven_full.py` (97.5%)
- Old learned solver: `C:\Users\grill\empirical_grilly_next\scripts\eval_iraven_learned.py`
- CubeMind benchmark: `C:\Users\grill\Documents\GitHub\cubemind\benchmarks\iraven.py`
- Block ops C++: `C:\Users\grill\Documents\GitHub\grilly\cpp\src\cubemind\block_ops.cpp`
- FFT shader: `C:\Users\grill\Documents\GitHub\grilly\shaders\fft-bind.glsl`
