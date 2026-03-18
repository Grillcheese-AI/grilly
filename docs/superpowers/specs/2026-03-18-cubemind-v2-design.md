# CubeMind v2 — Clean Reimplementation on grilly 0.5.0

**Date:** 2026-03-18
**Author:** grillcheese + Claude
**Repo:** `Grillcheese-AI/cubemind` (new, clean)
**Depends on:** grilly >= 0.5.0

## Summary

CubeMind v2 is a clean-room reimplementation of the CubeMind neuro-vector-symbolic architecture, rebuilt from scratch on grilly 0.5.0's GPU-first infrastructure. Every module validates against the 6 published papers (10 formal theorems). The old `empirical_grilly_next/cubemind/` (25 modules, ~8K lines of validation code) and `grillcheese/` (brain architecture) are reference — not copied.

### Objectives

1. Clean architecture — proper OOP, <1000 lines/file, each module has one job
2. GPU-first — all hot-path ops on VulkanTensor via grilly C++ backend
3. Paper-validated — every module has a test verifying the paper's math
4. 97.5%+ iRaven accuracy — match or beat v1 benchmark
5. Port surprise-momentum optimizer from grillcheese for biologically-grounded learning

---

## Architecture

### Pipeline

```
Input (text/vector)
  │
  ▼
PERCEPTION — BatchVSAEncoder → block-code (k×l)
  │
  ▼
ROUTING — CubeMindRouter: cosine sim to prototypes → DSelectK → top-k experts
  │
  ▼
MEMORY — VSACache: store block-code, compute surprise + stress
  │
  ▼
DETECTION — HMM Ensemble: forward algorithm → rule weights per expert
  │
  ▼
EXECUTION — HYLA hypernetwork: condition on block-code → output weights
           CVL: contrastive Q-value for action selection
  │
  ▼
ANSWER — codebook lookup: output block-code → decoded answer
```

All tensors are VulkanTensor. Data uploads once at perception, stays on GPU through the pipeline, downloads once at answer.

### Project Structure

```
cubemind/
├── pyproject.toml              # depends on grilly>=0.5.0
├── cubemind/
│   ├── __init__.py
│   ├── core.py                 # Strategy enum, K_BLOCKS, L_BLOCK, D_VSA, Hyperfan init
│   ├── model.py                # CubeMind: full pipeline orchestrator
│   │
│   ├── perception/
│   │   ├── __init__.py
│   │   └── encoder.py          # Text → block-code via grilly.BatchVSAEncoder
│   │
│   ├── routing/
│   │   ├── __init__.py
│   │   ├── router.py           # CubeMindRouter: prototype similarity
│   │   └── moe_gate.py         # DSelectK sparse expert gate
│   │
│   ├── reasoning/
│   │   ├── __init__.py
│   │   ├── hmm_rule.py         # HMM ensemble: Baum-Welch EM, Viterbi, forward-backward
│   │   └── combiner.py         # Combiner-Axial attention
│   │
│   ├── execution/
│   │   ├── __init__.py
│   │   ├── hyla.py             # HYLA hypernetwork (wraps grilly.nn.HYLAAttention)
│   │   ├── cvl.py              # Contrastive Value Learning (TD-free Q-values)
│   │   └── decoder.py          # Block-code → output decoding
│   │
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── cache.py            # VSACache: surprise/stress, hippocampal-style
│   │   ├── hippocampal.py      # DG/CA3 encode (ported from grillcheese)
│   │   └── replay.py           # Experience replay buffer
│   │
│   ├── ops/
│   │   ├── __init__.py
│   │   ├── block_codes.py      # Wraps grilly blockcode-bind/unbind/similarity shaders
│   │   └── hdc.py              # Wraps grilly HDC packed ops (32x compression)
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py          # End-to-end GPU training loop
│   │   ├── losses.py           # CIW, DROPS, BCE, cross-entropy
│   │   ├── disarm.py           # Wraps grilly.nn.DisARMSampler
│   │   ├── surprise_optim.py   # SurpriseMomentumOptimizer (ported from grillcheese)
│   │   └── hopfield_optim.py   # HopfieldSurpriseOptimizer (ported from grillcheese)
│   │
│   ├── safety/
│   │   ├── __init__.py
│   │   ├── debiasing.py        # Fairness constraints
│   │   └── dp_privacy.py       # Differential privacy
│   │
│   └── experimental/
│       ├── __init__.py
│       ├── bandits.py           # Contextual bandits
│       ├── burn_feed.py         # Ecological burn context
│       ├── theory_of_mind.py    # ToM modeling
│       ├── hyperattention.py    # HyperAttention variant
│       ├── vs_graph.py          # Vector-symbolic graphs
│       └── convergence.py       # Convergence monitoring
│
├── tests/
│   ├── test_block_codes.py      # Theorem 1: magnitude preservation
│   ├── test_block_kernel.py     # Theorem 7: PSD kernel
│   ├── test_hmm_gfsa.py         # Theorem 2: HMM = GFSA
│   ├── test_hmm_convergence.py  # Theorem 8: log-param convergence
│   ├── test_hyperfan.py         # Theorem 3: variance formula
│   ├── test_cvl.py              # Theorem 4: InfoNCE recovery
│   ├── test_combiner.py         # Theorem 5: sub-quadratic
│   ├── test_disarm.py           # Theorem 6: variance ordering
│   ├── test_router.py           # MoWM paper: routing correctness
│   ├── test_hyla.py             # MoWM paper: hypernetwork output
│   ├── test_cache.py            # Surprise/stress computation
│   ├── test_surprise_optim.py   # Surprise-momentum convergence
│   ├── test_pipeline.py         # End-to-end pipeline
│   └── test_iraven.py           # iRaven benchmark: ≥97.5%
│
├── benchmarks/
│   ├── iraven.py                # iRaven eval suite
│   └── xraven.py                # X-RAVEN (harder distribution)
│
├── configs/
│   └── default.yaml             # Hyperparameters
│
└── scripts/
    ├── train.py
    └── eval.py
```

---

## grilly Integration Map

| CubeMind Module | grilly Feature Used |
|----------------|---------------------|
| `ops/block_codes.py` | `_bridge.blockcode_bind/unbind/similarity` → SPIR-V shaders |
| `ops/hdc.py` | `_bridge.hdc_bind_packed/bundle/similarity/permute` → 32x compression |
| `execution/hyla.py` | `grilly.nn.HYLAAttention` (GPU-backed, tested) |
| `training/disarm.py` | `grilly.nn.DisARMSampler` (antithetic sampling) |
| `perception/encoder.py` | `grilly.experimental.language.BatchVSAEncoder` |
| `reasoning/combiner.py` | `grilly.nn.SympFormerBlock` + `grilly.nn.MultiheadAttention` |
| All tensors | `grilly.utils.VulkanTensor` (C++ backend, GPU-first) |
| Similarity search | `grilly.functional.faiss_distance/faiss_topk` |
| Training loop | `grilly.optim.SGD` + surprise-momentum on top |
| Neurogenesis | `sanger-gha.spv` (online PCA for 75% capacity expansion) |
| Adaptive cache | `grilly.backend.adaptive_kv` (TAPPA q-similarity) |

### What CubeMind implements itself (domain-specific)

- HMM ensemble Baum-Welch EM + Viterbi + forward-backward
- CubeMindRouter prototype matching + training
- CVL contrastive value learning (InfoNCE occupancy measure)
- VSACache surprise/stress computation
- DSelectK MoE gate
- CIW/DROPS losses
- Surprise-momentum optimizer (ported from grillcheese, adapted for VSA)
- Hippocampal DG/CA3 gradient episode storage
- Debiasing, DP privacy
- All experimental modules (bandits, burn feed, ToM, etc.)

---

## Validation Matrix

Each module maps to specific paper theorems:

| Module | Paper | Theorem | Test |
|--------|-------|---------|------|
| `ops/block_codes.py` | Formal Proofs | T1: L1 norm preserved after N bindings | `test_magnitude_preservation_1000_chains` |
| `ops/block_codes.py` | Formal Proofs | T7: Similarity kernel is PSD | `test_kernel_positive_semidefinite` |
| `reasoning/hmm_rule.py` | Formal Proofs | T2: HMM forward = GFSA absorbing probs | `test_hmm_gfsa_equivalence` |
| `reasoning/hmm_rule.py` | Formal Proofs | T8: Log-param HMM converges | `test_hmm_convergence_monotonic` |
| `execution/hyla.py` | Formal Proofs | T3: Hyperfan variance matches formula | `test_hyperfan_variance_empirical` |
| `execution/cvl.py` | Formal Proofs | T4: InfoNCE recovers Q-values | `test_cvl_infonce_recovery` |
| `reasoning/combiner.py` | Formal Proofs | T5: Sub-quadratic complexity | `test_combiner_scales_subquadratic` |
| `training/disarm.py` | Formal Proofs | T6: DisARM < ARM < REINFORCE variance | `test_disarm_variance_ordering` |
| `routing/router.py` | MoWM | Prototype similarity routing | `test_routing_matches_paper` |
| `execution/hyla.py` | MoWM | HYLA hypernetwork output | `test_hyla_conditioning` |
| `training/surprise_optim.py` | Grillcheese | Surprise-driven adaptive LR | `test_surprise_optim_convergence` |
| `model.py` | CubeMind main | Full pipeline end-to-end | `test_iraven_97_5_percent` |

---

## Ported from grillcheese

### Surprise-Momentum Optimizer

Source: `grillcheese/grillcheese/optim/surprise_momentum.py`

Pipeline:
1. Current gradient `g_t` arrives
2. Hippocampal CA3 retrieves k-nearest gradient episodes → recalled context `g_recall`
3. Instant surprise = `||g_t - g_recall||` (prediction error)
4. Biological momentum = EMA of surprise (like β₁ in Adam but adaptive)
5. Effective LR = `lr_base * (surprise_floor + surprise)` (high surprise → learn faster)
6. Weight update: `w -= effective_lr * (g_t + λ * g_recall)`

Dependencies ported:
- `HippocampalOptimizer` → `training/hippo_opt.py` (gradient episode storage)
- `GrillyHippocampalMemory` → `memory/hippocampal.py` (DG/CA3 encode)
- `OjaPlasticity` → use grilly's `sanger-gha.spv` shader instead
- `AmygdalaState` → merge into `memory/cache.py` (surprise/stress already there)

### Hopfield-Surprise Optimizer

Extends surprise-momentum with:
1. High-surprise gradient episodes stored as Hopfield patterns
2. Current gradient refined by iterative attractor convergence
3. Final update = gradient + biological momentum + attractor correction

---

## Key Differences from v1

| Aspect | v1 (empirical_grilly_next) | v2 (new repo) |
|--------|---------------------------|---------------|
| Tensor backend | `numpy` via deprecated `Compute()` | `VulkanTensor` via C++ bridge |
| Block-code ops | Python loops | `blockcode-bind.spv` GPU shader |
| HYLA | Local reimplementation | `grilly.nn.HYLAAttention` |
| DisARM | Local reimplementation | `grilly.nn.DisARMSampler` |
| Optimizer | numpy SGD | Surprise-momentum (ported from grillcheese) |
| Perception | Sequential encode | `BatchVSAEncoder` (GPU batch) |
| Code quality | "all over the place" validation code | Clean OOP, <1000 lines/file |
| Tests | Basic functionality | Paper theorem validation |
| Data on GPU | Ping-pong every op | Single upload/download per step |

---

## Implementation Order

1. **Scaffold** — create repo, pyproject.toml, directory structure
2. **ops/** — block_codes.py, hdc.py (wrappers around grilly shaders)
3. **core.py** — constants, Hyperfan init
4. **perception/** — encoder.py wrapping BatchVSAEncoder
5. **routing/** — router.py, moe_gate.py
6. **reasoning/** — hmm_rule.py (the hardest module — Baum-Welch + Viterbi)
7. **execution/** — hyla.py, cvl.py, decoder.py
8. **memory/** — cache.py, hippocampal.py, replay.py
9. **training/** — surprise_optim.py, hopfield_optim.py, losses.py, trainer.py
10. **model.py** — wire the full pipeline
11. **safety/** — debiasing.py, dp_privacy.py
12. **experimental/** — all experimental modules
13. **tests/** — theorem validation tests
14. **benchmarks/** — iRaven eval, X-RAVEN

Each step produces working, testable code before moving to the next.

---

## GitHub Setup

1. Delete existing `Grillcheese-AI/cubemind` repo
2. Create fresh `Grillcheese-AI/cubemind` repo
3. Old code archived at `C:\Users\grill\Documents\GitHub\cubemind-v1-archive`
4. Old validation code at `C:\Users\grill\empirical_grilly_next\cubemind\` (reference only)
5. Old brain code at `C:\Users\grill\Documents\GitHub\grillcheese\` (reference only)
