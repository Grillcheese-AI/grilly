# CubeMind v2 Plan A: Foundation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a clean CubeMind v2 repo with ops wrappers, core constants, perception encoder, and topic router — the foundation that Plans B and C build on.

**Architecture:** New repo at `C:\Users\grill\Documents\GitHub\cubemind`. Depends on grilly >= 0.5.0. Each module wraps grilly GPU ops. Block-code operations go through grilly's compiled SPIR-V shaders via `backend._bridge`. All tensors are numpy for now (VulkanTensor integration in Plan C's trainer).

**Tech Stack:** Python 3.12+, grilly 0.5.0, numpy, pytest

**Spec:** `docs/superpowers/specs/2026-03-18-cubemind-v2-design.md`

**Reference code:**
- Old CubeMind: `C:\Users\grill\empirical_grilly_next\cubemind\` (indexed in elephant-coder)
- Old grillcheese: `C:\Users\grill\Documents\GitHub\grillcheese\` (indexed in elephant-coder)
- grilly block ops: `grilly.experimental.vsa.block_ops.BlockCodeOps`

---

## File Structure (Plan A only)

### New Files

| File | Responsibility |
|------|---------------|
| `pyproject.toml` | Package config, grilly dependency |
| `cubemind/__init__.py` | Package entry, version |
| `cubemind/core.py` | Strategy enum, K_BLOCKS=16, L_BLOCK=128, D_VSA=2048, Hyperfan init |
| `cubemind/ops/__init__.py` | Ops subpackage |
| `cubemind/ops/block_codes.py` | Wraps grilly's BlockCodeOps + bridge shaders |
| `cubemind/ops/hdc.py` | Wraps grilly's HDC packed ops |
| `cubemind/perception/__init__.py` | Perception subpackage |
| `cubemind/perception/encoder.py` | Text → block-code via BatchVSAEncoder |
| `cubemind/routing/__init__.py` | Routing subpackage |
| `cubemind/routing/router.py` | CubeMindRouter: prototype similarity |
| `cubemind/routing/moe_gate.py` | DSelectK sparse expert gate |
| `tests/__init__.py` | Test package |
| `tests/test_block_codes.py` | Theorem 1 + 7 validation |
| `tests/test_perception.py` | Encoder shape + roundtrip tests |
| `tests/test_router.py` | Routing correctness tests |

---

## Task 1: Scaffold Repo + GitHub

**Files:**
- Create: `C:\Users\grill\Documents\GitHub\cubemind\pyproject.toml`
- Create: `C:\Users\grill\Documents\GitHub\cubemind\cubemind\__init__.py`
- Create: `C:\Users\grill\Documents\GitHub\cubemind\.gitignore`
- Create: `C:\Users\grill\Documents\GitHub\cubemind\README.md`

- [ ] **Step 1: Delete old GitHub repo and create new one**

```bash
gh repo delete Grillcheese-AI/cubemind --yes 2>/dev/null
mkdir -p C:/Users/grill/Documents/GitHub/cubemind
cd C:/Users/grill/Documents/GitHub/cubemind
git init
```

- [ ] **Step 2: Create pyproject.toml**

```toml
[project]
name = "cubemind"
version = "2.0.0"
description = "Neuro-vector-symbolic architecture for compositional reasoning on consumer hardware"
requires-python = ">=3.10"
dependencies = [
    "grilly>=0.5.0",
    "numpy>=1.24",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "ruff>=0.4"]

[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
```

- [ ] **Step 3: Create __init__.py**

```python
"""CubeMind v2 — neuro-vector-symbolic reasoning on grilly GPU backend."""
__version__ = "2.0.0"
```

- [ ] **Step 4: Create .gitignore and README.md**

.gitignore: standard Python + .venv, __pycache__, *.egg-info, dist, build, .pytest_cache

README.md: one-liner placeholder

- [ ] **Step 5: Create subdirectory structure**

```bash
mkdir -p cubemind/ops cubemind/perception cubemind/routing
mkdir -p cubemind/reasoning cubemind/execution cubemind/memory
mkdir -p cubemind/training cubemind/safety cubemind/experimental
mkdir -p tests benchmarks configs scripts
touch cubemind/ops/__init__.py cubemind/perception/__init__.py
touch cubemind/routing/__init__.py cubemind/reasoning/__init__.py
touch cubemind/execution/__init__.py cubemind/memory/__init__.py
touch cubemind/training/__init__.py cubemind/safety/__init__.py
touch cubemind/experimental/__init__.py tests/__init__.py
```

- [ ] **Step 6: Initial commit + push to GitHub**

```bash
git add -A
git commit -m "scaffold: cubemind v2 repo with directory structure"
gh repo create Grillcheese-AI/cubemind --public --source=. --push
```

---

## Task 2: core.py — Constants + Hyperfan Init

**Files:**
- Create: `cubemind/core.py`
- Create: `tests/test_core.py`

- [ ] **Step 1: Write failing test for Hyperfan init (Theorem 3)**

```python
# tests/test_core.py
import numpy as np
from cubemind.core import K_BLOCKS, L_BLOCK, D_VSA, Strategy, hyperfan_init, hyperfan_in_variance


def test_constants():
    assert K_BLOCKS == 16
    assert L_BLOCK == 128
    assert D_VSA == K_BLOCKS * L_BLOCK == 2048


def test_strategy_enum():
    assert Strategy.BLOCK_CODE.value == "block_code"
    assert Strategy.BLAKE3.value == "blake3"


def test_hyperfan_variance_formula():
    """Theorem 3: Hyperfan variance = act_factor / (bias_factor * fan_in * d_k * var_e)"""
    fan_in, d_k, l = 64, 128, 128
    var = hyperfan_in_variance(fan_in, d_k, l, has_bias=False, activation="gelu")
    var_e = 1.0 / l
    expected = 1.7 / (1.0 * fan_in * d_k * var_e)  # gelu factor = 1.7
    np.testing.assert_allclose(var, expected, rtol=1e-6)


def test_hyperfan_init_shape():
    W = hyperfan_init(fan_out=32, fan_in=64, d_k=128, l=128, seed=42)
    assert W.shape == (32 * 64, 128)
    assert W.dtype == np.float32


def test_hyperfan_init_variance_empirical():
    """Empirically verify Hyperfan init produces correct variance."""
    W = hyperfan_init(fan_out=256, fan_in=256, d_k=128, l=128, seed=42)
    expected_var = hyperfan_in_variance(256, 128, 128)
    empirical_var = np.var(W)
    # Should be within 20% for this sample size
    np.testing.assert_allclose(empirical_var, expected_var, rtol=0.2)
```

- [ ] **Step 2: Implement core.py**

Port from `empirical_grilly_next/cubemind/core.py` — clean version with proper docstrings. Reference the file via elephant-coder: `recall_file_memories("C:\Users\grill\empirical_grilly_next\cubemind\core.py")`.

- [ ] **Step 3: Run tests, commit**

```bash
cd C:/Users/grill/Documents/GitHub/cubemind
uv run pytest tests/test_core.py -v
git add cubemind/core.py tests/test_core.py
git commit -m "feat: core.py — constants, Strategy enum, Hyperfan init (Theorem 3)"
```

---

## Task 3: ops/block_codes.py — GPU Block-Code Operations

**Files:**
- Create: `cubemind/ops/block_codes.py`
- Create: `tests/test_block_codes.py`

- [ ] **Step 1: Write failing tests (Theorems 1 + 7)**

```python
# tests/test_block_codes.py
import numpy as np
from cubemind.ops.block_codes import BlockCodes


def test_bind_returns_valid_block_code():
    bc = BlockCodes(k=16, l=128)
    a = bc.random_discrete(seed=42)
    b = bc.random_discrete(seed=43)
    c = bc.bind(a, b)
    assert c.shape == (16, 128)
    # Each block sums to 1 (one-hot)
    for j in range(16):
        np.testing.assert_allclose(c[j].sum(), 1.0, atol=1e-6)


def test_unbind_roundtrip():
    bc = BlockCodes(k=16, l=128)
    a = bc.random_discrete(seed=42)
    b = bc.random_discrete(seed=43)
    c = bc.bind(a, b)
    recovered = bc.unbind(c, b)
    np.testing.assert_allclose(bc.similarity(recovered, a), 1.0, atol=1e-6)


def test_theorem1_magnitude_preservation_1000_chains():
    """Theorem 1: L1 norm preserved after 1000 successive bindings."""
    bc = BlockCodes(k=16, l=128)
    v = bc.random_discrete(seed=0)
    for i in range(1000):
        r = bc.random_discrete(seed=i + 1)
        v = bc.bind(v, r)
    for j in range(16):
        np.testing.assert_allclose(v[j].sum(), 1.0, atol=1e-6)


def test_theorem7_kernel_positive_semidefinite():
    """Theorem 7: Block-code similarity kernel is PSD."""
    bc = BlockCodes(k=16, l=128)
    n = 20
    vectors = [bc.random_discrete(seed=i) for i in range(n)]
    # Build Gram matrix
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = bc.similarity(vectors[i], vectors[j])
    # All eigenvalues >= 0
    eigenvalues = np.linalg.eigvalsh(K)
    assert np.all(eigenvalues >= -1e-10)


def test_similarity_self():
    bc = BlockCodes(k=16, l=128)
    a = bc.random_discrete(seed=42)
    assert abs(bc.similarity(a, a) - 1.0) < 1e-6


def test_codebook_discrete():
    bc = BlockCodes(k=16, l=128)
    codebook = bc.codebook_discrete(n=32, seed=42)
    assert codebook.shape == (32, 16, 128)


def test_cosine_to_pmf():
    bc = BlockCodes(k=16, l=128)
    sims = np.array([0.9, 0.5, 0.1, 0.01])
    pmf = bc.cosine_to_pmf(sims, temperature=0.1)
    np.testing.assert_allclose(pmf.sum(), 1.0, atol=1e-6)
    assert pmf[0] > pmf[1] > pmf[2] > pmf[3]
```

- [ ] **Step 2: Implement block_codes.py**

Wraps `grilly.experimental.vsa.block_ops.BlockCodeOps` and `grilly.backend._bridge.blockcode_*` for GPU acceleration. Falls back to BlockCodeOps (numpy) if bridge unavailable.

```python
"""Block-code VSA operations — GPU-accelerated via grilly shaders.

Wraps grilly's BlockCodeOps for the algebraic operations and
routes through the C++ bridge for GPU dispatch when available.
"""

import numpy as np

from cubemind.core import K_BLOCKS, L_BLOCK


class BlockCodes:
    """Block-code vector space operations.

    Args:
        k: Number of blocks (default: K_BLOCKS=16)
        l: Block length (default: L_BLOCK=128)
    """

    def __init__(self, k: int = K_BLOCKS, l: int = L_BLOCK):
        self.k = k
        self.l = l
        self.d = k * l
        self._ops = None
        self._bridge = None
        self._init_backends()

    def _init_backends(self):
        try:
            from grilly.experimental.vsa.block_ops import BlockCodeOps
            self._ops = BlockCodeOps
        except ImportError:
            pass
        try:
            from grilly.backend import _bridge
            self._bridge = _bridge
        except ImportError:
            pass

    def random_discrete(self, seed=None):
        if self._ops:
            return self._ops.random_discrete(self.k, self.l, seed=seed)
        rng = np.random.default_rng(seed)
        v = np.zeros((self.k, self.l), dtype=np.float32)
        for j in range(self.k):
            v[j, rng.integers(self.l)] = 1.0
        return v

    def codebook_discrete(self, n, seed=None):
        if self._ops:
            return self._ops.codebook_discrete(self.k, self.l, n, seed=seed)
        return np.stack([self.random_discrete(seed=seed+i if seed else None) for i in range(n)])

    def bind(self, a, b):
        # Try GPU shader first
        if self._bridge:
            try:
                result = self._bridge.blockcode_bind(a, b, self.k, self.l)
                if result is not None:
                    return result.reshape(self.k, self.l)
            except Exception:
                pass
        if self._ops:
            return self._ops.bind(a, b)
        # Pure numpy fallback: per-block circular convolution
        # ... (implement from v1 reference)

    def unbind(self, composite, key):
        if self._bridge:
            try:
                result = self._bridge.blockcode_unbind(composite, key, self.k, self.l)
                if result is not None:
                    return result.reshape(self.k, self.l)
            except Exception:
                pass
        if self._ops:
            return self._ops.unbind(composite, key)
        # ... numpy fallback

    def similarity(self, a, b):
        if self._ops:
            return self._ops.similarity(a, b)
        return float(np.sum(a * b) / self.k)

    def similarity_batch(self, query, codebook):
        if self._ops:
            return self._ops.similarity_batch(query, codebook)
        return np.array([self.similarity(query, cb) for cb in codebook])

    def cosine_to_pmf(self, similarities, temperature=0.1):
        if self._ops:
            return self._ops.cosine_to_pmf(similarities, temperature)
        scaled = similarities / temperature
        exp_s = np.exp(scaled - scaled.max())
        return exp_s / exp_s.sum()

    # ... bundle, discretize, cyclic_shift, from_flat, to_flat
```

- [ ] **Step 3: Run tests, commit**

```bash
uv run pytest tests/test_block_codes.py -v
git add cubemind/ops/block_codes.py tests/test_block_codes.py
git commit -m "feat: block_codes.py — GPU block-code ops (Theorems 1+7 validated)"
```

---

## Task 4: ops/hdc.py — HDC Packed Operations

**Files:**
- Create: `cubemind/ops/hdc.py`
- Create: `tests/test_hdc.py`

- [ ] **Step 1: Write tests**

Test bind_packed, bundle_packed, similarity_packed, permute_packed via grilly bridge.

- [ ] **Step 2: Implement hdc.py**

Thin wrapper around `grilly.backend._bridge.hdc_*` functions with numpy fallbacks.

- [ ] **Step 3: Run tests, commit**

```bash
uv run pytest tests/test_hdc.py -v
git commit -m "feat: hdc.py — HDC packed ops wrapper (32x compression)"
```

---

## Task 5: perception/encoder.py — Text to Block-Code

**Files:**
- Create: `cubemind/perception/encoder.py`
- Create: `tests/test_perception.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_perception.py
import numpy as np
from cubemind.perception.encoder import Encoder
from cubemind.core import K_BLOCKS, L_BLOCK


def test_encode_text_shape():
    enc = Encoder(k=K_BLOCKS, l=L_BLOCK)
    vec = enc.encode("hello world")
    assert vec.shape == (K_BLOCKS, L_BLOCK)


def test_encode_batch_shape():
    enc = Encoder(k=K_BLOCKS, l=L_BLOCK)
    vecs = enc.encode_batch(["hello", "world", "test"])
    assert vecs.shape == (3, K_BLOCKS, L_BLOCK)


def test_similar_texts_higher_similarity():
    enc = Encoder(k=K_BLOCKS, l=L_BLOCK)
    from cubemind.ops.block_codes import BlockCodes
    bc = BlockCodes()
    v1 = enc.encode("the cat sat on the mat")
    v2 = enc.encode("the cat sat on the rug")
    v3 = enc.encode("quantum chromodynamics predicts hadron masses")
    sim_close = bc.similarity(v1, v2)
    sim_far = bc.similarity(v1, v3)
    assert sim_close > sim_far
```

- [ ] **Step 2: Implement encoder.py**

Uses `grilly.experimental.language.BatchVSAEncoder` for GPU batch encoding, then discretizes to block codes.

- [ ] **Step 3: Run tests, commit**

```bash
uv run pytest tests/test_perception.py -v
git commit -m "feat: perception encoder — text to block-code via BatchVSAEncoder"
```

---

## Task 6: routing/router.py — CubeMindRouter

**Files:**
- Create: `cubemind/routing/router.py`
- Create: `tests/test_router.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_router.py
import numpy as np
from cubemind.routing.router import CubeMindRouter
from cubemind.ops.block_codes import BlockCodes
from cubemind.core import K_BLOCKS, L_BLOCK


def test_router_construction():
    bc = BlockCodes()
    prototypes = np.stack([bc.random_discrete(seed=i) for i in range(5)])
    names = ["science", "sports", "politics", "tech", "arts"]
    router = CubeMindRouter(topic_names=names, prototypes=prototypes, k=K_BLOCKS, l=L_BLOCK)
    assert router.topic_count == 5


def test_route_returns_best_match():
    bc = BlockCodes()
    proto_a = bc.random_discrete(seed=10)
    proto_b = bc.random_discrete(seed=20)
    prototypes = np.stack([proto_a, proto_b])
    router = CubeMindRouter(["topicA", "topicB"], prototypes, K_BLOCKS, L_BLOCK)
    topic, score = router.route_vector(proto_a)
    assert topic == "topicA"
    assert score > 0.9


def test_route_topk():
    bc = BlockCodes()
    prototypes = np.stack([bc.random_discrete(seed=i) for i in range(10)])
    names = [f"topic_{i}" for i in range(10)]
    router = CubeMindRouter(names, prototypes, K_BLOCKS, L_BLOCK, top_k=3)
    results = router.route_topk_vector(prototypes[5])
    assert len(results) == 3
    assert results[0][0] == "topic_5"


def test_save_load(tmp_path):
    bc = BlockCodes()
    prototypes = np.stack([bc.random_discrete(seed=i) for i in range(3)])
    router = CubeMindRouter(["a", "b", "c"], prototypes, K_BLOCKS, L_BLOCK)
    path = tmp_path / "router.npz"
    router.save(str(path))
    loaded = CubeMindRouter.load(str(path))
    assert loaded.topic_count == 3
```

- [ ] **Step 2: Implement router.py**

Port from `empirical_grilly_next/cubemind/router.py`. Use BlockCodes for similarity.

- [ ] **Step 3: Run tests, commit**

```bash
uv run pytest tests/test_router.py -v
git commit -m "feat: CubeMindRouter — prototype similarity routing with save/load"
```

---

## Task 7: routing/moe_gate.py — DSelectK Gate

**Files:**
- Create: `cubemind/routing/moe_gate.py`
- Create: `tests/test_moe_gate.py`

- [ ] **Step 1: Write tests**

Test that DSelectK produces sparse k-hot selection, gradients flow, output sums to ~1.

- [ ] **Step 2: Implement moe_gate.py**

Port from `empirical_grilly_next/cubemind/moe_gate.py`. Clean implementation with proper docstring.

- [ ] **Step 3: Run tests, commit**

```bash
uv run pytest tests/test_moe_gate.py -v
git commit -m "feat: DSelectK MoE gate — sparse expert selection"
```

---

## Plan A Complete — Summary

After all 7 tasks:

| Component | What it does | Validated by |
|-----------|-------------|-------------|
| Repo scaffold | Clean project with grilly dependency | builds + imports |
| core.py | Constants + Hyperfan init | Theorem 3 test |
| ops/block_codes.py | GPU block-code bind/unbind/similarity | Theorems 1 + 7 |
| ops/hdc.py | HDC packed operations (32x compression) | Bind/similarity tests |
| perception/encoder.py | Text → block-code | Shape + similarity tests |
| routing/router.py | Prototype similarity routing | Route correctness + save/load |
| routing/moe_gate.py | DSelectK sparse gating | Sparsity + gradient tests |

**Next:** Plan B adds reasoning (HMM ensemble), execution (HYLA, CVL), and memory (cache, hippocampal).
