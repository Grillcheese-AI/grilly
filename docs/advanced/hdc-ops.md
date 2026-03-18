# HDC & Block Codes

Hyperdimensional Computing (HDC) operations with packed binary representations, block-code circular convolution, and Sanger GHA for neurogenesis.

---

## Packed Binary HDC

HDC represents data as high-dimensional binary vectors (hypervectors). Grilly packs 32 bits per uint32, achieving 32x memory compression:

```python
# 10,240-dimensional binary vector = 320 uint32 values
# vs 10,240 float32 values (40 KB -> 1.25 KB)
```

Operations on packed vectors (XOR, popcount, similarity) run in the packed domain without unpacking.

---

## Block Codes

Block codes split a hypervector into blocks and apply permutations within blocks. This enables structured composition while preserving the algebraic properties of HDC:

- **Binding** (XOR): Combines two hypervectors into a new one
- **Bundling** (majority vote): Creates a prototype from multiple vectors
- **Circular convolution**: Sequence encoding via block-wise rotation

```python
# Block-code circular convolution shader
# Encodes sequences by rotating blocks of the hypervector
```

The block-code approach is used by CubeMind for neuro-vector-symbolic reasoning.

---

## Sanger GHA (Generalized Hebbian Algorithm)

Sanger's rule extracts principal components from streaming data without storing the full covariance matrix:

```python
# Used for neurogenesis — growing new neurons when
# existing representations are insufficient
```

The Sanger GHA shader computes incremental PCA updates, enabling dynamic network growth.

---

## DisARM Gradient Estimator

Training through discrete operations (block-code sign/argmax) requires gradient estimation since these operations are non-differentiable. Grilly implements the DisARM estimator:

```python
from grilly.nn.disarm import disarm_gradient

grad = disarm_gradient(
    logits,            # Pre-sigmoid logits
    loss_fn,           # Callable: discrete samples -> scalar loss
    num_samples=1,     # Antithetic pairs
    temperature=1.0,
)
```

DisARM uses antithetic coupling (paired samples `u` and `1-u`) to cancel common noise, achieving approximately 7.5x lower variance than REINFORCE.

---

## Shaders

| Shader | Description |
|--------|-------------|
| `hdc-block-code-bind.glsl` | XOR binding of packed hypervectors |
| `hdc-block-code-bundle.glsl` | Majority-vote bundling |
| `hdc-circular-conv.glsl` | Block-code circular convolution |
| `hdc-popcount.glsl` | Hamming distance via popcount |
| `sanger-gha.glsl` | Sanger GHA for incremental PCA |
| `disarm-*.glsl` | DisARM gradient estimation shaders |
