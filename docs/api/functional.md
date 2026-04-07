# API Reference: functional

`grilly.functional` provides stateless operations that mirror `torch.nn.functional`.

Source: [`functional/`](https://github.com/grillcheese-ai/grilly/tree/main/functional)

---

## Activations

| Function | Signature | Description |
|----------|-----------|-------------|
| `relu` | `relu(x)` | Rectified linear unit. |
| `gelu` | `gelu(x)` | Gaussian error linear unit. |
| `silu` | `silu(x)` | Sigmoid linear unit (Swish). |
| `softmax` | `softmax(x, dim=-1)` | Softmax normalization. |
| `softplus` | `softplus(x)` | Smooth ReLU approximation. |

---

## Linear

| Function | Signature | Description |
|----------|-----------|-------------|
| `linear` | `linear(x, weight, bias=None)` | Affine transformation: `x @ weight.T + bias`. |

---

## Attention

| Function | Signature | Description |
|----------|-----------|-------------|
| `attention` | `attention(q, k, v, mask=None)` | Standard scaled dot-product attention. |
| `flash_attention2` | `flash_attention2(q, k, v)` | Memory-efficient Flash Attention 2. |

---

## Normalization

| Function | Signature | Description |
|----------|-----------|-------------|
| `layer_norm` | `layer_norm(x, weight, bias, eps=1e-5)` | Layer normalization. |

---

## Dropout

| Function | Signature | Description |
|----------|-----------|-------------|
| `dropout` | `dropout(x, p=0.5, training=True)` | Randomly zero elements. |

---

## Loss Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `cross_entropy` | `cross_entropy(logits, targets)` | Cross-entropy loss. |
| `binary_cross_entropy` | `binary_cross_entropy(predictions, targets)` | Binary cross-entropy. |

---

## FFT

| Function | Signature | Description |
|----------|-----------|-------------|
| `fft` | `fft(x)` | Fast Fourier Transform. |
| `ifft` | `ifft(x)` | Inverse FFT. |
| `fft_magnitude` | `fft_magnitude(x)` | Magnitude of FFT output. |
| `fft_power_spectrum` | `fft_power_spectrum(x)` | Power spectrum from FFT. |

---

## Memory Operations

| Function | Signature | Description |
|----------|-----------|-------------|
| `memory_read` | `memory_read(memory, query)` | Read from memory via attention. |
| `memory_write` | `memory_write(memory, key, value)` | Write to memory. |
| `memory_context_aggregate` | `memory_context_aggregate(contexts)` | Aggregate memory contexts. |
| `memory_query_pooling` | `memory_query_pooling(queries)` | Pool memory queries. |
| `memory_inject_concat` | `memory_inject_concat(x, memory)` | Inject memory via concatenation. |
| `memory_inject_gate` | `memory_inject_gate(x, memory)` | Inject memory via gating. |
| `memory_inject_residual` | `memory_inject_residual(x, memory)` | Inject memory via residual. |

---

## Cell Operations

| Function | Signature | Description |
|----------|-----------|-------------|
| `place_cell` | `place_cell(position, centers)` | Place cell activation (spatial). |
| `time_cell` | `time_cell(time, centers)` | Time cell activation (temporal). |
| `theta_gamma_encoding` | `theta_gamma_encoding(phase, frequency)` | Theta-gamma neural encoding. |

---

## Learning Operations

| Function | Signature | Description |
|----------|-----------|-------------|
| `fisher_info` | `fisher_info(grads)` | Compute Fisher information matrix. |
| `ewc_penalty` | `ewc_penalty(params, fisher, old_params)` | Elastic Weight Consolidation penalty. |
| `natural_gradient` | `natural_gradient(grad, fisher)` | Natural gradient via Fisher. |
| `nlms_predict` | `nlms_predict(weights, x)` | NLMS prediction. |
| `nlms_update` | `nlms_update(weights, x, error, mu)` | NLMS weight update. |
| `whitening_transform` | `whitening_transform(x)` | Compute whitening transform. |
| `whitening_apply` | `whitening_apply(x, transform)` | Apply whitening. |

---

## Bridge Operations

| Function | Signature | Description |
|----------|-----------|-------------|
| `continuous_to_spikes` | `continuous_to_spikes(x, threshold)` | Convert continuous signal to spikes. |
| `spikes_to_continuous` | `spikes_to_continuous(spikes, decay)` | Decode spikes to continuous signal. |
| `bridge_temporal_weights` | `bridge_temporal_weights(t, decay)` | Temporal weighting for bridge. |

---

## Embedding Operations

| Function | Signature | Description |
|----------|-----------|-------------|
| `embedding_lookup` | `embedding_lookup(table, indices)` | Lookup embeddings by index. |
| `embedding_normalize` | `embedding_normalize(embeddings)` | L2-normalize embeddings. |
| `embedding_position` | `embedding_position(embeddings, positions)` | Add positional embeddings. |
| `embedding_pool` | `embedding_pool(embeddings, method)` | Pool embeddings. |
| `embedding_ffn` | `embedding_ffn(embeddings, weights)` | FFN on embeddings. |
| `embedding_attention` | `embedding_attention(q, k, v)` | Attention over embeddings. |

---

## FAISS Operations

| Function | Signature | Description |
|----------|-----------|-------------|
| `faiss_distance` | `faiss_distance(queries, keys)` | Compute distances. |
| `faiss_topk` | `faiss_topk(distances, k)` | Top-k selection. |
| `faiss_ivf_filter` | `faiss_ivf_filter(queries, centroids)` | IVF coarse filtering. |
| `faiss_kmeans_update` | `faiss_kmeans_update(centroids, data)` | K-means centroid update. |
| `faiss_quantize` | `faiss_quantize(data, codebook)` | Product quantization. |

---

## SNN Operations

| Function | Signature | Description |
|----------|-----------|-------------|
| `lif_step` | `lif_step(x, v, tau, v_threshold)` | Single LIF neuron step. Returns `(spike, v_new)`. |
| `if_step` | `if_step(x, v, v_threshold)` | Single IF neuron step. |
| `multi_step_forward` | `multi_step_forward(module, x, T)` | Run module over T time steps. |
| `seq_to_ann_forward` | `seq_to_ann_forward(module, x)` | Apply ANN module to temporal data. |
| `reset_net` | `reset_net(module)` | Reset all neuron membrane potentials. |
| `set_step_mode` | `set_step_mode(module, mode)` | Set single-step or multi-step mode. |

---

## PyTorch parity notes

- **Layout**: `linear(x, weight, bias)` uses `weight` of shape `(out_features, in_features)`, matching `torch.nn.functional.linear` and `nn.Linear.weight`.
- **Dtypes**: GPU paths expect float32 inputs unless documented otherwise.
- **Differences**: Not every `torch.nn.functional` symbol exists; Grilly adds domain-specific ops (SNN, memory, FAISS-like). See `docs/PYTORCH_PARITY_STATUS.md`.
- **Migration**: Step-by-step porting guidance lives in `docs/MIGRATION_PYTORCH.md`.
- **Tests**: Numerical checks vs numpy (and optional PyTorch) live under `tests/parity/`.
