# TAPPA & Adaptive KV Cache

TAPPA (Token-Aware Predictability-based Pruning for Attention) uses q-similarity to classify attention heads and allocate KV cache budgets adaptively.

---

## Q-Similarity

Q-similarity measures how predictable an attention head is by comparing consecutive query vectors:

```python
from grilly.backend._bridge import q_similarity

# queries: (batch, num_heads, seq_len, head_dim)
qsim = q_similarity(queries)  # (batch, num_heads)
```

- **High q-similarity** (close to 1.0): The head's queries are similar across positions. This is a "streaming" or "predictable" head -- it doesn't need many KV cache tokens.
- **Low q-similarity** (close to 0.0): The head's queries vary significantly. This is a "retrieval" or "unpredictable" head -- it needs more KV cache tokens to find relevant keys.

Dispatched via `attention-q-similarity.glsl`.

---

## Adaptive KV Cache Budgets

```python
from grilly.backend.adaptive_kv import compute_head_budgets, classify_heads

# Compute per-head token budgets
budgets = compute_head_budgets(
    queries,                   # (batch, num_heads, seq_len, head_dim)
    total_budget=2048,         # Total tokens across all heads
    min_budget_per_head=32,    # Floor per head
)
# budgets: (num_heads,) — token budget per head
```

Heads with low q-similarity (unpredictable) get more budget. Heads with high q-similarity (predictable) get less. The total budget is conserved.

---

## Head Classification

```python
head_types = classify_heads(queries, threshold=0.5)
# Returns per-head labels: "streaming", "retrieval", or "mixed"
```

This classification drives:

- **KV cache eviction**: Retrieval heads keep more tokens; streaming heads evict aggressively.
- **Selective momentum**: Only unpredictable heads get SympFormer acceleration (see [SympFormer](sympformer.md)).
- **Compute allocation**: Retrieval heads may use full attention; streaming heads can use linear approximation.

---

## Integration with SympFormer

TAPPA and SympFormer compose naturally:

1. TAPPA identifies which heads are unpredictable
2. SympFormer accelerates convergence for those heads
3. Predictable heads skip momentum overhead

```python
from grilly.nn.selective_momentum import SelectiveMomentumAttention

attn = SelectiveMomentumAttention(
    attention=nn.MultiheadAttention(512, 8),
    embed_dim=512,
    num_heads=8,
    threshold=0.5,
)
```

---

## Shader

The q-similarity computation runs on `attention-q-similarity.glsl`, which computes cosine similarity between consecutive query vectors for each head.
