# `_bridge` Migration Checklist (Deprecated `Compute()` Removal)

This checklist tracks the migration away from legacy `Compute()` callsites toward `grilly.backend._bridge` (and explicit CPU fallbacks where required).

## Phase E1 (Functional API) — Current Status

- [x] `functional/activations.py`
  - Replaced `Compute()` fallback in `relu`, `gelu`, `silu`, `softmax`.
  - Kept GPU-first path via `_bridge.*`; added explicit numpy CPU fallback.
- [x] `functional/linear.py`
  - Replaced `Compute().fnn.linear(...)` fallback with numpy matmul + bias fallback.
- [x] `functional/normalization.py`
  - Replaced `Compute().layernorm(...)` fallback with explicit numpy layer norm fallback.
- [x] `functional/attention.py`
  - Removed `_get_backend()` legacy path.
  - Routed through `_bridge.attention_scores`, `_bridge.attention_mask`, `_bridge.attention_output`, `_bridge.flash_attention2`.
  - Added explicit numpy attention fallback.

## Remaining Functional Callsites

- [x] `functional/dropout.py`
  - Removed `Compute()` path; now uses `_bridge.dropout` + explicit numpy fallback.
- [x] `functional/fft.py`
  - Removed backend factory path; explicit numpy FFT fallback retained.
- [x] `functional/loss.py`
  - Removed backend factory path; routes through `_bridge.cross_entropy_loss` when available.
- [x] `functional/embedding.py`
  - Replaced `Compute()` path with `_bridge.embedding_lookup` + CPU fallback.
- [x] `functional/memory.py`
  - Replaced `Compute()` backend construction with `_bridge` path + deterministic CPU fallback.
- [x] `functional/learning.py`
  - Replaced `Compute()` backend construction with `_bridge` path + deterministic CPU fallback.
- [x] `functional/faiss.py`
  - Replaced `Compute()` backend construction with `_bridge` path + CPU fallback.
- [x] `functional/cells.py`
  - Replaced `Compute()` backend construction with `_bridge` path + CPU fallback.
- [x] `functional/bridge.py`
  - Removed internal `Compute()` wrappers; uses `_bridge` helpers as canonical runtime path.

## Phase E2 (Core Runtime) — File-by-File Targets

- [ ] `nn/module.py`
  - Replace module-level backend initialization (`self._backend = Compute()`) with bridge-native runtime context.
  - Ensure lazy init and no implicit deprecated path on import.
- [ ] `utils/tensor_conversion.py`
  - Replace all `Compute()` instantiations in conversion/fallback flow with `_bridge`-native conversion helpers.
  - Keep CPU fallback explicit and observable (no silent legacy fallback).

## Phase E3 (Guardrails)

- [ ] Add CI lint/grep guard preventing new `from grilly import Compute` in runtime paths.
- [ ] Add targeted tests to assert functional modules do not instantiate deprecated API.
- [ ] Add migration notes in docs/changelog for users still depending on legacy behavior.

## JIT / Snippet Optimizations Added

- [x] `backend/jit.py`
  - Switched graph cache to true LRU behavior (`OrderedDict` + `move_to_end`).
  - Included scalar kwargs when tracing compiled variants to avoid stale-kwargs replay behavior.
- [x] `functional/embedding.py`
  - Vectorized sinusoidal positional encoding path (removed nested Python loops).
- [x] `functional/faiss.py`
  - Vectorized `faiss_kmeans_update` centroid accumulation (removed scalar loops).

## Completion Criteria

- Runtime code (`functional`, `nn`, `utils`) has no steady-state `Compute()` dependency.
- All migrated paths prefer `_bridge` and keep deterministic numpy CPU fallback.
- CI blocks reintroduction of deprecated API callsites.
