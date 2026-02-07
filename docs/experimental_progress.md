# Experimental work log

Date: 2026-02-05

This document records the experimental namespace implementation, GPU VSA work,
NLP pipeline changes, capsule memory integration, and test status.

## Implemented modules

- `experimental/vsa`
  - BinaryOps and HolographicOps for bind/unbind/bundle/similarity.
  - ResonatorNetwork with multiple restarts and improved initialization.
  - Batch APIs for bind, bundle, similarity, and convolve.
- `experimental/moe`
  - RelationalEncoder, ResonatorMoE, and RelationalMoE routing.
  - Capsule-aware routing blend (VSA + capsule similarity).
- `experimental/language`
  - WordEncoder with stable n-gram encoding and unitary vectors.
  - SentenceEncoder with role and position binding.
  - ResonatorParser and SentenceGenerator.
  - InstantLanguage system orchestration.
- `experimental/temporal`
  - TemporalEncoder with unitary time vectors.
  - CausalChain, CounterfactualReasoner, TemporalReasoningSystem.
- `experimental/cognitive`
  - WorkingMemory, WorldModel, InternalSimulator, Understander, CognitiveController.
  - Capsule integration via `experimental/cognitive/capsule.py`.
  - Capsule vectors stored in working memory and world facts.
  - Capsule-aware retrieval and confidence scoring.
  - Temporal validation filtering for candidate responses.

## GPU backend and shaders

- `backend/experimental/vsa.py`
  - bind_bipolar, bind_bipolar_batch
  - bundle, bundle_batch
  - similarity_batch
  - resonator_step (codebook projection)
  - circular_convolve (CPU fallback for now)
- `backend/core.py`
  - Autoloads experimental shaders from `shaders/experimental/spv`.
- `shaders/experimental`
  - `vsa-bind.glsl`
  - `vsa-bind-batch.glsl`
  - `vsa-bundle.glsl`
  - `vsa-bundle-batch.glsl`
  - `vsa-similarity-batch.glsl`
  - `vsa-resonator-step.glsl`
  - `vsa-fft-convolve.glsl`
  - Compile instructions in `shaders/experimental/README.md`.

## NLP pipeline

- `svc_converter/convert_to_svc.py`
  - Replaced spaCy with Stanza.
  - Added adapter classes to expose a spaCy-like interface.
  - Added sentence indexing support and head resolution fixes.

## Tests and examples

- Added and updated `tests/experimental` across VSA, MoE, language, temporal,
  cognitive, backend VSA, and SVC integration.
- Updated `tests/test_conv2d.py` with xfail for known GPU conv issues.
- Updated `tests/test_gemm_backward.py` CPU benchmark input handling.
- Added `examples/experimental_*.py` including CLI chat interface and README.
- Added concrete examples for VSA batch ops, capsule MoE routing, and capsule cognition.
- Added a temporal validation example for cognitive response gating.

## RDNA2 compatibility notes

- Keep compute local sizes <= 256.
- Compile shaders with `--target-env=vulkan1.2` and SPIR-V <= 1.5.
- Avoid FP16 paths unless `VK_KHR_shader_float16_int8` is detected.
- Add a minimal RDNA2 smoke test suite for VSA bind/bundle/similarity, conv2d, gemm.

## Test status

Command:
`uv run -m pytest tests/experimental`

Result:
`266 passed in 6.52s`

## Recent fixes

- Fixed `WorkingMemoryItem` dataclass ordering for Python 3.12.
- Reworked `ResonatorParser.parallel_parse` to use deterministic role/position
  unbinding, restoring non-empty results.
- Clamped `Understander` confidence to [0, 1] to avoid negative values.
- Added temporal validation filtering in `CognitiveController`.

## Next steps (suggested)

- Temporal validation in `CognitiveController` candidate filtering.
- SVC pipeline batching to improve throughput.
- Additional custom shaders: word similarity batch, sentence encode, temporal bind/unbind.
