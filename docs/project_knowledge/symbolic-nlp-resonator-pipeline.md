# Symbolic NLP via VSA — Resonator Network Decoding Pipeline

**Date:** 2026-03-19
**Status:** Architecture design — TextEncoder and ResonatorNetwork exist in grilly C++ codebase

## Overview

Instead of autoregressive Transformer token prediction, compress an entire sentence into a single 1280-byte hypervector (10240 bits bipolar). GPU-accelerated Resonator Network decomposes it back into discrete words in ~29 microseconds via Vulkan Hamming search.

## Pipeline

```
Raw Text → Tokenize → SemanticAssigner (FastText 300D → LSH → 10240D bipolar)
  → TextEncoder: Word ⊗ Role ⊗ Position per token → Bundle (majority vote)
  → BitpackedVec (320 uint32 words = 1280 bytes)
  → ResonatorNetwork: Unbind(role, position) → Hamming search vs codebook
  → Decoded words with confidence scores
```

## Step 1: Semantic Projection (SemanticAssigner)

- FastText 300D vectors → Locality Sensitive Hashing → 10240D bipolar space
- Random Gaussian projection preserves cosine similarity as Hamming distance
- Similar words get similar bitpacked hypervectors
- File: `grilly/cubemind/semantic_assigner.h`

```cpp
SemanticAssigner assigner(10240, 300);
// Lazy-loads LSH projections into bitpacked cache
```

## Step 2: Sentence Encoding (TextEncoder)

Three-way binding per token: Word ⊗ Role ⊗ Position
Then bundle all tokens via majority vote (superposition) into bitpacked uint32 blocks.

```cpp
TextEncoder encoder(10240, 300);
std::vector<std::string> tokens = {"the", "quick", "brown", "fox"};
std::vector<std::string> roles = {"det", "amod", "amod", "nsubj"};
std::vector<uint32_t> positions = {0, 1, 2, 3};

BitpackedVec sentence_vector = encoder.encode_sentence(tokens, roles, positions);
// sentence_vector: 320 uint32 words (1280 bytes) — entire sentence compressed
```

## Step 3: Resonator Network Decoding

To query "What word is at position 3 acting as nsubj?":
1. Unbind structural keys (role, position) from sentence vector — XOR is self-inverse
2. Result: noisy version of the word vector
3. Vulkan shader (`resonator-bitpacked.glsl`) blasts noisy query against entire vocab codebook
4. Hardware Hamming distance matching finds closest semantic word in ~29 microseconds

```cpp
ResonatorNetwork resonator(pool, batch, pipeCache, 10240);
// Load vocab codebook into VRAM (persistent)

auto decoded = resonator.generate_sentence(sentence_vector, roles, positions, /*explain_away=*/true);
// Returns: vector of (word, confidence) pairs
```

### Explaining Away Accumulator
After finding each word, subtract its analog magnitude from the bundle to prevent "echoing" — stops decoded words from interfering with subsequent queries. This is the `explain_away=true` flag.

## Why This Architecture Works

1. **Zero-Shot Composition** — No neural network needed to combine words. Hyperdimensional algebra handles composition via bind/bundle.
2. **Explainability** — If resonator decodes "fox" incorrectly, trace exact Hamming distances of superimposed components to find why.
3. **Vulkan Speed** — `dispatch_resonator` maps codebook to VRAM, 256 threads per workgroup. Bypasses CPU-bound NLP bottlenecks.
4. **Compression** — Entire sentence in 1280 bytes vs thousands of float32 embedding dimensions.

## Key Parameters

- VSA dimension: 10240 bits (bipolar ±1, bitpacked to 320 uint32)
- FastText input: 300D float vectors
- LSH projection: Gaussian random matrix (300 → 10240)
- Resonator: Vulkan Hamming search, 256 threads/workgroup
- Decoding time: ~29 microseconds per word query

## Relationship to I-RAVEN Pipeline

Same VSA algebra (bind/unbind/bundle) but:
- I-RAVEN uses **continuous block codes** (probability simplexes, circular convolution)
- NLP uses **bipolar bitpacked codes** (±1, XOR binding, Hamming distance)
- Both use the same mathematical framework, different numerical representations
- The continuous version is differentiable (for end-to-end CNN training)
- The bipolar version is faster (bitwise ops, hardware popcount)

## Grilly C++ Files

- `grilly/cubemind/semantic_assigner.h` — LSH projection from embeddings to VSA
- `grilly/cubemind/text_encoder.h` — Sentence encoding via three-way binding
- `grilly/cubemind/resonator.h` — Resonator network with GPU Hamming search
- `shaders/resonator-bitpacked.glsl` — Vulkan compute shader for codebook search
