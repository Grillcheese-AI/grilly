/// fused.cpp — Fused transformer ops (MLP-GELU, LayerNorm-Linear).
///
/// These ops keep intermediate activations in GPU shared memory (LDS),
/// eliminating VRAM round-trips between sequential operations.

#include "grilly/ops/fused.h"
#include <cstring>

namespace grilly {
namespace ops {

// ── Fused MLP: fc1 → GELU → fc2 ────────────────────────────────────────
//
// Shader: fused-mlp-gelu.spv
// 6 bindings: input, w1, b1, w2, b2, output
// Specialization constants: D_IN, D_HIDDEN, D_OUT
// Dispatch: (seq_len, 1, 1) — one workgroup per token

void fusedMlpGelu(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
                  const float* input, const float* w1, const float* b1,
                  const float* w2, const float* b2, float* output,
                  uint32_t seqLen, uint32_t dIn, uint32_t dHidden, uint32_t dOut) {

    const size_t inputBytes  = size_t(seqLen) * dIn * sizeof(float);
    const size_t w1Bytes     = size_t(dHidden) * dIn * sizeof(float);
    const size_t b1Bytes     = size_t(dHidden) * sizeof(float);
    const size_t w2Bytes     = size_t(dOut) * dHidden * sizeof(float);
    const size_t b2Bytes     = size_t(dOut) * sizeof(float);
    const size_t outputBytes = size_t(seqLen) * dOut * sizeof(float);

    GrillyBuffer bufInput  = pool.acquire(inputBytes);
    GrillyBuffer bufW1     = pool.acquire(w1Bytes);
    GrillyBuffer bufB1     = pool.acquire(b1Bytes);
    GrillyBuffer bufW2     = pool.acquire(w2Bytes);
    GrillyBuffer bufB2     = pool.acquire(b2Bytes);
    GrillyBuffer bufOutput = pool.acquire(outputBytes);

    pool.upload(bufInput, input, inputBytes);
    pool.upload(bufW1, w1, w1Bytes);
    pool.upload(bufB1, b1, b1Bytes);
    pool.upload(bufW2, w2, w2Bytes);
    pool.upload(bufB2, b2, b2Bytes);

    // Note: specialization constants (D_IN, D_HIDDEN, D_OUT) are baked into
    // the SPIR-V at compile time with defaults matching SigLIP2 (768, 3072, 768).
    // For different dimensions, recompile the shader with new constants.
    PipelineEntry pipe = cache.getOrCreate("fused-mlp-gelu", 6, 0);

    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {bufInput.handle,  0, inputBytes},
        {bufW1.handle,     0, w1Bytes},
        {bufB1.handle,     0, b1Bytes},
        {bufW2.handle,     0, w2Bytes},
        {bufB2.handle,     0, b2Bytes},
        {bufOutput.handle, 0, outputBytes},
    };
    VkDescriptorSet descSet = cache.allocDescriptorSet("fused-mlp-gelu", bufInfos);

    batch.begin();
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, seqLen, 1, 1,
                   nullptr, 0);
    batch.submit();

    pool.download(bufOutput, output, outputBytes);

    pool.release(bufInput);
    pool.release(bufW1);
    pool.release(bufB1);
    pool.release(bufW2);
    pool.release(bufB2);
    pool.release(bufOutput);
}

// ── Fused LayerNorm + Linear ────────────────────────────────────────────
//
// Shader: fused-layernorm-linear.spv
// 6 bindings: input, ln_weight, ln_bias, proj_w, proj_b, output
// Specialization constants: D_IN, D_OUT
// Dispatch: (seq_len, 1, 1)

void fusedLayernormLinear(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
                          const float* input, const float* lnWeight, const float* lnBias,
                          const float* projW, const float* projB, float* output,
                          uint32_t seqLen, uint32_t dIn, uint32_t dOut) {

    const size_t inputBytes  = size_t(seqLen) * dIn * sizeof(float);
    const size_t lnWBytes    = size_t(dIn) * sizeof(float);
    const size_t lnBBytes    = size_t(dIn) * sizeof(float);
    const size_t projWBytes  = size_t(dOut) * dIn * sizeof(float);
    const size_t projBBytes  = size_t(dOut) * sizeof(float);
    const size_t outputBytes = size_t(seqLen) * dOut * sizeof(float);

    GrillyBuffer bufInput   = pool.acquire(inputBytes);
    GrillyBuffer bufLNW     = pool.acquire(lnWBytes);
    GrillyBuffer bufLNB     = pool.acquire(lnBBytes);
    GrillyBuffer bufProjW   = pool.acquire(projWBytes);
    GrillyBuffer bufProjB   = pool.acquire(projBBytes);
    GrillyBuffer bufOutput  = pool.acquire(outputBytes);

    pool.upload(bufInput, input, inputBytes);
    pool.upload(bufLNW, lnWeight, lnWBytes);
    pool.upload(bufLNB, lnBias, lnBBytes);
    pool.upload(bufProjW, projW, projWBytes);
    pool.upload(bufProjB, projB, projBBytes);

    // Specialization constants baked at compile time (defaults: 768, 768).
    PipelineEntry pipe = cache.getOrCreate("fused-layernorm-linear", 6, 0);

    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {bufInput.handle,  0, inputBytes},
        {bufLNW.handle,    0, lnWBytes},
        {bufLNB.handle,    0, lnBBytes},
        {bufProjW.handle,  0, projWBytes},
        {bufProjB.handle,  0, projBBytes},
        {bufOutput.handle, 0, outputBytes},
    };
    VkDescriptorSet descSet = cache.allocDescriptorSet("fused-layernorm-linear", bufInfos);

    batch.begin();
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, seqLen, 1, 1,
                   nullptr, 0);
    batch.submit();

    pool.download(bufOutput, output, outputBytes);

    pool.release(bufInput);
    pool.release(bufLNW);
    pool.release(bufLNB);
    pool.release(bufProjW);
    pool.release(bufProjB);
    pool.release(bufOutput);
}

}  // namespace ops
}  // namespace grilly
