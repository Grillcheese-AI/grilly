#include "grilly/ops/activations.h"

#include <cstring>

namespace grilly {
namespace ops {

// ── Activation dispatch helper ────────────────────────────────────────────
//
// All forward activations share the same pattern:
//   - 2 buffers: input (binding 0), output (binding 1)
//   - 1 uint push constant: total_elements
//   - 1D workgroups: (total + 255) / 256
//
// This is the simplest GPU dispatch pattern in grilly. Each thread processes
// one element, applying the nonlinearity in-place. The workgroup size of 256
// is a good default for RDNA 2 (8 waves of 32 threads = 256).

static void activationForward(
    const std::string& shaderName,
    CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
    const float* input, float* output, uint32_t totalElements) {
    const size_t bytes = size_t(totalElements) * sizeof(float);

    GrillyBuffer bufIn  = pool.acquire(bytes);
    GrillyBuffer bufOut = pool.acquire(bytes);

    pool.upload(bufIn, input, bytes);

    PipelineEntry pipe = cache.getOrCreate(shaderName, 2, sizeof(uint32_t));

    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {bufIn.handle,  0, bytes},
        {bufOut.handle, 0, bytes},
    };
    VkDescriptorSet descSet = cache.allocDescriptorSet(shaderName, bufInfos);

    ActivationParams push{totalElements};
    uint32_t gx = (totalElements + 255) / 256;

    batch.begin();
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, 1, 1,
                   &push, sizeof(push));
    batch.submitDeferred();
    batch.waitForCompletion();

    pool.download(bufOut, output, bytes);

    pool.release(bufIn);
    pool.release(bufOut);
}

// ── Activation backward helper ────────────────────────────────────────────
//
// Backward passes have 3 buffers: grad_output, input (original), grad_input.
// Same push constant and dispatch pattern as forward.

static void activationBackward(
    const std::string& shaderName,
    CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
    const float* gradOutput, const float* input,
    float* gradInput, uint32_t totalElements) {
    const size_t bytes = size_t(totalElements) * sizeof(float);

    GrillyBuffer bufGradOut = pool.acquire(bytes);
    GrillyBuffer bufInput   = pool.acquire(bytes);
    GrillyBuffer bufGradIn  = pool.acquire(bytes);

    pool.upload(bufGradOut, gradOutput, bytes);
    pool.upload(bufInput, input, bytes);

    PipelineEntry pipe = cache.getOrCreate(shaderName, 3, sizeof(uint32_t));

    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {bufGradOut.handle, 0, bytes},
        {bufInput.handle,   0, bytes},
        {bufGradIn.handle,  0, bytes},
    };
    VkDescriptorSet descSet = cache.allocDescriptorSet(shaderName, bufInfos);

    ActivationParams push{totalElements};
    uint32_t gx = (totalElements + 255) / 256;

    batch.begin();
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, 1, 1,
                   &push, sizeof(push));
    batch.submitDeferred();
    batch.waitForCompletion();

    pool.download(bufGradIn, gradInput, bytes);

    pool.release(bufGradOut);
    pool.release(bufInput);
    pool.release(bufGradIn);
}

// ── Forward passes ────────────────────────────────────────────────────────

void relu(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
          const float* input, float* output, uint32_t totalElements) {
    activationForward("activation-relu", batch, pool, cache,
                      input, output, totalElements);
}

void gelu(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
          const float* input, float* output, uint32_t totalElements) {
    activationForward("activation-gelu", batch, pool, cache,
                      input, output, totalElements);
}

void silu(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
          const float* input, float* output, uint32_t totalElements) {
    activationForward("activation-silu", batch, pool, cache,
                      input, output, totalElements);
}

void tanh_act(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
              const float* input, float* output, uint32_t totalElements) {
    activationForward("activation-tanh", batch, pool, cache,
                      input, output, totalElements);
}

// ── Backward passes ──────────────────────────────────────────────────────

void reluBackward(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
                  const float* gradOutput, const float* input,
                  float* gradInput, uint32_t totalElements) {
    activationBackward("activation-relu-backward", batch, pool, cache,
                       gradOutput, input, gradInput, totalElements);
}

void geluBackward(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
                  const float* gradOutput, const float* input,
                  float* gradInput, uint32_t totalElements) {
    activationBackward("activation-gelu-backward", batch, pool, cache,
                       gradOutput, input, gradInput, totalElements);
}

void siluBackward(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
                  const float* gradOutput, const float* input,
                  float* gradInput, uint32_t totalElements) {
    activationBackward("activation-silu-backward", batch, pool, cache,
                       gradOutput, input, gradInput, totalElements);
}

// ── Tanh backward ────────────────────────────────────────────────────────
// Uses tanh_output (not raw input), so d/dx tanh = 1 - tanh^2.
// Shader: activation-tanh-backward — same 3-buffer pattern as other backwards.

void tanhBackward(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
                  const float* gradOutput, const float* tanhOutput,
                  float* gradInput, uint32_t totalElements) {
    activationBackward("activation-tanh-backward", batch, pool, cache,
                       gradOutput, tanhOutput, gradInput, totalElements);
}

// ── Softmax forward ──────────────────────────────────────────────────────
// 3-pass algorithm using the same shader with different pass_type:
//   Pass 0: find max per position (for numerical stability)
//   Pass 1: compute sum of exp(x - max)
//   Pass 2: normalize: output = exp(x - max) / sum_exp

void softmax(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
             const float* input, float* output,
             uint32_t batchSize, uint32_t seqLen, uint32_t features) {
    const uint32_t totalPositions = batchSize * seqLen;
    const uint32_t totalElements = totalPositions * features;
    const size_t dataBytes = size_t(totalElements) * sizeof(float);
    const size_t auxBytes  = size_t(totalPositions) * sizeof(float);

    GrillyBuffer bufInput  = pool.acquire(dataBytes);
    GrillyBuffer bufOutput = pool.acquire(dataBytes);
    GrillyBuffer bufMax    = pool.acquire(auxBytes);
    GrillyBuffer bufSumExp = pool.acquire(auxBytes);

    pool.upload(bufInput, input, dataBytes);

    PipelineEntry pipe = cache.getOrCreate("activation-softmax", 4,
                                           sizeof(SoftmaxParams));

    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {bufInput.handle,  0, dataBytes},
        {bufOutput.handle, 0, dataBytes},
        {bufMax.handle,    0, auxBytes},
        {bufSumExp.handle, 0, auxBytes},
    };
    VkDescriptorSet descSet = cache.allocDescriptorSet("activation-softmax",
                                                        bufInfos);

    uint32_t gx = (totalPositions + 255) / 256;

    batch.begin();

    // Pass 0: find max
    SoftmaxParams push0{batchSize, seqLen, features, 0, features};
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, 1, 1,
                   &push0, sizeof(push0));
    batch.barrier();

    // Pass 1: sum_exp
    SoftmaxParams push1{batchSize, seqLen, features, 1, features};
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, 1, 1,
                   &push1, sizeof(push1));
    batch.barrier();

    // Pass 2: normalize
    SoftmaxParams push2{batchSize, seqLen, features, 2, features};
    uint32_t gx2 = (totalElements + 255) / 256;
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx2, 1, 1,
                   &push2, sizeof(push2));

    batch.submitDeferred();
    batch.waitForCompletion();

    pool.download(bufOutput, output, dataBytes);

    pool.release(bufInput);
    pool.release(bufOutput);
    pool.release(bufMax);
    pool.release(bufSumExp);
}

// ── Softmax backward ─────────────────────────────────────────────────────

void softmaxBackward(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
                     const float* gradOutput, const float* softmaxOutput,
                     float* gradInput, uint32_t batchSize, uint32_t seqLen,
                     uint32_t numClasses) {
    const uint32_t total = batchSize * seqLen * numClasses;
    const size_t bytes = size_t(total) * sizeof(float);

    GrillyBuffer bufGradOut  = pool.acquire(bytes);
    GrillyBuffer bufSoftmax  = pool.acquire(bytes);
    GrillyBuffer bufGradIn   = pool.acquire(bytes);

    pool.upload(bufGradOut, gradOutput, bytes);
    pool.upload(bufSoftmax, softmaxOutput, bytes);

    PipelineEntry pipe = cache.getOrCreate("activation-softmax-backward", 3,
                                           sizeof(SoftmaxBackwardParams));

    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {bufGradOut.handle, 0, bytes},
        {bufSoftmax.handle, 0, bytes},
        {bufGradIn.handle,  0, bytes},
    };
    VkDescriptorSet descSet = cache.allocDescriptorSet(
        "activation-softmax-backward", bufInfos);

    SoftmaxBackwardParams push{batchSize, seqLen, numClasses};
    uint32_t gx = (batchSize * seqLen + 255) / 256;

    batch.begin();
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, 1, 1,
                   &push, sizeof(push));
    batch.submitDeferred();
    batch.waitForCompletion();

    pool.download(bufGradIn, gradInput, bytes);

    pool.release(bufGradOut);
    pool.release(bufSoftmax);
    pool.release(bufGradIn);
}

}  // namespace ops
}  // namespace grilly
