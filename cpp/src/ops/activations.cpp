#include "grilly/ops/activations.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

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

    // Staging pattern (see linear.cpp for the long-form rationale):
    // compute on DEVICE_LOCAL VRAM, stage-in via WC sequential-write
    // memory, stage-out via HOST_CACHED random-read memory. Without this
    // a 19 MB ReLU readback ran at 25 MB/s (~750 ms); with it the same
    // op runs in single-digit ms.
    GrillyBuffer bufInDL    = pool.acquireDeviceLocal(bytes);
    GrillyBuffer bufOutDL   = pool.acquireDeviceLocal(bytes);
    GrillyBuffer bufInStage = pool.acquire(bytes);
    GrillyBuffer bufOutStage = pool.acquireReadback(bytes);

    pool.upload(bufInStage, input, bytes);

    PipelineEntry pipe = cache.getOrCreate(shaderName, 2, sizeof(uint32_t));

    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {bufInDL.handle,  0, bytes},
        {bufOutDL.handle, 0, bytes},
    };
    VkDescriptorSet descSet = cache.allocDescriptorSet(shaderName, bufInfos);

    ActivationParams push{totalElements};
    uint32_t gx = (totalElements + 255) / 256;

    batch.begin();
    batch.copyBuffer(bufInStage, bufInDL, bytes);
    batch.transferComputeBarrier();
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, 1, 1,
                   &push, sizeof(push));
    batch.transferComputeBarrier();
    batch.copyBuffer(bufOutDL, bufOutStage, bytes);
    batch.submitDeferred();
    batch.waitForCompletion();

    pool.download(bufOutStage, output, bytes);

    pool.release(bufInDL);
    pool.release(bufOutDL);
    pool.release(bufInStage);
    pool.release(bufOutStage);
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

    // Staging pattern: 2 stage-in (gradOutput, input), 1 stage-out (gradInput)
    GrillyBuffer bufGradOutDL = pool.acquireDeviceLocal(bytes);
    GrillyBuffer bufInputDL   = pool.acquireDeviceLocal(bytes);
    GrillyBuffer bufGradInDL  = pool.acquireDeviceLocal(bytes);

    GrillyBuffer bufGradOutStage = pool.acquire(bytes);
    GrillyBuffer bufInputStage   = pool.acquire(bytes);
    GrillyBuffer bufGradInStage  = pool.acquireReadback(bytes);

    pool.upload(bufGradOutStage, gradOutput, bytes);
    pool.upload(bufInputStage, input, bytes);

    PipelineEntry pipe = cache.getOrCreate(shaderName, 3, sizeof(uint32_t));

    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {bufGradOutDL.handle, 0, bytes},
        {bufInputDL.handle,   0, bytes},
        {bufGradInDL.handle,  0, bytes},
    };
    VkDescriptorSet descSet = cache.allocDescriptorSet(shaderName, bufInfos);

    ActivationParams push{totalElements};
    uint32_t gx = (totalElements + 255) / 256;

    batch.begin();
    batch.copyBuffer(bufGradOutStage, bufGradOutDL, bytes);
    batch.copyBuffer(bufInputStage,   bufInputDL,   bytes);
    batch.transferComputeBarrier();
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, 1, 1,
                   &push, sizeof(push));
    batch.transferComputeBarrier();
    batch.copyBuffer(bufGradInDL, bufGradInStage, bytes);
    batch.submitDeferred();
    batch.waitForCompletion();

    pool.download(bufGradInStage, gradInput, bytes);

    pool.release(bufGradOutDL);
    pool.release(bufInputDL);
    pool.release(bufGradInDL);
    pool.release(bufGradOutStage);
    pool.release(bufInputStage);
    pool.release(bufGradInStage);
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

    // Compute buffers (DEVICE_LOCAL, cached VRAM). bufMax/bufSumExp are
    // GPU-only scratch (per-pass aux), so no host staging needed.
    GrillyBuffer bufInput  = pool.acquireDeviceLocal(dataBytes);
    GrillyBuffer bufOutput = pool.acquireDeviceLocal(dataBytes);
    GrillyBuffer bufMax    = pool.acquireDeviceLocal(auxBytes);
    GrillyBuffer bufSumExp = pool.acquireDeviceLocal(auxBytes);
    GrillyBuffer stgInput  = pool.acquire(dataBytes);          // CPU write (WC ok)
    GrillyBuffer stgOutput = pool.acquireReadback(dataBytes);  // CPU read (HOST_CACHED)

    pool.upload(stgInput, input, dataBytes);

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

    uint32_t gx  = (totalPositions + 255) / 256;
    uint32_t gx2 = (totalElements  + 255) / 256;
    SoftmaxParams push0{batchSize, seqLen, features, 0, features};  // max
    SoftmaxParams push1{batchSize, seqLen, features, 1, features};  // sum_exp
    SoftmaxParams push2{batchSize, seqLen, features, 2, features};  // normalize

    batch.begin();
    batch.copyBuffer(stgInput, bufInput, dataBytes);
    batch.transferComputeBarrier();
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, 1, 1,
                   &push0, sizeof(push0));
    batch.barrier();
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, 1, 1,
                   &push1, sizeof(push1));
    batch.barrier();
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx2, 1, 1,
                   &push2, sizeof(push2));
    batch.transferComputeBarrier();
    batch.copyBuffer(bufOutput, stgOutput, dataBytes);
    batch.submitDeferred();
    batch.waitForCompletion();

    pool.download(stgOutput, output, dataBytes);

    pool.release(bufInput);
    pool.release(bufOutput);
    pool.release(bufMax);
    pool.release(bufSumExp);
    pool.release(stgInput);
    pool.release(stgOutput);
}

// ── Softmax backward ─────────────────────────────────────────────────────

void softmaxBackward(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
                     const float* gradOutput, const float* softmaxOutput,
                     float* gradInput, uint32_t batchSize, uint32_t seqLen,
                     uint32_t numClasses) {
    const uint32_t total = batchSize * seqLen * numClasses;
    const size_t bytes = size_t(total) * sizeof(float);

    GrillyBuffer bufGradOut  = pool.acquireDeviceLocal(bytes);
    GrillyBuffer bufSoftmax  = pool.acquireDeviceLocal(bytes);
    GrillyBuffer bufGradIn   = pool.acquireDeviceLocal(bytes);
    GrillyBuffer stgGradOut  = pool.acquire(bytes);
    GrillyBuffer stgSoftmax  = pool.acquire(bytes);
    GrillyBuffer stgGradIn   = pool.acquireReadback(bytes);

    pool.upload(stgGradOut, gradOutput, bytes);
    pool.upload(stgSoftmax, softmaxOutput, bytes);

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
    batch.copyBuffer(stgGradOut, bufGradOut, bytes);
    batch.copyBuffer(stgSoftmax, bufSoftmax, bytes);
    batch.transferComputeBarrier();
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, 1, 1,
                   &push, sizeof(push));
    batch.transferComputeBarrier();
    batch.copyBuffer(bufGradIn, stgGradIn, bytes);
    batch.submitDeferred();
    batch.waitForCompletion();

    pool.download(stgGradIn, gradInput, bytes);

    pool.release(bufGradOut);
    pool.release(bufSoftmax);
    pool.release(bufGradIn);
    pool.release(stgGradOut);
    pool.release(stgSoftmax);
    pool.release(stgGradIn);
}

// ── Multiplication-free softmax (mf-softmax.glsl) ─────────────────────────
// Same 3-pass buffer layout as softmax; pass_type 0/1/2 with relu sums.

void mfSoftmax(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
               const float* input, float* output, uint32_t batchSize, uint32_t seqLen,
               uint32_t features) {
    const uint32_t totalPositions = batchSize * seqLen;
    const uint32_t totalElements = totalPositions * features;
    const size_t dataBytes = size_t(totalElements) * sizeof(float);
    const size_t auxBytes = size_t(totalPositions) * sizeof(float);

    GrillyBuffer bufInput = pool.acquireDeviceLocal(dataBytes);
    GrillyBuffer bufOutput = pool.acquireDeviceLocal(dataBytes);
    GrillyBuffer bufMax = pool.acquireDeviceLocal(auxBytes);
    GrillyBuffer bufSumPos = pool.acquireDeviceLocal(auxBytes);
    GrillyBuffer stgInput = pool.acquire(dataBytes);
    GrillyBuffer stgOutput = pool.acquireReadback(dataBytes);

    pool.upload(stgInput, input, dataBytes);

    PipelineEntry pipe =
        cache.getOrCreate("mf-softmax", 4, sizeof(SoftmaxParams));

    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {bufInput.handle, 0, dataBytes},
        {bufOutput.handle, 0, dataBytes},
        {bufMax.handle, 0, auxBytes},
        {bufSumPos.handle, 0, auxBytes},
    };
    VkDescriptorSet descSet = cache.allocDescriptorSet("mf-softmax", bufInfos);

    uint32_t gx = (totalPositions + 255) / 256;
    uint32_t gx2 = (totalElements + 255) / 256;
    SoftmaxParams push0{batchSize, seqLen, features, 0, features};
    SoftmaxParams push1{batchSize, seqLen, features, 1, features};
    SoftmaxParams push2{batchSize, seqLen, features, 2, features};

    batch.begin();
    batch.copyBuffer(stgInput, bufInput, dataBytes);
    batch.transferComputeBarrier();
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, 1, 1, &push0,
                   sizeof(push0));
    batch.barrier();
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, 1, 1, &push1,
                   sizeof(push1));
    batch.barrier();
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx2, 1, 1, &push2,
                   sizeof(push2));
    batch.transferComputeBarrier();
    batch.copyBuffer(bufOutput, stgOutput, dataBytes);
    batch.submitDeferred();
    batch.waitForCompletion();

    pool.download(stgOutput, output, dataBytes);

    pool.release(bufInput);
    pool.release(bufOutput);
    pool.release(bufMax);
    pool.release(bufSumPos);
    pool.release(stgInput);
    pool.release(stgOutput);
}

// Push layout must match mf-softplus.glsl: uint total_elements; float c;
struct MfSoftplusParams {
    uint32_t totalElements;
    float c;
};

void mfSoftplus(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
                const float* input, float* output, uint32_t totalElements,
                float beta) {
    if (beta <= 0.f) {
        throw std::invalid_argument("mfSoftplus: beta must be positive");
    }
    const float c = 4.f / (beta * beta);
    const size_t bytes = size_t(totalElements) * sizeof(float);

    GrillyBuffer bufIn = pool.acquireDeviceLocal(bytes);
    GrillyBuffer bufOut = pool.acquireDeviceLocal(bytes);
    GrillyBuffer stgIn = pool.acquire(bytes);
    GrillyBuffer stgOut = pool.acquireReadback(bytes);

    pool.upload(stgIn, input, bytes);

    PipelineEntry pipe =
        cache.getOrCreate("mf-softplus", 2, sizeof(MfSoftplusParams));

    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {bufIn.handle, 0, bytes},
        {bufOut.handle, 0, bytes},
    };
    VkDescriptorSet descSet = cache.allocDescriptorSet("mf-softplus", bufInfos);

    MfSoftplusParams push{totalElements, c};
    uint32_t gx = (totalElements + 255) / 256;

    batch.begin();
    batch.copyBuffer(stgIn, bufIn, bytes);
    batch.transferComputeBarrier();
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, 1, 1, &push,
                   sizeof(push));
    batch.transferComputeBarrier();
    batch.copyBuffer(bufOut, stgOut, bytes);
    batch.submitDeferred();
    batch.waitForCompletion();

    pool.download(stgOut, output, bytes);

    pool.release(bufIn);
    pool.release(bufOut);
    pool.release(stgIn);
    pool.release(stgOut);
}

void mfSigmoid(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
               const float* input, float* output, uint32_t totalElements) {
    activationForward("mf-sigmoid", batch, pool, cache, input, output,
                      totalElements);
}

}  // namespace ops
}  // namespace grilly
