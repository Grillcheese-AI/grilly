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
    batch.submit();

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
    batch.submit();

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

}  // namespace ops
}  // namespace grilly
