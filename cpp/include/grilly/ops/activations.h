#pragma once

#include <cstdint>

#include "grilly/buffer_pool.h"
#include "grilly/command_batch.h"
#include "grilly/pipeline_cache.h"

namespace grilly {
namespace ops {

/// Activation push constants — matches activation-*.glsl layout.
/// All activation shaders take a single uint: total_elements.
struct ActivationParams {
    uint32_t totalElements;
};

/// GPU activation functions. Each dispatches the corresponding SPIR-V shader
/// (activation-relu.spv, activation-gelu.spv, etc.) with 2 buffer bindings
/// (input, output) and 1D workgroups at 256 threads.
///
/// These are the simplest ops — good for OpGraph composition since they have
/// minimal dispatch overhead and benefit most from batched submission.

void relu(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
          const float* input, float* output, uint32_t totalElements);

void gelu(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
          const float* input, float* output, uint32_t totalElements);

void silu(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
          const float* input, float* output, uint32_t totalElements);

void tanh_act(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
              const float* input, float* output, uint32_t totalElements);

/// Activation backward passes. 3 buffer bindings (grad_output, input, grad_input).

void reluBackward(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
                  const float* gradOutput, const float* input,
                  float* gradInput, uint32_t totalElements);

void geluBackward(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
                  const float* gradOutput, const float* input,
                  float* gradInput, uint32_t totalElements);

void siluBackward(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
                  const float* gradOutput, const float* input,
                  float* gradInput, uint32_t totalElements);

}  // namespace ops
}  // namespace grilly
