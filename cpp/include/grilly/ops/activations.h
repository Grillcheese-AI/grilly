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

/// Tanh backward: uses tanh_output (not raw input) — d/dx tanh = 1 - tanh^2.
/// 3 buffer bindings: grad_output, tanh_output, grad_input.
void tanhBackward(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
                  const float* gradOutput, const float* tanhOutput,
                  float* gradInput, uint32_t totalElements);

/// Softmax push constants — matches activation-softmax.glsl layout.
struct SoftmaxParams {
    uint32_t batchSize;
    uint32_t seqLen;
    uint32_t features;
    uint32_t passType;
    uint32_t dim;
};

/// GPU Softmax forward (3-pass: max → sum_exp → normalize).
/// 4 buffers: input, output, max_vals, sum_exp.
void softmax(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
             const float* input, float* output,
             uint32_t batchSize, uint32_t seqLen, uint32_t features);

/// Softmax backward push constants.
struct SoftmaxBackwardParams {
    uint32_t batchSize;
    uint32_t seqLen;
    uint32_t numClasses;
};

/// GPU Softmax backward.
/// 3 buffers: grad_output, softmax_output, grad_input.
void softmaxBackward(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
                     const float* gradOutput, const float* softmaxOutput,
                     float* gradInput, uint32_t batchSize, uint32_t seqLen,
                     uint32_t numClasses);

/// Multiplication-free softmax (ReLU-normalized): shader ``mf-softmax`` (same 3-pass
/// layout as ``activation-softmax``).
void mfSoftmax(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
               const float* input, float* output, uint32_t batchSize, uint32_t seqLen,
               uint32_t features);

/// Algebraic softplus: ``0.5 * (x + sqrt(x*x + c))`` with ``c = 4/beta^2``. Shader
/// ``mf-softplus`` (2 buffers).
void mfSoftplus(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
                const float* input, float* output, uint32_t totalElements, float beta);

/// Rational sigmoid ``x / (1 + |x|)``. Shader ``mf-sigmoid`` (2 buffers).
void mfSigmoid(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
               const float* input, float* output, uint32_t totalElements);

}  // namespace ops
}  // namespace grilly
