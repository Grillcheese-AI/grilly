#pragma once

#include <Eigen/Dense>

#include <cstdint>
#include <optional>
#include <vector>

#include "grilly/buffer_pool.h"
#include "grilly/command_batch.h"
#include "grilly/pipeline_cache.h"

namespace grilly {
namespace ops {

/// GPU-accelerated linear projection: output = x @ W^T + bias.
///
/// Dynamic dtype via ``elemSize``:
///   - ``elemSize == 4`` (fp32): dispatches ``fnn-linear.glsl`` (tiled GEMM,
///     universal path, works on any device). x / weights / bias are
///     interpreted as ``float``; output is always ``float``.
///   - ``elemSize == 2`` (fp16): dispatches ``gemm-coopmat-shared.glsl`` if
///     the device supports ``VK_KHR_cooperative_matrix`` AND the shape is
///     aligned (M % 16 == 0, K % 16 == 0, N % 64 == 0). x / weights / bias
///     are ``float16_t`` bytes in the staging buffer; output remains fp32
///     (the coopmat accumulator runs in fp32 for numerical stability).
///     Falls back to an fp32 conversion + fnn-linear path if the device
///     lacks cooperative matrix support or the shape isn't aligned.
///
/// Bias handling: on the coopmat path, bias is applied via a small second
/// dispatch (``gemm-bias-add.glsl``) because ``coopMatStore`` can't
/// interleave an elementwise add with the tile-cooperative store.
///
/// Push constants layout (matches fnn-linear.glsl):
///   uint batch_seq;   // offset  0
///   uint input_dim;   // offset  4
///   uint output_dim;  // offset  8
///   uint has_bias;    // offset 12
///   uint elem_size;   // offset 16 — 4 for fp32, 2 for fp16
struct LinearParams {
    uint32_t batchSeq;
    uint32_t inputDim;
    uint32_t outputDim;
    uint32_t hasBias;
    uint32_t elemSize;  ///< 4 for fp32, 2 for fp16
};

/// Execute a linear (dense / fully-connected) layer on the GPU.
///
/// Pointer types are ``const void*`` / ``void*`` so the caller can pass
/// either fp32 or fp16 byte buffers transparently — the element size is
/// encoded in ``p.elemSize``.
///
/// @param batch   CommandBatch to record the dispatch into.
/// @param pool    BufferPool for acquiring/releasing GPU memory.
/// @param cache   PipelineCache for the fnn-linear / gemm-coopmat shaders.
/// @param x       Input matrix, row-major (batchSeq × inputDim).
/// @param weights Weight matrix, row-major (outputDim × inputDim).
/// @param bias    Optional bias vector (outputDim).  nullptr = no bias.
/// @param output  Output buffer, pre-allocated (batchSeq × outputDim × 4 bytes).
/// @param p       Dimension + dtype parameters.
void linear(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
            const void* x, const void* weights, const void* bias,
            void* output, const LinearParams& p);

/// CPU reference implementation using Eigen for correctness verification.
/// Returns a newly-allocated vector: output = x @ W^T + bias.
std::vector<float> linearCPU(const float* x, const float* weights,
                             const float* bias, const LinearParams& p);

/// Linear backward push constants — matches fnn-linear-backward.glsl.
/// Uses pass_type: 0=grad_input, 1=grad_weight, 2=grad_bias.
struct LinearBackwardParams {
    uint32_t batchSeq;
    uint32_t inputDim;
    uint32_t outputDim;
    uint32_t passType;
};

/// GPU linear backward. Produces grad_input, grad_weight, grad_bias.
/// 3-pass dispatch with barriers. Dtype-dynamic via ``p.elemSize``.
/// 6 buffers: grad_output(0), input(1), weights(2),
///            grad_input(3), grad_weight(4), grad_bias(5).
///
/// NOTE: the current backward shader (``fnn-linear-backward.glsl``) is
/// fp32-only. Calling with ``p.elemSize == 2`` will throw until a
/// cooperative-matrix backward shader lands. The ``void*`` / dynamic
/// elemSize interface is in place so the upgrade is a local shader change.
void linearBackward(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
                    const void* gradOutput, const void* input,
                    const void* weights,
                    void* gradInput, void* gradWeight, void* gradBias,
                    const LinearParams& p);

/// fp16 cooperative-matrix backward. Same math + signature as linearBackward,
/// but the grad_input / grad_weight GEMMs run through gemm-coopmat-shared
/// (fp16 in, fp32 accumulate). Inputs AND outputs are fp32 (conversion is
/// internal). Caller MUST ensure coopmat alignment: BS%16, out%16, in%64.
/// grad_bias is computed in fp32 via fnn-linear-backward pass 2.
void linearBackwardCoopmat(CommandBatch& batch, BufferPool& pool,
                           PipelineCache& cache,
                           const void* gradOutput, const void* input,
                           const void* weights,
                           void* gradInput, void* gradWeight, void* gradBias,
                           const LinearParams& p);

/// Dropout push constants — matches fnn-dropout.glsl.
struct DropoutParams {
    uint32_t totalElements;
    float dropoutProb;
    uint32_t isTraining;
};

/// GPU dropout (inverted scaling in forward).
/// 3 buffers: input(0), random_mask(1), output(2).
void dropout(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
             const float* input, const float* randomMask, float* output,
             uint32_t totalElements, float dropoutProb, bool isTraining);

}  // namespace ops
}  // namespace grilly
