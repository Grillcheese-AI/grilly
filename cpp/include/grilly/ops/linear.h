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
/// Ports backend/fnn.py:1823-1976 to native C++.
///
/// Push constants layout (must match fnn-linear.glsl):
///   uint batch_seq;   // offset  0
///   uint input_dim;   // offset  4
///   uint output_dim;  // offset  8
///   uint has_bias;    // offset 12
///
/// Buffers (binding order matches shader):
///   0: input   (batch_seq * input_dim  floats)
///   1: weights (output_dim * input_dim floats)
///   2: bias    (output_dim floats, or 1 dummy float)
///   3: output  (batch_seq * output_dim floats)
struct LinearParams {
    uint32_t batchSeq;
    uint32_t inputDim;
    uint32_t outputDim;
    uint32_t hasBias;
};

/// Execute a linear (dense / fully-connected) layer on the GPU.
///
/// All Vulkan work — buffer upload, pipeline bind, descriptor set,
/// dispatch, download — happens inside C++ with zero Python crossings.
///
/// @param batch   CommandBatch to record the dispatch into.
/// @param pool    BufferPool for acquiring/releasing GPU memory.
/// @param cache   PipelineCache for the fnn-linear shader.
/// @param x       Input matrix, row-major (batchSeq × inputDim).
/// @param weights Weight matrix, row-major (outputDim × inputDim).
/// @param bias    Optional bias vector (outputDim).  nullptr = no bias.
/// @param output  Output buffer, pre-allocated (batchSeq × outputDim).
/// @param p       Dimension parameters.
void linear(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
            const float* x, const float* weights, const float* bias,
            float* output, const LinearParams& p);

/// CPU reference implementation using Eigen for correctness verification.
/// Returns a newly-allocated vector: output = x @ W^T + bias.
std::vector<float> linearCPU(const float* x, const float* weights,
                             const float* bias, const LinearParams& p);

}  // namespace ops
}  // namespace grilly
