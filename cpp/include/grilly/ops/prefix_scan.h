#pragma once

#include <cstdint>

#include "grilly/buffer_pool.h"
#include "grilly/command_batch.h"
#include "grilly/pipeline_cache.h"

namespace grilly {
namespace ops {

/// Causal Linear-RNN prefix scan push constants.
/// Matches shaders/prefix-scan-causal.glsl + prefix-scan-causal-backward.glsl.
struct PrefixScanParams {
    uint32_t seqLen;     ///< must be <= subgroup size (32 Wave32 / 64 Wave64)
    uint32_t hiddenDim;
    uint32_t batchSize;
};

/// Causal linear-RNN forward: h_t = a_t * h_{t-1} + x_t.
///
/// Implemented as two hardware subgroup inclusive scans (log(a) and the
/// rescaled x). The recurrence is strictly causal: h_t depends only on
/// x_{0..t} and a_{0..t}.
///
/// Inputs / outputs are fp32 (B, S, D) row-major.
///   x: input sequence, shape (B, S, D)
///   a: decay gates in (0, 1], same shape
///   h: output hidden states, same shape (pre-allocated)
///
/// Buffer bindings (match the shader):
///   0: x (input)
///   1: a (decay)
///   2: h (output)
///
/// Constraint: seqLen <= subgroup size. Longer sequences need a hierarchical
/// scan — not implemented yet.
void prefixScanCausal(CommandBatch& batch, BufferPool& pool,
                      PipelineCache& cache,
                      const float* x, const float* a, float* h,
                      const PrefixScanParams& p);

/// Causal prefix scan backward: given dh, compute dx and da.
///
/// Math: dx_t = sum_{s>=t} (prod_{k=t+1..s} a_k) * dh_s  (anti-causal scan)
///       da_t = dx_t * h_{t-1}
///
/// Needs the forward x and h tensors saved from the forward pass.
///
/// Buffer bindings:
///   0: dh (grad into output)
///   1: a  (decay)
///   2: h  (forward output, for da computation)
///   3: x  (forward input, for da computation)
///   4: dx (grad w.r.t. input, output)
///   5: da (grad w.r.t. decay, output)
void prefixScanCausalBackward(CommandBatch& batch, BufferPool& pool,
                              PipelineCache& cache,
                              const float* dh, const float* a,
                              const float* h, const float* x,
                              float* dx, float* da,
                              const PrefixScanParams& p);

}  // namespace ops
}  // namespace grilly
