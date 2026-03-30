#pragma once
/// Native MoQE Training — persistent GPU weights, batched expert dispatch.
///
/// Expert weights (W + W^T) uploaded once, stay on GPU permanently.
/// Per-step: upload tiny masked activations, dispatch via batchedTiledLinear,
/// download results. Two submits per layer (forward + backward dx).
///
/// VRAM budget at d=2048, 16 layers:
///   Weights: 32MB × 32 experts = 1GB (W + W^T)
///   Working: 4 × maxSeq×d = ~32MB
///   Total: ~1.03GB of 12GB VRAM

#include <cstdint>
#include <vector>

#include "grilly/buffer_pool.h"
#include "grilly/command_batch.h"
#include "grilly/pipeline_cache.h"

namespace grilly {
namespace ops {

struct MoQELayerGPU {
    GrillyBuffer w0, w0_t;  // Expert 0: W(d,d) + W^T(d,d)
    GrillyBuffer w1, w1_t;  // Expert 1: W(d,d) + W^T(d,d)
};

struct MoQETrainCache {
    uint32_t nLayers = 0;
    uint32_t dModel = 0;
    uint32_t maxSeq = 0;

    std::vector<MoQELayerGPU> layers;

    // Dual working buffers for barrier-free 2-expert dispatch
    GrillyBuffer bufA0, bufA1;  // Input for expert 0, 1
    GrillyBuffer bufC0, bufC1;  // Output for expert 0, 1
};

/// Upload dequantized expert weights (W + W^T) to GPU.
/// weight_ptrs: [w0_L0, w1_L0, w0_L1, w1_L1, ...], each (d,d) row-major.
int moqe_train_upload(BufferPool& pool,
                      const std::vector<const float*>& weight_ptrs,
                      uint32_t dModel, uint32_t nLayers, uint32_t maxSeq);

MoQETrainCache& moqe_train_get_cache(int handle);

/// Release all GPU buffers. Call when done training or before re-init.
void moqe_train_release(BufferPool& pool, int handle);

/// Re-upload one expert's weights after optimizer step.
void moqe_train_update_expert(BufferPool& pool, MoQETrainCache& cache,
                               uint32_t layerIdx, int expertIdx,
                               const float* w, uint32_t dModel);

/// Forward BOTH experts, single submit, barrier-free.
/// batch.submit() blocks until GPU completes (fence wait).
void moqe_layer_forward_gpu(CommandBatch& batch, BufferPool& pool,
                             PipelineCache& cache, MoQETrainCache& tc,
                             uint32_t layerIdx,
                             const float* x0, uint32_t n0,
                             const float* x1, uint32_t n1,
                             float* out0, float* out1);

/// Backward dx for BOTH experts, single submit, barrier-free.
/// dx = d_out @ W (uses persistent W^T on GPU).
/// grad_W stays on CPU (caller does d_out^T @ x_in via numpy BLAS).
void moqe_layer_backward_dx_gpu(CommandBatch& batch, BufferPool& pool,
                                 PipelineCache& cache, MoQETrainCache& tc,
                                 uint32_t layerIdx,
                                 const float* d0, uint32_t n0,
                                 const float* d1, uint32_t n1,
                                 float* dx0, float* dx1);

}  // namespace ops
}  // namespace grilly
