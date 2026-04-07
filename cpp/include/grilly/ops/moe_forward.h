#pragma once
/// Fused MoE forward / backward — minimal Python crossings for training hot path.
///
/// `moe_forward_gpu` runs embedding, per-layer expert GEMMs (batched), residual,
/// and output projection on GPU. Router softmax and expert blending use CPU for
/// correctness and simplicity (small tensors); blend can move to a dedicated
/// shader in a follow-up.

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "grilly/buffer_pool.h"
#include "grilly/command_batch.h"
#include "grilly/pipeline_cache.h"

namespace grilly {
namespace ops {

struct MoeLayerGPU {
    GrillyBuffer routerW;
    GrillyBuffer routerB;
    std::vector<GrillyBuffer> expertW;
    std::vector<GrillyBuffer> expertWt;
    GrillyBuffer expertPacked;  // All experts packed contiguously (n_experts * d * d)
};

struct MoeGradients {
    std::vector<float> grad_embed;
    std::vector<float> grad_pos;
    std::vector<std::vector<float>> grad_experts;
    std::vector<std::vector<float>> grad_router_w;
    std::vector<std::vector<float>> grad_router_b;
    std::vector<float> grad_out_w;
};

struct MoeHandleCache {
    uint32_t vocab = 0;
    uint32_t d = 0;
    uint32_t maxSeq = 0;
    uint32_t nLayers = 0;
    uint32_t nExperts = 0;

    GrillyBuffer embedW;
    GrillyBuffer posW;
    GrillyBuffer outW;
    GrillyBuffer outWt;

    std::vector<MoeLayerGPU> layers;

    GrillyBuffer bufIds;
    GrillyBuffer bufPosSlice;
    GrillyBuffer bufX;
    std::vector<GrillyBuffer> bufExpertOut;
    GrillyBuffer bufBlended;
    GrillyBuffer bufLogits;

    // Saved activations for backward (one per layer + final)
    std::vector<GrillyBuffer> bufActivations;  // [0]=after embed, [l]=after layer l-1
    // Router weights per layer (saved from forward for backward)
    std::vector<std::vector<float>> fwd_router_weights;

    std::vector<float> cpu_embed;
    std::vector<float> cpu_pos;
    std::vector<float> cpu_out_w;
    std::vector<std::vector<float>> cpu_router_w;
    std::vector<std::vector<float>> cpu_router_b;
    std::vector<std::vector<float>> cpu_expert_w;
};

int moe_upload(BufferPool& pool,
               uint32_t vocab_size, uint32_t d_model, uint32_t max_seq,
               const float* embed_w, const float* pos_w,
               const std::vector<const float*>& expert_ws,
               const std::vector<const float*>& router_ws,
               const std::vector<const float*>& router_bs,
               const float* out_w,
               uint32_t n_layers, uint32_t n_experts);

MoeHandleCache& moe_get_cache(int handle);

void moe_release(BufferPool& pool, int handle);

void moe_update_weights(BufferPool& pool, MoeHandleCache& h,
                        const float* embed_w, const float* pos_w,
                        const std::vector<const float*>& expert_ws,
                        const std::vector<const float*>& router_ws,
                        const std::vector<const float*>& router_bs,
                        const float* out_w);

void moe_forward_gpu(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
                     MoeHandleCache& h, const int32_t* input_ids, uint32_t seq_len,
                     float* logits_out);

MoeGradients moe_backward_cpu(const MoeHandleCache& h,
                              const int32_t* input_ids, uint32_t seq_len,
                              const float* grad_logits);

MoeGradients moe_backward_gpu(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
                               MoeHandleCache& h, const int32_t* input_ids, uint32_t seq_len,
                               const float* grad_logits);

}  // namespace ops
}  // namespace grilly
