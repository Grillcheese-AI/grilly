#pragma once
/// VSA Language Model fused forward/backward — AdditionLinear FFN + MindForge LoRA.
///
/// Uses addition-linear shader (L1 distance, no matmul) for FFN layers and
/// fnn-linear for the output projection only.  GPU router + MindForge LoRA
/// on CPU (tiny).  Target: 10x faster than Python AdditionLinear loops.

#include <cstdint>
#include <vector>

#include "grilly/buffer_pool.h"
#include "grilly/command_batch.h"
#include "grilly/pipeline_cache.h"

namespace grilly {
namespace ops {

struct VsaLmLayerGPU {
    GrillyBuffer ffnUpW;     // (d_ffn, d) addition-linear patterns
    GrillyBuffer ffnUpB;     // (d_ffn,)
    GrillyBuffer ffnDownW;   // (d, d_ffn) addition-linear patterns
    GrillyBuffer ffnDownB;   // (d,)
    GrillyBuffer lnGamma;    // (d,)
    GrillyBuffer lnBeta;     // (d,)
};

struct VsaLmGradients {
    std::vector<float> grad_embed;                     // (vocab, d)
    std::vector<float> grad_pos;                       // (max_seq, d)
    std::vector<std::vector<float>> grad_ffn_up_w;     // n_layers of (d_ffn * d)
    std::vector<std::vector<float>> grad_ffn_up_b;     // n_layers of (d_ffn)
    std::vector<std::vector<float>> grad_ffn_down_w;   // n_layers of (d * d_ffn)
    std::vector<std::vector<float>> grad_ffn_down_b;   // n_layers of (d)
    std::vector<std::vector<float>> grad_ln_gamma;     // n_layers of (d)
    std::vector<std::vector<float>> grad_ln_beta;      // n_layers of (d)
    std::vector<float> grad_out_w;                     // (vocab, d)
};

struct VsaLmHandleCache {
    uint32_t vocab   = 0;
    uint32_t d       = 0;
    uint32_t dFfn    = 0;
    uint32_t maxSeq  = 0;
    uint32_t nLayers = 0;

    // GPU buffers
    GrillyBuffer embedW;
    GrillyBuffer posW;
    GrillyBuffer outW;
    GrillyBuffer outWt;  // transposed for backward

    std::vector<VsaLmLayerGPU> layers;

    GrillyBuffer bufIds;
    GrillyBuffer bufPosSlice;
    GrillyBuffer bufX;          // current hidden (seq, d)
    GrillyBuffer bufLnOut;      // layernorm output (seq, d)
    GrillyBuffer bufFfnUp;      // after addition-linear up (seq, d_ffn)
    GrillyBuffer bufSign;       // after sign activation (seq, d_ffn)
    GrillyBuffer bufFfnDown;    // after addition-linear down (seq, d)
    GrillyBuffer bufLogits;     // (seq, vocab)

    // LayerNorm scratch (mean + var buffers)
    GrillyBuffer bufLnMean;     // (seq,)
    GrillyBuffer bufLnVar;      // (seq,)

    // Saved activations for backward (n_layers + 1)
    std::vector<GrillyBuffer> bufActivations;

    // CPU mirrors for backward
    std::vector<float> cpu_embed;
    std::vector<float> cpu_pos;
    std::vector<float> cpu_out_w;
    std::vector<std::vector<float>> cpu_ffn_up_w;   // n_layers
    std::vector<std::vector<float>> cpu_ffn_up_b;
    std::vector<std::vector<float>> cpu_ffn_down_w;
    std::vector<std::vector<float>> cpu_ffn_down_b;
    std::vector<std::vector<float>> cpu_ln_gamma;
    std::vector<std::vector<float>> cpu_ln_beta;

    // MindForge weights (CPU only — tiny)
    std::vector<float> forge_W_proj;       // (d_hidden, d_vsa)
    std::vector<float> forge_W_h;          // (d_hidden, d_hidden*2)
    std::vector<float> forge_b_h;          // (d_hidden,)
    std::vector<float> forge_W_coeff;      // (n_basis, d_hidden)
    std::vector<float> forge_b_coeff;      // (n_basis,)
    std::vector<float> forge_A_basis;      // (n_basis, rank, d)
    std::vector<float> forge_B_basis;      // (n_basis, d, rank)
    std::vector<float> forge_layer_embs;   // (n_layers, d_hidden)
    uint32_t forge_d_hidden = 0;
    uint32_t forge_d_vsa    = 0;
    uint32_t forge_n_basis  = 0;
    uint32_t forge_rank     = 0;
};

int vsa_lm_upload(BufferPool& pool,
                  uint32_t vocab, uint32_t d, uint32_t d_ffn, uint32_t max_seq,
                  const float* embed_w, const float* pos_w,
                  const std::vector<const float*>& ffn_up_ws,
                  const std::vector<const float*>& ffn_up_bs,
                  const std::vector<const float*>& ffn_down_ws,
                  const std::vector<const float*>& ffn_down_bs,
                  const std::vector<const float*>& ln_gammas,
                  const std::vector<const float*>& ln_betas,
                  const float* out_w,
                  uint32_t n_layers);

VsaLmHandleCache& vsa_lm_get_cache(int handle);

void vsa_lm_release(BufferPool& pool, int handle);

void vsa_lm_forward_gpu(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
                        VsaLmHandleCache& h, const int32_t* input_ids,
                        uint32_t seq_len, float* logits_out);

VsaLmGradients vsa_lm_backward_cpu(const VsaLmHandleCache& h,
                                   const int32_t* input_ids, uint32_t seq_len,
                                   const float* grad_logits);

}  // namespace ops
}  // namespace grilly
