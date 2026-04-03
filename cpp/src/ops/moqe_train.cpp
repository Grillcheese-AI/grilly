/// moqe_train.cpp — Persistent-weight MoQE training with batched dispatch.
///
/// Fixes applied from Vulkan expert review:
///   1. batch.submit() has built-in fence wait — verified safe for buffer reuse
///   2. Removed unused bufXT/bufGW allocations (grad_W on CPU)
///   3. Added moqe_train_release() to prevent VRAM leaks
///   4. Barrier-free dual-expert dispatch within each layer

#include "grilly/ops/moqe_train.h"
#include "grilly/ops/batched_ops.h"

#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace grilly {
namespace ops {

static std::unordered_map<int, MoQETrainCache> g_moqe_caches;
static int g_next_moqe_handle = 1;

MoQETrainCache& moqe_train_get_cache(int handle) {
    auto it = g_moqe_caches.find(handle);
    if (it == g_moqe_caches.end())
        throw std::runtime_error("Invalid MoQE train cache handle");
    return it->second;
}

static void uploadWeightPair(BufferPool& pool,
                              GrillyBuffer& wBuf, GrillyBuffer& wtBuf,
                              const float* w, uint32_t d) {
    size_t bytes = size_t(d) * d * sizeof(float);

    wBuf = pool.acquire(bytes);
    pool.upload(wBuf, w, bytes);

    std::vector<float> wt(d * d);
    for (uint32_t r = 0; r < d; ++r)
        for (uint32_t c = 0; c < d; ++c)
            wt[c * d + r] = w[r * d + c];

    wtBuf = pool.acquire(bytes);
    pool.upload(wtBuf, wt.data(), bytes);
}

int moqe_train_upload(BufferPool& pool,
                      const std::vector<const float*>& weight_ptrs,
                      uint32_t dModel, uint32_t nLayers, uint32_t maxSeq) {

    MoQETrainCache tc;
    tc.nLayers = nLayers;
    tc.dModel = dModel;
    tc.maxSeq = maxSeq;

    tc.layers.resize(nLayers);
    size_t idx = 0;
    for (uint32_t l = 0; l < nLayers; ++l) {
        uploadWeightPair(pool, tc.layers[l].w0, tc.layers[l].w0_t,
                         weight_ptrs[idx++], dModel);
        uploadWeightPair(pool, tc.layers[l].w1, tc.layers[l].w1_t,
                         weight_ptrs[idx++], dModel);
    }

    // Working buffers: dual for barrier-free 2-expert dispatch
    // Reused across layers — safe because batch.submit() has fence wait
    size_t seqBytes = size_t(maxSeq) * dModel * sizeof(float);
    tc.bufA0 = pool.acquire(seqBytes);
    tc.bufA1 = pool.acquire(seqBytes);
    tc.bufC0 = pool.acquire(seqBytes);
    tc.bufC1 = pool.acquire(seqBytes);

    int handle = g_next_moqe_handle++;
    g_moqe_caches[handle] = std::move(tc);
    return handle;
}

void moqe_train_release(BufferPool& pool, int handle) {
    auto it = g_moqe_caches.find(handle);
    if (it == g_moqe_caches.end()) return;

    auto& tc = it->second;
    for (auto& lw : tc.layers) {
        pool.release(lw.w0);
        pool.release(lw.w0_t);
        pool.release(lw.w1);
        pool.release(lw.w1_t);
    }
    pool.release(tc.bufA0);
    pool.release(tc.bufA1);
    pool.release(tc.bufC0);
    pool.release(tc.bufC1);

    g_moqe_caches.erase(it);
}

void moqe_train_update_expert(BufferPool& pool, MoQETrainCache& cache,
                               uint32_t layerIdx, int expertIdx,
                               const float* w, uint32_t dModel) {
    auto& lw = cache.layers[layerIdx];
    GrillyBuffer& wBuf  = (expertIdx == 0) ? lw.w0 : lw.w1;
    GrillyBuffer& wtBuf = (expertIdx == 0) ? lw.w0_t : lw.w1_t;

    size_t bytes = size_t(dModel) * dModel * sizeof(float);

    // Re-upload W into EXISTING buffer (no new allocation)
    pool.upload(wBuf, w, bytes);

    // Transpose on CPU and re-upload W^T into EXISTING buffer
    std::vector<float> wt(dModel * dModel);
    for (uint32_t r = 0; r < dModel; ++r)
        for (uint32_t c = 0; c < dModel; ++c)
            wt[c * dModel + r] = w[r * dModel + c];
    pool.upload(wtBuf, wt.data(), bytes);
}

// ── Forward: 2 experts, 1 submit, barrier-free ──────────────────────

void moqe_layer_forward_gpu(CommandBatch& batch, BufferPool& pool,
                             PipelineCache& cache, MoQETrainCache& tc,
                             uint32_t layerIdx,
                             const float* x0, uint32_t n0,
                             const float* x1, uint32_t n1,
                             float* out0, float* out1) {

    const uint32_t d = tc.dModel;
    auto& lw = tc.layers[layerIdx];

    if (n0 > 0) pool.upload(tc.bufA0, x0, size_t(n0) * d * sizeof(float));
    if (n1 > 0) pool.upload(tc.bufA1, x1, size_t(n1) * d * sizeof(float));

    batch.begin();
    if (n0 > 0)
        batchedTiledLinear(batch, cache, tc.bufA0, lw.w0, nullptr, tc.bufC0,
                           n0, d, d);
    if (n1 > 0)
        batchedTiledLinear(batch, cache, tc.bufA1, lw.w1, nullptr, tc.bufC1,
                           n1, d, d);
    // No barrier between experts — independent reads/writes
    batch.submitDeferred();
    batch.waitForCompletion();  // Fence wait — safe to reuse buffers after this returns

    if (n0 > 0) pool.download(tc.bufC0, out0, size_t(n0) * d * sizeof(float));
    if (n1 > 0) pool.download(tc.bufC1, out1, size_t(n1) * d * sizeof(float));
}

// ── Backward dx: 2 experts, 1 submit, barrier-free ──────────────────

void moqe_layer_backward_dx_gpu(CommandBatch& batch, BufferPool& pool,
                                 PipelineCache& cache, MoQETrainCache& tc,
                                 uint32_t layerIdx,
                                 const float* d0, uint32_t n0,
                                 const float* d1, uint32_t n1,
                                 float* dx0, float* dx1) {

    const uint32_t d = tc.dModel;
    auto& lw = tc.layers[layerIdx];

    if (n0 > 0) pool.upload(tc.bufA0, d0, size_t(n0) * d * sizeof(float));
    if (n1 > 0) pool.upload(tc.bufA1, d1, size_t(n1) * d * sizeof(float));

    // dx = d_out @ W: batchedTiledLinear(d_out, W^T) → d_out @ (W^T)^T = d_out @ W
    batch.begin();
    if (n0 > 0)
        batchedTiledLinear(batch, cache, tc.bufA0, lw.w0_t, nullptr, tc.bufC0,
                           n0, d, d);
    if (n1 > 0)
        batchedTiledLinear(batch, cache, tc.bufA1, lw.w1_t, nullptr, tc.bufC1,
                           n1, d, d);
    batch.submitDeferred();
    batch.waitForCompletion();

    if (n0 > 0) pool.download(tc.bufC0, dx0, size_t(n0) * d * sizeof(float));
    if (n1 > 0) pool.download(tc.bufC1, dx1, size_t(n1) * d * sizeof(float));
}

}  // namespace ops
}  // namespace grilly
