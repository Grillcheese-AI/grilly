#include "grilly/ops/optimizer.h"

#include <cstring>

namespace grilly {
namespace ops {

// ═══════════════════════════════════════════════════════════════════════════
// GPU optimizer steps
//
// Each optimizer updates parameters in-place. The shader reads W, grad,
// momentum state (m, v), applies the update rule, and optionally zeros
// gradients (clear_grad=1) in a single dispatch.
//
// dispatch at (256,1,1) — one thread per weight.
// ═══════════════════════════════════════════════════════════════════════════

// ── Adam ─────────────────────────────────────────────────────────────────

void adamUpdate(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
                float* weights, float* grad, float* m, float* v,
                const AdamParams& p) {
    const size_t bytes = size_t(p.totalWeights) * sizeof(float);

    GrillyBuffer bufW    = pool.acquire(bytes);
    GrillyBuffer bufGrad = pool.acquire(bytes);
    GrillyBuffer bufM    = pool.acquire(bytes);
    GrillyBuffer bufV    = pool.acquire(bytes);

    pool.upload(bufW, weights, bytes);
    pool.upload(bufGrad, grad, bytes);
    pool.upload(bufM, m, bytes);
    pool.upload(bufV, v, bytes);

    PipelineEntry pipe = cache.getOrCreate("adam-update", 4,
                                           sizeof(AdamParams));

    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {bufW.handle,    0, bytes},
        {bufGrad.handle, 0, bytes},
        {bufM.handle,    0, bytes},
        {bufV.handle,    0, bytes},
    };
    VkDescriptorSet descSet = cache.allocDescriptorSet("adam-update", bufInfos);

    uint32_t gx = (p.totalWeights + 255) / 256;

    batch.begin();
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, 1, 1,
                   &p, sizeof(p));
    batch.submit();

    pool.download(bufW, weights, bytes);
    pool.download(bufGrad, grad, bytes);
    pool.download(bufM, m, bytes);
    pool.download(bufV, v, bytes);

    pool.release(bufW);
    pool.release(bufGrad);
    pool.release(bufM);
    pool.release(bufV);
}

// ── AdamW ────────────────────────────────────────────────────────────────

void adamwUpdate(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
                 float* weights, float* grad, float* m, float* v,
                 const AdamWParams& p) {
    const size_t bytes = size_t(p.totalWeights) * sizeof(float);

    GrillyBuffer bufW    = pool.acquire(bytes);
    GrillyBuffer bufGrad = pool.acquire(bytes);
    GrillyBuffer bufM    = pool.acquire(bytes);
    GrillyBuffer bufV    = pool.acquire(bytes);

    pool.upload(bufW, weights, bytes);
    pool.upload(bufGrad, grad, bytes);
    pool.upload(bufM, m, bytes);
    pool.upload(bufV, v, bytes);

    PipelineEntry pipe = cache.getOrCreate("adamw-update", 4,
                                           sizeof(AdamWParams));

    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {bufW.handle,    0, bytes},
        {bufGrad.handle, 0, bytes},
        {bufM.handle,    0, bytes},
        {bufV.handle,    0, bytes},
    };
    VkDescriptorSet descSet = cache.allocDescriptorSet("adamw-update",
                                                        bufInfos);

    uint32_t gx = (p.totalWeights + 255) / 256;

    batch.begin();
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, 1, 1,
                   &p, sizeof(p));
    batch.submit();

    pool.download(bufW, weights, bytes);
    pool.download(bufGrad, grad, bytes);
    pool.download(bufM, m, bytes);
    pool.download(bufV, v, bytes);

    pool.release(bufW);
    pool.release(bufGrad);
    pool.release(bufM);
    pool.release(bufV);
}

}  // namespace ops
}  // namespace grilly
