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

    // Adam updates all 4 buffers in-place: each is both stage-in and
    // stage-out. Use HOST_CACHED readback staging for all 4 since we read
    // them back to CPU at the end of every step.
    GrillyBuffer bufWDL    = pool.acquireDeviceLocal(bytes);
    GrillyBuffer bufGradDL = pool.acquireDeviceLocal(bytes);
    GrillyBuffer bufMDL    = pool.acquireDeviceLocal(bytes);
    GrillyBuffer bufVDL    = pool.acquireDeviceLocal(bytes);

    GrillyBuffer bufWStage    = pool.acquireReadback(bytes);
    GrillyBuffer bufGradStage = pool.acquireReadback(bytes);
    GrillyBuffer bufMStage    = pool.acquireReadback(bytes);
    GrillyBuffer bufVStage    = pool.acquireReadback(bytes);

    pool.upload(bufWStage,    weights, bytes);
    pool.upload(bufGradStage, grad,    bytes);
    pool.upload(bufMStage,    m,       bytes);
    pool.upload(bufVStage,    v,       bytes);

    PipelineEntry pipe = cache.getOrCreate("adam-update", 4,
                                           sizeof(AdamParams));

    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {bufWDL.handle,    0, bytes},
        {bufGradDL.handle, 0, bytes},
        {bufMDL.handle,    0, bytes},
        {bufVDL.handle,    0, bytes},
    };
    VkDescriptorSet descSet = cache.allocDescriptorSet("adam-update", bufInfos);

    uint32_t gx = (p.totalWeights + 255) / 256;

    batch.begin();
    batch.copyBuffer(bufWStage,    bufWDL,    bytes);
    batch.copyBuffer(bufGradStage, bufGradDL, bytes);
    batch.copyBuffer(bufMStage,    bufMDL,    bytes);
    batch.copyBuffer(bufVStage,    bufVDL,    bytes);
    batch.transferComputeBarrier();
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, 1, 1,
                   &p, sizeof(p));
    batch.transferComputeBarrier();
    batch.copyBuffer(bufWDL,    bufWStage,    bytes);
    batch.copyBuffer(bufGradDL, bufGradStage, bytes);
    batch.copyBuffer(bufMDL,    bufMStage,    bytes);
    batch.copyBuffer(bufVDL,    bufVStage,    bytes);
    batch.submitDeferred();
    batch.waitForCompletion();

    pool.download(bufWStage,    weights, bytes);
    pool.download(bufGradStage, grad,    bytes);
    pool.download(bufMStage,    m,       bytes);
    pool.download(bufVStage,    v,       bytes);

    pool.release(bufWDL);
    pool.release(bufGradDL);
    pool.release(bufMDL);
    pool.release(bufVDL);
    pool.release(bufWStage);
    pool.release(bufGradStage);
    pool.release(bufMStage);
    pool.release(bufVStage);
}

// ── AdamW ────────────────────────────────────────────────────────────────

void adamwUpdate(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
                 float* weights, float* grad, float* m, float* v,
                 const AdamWParams& p) {
    const size_t bytes = size_t(p.totalWeights) * sizeof(float);

    // Same staging pattern as adamUpdate — all 4 buffers in-place updated.
    GrillyBuffer bufWDL    = pool.acquireDeviceLocal(bytes);
    GrillyBuffer bufGradDL = pool.acquireDeviceLocal(bytes);
    GrillyBuffer bufMDL    = pool.acquireDeviceLocal(bytes);
    GrillyBuffer bufVDL    = pool.acquireDeviceLocal(bytes);

    GrillyBuffer bufWStage    = pool.acquireReadback(bytes);
    GrillyBuffer bufGradStage = pool.acquireReadback(bytes);
    GrillyBuffer bufMStage    = pool.acquireReadback(bytes);
    GrillyBuffer bufVStage    = pool.acquireReadback(bytes);

    pool.upload(bufWStage,    weights, bytes);
    pool.upload(bufGradStage, grad,    bytes);
    pool.upload(bufMStage,    m,       bytes);
    pool.upload(bufVStage,    v,       bytes);

    PipelineEntry pipe = cache.getOrCreate("adamw-update", 4,
                                           sizeof(AdamWParams));

    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {bufWDL.handle,    0, bytes},
        {bufGradDL.handle, 0, bytes},
        {bufMDL.handle,    0, bytes},
        {bufVDL.handle,    0, bytes},
    };
    VkDescriptorSet descSet = cache.allocDescriptorSet("adamw-update",
                                                        bufInfos);

    uint32_t gx = (p.totalWeights + 255) / 256;

    batch.begin();
    batch.copyBuffer(bufWStage,    bufWDL,    bytes);
    batch.copyBuffer(bufGradStage, bufGradDL, bytes);
    batch.copyBuffer(bufMStage,    bufMDL,    bytes);
    batch.copyBuffer(bufVStage,    bufVDL,    bytes);
    batch.transferComputeBarrier();
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, 1, 1,
                   &p, sizeof(p));
    batch.transferComputeBarrier();
    batch.copyBuffer(bufWDL,    bufWStage,    bytes);
    batch.copyBuffer(bufGradDL, bufGradStage, bytes);
    batch.copyBuffer(bufMDL,    bufMStage,    bytes);
    batch.copyBuffer(bufVDL,    bufVStage,    bytes);
    batch.submitDeferred();
    batch.waitForCompletion();

    pool.download(bufWStage,    weights, bytes);
    pool.download(bufGradStage, grad,    bytes);
    pool.download(bufMStage,    m,       bytes);
    pool.download(bufVStage,    v,       bytes);

    pool.release(bufWDL);
    pool.release(bufGradDL);
    pool.release(bufMDL);
    pool.release(bufVDL);
    pool.release(bufWStage);
    pool.release(bufGradStage);
    pool.release(bufMStage);
    pool.release(bufVStage);
}

}  // namespace ops
}  // namespace grilly
