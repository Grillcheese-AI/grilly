#include "grilly/ops/mingru.h"

#include <vector>
#include <stdexcept>

namespace grilly {
namespace ops {

void minGruForward(CommandBatch& batch, BufferPool& pool,
                   PipelineCache& cache,
                   const float* G, const float* V, const float* D,
                   float* H, const MinGruParams& p) {
    const size_t elemBytes = size_t(p.batchSize) * p.seqLen * p.hiddenDim * sizeof(float);

    // DEVICE_LOCAL compute buffers
    GrillyBuffer bG_DL = pool.acquireDeviceLocal(elemBytes);
    GrillyBuffer bV_DL = pool.acquireDeviceLocal(elemBytes);
    GrillyBuffer bD_DL = pool.acquireDeviceLocal(elemBytes);
    GrillyBuffer bH_DL = pool.acquireDeviceLocal(elemBytes);

    // Staging
    GrillyBuffer bG_Stage = pool.acquire(elemBytes);
    GrillyBuffer bV_Stage = pool.acquire(elemBytes);
    GrillyBuffer bD_Stage = pool.acquire(elemBytes);
    GrillyBuffer bH_Stage = pool.acquireReadback(elemBytes);

    pool.upload(bG_Stage, G, elemBytes);
    pool.upload(bV_Stage, V, elemBytes);
    pool.upload(bD_Stage, D, elemBytes);

    PipelineEntry pipe = cache.getOrCreate("mingru-forward", 4, 2 * sizeof(uint32_t));

    std::vector<VkDescriptorBufferInfo> bufs = {
        {bG_DL.handle, 0, elemBytes},
        {bV_DL.handle, 0, elemBytes},
        {bD_DL.handle, 0, elemBytes},
        {bH_DL.handle, 0, elemBytes},
    };
    VkDescriptorSet descSet = cache.allocDescriptorSet("mingru-forward", bufs);

    struct Push {
        uint32_t seqLen;
        uint32_t hiddenDim;
    } push = {p.seqLen, p.hiddenDim};

    uint32_t gx = (p.hiddenDim + 63u) / 64u;
    uint32_t gy = p.batchSize;

    batch.begin();
    batch.copyBuffer(bG_Stage, bG_DL, elemBytes);
    batch.copyBuffer(bV_Stage, bV_DL, elemBytes);
    batch.copyBuffer(bD_Stage, bD_DL, elemBytes);
    batch.transferComputeBarrier();

    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, gy, 1, &push, sizeof(push));

    batch.transferComputeBarrier();
    batch.copyBuffer(bH_DL, bH_Stage, elemBytes);
    batch.submitDeferred();
    batch.waitForCompletion();

    pool.download(bH_Stage, H, elemBytes);

    pool.release(bG_DL); pool.release(bV_DL); pool.release(bD_DL); pool.release(bH_DL);
    pool.release(bG_Stage); pool.release(bV_Stage); pool.release(bD_Stage); pool.release(bH_Stage);
}

void minGruBackward(CommandBatch& batch, BufferPool& pool,
                    PipelineCache& cache,
                    const float* gradH, const float* G, const float* V, const float* D,
                    const float* H,
                    float* gradG, float* gradV, float* gradD,
                    const MinGruParams& p) {
    const size_t elemBytes = size_t(p.batchSize) * p.seqLen * p.hiddenDim * sizeof(float);

    // Inputs (5 DEVICE_LOCAL)
    GrillyBuffer bGH_DL = pool.acquireDeviceLocal(elemBytes);
    GrillyBuffer bG_DL  = pool.acquireDeviceLocal(elemBytes);
    GrillyBuffer bV_DL  = pool.acquireDeviceLocal(elemBytes);
    GrillyBuffer bD_DL  = pool.acquireDeviceLocal(elemBytes);
    GrillyBuffer bH_DL  = pool.acquireDeviceLocal(elemBytes);

    // Outputs (3 DEVICE_LOCAL)
    GrillyBuffer bGG_DL = pool.acquireDeviceLocal(elemBytes);
    GrillyBuffer bGV_DL = pool.acquireDeviceLocal(elemBytes);
    GrillyBuffer bGD_DL = pool.acquireDeviceLocal(elemBytes);

    // Staging
    GrillyBuffer bGH_Stage = pool.acquire(elemBytes);
    GrillyBuffer bG_Stage  = pool.acquire(elemBytes);
    GrillyBuffer bV_Stage  = pool.acquire(elemBytes);
    GrillyBuffer bD_Stage  = pool.acquire(elemBytes);
    GrillyBuffer bH_Stage  = pool.acquire(elemBytes);
    
    GrillyBuffer bGG_Stage = pool.acquireReadback(elemBytes);
    GrillyBuffer bGV_Stage = pool.acquireReadback(elemBytes);
    GrillyBuffer bGD_Stage = pool.acquireReadback(elemBytes);

    pool.upload(bGH_Stage, gradH, elemBytes);
    pool.upload(bG_Stage,  G,     elemBytes);
    pool.upload(bV_Stage,  V,     elemBytes);
    pool.upload(bD_Stage,  D,     elemBytes);
    pool.upload(bH_Stage,  H,     elemBytes);

    PipelineEntry pipe = cache.getOrCreate("mingru-backward", 8, 2 * sizeof(uint32_t));

    std::vector<VkDescriptorBufferInfo> bufs = {
        {bGH_DL.handle, 0, elemBytes},
        {bG_DL.handle,  0, elemBytes},
        {bV_DL.handle,  0, elemBytes},
        {bD_DL.handle,  0, elemBytes},
        {bH_DL.handle,  0, elemBytes},
        {bGG_DL.handle, 0, elemBytes},
        {bGV_DL.handle, 0, elemBytes},
        {bGD_DL.handle, 0, elemBytes},
    };
    VkDescriptorSet descSet = cache.allocDescriptorSet("mingru-backward", bufs);

    struct Push {
        uint32_t seqLen;
        uint32_t hiddenDim;
    } push = {p.seqLen, p.hiddenDim};

    uint32_t gx = (p.hiddenDim + 63u) / 64u;
    uint32_t gy = p.batchSize;

    batch.begin();
    batch.copyBuffer(bGH_Stage, bGH_DL, elemBytes);
    batch.copyBuffer(bG_Stage, bG_DL, elemBytes);
    batch.copyBuffer(bV_Stage, bV_DL, elemBytes);
    batch.copyBuffer(bD_Stage, bD_DL, elemBytes);
    batch.copyBuffer(bH_Stage, bH_DL, elemBytes);
    batch.transferComputeBarrier();

    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, gy, 1, &push, sizeof(push));

    batch.transferComputeBarrier();
    batch.copyBuffer(bGG_DL, bGG_Stage, elemBytes);
    batch.copyBuffer(bGV_DL, bGV_Stage, elemBytes);
    batch.copyBuffer(bGD_DL, bGD_Stage, elemBytes);
    batch.submitDeferred();
    batch.waitForCompletion();

    pool.download(bGG_Stage, gradG, elemBytes);
    pool.download(bGV_Stage, gradV, elemBytes);
    pool.download(bGD_Stage, gradD, elemBytes);

    pool.release(bGH_DL); pool.release(bG_DL); pool.release(bV_DL); pool.release(bD_DL); pool.release(bH_DL);
    pool.release(bGG_DL); pool.release(bGV_DL); pool.release(bGD_DL);
    pool.release(bGH_Stage); pool.release(bG_Stage); pool.release(bV_Stage); pool.release(bD_Stage); pool.release(bH_Stage);
    pool.release(bGG_Stage); pool.release(bGV_Stage); pool.release(bGD_Stage);
}

}  // namespace ops
}  // namespace grilly
