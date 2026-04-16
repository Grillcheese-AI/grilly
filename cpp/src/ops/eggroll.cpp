#include "grilly/ops/eggroll.h"

#include <vector>

namespace grilly {
namespace ops {

void eggrollGenerate(CommandBatch& batch, BufferPool& pool,
                    PipelineCache& cache,
                    float* U, float* V,
                    const EggrollGenParams& p) {
    const size_t uBytes = size_t(p.dOut) * p.nWorkers * sizeof(float);
    const size_t vBytes = size_t(p.dIn) * p.nWorkers * sizeof(float);

    GrillyBuffer bU_DL = pool.acquireDeviceLocal(uBytes);
    GrillyBuffer bV_DL = pool.acquireDeviceLocal(vBytes);
    GrillyBuffer bU_Read = pool.acquireReadback(uBytes);
    GrillyBuffer bV_Read = pool.acquireReadback(vBytes);

    PipelineEntry pipe = cache.getOrCreate("eggroll-generate", 2, sizeof(EggrollGenParams));
    
    std::vector<VkDescriptorBufferInfo> bufs = {
        {bU_DL.handle, 0, uBytes},
        {bV_DL.handle, 0, vBytes},
    };
    VkDescriptorSet descSet = cache.allocDescriptorSet("eggroll-generate", bufs);

    uint32_t total = (p.dOut + p.dIn) * p.nWorkers;
    uint32_t gx = (total + 63u) / 64u;

    batch.begin();
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, 1, 1, &p, sizeof(p));
    batch.transferComputeBarrier();
    batch.copyBuffer(bU_DL, bU_Read, uBytes);
    batch.copyBuffer(bV_DL, bV_Read, vBytes);
    batch.submitDeferred();
    batch.waitForCompletion();

    pool.download(bU_Read, U, uBytes);
    pool.download(bV_Read, V, vBytes);

    pool.release(bU_DL); pool.release(bV_DL);
    pool.release(bU_Read); pool.release(bV_Read);
}

void eggrollUpdate(CommandBatch& batch, BufferPool& pool,
                  PipelineCache& cache,
                  float* W, float* Merit,
                  const float* U, const float* V,
                  const uint32_t* topIdx, const float* topWeights,
                  const EggrollUpdateParams& p) {
    const size_t wBytes = size_t(p.dOut) * p.dIn * sizeof(float);
    const size_t uBytes = size_t(p.dOut) * p.nWorkers * sizeof(float);
    const size_t vBytes = size_t(p.dIn) * p.nWorkers * sizeof(float);
    const size_t idxBytes = size_t(p.topK) * sizeof(uint32_t);
    const size_t fitBytes = size_t(p.topK) * sizeof(float);

    GrillyBuffer bW_DL = pool.acquireDeviceLocal(wBytes);
    GrillyBuffer bM_DL = pool.acquireDeviceLocal(wBytes);
    GrillyBuffer bU_DL = pool.acquireDeviceLocal(uBytes);
    GrillyBuffer bV_DL = pool.acquireDeviceLocal(vBytes);
    GrillyBuffer bIdx_DL = pool.acquireDeviceLocal(idxBytes);
    GrillyBuffer bFit_DL = pool.acquireDeviceLocal(fitBytes);

    GrillyBuffer bW_Stage = pool.acquire(wBytes);
    GrillyBuffer bM_Stage = pool.acquire(wBytes);
    GrillyBuffer bU_Stage = pool.acquire(uBytes);
    GrillyBuffer bV_Stage = pool.acquire(vBytes);
    GrillyBuffer bIdx_Stage = pool.acquire(idxBytes);
    GrillyBuffer bFit_Stage = pool.acquire(fitBytes);

    pool.upload(bW_Stage, W, wBytes);
    pool.upload(bM_Stage, Merit, wBytes);
    pool.upload(bU_Stage, U, uBytes);
    pool.upload(bV_Stage, V, vBytes);
    pool.upload(bIdx_Stage, (float*)topIdx, idxBytes); // cast for upload API
    pool.upload(bFit_Stage, topWeights, fitBytes);

    PipelineEntry pipe = cache.getOrCreate("eggroll-update", 6, sizeof(EggrollUpdateParams));
    
    std::vector<VkDescriptorBufferInfo> bufs = {
        {bW_DL.handle, 0, wBytes},
        {bM_DL.handle, 0, wBytes},
        {bU_DL.handle, 0, uBytes},
        {bV_DL.handle, 0, vBytes},
        {bIdx_DL.handle, 0, idxBytes},
        {bFit_DL.handle, 0, fitBytes},
    };
    VkDescriptorSet descSet = cache.allocDescriptorSet("eggroll-update", bufs);

    batch.begin();
    batch.copyBuffer(bW_Stage, bW_DL, wBytes);
    batch.copyBuffer(bM_Stage, bM_DL, wBytes);
    batch.copyBuffer(bU_Stage, bU_DL, uBytes);
    batch.copyBuffer(bV_Stage, bV_DL, vBytes);
    batch.copyBuffer(bIdx_Stage, bIdx_DL, idxBytes);
    batch.copyBuffer(bFit_Stage, bFit_DL, fitBytes);
    batch.transferComputeBarrier();

    uint32_t gx = (p.dIn + 15u) / 16u;
    uint32_t gy = (p.dOut + 15u) / 16u;
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, gy, 1, &p, sizeof(p));

    batch.transferComputeBarrier();
    batch.copyBuffer(bW_DL, bW_Stage, wBytes);
    batch.copyBuffer(bM_DL, bM_Stage, wBytes);
    batch.submitDeferred();
    batch.waitForCompletion();

    pool.download(bW_Stage, W, wBytes);
    pool.download(bM_Stage, Merit, wBytes);

    pool.release(bW_DL); pool.release(bM_DL); pool.release(bU_DL); pool.release(bV_DL);
    pool.release(bIdx_DL); pool.release(bFit_DL);
    pool.release(bW_Stage); pool.release(bM_Stage); pool.release(bU_Stage); pool.release(bV_Stage);
    pool.release(bIdx_Stage); pool.release(bFit_Stage);
}

}  // namespace ops
}  // namespace grilly
