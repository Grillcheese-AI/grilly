#include "grilly/ops/bandit.h"

#include <vector>
#include <stdexcept>

namespace grilly {
namespace ops {

void banditSolve(CommandBatch& batch, BufferPool& pool,
                 PipelineCache& cache,
                 const float* muHat, const float* N,
                 float* targetW, uint32_t* stopFlags,
                 const BanditParams& p) {
    const size_t muBytes = size_t(p.nArms) * p.nInstances * sizeof(float);
    const size_t stopBytes = size_t(p.nInstances) * sizeof(uint32_t);

    GrillyBuffer bMu_DL = pool.acquireDeviceLocal(muBytes);
    GrillyBuffer bN_DL  = pool.acquireDeviceLocal(muBytes);
    GrillyBuffer bW_DL  = pool.acquireDeviceLocal(muBytes);
    GrillyBuffer bS_DL  = pool.acquireDeviceLocal(stopBytes);

    GrillyBuffer bMu_Stage = pool.acquire(muBytes);
    GrillyBuffer bN_Stage  = pool.acquire(muBytes);
    GrillyBuffer bW_Stage  = pool.acquireReadback(muBytes);
    GrillyBuffer bS_Stage  = pool.acquireReadback(stopBytes);

    pool.upload(bMu_Stage, muHat, muBytes);
    pool.upload(bN_Stage,  N,     muBytes);

    PipelineEntry pipe = cache.getOrCreate("bandit-solve", 4, sizeof(BanditParams));

    std::vector<VkDescriptorBufferInfo> bufs = {
        {bMu_DL.handle, 0, muBytes},
        {bN_DL.handle,  0, muBytes},
        {bW_DL.handle,  0, muBytes},
        {bS_DL.handle,  0, stopBytes},
    };
    VkDescriptorSet descSet = cache.allocDescriptorSet("bandit-solve", bufs);

    uint32_t gx = (p.nInstances + 63u) / 64u;

    batch.begin();
    batch.copyBuffer(bMu_Stage, bMu_DL, muBytes);
    batch.copyBuffer(bN_Stage,  bN_DL,  muBytes);
    batch.transferComputeBarrier();

    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, 1, 1, &p, sizeof(p));

    batch.transferComputeBarrier();
    batch.copyBuffer(bW_DL, bW_Stage, muBytes);
    batch.copyBuffer(bS_DL, bS_Stage, stopBytes);
    batch.submitDeferred();
    batch.waitForCompletion();

    pool.download(bW_Stage, targetW, muBytes);
    pool.download(bS_Stage, (float*)stopFlags, stopBytes); // download use float* but handle uint32* cast

    pool.release(bMu_DL); pool.release(bN_DL); pool.release(bW_DL); pool.release(bS_DL);
    pool.release(bMu_Stage); pool.release(bN_Stage); pool.release(bW_Stage); pool.release(bS_Stage);
}

}  // namespace ops
}  // namespace grilly
