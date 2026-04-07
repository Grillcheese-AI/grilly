#include "grilly/ops/prefix_scan.h"

#include <cstring>
#include <stdexcept>
#include <vector>

namespace grilly {
namespace ops {

// Uses the staging pattern (DEVICE_LOCAL compute + WC stage-in + HOST_CACHED
// stage-out) — same as linear.cpp. See linear.cpp for the rationale.

void prefixScanCausal(CommandBatch& batch, BufferPool& pool,
                      PipelineCache& cache,
                      const float* x, const float* a, float* h,
                      const PrefixScanParams& p) {
    const size_t elemBytes = size_t(p.batchSize) * p.seqLen * p.hiddenDim *
                             sizeof(float);

    // DEVICE_LOCAL compute buffers
    GrillyBuffer bufXDL = pool.acquireDeviceLocal(elemBytes);
    GrillyBuffer bufADL = pool.acquireDeviceLocal(elemBytes);
    GrillyBuffer bufHDL = pool.acquireDeviceLocal(elemBytes);

    // Host staging
    GrillyBuffer bufXStage = pool.acquire(elemBytes);
    GrillyBuffer bufAStage = pool.acquire(elemBytes);
    GrillyBuffer bufHStage = pool.acquireReadback(elemBytes);

    pool.upload(bufXStage, x, elemBytes);
    pool.upload(bufAStage, a, elemBytes);

    PipelineEntry pipe = cache.getOrCreate(
        "prefix-scan-causal", 3, 2 * sizeof(uint32_t));

    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {bufXDL.handle, 0, elemBytes},
        {bufADL.handle, 0, elemBytes},
        {bufHDL.handle, 0, elemBytes},
    };
    VkDescriptorSet descSet =
        cache.allocDescriptorSet("prefix-scan-causal", bufInfos);

    struct Push {
        uint32_t seqLen;
        uint32_t hiddenDim;
    } push = {p.seqLen, p.hiddenDim};

    // One workgroup per (hidden_dim, batch) pair. Local size = 32 threads
    // (one per time step) — shader enforces seqLen <= 32.
    uint32_t gx = p.hiddenDim;
    uint32_t gy = p.batchSize;

    batch.begin();

    batch.copyBuffer(bufXStage, bufXDL, elemBytes);
    batch.copyBuffer(bufAStage, bufADL, elemBytes);
    batch.transferComputeBarrier();

    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, gy, 1,
                   &push, sizeof(push));

    batch.transferComputeBarrier();
    batch.copyBuffer(bufHDL, bufHStage, elemBytes);
    batch.submitDeferred();
    batch.waitForCompletion();

    pool.download(bufHStage, h, elemBytes);

    pool.release(bufXDL);
    pool.release(bufADL);
    pool.release(bufHDL);
    pool.release(bufXStage);
    pool.release(bufAStage);
    pool.release(bufHStage);
}

void prefixScanCausalBackward(CommandBatch& batch, BufferPool& pool,
                              PipelineCache& cache,
                              const float* dh, const float* a,
                              const float* h, const float* x,
                              float* dx, float* da,
                              const PrefixScanParams& p) {
    const size_t elemBytes = size_t(p.batchSize) * p.seqLen * p.hiddenDim *
                             sizeof(float);

    // DEVICE_LOCAL compute buffers (4 inputs, 2 outputs)
    GrillyBuffer bufDhDL = pool.acquireDeviceLocal(elemBytes);
    GrillyBuffer bufADL  = pool.acquireDeviceLocal(elemBytes);
    GrillyBuffer bufHDL  = pool.acquireDeviceLocal(elemBytes);
    GrillyBuffer bufXDL  = pool.acquireDeviceLocal(elemBytes);
    GrillyBuffer bufDxDL = pool.acquireDeviceLocal(elemBytes);
    GrillyBuffer bufDaDL = pool.acquireDeviceLocal(elemBytes);

    // Staging
    GrillyBuffer bufDhStage = pool.acquire(elemBytes);
    GrillyBuffer bufAStage  = pool.acquire(elemBytes);
    GrillyBuffer bufHStage  = pool.acquire(elemBytes);
    GrillyBuffer bufXStage  = pool.acquire(elemBytes);
    GrillyBuffer bufDxStage = pool.acquireReadback(elemBytes);
    GrillyBuffer bufDaStage = pool.acquireReadback(elemBytes);

    pool.upload(bufDhStage, dh, elemBytes);
    pool.upload(bufAStage,  a,  elemBytes);
    pool.upload(bufHStage,  h,  elemBytes);
    pool.upload(bufXStage,  x,  elemBytes);

    PipelineEntry pipe = cache.getOrCreate(
        "prefix-scan-causal-backward", 6, 2 * sizeof(uint32_t));

    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {bufDhDL.handle, 0, elemBytes},
        {bufADL.handle,  0, elemBytes},
        {bufHDL.handle,  0, elemBytes},
        {bufXDL.handle,  0, elemBytes},
        {bufDxDL.handle, 0, elemBytes},
        {bufDaDL.handle, 0, elemBytes},
    };
    VkDescriptorSet descSet = cache.allocDescriptorSet(
        "prefix-scan-causal-backward", bufInfos);

    struct Push {
        uint32_t seqLen;
        uint32_t hiddenDim;
    } push = {p.seqLen, p.hiddenDim};

    uint32_t gx = p.hiddenDim;
    uint32_t gy = p.batchSize;

    batch.begin();

    batch.copyBuffer(bufDhStage, bufDhDL, elemBytes);
    batch.copyBuffer(bufAStage,  bufADL,  elemBytes);
    batch.copyBuffer(bufHStage,  bufHDL,  elemBytes);
    batch.copyBuffer(bufXStage,  bufXDL,  elemBytes);
    batch.transferComputeBarrier();

    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, gy, 1,
                   &push, sizeof(push));

    batch.transferComputeBarrier();
    batch.copyBuffer(bufDxDL, bufDxStage, elemBytes);
    batch.copyBuffer(bufDaDL, bufDaStage, elemBytes);
    batch.submitDeferred();
    batch.waitForCompletion();

    pool.download(bufDxStage, dx, elemBytes);
    pool.download(bufDaStage, da, elemBytes);

    pool.release(bufDhDL);
    pool.release(bufADL);
    pool.release(bufHDL);
    pool.release(bufXDL);
    pool.release(bufDxDL);
    pool.release(bufDaDL);
    pool.release(bufDhStage);
    pool.release(bufAStage);
    pool.release(bufHStage);
    pool.release(bufXStage);
    pool.release(bufDxStage);
    pool.release(bufDaStage);
}

}  // namespace ops
}  // namespace grilly
