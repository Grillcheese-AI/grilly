#include "grilly/ops/ternary.h"

#include <vector>

namespace grilly {
namespace ops {

// ── Ternary weight-only GEMM (BitNet b1.58), multiply-free ───────────────
// DEVICE_LOCAL + staging (per linear.cpp): compute reads must hit cached VRAM
// or RDNA2 runs them at ~0.05 GB/s. 2 stage-in (activations, packed weights),
// 1 stage-out (output), one submit.

void ternaryGemm(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
                 const float* activations, const uint32_t* weightsPacked,
                 float* output, uint32_t wordsPerRow,
                 const TernaryGemmParams& p) {
    const size_t actBytes = size_t(p.M) * p.K * sizeof(float);
    const size_t wBytes   = size_t(p.N) * wordsPerRow * sizeof(uint32_t);
    const size_t outBytes = size_t(p.M) * p.N * sizeof(float);

    GrillyBuffer bufADL = pool.acquireDeviceLocal(actBytes);
    GrillyBuffer bufWDL = pool.acquireDeviceLocal(wBytes);
    GrillyBuffer bufODL = pool.acquireDeviceLocal(outBytes);

    GrillyBuffer bufAStage = pool.acquire(actBytes);
    GrillyBuffer bufWStage = pool.acquire(wBytes);
    GrillyBuffer bufOStage = pool.acquireReadback(outBytes);

    pool.upload(bufAStage, activations, actBytes);
    pool.upload(bufWStage, reinterpret_cast<const float*>(weightsPacked), wBytes);

    PipelineEntry pipe = cache.getOrCreate("ternary-gemm", 3,
                                           sizeof(TernaryGemmParams));

    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {bufADL.handle, 0, actBytes},
        {bufWDL.handle, 0, wBytes},
        {bufODL.handle, 0, outBytes},
    };
    VkDescriptorSet descSet = cache.allocDescriptorSet("ternary-gemm", bufInfos);

    // 16x16 workgroup over (N, M) — matches the shader's local_size and the
    // int8-gemm dispatch convention (x=N col, y=M row).
    const uint32_t gx = (p.N + 15u) / 16u;
    const uint32_t gy = (p.M + 15u) / 16u;

    batch.begin();
    batch.copyBuffer(bufAStage, bufADL, actBytes);
    batch.copyBuffer(bufWStage, bufWDL, wBytes);
    batch.transferComputeBarrier();
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, gy, 1,
                   &p, sizeof(p));
    batch.transferComputeBarrier();
    batch.copyBuffer(bufODL, bufOStage, outBytes);
    batch.submitDeferred();
    batch.waitForCompletion();

    pool.download(bufOStage, output, outBytes);

    pool.release(bufADL); pool.release(bufWDL); pool.release(bufODL);
    pool.release(bufAStage); pool.release(bufWStage); pool.release(bufOStage);
}

}  // namespace ops
}  // namespace grilly
