#include "grilly/ops/snn.h"

#include <cstring>
#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace grilly {
namespace ops {

// ═══════════════════════════════════════════════════════════════════════════
// SNN standalone dispatch ops
//
// Each function follows the standard grilly GPU dispatch pattern:
//   1. Compute buffer sizes from params
//   2. Acquire GPU buffers from the pool
//   3. Upload input data via persistent mapping (single memcpy)
//   4. Get or create pipeline for the shader
//   5. Allocate descriptor set binding buffers to shader bindings
//   6. Record dispatch into CommandBatch and submit
//   7. Download results
//   8. Release buffers back to pool
//
// Push constant structs are memcpy'd directly — their layout matches
// the GLSL layout(push_constant) exactly (uint32/float, tightly packed).
// ═══════════════════════════════════════════════════════════════════════════

// ── LIF neuron step ──────────────────────────────────────────────────────

void lifStep(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
             const float* input, float* vMem, float* tRefrac, float* spikes,
             const LIFParams& p) {
    const size_t bytes = size_t(p.nNeurons) * sizeof(float);

    GrillyBuffer bufInput   = pool.acquire(bytes);
    GrillyBuffer bufVMem    = pool.acquire(bytes);
    GrillyBuffer bufRefrac  = pool.acquire(bytes);
    GrillyBuffer bufSpikes  = pool.acquire(bytes);

    pool.upload(bufInput, input, bytes);
    pool.upload(bufVMem, vMem, bytes);
    pool.upload(bufRefrac, tRefrac, bytes);

    PipelineEntry pipe = cache.getOrCreate("lif-neuron", 4, sizeof(LIFParams));

    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {bufInput.handle,  0, bytes},
        {bufVMem.handle,   0, bytes},
        {bufRefrac.handle, 0, bytes},
        {bufSpikes.handle, 0, bytes},
    };
    VkDescriptorSet descSet = cache.allocDescriptorSet("lif-neuron", bufInfos);

    uint32_t gx = (p.nNeurons + 255) / 256;

    batch.begin();
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, 1, 1,
                   &p, sizeof(p));
    batch.submitDeferred();
    batch.waitForCompletion();

    pool.download(bufVMem, vMem, bytes);
    pool.download(bufRefrac, tRefrac, bytes);
    pool.download(bufSpikes, spikes, bytes);

    pool.release(bufInput);
    pool.release(bufVMem);
    pool.release(bufRefrac);
    pool.release(bufSpikes);
}

// ── SNN node forward ─────────────────────────────────────────────────────

void snnNodeForward(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
                    const float* xIn, float* vMem, float* spikes, float* hOut,
                    const float* tauParam, const SNNNodeForwardParams& p) {
    const size_t bytes = size_t(p.nElements) * sizeof(float);

    GrillyBuffer bufXIn     = pool.acquire(bytes);
    GrillyBuffer bufVMem    = pool.acquire(bytes);
    GrillyBuffer bufSpikes  = pool.acquire(bytes);
    GrillyBuffer bufHOut    = pool.acquire(bytes);
    GrillyBuffer bufTau     = pool.acquire(bytes);

    pool.upload(bufXIn, xIn, bytes);
    pool.upload(bufVMem, vMem, bytes);
    pool.upload(bufTau, tauParam, bytes);

    PipelineEntry pipe = cache.getOrCreate("snn-node-forward", 5,
                                           sizeof(SNNNodeForwardParams));

    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {bufXIn.handle,    0, bytes},
        {bufVMem.handle,   0, bytes},
        {bufSpikes.handle, 0, bytes},
        {bufHOut.handle,   0, bytes},
        {bufTau.handle,    0, bytes},
    };
    VkDescriptorSet descSet = cache.allocDescriptorSet("snn-node-forward",
                                                        bufInfos);

    uint32_t gx = (p.nElements + 255) / 256;

    batch.begin();
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, 1, 1,
                   &p, sizeof(p));
    batch.submitDeferred();
    batch.waitForCompletion();

    pool.download(bufVMem, vMem, bytes);
    pool.download(bufSpikes, spikes, bytes);
    pool.download(bufHOut, hOut, bytes);

    pool.release(bufXIn);
    pool.release(bufVMem);
    pool.release(bufSpikes);
    pool.release(bufHOut);
    pool.release(bufTau);
}

// ── SNN node backward ───────────────────────────────────────────────────

void snnNodeBackward(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
                     const float* gradSpike, const float* hCache,
                     float* gradX, const SNNNodeBackwardParams& p) {
    const size_t bytes = size_t(p.nElements) * sizeof(float);

    GrillyBuffer bufGradSpike = pool.acquire(bytes);
    GrillyBuffer bufHCache    = pool.acquire(bytes);
    GrillyBuffer bufGradX     = pool.acquire(bytes);

    pool.upload(bufGradSpike, gradSpike, bytes);
    pool.upload(bufHCache, hCache, bytes);

    PipelineEntry pipe = cache.getOrCreate("snn-node-backward", 3,
                                           sizeof(SNNNodeBackwardParams));

    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {bufGradSpike.handle, 0, bytes},
        {bufHCache.handle,    0, bytes},
        {bufGradX.handle,     0, bytes},
    };
    VkDescriptorSet descSet = cache.allocDescriptorSet("snn-node-backward",
                                                        bufInfos);

    uint32_t gx = (p.nElements + 255) / 256;

    batch.begin();
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, 1, 1,
                   &p, sizeof(p));
    batch.submitDeferred();
    batch.waitForCompletion();

    pool.download(bufGradX, gradX, bytes);

    pool.release(bufGradSpike);
    pool.release(bufHCache);
    pool.release(bufGradX);
}

// ── Hebbian learning ─────────────────────────────────────────────────────
// 2D dispatch at (16, 16) workgroup: dispatches over (preDim, postDim).

void hebbianLearning(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
                     const float* pre, const float* post, float* weights,
                     const HebbianParams& p) {
    const size_t preBytes    = size_t(p.batchSize) * p.timeSteps * p.preDim * sizeof(float);
    const size_t postBytes   = size_t(p.batchSize) * p.timeSteps * p.postDim * sizeof(float);
    const size_t weightBytes = size_t(p.preDim) * p.postDim * sizeof(float);

    GrillyBuffer bufPre     = pool.acquire(preBytes);
    GrillyBuffer bufPost    = pool.acquire(postBytes);
    GrillyBuffer bufWeights = pool.acquire(weightBytes);

    pool.upload(bufPre, pre, preBytes);
    pool.upload(bufPost, post, postBytes);
    pool.upload(bufWeights, weights, weightBytes);

    PipelineEntry pipe = cache.getOrCreate("hebbian-learning", 3,
                                           sizeof(HebbianParams));

    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {bufPre.handle,     0, preBytes},
        {bufPost.handle,    0, postBytes},
        {bufWeights.handle, 0, weightBytes},
    };
    VkDescriptorSet descSet = cache.allocDescriptorSet("hebbian-learning",
                                                        bufInfos);

    uint32_t gx = (p.preDim + 15) / 16;
    uint32_t gy = (p.postDim + 15) / 16;

    batch.begin();
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, gy, 1,
                   &p, sizeof(p));
    batch.submitDeferred();
    batch.waitForCompletion();

    pool.download(bufWeights, weights, weightBytes);

    pool.release(bufPre);
    pool.release(bufPost);
    pool.release(bufWeights);
}

// ── STDP learning ────────────────────────────────────────────────────────
// Two-pass dispatch: pass 0 updates traces, pass 1 updates weights.

void stdpLearning(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
                  const float* pre, const float* post, float* weights,
                  float* preTrace, float* postTrace, const STDPParams& p) {
    const size_t preBytes      = size_t(p.batchSize) * p.timeSteps * p.preDim * sizeof(float);
    const size_t postBytes     = size_t(p.batchSize) * p.timeSteps * p.postDim * sizeof(float);
    const size_t weightBytes   = size_t(p.preDim) * p.postDim * sizeof(float);
    const size_t preTraceBytes = size_t(p.batchSize) * p.preDim * sizeof(float);
    const size_t postTraceBytes = size_t(p.batchSize) * p.postDim * sizeof(float);

    GrillyBuffer bufPre       = pool.acquire(preBytes);
    GrillyBuffer bufPost      = pool.acquire(postBytes);
    GrillyBuffer bufWeights   = pool.acquire(weightBytes);
    GrillyBuffer bufPreTrace  = pool.acquire(preTraceBytes);
    GrillyBuffer bufPostTrace = pool.acquire(postTraceBytes);

    pool.upload(bufPre, pre, preBytes);
    pool.upload(bufPost, post, postBytes);
    pool.upload(bufWeights, weights, weightBytes);
    pool.upload(bufPreTrace, preTrace, preTraceBytes);
    pool.upload(bufPostTrace, postTrace, postTraceBytes);

    PipelineEntry pipe = cache.getOrCreate("stdp-learning", 5,
                                           sizeof(STDPParams));

    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {bufPre.handle,       0, preBytes},
        {bufPost.handle,      0, postBytes},
        {bufWeights.handle,   0, weightBytes},
        {bufPreTrace.handle,  0, preTraceBytes},
        {bufPostTrace.handle, 0, postTraceBytes},
    };
    VkDescriptorSet descSet = cache.allocDescriptorSet("stdp-learning",
                                                        bufInfos);

    uint32_t gx = (p.preDim + 15) / 16;
    uint32_t gy = (p.postDim + 15) / 16;

    // Two-pass: barriers ensure traces are computed before weight update
    batch.begin();

    // Pass 0: update traces
    STDPParams push0 = p;
    push0.passType = 0;
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, gy, 1,
                   &push0, sizeof(push0));
    batch.barrier();

    // Pass 1: update weights
    STDPParams push1 = p;
    push1.passType = 1;
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, gy, 1,
                   &push1, sizeof(push1));

    batch.submitDeferred();
    batch.waitForCompletion();

    pool.download(bufWeights, weights, weightBytes);
    pool.download(bufPreTrace, preTrace, preTraceBytes);
    pool.download(bufPostTrace, postTrace, postTraceBytes);

    pool.release(bufPre);
    pool.release(bufPost);
    pool.release(bufWeights);
    pool.release(bufPreTrace);
    pool.release(bufPostTrace);
}

// ── Synapse filter ───────────────────────────────────────────────────────
// Simplest SNN op: exponential decay filter y = y * decay + x.
// Only 2 buffers, no output buffer — state is updated in-place.

void synapseFilter(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
                   const float* xIn, float* yState,
                   const SynapseFilterParams& p) {
    const size_t bytes = size_t(p.nElements) * sizeof(float);

    GrillyBuffer bufXIn    = pool.acquire(bytes);
    GrillyBuffer bufYState = pool.acquire(bytes);

    pool.upload(bufXIn, xIn, bytes);
    pool.upload(bufYState, yState, bytes);

    PipelineEntry pipe = cache.getOrCreate("snn-synapse-filter", 2,
                                           sizeof(SynapseFilterParams));

    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {bufXIn.handle,    0, bytes},
        {bufYState.handle, 0, bytes},
    };
    VkDescriptorSet descSet = cache.allocDescriptorSet("snn-synapse-filter",
                                                        bufInfos);

    uint32_t gx = (p.nElements + 255) / 256;

    batch.begin();
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, 1, 1,
                   &p, sizeof(p));
    batch.submitDeferred();
    batch.waitForCompletion();

    pool.download(bufYState, yState, bytes);

    pool.release(bufXIn);
    pool.release(bufYState);
}

// ── GIF neuron step ──────────────────────────────────────────────────────
// Most complex SNN neuron: 8 buffers, 14 push constant fields.

void gifNeuronStep(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
                   const float* input, float* vMem, float* iAdapt,
                   float* gInput, float* gForget, float* tRefrac,
                   float* spikes, float* tLastSpike,
                   const GIFParams& p) {
    const size_t bytes = size_t(p.nNeurons) * sizeof(float);

    GrillyBuffer bufInput      = pool.acquire(bytes);
    GrillyBuffer bufVMem       = pool.acquire(bytes);
    GrillyBuffer bufIAdapt     = pool.acquire(bytes);
    GrillyBuffer bufGInput     = pool.acquire(bytes);
    GrillyBuffer bufGForget    = pool.acquire(bytes);
    GrillyBuffer bufTRefrac    = pool.acquire(bytes);
    GrillyBuffer bufSpikes     = pool.acquire(bytes);
    GrillyBuffer bufTLastSpike = pool.acquire(bytes);

    pool.upload(bufInput, input, bytes);
    pool.upload(bufVMem, vMem, bytes);
    pool.upload(bufIAdapt, iAdapt, bytes);
    pool.upload(bufGInput, gInput, bytes);
    pool.upload(bufGForget, gForget, bytes);
    pool.upload(bufTRefrac, tRefrac, bytes);
    pool.upload(bufTLastSpike, tLastSpike, bytes);

    PipelineEntry pipe = cache.getOrCreate("gif-neuron", 8, sizeof(GIFParams));

    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {bufInput.handle,      0, bytes},
        {bufVMem.handle,       0, bytes},
        {bufIAdapt.handle,     0, bytes},
        {bufGInput.handle,     0, bytes},
        {bufGForget.handle,    0, bytes},
        {bufTRefrac.handle,    0, bytes},
        {bufSpikes.handle,     0, bytes},
        {bufTLastSpike.handle, 0, bytes},
    };
    VkDescriptorSet descSet = cache.allocDescriptorSet("gif-neuron", bufInfos);

    uint32_t gx = (p.nNeurons + 255) / 256;

    batch.begin();
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, 1, 1,
                   &p, sizeof(p));
    batch.submitDeferred();
    batch.waitForCompletion();

    pool.download(bufVMem, vMem, bytes);
    pool.download(bufIAdapt, iAdapt, bytes);
    pool.download(bufGInput, gInput, bytes);
    pool.download(bufGForget, gForget, bytes);
    pool.download(bufTRefrac, tRefrac, bytes);
    pool.download(bufSpikes, spikes, bytes);
    pool.download(bufTLastSpike, tLastSpike, bytes);

    pool.release(bufInput);
    pool.release(bufVMem);
    pool.release(bufIAdapt);
    pool.release(bufGInput);
    pool.release(bufGForget);
    pool.release(bufTRefrac);
    pool.release(bufSpikes);
    pool.release(bufTLastSpike);
}

// ── Event-driven sparse synaptic scatter ─────────────────────────────────

void spikeScatter(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
                  const float* firedIdx, const float* firedCount,
                  const float* weights, float* iAcc,
                  uint32_t nFired, const SpikeScatterParams& p) {
    const size_t accBytes = size_t(p.n) * sizeof(float);
    const size_t wBytes   = size_t(p.n) * p.n * sizeof(float);
    const size_t cntBytes = sizeof(uint32_t);
    // iAcc is pre-zeroed by the caller; with no spikes there is nothing to add.
    if (nFired == 0) return;
    const size_t idxBytes = size_t(nFired) * sizeof(uint32_t);

    GrillyBuffer bufIdx = pool.acquire(idxBytes);
    GrillyBuffer bufCnt = pool.acquire(cntBytes);
    GrillyBuffer bufW   = pool.acquire(wBytes);
    GrillyBuffer bufAcc = pool.acquire(accBytes);

    pool.upload(bufIdx, firedIdx, idxBytes);
    pool.upload(bufCnt, firedCount, cntBytes);
    pool.upload(bufW, weights, wBytes);
    pool.upload(bufAcc, iAcc, accBytes);  // zeroed

    PipelineEntry pipe = cache.getOrCreate("spike-scatter", 4,
                                           sizeof(SpikeScatterParams));

    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {bufIdx.handle, 0, idxBytes},
        {bufCnt.handle, 0, cntBytes},
        {bufW.handle,   0, wBytes},
        {bufAcc.handle, 0, accBytes},
    };
    VkDescriptorSet descSet = cache.allocDescriptorSet("spike-scatter", bufInfos);

    uint32_t gx = static_cast<uint32_t>((uint64_t(p.n) + 255) / 256);

    batch.begin();
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, 1, 1,
                   &p, sizeof(p));
    batch.submitDeferred();
    batch.waitForCompletion();

    pool.download(bufAcc, iAcc, accBytes);

    pool.release(bufIdx);
    pool.release(bufCnt);
    pool.release(bufW);
    pool.release(bufAcc);
}

// ── Resident-weight benchmark loop ───────────────────────────────────────

void residentBench(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
                   uint32_t mode, const float* firedIdx,
                   const float* firedCount, const float* spikes,
                   const float* weights, float* iAcc,
                   uint32_t nFired, uint32_t n, uint32_t iters,
                   uint32_t batched) {
    const size_t accBytes = size_t(n) * sizeof(float);
    const size_t wBytes   = size_t(n) * n * sizeof(float);
    const size_t spkBytes = size_t(n) * sizeof(float);
    const size_t cntBytes = sizeof(uint32_t);
    const size_t idxBytes = size_t(nFired ? nFired : 1) * sizeof(uint32_t);

    GrillyBuffer bufIdx = pool.acquire(idxBytes);
    GrillyBuffer bufCnt = pool.acquire(cntBytes);
    GrillyBuffer bufSpk = pool.acquire(spkBytes);
    GrillyBuffer bufW   = pool.acquire(wBytes);
    GrillyBuffer bufAcc = pool.acquire(accBytes);

    // Upload everything ONCE — W stays resident across all iters.
    if (nFired) pool.upload(bufIdx, firedIdx, size_t(nFired) * sizeof(uint32_t));
    pool.upload(bufCnt, firedCount, cntBytes);
    pool.upload(bufSpk, spikes, spkBytes);
    pool.upload(bufW, weights, wBytes);
    pool.upload(bufAcc, iAcc, accBytes);

    SpikeScatterParams p{n};
    const char* shader = (mode == 0) ? "spike-scatter" : "synapse-dense";
    uint32_t nBuf = (mode == 0) ? 4u : 3u;

    PipelineEntry pipe = cache.getOrCreate(shader, nBuf, sizeof(p));
    std::vector<VkDescriptorBufferInfo> bufInfos;
    if (mode == 0) {
        bufInfos = {
            {bufIdx.handle, 0, idxBytes},
            {bufCnt.handle, 0, cntBytes},
            {bufW.handle,   0, wBytes},
            {bufAcc.handle, 0, accBytes},
        };
    } else {
        bufInfos = {
            {bufSpk.handle, 0, spkBytes},
            {bufW.handle,   0, wBytes},
            {bufAcc.handle, 0, accBytes},
        };
    }
    VkDescriptorSet descSet = cache.allocDescriptorSet(shader, bufInfos);

    uint32_t gx = static_cast<uint32_t>((uint64_t(n) + 255) / 256);

    if (batched) {
        // One submit for all iters; compute barrier serializes timesteps.
        batch.begin();
        for (uint32_t it = 0; it < iters; ++it) {
            batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, 1, 1,
                           &p, sizeof(p));
            batch.barrier();
        }
        batch.submitDeferred();
        batch.waitForCompletion();
    } else {
        // One submit + host wait per iter (per-step overhead floor).
        for (uint32_t it = 0; it < iters; ++it) {
            batch.begin();
            batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, 1, 1,
                           &p, sizeof(p));
            batch.submitDeferred();
            batch.waitForCompletion();
        }
    }

    pool.download(bufAcc, iAcc, accBytes);

    pool.release(bufIdx);
    pool.release(bufCnt);
    pool.release(bufSpk);
    pool.release(bufW);
    pool.release(bufAcc);
}

// ── Batched event-driven propagation (production primitive) ──────────────

void spikePropagateBatch(CommandBatch& batch, BufferPool& pool,
                         PipelineCache& cache, const float* firedIdx,
                         const float* firedOffsets, const float* firedCounts,
                         const float* weights, float* out, const float* firedVals,
                         uint32_t nFiredTotal, uint32_t nIn, uint32_t nOut,
                         uint32_t M) {
    const size_t idxBytes = size_t(nFiredTotal ? nFiredTotal : 1) * sizeof(uint32_t);
    const size_t valBytes = size_t(nFiredTotal ? nFiredTotal : 1) * sizeof(float);
    const size_t offBytes = size_t(M) * sizeof(uint32_t);
    const size_t cntBytes = size_t(M) * sizeof(uint32_t);
    const size_t wBytes   = size_t(nIn) * nOut * sizeof(float);
    const size_t outBytes = size_t(M) * nOut * sizeof(float);

    // DEVICE_LOCAL compute buffers (cached VRAM — the shader reads these).
    GrillyBuffer bufIdx = pool.acquireDeviceLocal(idxBytes);
    GrillyBuffer bufOff = pool.acquireDeviceLocal(offBytes);
    GrillyBuffer bufCnt = pool.acquireDeviceLocal(cntBytes);
    GrillyBuffer bufW   = pool.acquireDeviceLocal(wBytes);
    GrillyBuffer bufOut = pool.acquireDeviceLocal(outBytes);
    GrillyBuffer bufVal = pool.acquireDeviceLocal(valBytes);

    // Stage-IN (CPU writes -> WC memory, fast sequential memcpy).
    GrillyBuffer stgIdx = pool.acquire(idxBytes);
    GrillyBuffer stgOff = pool.acquire(offBytes);
    GrillyBuffer stgCnt = pool.acquire(cntBytes);
    GrillyBuffer stgW   = pool.acquire(wBytes);
    GrillyBuffer stgVal = pool.acquire(valBytes);
    // Stage-OUT (CPU reads -> MUST be HOST_CACHED, else readback is ~25 MB/s).
    GrillyBuffer stgOut = pool.acquireReadback(outBytes);

    using clk = std::chrono::high_resolution_clock;
    const bool prof = std::getenv("PPB_PROF") != nullptr;
    auto t0 = clk::now();

    if (nFiredTotal) {
        pool.upload(stgIdx, firedIdx, size_t(nFiredTotal) * sizeof(uint32_t));
        pool.upload(stgVal, firedVals, size_t(nFiredTotal) * sizeof(float));
    }
    pool.upload(stgOff, firedOffsets, offBytes);
    pool.upload(stgCnt, firedCounts, cntBytes);
    pool.upload(stgW, weights, wBytes);
    auto t1 = clk::now();

    SpikePropagateBatchParams p{nOut, M};
    PipelineEntry pipe = cache.getOrCreate("spike-propagate-batch", 6,
                                           sizeof(SpikePropagateBatchParams));
    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {bufIdx.handle, 0, idxBytes},
        {bufOff.handle, 0, offBytes},
        {bufCnt.handle, 0, cntBytes},
        {bufW.handle,   0, wBytes},
        {bufOut.handle, 0, outBytes},
        {bufVal.handle, 0, valBytes},
    };
    VkDescriptorSet descSet =
        cache.allocDescriptorSet("spike-propagate-batch", bufInfos);
    auto t2 = clk::now();

    uint32_t gx = static_cast<uint32_t>((uint64_t(M) * nOut + 255) / 256);

    // Single command buffer: stage-in DMA -> barrier -> compute -> barrier -> stage-out DMA
    batch.begin();
    if (nFiredTotal) {
        batch.copyBuffer(stgIdx, bufIdx, idxBytes);
        batch.copyBuffer(stgVal, bufVal, valBytes);
    }
    batch.copyBuffer(stgOff, bufOff, offBytes);
    batch.copyBuffer(stgCnt, bufCnt, cntBytes);
    batch.copyBuffer(stgW,   bufW,   wBytes);
    batch.transferComputeBarrier();
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, 1, 1,
                   &p, sizeof(p));
    batch.transferComputeBarrier();
    batch.copyBuffer(bufOut, stgOut, outBytes);
    batch.submitDeferred();
    batch.waitForCompletion();
    auto t3 = clk::now();

    pool.download(stgOut, out, outBytes);  // HOST_CACHED ~7 GB/s
    auto t4 = clk::now();

    if (prof) {
        auto ms = [](auto a, auto b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        std::fprintf(stderr,
            "[PPB] M=%u nIn=%u nOut=%u nFired=%u | upload=%.3f desc=%.3f "
            "compute+dma=%.3f download=%.3f total=%.3f ms\n",
            M, nIn, nOut, nFiredTotal, ms(t0, t1), ms(t1, t2),
            ms(t2, t3), ms(t3, t4), ms(t0, t4));
    }

    pool.release(bufIdx);
    pool.release(bufOff);
    pool.release(bufCnt);
    pool.release(bufW);
    pool.release(bufOut);
    pool.release(bufVal);
    pool.release(stgIdx);
    pool.release(stgOff);
    pool.release(stgCnt);
    pool.release(stgW);
    pool.release(stgVal);
    pool.release(stgOut);
}

}  // namespace ops
}  // namespace grilly
