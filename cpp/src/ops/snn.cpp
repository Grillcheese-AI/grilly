#include "grilly/ops/snn.h"

#include <cstring>

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

}  // namespace ops
}  // namespace grilly
