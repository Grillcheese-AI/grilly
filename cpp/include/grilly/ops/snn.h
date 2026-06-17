#pragma once

#include <cstdint>

#include "grilly/buffer_pool.h"
#include "grilly/command_batch.h"
#include "grilly/pipeline_cache.h"

namespace grilly {
namespace ops {

// ═══════════════════════════════════════════════════════════════════════════
// SNN standalone dispatch ops
//
// These are standalone GPU dispatch functions that mirror the legacy Python
// backend.snn module. They dispatch pre-compiled SPIR-V shaders via the
// standard acquire → upload → dispatch → download → release pattern.
//
// The push constant structs match the GLSL layout(push_constant) exactly.
// ═══════════════════════════════════════════════════════════════════════════

// ── LIF neuron step ──────────────────────────────────────────────────────
// Shader: lif-neuron.spv
// local_size: (256, 1, 1)
// Buffers: input(0), v_mem(1) rw, t_refrac(2) rw, spikes(3) write

struct LIFParams {
    uint32_t nNeurons;
    float dt;
    float tauMem;
    float vRest;
    float vReset;
    float vThresh;
    float rMem;
    float tRefracPeriod;
};

void lifStep(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
             const float* input, float* vMem, float* tRefrac, float* spikes,
             const LIFParams& p);

// ── SNN node forward ─────────────────────────────────────────────────────
// Shader: snn-node-forward.spv
// local_size: (256, 1, 1)
// Buffers: x_in(0), v_mem(1) rw, spikes(2), h_out(3), tau_param(4)

struct SNNNodeForwardParams {
    uint32_t nElements;
    uint32_t neuronType;  // 0=IF, 1=LIF, 2=PLIF
    float tau;
    float vThreshold;
    float vReset;
    uint32_t resetMode;    // 0=subtract, 1=zero
    uint32_t decayInput;   // 0=no, 1=yes
};

void snnNodeForward(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
                    const float* xIn, float* vMem, float* spikes, float* hOut,
                    const float* tauParam, const SNNNodeForwardParams& p);

// ── SNN node backward ───────────────────────────────────────────────────
// Shader: snn-node-backward.spv
// local_size: (256, 1, 1)
// Buffers: grad_spike(0), h_cache(1), grad_x(2)

struct SNNNodeBackwardParams {
    uint32_t nElements;
    float alpha;
    uint32_t surrogateType;  // 0=ATan, 1=Sigmoid, 2=FastSigmoid
    float vThreshold;
};

void snnNodeBackward(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
                     const float* gradSpike, const float* hCache,
                     float* gradX, const SNNNodeBackwardParams& p);

// ── Hebbian learning ─────────────────────────────────────────────────────
// Shader: hebbian-learning.spv
// local_size: (16, 16, 1)
// Buffers: pre(0), post(1), weights(2) rw

struct HebbianParams {
    uint32_t batchSize;
    uint32_t timeSteps;
    uint32_t preDim;
    uint32_t postDim;
    float learningRate;
    float weightDecay;
};

void hebbianLearning(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
                     const float* pre, const float* post, float* weights,
                     const HebbianParams& p);

// ── STDP learning ────────────────────────────────────────────────────────
// Shader: stdp-learning.spv
// local_size: (16, 16, 1)
// Buffers: pre(0), post(1), weights(2) rw, pre_trace(3) rw, post_trace(4) rw
// Two-pass: pass 0 = update traces, pass 1 = update weights

struct STDPParams {
    uint32_t batchSize;
    uint32_t timeSteps;
    uint32_t preDim;
    uint32_t postDim;
    float lrPotentiation;
    float lrDepression;
    float traceDecay;
    uint32_t passType;
};

void stdpLearning(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
                  const float* pre, const float* post, float* weights,
                  float* preTrace, float* postTrace, const STDPParams& p);

// ── Synapse filter ───────────────────────────────────────────────────────
// Shader: snn-synapse-filter.spv
// local_size: (256, 1, 1)
// Buffers: x_in(0), y_state(1) rw

struct SynapseFilterParams {
    uint32_t nElements;
    float decay;
};

void synapseFilter(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
                   const float* xIn, float* yState,
                   const SynapseFilterParams& p);

// ── GIF neuron step ──────────────────────────────────────────────────────
// Shader: gif-neuron.spv
// local_size: (256, 1, 1)
// Buffers: input(0), v_mem(1) rw, i_adapt(2) rw, g_input(3) rw,
//          g_forget(4) rw, t_refrac(5) rw, spikes(6) write,
//          t_last_spike(7) rw

struct GIFParams {
    uint32_t nNeurons;
    float dt;
    float currentTime;
    float tauMem;
    float vRest;
    float vReset;
    float vThresh;
    float rMem;
    float tauAdapt;
    float deltaAdapt;
    float bAdapt;
    float tauGate;
    float gateStrength;
    float tRefracPeriod;
};

void gifNeuronStep(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
                   const float* input, float* vMem, float* iAdapt,
                   float* gInput, float* gForget, float* tRefrac,
                   float* spikes, float* tLastSpike,
                   const GIFParams& p);

// ── Event-driven sparse synaptic scatter ─────────────────────────────────
// Shader: spike-scatter.spv
// local_size: (256, 1, 1)
// Buffers: fired_idx(0) uint, fired_count(1) uint, weights(2), i_acc(3) rw
// Grid = ceil(nFired*N / 256). firedIdx/firedCount are uint32 data passed as
// raw float* bytes (upload is a plain memcpy). iAcc must be pre-zeroed.

struct SpikeScatterParams {
    uint32_t n;  // post dimension (matches GLSL push_constant {uint N;})
};

void spikeScatter(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
                  const float* firedIdx, const float* firedCount,
                  const float* weights, float* iAcc,
                  uint32_t nFired, const SpikeScatterParams& p);

// ── Resident-weight benchmark loop ───────────────────────────────────────
// Uploads W (and inputs) ONCE, then dispatches the chosen kernel `iters` times
// into the resident buffers. Models the real SNN loop (W fixed, spikes stream).
// mode 0 = spike-scatter (gather over fired list), 1 = synapse-dense (full GEMV).
// Caller times the whole call and divides by iters for per-step resident cost.

void residentBench(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
                   uint32_t mode, const float* firedIdx,
                   const float* firedCount, const float* spikes,
                   const float* weights, float* iAcc,
                   uint32_t nFired, uint32_t n, uint32_t iters,
                   uint32_t batched);

// ── Batched event-driven propagation (production primitive) ──────────────
// Shader: spike-propagate-batch.spv. Propagates M spike vectors through one
// resident (N_in x N_out) W in a single dispatch/submit.
// out[m,post] = sum over fired pre of vector m of W[pre*N_out+post].
// fired_idx is the concatenation of all vectors' fired indices; firedOffsets[M]
// and firedCounts[M] slice it. uint inputs passed as raw float* bytes.

struct SpikePropagateBatchParams {
    uint32_t nOut;
    uint32_t m;
};

void spikePropagateBatch(CommandBatch& batch, BufferPool& pool,
                         PipelineCache& cache, const float* firedIdx,
                         const float* firedOffsets, const float* firedCounts,
                         const float* weights, float* out, const float* firedVals,
                         uint32_t nFiredTotal, uint32_t nIn, uint32_t nOut,
                         uint32_t M);

}  // namespace ops
}  // namespace grilly
