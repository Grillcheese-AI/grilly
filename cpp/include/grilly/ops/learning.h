#pragma once

#include <cstdint>

#include "grilly/buffer_pool.h"
#include "grilly/command_batch.h"
#include "grilly/pipeline_cache.h"

namespace grilly {
namespace ops {

// ═══════════════════════════════════════════════════════════════════════════
// Learning / auxiliary ops — SSM, Fisher info, EWC, NLMS, spike bridges
// ═══════════════════════════════════════════════════════════════════════════

// ── SSM fused math (Mamba-style selective scan) ──────────────────────────
// Shader: ssm-fused-math.spv
// Buffers: gate(0), value(1), decay(2), scan_out(3)

struct SSMFusedParams {
    uint32_t batchSize;
    uint32_t seqLen;
    uint32_t features;
    float minDecay;
    float maxDecay;
};

void ssmFusedMath(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
                  const float* gate, const float* value, const float* decay,
                  float* scanOut, const SSMFusedParams& p);

// ── Fisher information diagonal ──────────────────────────────────────────
// Shader: fisher-info.spv
// Buffers: grads(0), fisher(1) rw

struct FisherInfoParams {
    uint32_t numParams;
    float momentum;
    uint32_t useEMA;
    uint32_t resetFisher;
};

void fisherInfoUpdate(CommandBatch& batch, BufferPool& pool,
                      PipelineCache& cache,
                      const float* grads, float* fisher,
                      const FisherInfoParams& p);

// ── EWC penalty ──────────────────────────────────────────────────────────
// Shader: fisher-ewc-penalty.spv
// Buffers: current_params(0), optimal_params(1), fisher(2), penalty(3)

struct EWCPenaltyParams {
    uint32_t numParams;
    float lambda;
};

void ewcPenalty(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
                const float* currentParams, const float* optimalParams,
                const float* fisher, float* penalty,
                const EWCPenaltyParams& p);

// ── NLMS predict ─────────────────────────────────────────────────────────
// Shader: nlms-predict.spv
// Buffers: x(0), w(1), bias(2), y_pred(3)

struct NLMSPredictParams {
    uint32_t batchSize;
    uint32_t nFeatures;
};

void nlmsPredict(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
                 const float* x, const float* w, const float* bias,
                 float* yPred, const NLMSPredictParams& p);

// ── NLMS update ──────────────────────────────────────────────────────────
// Shader: nlms-update.spv
// Buffers: x(0), y_pred(1), y_true(2), w(3) rw, bias(4) rw,
//          mu(5) rw, error(6) write

struct NLMSUpdateParams {
    uint32_t nFeatures;
    float muDecay;
    float muMin;
    float biasLrScale;
    float epsilon;
};

void nlmsUpdate(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
                const float* x, const float* yPred, const float* yTrue,
                float* w, float* bias, float* mu, float* error,
                const NLMSUpdateParams& p);

// ── Continuous-to-spike bridge ───────────────────────────────────────────
// Shader: bridge-continuous-to-spike.spv
// 2-pass: project → encode
// Buffers: features(0), spikes(1), W(2), b(3), random(4), temp(5)

struct ContinuousToSpikeParams {
    uint32_t batchSize;
    uint32_t numTimesteps;
    uint32_t inputDim;
    uint32_t spikeDim;
    uint32_t encodingType;   // 0=Poisson, 1=Temporal
    uint32_t useProjection;
    uint32_t passType;
};

void continuousToSpikes(CommandBatch& batch, BufferPool& pool,
                        PipelineCache& cache,
                        const float* features, float* spikes,
                        const float* W, const float* b,
                        const float* random, float* temp,
                        const ContinuousToSpikeParams& p);

// ── Spike-to-continuous bridge ───────────────────────────────────────────
// Shader: bridge-spike-to-continuous.spv
// 2-pass: encode → project
// Buffers: spikes(0), features(1), weights(2), W(3), b(4), temp(5)

struct SpikeToContinuousParams {
    uint32_t batchSize;
    uint32_t totalTime;
    uint32_t spikeDim;
    uint32_t outputDim;
    uint32_t timeWindow;
    uint32_t encodingType;   // 0=Rate, 1=Temporal, 2=Phase
    uint32_t useProjection;
    uint32_t passType;
};

void spikesToContinuous(CommandBatch& batch, BufferPool& pool,
                        PipelineCache& cache,
                        const float* spikes, float* features,
                        const float* weights, const float* W,
                        const float* b, float* temp,
                        const SpikeToContinuousParams& p);

}  // namespace ops
}  // namespace grilly
