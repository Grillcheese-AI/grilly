#include "grilly/ops/linear.h"

#include <cstring>
#include <stdexcept>

namespace grilly {
namespace ops {

// ── GPU linear (port of fnn.py:1823-1976) ───────────────────────────────────
//
// In the Python backend, linear() makes ~12 ctypes FFI calls:
//   acquire buffer × 4, upload × 3, get_or_create_pipeline, get_descriptor_set,
//   dispatch_compute (which internally does: reset cmd, begin, bind pipeline,
//   bind descriptors, push constants, dispatch, end, submit, wait fence),
//   download, release × 4.
//
// Here ALL of that is native C++ — zero Python crossings. The CommandBatch
// records everything into a single command buffer submission.

void linear(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
            const float* x, const float* weights, const float* bias,
            float* output, const LinearParams& p) {
    // ── Buffer sizes ──
    const size_t inputBytes  = size_t(p.batchSeq) * p.inputDim * sizeof(float);
    const size_t weightBytes = size_t(p.outputDim) * p.inputDim * sizeof(float);
    const size_t biasBytes   = p.hasBias ? size_t(p.outputDim) * sizeof(float)
                                         : sizeof(float);  // dummy
    const size_t outputBytes = size_t(p.batchSeq) * p.outputDim * sizeof(float);

    // ── Acquire buffers (bucket-rounded, persistent mapping) ──
    GrillyBuffer bufInput   = pool.acquire(inputBytes);
    GrillyBuffer bufWeights = pool.acquire(weightBytes);
    GrillyBuffer bufBias    = pool.acquire(biasBytes);
    GrillyBuffer bufOutput  = pool.acquire(outputBytes);

    // ── Upload via persistent mapping (single memcpy each, no vkMap/vkUnmap) ──
    pool.upload(bufInput, x, inputBytes);
    pool.upload(bufWeights, weights, weightBytes);
    if (p.hasBias && bias) {
        pool.upload(bufBias, bias, p.outputDim * sizeof(float));
    }

    // ── Get or create pipeline (4 buffers, 16 bytes push constants) ──
    PipelineEntry pipe = cache.getOrCreate("fnn-linear", 4, 16);

    // ── Allocate descriptor set (LRU cached) ──
    std::vector<VkDescriptorBufferInfo> bufferInfos(4);
    bufferInfos[0] = {bufInput.handle,   0, inputBytes};
    bufferInfos[1] = {bufWeights.handle, 0, weightBytes};
    bufferInfos[2] = {bufBias.handle,    0, biasBytes};
    bufferInfos[3] = {bufOutput.handle,  0, outputBytes};

    VkDescriptorSet descSet = cache.allocDescriptorSet("fnn-linear", bufferInfos);

    // ── Push constants: batch_seq, input_dim, output_dim, has_bias ──
    // Matches fnn-linear.glsl layout (4 × uint32 = 16 bytes).
    // Python packs these via struct.pack("IIII", ...) — we just memcpy the struct.
    LinearParams pushData = p;

    // ── Dispatch ──
    // 2D workgroups at 16×16 (must match fnn-linear.glsl local_size)
    uint32_t gx = (p.outputDim + 15) / 16;
    uint32_t gy = (p.batchSeq + 15) / 16;

    batch.begin();
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, gy, 1,
                   &pushData, sizeof(pushData));
    batch.submit();

    // ── Download result (persistent mapping — single memcpy, no vkMap) ──
    pool.download(bufOutput, output, outputBytes);

    // ── Release buffers back to pool ──
    pool.release(bufInput);
    pool.release(bufWeights);
    pool.release(bufBias);
    pool.release(bufOutput);
}

// ── CPU reference using Eigen (for correctness verification) ────────────────
//
// Eigen::Map wraps raw float* without copying, then the matrix multiply
// compiles to optimized SIMD (AVX/SSE) via Eigen's expression templates.
// This gives us a high-quality CPU baseline to verify GPU results against.

std::vector<float> linearCPU(const float* x, const float* weights,
                             const float* bias, const LinearParams& p) {
    using Eigen::Map;
    using Eigen::MatrixXf;
    using Eigen::RowMajor;
    using RowMajorMap = Map<const Eigen::Matrix<float, Eigen::Dynamic,
                                                Eigen::Dynamic, RowMajor>>;

    // Map input matrices (zero-copy views over the raw pointers)
    RowMajorMap xMat(x, p.batchSeq, p.inputDim);
    RowMajorMap wMat(weights, p.outputDim, p.inputDim);

    // output = x @ W^T  (Eigen handles the transpose internally)
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, RowMajor> result =
        xMat * wMat.transpose();

    // Add bias if present
    if (p.hasBias && bias) {
        Map<const Eigen::VectorXf> bVec(bias, p.outputDim);
        result.rowwise() += bVec.transpose();
    }

    // Copy to output vector
    std::vector<float> out(p.batchSeq * p.outputDim);
    std::memcpy(out.data(), result.data(), out.size() * sizeof(float));
    return out;
}

// ── GPU linear backward ──────────────────────────────────────────────────
//
// 3-pass dispatch using the same "fnn-linear-backward" shader:
//   Pass 0: grad_input = grad_output @ W     (reverse of x @ W^T)
//   Pass 1: grad_weight = grad_output^T @ x  (outer product accumulation)
//   Pass 2: grad_bias = sum(grad_output, dim=0)
//
// 6 buffers: grad_output, input, weights, grad_input, grad_weight, grad_bias.
// Workgroups: 2D at (16,16) for passes 0 and 1, 1D for pass 2.

void linearBackward(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
                    const float* gradOutput, const float* input,
                    const float* weights,
                    float* gradInput, float* gradWeight, float* gradBias,
                    const LinearParams& p) {
    const size_t gradOutBytes  = size_t(p.batchSeq) * p.outputDim * sizeof(float);
    const size_t inputBytes    = size_t(p.batchSeq) * p.inputDim * sizeof(float);
    const size_t weightBytes   = size_t(p.outputDim) * p.inputDim * sizeof(float);
    const size_t gradInBytes   = inputBytes;
    const size_t gradWBytes    = weightBytes;
    const size_t gradBiasBytes = size_t(p.outputDim) * sizeof(float);

    GrillyBuffer bufGradOut  = pool.acquire(gradOutBytes);
    GrillyBuffer bufInput    = pool.acquire(inputBytes);
    GrillyBuffer bufWeights  = pool.acquire(weightBytes);
    GrillyBuffer bufGradIn   = pool.acquire(gradInBytes);
    GrillyBuffer bufGradW    = pool.acquire(gradWBytes);
    GrillyBuffer bufGradBias = pool.acquire(gradBiasBytes);

    pool.upload(bufGradOut, gradOutput, gradOutBytes);
    pool.upload(bufInput, input, inputBytes);
    pool.upload(bufWeights, weights, weightBytes);

    // Zero grad outputs
    std::vector<float> zerosIn(p.batchSeq * p.inputDim, 0.0f);
    std::vector<float> zerosW(p.outputDim * p.inputDim, 0.0f);
    std::vector<float> zerosB(p.outputDim, 0.0f);
    pool.upload(bufGradIn, zerosIn.data(), gradInBytes);
    pool.upload(bufGradW, zerosW.data(), gradWBytes);
    pool.upload(bufGradBias, zerosB.data(), gradBiasBytes);

    LinearBackwardParams bwdParams{p.batchSeq, p.inputDim, p.outputDim, 0};

    PipelineEntry pipe = cache.getOrCreate("fnn-linear-backward", 6,
                                           sizeof(LinearBackwardParams));

    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {bufGradOut.handle,  0, gradOutBytes},
        {bufInput.handle,    0, inputBytes},
        {bufWeights.handle,  0, weightBytes},
        {bufGradIn.handle,   0, gradInBytes},
        {bufGradW.handle,    0, gradWBytes},
        {bufGradBias.handle, 0, gradBiasBytes},
    };
    VkDescriptorSet descSet = cache.allocDescriptorSet("fnn-linear-backward",
                                                        bufInfos);

    batch.begin();

    // Pass 0: grad_input
    bwdParams.passType = 0;
    uint32_t gx0 = (p.inputDim + 15) / 16;
    uint32_t gy0 = (p.batchSeq + 15) / 16;
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx0, gy0, 1,
                   &bwdParams, sizeof(bwdParams));
    batch.barrier();

    // Pass 1: grad_weight
    bwdParams.passType = 1;
    uint32_t gx1 = (p.inputDim + 15) / 16;
    uint32_t gy1 = (p.outputDim + 15) / 16;
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx1, gy1, 1,
                   &bwdParams, sizeof(bwdParams));
    batch.barrier();

    // Pass 2: grad_bias
    bwdParams.passType = 2;
    uint32_t gx2 = (p.outputDim + 255) / 256;
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx2, 1, 1,
                   &bwdParams, sizeof(bwdParams));

    batch.submit();

    pool.download(bufGradIn, gradInput, gradInBytes);
    pool.download(bufGradW, gradWeight, gradWBytes);
    pool.download(bufGradBias, gradBias, gradBiasBytes);

    pool.release(bufGradOut);
    pool.release(bufInput);
    pool.release(bufWeights);
    pool.release(bufGradIn);
    pool.release(bufGradW);
    pool.release(bufGradBias);
}

// ── GPU dropout ──────────────────────────────────────────────────────────

void dropout(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
             const float* input, const float* randomMask, float* output,
             uint32_t totalElements, float dropoutProb, bool isTraining) {
    const size_t bytes = size_t(totalElements) * sizeof(float);

    GrillyBuffer bufInput  = pool.acquire(bytes);
    GrillyBuffer bufRandom = pool.acquire(bytes);
    GrillyBuffer bufOutput = pool.acquire(bytes);

    pool.upload(bufInput, input, bytes);
    pool.upload(bufRandom, randomMask, bytes);

    PipelineEntry pipe = cache.getOrCreate("fnn-dropout", 3,
                                           sizeof(DropoutParams));

    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {bufInput.handle,  0, bytes},
        {bufRandom.handle, 0, bytes},
        {bufOutput.handle, 0, bytes},
    };
    VkDescriptorSet descSet = cache.allocDescriptorSet("fnn-dropout", bufInfos);

    DropoutParams push{totalElements, dropoutProb, isTraining ? 1u : 0u};
    uint32_t gx = (totalElements + 255) / 256;

    batch.begin();
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, 1, 1,
                   &push, sizeof(push));
    batch.submit();

    pool.download(bufOutput, output, bytes);

    pool.release(bufInput);
    pool.release(bufRandom);
    pool.release(bufOutput);
}

}  // namespace ops
}  // namespace grilly
