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

}  // namespace ops
}  // namespace grilly
