#include "grilly/ops/linear.h"

#include <cstring>
#include <stdexcept>

namespace grilly {
namespace ops {

// ── GPU linear with explicit DEVICE_LOCAL + staging pattern ────────────────
//
// On AMD/Windows even with Resizable BAR enabled, the DEVICE_LOCAL +
// HOST_VISIBLE memory type that VMA selects for ``BufferPool::acquire``
// lands in WC-mapped memory that bypasses the GPU's L2 cache. Compute
// kernels reading from it run at ~0.05 GB/s — slower than a SATA SSD,
// roughly 0.04% of theoretical VRAM bandwidth (432 GB/s on RX 6750 XT).
// See sandbox/vsa_lm/grilly_gpu_path_test.py for the smoking-gun profile.
//
// The fix: compute buffers go through ``acquireDeviceLocal`` (DEVICE_LOCAL
// only, full cached VRAM, ~432 GB/s), and we move data in/out via small
// staging buffers from the regular pool. The staging buffers are slow for
// GPU compute reads but fine for ``vkCmdCopyBuffer`` transfers, which use
// the GPU's dedicated DMA engine and run at PCIe speed (~25 GB/s).
//
// All 3 staging-in copies, the compute dispatch, and the 1 staging-out
// copy are batched into a single command buffer with a single submit/wait,
// so the dispatch overhead is unchanged from the old fast-path.

void linear(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
            const float* x, const float* weights, const float* bias,
            float* output, const LinearParams& p) {
    // ── Buffer sizes ──
    const size_t inputBytes  = size_t(p.batchSeq) * p.inputDim * sizeof(float);
    const size_t weightBytes = size_t(p.outputDim) * p.inputDim * sizeof(float);
    const size_t biasBytes   = p.hasBias ? size_t(p.outputDim) * sizeof(float)
                                         : sizeof(float);  // dummy
    const size_t outputBytes = size_t(p.batchSeq) * p.outputDim * sizeof(float);

    // ── Acquire DEVICE_LOCAL compute buffers (cached VRAM, fast GPU access) ──
    GrillyBuffer bufInputDL   = pool.acquireDeviceLocal(inputBytes);
    GrillyBuffer bufWeightsDL = pool.acquireDeviceLocal(weightBytes);
    GrillyBuffer bufBiasDL    = pool.acquireDeviceLocal(biasBytes);
    GrillyBuffer bufOutputDL  = pool.acquireDeviceLocal(outputBytes);

    // ── Acquire host-visible staging buffers ──
    // Stage-IN buffers (CPU writes only): WC memory is fast for sequential
    // memcpy at ~9 GB/s — pool.acquire() is the right choice.
    GrillyBuffer bufInputStage   = pool.acquire(inputBytes);
    GrillyBuffer bufWeightsStage = pool.acquire(weightBytes);
    GrillyBuffer bufBiasStage    = pool.acquire(biasBytes);
    // Stage-OUT buffer (CPU reads from it): MUST be HOST_CACHED random-read
    // memory. WC memory is uncached on the CPU side and a 19 MB readback
    // memcpy ran at ~25 MB/s (749 ms — slower than the 9 ms GPU compute!).
    // HOST_CACHED via acquireReadback gives ~7 GB/s for the same memcpy.
    GrillyBuffer bufOutputStage  = pool.acquireReadback(outputBytes);

    // ── memcpy CPU → staging (no GPU sync needed, persistent mapping) ──
    pool.upload(bufInputStage, x, inputBytes);
    pool.upload(bufWeightsStage, weights, weightBytes);
    if (p.hasBias && bias) {
        pool.upload(bufBiasStage, bias, p.outputDim * sizeof(float));
    }

    // ── Get or create pipeline (4 buffers, 16 bytes push constants) ──
    PipelineEntry pipe = cache.getOrCreate("fnn-linear", 4, 16);

    // ── Allocate descriptor set bound to DEVICE_LOCAL buffers ──
    // The descriptor cache keys on (shader_name, [(buffer.handle, range)]),
    // so as long as the pool returns stable handles for repeated bucket
    // requests (LIFO), this hits across calls.
    std::vector<VkDescriptorBufferInfo> bufferInfos(4);
    bufferInfos[0] = {bufInputDL.handle,   0, inputBytes};
    bufferInfos[1] = {bufWeightsDL.handle, 0, weightBytes};
    bufferInfos[2] = {bufBiasDL.handle,    0, biasBytes};
    bufferInfos[3] = {bufOutputDL.handle,  0, outputBytes};

    VkDescriptorSet descSet = cache.allocDescriptorSet("fnn-linear", bufferInfos);

    LinearParams pushData = p;

    uint32_t gx = (p.outputDim + 15) / 16;
    uint32_t gy = (p.batchSeq + 15) / 16;

    // ── Single command buffer: stage-in → barrier → compute → barrier → stage-out ──
    batch.begin();

    // Stage-in: DMA copy host-visible staging → DEVICE_LOCAL VRAM
    batch.copyBuffer(bufInputStage,   bufInputDL,   inputBytes);
    batch.copyBuffer(bufWeightsStage, bufWeightsDL, weightBytes);
    if (p.hasBias && bias) {
        batch.copyBuffer(bufBiasStage, bufBiasDL, p.outputDim * sizeof(float));
    }

    // Barrier: TRANSFER_WRITE → SHADER_READ
    batch.transferComputeBarrier();

    // Compute on DEVICE_LOCAL buffers (full ~432 GB/s VRAM bandwidth)
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, gy, 1,
                   &pushData, sizeof(pushData));

    // Barrier: SHADER_WRITE → TRANSFER_READ
    batch.transferComputeBarrier();

    // Stage-out: DMA copy DEVICE_LOCAL → host-visible HOST_CACHED staging
    batch.copyBuffer(bufOutputDL, bufOutputStage, outputBytes);

    batch.submitDeferred();
    batch.waitForCompletion();

    // ── memcpy staging → CPU output (HOST_CACHED, ~7 GB/s) ──
    pool.download(bufOutputStage, output, outputBytes);

    // ── Release buffers back to their respective pools ──
    pool.release(bufInputDL);
    pool.release(bufWeightsDL);
    pool.release(bufBiasDL);
    pool.release(bufOutputDL);
    pool.release(bufInputStage);
    pool.release(bufWeightsStage);
    pool.release(bufBiasStage);
    pool.release(bufOutputStage);
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

    // Staging pattern: 3 stage-in (gradOut, input, weights),
    // 3 stage-out (gradIn, gradW, gradBias). All compute on DEVICE_LOCAL.
    GrillyBuffer bufGradOutDL  = pool.acquireDeviceLocal(gradOutBytes);
    GrillyBuffer bufInputDL    = pool.acquireDeviceLocal(inputBytes);
    GrillyBuffer bufWeightsDL  = pool.acquireDeviceLocal(weightBytes);
    GrillyBuffer bufGradInDL   = pool.acquireDeviceLocal(gradInBytes);
    GrillyBuffer bufGradWDL    = pool.acquireDeviceLocal(gradWBytes);
    GrillyBuffer bufGradBiasDL = pool.acquireDeviceLocal(gradBiasBytes);

    GrillyBuffer bufGradOutStage = pool.acquire(gradOutBytes);
    GrillyBuffer bufInputStage   = pool.acquire(inputBytes);
    GrillyBuffer bufWeightsStage = pool.acquire(weightBytes);
    GrillyBuffer bufGradInStage   = pool.acquireReadback(gradInBytes);
    GrillyBuffer bufGradWStage    = pool.acquireReadback(gradWBytes);
    GrillyBuffer bufGradBiasStage = pool.acquireReadback(gradBiasBytes);

    pool.upload(bufGradOutStage, gradOutput, gradOutBytes);
    pool.upload(bufInputStage,   input, inputBytes);
    pool.upload(bufWeightsStage, weights, weightBytes);

    // The grad output buffers must start at zero — pass 1 (grad_weight) and
    // pass 2 (grad_bias) accumulate via atomic adds in the shader. We zero
    // them on the GPU side via vkCmdFillBuffer rather than uploading zeros
    // through staging (which was the old code path).
    // (Workaround: upload zeros to a small temporary stage and copy. The
    // simpler path: keep the upload-zeros-via-stage approach since we need
    // to reset every call.)
    std::vector<float> zerosIn(p.batchSeq * p.inputDim, 0.0f);
    std::vector<float> zerosW(p.outputDim * p.inputDim, 0.0f);
    std::vector<float> zerosB(p.outputDim, 0.0f);
    // Reuse the readback stage buffers as upload-zeros source: they're
    // host-visible (HOST_CACHED), CPU-write is fine even though it's not
    // optimal for sequential write — total bytes is small relative to GPU
    // compute. Upload then DMA copy in the command buffer.
    pool.upload(bufGradInStage,   zerosIn.data(), gradInBytes);
    pool.upload(bufGradWStage,    zerosW.data(),  gradWBytes);
    pool.upload(bufGradBiasStage, zerosB.data(),  gradBiasBytes);

    LinearBackwardParams bwdParams{p.batchSeq, p.inputDim, p.outputDim, 0};

    PipelineEntry pipe = cache.getOrCreate("fnn-linear-backward", 6,
                                           sizeof(LinearBackwardParams));

    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {bufGradOutDL.handle,  0, gradOutBytes},
        {bufInputDL.handle,    0, inputBytes},
        {bufWeightsDL.handle,  0, weightBytes},
        {bufGradInDL.handle,   0, gradInBytes},
        {bufGradWDL.handle,    0, gradWBytes},
        {bufGradBiasDL.handle, 0, gradBiasBytes},
    };
    VkDescriptorSet descSet = cache.allocDescriptorSet("fnn-linear-backward",
                                                        bufInfos);

    batch.begin();

    // Stage-in: copy all 6 staging buffers (3 inputs + 3 zeroed grads) to DL
    batch.copyBuffer(bufGradOutStage, bufGradOutDL, gradOutBytes);
    batch.copyBuffer(bufInputStage,   bufInputDL,   inputBytes);
    batch.copyBuffer(bufWeightsStage, bufWeightsDL, weightBytes);
    batch.copyBuffer(bufGradInStage,   bufGradInDL,   gradInBytes);
    batch.copyBuffer(bufGradWStage,    bufGradWDL,    gradWBytes);
    batch.copyBuffer(bufGradBiasStage, bufGradBiasDL, gradBiasBytes);

    batch.transferComputeBarrier();

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

    batch.transferComputeBarrier();

    // Stage-out: copy 3 grad buffers from DL → HOST_CACHED readback staging
    batch.copyBuffer(bufGradInDL,   bufGradInStage,   gradInBytes);
    batch.copyBuffer(bufGradWDL,    bufGradWStage,    gradWBytes);
    batch.copyBuffer(bufGradBiasDL, bufGradBiasStage, gradBiasBytes);

    batch.submitDeferred();
    batch.waitForCompletion();

    pool.download(bufGradInStage,   gradInput,  gradInBytes);
    pool.download(bufGradWStage,    gradWeight, gradWBytes);
    pool.download(bufGradBiasStage, gradBias,   gradBiasBytes);

    pool.release(bufGradOutDL);
    pool.release(bufInputDL);
    pool.release(bufWeightsDL);
    pool.release(bufGradInDL);
    pool.release(bufGradWDL);
    pool.release(bufGradBiasDL);
    pool.release(bufGradOutStage);
    pool.release(bufInputStage);
    pool.release(bufWeightsStage);
    pool.release(bufGradInStage);
    pool.release(bufGradWStage);
    pool.release(bufGradBiasStage);
}

// ── GPU dropout ──────────────────────────────────────────────────────────

void dropout(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
             const float* input, const float* randomMask, float* output,
             uint32_t totalElements, float dropoutProb, bool isTraining) {
    const size_t bytes = size_t(totalElements) * sizeof(float);

    // Staging pattern: 2 stage-in (input, randomMask), 1 stage-out (output)
    GrillyBuffer bufInputDL  = pool.acquireDeviceLocal(bytes);
    GrillyBuffer bufRandomDL = pool.acquireDeviceLocal(bytes);
    GrillyBuffer bufOutputDL = pool.acquireDeviceLocal(bytes);

    GrillyBuffer bufInputStage  = pool.acquire(bytes);
    GrillyBuffer bufRandomStage = pool.acquire(bytes);
    GrillyBuffer bufOutputStage = pool.acquireReadback(bytes);

    pool.upload(bufInputStage,  input,      bytes);
    pool.upload(bufRandomStage, randomMask, bytes);

    PipelineEntry pipe = cache.getOrCreate("fnn-dropout", 3,
                                           sizeof(DropoutParams));

    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {bufInputDL.handle,  0, bytes},
        {bufRandomDL.handle, 0, bytes},
        {bufOutputDL.handle, 0, bytes},
    };
    VkDescriptorSet descSet = cache.allocDescriptorSet("fnn-dropout", bufInfos);

    DropoutParams push{totalElements, dropoutProb, isTraining ? 1u : 0u};
    uint32_t gx = (totalElements + 255) / 256;

    batch.begin();
    batch.copyBuffer(bufInputStage,  bufInputDL,  bytes);
    batch.copyBuffer(bufRandomStage, bufRandomDL, bytes);
    batch.transferComputeBarrier();
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, 1, 1,
                   &push, sizeof(push));
    batch.transferComputeBarrier();
    batch.copyBuffer(bufOutputDL, bufOutputStage, bytes);
    batch.submitDeferred();
    batch.waitForCompletion();

    pool.download(bufOutputStage, output, bytes);

    pool.release(bufInputDL);
    pool.release(bufRandomDL);
    pool.release(bufOutputDL);
    pool.release(bufInputStage);
    pool.release(bufRandomStage);
    pool.release(bufOutputStage);
}

}  // namespace ops
}  // namespace grilly
