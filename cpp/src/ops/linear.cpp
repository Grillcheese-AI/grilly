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
            const void* x, const void* weights, const void* bias,
            void* output, const LinearParams& p) {
    // ── Byte sizes (dynamic — fp32 or fp16 determined by p.elemSize) ──
    const uint32_t inElem  = p.elemSize;          // 2 for fp16, 4 for fp32
    const size_t inputBytes  = size_t(p.batchSeq) * p.inputDim  * inElem;
    const size_t weightBytes = size_t(p.outputDim) * p.inputDim * inElem;
    // Bias is ALWAYS fp32 regardless of input dtype. The fp32 bias matches
    // both fnn-linear's 3rd binding and gemm-bias-add's accumulator, and
    // bias is small enough (outputDim floats) that the bandwidth cost of
    // fp32 vs fp16 is negligible.
    const size_t biasBytes   = p.hasBias ? size_t(p.outputDim) * sizeof(float)
                                         : sizeof(float);  // dummy
    // The output is ALWAYS fp32 regardless of input dtype — coopmat
    // accumulator runs in fp32 for numerical stability, and fnn-linear
    // also writes fp32. The Python binding converts back to fp16 if
    // requested by the caller's dtype.
    const size_t outputBytes = size_t(p.batchSeq) * p.outputDim * sizeof(float);

    // ── Shader selection ──
    // Coopmat requirements:
    //   - fp16 input (elemSize == 2)
    //   - device exposes VK_KHR_cooperative_matrix
    //   - the compiled SPIR-V is loaded in the pipeline cache
    //   - shape aligned to the shader's tile (M%16, K%16, N%64)
    const bool shapeAligned =
        (p.batchSeq  % 16u == 0u) &&
        (p.inputDim  % 16u == 0u) &&
        (p.outputDim % 64u == 0u);
    const bool useCoopMat =
        inElem == 2u &&
        cache.getDevice().hasCooperativeMatrix() &&
        cache.hasShader("gemm-coopmat-shared") &&
        shapeAligned;

    // fp16 input without a coopmat path is not supported in this function —
    // the fallback fnn-linear shader is fp32-only. Callers must either use
    // fp32 input or run on a device that supports cooperative matrix.
    if (inElem == 2u && !useCoopMat) {
        throw std::runtime_error(
            "linear(): fp16 input requested but cooperative matrix path is "
            "unavailable (missing device support, shader, or shape "
            "alignment — M%16, K%16, N%64 required).");
    }

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

    // ── memcpy CPU → staging (raw bytes, dtype-agnostic for x/weights) ──
    pool.upload(bufInputStage,
                reinterpret_cast<const float*>(x), inputBytes);
    pool.upload(bufWeightsStage,
                reinterpret_cast<const float*>(weights), weightBytes);
    if (p.hasBias && bias) {
        // Bias is always fp32 (see biasBytes computation above).
        pool.upload(bufBiasStage,
                    reinterpret_cast<const float*>(bias),
                    size_t(p.outputDim) * sizeof(float));
    }

    // ── Get or create pipeline ──
    const std::string shaderName = useCoopMat ? "gemm-coopmat-shared"
                                               : "fnn-linear";
    // gemm-coopmat-shared has 3 bindings (A, B, C); push constants = 16 bytes
    // ({M,K,N,transpose_b}). fnn-linear has 4 bindings; push 16 bytes too.
    const uint32_t numBindings = useCoopMat ? 3u : 4u;
    const uint32_t pushSize    = 16u;
    PipelineEntry pipe = cache.getOrCreate(shaderName, numBindings, pushSize);

    // ── Allocate descriptor set ──
    std::vector<VkDescriptorBufferInfo> bufferInfos;
    if (useCoopMat) {
        bufferInfos = {
            {bufInputDL.handle,   0, inputBytes},
            {bufWeightsDL.handle, 0, weightBytes},
            {bufOutputDL.handle,  0, outputBytes},
        };
    } else {
        bufferInfos = {
            {bufInputDL.handle,   0, inputBytes},
            {bufWeightsDL.handle, 0, weightBytes},
            {bufBiasDL.handle,    0, biasBytes},
            {bufOutputDL.handle,  0, outputBytes},
        };
    }
    VkDescriptorSet descSet = cache.allocDescriptorSet(shaderName, bufferInfos);

    // Dispatch grid depends on the shader's output tile.
    uint32_t gx, gy;
    if (useCoopMat) {
        // gemm-coopmat-shared writes a 16×64 (M×N) tile per workgroup.
        gx = (p.outputDim + 63u) / 64u;
        gy = (p.batchSeq  + 15u) / 16u;
    } else {
        // fnn-linear writes a 16×16 tile per workgroup.
        gx = (p.outputDim + 15u) / 16u;
        gy = (p.batchSeq  + 15u) / 16u;
    }

    // ── Single command buffer: stage-in → barrier → compute → barrier → stage-out ──
    batch.begin();

    // Stage-in: DMA copy host-visible staging → DEVICE_LOCAL VRAM.
    // Bias goes to DL up front for both paths — fnn-linear reads it via
    // binding 2, and the coopmat bias-add post-pass reads it via binding 1.
    batch.copyBuffer(bufInputStage,   bufInputDL,   inputBytes);
    batch.copyBuffer(bufWeightsStage, bufWeightsDL, weightBytes);
    if (p.hasBias && bias) {
        batch.copyBuffer(bufBiasStage, bufBiasDL,
                         size_t(p.outputDim) * sizeof(float));
    }

    batch.transferComputeBarrier();

    if (useCoopMat) {
        // Coopmat push constants: {M, K, N, transpose_b} (16 bytes).
        // The linear op computes y = x . W^T, with W stored (outputDim,inputDim)
        // = (N,K). transpose_b=1 tells the kernel B is weights and to read W^T.
        struct CoopPush {
            uint32_t M;
            uint32_t K;
            uint32_t N;
            uint32_t transpose_b;
        } coopPush = {p.batchSeq, p.inputDim, p.outputDim, 1u};
        batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, gy, 1,
                       &coopPush, sizeof(coopPush));
    } else {
        // fnn-linear push constants: {batch, in, out, hasBias} (16 bytes)
        struct FnnPush {
            uint32_t batchSeq;
            uint32_t inputDim;
            uint32_t outputDim;
            uint32_t hasBias;
        } fnnPush = {p.batchSeq, p.inputDim, p.outputDim, p.hasBias};
        batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, gy, 1,
                       &fnnPush, sizeof(fnnPush));
    }

    // ── Bias post-pass (coopmat only; fnn-linear applies bias inline) ──
    // Bias was already copied to bufBiasDL during the stage-in phase above,
    // so the post-pass just needs a GEMM-write → bias-read barrier and a
    // dispatch of the gemm-bias-add kernel.
    if (useCoopMat && p.hasBias && bias &&
        cache.hasShader("gemm-bias-add")) {
        batch.barrier();  // SHADER_WRITE (GEMM) → SHADER_READ (bias-add)

        // gemm-bias-add: 2 bindings (C, bias), 8 bytes push {totalElements, N}
        PipelineEntry biasPipe =
            cache.getOrCreate("gemm-bias-add", 2, 2 * sizeof(uint32_t));
        std::vector<VkDescriptorBufferInfo> biasInfos = {
            {bufOutputDL.handle, 0, outputBytes},
            {bufBiasDL.handle,   0, size_t(p.outputDim) * sizeof(float)},
        };
        VkDescriptorSet biasSet =
            cache.allocDescriptorSet("gemm-bias-add", biasInfos);
        struct BiasPush {
            uint32_t totalElements;
            uint32_t N;
        } biasPush = {p.batchSeq * p.outputDim, p.outputDim};
        uint32_t biasGx = (biasPush.totalElements + 255u) / 256u;
        batch.dispatch(biasPipe.pipeline, biasPipe.layout, biasSet,
                       biasGx, 1, 1, &biasPush, sizeof(biasPush));
    }

    batch.transferComputeBarrier();

    // Stage-out: DMA copy DEVICE_LOCAL → host-visible HOST_CACHED staging
    batch.copyBuffer(bufOutputDL, bufOutputStage, outputBytes);

    batch.submitDeferred();
    batch.waitForCompletion();

    // ── memcpy staging → CPU output (HOST_CACHED, ~7 GB/s) ──
    // Output is always fp32, regardless of input dtype.
    pool.download(bufOutputStage, reinterpret_cast<float*>(output), outputBytes);

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
                    const void* gradOutput, const void* input,
                    const void* weights,
                    void* gradInput, void* gradWeight, void* gradBias,
                    const LinearParams& p) {
    // The fnn-linear-backward shader is fp32-only; reject fp16 input until a
    // coopmat backward shader lands. The void* interface is in place so the
    // switchover is local.
    if (p.elemSize != 4u) {
        throw std::runtime_error(
            "linearBackward(): currently requires fp32 (elemSize=4). fp16 "
            "backward needs a cooperative matrix backward shader — TODO.");
    }

    // Dynamic byte calculation. With elemSize==4 today these match the old
    // sizeof(float) computations, so existing callers see no behavior change.
    const size_t gradOutBytes  = size_t(p.batchSeq) * p.outputDim * p.elemSize;
    const size_t inputBytes    = size_t(p.batchSeq) * p.inputDim  * p.elemSize;
    const size_t weightBytes   = size_t(p.outputDim) * p.inputDim * p.elemSize;
    const size_t gradInBytes   = inputBytes;
    const size_t gradWBytes    = weightBytes;
    const size_t gradBiasBytes = size_t(p.outputDim) * p.elemSize;

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

    pool.upload(bufGradOutStage,
                reinterpret_cast<const float*>(gradOutput), gradOutBytes);
    pool.upload(bufInputStage,
                reinterpret_cast<const float*>(input), inputBytes);
    pool.upload(bufWeightsStage,
                reinterpret_cast<const float*>(weights), weightBytes);

    // The grad buffers must start at zero — pass 1 (grad_weight) and
    // pass 2 (grad_bias) accumulate via atomic adds in the shader. Use
    // raw byte vectors so zeroing works identically for fp32 and fp16
    // (whenever the fp16 backward shader lands). Reuse the readback stage
    // buffers as upload-zeros source — HOST_CACHED, CPU-write is fine.
    std::vector<uint8_t> zerosIn(gradInBytes, 0);
    std::vector<uint8_t> zerosW(gradWBytes, 0);
    std::vector<uint8_t> zerosB(gradBiasBytes, 0);
    pool.upload(bufGradInStage,
                reinterpret_cast<const float*>(zerosIn.data()), gradInBytes);
    pool.upload(bufGradWStage,
                reinterpret_cast<const float*>(zerosW.data()), gradWBytes);
    pool.upload(bufGradBiasStage,
                reinterpret_cast<const float*>(zerosB.data()), gradBiasBytes);

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

    pool.download(bufGradInStage,
                  reinterpret_cast<float*>(gradInput), gradInBytes);
    pool.download(bufGradWStage,
                  reinterpret_cast<float*>(gradWeight), gradWBytes);
    pool.download(bufGradBiasStage,
                  reinterpret_cast<float*>(gradBias), gradBiasBytes);

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
