#include "grilly/ops/swizzle.h"

#include <cstring>

namespace grilly {
namespace ops {

// ── Subgroup-Swizzled Block Caching Implementation ──────────────────────
//
// The memory layout transformation:
//
// Standard:  tensor[b][h][s][d]  where s = seq position, d = head dim element
//            Memory address = b*(H*S*D) + h*(S*D) + s*D + d
//
// Swizzled:  tensor[b][s/W][d/W][sw][dw]  where W = wave size
//            s_block = s / W,  s_wave = s % W
//            d_block = d / W,  d_wave = d % W
//            Memory address = b*(H*S_pad*D_pad) + h*(S_pad*D_pad)
//                           + s_block*(D_blocks*W*W) + d_block*(W*W)
//                           + s_wave*W + d_wave
//
// Why this works on AMD RDNA 2:
//
//   AMD's memory controller uses 256-byte (64-float) interleaving across
//   channels. In standard layout, 32 adjacent threads reading positions
//   s=0..31 for the same d all hit addresses that are D floats apart.
//   If D isn't a multiple of 64, some threads hit the same channel.
//
//   In swizzled layout, the inner two dimensions [s_wave][d_wave] form a
//   W×W block (32×32 = 1024 elements = 4096 bytes). Within this block,
//   thread i in the wave reads element [i][d_wave] — which is at offset
//   i*W + d_wave. Since W=32 and the memory controller interleaves at
//   64-element boundaries, threads 0..31 span exactly half a channel
//   block, and the next group of 32 (if Wave64) spans the other half.
//   No two threads in a wave hit the same channel.
//
//   Furthermore, each thread reads 4 consecutive floats (vec4 = 16 bytes)
//   by setting d_wave to be aligned to 4. This maps to a single 128-bit
//   VMEM instruction (buffer_load_dwordx4).
//
// subgroupQuadBroadcast usage:
//   After loading, threads need each other's K/V data for the matmul.
//   Instead of writing to shared memory (LDS) and reading back, we use
//   Vulkan subgroup operations:
//     float4 my_data = texelFetch(kv_cache, my_swizzled_index);
//     float4 neighbor_data = subgroupQuadBroadcast(my_data, lane_id);
//   This is a register-to-register transfer with zero latency.

void swizzleCPU(const float* input, float* output,
                uint32_t batchSize, uint32_t numHeads,
                uint32_t seqLen, uint32_t headDim,
                uint32_t waveSize) {
    uint32_t seqPadded = ((seqLen + waveSize - 1) / waveSize) * waveSize;
    uint32_t dimPadded = ((headDim + waveSize - 1) / waveSize) * waveSize;
    uint32_t dimBlocks = dimPadded / waveSize;

    // Zero output (padded regions should be zero)
    size_t outSize = size_t(batchSize) * numHeads * seqPadded * dimPadded;
    std::memset(output, 0, outSize * sizeof(float));

    for (uint32_t b = 0; b < batchSize; ++b) {
        for (uint32_t h = 0; h < numHeads; ++h) {
            for (uint32_t s = 0; s < seqLen; ++s) {
                for (uint32_t d = 0; d < headDim; ++d) {
                    // Source: standard layout
                    size_t srcIdx = (size_t(b) * numHeads + h) * seqLen *
                                    headDim + s * headDim + d;

                    // Destination: swizzled layout
                    uint32_t sBlock = s / waveSize;
                    uint32_t sWave  = s % waveSize;
                    uint32_t dBlock = d / waveSize;
                    uint32_t dWave  = d % waveSize;

                    size_t dstIdx = (size_t(b) * numHeads + h) * seqPadded *
                                    dimPadded
                                    + sBlock * (dimBlocks * waveSize * waveSize)
                                    + dBlock * (waveSize * waveSize)
                                    + sWave * waveSize
                                    + dWave;

                    output[dstIdx] = input[srcIdx];
                }
            }
        }
    }
}

void unswizzleCPU(const float* input, float* output,
                  uint32_t batchSize, uint32_t numHeads,
                  uint32_t seqLen, uint32_t headDim,
                  uint32_t waveSize) {
    uint32_t seqPadded = ((seqLen + waveSize - 1) / waveSize) * waveSize;
    uint32_t dimPadded = ((headDim + waveSize - 1) / waveSize) * waveSize;
    uint32_t dimBlocks = dimPadded / waveSize;

    for (uint32_t b = 0; b < batchSize; ++b) {
        for (uint32_t h = 0; h < numHeads; ++h) {
            for (uint32_t s = 0; s < seqLen; ++s) {
                for (uint32_t d = 0; d < headDim; ++d) {
                    // Source: swizzled layout
                    uint32_t sBlock = s / waveSize;
                    uint32_t sWave  = s % waveSize;
                    uint32_t dBlock = d / waveSize;
                    uint32_t dWave  = d % waveSize;

                    size_t srcIdx = (size_t(b) * numHeads + h) * seqPadded *
                                    dimPadded
                                    + sBlock * (dimBlocks * waveSize * waveSize)
                                    + dBlock * (waveSize * waveSize)
                                    + sWave * waveSize
                                    + dWave;

                    // Destination: standard layout
                    size_t dstIdx = (size_t(b) * numHeads + h) * seqLen *
                                    headDim + s * headDim + d;

                    output[dstIdx] = input[srcIdx];
                }
            }
        }
    }
}

// ── GPU swizzle dispatch ────────────────────────────────────────────────
//
// If a kv-swizzle.spv shader is available, dispatch on GPU. Otherwise
// fall back to the CPU path. The GPU shader would look like:
//
//   layout(local_size_x = 32) in;  // Wave32
//   void main() {
//       uint tid = gl_GlobalInvocationID.x;
//       // Compute source and destination indices
//       // Each thread handles one element
//       uint s = ...; uint d = ...;
//       uint sBlock = s / waveSize; uint sWave = s % waveSize;
//       uint dBlock = d / waveSize; uint dWave = d % waveSize;
//       uint srcIdx = (b*H + h)*S*D + s*D + d;
//       uint dstIdx = (b*H + h)*S_pad*D_pad + sBlock*(dimBlocks*W*W)
//                   + dBlock*(W*W) + sWave*W + dWave;
//       output_buf[dstIdx] = input_buf[srcIdx];
//   }

// ── GPU swizzle dispatch ────────────────────────────────────────────────
//
// The kv-swizzle.spv shader maps each thread to one element of the tensor
// and computes the swizzled index using the same formula as the CPU path.
// Each thread reads input[srcIdx] and writes output[dstIdx] (or vice versa
// for unswizzle, controlled by the direction push constant).
//
// Layout:
//   local_size_x = 256  (8 waves of 32 for RDNA 2)
//   Binding 0: input buffer
//   Binding 1: output buffer
//   Push constants: SwizzleParams (6 × uint32 = 24 bytes)
//
// The shader is a pure memory reorganization — no arithmetic on values.
// GPU dispatch is worthwhile because:
//   1. It avoids the CPU→GPU upload of the swizzled result
//   2. GPU memory bandwidth (288 GB/s on RX 6750 XT) far exceeds PCIe
//      bandwidth (~14 GB/s) for the copy
//   3. The operation is embarrassingly parallel — each thread is independent

void swizzle(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
             const float* input, float* output,
             uint32_t batchSize, uint32_t numHeads,
             uint32_t seqLen, uint32_t headDim,
             uint32_t waveSize, bool reverse) {
    // Try GPU path first: dispatch kv-swizzle shader if available
    if (cache.hasShader("kv-swizzle")) {
        uint32_t seqPadded = ((seqLen + waveSize - 1) / waveSize) * waveSize;
        uint32_t dimPadded = ((headDim + waveSize - 1) / waveSize) * waveSize;

        size_t inputBytes = reverse
            ? size_t(batchSize) * numHeads * seqPadded * dimPadded * sizeof(float)
            : size_t(batchSize) * numHeads * seqLen * headDim * sizeof(float);
        size_t outputBytes = reverse
            ? size_t(batchSize) * numHeads * seqLen * headDim * sizeof(float)
            : size_t(batchSize) * numHeads * seqPadded * dimPadded * sizeof(float);

        GrillyBuffer bufIn  = pool.acquire(inputBytes);
        GrillyBuffer bufOut = pool.acquire(outputBytes);

        pool.upload(bufIn, input, inputBytes);

        // Zero output buffer (padded regions must be zero for swizzle)
        if (!reverse) {
            std::vector<float> zeros(outputBytes / sizeof(float), 0.0f);
            pool.upload(bufOut, zeros.data(), outputBytes);
        }

        PipelineEntry pipe = cache.getOrCreate("kv-swizzle", 2,
                                                sizeof(SwizzleParams));

        std::vector<VkDescriptorBufferInfo> bufInfos = {
            {bufIn.handle,  0, inputBytes},
            {bufOut.handle, 0, outputBytes},
        };
        VkDescriptorSet descSet = cache.allocDescriptorSet("kv-swizzle",
                                                            bufInfos);

        SwizzleParams push{batchSize, numHeads, seqLen, headDim, waveSize,
                           reverse ? 1u : 0u};

        // Total elements to process (each thread handles one element)
        uint32_t totalElements = batchSize * numHeads * seqLen * headDim;
        uint32_t gx = (totalElements + 255) / 256;

        batch.begin();
        batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, 1, 1,
                       &push, sizeof(push));
        batch.submit();

        pool.download(bufOut, output, outputBytes);

        pool.release(bufIn);
        pool.release(bufOut);
        return;
    }

    // Fallback: CPU path (when kv-swizzle.spv isn't loaded)
    if (reverse) {
        unswizzleCPU(input, output, batchSize, numHeads, seqLen, headDim,
                     waveSize);
    } else {
        swizzleCPU(input, output, batchSize, numHeads, seqLen, headDim,
                   waveSize);
    }
}

}  // namespace ops
}  // namespace grilly
