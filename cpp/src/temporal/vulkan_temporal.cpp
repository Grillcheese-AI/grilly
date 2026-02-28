#include "grilly/temporal/vulkan_temporal.h"

#include <cstring>

namespace grilly {
namespace temporal {

BatchShiftResult VulkanTemporalDispatcher::dispatch(
        CommandBatch& batch,
        BufferPool& pool,
        PipelineCache& pipeCache,
        const uint32_t* input,
        uint32_t batch_size,
        uint32_t words_per_vec,
        uint32_t shift_amount,
        uint32_t mode) {

    const size_t vec_bytes = size_t(words_per_vec) * sizeof(uint32_t);
    const size_t total_bytes = size_t(batch_size) * vec_bytes;

    BatchShiftResult result;
    result.batch_size = batch_size;
    result.words_per_vec = words_per_vec;
    result.data.resize(size_t(batch_size) * words_per_vec);

    // CPU fallback if shader not loaded
    if (!pipeCache.hasShader("circular-shift")) {
        // Use the CPU TemporalEncoder for each vector
        for (uint32_t i = 0; i < batch_size; ++i) {
            const uint32_t* src = input + size_t(i) * words_per_vec;
            uint32_t* dst = result.data.data() + size_t(i) * words_per_vec;

            uint32_t total_bits = words_per_vec * 32;
            uint32_t effective = shift_amount % total_bits;
            if (mode == 1) {
                effective = total_bits - effective;
            }
            if (effective == 0) {
                std::memcpy(dst, src, vec_bytes);
                continue;
            }

            uint32_t ws = effective / 32;
            uint32_t bs = effective % 32;
            if (bs == 0) {
                for (uint32_t j = 0; j < words_per_vec; ++j) {
                    uint32_t si = (j + words_per_vec - ws) % words_per_vec;
                    dst[j] = src[si];
                }
            } else {
                uint32_t comp = 32 - bs;
                for (uint32_t j = 0; j < words_per_vec; ++j) {
                    uint32_t hi = (j + words_per_vec - ws) % words_per_vec;
                    uint32_t lo = (hi + words_per_vec - 1) % words_per_vec;
                    dst[j] = (src[hi] >> bs) | (src[lo] << comp);
                }
            }
        }
        return result;
    }

    // ── GPU Path ──────────────────────────────────────────────────────────

    // 1. Acquire buffers
    GrillyBuffer bufIn  = pool.acquire(total_bytes);
    GrillyBuffer bufOut = pool.acquire(total_bytes);

    // 2. Upload input (contiguous: all timelines packed)
    pool.upload(bufIn, reinterpret_cast<const float*>(input), total_bytes);

    // 3. Get or create compute pipeline (2 buffers, 12 bytes push constants)
    PipelineEntry pipe = pipeCache.getOrCreate(
        "circular-shift", 2, sizeof(CircularShiftParams));

    // 4. Build descriptor buffer infos
    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {bufIn.handle,  0, total_bytes},   // binding 0: InputState
        {bufOut.handle, 0, total_bytes},   // binding 1: OutputState
    };

    // 5. Allocate descriptor set
    VkDescriptorSet descSet = pipeCache.allocDescriptorSet(
        "circular-shift", bufInfos);

    // 6. Push constants
    CircularShiftParams push;
    push.words_per_vec = words_per_vec;
    push.shift_amount  = shift_amount;
    push.mode          = mode;

    // 7. Dispatch: one workgroup per timeline
    //    Each WG = 320 threads (local_size_x = 320)
    batch.begin();
    batch.dispatch(pipe.pipeline, pipe.layout, descSet,
                   batch_size, 1, 1,
                   &push, sizeof(push));
    batch.submit();

    // 8. Download results
    pool.download(bufOut, reinterpret_cast<float*>(result.data.data()),
                  total_bytes);

    // 9. Release buffers
    pool.release(bufIn);
    pool.release(bufOut);

    return result;
}

BatchShiftResult VulkanTemporalDispatcher::batch_counterfactuals(
        CommandBatch& batch,
        BufferPool& pool,
        PipelineCache& pipeCache,
        const uint32_t* base_timeline,
        const uint32_t* actual_facts,
        const uint32_t* what_if_facts,
        uint32_t n,
        uint32_t words_per_vec,
        uint32_t dt) {

    const size_t vec_words = size_t(words_per_vec);

    // ── CPU Phase: XOR interventions (instant, ~nanoseconds per vec) ─────
    //
    // For each counterfactual branch i:
    //   intervened[i] = base_timeline ^ actual_facts[i] ^ what_if_facts[i]
    //
    // This simultaneously erases the actual fact and inserts the "what if".
    // XOR is self-inverse, so: base ^ fact ^ fact = base (no-op if same).
    //
    std::vector<uint32_t> intervened(size_t(n) * vec_words);

    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t* actual = actual_facts + size_t(i) * vec_words;
        const uint32_t* whatif = what_if_facts + size_t(i) * vec_words;
        uint32_t* dst = intervened.data() + size_t(i) * vec_words;

        for (uint32_t j = 0; j < vec_words; ++j) {
            dst[j] = base_timeline[j] ^ actual[j] ^ whatif[j];
        }
    }

    // ── GPU Phase: Circular shift all N alternate timelines ──────────────
    //
    // dispatch(N workgroups) shifts all branches forward by dt time steps.
    //
    return dispatch(batch, pool, pipeCache,
                    intervened.data(), n, words_per_vec, dt, 0);
}

}  // namespace temporal
}  // namespace grilly
