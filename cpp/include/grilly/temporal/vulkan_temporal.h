#pragma once

#include <cstdint>
#include <vector>

#include "grilly/buffer_pool.h"
#include "grilly/command_batch.h"
#include "grilly/cubemind/types.h"
#include "grilly/pipeline_cache.h"

namespace grilly {
namespace temporal {

/// Push constants matching circular-shift.glsl layout.
struct CircularShiftParams {
    uint32_t words_per_vec;  // 320 for 10240d
    uint32_t shift_amount;   // T value (bits to shift)
    uint32_t mode;           // 0 = right (bind), 1 = left (unbind)
};

/// Result of a batched temporal shift on GPU.
struct BatchShiftResult {
    std::vector<uint32_t> data;   // Contiguous: batch_size * words_per_vec
    uint32_t batch_size;
    uint32_t words_per_vec;
};

/// GPU-accelerated batched temporal shift dispatcher.
///
/// Loads the circular-shift.spv shader and dispatches N timelines
/// in a single vkCmdDispatch(N, 1, 1). Each workgroup (320 threads)
/// handles one 10240-bit vector, computing the circular bit shift
/// for time binding or unbinding.
///
/// Memory layout:
///   Input buffer:  [vec_0: 320 words][vec_1: 320 words]...[vec_N-1]
///   Output buffer: [shifted_0][shifted_1]...[shifted_N-1]
///
/// On the A40 with 84 SMs and 32-thread warps:
///   - 320 threads/WG = 10 warps/WG
///   - 128 WGs = 1280 warps → fully occupies all SMs
///   - Expected latency: ~30us for 128 timelines
///
class VulkanTemporalDispatcher {
public:
    /// Batch circular shift on GPU.
    ///
    /// @param batch       CommandBatch for recording the dispatch
    /// @param pool        BufferPool for GPU buffer management
    /// @param pipeCache   PipelineCache for shader pipeline lookup
    /// @param input       Contiguous bitpacked vectors (batch_size * words_per_vec)
    /// @param batch_size  Number of timelines to shift
    /// @param words_per_vec  uint32 words per vector (default 320)
    /// @param shift_amount   Bits to shift (time step)
    /// @param mode        0 = right shift (bind), 1 = left shift (unbind)
    /// @return            Shifted output vectors (contiguous)
    static BatchShiftResult dispatch(
        CommandBatch& batch,
        BufferPool& pool,
        PipelineCache& pipeCache,
        const uint32_t* input,
        uint32_t batch_size,
        uint32_t words_per_vec,
        uint32_t shift_amount,
        uint32_t mode = 0);

    /// Batch counterfactual evaluation:
    ///   1. XOR-erase actual facts from base timelines (CPU, instant)
    ///   2. XOR-insert counterfactual facts (CPU, instant)
    ///   3. GPU circular shift all N timelines forward by dt steps
    ///
    /// @param batch         CommandBatch for GPU dispatch
    /// @param pool          BufferPool for buffer management
    /// @param pipeCache     PipelineCache for shader pipeline
    /// @param base_timeline Single base timeline (words_per_vec uint32s)
    /// @param actual_facts  N fact vectors to erase (N * words_per_vec)
    /// @param what_if_facts N counterfactual facts to insert (N * words_per_vec)
    /// @param n             Number of counterfactual branches
    /// @param words_per_vec Words per vector (default 320)
    /// @param dt            Time steps to project forward (default 1)
    /// @return              N shifted alternate futures (contiguous)
    static BatchShiftResult batch_counterfactuals(
        CommandBatch& batch,
        BufferPool& pool,
        PipelineCache& pipeCache,
        const uint32_t* base_timeline,
        const uint32_t* actual_facts,
        const uint32_t* what_if_facts,
        uint32_t n,
        uint32_t words_per_vec,
        uint32_t dt = 1);
};

}  // namespace temporal
}  // namespace grilly
