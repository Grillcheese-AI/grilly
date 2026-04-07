#pragma once
/// Vulkan batched command buffer with timeline semaphore support.

#include <vulkan/vulkan.h>

#include <cstdint>

#include "grilly/vulkan/vk_device.h"
#include "grilly/vulkan/vk_buffer_pool.h"

namespace grilly {

/// Batched command buffer recorder with single-submit semantics.
class CommandBatch {
public:
    explicit CommandBatch(GrillyDevice& device);
    ~CommandBatch();

    CommandBatch(const CommandBatch&) = delete;
    CommandBatch& operator=(const CommandBatch&) = delete;

    void begin();

    /// Start recording if not already. Safe to call multiple times.
    void ensureRecording();

    void dispatch(VkPipeline pipeline, VkPipelineLayout layout,
                  VkDescriptorSet descSet,
                  uint32_t gx, uint32_t gy = 1, uint32_t gz = 1,
                  const void* pushData = nullptr, uint32_t pushSize = 0);

    void barrier();

    /// Bidirectional TRANSFER ↔ COMPUTE memory barrier — covers both
    /// stage-in → compute (TRANSFER_WRITE → SHADER_READ) and
    /// compute → stage-out (SHADER_WRITE → TRANSFER_READ). Used by the
    /// staging-buffer pattern in cpp/src/ops/linear.cpp.
    void transferComputeBarrier();

    /// GPU-to-GPU buffer copy (no CPU involvement).
    void copyBuffer(const GrillyBuffer& src, GrillyBuffer& dst, size_t bytes);

    void submit();
    void submitAsync(VkSemaphore timeline, uint64_t signalValue);

    /// Submit without waiting — GPU runs while CPU continues.
    void submitDeferred();

    /// Block until the last submitted work finishes. No-op if nothing pending.
    void waitForCompletion();

    /// Number of dispatches recorded in current command buffer.
    uint32_t dispatchCount() const { return dispatchCount_; }

    bool isRecording() const { return recording_; }
    bool isPending() const { return pending_; }
    VkCommandBuffer cmdBuffer() const { return cmd_; }

private:
    GrillyDevice& device_;
    VkCommandPool pool_ = VK_NULL_HANDLE;
    VkCommandBuffer cmd_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
    bool recording_ = false;
    bool pending_ = false;
    uint32_t dispatchCount_ = 0;
};

}  // namespace grilly
