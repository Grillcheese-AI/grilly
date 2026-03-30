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

    void dispatch(VkPipeline pipeline, VkPipelineLayout layout,
                  VkDescriptorSet descSet,
                  uint32_t gx, uint32_t gy = 1, uint32_t gz = 1,
                  const void* pushData = nullptr, uint32_t pushSize = 0);

    void barrier();

    /// GPU-to-GPU buffer copy (no CPU involvement).
    /// Must be called between begin() and submit().
    void copyBuffer(const GrillyBuffer& src, GrillyBuffer& dst, size_t bytes);

    void submit();
    void submitAsync(VkSemaphore timeline, uint64_t signalValue);

    bool isRecording() const { return recording_; }
    VkCommandBuffer cmdBuffer() const { return cmd_; }

private:
    GrillyDevice& device_;
    VkCommandPool pool_ = VK_NULL_HANDLE;
    VkCommandBuffer cmd_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
    bool recording_ = false;
};

}  // namespace grilly
