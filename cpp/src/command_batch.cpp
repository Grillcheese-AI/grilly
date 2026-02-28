#include "grilly/command_batch.h"

#include <cstring>
#include <stdexcept>

namespace grilly {

// ── Helper ──────────────────────────────────────────────────────────────────
static void vkCheck(VkResult result, const char* msg) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(std::string(msg) +
                                 " (VkResult=" + std::to_string(result) + ")");
    }
}

static constexpr uint64_t kFenceTimeoutNs = 2'000'000'000;  // 2 seconds

// ── Construction / destruction ──────────────────────────────────────────────

CommandBatch::CommandBatch(GrillyDevice& device) : device_(device) {
    // Create a dedicated command pool with reset-per-buffer capability
    // (same flags as core.py:343-348)
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = device_.queueFamily();
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    vkCheck(vkCreateCommandPool(device_.device(), &poolInfo, nullptr, &pool_),
            "vkCreateCommandPool failed");

    // Allocate one primary command buffer
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = pool_;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    vkCheck(vkAllocateCommandBuffers(device_.device(), &allocInfo, &cmd_),
            "vkAllocateCommandBuffers failed");

    // Create fence (starts signaled so first wait is a no-op)
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    vkCheck(vkCreateFence(device_.device(), &fenceInfo, nullptr, &fence_),
            "vkCreateFence failed");
}

CommandBatch::~CommandBatch() {
    VkDevice dev = device_.device();
    if (dev == VK_NULL_HANDLE)
        return;

    // Wait for any pending work
    if (fence_ != VK_NULL_HANDLE)
        vkWaitForFences(dev, 1, &fence_, VK_TRUE, kFenceTimeoutNs);

    if (fence_ != VK_NULL_HANDLE)
        vkDestroyFence(dev, fence_, nullptr);
    if (cmd_ != VK_NULL_HANDLE)
        vkFreeCommandBuffers(dev, pool_, 1, &cmd_);
    if (pool_ != VK_NULL_HANDLE)
        vkDestroyCommandPool(dev, pool_, nullptr);
}

// ── Recording ───────────────────────────────────────────────────────────────

void CommandBatch::begin() {
    if (recording_)
        throw std::runtime_error("CommandBatch::begin() called while already recording");

    vkResetCommandBuffer(cmd_, 0);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkCheck(vkBeginCommandBuffer(cmd_, &beginInfo),
            "vkBeginCommandBuffer failed");

    recording_ = true;
}

void CommandBatch::dispatch(VkPipeline pipeline, VkPipelineLayout layout,
                            VkDescriptorSet descSet, uint32_t gx,
                            uint32_t gy, uint32_t gz,
                            const void* pushData, uint32_t pushSize) {
    if (!recording_)
        throw std::runtime_error("CommandBatch::dispatch() called without begin()");

    vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0,
                            1, &descSet, 0, nullptr);

    if (pushData && pushSize > 0) {
        vkCmdPushConstants(cmd_, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           pushSize, pushData);
    }

    vkCmdDispatch(cmd_, gx, gy, gz);
}

void CommandBatch::barrier() {
    if (!recording_)
        return;

    // COMPUTE → COMPUTE memory barrier (SHADER_WRITE → SHADER_READ)
    // Same logic as core.py:854-878
    VkMemoryBarrier memBarrier{};
    memBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(cmd_,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,  // src
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,  // dst
                         0,          // dependency flags
                         1, &memBarrier,   // memory barriers
                         0, nullptr,       // buffer memory barriers
                         0, nullptr);      // image memory barriers
}

// ── Submission ──────────────────────────────────────────────────────────────

void CommandBatch::submit() {
    if (!recording_)
        return;

    vkCheck(vkEndCommandBuffer(cmd_), "vkEndCommandBuffer failed");
    recording_ = false;

    // Wait for any prior submission to complete, then reset fence
    vkWaitForFences(device_.device(), 1, &fence_, VK_TRUE, kFenceTimeoutNs);
    vkResetFences(device_.device(), 1, &fence_);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd_;

    vkCheck(vkQueueSubmit(device_.computeQueue(), 1, &submitInfo, fence_),
            "vkQueueSubmit failed");

    // Wait for completion so callers can read back results
    vkCheck(
        vkWaitForFences(device_.device(), 1, &fence_, VK_TRUE, kFenceTimeoutNs),
        "vkWaitForFences timed out");
}

void CommandBatch::submitAsync(VkSemaphore timeline, uint64_t signalValue) {
    if (!recording_)
        return;

    vkCheck(vkEndCommandBuffer(cmd_), "vkEndCommandBuffer failed");
    recording_ = false;

    // Wait for prior work
    vkWaitForFences(device_.device(), 1, &fence_, VK_TRUE, kFenceTimeoutNs);
    vkResetFences(device_.device(), 1, &fence_);

    // Timeline semaphore signal info (Vulkan 1.2 / VK_KHR_timeline_semaphore)
    VkTimelineSemaphoreSubmitInfo timelineInfo{};
    timelineInfo.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    timelineInfo.signalSemaphoreValueCount = 1;
    timelineInfo.pSignalSemaphoreValues = &signalValue;

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.pNext = &timelineInfo;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd_;
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &timeline;

    vkCheck(vkQueueSubmit(device_.computeQueue(), 1, &submitInfo, fence_),
            "vkQueueSubmit (async) failed");
}

}  // namespace grilly
