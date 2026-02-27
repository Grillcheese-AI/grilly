#include "grilly/buffer_pool.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace grilly {

// ── Helper ──────────────────────────────────────────────────────────────────
static void vkCheck(VkResult result, const char* msg) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(std::string(msg) +
                                 " (VkResult=" + std::to_string(result) + ")");
    }
}

// ── Construction / destruction ──────────────────────────────────────────────

BufferPool::BufferPool(GrillyDevice& device) : device_(device) {
    // Create VMA allocator.
    // VMA_IMPLEMENTATION is compiled in device.cpp — here we just use the API.
    VmaAllocatorCreateInfo allocInfo{};
    allocInfo.physicalDevice = device_.physicalDevice();
    allocInfo.device = device_.device();
    allocInfo.instance = device_.instance();
    allocInfo.vulkanApiVersion = VK_MAKE_API_VERSION(0, 1, 3, 0);

    // Enable VK_EXT_memory_priority support in VMA — prevents WDDM eviction
    if (device_.hasExtension("VK_EXT_memory_priority")) {
        allocInfo.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_PRIORITY_BIT;
    }
    // Enable VK_EXT_memory_budget for better allocation decisions
    if (device_.hasExtension("VK_EXT_memory_budget")) {
        allocInfo.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
    }

    vkCheck(vmaCreateAllocator(&allocInfo, &allocator_),
            "vmaCreateAllocator failed");

    std::cout << "[OK] VMA allocator initialized (C++ native)" << std::endl;
}

BufferPool::~BufferPool() {
    // Destroy all pooled buffers first
    for (auto& [bucketSize, vec] : buckets_) {
        for (auto& buf : vec) {
            if (buf.handle != VK_NULL_HANDLE)
                vmaDestroyBuffer(allocator_, buf.handle, buf.allocation);
        }
    }
    buckets_.clear();

    if (allocator_ != VK_NULL_HANDLE)
        vmaDestroyAllocator(allocator_);
}

// ── Bucket sizing (port of buffer_pool.py:285-291) ─────────────────────────

size_t BufferPool::sizeToBucket(size_t size) const {
    if (size == 0)
        return size_t(1) << kMinBucketPower;

    // Round up to next power of 2
    int power = kMinBucketPower;
    size_t bucket = size_t(1) << power;
    while (bucket < size && power < kMaxBucketPower) {
        ++power;
        bucket = size_t(1) << power;
    }

    // For sizes exceeding the max bucket, return the exact size.
    // These allocations bypass the pool (too large to cache).
    if (bucket < size)
        return size;

    return bucket;
}

// ── Buffer allocation via VMA ───────────────────────────────────────────────

GrillyBuffer BufferPool::allocateBuffer(size_t bucketSize) {
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = bucketSize;
    bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    // VMA_MEMORY_USAGE_AUTO lets VMA pick the best heap.
    // MAPPED_BIT gives us a persistent CPU pointer — the key win over Python
    // which does vkMapMemory/vkUnmapMemory on every upload/download cycle.
    // HOST_ACCESS_SEQUENTIAL_WRITE_BIT hints that we write linearly (memcpy).
    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
                      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

    GrillyBuffer buf{};
    buf.bucketSize = bucketSize;

    vkCheck(vmaCreateBuffer(allocator_, &bufferInfo, &allocInfo, &buf.handle,
                            &buf.allocation, &buf.info),
            "vmaCreateBuffer failed");

    buf.mappedPtr = buf.info.pMappedData;
    return buf;
}

// ── Acquire (port of buffer_pool.py:293-369) ────────────────────────────────

GrillyBuffer BufferPool::acquire(size_t size) {
    size_t bucket = sizeToBucket(size);

    std::lock_guard<std::mutex> lock(mutex_);
    stats_.totalAcquired++;

    // Try reuse from bucket
    auto it = buckets_.find(bucket);
    if (it != buckets_.end() && !it->second.empty()) {
        GrillyBuffer buf = it->second.back();
        it->second.pop_back();
        buf.size = size;
        stats_.hits++;
        return buf;
    }

    // Allocate new
    stats_.misses++;
    stats_.allocations++;
    GrillyBuffer buf = allocateBuffer(bucket);
    buf.size = size;
    return buf;
}

// ── Release ─────────────────────────────────────────────────────────────────

void BufferPool::release(GrillyBuffer& buf) {
    if (buf.handle == VK_NULL_HANDLE)
        return;

    std::lock_guard<std::mutex> lock(mutex_);
    stats_.totalReleased++;

    auto& vec = buckets_[buf.bucketSize];
    if (vec.size() < kMaxBuffersPerBucket) {
        vec.push_back(buf);
    } else {
        // Bucket full — destroy immediately
        vmaDestroyBuffer(allocator_, buf.handle, buf.allocation);
    }

    // Null out the caller's handle so they don't double-free
    buf.handle = VK_NULL_HANDLE;
    buf.allocation = VK_NULL_HANDLE;
    buf.mappedPtr = nullptr;
}

// ── Device-Local Buffer ────────────────────────────────────────────────────
// GPU-only (VRAM) buffers have ~20x more bandwidth than host-visible on
// discrete GPUs. At 490K × 320 × 4 = 627 MB, the difference between
// VRAM (288 GB/s on RDNA 2) vs system RAM (14 GB/s over PCIe 4.0) is
// the difference between <1ms and ~45ms per Hamming search.

GrillyBuffer BufferPool::acquireDeviceLocal(size_t size) {
    size_t bucket = sizeToBucket(size);

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = bucket;
    bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                       VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    allocInfo.priority = 1.0f;  // Maximum priority — keep in VRAM, don't evict

    GrillyBuffer buf{};
    buf.bucketSize = bucket;
    buf.size = size;

    vkCheck(vmaCreateBuffer(allocator_, &bufferInfo, &allocInfo, &buf.handle,
                            &buf.allocation, &buf.info),
            "vmaCreateBuffer (device-local) failed");

    buf.mappedPtr = nullptr;  // Not host-visible
    return buf;
}

GrillyBuffer BufferPool::acquirePreferDeviceLocal(size_t size) {
    size_t bucket = sizeToBucket(size);

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = bucket;
    bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    // Request host-visible mapping but PREFER device-local.
    // On AMD with ReBAR (256 MB BAR → 8+ GB), VMA places this in VRAM
    // with host-visible mapping — the sweet spot for CubeMind cache:
    // GPU reads at VRAM speed, CPU writes via memcpy for updates.
    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
                      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
    allocInfo.preferredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    GrillyBuffer buf{};
    buf.bucketSize = bucket;
    buf.size = size;

    vkCheck(vmaCreateBuffer(allocator_, &bufferInfo, &allocInfo, &buf.handle,
                            &buf.allocation, &buf.info),
            "vmaCreateBuffer (prefer-device-local) failed");

    buf.mappedPtr = buf.info.pMappedData;
    return buf;
}

GrillyBuffer BufferPool::acquireReadback(size_t size) {
    size_t bucket = sizeToBucket(size);

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = bucket;
    bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                       VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
    // RANDOM_BIT: maps into cached system RAM (L1/L2/L3) instead of
    // Write-Combined memory. CPU reads are ~10 GB/s instead of ~39 MB/s.
    allocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
                      VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;

    GrillyBuffer buf{};
    buf.bucketSize = bucket;
    buf.size = size;

    vkCheck(vmaCreateBuffer(allocator_, &bufferInfo, &allocInfo, &buf.handle,
                            &buf.allocation, &buf.info),
            "vmaCreateBuffer (readback) failed");

    buf.mappedPtr = buf.info.pMappedData;
    return buf;
}

void BufferPool::uploadStaged(GrillyBuffer& deviceBuf, const void* data,
                               size_t bytes) {
    // 1. Create a temporary host-visible staging buffer
    VkBufferCreateInfo stagingInfo{};
    stagingInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    stagingInfo.size = bytes;
    stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    stagingInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo stagingAlloc{};
    stagingAlloc.usage = VMA_MEMORY_USAGE_AUTO;
    stagingAlloc.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
                         VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

    VkBuffer stagingBuf;
    VmaAllocation stagingMem;
    VmaAllocationInfo stagingMemInfo;
    vkCheck(vmaCreateBuffer(allocator_, &stagingInfo, &stagingAlloc,
                            &stagingBuf, &stagingMem, &stagingMemInfo),
            "staging buffer alloc failed");

    // 2. Copy data into staging buffer
    std::memcpy(stagingMemInfo.pMappedData, data, bytes);
    vmaFlushAllocation(allocator_, stagingMem, 0, bytes);

    // 3. Create a one-shot command buffer for the transfer
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    poolInfo.queueFamilyIndex = device_.queueFamily();

    VkCommandPool cmdPool;
    vkCheck(vkCreateCommandPool(device_.device(), &poolInfo, nullptr, &cmdPool),
            "vkCreateCommandPool for staging failed");

    VkCommandBufferAllocateInfo cmdAllocInfo{};
    cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAllocInfo.commandPool = cmdPool;
    cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandBufferCount = 1;

    VkCommandBuffer cmd;
    vkCheck(vkAllocateCommandBuffers(device_.device(), &cmdAllocInfo, &cmd),
            "vkAllocateCommandBuffers for staging failed");

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    VkBufferCopy copyRegion{};
    copyRegion.size = bytes;
    vkCmdCopyBuffer(cmd, stagingBuf, deviceBuf.handle, 1, &copyRegion);

    vkEndCommandBuffer(cmd);

    // 4. Submit and wait
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence;
    vkCheck(vkCreateFence(device_.device(), &fenceInfo, nullptr, &fence),
            "staging fence creation failed");

    vkCheck(vkQueueSubmit(device_.computeQueue(), 1, &submitInfo, fence),
            "staging vkQueueSubmit failed");
    vkCheck(vkWaitForFences(device_.device(), 1, &fence, VK_TRUE, UINT64_MAX),
            "staging vkWaitForFences failed");

    // 5. Cleanup
    vkDestroyFence(device_.device(), fence, nullptr);
    vkDestroyCommandPool(device_.device(), cmdPool, nullptr);
    vmaDestroyBuffer(allocator_, stagingBuf, stagingMem);
}

void BufferPool::downloadStaged(const GrillyBuffer& deviceBuf, void* out,
                                  size_t bytes) {
    // 1. Create a temporary host-visible staging buffer for readback
    VkBufferCreateInfo stagingInfo{};
    stagingInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    stagingInfo.size = bytes;
    stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    stagingInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo stagingAlloc{};
    stagingAlloc.usage = VMA_MEMORY_USAGE_AUTO;
    stagingAlloc.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
                         VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;

    VkBuffer stagingBuf;
    VmaAllocation stagingMem;
    VmaAllocationInfo stagingMemInfo;
    vkCheck(vmaCreateBuffer(allocator_, &stagingInfo, &stagingAlloc,
                            &stagingBuf, &stagingMem, &stagingMemInfo),
            "staging readback buffer alloc failed");

    // 2. Copy from device-local to staging via DMA
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    poolInfo.queueFamilyIndex = device_.queueFamily();

    VkCommandPool cmdPool;
    vkCheck(vkCreateCommandPool(device_.device(), &poolInfo, nullptr, &cmdPool),
            "vkCreateCommandPool for readback failed");

    VkCommandBufferAllocateInfo cmdAllocInfo{};
    cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAllocInfo.commandPool = cmdPool;
    cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandBufferCount = 1;

    VkCommandBuffer cmd;
    vkCheck(vkAllocateCommandBuffers(device_.device(), &cmdAllocInfo, &cmd),
            "vkAllocateCommandBuffers for readback failed");

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    VkBufferCopy copyRegion{};
    copyRegion.size = bytes;
    vkCmdCopyBuffer(cmd, deviceBuf.handle, stagingBuf, 1, &copyRegion);

    vkEndCommandBuffer(cmd);

    // 3. Submit and wait
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence;
    vkCheck(vkCreateFence(device_.device(), &fenceInfo, nullptr, &fence),
            "readback fence creation failed");

    vkCheck(vkQueueSubmit(device_.computeQueue(), 1, &submitInfo, fence),
            "readback vkQueueSubmit failed");
    vkCheck(vkWaitForFences(device_.device(), 1, &fence, VK_TRUE, UINT64_MAX),
            "readback vkWaitForFences failed");

    // 4. Invalidate + copy to output
    vmaInvalidateAllocation(allocator_, stagingMem, 0, bytes);
    std::memcpy(out, stagingMemInfo.pMappedData, bytes);

    // 5. Cleanup
    vkDestroyFence(device_.device(), fence, nullptr);
    vkDestroyCommandPool(device_.device(), cmdPool, nullptr);
    vmaDestroyBuffer(allocator_, stagingBuf, stagingMem);
}

// ── Upload / Download ───────────────────────────────────────────────────────
// With VMA persistent mapping, these are just memcpy + flush/invalidate.
// The Python backend does vkMapMemory → ctypes.memmove → vkUnmapMemory
// for every single transfer — 3 FFI calls. We do 0 Vulkan calls here.

void BufferPool::upload(GrillyBuffer& buf, const float* data, size_t bytes) {
    if (!buf.mappedPtr)
        throw std::runtime_error("Buffer has no persistent mapping");
    std::memcpy(buf.mappedPtr, data, bytes);
    // Flush to make writes visible to the GPU.
    // For HOST_COHERENT memory this is a no-op, but VMA may choose
    // non-coherent memory for performance — flush is always safe.
    vmaFlushAllocation(allocator_, buf.allocation, 0, bytes);
}

void BufferPool::download(const GrillyBuffer& buf, float* out, size_t bytes) {
    if (!buf.mappedPtr)
        throw std::runtime_error("Buffer has no persistent mapping");
    // Invalidate to see GPU writes on the CPU side.
    vmaInvalidateAllocation(allocator_, buf.allocation, 0, bytes);
    std::memcpy(out, buf.mappedPtr, bytes);
}

// ── Stats ───────────────────────────────────────────────────────────────────

BufferPool::Stats BufferPool::stats() const { return stats_; }

}  // namespace grilly
