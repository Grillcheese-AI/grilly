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
    VkDevice dev = device_.device();

    // Clean up persistent transfer context
    if (transferInitialized_) {
        if (transferFence_ != VK_NULL_HANDLE) {
            vkWaitForFences(dev, 1, &transferFence_, VK_TRUE, UINT64_MAX);
            vkDestroyFence(dev, transferFence_, nullptr);
        }
        if (transferCmd_ != VK_NULL_HANDLE)
            vkFreeCommandBuffers(dev, transferPool_, 1, &transferCmd_);
        if (transferPool_ != VK_NULL_HANDLE)
            vkDestroyCommandPool(dev, transferPool_, nullptr);
    }

    // Destroy all pooled buffers (host-visible bucket pool)
    for (auto& [bucketSize, vec] : buckets_) {
        for (auto& buf : vec) {
            if (buf.handle != VK_NULL_HANDLE)
                vmaDestroyBuffer(allocator_, buf.handle, buf.allocation);
        }
    }
    buckets_.clear();

    // Destroy all pooled buffers (device-local bucket pool)
    for (auto& [bucketSize, vec] : dlBuckets_) {
        for (auto& buf : vec) {
            if (buf.handle != VK_NULL_HANDLE)
                vmaDestroyBuffer(allocator_, buf.handle, buf.allocation);
        }
    }
    dlBuckets_.clear();

    // Destroy all pooled buffers (readback bucket pool)
    for (auto& [bucketSize, vec] : readbackBuckets_) {
        for (auto& buf : vec) {
            if (buf.handle != VK_NULL_HANDLE)
                vmaDestroyBuffer(allocator_, buf.handle, buf.allocation);
        }
    }
    readbackBuckets_.clear();

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

    // REQUIRE device-local memory.
    //
    // Background: ``preferredFlags`` is a soft hint VMA can ignore. Combined
    // with ``HOST_ACCESS_SEQUENTIAL_WRITE_BIT``, on Windows + AMD/RDNA the
    // auto-selector lands on memoryType[1] (HOST_VISIBLE | HOST_COHERENT
    // *only* — no DEVICE_LOCAL, no HOST_CACHED). The buffer ends up in
    // host-uncached BAR memory, and every GPU read becomes a single-byte
    // PCIe transaction → measured 0.1 GB/s effective bandwidth, ~100x slower
    // than CPU/numpy. See sandbox/vsa_lm/grilly_gpu_path_test.py for the
    // smoking-gun profile (gc.relu on 4.7M elements: 757 ms).
    //
    // Using ``requiredFlags`` forces VMA to *only* consider memory types with
    // DEVICE_LOCAL_BIT. On systems with Resizable BAR (the common case for
    // modern AMD + Windows + AGESA 2020+), VMA picks the
    // DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT memory type — full VRAM
    // bandwidth (~432 GB/s on RX 6750 XT) + CPU mapping for fast uploads.
    //
    // On systems without ReBAR, this allocation will FAIL — which is the
    // correct behavior, because the silent slow path was a footgun. Users
    // without ReBAR should enable it in BIOS, or callers needing the legacy
    // host-mapped path should use ``acquirePreferDeviceLocal`` which is
    // explicitly soft-preferred for that case.
    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
                      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
    allocInfo.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

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

    // Route to the right bucket pool based on memory class. The three pools
    // MUST stay separate:
    //   - dlBuckets_       : DEVICE_LOCAL only (mappedPtr=null, GPU compute)
    //   - readbackBuckets_ : HOST_CACHED random read (CPU reads from GPU)
    //   - buckets_         : default WC sequential write (CPU writes to GPU)
    // Picking up a DL buffer via ``acquire()`` would crash trying to memcpy
    // into a null mappedPtr; picking up a WC buffer via ``acquireReadback``
    // would silently destroy CPU-read perf (the original bug we're fixing).
    auto& vec = buf.deviceLocal ? dlBuckets_[buf.bucketSize]
              : buf.readback    ? readbackBuckets_[buf.bucketSize]
                                 : buckets_[buf.bucketSize];
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

    std::lock_guard<std::mutex> lock(mutex_);
    stats_.totalAcquired++;

    // Try reuse from the DL bucket pool first (LIFO returns the same handle
    // most often, which keeps the descriptor cache hitting on repeat calls).
    auto it = dlBuckets_.find(bucket);
    if (it != dlBuckets_.end() && !it->second.empty()) {
        GrillyBuffer buf = it->second.back();
        it->second.pop_back();
        buf.size = size;
        stats_.hits++;
        return buf;
    }

    stats_.misses++;
    stats_.allocations++;

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = bucket;
    // We need both TRANSFER_DST (for stage-in copies) and TRANSFER_SRC (for
    // stage-out copies) since the staging pattern in cpp/src/ops/linear.cpp
    // copies output back from DL → host-visible staging.
    bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                       VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                       VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    allocInfo.priority = 1.0f;  // Maximum priority — keep in VRAM, don't evict

    GrillyBuffer buf{};
    buf.bucketSize = bucket;
    buf.size = size;
    buf.deviceLocal = true;  // routes to dlBuckets_ on release

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

    std::lock_guard<std::mutex> lock(mutex_);
    stats_.totalAcquired++;

    // Try reuse from the readback bucket pool first.
    auto it = readbackBuckets_.find(bucket);
    if (it != readbackBuckets_.end() && !it->second.empty()) {
        GrillyBuffer buf = it->second.back();
        it->second.pop_back();
        buf.size = size;
        stats_.hits++;
        return buf;
    }

    stats_.misses++;
    stats_.allocations++;

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
    buf.readback = true;  // routes to readbackBuckets_ on release

    vkCheck(vmaCreateBuffer(allocator_, &bufferInfo, &allocInfo, &buf.handle,
                            &buf.allocation, &buf.info),
            "vmaCreateBuffer (readback) failed");

    buf.mappedPtr = buf.info.pMappedData;
    return buf;
}

// ── Persistent transfer context ────────────────────────────────────────────

void BufferPool::ensureTransferContext() {
    if (transferInitialized_) return;

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = device_.queueFamily();
    vkCheck(vkCreateCommandPool(device_.device(), &poolInfo, nullptr, &transferPool_),
            "transfer pool creation failed");

    VkCommandBufferAllocateInfo cmdAllocInfo{};
    cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAllocInfo.commandPool = transferPool_;
    cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandBufferCount = 1;
    vkCheck(vkAllocateCommandBuffers(device_.device(), &cmdAllocInfo, &transferCmd_),
            "transfer cmd alloc failed");

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    vkCheck(vkCreateFence(device_.device(), &fenceInfo, nullptr, &transferFence_),
            "transfer fence creation failed");

    transferInitialized_ = true;
}

void BufferPool::transferSubmitAndWait() {
    vkEndCommandBuffer(transferCmd_);

    vkWaitForFences(device_.device(), 1, &transferFence_, VK_TRUE, UINT64_MAX);
    vkResetFences(device_.device(), 1, &transferFence_);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &transferCmd_;

    vkCheck(vkQueueSubmit(device_.computeQueue(), 1, &submitInfo, transferFence_),
            "transfer submit failed");
    vkCheck(vkWaitForFences(device_.device(), 1, &transferFence_, VK_TRUE, UINT64_MAX),
            "transfer wait failed");
}

void BufferPool::uploadStaged(GrillyBuffer& deviceBuf, const void* data,
                               size_t bytes) {
    ensureTransferContext();

    // 1. Create staging buffer (TODO: pool these too)
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

    // 2. Copy data into staging
    std::memcpy(stagingMemInfo.pMappedData, data, bytes);
    vmaFlushAllocation(allocator_, stagingMem, 0, bytes);

    // 3. Record transfer using persistent context
    vkResetCommandBuffer(transferCmd_, 0);
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(transferCmd_, &beginInfo);

    VkBufferCopy copyRegion{};
    copyRegion.size = bytes;
    vkCmdCopyBuffer(transferCmd_, stagingBuf, deviceBuf.handle, 1, &copyRegion);

    // 4. Submit + wait (reuses persistent fence)
    transferSubmitAndWait();

    // 5. Cleanup staging only
    vmaDestroyBuffer(allocator_, stagingBuf, stagingMem);
}

void BufferPool::downloadStaged(const GrillyBuffer& deviceBuf, void* out,
                                  size_t bytes) {
    ensureTransferContext();

    // 1. Create staging readback buffer (TODO: pool these)
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

    // 2. Record copy using persistent transfer context
    vkResetCommandBuffer(transferCmd_, 0);
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(transferCmd_, &beginInfo);

    VkBufferCopy copyRegion{};
    copyRegion.size = bytes;
    vkCmdCopyBuffer(transferCmd_, deviceBuf.handle, stagingBuf, 1, &copyRegion);

    // 3. Submit + wait (reuses persistent fence)
    transferSubmitAndWait();

    // 4. Invalidate + copy to output
    vmaInvalidateAllocation(allocator_, stagingMem, 0, bytes);
    std::memcpy(out, stagingMemInfo.pMappedData, bytes);

    // 5. Cleanup staging only
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
