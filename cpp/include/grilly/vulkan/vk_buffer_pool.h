#pragma once
/// Vulkan buffer pool implementation (VMA-backed).

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "grilly/vulkan/vk_device.h"

namespace grilly {

/// A single GPU buffer backed by VMA.
struct GrillyBuffer {
    VkBuffer handle = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VmaAllocationInfo info{};
    size_t size = 0;
    size_t bucketSize = 0;
    void* mappedPtr = nullptr;
};

/// VMA-backed buffer pool with power-of-2 bucket reuse.
class BufferPool {
public:
    explicit BufferPool(GrillyDevice& device);
    ~BufferPool();

    BufferPool(const BufferPool&) = delete;
    BufferPool& operator=(const BufferPool&) = delete;

    GrillyBuffer acquire(size_t size);
    GrillyBuffer acquireDeviceLocal(size_t size);
    GrillyBuffer acquirePreferDeviceLocal(size_t size);
    GrillyBuffer acquireReadback(size_t size);
    void release(GrillyBuffer& buf);

    void upload(GrillyBuffer& buf, const float* data, size_t bytes);
    void uploadStaged(GrillyBuffer& deviceBuf, const void* data, size_t bytes);
    void download(const GrillyBuffer& buf, float* out, size_t bytes);
    void downloadStaged(const GrillyBuffer& deviceBuf, void* out, size_t bytes);

    VmaAllocator allocator() const { return allocator_; }

    struct Stats {
        uint64_t hits = 0;
        uint64_t misses = 0;
        uint64_t allocations = 0;
        uint64_t totalAcquired = 0;
        uint64_t totalReleased = 0;
    };
    Stats stats() const;

private:
    static constexpr int kMinBucketPower = 8;
    static constexpr int kMaxBucketPower = 28;
    static constexpr size_t kMaxBuffersPerBucket = 32;

    size_t sizeToBucket(size_t size) const;
    GrillyBuffer allocateBuffer(size_t bucketSize);

    GrillyDevice& device_;
    VmaAllocator allocator_ = VK_NULL_HANDLE;
    std::mutex mutex_;
    std::unordered_map<size_t, std::vector<GrillyBuffer>> buckets_;
    Stats stats_{};
};

}  // namespace grilly
