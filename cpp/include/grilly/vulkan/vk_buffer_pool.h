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
    /// True if this buffer was acquired via ``acquireDeviceLocal`` (no host
    /// visibility, mapped to GPU-only VRAM). ``release`` uses this to route
    /// the buffer back to the device-local bucket pool, so a regular
    /// ``acquire`` won't accidentally pick up a DL buffer and crash trying
    /// to memcpy into a null mappedPtr.
    bool deviceLocal = false;
    /// True if this buffer was acquired via ``acquireReadback`` — backed
    /// by HOST_CACHED memory for fast CPU read-after-GPU-write. Released
    /// buffers go to a separate readback pool so a future ``acquire``
    /// (which expects WC sequential-write memory) doesn't waste a cached
    /// readback slot on a write-only workload.
    bool readback = false;
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
    void download(const GrillyBuffer& buf, float* out, size_t bytes, size_t srcOffset = 0);
    void downloadStaged(const GrillyBuffer& deviceBuf, void* out, size_t bytes, size_t srcOffset = 0);

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
    /// Pool for host-visible WC buffers (acquire / release default path).
    /// Optimized for CPU sequential writes (memcpy CPU → buffer).
    std::unordered_map<size_t, std::vector<GrillyBuffer>> buckets_;
    /// Pool for DEVICE_LOCAL-only buffers (acquireDeviceLocal / release).
    /// Kept separate so a host-visible acquire never picks up a DL buffer.
    std::unordered_map<size_t, std::vector<GrillyBuffer>> dlBuckets_;
    /// Pool for HOST_CACHED readback buffers (acquireReadback / release).
    /// Optimized for CPU random reads (memcpy buffer → CPU output) at
    /// cached system RAM speed (~10 GB/s vs ~25 MB/s for WC reads).
    std::unordered_map<size_t, std::vector<GrillyBuffer>> readbackBuckets_;
    Stats stats_{};

    // Persistent transfer context (avoids cmd pool/fence recreation per transfer)
    VkCommandPool transferPool_ = VK_NULL_HANDLE;
    VkCommandBuffer transferCmd_ = VK_NULL_HANDLE;
    VkFence transferFence_ = VK_NULL_HANDLE;
    bool transferInitialized_ = false;

    void ensureTransferContext();
    void transferSubmitAndWait();
};

}  // namespace grilly
