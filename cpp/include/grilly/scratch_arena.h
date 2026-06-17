#pragma once
/// Scratch arena allocator for temporary GPU buffers.
///
/// Reduces allocation overhead by batching buffer acquisitions and releasing
/// them all at once after a computation phase (e.g., forward/backward pass).
///
/// Usage:
///   ScratchArena arena(backend.pool());
///   auto buf1 = arena.allocateDeviceLocal(size1);
///   auto buf2 = arena.acquire(size2);
///   // ... use buffers ...
///   arena.reset();  // Returns all buffers to pool at once

#include "grilly/vulkan/vk_buffer_pool.h"
#include <memory>
#include <vector>

namespace grilly {

class ScratchArena {
public:
    explicit ScratchArena(BufferPool& pool) : pool_(pool) {}
    
    ~ScratchArena() {
        reset();
    }
    
    ScratchArena(const ScratchArena&) = delete;
    ScratchArena& operator=(const ScratchArena&) = delete;
    
    /// Acquire a host-visible buffer from the arena
    GrillyBuffer acquire(size_t size) {
        auto buf = pool_.acquire(size);
        acquired_.push_back(buf);
        return buf;
    }
    
    /// Acquire a device-local (GPU-only) buffer from the arena
    GrillyBuffer acquireDeviceLocal(size_t size) {
        auto buf = pool_.acquireDeviceLocal(size);
        acquired_.push_back(buf);
        return buf;
    }
    
    /// Acquire a readback-optimized buffer from the arena
    GrillyBuffer acquireReadback(size_t size) {
        auto buf = pool_.acquireReadback(size);
        acquired_.push_back(buf);
        return buf;
    }
    
    /// Acquire a prefer-device-local buffer from the arena
    GrillyBuffer acquirePreferDeviceLocal(size_t size) {
        auto buf = pool_.acquirePreferDeviceLocal(size);
        acquired_.push_back(buf);
        return buf;
    }
    
    /// Return all acquired buffers to the pool at once.
    /// Call this after a batch of operations completes (e.g., after endBatch()).
    void reset() {
        for (auto& buf : acquired_) {
            if (buf.handle != VK_NULL_HANDLE) {
                pool_.release(buf);
            }
        }
        acquired_.clear();
    }
    
    /// Number of buffers currently held by this arena
    size_t bufferCount() const {
        return acquired_.size();
    }
    
    /// Total bytes allocated (approximate, uses bucket sizes)
    size_t totalBytes() const {
        size_t total = 0;
        for (const auto& buf : acquired_) {
            total += buf.bucketSize;
        }
        return total;
    }

private:
    BufferPool& pool_;
    std::vector<GrillyBuffer> acquired_;
};

}  // namespace grilly
