#pragma once

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include <cstdint>
#include <mutex>
#include <vector>

#include "grilly/device.h"

namespace grilly {
namespace experimental {

// ── Paged Latent Vector Pool ────────────────────────────────────────────
//
// A memory pool specifically designed for MLA-compressed KV cache latents,
// using AMD SAM (Smart Access Memory) / ReBAR (Resizable BAR) for direct
// CPU→VRAM writes.
//
// Traditional GPU memory flow:
//   CPU → staging buffer (HOST_VISIBLE) → copy → VRAM (DEVICE_LOCAL)
//   Two allocations, one explicit transfer, one fence wait.
//
// ReBAR flow:
//   CPU → VRAM (DEVICE_LOCAL | HOST_VISIBLE)
//   One allocation, zero transfers. CPU writes land directly in VRAM
//   at PCIe bandwidth (~14 GB/s on PCIe 4.0 x16).
//
// On AMD RDNA 2 with SAM enabled, the full 12 GB VRAM is mapped into
// the CPU's address space. VMA automatically selects this memory type
// when we request DEVICE_LOCAL + HOST_VISIBLE with
// VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT.
//
// The "paged" design uses fixed-size pages of latent vectors. Each page
// holds a configurable number of token latents. Pages can be independently
// allocated, mapped, and freed — enabling efficient partial cache eviction
// without defragmenting the entire pool.
//
// Memory flags (matching the user's API):
//   MEM_DEVICE_LOCAL  = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
//   MEM_HOST_VISIBLE  = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
//   Combined = ReBAR/SAM "instant host-write" memory

/// Memory flag constants matching the grilly Python API
enum MemoryFlags : uint32_t {
    MEM_DEVICE_LOCAL  = 0x01,
    MEM_HOST_VISIBLE  = 0x02,
    MEM_HOST_COHERENT = 0x04,
    MEM_HOST_CACHED   = 0x08,
};

/// Configuration for the paged latent pool
struct PagedLatentPoolConfig {
    uint32_t maxTokens;         // Maximum total tokens across all pages
    uint32_t latentDim;         // Latent vector dimension (d_h / compression_ratio)
    uint32_t numHeads;          // Number of attention heads
    uint32_t tokensPerPage;     // Tokens per page (default: 256)
    uint32_t memoryFlags;       // MEM_DEVICE_LOCAL | MEM_HOST_VISIBLE etc.
};

/// A single page in the latent pool
struct LatentPage {
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    void* mappedPtr = nullptr;      // Persistent mapping (ReBAR direct access)
    uint32_t tokenCount = 0;        // Current tokens stored in this page
    uint32_t capacity = 0;          // Max tokens this page can hold
    bool active = false;            // Is this page currently allocated?
};

/// Per-token metadata for the paged pool
struct LatentTokenInfo {
    uint32_t pageIndex;             // Which page this token lives in
    uint32_t offsetInPage;          // Offset within the page
    uint32_t globalTokenId;         // Original sequence position
    float cumulativeScore;          // H2O eviction score
};

/// Paged latent vector pool with ReBAR/SAM support.
///
/// Usage:
///   auto pool = PagedLatentPool(device, config);
///   uint32_t page = pool.allocatePage();
///   float* ptr = pool.getWritePtr(page, tokenOffset);
///   memcpy(ptr, latentData, bytes);  // Direct VRAM write via ReBAR
///   // No staging buffer, no vkCmdCopy, no fence wait needed
///
class PagedLatentPool {
public:
    PagedLatentPool(GrillyDevice& device, const PagedLatentPoolConfig& config);
    ~PagedLatentPool();

    PagedLatentPool(const PagedLatentPool&) = delete;
    PagedLatentPool& operator=(const PagedLatentPool&) = delete;

    /// Allocate a new page. Returns page index.
    uint32_t allocatePage();

    /// Free a page (returns it to the free list).
    void freePage(uint32_t pageIndex);

    /// Get a direct write pointer to a token's latent vector within a page.
    /// On ReBAR/SAM hardware, this is a direct VRAM pointer — writes land
    /// in GPU memory immediately without any transfer command.
    float* getWritePtr(uint32_t pageIndex, uint32_t tokenOffset);

    /// Get a read pointer (for CPU-side verification)
    const float* getReadPtr(uint32_t pageIndex, uint32_t tokenOffset) const;

    /// Write a latent vector to the pool (convenience wrapper around getWritePtr)
    void writeLatent(uint32_t pageIndex, uint32_t tokenOffset,
                     const float* latent, uint32_t latentDim);

    /// Get the Vulkan buffer handle for a page (for GPU shader binding)
    VkBuffer getPageBuffer(uint32_t pageIndex) const;

    /// Get buffer size for a page
    VkDeviceSize getPageBufferSize(uint32_t pageIndex) const;

    /// Append tokens to the pool. Automatically allocates new pages as needed.
    /// Returns the (pageIndex, offsetInPage) for each appended token.
    std::vector<std::pair<uint32_t, uint32_t>> appendTokens(
        const float* latents, uint32_t numTokens, uint32_t latentDim);

    /// Get total number of tokens currently stored
    uint32_t totalTokens() const;

    /// Get total number of allocated pages
    uint32_t allocatedPages() const;

    /// Check if ReBAR/SAM is available (DEVICE_LOCAL + HOST_VISIBLE memory)
    bool hasReBAR() const { return hasReBAR_; }

    const PagedLatentPoolConfig& config() const { return config_; }

    struct PoolStats {
        uint32_t totalPages;
        uint32_t activePages;
        uint32_t totalTokens;
        uint32_t maxTokens;
        bool rebarAvailable;
        size_t totalMemoryBytes;
        size_t usedMemoryBytes;
    };
    PoolStats stats() const;

private:
    VmaAllocationCreateInfo buildAllocInfo() const;

    GrillyDevice& device_;
    VmaAllocator allocator_ = VK_NULL_HANDLE;
    PagedLatentPoolConfig config_;
    bool hasReBAR_ = false;

    std::vector<LatentPage> pages_;
    std::vector<uint32_t> freePages_;  // Free page indices
    std::vector<LatentTokenInfo> tokenInfo_;

    mutable std::mutex mutex_;
};

}  // namespace experimental
}  // namespace grilly
