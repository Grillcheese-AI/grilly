#include "grilly/experimental/paged_latent_pool.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace grilly {
namespace experimental {

// ── ReBAR / SAM Detection ───────────────────────────────────────────────
//
// AMD Smart Access Memory (SAM) and NVIDIA Resizable BAR expose the GPU's
// entire VRAM through a PCIe BAR that the CPU can map directly. VMA
// reports this as a memory type with both DEVICE_LOCAL and HOST_VISIBLE.
//
// To detect ReBAR availability, we query VMA for memory types and check
// if any has both flags. If found, allocations using our buildAllocInfo()
// will land in this memory — CPU writes go directly to VRAM.
//
// Without ReBAR, VMA falls back to HOST_VISIBLE-only memory (system RAM
// with GPU access via PCIe), which is slower for GPU reads but still
// works correctly. The persistent mapping means upload is still just
// memcpy — no vkMapMemory/vkUnmapMemory overhead.

static bool detectReBAR(VmaAllocator allocator) {
    // Check if any memory type has both DEVICE_LOCAL and HOST_VISIBLE
    const VkPhysicalDeviceMemoryProperties* memProps = nullptr;
    vmaGetMemoryProperties(allocator, &memProps);

    if (!memProps) return false;

    for (uint32_t i = 0; i < memProps->memoryTypeCount; ++i) {
        uint32_t flags = memProps->memoryTypes[i].propertyFlags;
        if ((flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
            (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
            return true;
        }
    }
    return false;
}

PagedLatentPool::PagedLatentPool(GrillyDevice& device,
                                  const PagedLatentPoolConfig& config)
    : device_(device), config_(config) {
    // Create VMA allocator (separate from BufferPool's allocator for
    // independent memory management)
    VmaAllocatorCreateInfo allocInfo{};
    allocInfo.vulkanApiVersion = VK_API_VERSION_1_1;
    allocInfo.physicalDevice = device_.physicalDevice();
    allocInfo.device = device_.device();
    allocInfo.instance = device_.instance();

    if (vmaCreateAllocator(&allocInfo, &allocator_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create VMA allocator for paged pool");
    }

    hasReBAR_ = detectReBAR(allocator_);

    // Pre-calculate total pages needed
    if (config_.tokensPerPage == 0) {
        config_.tokensPerPage = 256;  // default
    }
    uint32_t totalPages = (config_.maxTokens + config_.tokensPerPage - 1) /
                          config_.tokensPerPage;

    // Reserve page vector (but don't allocate pages yet — lazy allocation)
    pages_.resize(totalPages);
    for (uint32_t i = 0; i < totalPages; ++i) {
        pages_[i].capacity = config_.tokensPerPage;
        pages_[i].active = false;
    }

    tokenInfo_.reserve(config_.maxTokens);
}

PagedLatentPool::~PagedLatentPool() {
    // Free all active pages
    for (auto& page : pages_) {
        if (page.active && page.buffer != VK_NULL_HANDLE) {
            vmaDestroyBuffer(allocator_, page.buffer, page.allocation);
        }
    }
    if (allocator_ != VK_NULL_HANDLE) {
        vmaDestroyAllocator(allocator_);
    }
}

VmaAllocationCreateInfo PagedLatentPool::buildAllocInfo() const {
    VmaAllocationCreateInfo allocInfo{};
    allocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;

    if ((config_.memoryFlags & MEM_DEVICE_LOCAL) &&
        (config_.memoryFlags & MEM_HOST_VISIBLE)) {
        // ReBAR/SAM path: request DEVICE_LOCAL memory with host write access.
        // VMA will automatically select a memory type that has both flags
        // if available (ReBAR), or fall back to HOST_VISIBLE-only if not.
        allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
        allocInfo.flags |= VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
        allocInfo.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
        allocInfo.preferredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    } else if (config_.memoryFlags & MEM_DEVICE_LOCAL) {
        allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    } else {
        allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
        allocInfo.flags |= VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
    }

    return allocInfo;
}

uint32_t PagedLatentPool::allocatePage() {
    std::lock_guard<std::mutex> lock(mutex_);

    // Find a free page
    uint32_t pageIdx = UINT32_MAX;

    if (!freePages_.empty()) {
        pageIdx = freePages_.back();
        freePages_.pop_back();
    } else {
        // Find first inactive page
        for (uint32_t i = 0; i < pages_.size(); ++i) {
            if (!pages_[i].active) {
                pageIdx = i;
                break;
            }
        }
    }

    if (pageIdx == UINT32_MAX) {
        throw std::runtime_error("Paged latent pool exhausted — all pages allocated");
    }

    auto& page = pages_[pageIdx];

    // Allocate Vulkan buffer for this page
    size_t pageBytes = size_t(page.capacity) * config_.numHeads *
                       config_.latentDim * sizeof(float);

    VkBufferCreateInfo bufInfo{};
    bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufInfo.size = pageBytes;
    bufInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocCI = buildAllocInfo();
    VmaAllocationInfo resultInfo{};

    VkResult result = vmaCreateBuffer(allocator_, &bufInfo, &allocCI,
                                       &page.buffer, &page.allocation,
                                       &resultInfo);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate paged latent buffer");
    }

    page.mappedPtr = resultInfo.pMappedData;
    page.tokenCount = 0;
    page.active = true;

    return pageIdx;
}

void PagedLatentPool::freePage(uint32_t pageIndex) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (pageIndex >= pages_.size() || !pages_[pageIndex].active) return;

    auto& page = pages_[pageIndex];
    vmaDestroyBuffer(allocator_, page.buffer, page.allocation);
    page.buffer = VK_NULL_HANDLE;
    page.allocation = VK_NULL_HANDLE;
    page.mappedPtr = nullptr;
    page.tokenCount = 0;
    page.active = false;

    freePages_.push_back(pageIndex);
}

float* PagedLatentPool::getWritePtr(uint32_t pageIndex, uint32_t tokenOffset) {
    if (pageIndex >= pages_.size() || !pages_[pageIndex].active) {
        throw std::runtime_error("Invalid page index");
    }
    auto& page = pages_[pageIndex];
    if (tokenOffset >= page.capacity) {
        throw std::runtime_error("Token offset exceeds page capacity");
    }
    size_t byteOffset = size_t(tokenOffset) * config_.numHeads *
                        config_.latentDim * sizeof(float);
    return reinterpret_cast<float*>(
        static_cast<uint8_t*>(page.mappedPtr) + byteOffset);
}

const float* PagedLatentPool::getReadPtr(uint32_t pageIndex,
                                          uint32_t tokenOffset) const {
    if (pageIndex >= pages_.size() || !pages_[pageIndex].active) {
        throw std::runtime_error("Invalid page index");
    }
    const auto& page = pages_[pageIndex];
    size_t byteOffset = size_t(tokenOffset) * config_.numHeads *
                        config_.latentDim * sizeof(float);
    return reinterpret_cast<const float*>(
        static_cast<const uint8_t*>(page.mappedPtr) + byteOffset);
}

void PagedLatentPool::writeLatent(uint32_t pageIndex, uint32_t tokenOffset,
                                   const float* latent, uint32_t latentDim) {
    float* dst = getWritePtr(pageIndex, tokenOffset);
    std::memcpy(dst, latent, size_t(config_.numHeads) * latentDim * sizeof(float));

    // Update token count if this extends the page
    auto& page = pages_[pageIndex];
    if (tokenOffset >= page.tokenCount) {
        page.tokenCount = tokenOffset + 1;
    }
}

VkBuffer PagedLatentPool::getPageBuffer(uint32_t pageIndex) const {
    if (pageIndex >= pages_.size() || !pages_[pageIndex].active) {
        throw std::runtime_error("Invalid page index");
    }
    return pages_[pageIndex].buffer;
}

VkDeviceSize PagedLatentPool::getPageBufferSize(uint32_t pageIndex) const {
    if (pageIndex >= pages_.size() || !pages_[pageIndex].active) {
        throw std::runtime_error("Invalid page index");
    }
    return VkDeviceSize(pages_[pageIndex].capacity) * config_.numHeads *
           config_.latentDim * sizeof(float);
}

std::vector<std::pair<uint32_t, uint32_t>> PagedLatentPool::appendTokens(
    const float* latents, uint32_t numTokens, uint32_t latentDim) {
    std::vector<std::pair<uint32_t, uint32_t>> locations;
    locations.reserve(numTokens);

    uint32_t remaining = numTokens;
    uint32_t srcOffset = 0;

    while (remaining > 0) {
        // Find a page with space, or allocate a new one
        uint32_t targetPage = UINT32_MAX;
        for (uint32_t i = 0; i < pages_.size(); ++i) {
            if (pages_[i].active &&
                pages_[i].tokenCount < pages_[i].capacity) {
                targetPage = i;
                break;
            }
        }
        if (targetPage == UINT32_MAX) {
            targetPage = allocatePage();
        }

        auto& page = pages_[targetPage];
        uint32_t space = page.capacity - page.tokenCount;
        uint32_t toWrite = std::min(remaining, space);

        for (uint32_t t = 0; t < toWrite; ++t) {
            uint32_t offset = page.tokenCount + t;
            size_t srcBytes = size_t(config_.numHeads) * latentDim;
            const float* src = latents + (srcOffset + t) * srcBytes;
            writeLatent(targetPage, offset, src, latentDim);
            locations.push_back({targetPage, offset});
        }

        page.tokenCount += toWrite;
        remaining -= toWrite;
        srcOffset += toWrite;
    }

    return locations;
}

uint32_t PagedLatentPool::totalTokens() const {
    uint32_t total = 0;
    for (const auto& page : pages_) {
        if (page.active) total += page.tokenCount;
    }
    return total;
}

uint32_t PagedLatentPool::allocatedPages() const {
    uint32_t count = 0;
    for (const auto& page : pages_) {
        if (page.active) count++;
    }
    return count;
}

PagedLatentPool::PoolStats PagedLatentPool::stats() const {
    PoolStats s{};
    s.totalPages = static_cast<uint32_t>(pages_.size());
    s.maxTokens = config_.maxTokens;
    s.rebarAvailable = hasReBAR_;

    for (const auto& page : pages_) {
        if (page.active) {
            s.activePages++;
            s.totalTokens += page.tokenCount;
            s.totalMemoryBytes += size_t(page.capacity) * config_.numHeads *
                                  config_.latentDim * sizeof(float);
            s.usedMemoryBytes += size_t(page.tokenCount) * config_.numHeads *
                                 config_.latentDim * sizeof(float);
        }
    }

    return s;
}

}  // namespace experimental
}  // namespace grilly
