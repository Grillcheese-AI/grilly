#pragma once
/// Vulkan pipeline and descriptor set cache.

#include <vulkan/vulkan.h>

#include <cstdint>
#include <list>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "grilly/vulkan/vk_device.h"

namespace grilly {

/// Cached pipeline + layout + descriptor set layout for a single shader.
struct PipelineEntry {
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout descLayout = VK_NULL_HANDLE;
    VkShaderModule shaderModule = VK_NULL_HANDLE;
};

/// Pipeline and descriptor set cache with LRU eviction.
class PipelineCache {
public:
    // Pool must hold MORE sets than kMaxCachedDescSets (so eviction frees a
    // cached set before the pool itself is exhausted -- the OOM "evict all" path
    // would otherwise free IN-FLIGHT sets). maxStorageBuffers covers
    // kMaxCachedDescSets * (max bindings/set, <=8) with margin.
    PipelineCache(GrillyDevice& device, uint32_t maxDescriptorSets = 8192,
                  uint32_t maxStorageBuffers = 131072);
    ~PipelineCache();

    PipelineCache(const PipelineCache&) = delete;
    PipelineCache& operator=(const PipelineCache&) = delete;

    void loadSPIRV(const std::string& name, const std::vector<uint8_t>& code);
    void loadSPIRVFile(const std::string& name, const std::string& path);

    PipelineEntry getOrCreate(const std::string& name, uint32_t numBuffers,
                              uint32_t pushConstSize = 0);

    VkDescriptorSet allocDescriptorSet(
        const std::string& name,
        const std::vector<VkDescriptorBufferInfo>& buffers);

    /// Free ALL cached descriptor sets and reset the LRU. Call ONLY when the GPU
    /// is idle (no in-flight command buffer references the sets). The descriptor
    /// cache is keyed by (shader, buffer handles); across steps the buffer pool
    /// recycles handles, so a stale cached set can be falsely reused for a
    /// different logical buffer -> nondeterministic/wrong results. Clearing at
    /// each step boundary (TapeContext::begin(), where prior synchronous submits
    /// have completed) forces correct fresh bindings each step.
    void clearDescriptorCache();

    bool hasShader(const std::string& name) const {
        return spirvCode_.count(name) > 0;
    }

    /// Access the underlying device for capability queries
    /// (e.g. ``hasCooperativeMatrix()``).
    GrillyDevice& getDevice() { return device_; }
    const GrillyDevice& getDevice() const { return device_; }

    struct CacheStats {
        uint64_t hits = 0;
        uint64_t misses = 0;
        uint64_t evictions = 0;
        size_t cachedSets = 0;
    };
    CacheStats cacheStats() const;

private:
    VkDescriptorSetLayout createDescLayout(uint32_t numBuffers);
    VkPipelineLayout createPipeLayout(VkDescriptorSetLayout descLayout,
                                       uint32_t pushConstSize);
    VkPipeline createPipeline(const std::vector<uint8_t>& spirv,
                              VkPipelineLayout pipeLayout,
                              VkShaderModule& outModule);

    GrillyDevice& device_;
    VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;

    std::unordered_map<std::string, std::vector<uint8_t>> spirvCode_;
    std::unordered_map<std::string, PipelineEntry> pipelines_;

    struct DescCacheKey {
        std::string shaderName;
        std::vector<std::pair<VkBuffer, VkDeviceSize>> bufferBindings;
        bool operator==(const DescCacheKey& o) const;
    };
    struct DescCacheKeyHash {
        size_t operator()(const DescCacheKey& k) const;
    };

    // INVARIANT: must EXCEED the number of distinct descriptor sets allocated in
    // any single un-submitted command batch. The descriptor-set cache is keyed by
    // (shader, buffer handles), so every dispatch in a batch with fresh buffers
    // takes a distinct set; on a miss when full, the LRU evicts+frees a set. If
    // that evicted set is still recorded in the in-flight command buffer (i.e. the
    // batch needs > kMaxCachedDescSets sets), the submit reads a freed set ->
    // garbage/zero results. The Cubby trunk's backward batch is ~15 sets/layer, so
    // L=6 (~90) was just under the old cap of 100 and L>=12 silently corrupted
    // grads. 4096 covers ~L=250 with margin; pool maxSets (8192) stays above it so
    // the pool never OOMs first.
    static constexpr size_t kMaxCachedDescSets = 4096;

    using LRUList = std::list<DescCacheKey>;
    LRUList lruList_;
    struct DescCacheEntry {
        VkDescriptorSet set;
        LRUList::iterator lruIter;
    };
    std::unordered_map<DescCacheKey, DescCacheEntry, DescCacheKeyHash> descCache_;

    mutable std::mutex mutex_;
    CacheStats stats_{};
};

}  // namespace grilly
