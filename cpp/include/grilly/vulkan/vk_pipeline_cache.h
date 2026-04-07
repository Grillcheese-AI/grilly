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
    PipelineCache(GrillyDevice& device, uint32_t maxDescriptorSets = 500,
                  uint32_t maxStorageBuffers = 1000);
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

    static constexpr size_t kMaxCachedDescSets = 100;

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
