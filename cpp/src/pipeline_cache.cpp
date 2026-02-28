#include "grilly/pipeline_cache.h"

#include <fstream>
#include <functional>
#include <stdexcept>

namespace grilly {

// ── Helper ──────────────────────────────────────────────────────────────────
static void vkCheck(VkResult result, const char* msg) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(std::string(msg) +
                                 " (VkResult=" + std::to_string(result) + ")");
    }
}

// ── DescCacheKey equality + hash ────────────────────────────────────────────

bool PipelineCache::DescCacheKey::operator==(const DescCacheKey& o) const {
    return shaderName == o.shaderName && bufferBindings == o.bufferBindings;
}

size_t PipelineCache::DescCacheKeyHash::operator()(
    const DescCacheKey& k) const {
    size_t h = std::hash<std::string>{}(k.shaderName);
    for (const auto& [buf, sz] : k.bufferBindings) {
        // Mix buffer handle (pointer-sized) and size into the hash
        h ^= std::hash<uint64_t>{}(reinterpret_cast<uint64_t>(buf)) +
             0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<VkDeviceSize>{}(sz) + 0x9e3779b9 + (h << 6) +
             (h >> 2);
    }
    return h;
}

// ── Construction / destruction ──────────────────────────────────────────────

PipelineCache::PipelineCache(GrillyDevice& device,
                             uint32_t maxDescriptorSets,
                             uint32_t maxStorageBuffers)
    : device_(device) {
    // Create descriptor pool (mirrors core.py:378-391)
    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = maxStorageBuffers;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    poolInfo.maxSets = maxDescriptorSets;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;

    vkCheck(vkCreateDescriptorPool(device_.device(), &poolInfo, nullptr,
                                   &descriptorPool_),
            "vkCreateDescriptorPool failed");
}

PipelineCache::~PipelineCache() {
    VkDevice dev = device_.device();
    if (dev == VK_NULL_HANDLE)
        return;

    // Destroy all cached pipelines
    for (auto& [name, entry] : pipelines_) {
        if (entry.pipeline != VK_NULL_HANDLE)
            vkDestroyPipeline(dev, entry.pipeline, nullptr);
        if (entry.layout != VK_NULL_HANDLE)
            vkDestroyPipelineLayout(dev, entry.layout, nullptr);
        if (entry.descLayout != VK_NULL_HANDLE)
            vkDestroyDescriptorSetLayout(dev, entry.descLayout, nullptr);
        if (entry.shaderModule != VK_NULL_HANDLE)
            vkDestroyShaderModule(dev, entry.shaderModule, nullptr);
    }

    if (descriptorPool_ != VK_NULL_HANDLE)
        vkDestroyDescriptorPool(dev, descriptorPool_, nullptr);
}

// ── SPIR-V loading ──────────────────────────────────────────────────────────

void PipelineCache::loadSPIRV(const std::string& name,
                               const std::vector<uint8_t>& code) {
    std::lock_guard<std::mutex> lock(mutex_);
    spirvCode_[name] = code;
}

void PipelineCache::loadSPIRVFile(const std::string& name,
                                   const std::string& path) {
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("Failed to open shader: " + path);

    size_t fileSize = static_cast<size_t>(file.tellg());
    std::vector<uint8_t> code(fileSize);
    file.seekg(0);
    file.read(reinterpret_cast<char*>(code.data()), fileSize);

    loadSPIRV(name, code);
}

// ── Internal: create Vulkan objects ─────────────────────────────────────────

VkDescriptorSetLayout PipelineCache::createDescLayout(uint32_t numBuffers) {
    // Mirrors pipelines.py:29-48
    std::vector<VkDescriptorSetLayoutBinding> bindings(numBuffers);
    for (uint32_t i = 0; i < numBuffers; ++i) {
        bindings[i] = {};
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = numBuffers;
    layoutInfo.pBindings = bindings.data();

    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    vkCheck(
        vkCreateDescriptorSetLayout(device_.device(), &layoutInfo, nullptr,
                                    &layout),
        "vkCreateDescriptorSetLayout failed");
    return layout;
}

VkPipelineLayout PipelineCache::createPipeLayout(
    VkDescriptorSetLayout descLayout, uint32_t pushConstSize) {
    // Mirrors pipelines.py:50-66
    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset = 0;
    pushRange.size = pushConstSize;

    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &descLayout;
    if (pushConstSize > 0) {
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges = &pushRange;
    }

    VkPipelineLayout layout = VK_NULL_HANDLE;
    vkCheck(
        vkCreatePipelineLayout(device_.device(), &layoutInfo, nullptr,
                               &layout),
        "vkCreatePipelineLayout failed");
    return layout;
}

VkPipeline PipelineCache::createPipeline(const std::vector<uint8_t>& spirv,
                                         VkPipelineLayout pipeLayout,
                                         VkShaderModule& outModule) {
    // Mirrors pipelines.py:68-97
    VkShaderModuleCreateInfo moduleInfo{};
    moduleInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    moduleInfo.codeSize = spirv.size();
    moduleInfo.pCode = reinterpret_cast<const uint32_t*>(spirv.data());

    vkCheck(vkCreateShaderModule(device_.device(), &moduleInfo, nullptr,
                                 &outModule),
            "vkCreateShaderModule failed");

    VkPipelineShaderStageCreateInfo stageInfo{};
    stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageInfo.module = outModule;
    stageInfo.pName = "main";

    VkComputePipelineCreateInfo pipeInfo{};
    pipeInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeInfo.stage = stageInfo;
    pipeInfo.layout = pipeLayout;

    VkPipeline pipeline = VK_NULL_HANDLE;
    vkCheck(vkCreateComputePipelines(device_.device(), VK_NULL_HANDLE, 1,
                                     &pipeInfo, nullptr, &pipeline),
            "vkCreateComputePipelines failed");

    return pipeline;
}

// ── Public: get or create pipeline ──────────────────────────────────────────

PipelineEntry PipelineCache::getOrCreate(const std::string& name,
                                          uint32_t numBuffers,
                                          uint32_t pushConstSize) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Check flat_map cache
    auto it = pipelines_.find(name);
    if (it != pipelines_.end())
        return it->second;

    // Need SPIR-V code
    auto spirvIt = spirvCode_.find(name);
    if (spirvIt == spirvCode_.end()) {
        throw std::runtime_error(
            "Shader '" + name +
            "' not loaded. Call loadSPIRV() or loadSPIRVFile() first.");
    }

    PipelineEntry entry{};
    entry.descLayout = createDescLayout(numBuffers);
    entry.layout = createPipeLayout(entry.descLayout, pushConstSize);
    entry.pipeline =
        createPipeline(spirvIt->second, entry.layout, entry.shaderModule);

    pipelines_.insert({name, entry});
    return entry;
}

// ── Public: descriptor set allocation with LRU cache ────────────────────────

VkDescriptorSet PipelineCache::allocDescriptorSet(
    const std::string& name,
    const std::vector<VkDescriptorBufferInfo>& buffers) {

    std::lock_guard<std::mutex> lock(mutex_);

    // Build cache key from shader name + buffer handles + sizes
    DescCacheKey key;
    key.shaderName = name;
    key.bufferBindings.reserve(buffers.size());
    for (const auto& bi : buffers)
        key.bufferBindings.emplace_back(bi.buffer, bi.range);

    // ── Cache hit? ──
    auto cacheIt = descCache_.find(key);
    if (cacheIt != descCache_.end()) {
        // Move to front of LRU
        lruList_.splice(lruList_.begin(), lruList_, cacheIt->second.lruIter);
        stats_.hits++;
        return cacheIt->second.set;
    }

    // ── Cache miss — evict LRU if full (mirrors pipelines.py:191-196) ──
    stats_.misses++;
    while (descCache_.size() >= kMaxCachedDescSets) {
        // Evict least recently used (back of list)
        auto& lruKey = lruList_.back();
        auto evictIt = descCache_.find(lruKey);
        if (evictIt != descCache_.end()) {
            vkFreeDescriptorSets(device_.device(), descriptorPool_, 1,
                                 &evictIt->second.set);
            descCache_.erase(evictIt);
        }
        lruList_.pop_back();
        stats_.evictions++;
    }

    // ── Allocate new descriptor set ──
    auto pipeIt = pipelines_.find(name);
    if (pipeIt == pipelines_.end())
        throw std::runtime_error("Pipeline '" + name + "' not created yet");

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool_;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &pipeIt->second.descLayout;

    VkDescriptorSet descSet = VK_NULL_HANDLE;
    VkResult result =
        vkAllocateDescriptorSets(device_.device(), &allocInfo, &descSet);

    if (result == VK_ERROR_OUT_OF_POOL_MEMORY ||
        result == VK_ERROR_FRAGMENTED_POOL) {
        // Emergency evict all and retry
        for (auto& [k, v] : descCache_)
            vkFreeDescriptorSets(device_.device(), descriptorPool_, 1,
                                 &v.set);
        descCache_.clear();
        lruList_.clear();
        vkCheck(
            vkAllocateDescriptorSets(device_.device(), &allocInfo, &descSet),
            "vkAllocateDescriptorSets failed after pool reset");
    } else {
        vkCheck(result, "vkAllocateDescriptorSets failed");
    }

    // ── Write buffer bindings (mirrors pipelines.py:144-161) ──
    std::vector<VkWriteDescriptorSet> writes(buffers.size());
    for (size_t i = 0; i < buffers.size(); ++i) {
        writes[i] = {};
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = descSet;
        writes[i].dstBinding = static_cast<uint32_t>(i);
        writes[i].dstArrayElement = 0;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &buffers[i];
    }
    vkUpdateDescriptorSets(device_.device(),
                           static_cast<uint32_t>(writes.size()), writes.data(),
                           0, nullptr);

    // ── Insert into LRU cache ──
    lruList_.push_front(key);
    descCache_[key] = {descSet, lruList_.begin()};

    return descSet;
}

PipelineCache::CacheStats PipelineCache::cacheStats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    CacheStats s = stats_;
    s.cachedSets = descCache_.size();
    return s;
}

}  // namespace grilly
