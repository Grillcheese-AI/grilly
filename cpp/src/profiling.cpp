#include "grilly/profiling.h"

#include <stdexcept>
#include <cstring>

namespace grilly {

static void vkCheck(VkResult result, const char* msg) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(std::string(msg) +
                                 " (VkResult=" + std::to_string(result) + ")");
    }
}

ProfilingContext::ProfilingContext(GrillyDevice& device, CommandBatch* batch)
    : device_(device), batch_(batch) {
    // Query timestamp period from physical device properties
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(device_.physicalDevice(), &props);
    timestampPeriod_ = props.limits.timestampPeriod;  // Already in nanoseconds
}

ProfilingContext::~ProfilingContext() {
    destroyQueryPool();
}

void ProfilingContext::setEnabled(bool enable) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (enable && !enabled_) {
        ensureQueryPool();
    }
    enabled_ = enable;
}

void ProfilingContext::ensureQueryPool() {
    if (queryPool_ != VK_NULL_HANDLE) return;
    
    VkQueryPoolCreateInfo queryInfo{};
    queryInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    queryInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
    queryInfo.queryCount = maxQueries_;
    
    vkCheck(vkCreateQueryPool(device_.device(), &queryInfo, nullptr, &queryPool_),
            "vkCreateQueryPool failed");
}

void ProfilingContext::destroyQueryPool() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queryPool_ != VK_NULL_HANDLE) {
        vkDestroyQueryPool(device_.device(), queryPool_, nullptr);
        queryPool_ = VK_NULL_HANDLE;
    }
    queryCount_ = 0;
    currentQueryIndex_ = 0;
    pendingResults_.clear();
}

void ProfilingContext::beginDispatch(const std::string& shaderName,
                                      uint32_t gx, uint32_t gy, uint32_t gz) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!enabled_ || !batch_ || !batch_->isRecording()) return;
    
    if (currentQueryIndex_ >= maxQueries_) {
        throw std::runtime_error(
            "ProfilingContext: exceeded maximum query count (" +
            std::to_string(maxQueries_) + "). "
            "Increase maxQueries_ or reduce dispatches per batch.");
    }
    
    // Record start timestamp
    uint32_t queryIndex = currentQueryIndex_ * 2;  // 2 queries per dispatch (start/end)
    vkCmdWriteTimestamp(batch_->cmdBuffer(),
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        queryPool_, queryIndex);
    
    // Store metadata for later retrieval
    ProfileResult result;
    result.shaderName = shaderName;
    result.workgroupX = gx;
    result.workgroupY = gy;
    result.workgroupZ = gz;
    pendingResults_.push_back(result);
}

void ProfilingContext::endDispatch() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!enabled_ || !batch_ || !batch_->isRecording()) return;
    
    if (currentQueryIndex_ >= maxQueries_) return;
    
    // Record end timestamp
    uint32_t queryIndex = currentQueryIndex_ * 2 + 1;
    vkCmdWriteTimestamp(batch_->cmdBuffer(),
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        queryPool_, queryIndex);
    
    currentQueryIndex_++;
    queryCount_++;
}

std::vector<ProfileResult> ProfilingContext::getResults() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!enabled_ || queryPool_ == VK_NULL_HANDLE || queryCount_ == 0) {
        return {};
    }
    
    // Retrieve all query results
    std::vector<uint64_t> timestamps(queryCount_ * 2);
    VkResult result = vkGetQueryPoolResults(
        device_.device(),
        queryPool_,
        0,                              // firstQuery
        static_cast<uint32_t>(timestamps.size()),
        timestamps.size() * sizeof(uint64_t),
        timestamps.data(),
        sizeof(uint64_t),               // stride
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT
    );
    
    if (result != VK_SUCCESS) {
        throw std::runtime_error("vkGetQueryPoolResults failed: " +
                                 std::to_string(result));
    }
    
    // Compute durations and populate results
    std::vector<ProfileResult> results;
    results.reserve(queryCount_);
    
    for (uint32_t i = 0; i < queryCount_ && i < pendingResults_.size(); ++i) {
        ProfileResult& r = pendingResults_[i];
        r.gpuStartNs = static_cast<uint64_t>(timestamps[i * 2] * timestampPeriod_);
        r.gpuEndNs = static_cast<uint64_t>(timestamps[i * 2 + 1] * timestampPeriod_);
        r.gpuDurationNs = r.gpuEndNs - r.gpuStartNs;
        results.push_back(r);
    }
    
    return results;
}

void ProfilingContext::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    currentQueryIndex_ = 0;
    queryCount_ = 0;
    pendingResults_.clear();
}

}  // namespace grilly
