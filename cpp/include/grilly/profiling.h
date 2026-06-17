#pragma once
/// GPU profiling hooks for Vulkan command buffers with timeline semaphore support.
///
/// Provides optional timing instrumentation for compute dispatches using Vulkan
/// query pools. When enabled, records GPU timestamps for each dispatch and allows
/// retrieval of profile data after execution.
///
/// Usage:
///   CommandBatch batch(device);
///   ProfilingContext profiler(device, &batch);
///   profiler.setEnabled(true);
///   
///   // ... record dispatches ...
///   batch.dispatch(...);
///   
///   batch.submit();
///   batch.waitForCompletion();
///   
///   auto results = profiler.getResults();
///   for (const auto& r : results) {
///       std::cout << r.shaderName << ": " << r.gpuDurationNs << " ns\n";
///   }

#include <vulkan/vulkan.h>
#include <string>
#include <vector>
#include <mutex>
#include <cstdint>

#include "grilly/vulkan/vk_device.h"
#include "grilly/vulkan/vk_command_batch.h"

namespace grilly {

/// Single dispatch profiling result
struct ProfileResult {
    std::string shaderName;
    uint64_t gpuStartNs = 0;      ///< Absolute GPU timestamp at start
    uint64_t gpuEndNs = 0;        ///< Absolute GPU timestamp at end
    uint64_t gpuDurationNs = 0;   ///< Duration in nanoseconds
    uint32_t workgroupX = 0;
    uint32_t workgroupY = 0;
    uint32_t workgroupZ = 0;
};

/// Profiling context attached to a CommandBatch
class ProfilingContext {
public:
    explicit ProfilingContext(GrillyDevice& device, CommandBatch* batch = nullptr);
    ~ProfilingContext();
    
    ProfilingContext(const ProfilingContext&) = delete;
    ProfilingContext& operator=(const ProfilingContext&) = delete;
    
    /// Enable or disable profiling. When disabled, no queries are recorded.
    void setEnabled(bool enable);
    bool isEnabled() const { return enabled_; }
    
    /// Attach to a command batch (can be changed between batches)
    void setCommandBatch(CommandBatch* batch) { batch_ = batch; }
    CommandBatch* getCommandBatch() const { return batch_; }
    
    /// Record start timestamp for a dispatch. Call before batch->dispatch().
    void beginDispatch(const std::string& shaderName,
                       uint32_t gx, uint32_t gy, uint32_t gz);
    
    /// Record end timestamp for a dispatch. Call after batch->dispatch().
    void endDispatch();
    
    /// Must be called after batch submission completes. Retrieves all results.
    std::vector<ProfileResult> getResults();
    
    /// Reset all query state (call after retrieving results)
    void reset();
    
    /// Get timestamp period in nanoseconds (converts query results to time)
    double getTimestampPeriod() const { return timestampPeriod_; }
    
private:
    GrillyDevice& device_;
    CommandBatch* batch_ = nullptr;
    bool enabled_ = false;
    
    VkQueryPool queryPool_ = VK_NULL_HANDLE;
    uint32_t queryCount_ = 0;
    uint32_t maxQueries_ = 1024;  ///< Maximum dispatches per batch
    
    std::vector<ProfileResult> pendingResults_;
    uint32_t currentQueryIndex_ = 0;
    
    double timestampPeriod_ = 1.0;  ///< Device timestamp period in ns
    mutable std::mutex mutex_;
    
    void ensureQueryPool();
    void destroyQueryPool();
};

}  // namespace grilly
