#pragma once
/// VulkanBackend — ComputeBackend implementation using Vulkan + VMA.
///
/// Composes the existing GrillyDevice, BufferPool, PipelineCache, and
/// CommandBatch into a single ComputeBackend interface. Maps opaque uint64_t
/// buffer handles to GrillyBuffer structs via an internal table.

#include "grilly/compute_backend.h"
#include "grilly/op_graph.h"
#include "grilly/vulkan/vk_buffer_pool.h"
#include "grilly/vulkan/vk_command_batch.h"
#include "grilly/vulkan/vk_device.h"
#include "grilly/vulkan/vk_pipeline_cache.h"

#include <mutex>
#include <unordered_map>

namespace grilly {

class VulkanBackend : public ComputeBackend {
public:
    VulkanBackend();
    ~VulkanBackend() override;

    // ── ComputeBackend interface ──

    std::string name() const override { return "vulkan"; }
    std::string deviceName() const override;

    uint64_t createBuffer(const BufferDesc& desc) override;
    void destroyBuffer(uint64_t buf) override;
    void upload(uint64_t buf, const void* data, size_t bytes) override;
    void download(uint64_t buf, void* out, size_t bytes) override;

    void loadShader(const std::string& name,
                    const std::string& path) override;
    void loadShaderDir(const std::string& dir) override;
    bool hasShader(const std::string& name) const override;

    void beginBatch() override;
    void dispatch(const std::string& shader,
                  const std::vector<uint64_t>& buffers,
                  uint32_t gx, uint32_t gy, uint32_t gz,
                  const void* push = nullptr,
                  uint32_t pushBytes = 0) override;
    void barrier() override;
    void endBatch() override;

    bool hasCooperativeMatrix() const override;
    bool hasFloat16() const override;

    // ── Direct access to Vulkan internals (for ops that need it) ──

    GrillyDevice& device() { return device_; }
    BufferPool& pool() { return pool_; }
    PipelineCache& cache() { return cache_; }
    CommandBatch& batch() { return batch_; }

    /// Resolve an opaque handle to its GrillyBuffer.
    GrillyBuffer& resolveBuffer(uint64_t handle);
    const GrillyBuffer& resolveBuffer(uint64_t handle) const;

    // ── Graph recording mode ──
    // When enabled, dispatch() records ops into an OpGraph instead of
    // directly into CommandBatch. endBatch() runs optimize() + execute()
    // for automatic fusion and barrier elimination.
    void setGraphMode(bool enable) { graphMode_ = enable; }
    bool graphMode() const { return graphMode_; }
    OpGraph& opGraph() { return graph_; }

private:
    GrillyDevice device_;
    BufferPool pool_;
    PipelineCache cache_;
    CommandBatch batch_;

    // Graph recording mode
    bool graphMode_ = false;
    OpGraph graph_;

    // Handle -> GrillyBuffer mapping
    mutable std::mutex handleMutex_;
    uint64_t nextHandle_ = 1;
    std::unordered_map<uint64_t, GrillyBuffer> handles_;
};

}  // namespace grilly
