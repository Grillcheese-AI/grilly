#include "grilly/vulkan/vk_backend.h"

#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace grilly {

// ── Construction / destruction ──────────────────────────────────────────────

VulkanBackend::VulkanBackend()
    : device_(), pool_(device_), cache_(device_), batch_(device_) {}

VulkanBackend::~VulkanBackend() {
    // Release all tracked buffers
    auto alloc = pool_.allocator();
    for (auto& [handle, bufPtr] : handles_) {
        if (bufPtr && bufPtr->handle != VK_NULL_HANDLE) {
            // Exception safety: don't throw during destruction
            try {
                vmaDestroyBuffer(alloc, bufPtr->handle, bufPtr->allocation);
            } catch (...) {
                // Log but don't propagate - throwing from destructor causes std::terminate
                std::cerr << "[WARN] vmaDestroyBuffer failed for handle " 
                          << handle << " during cleanup" << std::endl;
            }
        }
    }
    handles_.clear();
}

// ── Identity ────────────────────────────────────────────────────────────────

std::string VulkanBackend::deviceName() const {
    return device_.deviceName();
}

// ── Buffer management ───────────────────────────────────────────────────────

uint64_t VulkanBackend::createBuffer(const BufferDesc& desc) {
    auto buf = std::make_shared<GrillyBuffer>();
    switch (desc.usage) {
        case BufferDesc::DeviceLocal:
            *buf = pool_.acquireDeviceLocal(desc.size);
            break;
        case BufferDesc::Readback:
            *buf = pool_.acquireReadback(desc.size);
            break;
        case BufferDesc::PreferDevice:
            *buf = pool_.acquirePreferDeviceLocal(desc.size);
            break;
        case BufferDesc::HostVisible:
        default:
            *buf = pool_.acquire(desc.size);
            break;
    }

    std::lock_guard<std::mutex> lock(handleMutex_);
    uint64_t h = nextHandle_++;
    handles_[h] = buf;
    return h;
}

void VulkanBackend::destroyBuffer(uint64_t handle) {
    std::lock_guard<std::mutex> lock(handleMutex_);
    auto it = handles_.find(handle);
    if (it == handles_.end()) return;

    auto alloc = pool_.allocator();
    auto& bufPtr = it->second;
    if (bufPtr && bufPtr->handle != VK_NULL_HANDLE) {
        try {
            vmaDestroyBuffer(alloc, bufPtr->handle, bufPtr->allocation);
        } catch (...) {
            std::cerr << "[WARN] vmaDestroyBuffer failed for handle " 
                      << handle << std::endl;
        }
    }
    handles_.erase(it);
}

void VulkanBackend::upload(uint64_t handle, const void* data, size_t bytes) {
    auto buf = resolveBuffer(handle);
    if (buf->mappedPtr) {
        // Host-visible: direct memcpy
        pool_.upload(*buf, static_cast<const float*>(data), bytes);
    } else {
        // Device-local: staged upload
        pool_.uploadStaged(*buf, data, bytes);
    }
}

void VulkanBackend::download(uint64_t handle, void* out, size_t bytes) {
    auto buf = resolveBuffer(handle);
    if (buf->mappedPtr) {
        pool_.download(*buf, static_cast<float*>(out), bytes);
    } else {
        pool_.downloadStaged(*buf, out, bytes);
    }
}

std::shared_ptr<GrillyBuffer> VulkanBackend::resolveBuffer(uint64_t handle) {
    std::lock_guard<std::mutex> lock(handleMutex_);
    auto it = handles_.find(handle);
    if (it == handles_.end())
        throw std::runtime_error("Invalid buffer handle: " +
                                 std::to_string(handle));
    return it->second;
}

std::shared_ptr<const GrillyBuffer> VulkanBackend::resolveBuffer(uint64_t handle) const {
    std::lock_guard<std::mutex> lock(handleMutex_);
    auto it = handles_.find(handle);
    if (it == handles_.end())
        throw std::runtime_error("Invalid buffer handle: " +
                                 std::to_string(handle));
    return it->second;
}

// ── Shader management ───────────────────────────────────────────────────────

void VulkanBackend::loadShader(const std::string& name,
                               const std::string& path) {
    cache_.loadSPIRVFile(name, path);
}

void VulkanBackend::loadShaderDir(const std::string& dir) {
    namespace fs = std::filesystem;
    fs::path dirPath(dir);
    if (!fs::exists(dirPath))
        throw std::runtime_error("Shader directory not found: " + dir);

    int count = 0;
    for (const auto& entry : fs::directory_iterator(dirPath)) {
        if (entry.path().extension() == ".spv") {
            std::string shaderName = entry.path().stem().string();
            cache_.loadSPIRVFile(shaderName, entry.path().string());
            count++;
        }
    }
    std::cout << "[OK] Loaded " << count << " SPIR-V shaders from " << dir
              << std::endl;
}

bool VulkanBackend::hasShader(const std::string& name) const {
    return cache_.hasShader(name);
}

// ── Compute dispatch ────────────────────────────────────────────────────────

void VulkanBackend::beginBatch() {
    if (graphMode_) {
        graph_.clear();
        return;
    }
    batch_.begin();
}

void VulkanBackend::dispatch(const std::string& shader,
                             const std::vector<uint64_t>& buffers,
                             uint32_t gx, uint32_t gy, uint32_t gz,
                             const void* push, uint32_t pushBytes) {
    // Build descriptor buffer infos from opaque handles
    std::vector<VkDescriptorBufferInfo> bufInfos;
    bufInfos.reserve(buffers.size());
    for (uint64_t h : buffers) {
        auto buf = resolveBuffer(h);
        bufInfos.push_back({buf->handle, 0, buf->size});
    }

    if (graphMode_) {
        // Record into OpGraph for deferred optimization + execution
        graph_.addOp(shader, static_cast<uint32_t>(buffers.size()),
                     bufInfos, push, pushBytes, gx, gy, gz);
        return;
    }

    // Immediate mode: dispatch directly into CommandBatch
    auto entry = cache_.getOrCreate(
        shader, static_cast<uint32_t>(buffers.size()), pushBytes);
    VkDescriptorSet descSet = cache_.allocDescriptorSet(shader, bufInfos);
    batch_.dispatch(entry.pipeline, entry.layout, descSet,
                    gx, gy, gz, push, pushBytes);
}

void VulkanBackend::barrier() {
    if (graphMode_) return;  // optimize() handles barriers
    batch_.barrier();
}

void VulkanBackend::endBatch() {
    if (graphMode_) {
        if (graph_.size() == 0) return;
        graph_.optimize(cache_);
        graph_.execute(batch_, cache_);
        return;
    }
    batch_.submit();
}

// ── Capability queries ──────────────────────────────────────────────────────

bool VulkanBackend::hasCooperativeMatrix() const {
    return device_.hasCooperativeMatrix();
}

bool VulkanBackend::hasFloat16() const {
    return device_.hasFloat16();
}

// ── Factory ─────────────────────────────────────────────────────────────────

std::unique_ptr<ComputeBackend> createBackend(const std::string& type) {
    if (type == "vulkan") {
        return std::make_unique<VulkanBackend>();
    }
    throw std::runtime_error(
        "Unknown backend type: '" + type + "'. "
        "Supported: vulkan. Coming soon: opengl, opencl.");
}

}  // namespace grilly
