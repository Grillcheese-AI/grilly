#pragma once
/// Vulkan device implementation.
/// Physical location: vulkan/ — but namespace stays grilly:: for compatibility.
/// When OpenGL/OpenCL backends are added, they'll have their own subdirectories
/// and the ComputeBackend interface provides the abstraction boundary.

#include <vulkan/vulkan.h>

#include <cstdint>
#include <set>
#include <string>
#include <vector>

namespace grilly {

/// Core Vulkan device wrapper.
/// Ports backend/core.py:157-477 — instance creation, GPU selection, device
/// + queue init, and extension chain building. All done in native C++ so the
/// struct-creation overhead that plagues the ctypes path is eliminated.
class GrillyDevice {
public:
    GrillyDevice();
    ~GrillyDevice();

    // Non-copyable, movable
    GrillyDevice(const GrillyDevice&) = delete;
    GrillyDevice& operator=(const GrillyDevice&) = delete;
    GrillyDevice(GrillyDevice&& other) noexcept;
    GrillyDevice& operator=(GrillyDevice&& other) noexcept;

    VkDevice device() const { return device_; }
    VkPhysicalDevice physicalDevice() const { return physicalDevice_; }
    VkQueue computeQueue() const { return queue_; }
    uint32_t queueFamily() const { return queueFamily_; }
    VkInstance instance() const { return instance_; }

    bool hasExtension(const std::string& name) const;
    bool hasCooperativeMatrix() const;
    bool hasFloat16() const;

    /// GPU name reported by the driver.
    const std::string& deviceName() const { return deviceName_; }

private:
    void initVulkan();
    VkPhysicalDevice selectGPU(const std::vector<VkPhysicalDevice>& devices);

    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    uint32_t queueFamily_ = 0;

    std::set<std::string> enabledExtensions_;
    std::string deviceName_;
};

}  // namespace grilly
