// VMA implementation lives in exactly one translation unit.
// Must come before any include of vk_mem_alloc.h.
#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#include "grilly/device.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace grilly {

// ── Desired device extensions (mirrors core.py:144-155) ─────────────────────
static const char* kDesiredExtensions[] = {
    "VK_KHR_cooperative_matrix",
    "VK_KHR_shader_atomic_int64",
    "VK_EXT_shader_atomic_float",
    "VK_EXT_shader_atomic_float2",
    "VK_KHR_shader_float16_int8",
    "VK_KHR_16bit_storage",
    "VK_KHR_8bit_storage",
    "VK_KHR_storage_buffer_storage_class",
    "VK_KHR_vulkan_memory_model",
    // Timeline semaphores for async command batch submission
    "VK_KHR_timeline_semaphore",
    // Memory priority: prevent WDDM from evicting VRAM allocations
    "VK_EXT_memory_priority",
    "VK_EXT_pageable_device_local_memory",
    "VK_EXT_memory_budget",
};
static constexpr size_t kNumDesiredExtensions =
    sizeof(kDesiredExtensions) / sizeof(kDesiredExtensions[0]);

// ── Helper: check VkResult ──────────────────────────────────────────────────
static void vkCheck(VkResult result, const char* msg) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(std::string(msg) +
                                 " (VkResult=" + std::to_string(result) + ")");
    }
}

// ── Construction / destruction ──────────────────────────────────────────────

GrillyDevice::GrillyDevice() { initVulkan(); }

GrillyDevice::~GrillyDevice() {
    if (device_ != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device_);
        vkDestroyDevice(device_, nullptr);
    }
    if (instance_ != VK_NULL_HANDLE) {
        vkDestroyInstance(instance_, nullptr);
    }
}

GrillyDevice::GrillyDevice(GrillyDevice&& other) noexcept
    : instance_(other.instance_),
      physicalDevice_(other.physicalDevice_),
      device_(other.device_),
      queue_(other.queue_),
      queueFamily_(other.queueFamily_),
      enabledExtensions_(std::move(other.enabledExtensions_)),
      deviceName_(std::move(other.deviceName_)) {
    other.instance_ = VK_NULL_HANDLE;
    other.physicalDevice_ = VK_NULL_HANDLE;
    other.device_ = VK_NULL_HANDLE;
    other.queue_ = VK_NULL_HANDLE;
}

GrillyDevice& GrillyDevice::operator=(GrillyDevice&& other) noexcept {
    if (this != &other) {
        // Destroy current resources
        if (device_ != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(device_);
            vkDestroyDevice(device_, nullptr);
        }
        if (instance_ != VK_NULL_HANDLE)
            vkDestroyInstance(instance_, nullptr);

        instance_ = other.instance_;
        physicalDevice_ = other.physicalDevice_;
        device_ = other.device_;
        queue_ = other.queue_;
        queueFamily_ = other.queueFamily_;
        enabledExtensions_ = std::move(other.enabledExtensions_);
        deviceName_ = std::move(other.deviceName_);

        other.instance_ = VK_NULL_HANDLE;
        other.physicalDevice_ = VK_NULL_HANDLE;
        other.device_ = VK_NULL_HANDLE;
        other.queue_ = VK_NULL_HANDLE;
    }
    return *this;
}

// ── Vulkan initialization (port of core.py:157-395) ────────────────────────

void GrillyDevice::initVulkan() {
    // ── Create instance (Vulkan 1.3 for subgroup size control, memory priority) ──
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "GrillCheese";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "grilly-cpp";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_MAKE_API_VERSION(0, 1, 3, 0);

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    vkCheck(vkCreateInstance(&createInfo, nullptr, &instance_),
            "vkCreateInstance failed");

    // ── Enumerate physical devices ──
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance_, &deviceCount, nullptr);
    if (deviceCount == 0)
        throw std::runtime_error("No Vulkan physical devices found");

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance_, &deviceCount, devices.data());

    // ── Select GPU (mirrors core.py:397-477) ──
    physicalDevice_ = selectGPU(devices);

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physicalDevice_, &props);
    deviceName_ = props.deviceName;
    std::cout << "[OK] Using GPU: " << deviceName_ << std::endl;

    // ── Enumerate available device extensions ──
    uint32_t extCount = 0;
    vkEnumerateDeviceExtensionProperties(physicalDevice_, nullptr, &extCount,
                                         nullptr);
    std::vector<VkExtensionProperties> availableExts(extCount);
    vkEnumerateDeviceExtensionProperties(physicalDevice_, nullptr, &extCount,
                                         availableExts.data());

    std::set<std::string> availableSet;
    for (const auto& ext : availableExts)
        availableSet.insert(ext.extensionName);

    // Enable desired extensions that are available
    std::vector<const char*> enabledExtPtrs;
    for (size_t i = 0; i < kNumDesiredExtensions; ++i) {
        if (availableSet.count(kDesiredExtensions[i])) {
            enabledExtPtrs.push_back(kDesiredExtensions[i]);
            enabledExtensions_.insert(kDesiredExtensions[i]);
        }
    }

    if (!enabledExtensions_.empty()) {
        std::cout << "[OK] Vulkan extensions:";
        for (const auto& e : enabledExtensions_)
            std::cout << " " << e;
        std::cout << std::endl;
    }

    // ── Find compute queue family ──
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice_,
                                              &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice_,
                                              &queueFamilyCount,
                                              queueFamilies.data());

    queueFamily_ = UINT32_MAX;
    for (uint32_t i = 0; i < queueFamilyCount; ++i) {
        if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            queueFamily_ = i;
            break;
        }
    }
    if (queueFamily_ == UINT32_MAX)
        throw std::runtime_error("No compute queue family found");

    // ── Create logical device ──
    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueCreateInfo{};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = queueFamily_;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    // ── Build pNext feature chain (mirrors core.py:240-326) ──
    // We query each feature struct from the physical device and only request
    // what the hardware actually supports.  Built bottom-up so each struct's
    // pNext points to the previous one.

    VkPhysicalDeviceFeatures deviceFeatures{};

    // Enable shaderInt64 — required for uint64_t types in GLSL shaders.
    // Needed by hamming-top1.glsl (GL_ARB_gpu_shader_int64).
    {
        VkPhysicalDeviceFeatures supported{};
        vkGetPhysicalDeviceFeatures(physicalDevice_, &supported);
        if (supported.shaderInt64) {
            deviceFeatures.shaderInt64 = VK_TRUE;
            std::cout << "[OK] shaderInt64 enabled" << std::endl;
        }
    }

    // Base pNext chain pointer — we'll prepend to this as we add features
    void* pNextChain = nullptr;

    // Shader atomic int64 (VK_KHR_shader_atomic_int64) — atomicMin on uint64
    // Required by hamming-top1.glsl for race-free GPU argmin.
    VkPhysicalDeviceShaderAtomicInt64Features atomicInt64Features{};
    atomicInt64Features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_INT64_FEATURES;
    if (enabledExtensions_.count("VK_KHR_shader_atomic_int64")) {
        VkPhysicalDeviceShaderAtomicInt64Features query{};
        query.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_INT64_FEATURES;
        VkPhysicalDeviceFeatures2 features2Query{};
        features2Query.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        features2Query.pNext = &query;
        vkGetPhysicalDeviceFeatures2(physicalDevice_, &features2Query);

        atomicInt64Features.shaderBufferInt64Atomics =
            query.shaderBufferInt64Atomics;
        atomicInt64Features.shaderSharedInt64Atomics =
            query.shaderSharedInt64Atomics;
        atomicInt64Features.pNext = pNextChain;
        pNextChain = &atomicInt64Features;
        if (query.shaderBufferInt64Atomics)
            std::cout << "[OK] shaderBufferInt64Atomics enabled" << std::endl;
    }

    // Timeline semaphore features (Vulkan 1.2)
    VkPhysicalDeviceTimelineSemaphoreFeatures timelineSemFeatures{};
    timelineSemFeatures.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
    if (enabledExtensions_.count("VK_KHR_timeline_semaphore")) {
        VkPhysicalDeviceTimelineSemaphoreFeatures query{};
        query.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
        VkPhysicalDeviceFeatures2 features2Query{};
        features2Query.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        features2Query.pNext = &query;
        vkGetPhysicalDeviceFeatures2(physicalDevice_, &features2Query);

        timelineSemFeatures.timelineSemaphore = query.timelineSemaphore;
        timelineSemFeatures.pNext = pNextChain;
        pNextChain = &timelineSemFeatures;
    }

    // 16-bit storage features
    VkPhysicalDevice16BitStorageFeatures storage16Features{};
    storage16Features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
    if (enabledExtensions_.count("VK_KHR_16bit_storage")) {
        VkPhysicalDevice16BitStorageFeatures query{};
        query.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
        VkPhysicalDeviceFeatures2 features2Query{};
        features2Query.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        features2Query.pNext = &query;
        vkGetPhysicalDeviceFeatures2(physicalDevice_, &features2Query);

        storage16Features.storageBuffer16BitAccess =
            query.storageBuffer16BitAccess;
        storage16Features.pNext = pNextChain;
        pNextChain = &storage16Features;
    }

    // Float16/Int8 shader features
    VkPhysicalDeviceShaderFloat16Int8Features fp16Int8Features{};
    fp16Int8Features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
    if (enabledExtensions_.count("VK_KHR_shader_float16_int8")) {
        VkPhysicalDeviceShaderFloat16Int8Features query{};
        query.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
        VkPhysicalDeviceFeatures2 features2Query{};
        features2Query.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        features2Query.pNext = &query;
        vkGetPhysicalDeviceFeatures2(physicalDevice_, &features2Query);

        fp16Int8Features.shaderFloat16 = query.shaderFloat16;
        fp16Int8Features.shaderInt8 = query.shaderInt8;
        fp16Int8Features.pNext = pNextChain;
        pNextChain = &fp16Int8Features;
    }

    // Memory priority (VK_EXT_memory_priority) — prevent WDDM VRAM eviction
    VkPhysicalDeviceMemoryPriorityFeaturesEXT memPriorityFeatures{};
    memPriorityFeatures.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PRIORITY_FEATURES_EXT;
    if (enabledExtensions_.count("VK_EXT_memory_priority")) {
        VkPhysicalDeviceMemoryPriorityFeaturesEXT query{};
        query.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PRIORITY_FEATURES_EXT;
        VkPhysicalDeviceFeatures2 features2Query{};
        features2Query.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        features2Query.pNext = &query;
        vkGetPhysicalDeviceFeatures2(physicalDevice_, &features2Query);

        memPriorityFeatures.memoryPriority = query.memoryPriority;
        memPriorityFeatures.pNext = pNextChain;
        pNextChain = &memPriorityFeatures;
        if (query.memoryPriority)
            std::cout << "[OK] VK_EXT_memory_priority enabled" << std::endl;
    }

    // Pageable device-local memory (VK_EXT_pageable_device_local_memory)
    VkPhysicalDevicePageableDeviceLocalMemoryFeaturesEXT pageableFeatures{};
    pageableFeatures.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PAGEABLE_DEVICE_LOCAL_MEMORY_FEATURES_EXT;
    if (enabledExtensions_.count("VK_EXT_pageable_device_local_memory")) {
        VkPhysicalDevicePageableDeviceLocalMemoryFeaturesEXT query{};
        query.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PAGEABLE_DEVICE_LOCAL_MEMORY_FEATURES_EXT;
        VkPhysicalDeviceFeatures2 features2Query{};
        features2Query.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        features2Query.pNext = &query;
        vkGetPhysicalDeviceFeatures2(physicalDevice_, &features2Query);

        pageableFeatures.pageableDeviceLocalMemory =
            query.pageableDeviceLocalMemory;
        pageableFeatures.pNext = pNextChain;
        pNextChain = &pageableFeatures;
        if (query.pageableDeviceLocalMemory)
            std::cout << "[OK] VK_EXT_pageable_device_local_memory enabled"
                      << std::endl;
    }

    // Wrap features in VkPhysicalDeviceFeatures2
    VkPhysicalDeviceFeatures2 features2{};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.features = deviceFeatures;
    features2.pNext = pNextChain;

    VkDeviceCreateInfo deviceCreateInfo{};
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
    deviceCreateInfo.enabledExtensionCount =
        static_cast<uint32_t>(enabledExtPtrs.size());
    deviceCreateInfo.ppEnabledExtensionNames = enabledExtPtrs.data();
    deviceCreateInfo.pEnabledFeatures = nullptr;  // features via pNext
    deviceCreateInfo.pNext = &features2;

    vkCheck(vkCreateDevice(physicalDevice_, &deviceCreateInfo, nullptr,
                           &device_),
            "vkCreateDevice failed");

    vkGetDeviceQueue(device_, queueFamily_, 0, &queue_);

    std::cout << "[OK] Vulkan device initialized (C++ backend)" << std::endl;
}

// ── GPU selection (port of core.py:397-477) ─────────────────────────────────

VkPhysicalDevice GrillyDevice::selectGPU(
    const std::vector<VkPhysicalDevice>& devices) {
    struct GPUInfo {
        VkPhysicalDevice device;
        VkPhysicalDeviceProperties props;
        std::string name;
    };

    std::vector<GPUInfo> entries;
    for (auto dev : devices) {
        GPUInfo info;
        info.device = dev;
        vkGetPhysicalDeviceProperties(dev, &info.props);
        info.name = info.props.deviceName;
        entries.push_back(info);
    }

    // Log device inventory
    std::cout << "[OK] Vulkan devices enumerated:";
    for (size_t i = 0; i < entries.size(); ++i) {
        std::cout << " " << i << ":" << entries[i].name << " (type "
                  << entries[i].props.deviceType << ", vendor 0x" << std::hex
                  << entries[i].props.vendorID << std::dec << ")";
    }
    std::cout << std::endl;

    // Check VK_GPU_INDEX env var
    const char* envIdx = std::getenv("VK_GPU_INDEX");
    if (envIdx) {
        int idx = std::atoi(envIdx);
        if (idx >= 0 && static_cast<size_t>(idx) < entries.size())
            return entries[idx].device;
    }

    // Filter out CPU / llvmpipe
    auto isLlvmpipe = [](const std::string& name) {
        std::string lower = name;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        return lower.find("llvmpipe") != std::string::npos;
    };

    std::vector<GPUInfo> candidates;
    for (const auto& e : entries) {
        if (e.props.deviceType != VK_PHYSICAL_DEVICE_TYPE_CPU &&
            !isLlvmpipe(e.name)) {
            candidates.push_back(e);
        }
    }
    if (candidates.empty())
        candidates = entries;

    // Prefer discrete GPUs
    std::vector<GPUInfo> discrete;
    for (const auto& e : candidates) {
        if (e.props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
            discrete.push_back(e);
    }
    if (!discrete.empty())
        candidates = discrete;

    // Vendor preference: NVIDIA (0x10DE) then AMD (0x1002)
    for (uint32_t vendor : {0x10DEu, 0x1002u}) {
        for (const auto& e : candidates) {
            if (e.props.vendorID == vendor)
                return e.device;
        }
    }

    // If all are CPU, check ALLOW_CPU_VULKAN
    bool allCPU = std::all_of(candidates.begin(), candidates.end(),
                              [&](const GPUInfo& e) {
                                  return e.props.deviceType ==
                                             VK_PHYSICAL_DEVICE_TYPE_CPU ||
                                         isLlvmpipe(e.name);
                              });
    if (allCPU) {
        const char* allowCpu = std::getenv("ALLOW_CPU_VULKAN");
        if (!allowCpu || (std::string(allowCpu) != "1" &&
                          std::string(allowCpu) != "true")) {
            throw std::runtime_error(
                "No discrete Vulkan GPU found. "
                "Set ALLOW_CPU_VULKAN=1 to allow CPU/llvmpipe fallback.");
        }
    }

    return candidates[0].device;
}

// ── Extension queries ───────────────────────────────────────────────────────

bool GrillyDevice::hasExtension(const std::string& name) const {
    return enabledExtensions_.count(name) > 0;
}

bool GrillyDevice::hasCooperativeMatrix() const {
    return hasExtension("VK_KHR_cooperative_matrix");
}

bool GrillyDevice::hasFloat16() const {
    return hasExtension("VK_KHR_shader_float16_int8") &&
           hasExtension("VK_KHR_16bit_storage");
}

}  // namespace grilly
