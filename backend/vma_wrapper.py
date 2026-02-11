"""
VMA (Vulkan Memory Allocator) Python Wrapper

Uses ctypes to interface with VMA compiled as a shared library.
This avoids FFI compatibility issues between different Python Vulkan bindings.

VMA provides:
- AMD/NVIDIA/Intel optimized memory allocation
- Sub-allocation from large memory blocks
- Automatic memory type selection
- Persistent mapping support

Build VMA:
    See build_vma.py or run: python -m grilly.backend.vma_wrapper --build
"""

import ctypes
import logging
import os
import platform
import sys
from ctypes import (
    POINTER,
    Structure,
    byref,
    c_float,
    c_int,
    c_uint32,
    c_uint64,
    c_void_p,
)

import numpy as np

logger = logging.getLogger(__name__)

# VMA constants
VMA_MEMORY_USAGE_UNKNOWN = 0
VMA_MEMORY_USAGE_GPU_ONLY = 1
VMA_MEMORY_USAGE_CPU_ONLY = 2
VMA_MEMORY_USAGE_CPU_TO_GPU = 3
VMA_MEMORY_USAGE_GPU_TO_CPU = 4
VMA_MEMORY_USAGE_CPU_COPY = 5
VMA_MEMORY_USAGE_GPU_LAZILY_ALLOCATED = 6
VMA_MEMORY_USAGE_AUTO = 7
VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE = 8
VMA_MEMORY_USAGE_AUTO_PREFER_HOST = 9

VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT = 0x00000001
VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT = 0x00000002
VMA_ALLOCATION_CREATE_MAPPED_BIT = 0x00000004
VMA_ALLOCATION_CREATE_USER_DATA_COPY_STRING_BIT = 0x00000020
VMA_ALLOCATION_CREATE_UPPER_ADDRESS_BIT = 0x00000040
VMA_ALLOCATION_CREATE_DONT_BIND_BIT = 0x00000080
VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT = 0x00000100
VMA_ALLOCATION_CREATE_CAN_ALIAS_BIT = 0x00000200
VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT = 0x00000400
VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT = 0x00000800
VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT = 0x00001000
VMA_ALLOCATION_CREATE_STRATEGY_MIN_MEMORY_BIT = 0x00010000
VMA_ALLOCATION_CREATE_STRATEGY_MIN_TIME_BIT = 0x00020000
VMA_ALLOCATION_CREATE_STRATEGY_MIN_OFFSET_BIT = 0x00040000

VK_SUCCESS = 0


class VmaVulkanFunctions(Structure):
    """Pointers to Vulkan functions used by VMA.

    Must match the VmaVulkanFunctions struct in vk_mem_alloc.h.
    The struct layout depends on VMA_VULKAN_VERSION and feature macros.
    For Vulkan 1.1+ (VMA default), includes all KHR extension fields.
    """

    _fields_ = [
        # Core functions
        ("vkGetInstanceProcAddr", c_void_p),
        ("vkGetDeviceProcAddr", c_void_p),
        ("vkGetPhysicalDeviceProperties", c_void_p),
        ("vkGetPhysicalDeviceMemoryProperties", c_void_p),
        ("vkAllocateMemory", c_void_p),
        ("vkFreeMemory", c_void_p),
        ("vkMapMemory", c_void_p),
        ("vkUnmapMemory", c_void_p),
        ("vkFlushMappedMemoryRanges", c_void_p),
        ("vkInvalidateMappedMemoryRanges", c_void_p),
        ("vkBindBufferMemory", c_void_p),
        ("vkBindImageMemory", c_void_p),
        ("vkGetBufferMemoryRequirements", c_void_p),
        ("vkGetImageMemoryRequirements", c_void_p),
        ("vkCreateBuffer", c_void_p),
        ("vkDestroyBuffer", c_void_p),
        ("vkCreateImage", c_void_p),
        ("vkDestroyImage", c_void_p),
        ("vkCmdCopyBuffer", c_void_p),
        # VMA_DEDICATED_ALLOCATION || VMA_VULKAN_VERSION >= 1001000
        ("vkGetBufferMemoryRequirements2KHR", c_void_p),
        ("vkGetImageMemoryRequirements2KHR", c_void_p),
        # VMA_BIND_MEMORY2 || VMA_VULKAN_VERSION >= 1001000
        ("vkBindBufferMemory2KHR", c_void_p),
        ("vkBindImageMemory2KHR", c_void_p),
        # VMA_MEMORY_BUDGET || VMA_VULKAN_VERSION >= 1001000
        ("vkGetPhysicalDeviceMemoryProperties2KHR", c_void_p),
        # VMA_KHR_MAINTENANCE4 || VMA_VULKAN_VERSION >= 1003000
        ("vkGetDeviceBufferMemoryRequirements", c_void_p),
        ("vkGetDeviceImageMemoryRequirements", c_void_p),
        # VMA_EXTERNAL_MEMORY_WIN32 (always present, may be NULL)
        ("vkGetMemoryWin32HandleKHR", c_void_p),
    ]


class VmaAllocatorCreateInfo(Structure):
    """Parameters for creating VMA allocator"""

    _fields_ = [
        ("flags", c_uint32),
        ("physicalDevice", c_void_p),
        ("device", c_void_p),
        ("preferredLargeHeapBlockSize", c_uint64),
        ("pAllocationCallbacks", c_void_p),
        ("pDeviceMemoryCallbacks", c_void_p),
        ("pHeapSizeLimit", c_void_p),
        ("pVulkanFunctions", POINTER(VmaVulkanFunctions)),
        ("instance", c_void_p),
        ("vulkanApiVersion", c_uint32),
        ("pTypeExternalMemoryHandleTypes", c_void_p),
    ]


class VmaAllocationCreateInfo(Structure):
    """Parameters for creating an allocation"""

    _fields_ = [
        ("flags", c_uint32),
        ("usage", c_uint32),  # VmaMemoryUsage
        ("requiredFlags", c_uint32),
        ("preferredFlags", c_uint32),
        ("memoryTypeBits", c_uint32),
        ("pool", c_void_p),
        ("pUserData", c_void_p),
        ("priority", c_float),
    ]


class VmaAllocationInfo(Structure):
    """Returned info about an allocation"""

    _fields_ = [
        ("memoryType", c_uint32),
        ("deviceMemory", c_void_p),
        ("offset", c_uint64),
        ("size", c_uint64),
        ("pMappedData", c_void_p),
        ("pUserData", c_void_p),
        ("pName", c_void_p),
    ]


class VkBufferCreateInfo(Structure):
    """VkBufferCreateInfo structure"""

    _fields_ = [
        ("sType", c_uint32),
        ("pNext", c_void_p),
        ("flags", c_uint32),
        ("size", c_uint64),
        ("usage", c_uint32),
        ("sharingMode", c_uint32),
        ("queueFamilyIndexCount", c_uint32),
        ("pQueueFamilyIndices", c_void_p),
    ]


# VMA Library loader
_vma_lib = None
_vma_lib_path = None


def _get_vma_lib_path():
    """Get path to VMA shared library"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if platform.system() == "Windows":
        lib_name = "vma.dll"
    elif platform.system() == "Darwin":
        lib_name = "libvma.dylib"
    else:
        lib_name = "libvma.so"

    # Check multiple locations
    search_paths = [
        os.path.join(base_dir, "lib", lib_name),
        os.path.join(base_dir, lib_name),
        os.path.join(base_dir, "backend", lib_name),
        os.path.join(base_dir, "VulkanMemoryAllocator", "build", lib_name),
        os.path.join(base_dir, "VulkanMemoryAllocator", "build", "Release", lib_name),
    ]

    for path in search_paths:
        if os.path.exists(path):
            return path

    return None


def _load_vma_library():
    """Load VMA shared library"""
    global _vma_lib, _vma_lib_path

    if _vma_lib is not None:
        return _vma_lib

    lib_path = _get_vma_lib_path()
    if lib_path is None:
        raise RuntimeError(
            "VMA library not found. Build it with: python -m grilly.backend.vma_wrapper --build"
        )

    try:
        _vma_lib = ctypes.CDLL(lib_path)
        _vma_lib_path = lib_path
        _setup_vma_functions(_vma_lib)
        logger.info(f"Loaded VMA library from {lib_path}")
        return _vma_lib
    except Exception as e:
        raise RuntimeError(f"Failed to load VMA library from {lib_path}: {e}")


def _setup_vma_functions(lib):
    """Set up function signatures for VMA library"""
    # vmaCreateAllocator
    lib.vmaCreateAllocator.argtypes = [POINTER(VmaAllocatorCreateInfo), POINTER(c_void_p)]
    lib.vmaCreateAllocator.restype = c_int

    # vmaDestroyAllocator
    lib.vmaDestroyAllocator.argtypes = [c_void_p]
    lib.vmaDestroyAllocator.restype = None

    # vmaCreateBuffer
    lib.vmaCreateBuffer.argtypes = [
        c_void_p,  # allocator
        POINTER(VkBufferCreateInfo),
        POINTER(VmaAllocationCreateInfo),
        POINTER(c_void_p),  # pBuffer
        POINTER(c_void_p),  # pAllocation
        POINTER(VmaAllocationInfo),  # pAllocationInfo
    ]
    lib.vmaCreateBuffer.restype = c_int

    # vmaDestroyBuffer
    lib.vmaDestroyBuffer.argtypes = [c_void_p, c_void_p, c_void_p]
    lib.vmaDestroyBuffer.restype = None

    # vmaMapMemory
    lib.vmaMapMemory.argtypes = [c_void_p, c_void_p, POINTER(c_void_p)]
    lib.vmaMapMemory.restype = c_int

    # vmaUnmapMemory
    lib.vmaUnmapMemory.argtypes = [c_void_p, c_void_p]
    lib.vmaUnmapMemory.restype = None

    # vmaFlushAllocation
    lib.vmaFlushAllocation.argtypes = [c_void_p, c_void_p, c_uint64, c_uint64]
    lib.vmaFlushAllocation.restype = c_int

    # vmaInvalidateAllocation
    lib.vmaInvalidateAllocation.argtypes = [c_void_p, c_void_p, c_uint64, c_uint64]
    lib.vmaInvalidateAllocation.restype = c_int


def is_vma_available():
    """Check if VMA library is available"""
    try:
        _load_vma_library()
        return True
    except Exception:
        return False


class VMAAllocator:
    """
    VMA Allocator wrapper.

    Provides AMD/NVIDIA/Intel optimized GPU memory allocation.

    Example:
        >>> allocator = VMAAllocator(vulkan_core)
        >>> buffer = allocator.create_buffer(1024, usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        >>> # ... use buffer ...
        >>> allocator.destroy_buffer(buffer)
    """

    def __init__(self, core):
        """
        Initialize VMA allocator.

        Args:
            core: VulkanCore instance
        """
        self.core = core
        self._lib = _load_vma_library()
        self._allocator = None
        self._init_allocator()

    def _init_allocator(self):
        """Initialize VMA allocator with Vulkan handles"""
        import vulkan as vk

        # Get Vulkan handles as integers
        physical_device_int = int(vk.ffi.cast("uintptr_t", self.core.physical_device))
        device_int = int(vk.ffi.cast("uintptr_t", self.core.device))
        instance_int = (
            int(vk.ffi.cast("uintptr_t", self.core.instance))
            if hasattr(self.core, "instance")
            else 0
        )

        # Load vulkan-1.dll
        if platform.system() == "Windows":
            vulkan_dll = ctypes.WinDLL("vulkan-1.dll")
        else:
            vulkan_dll = ctypes.CDLL("libvulkan.so.1")

        # VMA with VMA_DYNAMIC_VULKAN_FUNCTIONS=1 only needs these two functions
        # It will load all other functions dynamically using these
        vulkan_functions = VmaVulkanFunctions()
        vulkan_functions.vkGetInstanceProcAddr = ctypes.cast(
            vulkan_dll.vkGetInstanceProcAddr, c_void_p
        ).value
        vulkan_functions.vkGetDeviceProcAddr = ctypes.cast(
            vulkan_dll.vkGetDeviceProcAddr, c_void_p
        ).value

        logger.debug(f"vkGetInstanceProcAddr: {hex(vulkan_functions.vkGetInstanceProcAddr)}")
        logger.debug(f"vkGetDeviceProcAddr: {hex(vulkan_functions.vkGetDeviceProcAddr)}")

        # Create allocator - VMA will load all other functions dynamically
        create_info = VmaAllocatorCreateInfo()
        create_info.flags = 0
        create_info.physicalDevice = c_void_p(physical_device_int)
        create_info.device = c_void_p(device_int)
        create_info.instance = c_void_p(instance_int)
        create_info.vulkanApiVersion = 0x00401000  # VK_API_VERSION_1_1
        create_info.pVulkanFunctions = ctypes.pointer(vulkan_functions)

        allocator = c_void_p()
        result = self._lib.vmaCreateAllocator(byref(create_info), byref(allocator))

        if result != VK_SUCCESS:
            raise RuntimeError(f"vmaCreateAllocator failed with code {result}")

        self._allocator = allocator
        logger.info("VMA allocator initialized successfully")

    def create_buffer(self, size: int, usage: int, memory_usage: int = VMA_MEMORY_USAGE_AUTO):
        """
        Create a buffer with VMA allocation.

        Args:
            size: Buffer size in bytes
            usage: Vulkan buffer usage flags
            memory_usage: VMA memory usage hint

        Returns:
            VMABuffer instance
        """
        VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO = 12
        VK_SHARING_MODE_EXCLUSIVE = 0

        buffer_info = VkBufferCreateInfo()
        buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO
        buffer_info.size = size
        buffer_info.usage = usage
        buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE

        alloc_info = VmaAllocationCreateInfo()
        alloc_info.usage = memory_usage
        alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT

        buffer = c_void_p()
        allocation = c_void_p()
        allocation_info = VmaAllocationInfo()

        result = self._lib.vmaCreateBuffer(
            self._allocator,
            byref(buffer_info),
            byref(alloc_info),
            byref(buffer),
            byref(allocation),
            byref(allocation_info),
        )

        if result != VK_SUCCESS:
            raise RuntimeError(f"vmaCreateBuffer failed with code {result}")

        return VMABuffer(
            allocator=self,
            handle=buffer.value,
            allocation=allocation.value,
            size=size,
            mapped_ptr=allocation_info.pMappedData,
        )

    def destroy_buffer(self, buffer: "VMABuffer"):
        """Destroy a VMA-allocated buffer"""
        if buffer.handle and buffer.allocation:
            self._lib.vmaDestroyBuffer(
                self._allocator, c_void_p(buffer.handle), c_void_p(buffer.allocation)
            )
            buffer.handle = None
            buffer.allocation = None

    def map_memory(self, allocation) -> c_void_p:
        """Map allocation memory for CPU access"""
        mapped_ptr = c_void_p()
        result = self._lib.vmaMapMemory(self._allocator, c_void_p(allocation), byref(mapped_ptr))
        if result != VK_SUCCESS:
            raise RuntimeError(f"vmaMapMemory failed with code {result}")
        return mapped_ptr.value

    def unmap_memory(self, allocation):
        """Unmap allocation memory"""
        self._lib.vmaUnmapMemory(self._allocator, c_void_p(allocation))

    def flush_allocation(self, allocation, offset: int = 0, size: int = 0xFFFFFFFFFFFFFFFF):
        """Flush allocation to make CPU writes visible to GPU"""
        result = self._lib.vmaFlushAllocation(
            self._allocator, c_void_p(allocation), c_uint64(offset), c_uint64(size)
        )
        return result == VK_SUCCESS

    def invalidate_allocation(self, allocation, offset: int = 0, size: int = 0xFFFFFFFFFFFFFFFF):
        """Invalidate allocation to make GPU writes visible to CPU"""
        result = self._lib.vmaInvalidateAllocation(
            self._allocator, c_void_p(allocation), c_uint64(offset), c_uint64(size)
        )
        return result == VK_SUCCESS

    def __del__(self):
        """Cleanup allocator"""
        if self._allocator:
            try:
                self._lib.vmaDestroyAllocator(self._allocator)
            except Exception:
                pass


class VMABuffer:
    """
    A buffer allocated via VMA.

    Attributes:
        handle: Vulkan buffer handle (as integer)
        allocation: VMA allocation handle
        size: Buffer size in bytes
    """

    __slots__ = (
        "allocator",
        "handle",
        "allocation",
        "size",
        "mapped_ptr",
        "in_use",
        "last_used",
        "bucket_size",
        "_vk_handle",
    )

    def __init__(self, allocator, handle, allocation, size, mapped_ptr=None):
        """Initialize the instance."""

        self.allocator = allocator
        self.handle = handle
        self.allocation = allocation
        self.size = size
        self.mapped_ptr = mapped_ptr
        self.in_use = True
        self.last_used = 0
        self.bucket_size = size
        self._vk_handle = None

    def get_vulkan_handle(self):
        """Get buffer handle compatible with vulkan package"""
        if self._vk_handle is None and self.handle is not None:
            import vulkan as vk

            self._vk_handle = vk.ffi.cast("VkBuffer", self.handle)
        return self._vk_handle

    def upload(self, data: np.ndarray):
        """Upload numpy array to buffer"""
        # Map memory
        mapped = self.allocator.map_memory(self.allocation)

        # Copy data
        data_bytes = data.tobytes()
        ctypes.memmove(mapped, data_bytes, len(data_bytes))

        # Flush and unmap
        self.allocator.flush_allocation(self.allocation)
        self.allocator.unmap_memory(self.allocation)

    def download(self, size: int, dtype=np.float32) -> np.ndarray:
        """Download data from buffer to numpy array"""
        # Invalidate cache
        self.allocator.invalidate_allocation(self.allocation)

        # Map memory
        mapped = self.allocator.map_memory(self.allocation)

        # Copy to numpy
        buffer = (ctypes.c_char * size).from_address(mapped)
        result = np.frombuffer(bytes(buffer), dtype=dtype).copy()

        # Unmap
        self.allocator.unmap_memory(self.allocation)

        return result

    def release(self):
        """Destroy this buffer"""
        if self.allocator and self.handle:
            self.allocator.destroy_buffer(self)
            self.in_use = False

    def __del__(self):
        """Auto-cleanup"""
        if getattr(self, "in_use", False) and getattr(self, "handle", None):
            try:
                self.release()
            except Exception:
                pass


def build_vma_library():
    """
    Build VMA as a shared library.

    Requires:
    - Windows: Visual Studio Build Tools with C++ workload
    - Linux: GCC/Clang and development headers
    - Vulkan SDK installed
    """
    import shutil
    import subprocess

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vma_dir = os.path.join(base_dir, "VulkanMemoryAllocator")

    if not os.path.exists(vma_dir):
        raise RuntimeError(f"VMA source not found at {vma_dir}")

    # Create build directory
    build_dir = os.path.join(vma_dir, "build")
    os.makedirs(build_dir, exist_ok=True)

    # Create wrapper source that exports VMA functions
    wrapper_src = os.path.join(build_dir, "vma_wrapper.cpp")
    with open(wrapper_src, "w") as f:
        f.write("""
// VMA Wrapper - exports VMA functions as a shared library
#define VMA_IMPLEMENTATION
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1

#ifdef _WIN32
#define VMA_EXPORT __declspec(dllexport)
#else
#define VMA_EXPORT __attribute__((visibility("default")))
#endif

#include "../include/vk_mem_alloc.h"

// Export all VMA functions
extern "C" {
    VMA_EXPORT VkResult vmaCreateAllocator(const VmaAllocatorCreateInfo* pCreateInfo, VmaAllocator* pAllocator) {
        return ::vmaCreateAllocator(pCreateInfo, pAllocator);
    }

    VMA_EXPORT void vmaDestroyAllocator(VmaAllocator allocator) {
        ::vmaDestroyAllocator(allocator);
    }

    VMA_EXPORT VkResult vmaCreateBuffer(
        VmaAllocator allocator,
        const VkBufferCreateInfo* pBufferCreateInfo,
        const VmaAllocationCreateInfo* pAllocationCreateInfo,
        VkBuffer* pBuffer,
        VmaAllocation* pAllocation,
        VmaAllocationInfo* pAllocationInfo
    ) {
        return ::vmaCreateBuffer(allocator, pBufferCreateInfo, pAllocationCreateInfo, pBuffer, pAllocation, pAllocationInfo);
    }

    VMA_EXPORT void vmaDestroyBuffer(VmaAllocator allocator, VkBuffer buffer, VmaAllocation allocation) {
        ::vmaDestroyBuffer(allocator, buffer, allocation);
    }

    VMA_EXPORT VkResult vmaMapMemory(VmaAllocator allocator, VmaAllocation allocation, void** ppData) {
        return ::vmaMapMemory(allocator, allocation, ppData);
    }

    VMA_EXPORT void vmaUnmapMemory(VmaAllocator allocator, VmaAllocation allocation) {
        ::vmaUnmapMemory(allocator, allocation);
    }

    VMA_EXPORT VkResult vmaFlushAllocation(VmaAllocator allocator, VmaAllocation allocation, VkDeviceSize offset, VkDeviceSize size) {
        return ::vmaFlushAllocation(allocator, allocation, offset, size);
    }

    VMA_EXPORT VkResult vmaInvalidateAllocation(VmaAllocator allocator, VmaAllocation allocation, VkDeviceSize offset, VkDeviceSize size) {
        return ::vmaInvalidateAllocation(allocator, allocation, offset, size);
    }
}
""")

    print(f"Building VMA library in {build_dir}...")

    if platform.system() == "Windows":
        # Find Vulkan SDK
        vulkan_sdk = os.environ.get("VULKAN_SDK", r"C:\VulkanSDK\1.3.290.0")
        if not os.path.exists(vulkan_sdk):
            # Try to find it
            vulkan_base = r"C:\VulkanSDK"
            if os.path.exists(vulkan_base):
                versions = os.listdir(vulkan_base)
                if versions:
                    vulkan_sdk = os.path.join(vulkan_base, sorted(versions)[-1])

        vulkan_include = os.path.join(vulkan_sdk, "Include")
        os.path.join(vulkan_sdk, "Lib")

        # Use CMake
        cmake_cmd = [
            "cmake",
            "..",
            f"-DVULKAN_SDK={vulkan_sdk}",
            "-DCMAKE_BUILD_TYPE=Release",
            "-G",
            "Visual Studio 17 2022",
            "-A",
            "x64",
        ]

        # Create CMakeLists.txt
        cmake_file = os.path.join(build_dir, "CMakeLists.txt")
        with open(cmake_file, "w") as f:
            f.write("""
cmake_minimum_required(VERSION 3.10)
project(vma_wrapper)

set(CMAKE_CXX_STANDARD 11)

find_package(Vulkan REQUIRED)

add_library(vma SHARED vma_wrapper.cpp)
target_include_directories(vma PRIVATE ${Vulkan_INCLUDE_DIRS})
target_link_libraries(vma PRIVATE ${Vulkan_LIBRARIES})

if(WIN32)
    set_target_properties(vma PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}"
        LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}"
    )
endif()
""")

        # Run CMake
        subprocess.run(cmake_cmd, cwd=build_dir, check=True)
        subprocess.run(["cmake", "--build", ".", "--config", "Release"], cwd=build_dir, check=True)

        # Copy DLL to lib directory
        lib_dir = os.path.join(base_dir, "lib")
        os.makedirs(lib_dir, exist_ok=True)

        dll_src = os.path.join(build_dir, "Release", "vma.dll")
        if not os.path.exists(dll_src):
            dll_src = os.path.join(build_dir, "vma.dll")

        if os.path.exists(dll_src):
            dll_dst = os.path.join(lib_dir, "vma.dll")
            shutil.copy2(dll_src, dll_dst)
            print(f"VMA library built: {dll_dst}")
        else:
            print("Warning: Could not find built DLL")

    else:
        # Linux/macOS - use g++
        vulkan_include = "/usr/include"
        if os.path.exists("/usr/include/vulkan"):
            vulkan_include = "/usr/include"

        output_lib = "libvma.so" if platform.system() == "Linux" else "libvma.dylib"
        output_path = os.path.join(build_dir, output_lib)

        compile_cmd = [
            "g++",
            "-shared",
            "-fPIC",
            "-O2",
            "-std=c++11",
            f"-I{vulkan_include}",
            "-I" + os.path.join(vma_dir, "include"),
            wrapper_src,
            "-o",
            output_path,
            "-lvulkan",
        ]

        subprocess.run(compile_cmd, check=True)

        # Copy to lib directory
        lib_dir = os.path.join(base_dir, "lib")
        os.makedirs(lib_dir, exist_ok=True)
        dst = os.path.join(lib_dir, output_lib)
        shutil.copy2(output_path, dst)
        print(f"VMA library built: {dst}")


if __name__ == "__main__":
    if "--build" in sys.argv:
        build_vma_library()
    else:
        print("VMA Wrapper")
        print("  --build    Build VMA shared library")
        print()
        print(f"VMA available: {is_vma_available()}")
