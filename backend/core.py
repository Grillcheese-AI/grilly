"""
Core Vulkan initialization, buffer management, and dispatch operations.
"""

import numpy as np
import ctypes
from pathlib import Path
from .base import VULKAN_AVAILABLE

if VULKAN_AVAILABLE:
    from vulkan import *


class VulkanCore:
    """Core Vulkan operations: initialization, buffers, and dispatch"""
    
    def __init__(self, shader_dir: str = None):
        if not VULKAN_AVAILABLE:
            raise RuntimeError("Vulkan not available")
        import os
        # Disable Mesa device_select layer which can force CPU llvmpipe
        os.environ.setdefault("VK_LOADER_LAYERS_DISABLE", "VK_LAYER_MESA_device_select")
        
        # Default to shaders directory relative to this file
        if shader_dir is None:
            shader_dir = Path(__file__).parent.parent / "shaders"
        
        self.shader_dir = Path(shader_dir)
        self.shaders = self._load_shaders()
        # Shaders loaded silently - remove verbose logging
        
        # Initialize Vulkan
        self._init_vulkan()
    
    def _load_shaders(self):
        """Load all .spv files"""
        shaders = {}
        spv_dir = Path(self.shader_dir) / "spv"
        if not spv_dir.exists():
            spv_dir = Path(self.shader_dir)
        
        for spv_file in spv_dir.glob("*.spv"):
            name = spv_file.stem
            with open(spv_file, 'rb') as f:
                shaders[name] = f.read()
        
        # Check for missing shaders and warn
        required_shaders = ['fnn-xavier-init', 'convd_im2col', 'conv2d-backward-weight', 
                            'conv2d-backward-input', 'conv2d-forward', 'gemm_mnk']
        for shader_name in required_shaders:
            if shader_name not in shaders:
                print(f"[WARNING] Shader {shader_name}.spv not found - GPU Xavier init will use CPU fallback")
                print(f"  To compile: glslc {shader_name}.glsl -o spv/{shader_name}.spv")
        
        return shaders
    
    def _init_vulkan(self):
        """Initialize Vulkan instance, device, queue"""
        # Create instance
        app_info = VkApplicationInfo(
            sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName="GrillCheese",
            applicationVersion=VK_MAKE_VERSION(1, 0, 0),
            pEngineName="SNN-Compute",
            engineVersion=VK_MAKE_VERSION(1, 0, 0),
            apiVersion=VK_API_VERSION_1_0
        )
        
        

        create_info = VkInstanceCreateInfo(
            sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationInfo=app_info
        )
        
        self.instance = vkCreateInstance(create_info, None)
        
        # Select GPU (prefer discrete NVIDIA/AMD, avoid llvmpipe/CPU)
        physical_devices = vkEnumeratePhysicalDevices(self.instance)
        self.physical_device = self._select_gpu(physical_devices)
        
        # Get device properties and features
        props = vkGetPhysicalDeviceProperties(self.physical_device)
        features = vkGetPhysicalDeviceFeatures(self.physical_device)
        # Log once per process
        if not hasattr(VulkanCore, "_logged_device"):
            print(f"[OK] Using GPU: {props.deviceName}")
            VulkanCore._logged_device = True
        
        # Store device properties and features for capability queries
        self.device_properties = props
        self.device_features = features
        
        # Check for tiling-related capabilities
        self.tiling_support = self._check_tiling_support()
        
        # Find compute queue family
        queue_families = vkGetPhysicalDeviceQueueFamilyProperties(self.physical_device)
        compute_queue_family = None
        for i, family in enumerate(queue_families):
            if family.queueFlags & VK_QUEUE_COMPUTE_BIT:
                compute_queue_family = i
                break
        
        if compute_queue_family is None:
            raise RuntimeError("No compute queue found")
        
        # Create logical device
        queue_create_info = VkDeviceQueueCreateInfo(
            sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            queueFamilyIndex=compute_queue_family,
            queueCount=1,
            pQueuePriorities=[1.0]
        )
        
        # Request sparse features if available (for tiling support)
        device_features = VkPhysicalDeviceFeatures()
        try:
            # Try to enable sparse features for tiling
            if hasattr(self.device_features, 'sparseBinding'):
                device_features.sparseBinding = self.device_features.sparseBinding
            if hasattr(self.device_features, 'sparseResidencyBuffer'):
                device_features.sparseResidencyBuffer = self.device_features.sparseResidencyBuffer
        except:
            pass

        device_create_info = VkDeviceCreateInfo(
            sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            queueCreateInfoCount=1,
            pQueueCreateInfos=[queue_create_info],
            pEnabledFeatures=device_features
        )
        
        self.device = vkCreateDevice(self.physical_device, device_create_info, None)
        self.queue = vkGetDeviceQueue(self.device, compute_queue_family, 0)
        self.compute_queue_family = compute_queue_family
        
        # Create command pool
        pool_info = VkCommandPoolCreateInfo(
            sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            queueFamilyIndex=compute_queue_family,
            flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
        )
        self.command_pool = vkCreateCommandPool(self.device, pool_info, None)
        
        # Create descriptor pool (large enough for all shaders)
        pool_sizes = [
            VkDescriptorPoolSize(
                type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1000
            )
        ]
        
        pool_info = VkDescriptorPoolCreateInfo(
            sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            flags=VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
            maxSets=500,
            poolSizeCount=len(pool_sizes),
            pPoolSizes=pool_sizes
        )
        
        self.descriptor_pool = vkCreateDescriptorPool(self.device, pool_info, None)
        
        if not hasattr(VulkanCore, "_logged_init"):
            print("[OK] Vulkan device initialized")
            VulkanCore._logged_init = True
    
    def _select_gpu(self, devices):
        """
        Select best GPU: prefer explicit env, then discrete NVIDIA/AMD, avoid CPU/llvmpipe.
        Raises by default if only CPU adapters are found (set ALLOW_CPU_VULKAN=1 to permit).
        """
        import os

        # Build list of candidates with readable info
        def _info(idx, dev):
            props = vkGetPhysicalDeviceProperties(dev)
            name = props.deviceName.decode('utf-8') if isinstance(props.deviceName, bytes) else props.deviceName
            return {
                "idx": idx,
                "device": dev,
                "type": props.deviceType,
                "vendor": props.vendorID,
                "name": name,
            }

        entries = [_info(i, d) for i, d in enumerate(devices)]

        # Log device inventory once to help diagnose CPU binding
        if not hasattr(VulkanCore, "_logged_devices"):
            readable = ", ".join([f"{e['idx']}:{e['name']} (type {e['type']}, vendor 0x{e['vendor']:04X})" for e in entries])
            print(f"[OK] Vulkan devices enumerated: {readable}")
            VulkanCore._logged_devices = True

        # If env set, try that index first (even if CPU—user requested explicitly)
        env_idx = os.getenv("VK_GPU_INDEX")
        if env_idx is not None:
            try:
                idx = int(env_idx)
                if 0 <= idx < len(entries):
                    return entries[idx]["device"]
            except Exception:
                pass

        # Filter out CPU / llvmpipe soft devices
        non_cpu = [e for e in entries if e["type"] != VK_PHYSICAL_DEVICE_TYPE_CPU and "llvmpipe" not in str(e["name"]).lower()]
        candidates = non_cpu or entries

        # Prefer discrete GPUs
        discrete = [e for e in candidates if e["type"] == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU]
        if discrete:
            candidates = discrete

        # Vendor preference: NVIDIA then AMD
        for vendor in (0x10DE, 0x1002):
            for e in candidates:
                if e["vendor"] == vendor:
                    return e["device"]

        # If only CPU/llvmpipe remain, fail loudly unless overridden
        if all(e["type"] == VK_PHYSICAL_DEVICE_TYPE_CPU or "llvmpipe" in str(e["name"]).lower() for e in candidates):
            if os.getenv("ALLOW_CPU_VULKAN", "0").lower() not in ("1", "true", "yes"):
                readable = ", ".join([e["name"] for e in entries])
                raise RuntimeError(
                    f"No discrete Vulkan GPU found (candidates: {readable}). "
                    "Set ALLOW_CPU_VULKAN=1 to allow CPU/llvmpipe fallback."
                )

        # Fallback to first remaining candidate
        return candidates[0]["device"]
    
    def _check_tiling_support(self) -> dict:
        """
        Check for tiling-related GPU capabilities
        
        Returns:
            Dictionary with tiling support information:
            - sparse_binding: Sparse buffer/image binding support
            - sparse_residency: Sparse residency support
            - sparse_residency_aliased: Aliased sparse residency
            - optimal_tiling: Optimal image tiling support (always True for modern GPUs)
            - shader_image_gather_extended: Extended image gather (tiling-friendly)
            - device_name: GPU name
            - vendor_id: Vendor ID (AMD=0x1002, NVIDIA=0x10DE, Intel=0x8086)
        """
        if not VULKAN_AVAILABLE:
            return {'available': False}
        
        try:
            features = self.device_features
            props = self.device_properties
            
            tiling_info = {
                'available': True,
                'sparse_binding': getattr(features, 'sparseBinding', False),
                'sparse_residency': getattr(features, 'sparseResidencyBuffer', False),
                'sparse_residency_aliased': getattr(features, 'sparseResidencyAliased', False),
                'shader_image_gather_extended': getattr(features, 'shaderImageGatherExtended', False),
                'device_name': props.deviceName.decode('utf-8') if isinstance(props.deviceName, bytes) else props.deviceName,
                'vendor_id': props.vendorID,
                'device_type': props.deviceType,
                'optimal_tiling': True,  # All modern GPUs support optimal tiling
            }
            
            # Check for vendor-specific tiling features
            vendor_name = "Unknown"
            if props.vendorID == 0x1002:
                vendor_name = "AMD"
                # AMD GPUs typically have good tiling support
                tiling_info['amd_optimized'] = True
            elif props.vendorID == 0x10DE:
                vendor_name = "NVIDIA"
                # NVIDIA GPUs have excellent tiling support
                tiling_info['nvidia_optimized'] = True
            elif props.vendorID == 0x8086:
                vendor_name = "Intel"
                tiling_info['intel_optimized'] = True
            
            tiling_info['vendor'] = vendor_name
            
            # Log tiling capabilities
            if not hasattr(VulkanCore, "_logged_tiling"):
                if tiling_info['sparse_binding']:
                    print(f"[OK] Tiling support: Sparse binding enabled")
                if tiling_info['sparse_residency']:
                    print(f"[OK] Tiling support: Sparse residency enabled")
                VulkanCore._logged_tiling = True
            
            return tiling_info
            
        except Exception as e:
            print(f"[WARNING] Failed to check tiling support: {e}")
            return {'available': False, 'error': str(e)}
    
    def get_tiling_info(self) -> dict:
        """
        Get tiling support information
        
        Returns:
            Dictionary with tiling capabilities
        """
        return getattr(self, 'tiling_support', {'available': False})
    
    def _create_buffer(self, size: int, usage: int):
        """Create Vulkan buffer and allocate memory"""
        buffer_info = VkBufferCreateInfo(
            sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            size=size,
            usage=usage,
            sharingMode=VK_SHARING_MODE_EXCLUSIVE
        )
        
        buffer = vkCreateBuffer(self.device, buffer_info, None)
        
        # Get memory requirements
        mem_req = vkGetBufferMemoryRequirements(self.device, buffer)
        
        # Allocate memory
        mem_props = vkGetPhysicalDeviceMemoryProperties(self.physical_device)
        mem_type_index = self._find_memory_type(
            mem_req.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            mem_props
        )
        
        alloc_info = VkMemoryAllocateInfo(
            sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize=mem_req.size,
            memoryTypeIndex=mem_type_index
        )
        
        memory = vkAllocateMemory(self.device, alloc_info, None)
        vkBindBufferMemory(self.device, buffer, memory, 0)
        
        return buffer, memory
    
    def _find_memory_type(self, type_filter, properties, mem_props):
        """Find suitable memory type"""
        for i in range(mem_props.memoryTypeCount):
            if (type_filter & (1 << i)) and \
               (mem_props.memoryTypes[i].propertyFlags & properties) == properties:
                return i
        raise RuntimeError("Failed to find suitable memory type")
    
    def _upload_buffer(self, buffer, memory, data: np.ndarray):
        """Upload numpy array to GPU buffer"""
        data_ptr = vkMapMemory(self.device, memory, 0, data.nbytes, 0)
        memview = memoryview(data_ptr)
        memview[:data.nbytes] = data.tobytes()
        vkUnmapMemory(self.device, memory)
    
    def _download_buffer(self, memory, size: int, dtype=np.float32) -> np.ndarray:
        """Download GPU buffer to numpy array"""
        data_ptr = vkMapMemory(self.device, memory, 0, size, 0)
        memview = memoryview(data_ptr)
        element_size = np.dtype(dtype).itemsize
        count = size // element_size
        result = np.frombuffer(memview, dtype=dtype, count=count).copy()
        vkUnmapMemory(self.device, memory)
        return result
    
    def _dispatch_compute(self, pipeline, pipeline_layout, descriptor_set, 
                         workgroup_x: int, push_constants: bytes = None, 
                         workgroup_y: int = 1, workgroup_z: int = 1):
        """Dispatch compute shader"""
        # Allocate command buffer
        alloc_info = VkCommandBufferAllocateInfo(
            sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            commandPool=self.command_pool,
            level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1
        )
        
        command_buffer = vkAllocateCommandBuffers(self.device, alloc_info)[0]
        
        # Begin command buffer
        begin_info = VkCommandBufferBeginInfo(
            sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
        )
        vkBeginCommandBuffer(command_buffer, begin_info)
        
        # Bind pipeline and descriptor set
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline)
        vkCmdBindDescriptorSets(
            command_buffer,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline_layout,
            0, 1, [descriptor_set],
            0, None
        )
        
        # Push constants if provided
        if push_constants:
            push_buf = ctypes.create_string_buffer(push_constants)
            vkCmdPushConstants(
                command_buffer,
                pipeline_layout,
                VK_SHADER_STAGE_COMPUTE_BIT,
                0,
                len(push_constants),
                ctypes.addressof(push_buf)
            )
        
        # Dispatch
        vkCmdDispatch(command_buffer, workgroup_x, workgroup_y, workgroup_z)
        
        vkEndCommandBuffer(command_buffer)
        
        # Submit
        submit_info = VkSubmitInfo(
            sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,
            commandBufferCount=1,
            pCommandBuffers=[command_buffer]
        )
        
        vkQueueSubmit(self.queue, 1, [submit_info], None)
        vkQueueWaitIdle(self.queue)
        
        # Free command buffer
        vkFreeCommandBuffers(self.device, self.command_pool, 1, [command_buffer])
    
    def cleanup(self):
        """Cleanup Vulkan resources"""
        if hasattr(self, 'device') and self.device:
            device = self.device
            self.device = None  # Prevent double cleanup
            if hasattr(self, 'descriptor_pool') and self.descriptor_pool:
                vkDestroyDescriptorPool(device, self.descriptor_pool, None)
                self.descriptor_pool = None
            if hasattr(self, 'command_pool') and self.command_pool:
                vkDestroyCommandPool(device, self.command_pool, None)
                self.command_pool = None
            vkDestroyDevice(device, None)

        if hasattr(self, 'instance') and self.instance:
            instance = self.instance
            self.instance = None  # Prevent double cleanup
            vkDestroyInstance(instance, None)

