"""
Core Vulkan initialization, buffer management, and dispatch operations.
"""

import ctypes
import logging
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .base import VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VULKAN_AVAILABLE

if VULKAN_AVAILABLE:
    from vulkan import *

logger = logging.getLogger(__name__)

import os as _os

_DEBUG_UPLOAD = _os.getenv("GRILLY_DEBUG_UPLOAD", "0") == "1"

# Enable provisional Vulkan extensions (required since spec 1.2.171)
_os.environ.setdefault("VK_ENABLE_BETA_EXTENSIONS", "1")


class VulkanCore:
    """Core Vulkan operations: initialization, buffers, and dispatch"""

    def __init__(self, shader_dir: str = None):
        """Initialize the instance."""

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
        """Load all .spv files from main and experimental directories"""
        shaders = {}

        # Load from main spv directory
        spv_dir = Path(self.shader_dir) / "spv"
        if not spv_dir.exists():
            spv_dir = Path(self.shader_dir)

        required_shaders = [
            "fnn-xavier-init",
            "convd_im2col",
            "conv2d-backward-weight",
            "conv2d-backward-input",
            "conv2d-forward",
            "gemm_mnk",
            "loss-cross-entropy",
            # Optimizer kernels required for native training throughput.
            "adam-update",
            "adamw-update",
            # Native SSM fused recurrence path
            "ssm-fused-math",
            "ssm-fused-uv",
            # LoRA training/inference shaders
            "lora-forward",
            "lora-backward",
            # SSD chunked scan for training mode
            "ssd-scan-chunks",
        ]
        self._ensure_required_shaders(required_shaders, spv_dir)

        for spv_file in spv_dir.glob("*.spv"):
            name = spv_file.stem
            with open(spv_file, "rb") as f:
                shaders[name] = f.read()

        # Also load from experimental/spv directory
        experimental_spv_dir = Path(self.shader_dir) / "experimental" / "spv"
        if experimental_spv_dir.exists():
            for spv_file in experimental_spv_dir.glob("*.spv"):
                name = spv_file.stem
                # Only add if not already loaded (main directory takes precedence)
                if name not in shaders:
                    with open(spv_file, "rb") as f:
                        shaders[name] = f.read()

        # Check for missing shaders and warn
        for shader_name in required_shaders:
            if shader_name not in shaders:
                print(
                    f"[WARNING] Shader {shader_name}.spv not found - GPU Xavier init will use CPU fallback"
                )
                print(f"  To compile: glslc {shader_name}.glsl -o spv/{shader_name}.spv")

        return shaders

    def _ensure_required_shaders(self, shader_names: list[str], spv_dir: Path) -> None:
        """Best-effort compile of missing shader binaries from GLSL sources."""
        missing = [name for name in shader_names if not (spv_dir / f"{name}.spv").exists()]
        if not missing:
            return

        glslc = shutil.which("glslc")
        if glslc is None:
            logger.debug("glslc not found; missing shaders will use CPU fallback: %s", missing)
            return

        for shader_name in missing:
            src_path = Path(self.shader_dir) / f"{shader_name}.glsl"
            out_path = spv_dir / f"{shader_name}.spv"
            if not src_path.exists():
                continue

            try:
                spv_dir.mkdir(parents=True, exist_ok=True)
                subprocess.run(
                    [
                        glslc,
                        "-fshader-stage=compute",
                        str(src_path),
                        "-o",
                        str(out_path),
                        "--target-env=vulkan1.0",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                logger.info("Compiled missing shader %s -> %s", src_path.name, out_path.name)
            except Exception as exc:
                logger.warning("Failed to compile shader %s: %s", src_path.name, exc)

    # Extensions we want to enable when available
    _DESIRED_EXTENSIONS = [
        "VK_KHR_cooperative_matrix",
        "VK_KHR_shader_atomic_int64",
        "VK_EXT_shader_atomic_float",
        "VK_EXT_shader_atomic_float2",
        "VK_KHR_shader_float16_int8",
        "VK_KHR_16bit_storage",
        "VK_KHR_8bit_storage",
        "VK_KHR_storage_buffer_storage_class",
        "VK_KHR_vulkan_memory_model",
    ]

    def _init_vulkan(self):
        """Initialize Vulkan instance, device, queue"""
        # Create instance (Vulkan 1.1 for subgroup ops + cooperative matrix)
        app_info = VkApplicationInfo(
            sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName="GrillCheese",
            applicationVersion=VK_MAKE_VERSION(1, 0, 0),
            pEngineName="SNN-Compute",
            engineVersion=VK_MAKE_VERSION(1, 0, 0),
            apiVersion=VK_MAKE_VERSION(1, 1, 0),
        )

        create_info = VkInstanceCreateInfo(
            sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, pApplicationInfo=app_info
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

        # ── Enumerate and enable device extensions ──
        available_exts = set()
        for e in vkEnumerateDeviceExtensionProperties(self.physical_device, None):
            name = e.extensionName
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            available_exts.add(name.rstrip("\x00"))

        enabled_extensions = [e for e in self._DESIRED_EXTENSIONS if e in available_exts]
        self.enabled_extensions = set(enabled_extensions)

        if not hasattr(VulkanCore, "_logged_exts"):
            if enabled_extensions:
                print(f"[OK] Vulkan extensions: {', '.join(enabled_extensions)}")
            VulkanCore._logged_exts = True

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
            pQueuePriorities=[1.0],
        )

        # Base features (sparse binding for tiling)
        device_features = VkPhysicalDeviceFeatures()
        try:
            if hasattr(self.device_features, "sparseBinding"):
                device_features.sparseBinding = self.device_features.sparseBinding
            if hasattr(self.device_features, "sparseResidencyBuffer"):
                device_features.sparseResidencyBuffer = self.device_features.sparseResidencyBuffer
        except Exception:
            pass

        # ── Build pNext feature chain for enabled extensions ──
        # Query actual device feature support first, then only request what
        # the hardware reports.  vulkan-python pNext must be set at
        # construction time (void* can't be reassigned), so we build the
        # chain bottom-up.
        pnext_tail = None

        if "VK_KHR_16bit_storage" in self.enabled_extensions:
            q = VkPhysicalDevice16BitStorageFeatures(
                sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES,
            )
            qf = VkPhysicalDeviceFeatures2(
                sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, pNext=q
            )
            vkGetPhysicalDeviceFeatures2(self.physical_device, qf)
            pnext_tail = VkPhysicalDevice16BitStorageFeatures(
                sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES,
                storageBuffer16BitAccess=q.storageBuffer16BitAccess,
                pNext=pnext_tail,
            )

        if "VK_KHR_shader_float16_int8" in self.enabled_extensions:
            q = VkPhysicalDeviceShaderFloat16Int8Features(
                sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES,
            )
            qf = VkPhysicalDeviceFeatures2(
                sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, pNext=q
            )
            vkGetPhysicalDeviceFeatures2(self.physical_device, qf)
            pnext_tail = VkPhysicalDeviceShaderFloat16Int8Features(
                sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES,
                shaderFloat16=q.shaderFloat16,
                shaderInt8=q.shaderInt8,
                pNext=pnext_tail,
            )

        if "VK_EXT_shader_atomic_float" in self.enabled_extensions:
            q = VkPhysicalDeviceShaderAtomicFloatFeaturesEXT(
                sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT,
            )
            qf = VkPhysicalDeviceFeatures2(
                sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, pNext=q
            )
            vkGetPhysicalDeviceFeatures2(self.physical_device, qf)
            pnext_tail = VkPhysicalDeviceShaderAtomicFloatFeaturesEXT(
                sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT,
                shaderBufferFloat32Atomics=q.shaderBufferFloat32Atomics,
                shaderBufferFloat32AtomicAdd=q.shaderBufferFloat32AtomicAdd,
                shaderSharedFloat32Atomics=q.shaderSharedFloat32Atomics,
                shaderSharedFloat32AtomicAdd=q.shaderSharedFloat32AtomicAdd,
                pNext=pnext_tail,
            )

        if "VK_KHR_shader_atomic_int64" in self.enabled_extensions:
            q = VkPhysicalDeviceShaderAtomicInt64Features(
                sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_INT64_FEATURES,
            )
            qf = VkPhysicalDeviceFeatures2(
                sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, pNext=q
            )
            vkGetPhysicalDeviceFeatures2(self.physical_device, qf)
            pnext_tail = VkPhysicalDeviceShaderAtomicInt64Features(
                sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_INT64_FEATURES,
                shaderBufferInt64Atomics=q.shaderBufferInt64Atomics,
                shaderSharedInt64Atomics=q.shaderSharedInt64Atomics,
                pNext=pnext_tail,
            )

        if "VK_KHR_cooperative_matrix" in self.enabled_extensions:
            q = VkPhysicalDeviceCooperativeMatrixFeaturesKHR(
                sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR,
            )
            qf = VkPhysicalDeviceFeatures2(
                sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, pNext=q
            )
            vkGetPhysicalDeviceFeatures2(self.physical_device, qf)
            pnext_tail = VkPhysicalDeviceCooperativeMatrixFeaturesKHR(
                sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR,
                cooperativeMatrix=q.cooperativeMatrix,
                pNext=pnext_tail,
            )

        # Wrap base features + pNext chain in VkPhysicalDeviceFeatures2
        features2 = VkPhysicalDeviceFeatures2(
            sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
            features=device_features,
            pNext=pnext_tail,
        )

        device_create_info = VkDeviceCreateInfo(
            sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            queueCreateInfoCount=1,
            pQueueCreateInfos=[queue_create_info],
            enabledExtensionCount=len(enabled_extensions),
            ppEnabledExtensionNames=enabled_extensions,
            pEnabledFeatures=None,  # features in pNext chain via features2
            pNext=features2,
        )

        self.device = vkCreateDevice(self.physical_device, device_create_info, None)
        self.queue = vkGetDeviceQueue(self.device, compute_queue_family, 0)
        self.compute_queue_family = compute_queue_family

        # Create command pool
        pool_info = VkCommandPoolCreateInfo(
            sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            queueFamilyIndex=compute_queue_family,
            flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        )
        self.command_pool = vkCreateCommandPool(self.device, pool_info, None)

        # Pre-allocate a reusable command buffer (pool uses RESET_COMMAND_BUFFER_BIT)
        cmd_alloc_info = VkCommandBufferAllocateInfo(
            sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            commandPool=self.command_pool,
            level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1,
        )
        self._cmd_buffer = vkAllocateCommandBuffers(self.device, cmd_alloc_info)[0]

        # Create a reusable fence for dispatch synchronization.
        # Starts signaled so the first _dispatch_compute wait is a no-op.
        fence_info = VkFenceCreateInfo(
            sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            flags=VK_FENCE_CREATE_SIGNALED_BIT,
        )
        self._fence = vkCreateFence(self.device, fence_info, None)

        # Batch command buffer + fence for CommandRecorder (multi-dispatch chaining).
        # Separate from single-shot _cmd_buffer/_fence to avoid conflicts.
        batch_alloc = VkCommandBufferAllocateInfo(
            sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            commandPool=self.command_pool,
            level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1,
        )
        self._batch_cmd_buffer = vkAllocateCommandBuffers(self.device, batch_alloc)[0]
        self._batch_fence = vkCreateFence(self.device, fence_info, None)

        # Create descriptor pool (large enough for all shaders)
        pool_sizes = [
            VkDescriptorPoolSize(type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorCount=1000)
        ]

        pool_info = VkDescriptorPoolCreateInfo(
            sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            flags=VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
            maxSets=500,
            poolSizeCount=len(pool_sizes),
            pPoolSizes=pool_sizes,
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
            """Execute info."""

            props = vkGetPhysicalDeviceProperties(dev)
            name = (
                props.deviceName.decode("utf-8")
                if isinstance(props.deviceName, bytes)
                else props.deviceName
            )
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
            readable = ", ".join(
                [
                    f"{e['idx']}:{e['name']} (type {e['type']}, vendor 0x{e['vendor']:04X})"
                    for e in entries
                ]
            )
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
        non_cpu = [
            e
            for e in entries
            if e["type"] != VK_PHYSICAL_DEVICE_TYPE_CPU and "llvmpipe" not in str(e["name"]).lower()
        ]
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
        if all(
            e["type"] == VK_PHYSICAL_DEVICE_TYPE_CPU or "llvmpipe" in str(e["name"]).lower()
            for e in candidates
        ):
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
            return {"available": False}

        try:
            features = self.device_features
            props = self.device_properties

            tiling_info = {
                "available": True,
                "sparse_binding": getattr(features, "sparseBinding", False),
                "sparse_residency": getattr(features, "sparseResidencyBuffer", False),
                "sparse_residency_aliased": getattr(features, "sparseResidencyAliased", False),
                "shader_image_gather_extended": getattr(
                    features, "shaderImageGatherExtended", False
                ),
                "device_name": props.deviceName.decode("utf-8")
                if isinstance(props.deviceName, bytes)
                else props.deviceName,
                "vendor_id": props.vendorID,
                "device_type": props.deviceType,
                "optimal_tiling": True,  # All modern GPUs support optimal tiling
            }

            # Check for vendor-specific tiling features
            vendor_name = "Unknown"
            if props.vendorID == 0x1002:
                vendor_name = "AMD"
                # AMD GPUs typically have good tiling support
                tiling_info["amd_optimized"] = True
            elif props.vendorID == 0x10DE:
                vendor_name = "NVIDIA"
                # NVIDIA GPUs have excellent tiling support
                tiling_info["nvidia_optimized"] = True
            elif props.vendorID == 0x8086:
                vendor_name = "Intel"
                tiling_info["intel_optimized"] = True

            tiling_info["vendor"] = vendor_name

            # Log tiling capabilities
            if not hasattr(VulkanCore, "_logged_tiling"):
                if tiling_info["sparse_binding"]:
                    print("[OK] Tiling support: Sparse binding enabled")
                if tiling_info["sparse_residency"]:
                    print("[OK] Tiling support: Sparse residency enabled")
                VulkanCore._logged_tiling = True

            return tiling_info

        except Exception as e:
            print(f"[WARNING] Failed to check tiling support: {e}")
            return {"available": False, "error": str(e)}

    def get_tiling_info(self) -> dict:
        """
        Get tiling support information

        Returns:
            Dictionary with tiling capabilities
        """
        return getattr(self, "tiling_support", {"available": False})

    def has_extension(self, ext_name: str) -> bool:
        """Check if a Vulkan device extension is enabled."""
        return ext_name in getattr(self, "enabled_extensions", set())

    @property
    def has_cooperative_matrix(self) -> bool:
        """True when VK_KHR_cooperative_matrix is enabled."""
        return self.has_extension("VK_KHR_cooperative_matrix")

    @property
    def has_float16(self) -> bool:
        """True when fp16 shader arithmetic + storage buffer access are enabled."""
        return self.has_extension("VK_KHR_shader_float16_int8") and self.has_extension(
            "VK_KHR_16bit_storage"
        )

    def _create_buffer(self, size: int, usage: int = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT):
        """Create Vulkan buffer and allocate memory.

        usage defaults to VK_BUFFER_USAGE_STORAGE_BUFFER_BIT so callers that
        only need a generic storage buffer can omit the flag (backward
        compatibility with earlier call sites).
        """
        buffer_info = VkBufferCreateInfo(
            sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            size=size,
            usage=usage,
            sharingMode=VK_SHARING_MODE_EXCLUSIVE,
        )

        buffer = vkCreateBuffer(self.device, buffer_info, None)

        # Get memory requirements
        mem_req = vkGetBufferMemoryRequirements(self.device, buffer)

        # Allocate memory
        mem_props = vkGetPhysicalDeviceMemoryProperties(self.physical_device)
        mem_type_index = self._find_memory_type(
            mem_req.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            mem_props,
        )

        alloc_info = VkMemoryAllocateInfo(
            sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize=mem_req.size,
            memoryTypeIndex=mem_type_index,
        )

        memory = vkAllocateMemory(self.device, alloc_info, None)
        vkBindBufferMemory(self.device, buffer, memory, 0)

        return buffer, memory

    def _find_memory_type(self, type_filter, properties, mem_props):
        """Find suitable memory type"""
        for i in range(mem_props.memoryTypeCount):
            if (type_filter & (1 << i)) and (
                mem_props.memoryTypes[i].propertyFlags & properties
            ) == properties:
                return i
        raise RuntimeError("Failed to find suitable memory type")

    def _upload_buffer(self, buffer, memory, data: np.ndarray):
        """Upload numpy array to GPU buffer"""
        arr = np.ascontiguousarray(data)
        upload_size = int(arr.nbytes)
        if upload_size == 0:
            return

        # Synchronous driver query — only pay this cost under debug flag.
        if _DEBUG_UPLOAD:
            mem_req = vkGetBufferMemoryRequirements(self.device, buffer)
            if upload_size > int(mem_req.size):
                raise ValueError(
                    f"Upload size {upload_size} exceeds buffer allocation {int(mem_req.size)}"
                )

        data_ptr = vkMapMemory(self.device, memory, 0, upload_size, 0)
        try:
            dst = memoryview(data_ptr)
            src = memoryview(arr).cast("B")
            # Chunked copies are more stable for large uploads on some drivers.
            chunk_bytes = 16 * 1024 * 1024
            for offset in range(0, upload_size, chunk_bytes):
                end = min(offset + chunk_bytes, upload_size)
                dst[offset:end] = src[offset:end]
        finally:
            vkUnmapMemory(self.device, memory)

    def _download_buffer(self, memory, size: int, dtype=np.float32) -> np.ndarray:
        """Download GPU buffer to numpy array"""
        if size <= 0:
            return np.empty(0, dtype=dtype)

        data_ptr = vkMapMemory(self.device, memory, 0, size, 0)
        try:
            element_size = np.dtype(dtype).itemsize
            count = size // element_size
            if count <= 0:
                return np.empty(0, dtype=dtype)

            requested_bytes = int(count * element_size)
            try:
                memview = memoryview(data_ptr)
                available = int(len(memview))
            except Exception:
                memview = None
                available = 0

            if memview is not None and available >= requested_bytes:
                return np.frombuffer(memview[:requested_bytes], dtype=dtype, count=count).copy()

            # Fallback path for drivers/bindings where mapped pointers expose an
            # undersized or non-buffer-compatible memoryview.
            raw = ctypes.string_at(data_ptr, requested_bytes)
            return np.frombuffer(raw, dtype=dtype, count=count).copy()
        finally:
            vkUnmapMemory(self.device, memory)

    def _dispatch_compute(
        self,
        pipeline,
        pipeline_layout,
        descriptor_set,
        workgroup_x: int,
        push_constants: bytes = None,
        workgroup_y: int = 1,
        workgroup_z: int = 1,
    ):
        """Dispatch compute shader using pre-allocated command buffer."""
        command_buffer = self._cmd_buffer

        # Reset and re-record the reusable command buffer
        vkResetCommandBuffer(command_buffer, 0)

        # Begin command buffer
        begin_info = VkCommandBufferBeginInfo(
            sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        )
        vkBeginCommandBuffer(command_buffer, begin_info)

        # Bind pipeline and descriptor set
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline)
        vkCmdBindDescriptorSets(
            command_buffer,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline_layout,
            0,
            1,
            [descriptor_set],
            0,
            None,
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
                ctypes.addressof(push_buf),
            )

        # Dispatch
        vkCmdDispatch(command_buffer, workgroup_x, workgroup_y, workgroup_z)

        vkEndCommandBuffer(command_buffer)

        # Wait for previous dispatch to finish, then reset the fence.
        vkWaitForFences(self.device, 1, [self._fence], VK_TRUE, 2_000_000_000)
        vkResetFences(self.device, 1, [self._fence])

        # Submit with fence — avoids the heavier vkQueueWaitIdle drain.
        submit_info = VkSubmitInfo(
            sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,
            commandBufferCount=1,
            pCommandBuffers=[command_buffer],
        )

        vkQueueSubmit(self.queue, 1, [submit_info], self._fence)

        # Wait immediately so callers can read back results.
        # This is still faster than vkQueueWaitIdle (no full queue drain).
        vkWaitForFences(self.device, 1, [self._fence], VK_TRUE, 2_000_000_000)

    def _wait_fence(self, timeout_ns: int = 2_000_000_000):
        """Wait for the most recent dispatch to complete."""
        vkWaitForFences(self.device, 1, [self._fence], VK_TRUE, timeout_ns)

    def cleanup(self):
        """Cleanup Vulkan resources"""
        if hasattr(self, "device") and self.device:
            device = self.device
            self.device = None  # Prevent double cleanup
            if hasattr(self, "_batch_fence") and self._batch_fence:
                vkDestroyFence(device, self._batch_fence, None)
                self._batch_fence = None
            if hasattr(self, "_fence") and self._fence:
                vkDestroyFence(device, self._fence, None)
                self._fence = None
            if hasattr(self, "descriptor_pool") and self.descriptor_pool:
                vkDestroyDescriptorPool(device, self.descriptor_pool, None)
                self.descriptor_pool = None
            if hasattr(self, "command_pool") and self.command_pool:
                # Free pre-allocated command buffers before destroying pool
                bufs_to_free = []
                for attr in ("_cmd_buffer", "_batch_cmd_buffer"):
                    cb = getattr(self, attr, None)
                    if cb is not None:
                        bufs_to_free.append(cb)
                        setattr(self, attr, None)
                if bufs_to_free:
                    vkFreeCommandBuffers(device, self.command_pool, len(bufs_to_free), bufs_to_free)
                vkDestroyCommandPool(device, self.command_pool, None)
                self.command_pool = None
            vkDestroyDevice(device, None)

        if hasattr(self, "instance") and self.instance:
            instance = self.instance
            self.instance = None  # Prevent double cleanup
            vkDestroyInstance(instance, None)

    # ------------------------------------------------------------------
    # Multi-dispatch command recording
    # ------------------------------------------------------------------
    def record_commands(self) -> "CommandRecorder":
        """Create a CommandRecorder for chaining multiple dispatches.

        Usage::

            with core.record_commands() as rec:
                rec.dispatch(pipeline, layout, desc, (gx, gy, gz), push)
                rec.barrier()
                rec.dispatch(pipeline2, layout2, desc2, (gx2,), push2)
            # single fence wait on __exit__
        """
        return CommandRecorder(self)


class CommandRecorder:
    """Records multiple dispatches into a single command buffer submission.

    Eliminates per-dispatch fence waits by chaining dispatches with
    pipeline barriers inside one command buffer, then submitting once.
    """

    def __init__(self, core: VulkanCore):
        self._core = core
        self._cmd = core._batch_cmd_buffer
        self._fence = core._batch_fence
        self._recording = False

    def begin(self):
        """Reset and begin recording into the batch command buffer."""
        vkResetCommandBuffer(self._cmd, 0)
        begin_info = VkCommandBufferBeginInfo(
            sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        )
        vkBeginCommandBuffer(self._cmd, begin_info)
        self._recording = True

    def dispatch(self, pipeline, pipeline_layout, descriptor_set,
                 workgroups, push_constants=None):
        """Record a compute dispatch (no submit).

        Args:
            workgroups: tuple (x,) or (x, y) or (x, y, z)
        """
        if not self._recording:
            raise RuntimeError("CommandRecorder.begin() not called")

        vkCmdBindPipeline(self._cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline)
        vkCmdBindDescriptorSets(
            self._cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline_layout, 0, 1, [descriptor_set], 0, None,
        )

        if push_constants:
            push_buf = ctypes.create_string_buffer(push_constants)
            vkCmdPushConstants(
                self._cmd, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                0, len(push_constants), ctypes.addressof(push_buf),
            )

        if isinstance(workgroups, (tuple, list)):
            gx = workgroups[0]
            gy = workgroups[1] if len(workgroups) > 1 else 1
            gz = workgroups[2] if len(workgroups) > 2 else 1
        else:
            gx, gy, gz = workgroups, 1, 1

        vkCmdDispatch(self._cmd, gx, gy, gz)

    def barrier(self):
        """Insert COMPUTE→COMPUTE memory barrier (SHADER_WRITE→SHADER_READ)."""
        if not self._recording:
            return
        try:
            # VkMemoryBarrier
            VK_STRUCTURE_TYPE_MEMORY_BARRIER = 46
            VK_ACCESS_SHADER_WRITE_BIT = 0x00000040
            VK_ACCESS_SHADER_READ_BIT = 0x00000020
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT = 0x00000800

            mem_barrier = VkMemoryBarrier(
                sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT,
                dstAccessMask=VK_ACCESS_SHADER_READ_BIT,
            )
            vkCmdPipelineBarrier(
                self._cmd,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,  # srcStageMask
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,  # dstStageMask
                0,  # dependencyFlags
                1, [mem_barrier],  # memoryBarriers
                0, None,  # bufferMemoryBarriers
                0, None,  # imageMemoryBarriers
            )
        except Exception as exc:
            logger.warning("CommandRecorder.barrier() failed: %s", exc)

    def transfer_barrier(self):
        """Insert COMPUTE→TRANSFER barrier (SHADER_WRITE→TRANSFER_READ)."""
        if not self._recording:
            return
        try:
            VK_STRUCTURE_TYPE_MEMORY_BARRIER = 46
            VK_ACCESS_SHADER_WRITE_BIT = 0x00000040
            VK_ACCESS_TRANSFER_READ_BIT = 0x00000800
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT = 0x00000800
            VK_PIPELINE_STAGE_TRANSFER_BIT = 0x00001000

            mem_barrier = VkMemoryBarrier(
                sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT,
                dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT,
            )
            vkCmdPipelineBarrier(
                self._cmd,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                0,
                1, [mem_barrier],
                0, None,
                0, None,
            )
        except Exception as exc:
            logger.warning("CommandRecorder.transfer_barrier() failed: %s", exc)

    def copy_buffer(self, src_handle, dst_handle, size: int):
        """Record vkCmdCopyBuffer for staging transfers."""
        if not self._recording:
            raise RuntimeError("CommandRecorder.begin() not called")
        region = VkBufferCopy(srcOffset=0, dstOffset=0, size=size)
        vkCmdCopyBuffer(self._cmd, src_handle, dst_handle, 1, [region])

    def submit_and_wait(self):
        """End recording, submit, and wait for completion."""
        if not self._recording:
            return
        vkEndCommandBuffer(self._cmd)
        self._recording = False

        vkWaitForFences(self._core.device, 1, [self._fence], VK_TRUE, 2_000_000_000)
        vkResetFences(self._core.device, 1, [self._fence])

        submit_info = VkSubmitInfo(
            sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,
            commandBufferCount=1,
            pCommandBuffers=[self._cmd],
        )
        vkQueueSubmit(self._core.queue, 1, [submit_info], self._fence)
        vkWaitForFences(self._core.device, 1, [self._fence], VK_TRUE, 2_000_000_000)

    def __enter__(self):
        self.begin()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._recording:
            if exc_type is None:
                self.submit_and_wait()
            else:
                # On exception, end the buffer but don't submit
                try:
                    vkEndCommandBuffer(self._cmd)
                except Exception:
                    pass
                self._recording = False
        return False
