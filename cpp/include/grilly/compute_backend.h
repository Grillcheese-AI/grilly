#pragma once
/// Abstract compute backend interface for GPU dispatch.
///
/// Grilly supports multiple GPU APIs through this interface:
///   - VulkanBackend (current, ships now)
///   - OpenGLBackend (future)
///   - OpenCLBackend (future)
///
/// The ops layer (ops::linear, ops::relu, etc.) currently uses Vulkan types
/// directly for performance. This interface provides a higher-level abstraction
/// point for code that doesn't need Vulkan-specific features — e.g., Python
/// bindings, op graph execution, and new backend-agnostic modules.
///
/// Design notes (informed by Francisco Letterio / DevSH SPIR-V talk):
/// - Shader loading is backend-specific: Vulkan loads .spv, OpenGL loads .glsl,
///   OpenCL loads .cl. Each backend handles its own format.
/// - Push constants as raw bytes: Vulkan maps directly, OpenGL uses UBOs,
///   OpenCL uses kernel args. Each backend translates internally.
/// - Buffer handles are opaque uint64_t: Vulkan wraps VkBuffer+VmaAllocation,
///   OpenGL wraps GLuint, OpenCL wraps cl_mem.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace grilly {

/// Describes how a buffer will be used.
struct BufferDesc {
    size_t size = 0;
    enum Usage {
        HostVisible,    ///< CPU-writable, GPU-readable (staging/upload)
        DeviceLocal,    ///< GPU-only VRAM (fastest GPU access)
        Readback,       ///< GPU-writable, CPU-readable (cached for fast reads)
        PreferDevice    ///< Prefer VRAM, fall back to host-visible (ReBAR/SAM)
    } usage = HostVisible;
};

/// Abstract interface for GPU compute backends.
///
/// All buffer handles are opaque uint64_t values. The backend maps them
/// internally to its native buffer type (VkBuffer, GLuint, cl_mem, etc.).
///
/// Usage:
///   auto backend = createBackend("vulkan");
///   backend->loadShaderDir("/path/to/shaders/spv");
///   uint64_t buf = backend->createBuffer({1024, BufferDesc::DeviceLocal});
///   backend->upload(buf, data, 1024);
///   backend->beginBatch();
///   backend->dispatch("fnn-linear", {buf, wBuf, outBuf}, gx, 1, 1, &push, 16);
///   backend->endBatch();
///   backend->download(outBuf, result, outputBytes);
class ComputeBackend {
public:
    virtual ~ComputeBackend() = default;

    // ── Identity ──

    /// Backend name: "vulkan", "opengl", "opencl"
    virtual std::string name() const = 0;

    /// GPU device name reported by the driver.
    virtual std::string deviceName() const = 0;

    // ── Buffer management ──

    /// Create a GPU buffer. Returns an opaque handle.
    virtual uint64_t createBuffer(const BufferDesc& desc) = 0;

    /// Destroy a buffer and free its memory.
    virtual void destroyBuffer(uint64_t buf) = 0;

    /// Upload CPU data to a buffer.
    virtual void upload(uint64_t buf, const void* data, size_t bytes) = 0;

    /// Download GPU data from a buffer.
    virtual void download(uint64_t buf, void* out, size_t bytes) = 0;

    // ── Shader management ──

    /// Load a single shader from a file. The backend determines the format
    /// (Vulkan: .spv, OpenGL: .glsl, OpenCL: .cl).
    virtual void loadShader(const std::string& name,
                            const std::string& path) = 0;

    /// Load all shaders from a directory (matching the backend's extension).
    virtual void loadShaderDir(const std::string& dir) = 0;

    /// Check if a shader has been loaded.
    virtual bool hasShader(const std::string& name) const = 0;

    // ── Compute dispatch ──

    /// Begin recording a batch of dispatches.
    virtual void beginBatch() = 0;

    /// Record a compute dispatch into the current batch.
    /// @param shader   Shader name (as registered via loadShader)
    /// @param buffers  Opaque buffer handles to bind as storage buffers
    /// @param gx,gy,gz Workgroup counts
    /// @param push     Push constant / uniform data (raw bytes)
    /// @param pushBytes Size of push data
    virtual void dispatch(const std::string& shader,
                          const std::vector<uint64_t>& buffers,
                          uint32_t gx, uint32_t gy, uint32_t gz,
                          const void* push = nullptr,
                          uint32_t pushBytes = 0) = 0;

    /// Insert a memory barrier between dispatches.
    virtual void barrier() = 0;

    /// Submit the current batch and wait for completion.
    virtual void endBatch() = 0;

    // ── Capability queries ──

    /// Check for cooperative matrix support (Vulkan tensor cores / WMMA).
    virtual bool hasCooperativeMatrix() const { return false; }

    /// Check for float16 shader support.
    virtual bool hasFloat16() const { return false; }
};

/// Create a backend by name. Currently only "vulkan" is supported.
/// Throws if the requested backend is not available.
std::unique_ptr<ComputeBackend> createBackend(
    const std::string& type = "vulkan");

}  // namespace grilly
