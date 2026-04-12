cmake_minimum_required(VERSION 3.20...4.2)

# CMake 4.x policies
if(POLICY CMP0144)
    cmake_policy(SET CMP0144 NEW)  # honor <PACKAGE>_ROOT variables
endif()
if(POLICY CMP0169)
    cmake_policy(SET CMP0169 OLD)  # allow FetchContent_Populate
endif()
# Boost removed in 0.5.1 — was only used for flat_map

project(grilly_core LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(FetchContent)

# ── Find Vulkan SDK (optional for pure-Python PyPI builds) ───────────────
find_package(Vulkan QUIET)
if(Vulkan_FOUND)
    message(STATUS "Vulkan found: ${Vulkan_LIBRARIES}")
    message(STATUS "Vulkan include: ${Vulkan_INCLUDE_DIRS}")
else()
    message(WARNING "Vulkan SDK not found — fetching headers only for compilation. "
                    "GPU acceleration requires Vulkan runtime at execution time.")
    # Fetch Vulkan-Headers so VMA and our code can compile without the full SDK
    FetchContent_Declare(VulkanHeaders
        GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
        GIT_TAG        v1.3.283
        GIT_SHALLOW    ON
    )
    FetchContent_MakeAvailable(VulkanHeaders)
    # Create a stub Vulkan::Vulkan target (headers only, no library)
    add_library(Vulkan::Vulkan INTERFACE IMPORTED)
    target_include_directories(Vulkan::Vulkan INTERFACE "${vulkanheaders_SOURCE_DIR}/include")
endif()

# ── Boost REMOVED ───────────────────────────────────────────────────────
# Was only used for boost::container::flat_map in vk_pipeline_cache.h.
# Replaced with std::unordered_map to eliminate the 100MB Boost download
# that was the #1 build time bottleneck.

# ── Eigen (header-only) ─────────────────────────────────────────────────
# Try system Eigen3 first, but verify the targets file actually exists.
# On Windows, partial installs can have Eigen3Config.cmake without
# Eigen3Targets.cmake, causing a confusing CMake error.
find_package(Eigen3 3.4 CONFIG QUIET)
if(Eigen3_FOUND AND TARGET Eigen3::Eigen)
    message(STATUS "Eigen3 found: ${Eigen3_VERSION}")
else()
    if(Eigen3_FOUND)
        message(STATUS "Eigen3 config found but Eigen3::Eigen target missing — fetching instead")
    else()
        message(STATUS "Eigen3 not found — fetching via FetchContent")
    endif()
    # Reset to prevent stale config from interfering
    unset(Eigen3_FOUND)
    unset(Eigen3_DIR CACHE)
    FetchContent_Declare(Eigen3
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG        3.4.0
        GIT_SHALLOW    ON
    )
    set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_DEMOS OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_CMAKE_PACKAGE OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(Eigen3)
endif()

# ── VMA (header-only) ───────────────────────────────────────────────────
set(VMA_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/VulkanMemoryAllocator/include")
if(EXISTS "${VMA_INCLUDE_DIR}/vk_mem_alloc.h")
    message(STATUS "VMA found in third_party/")
else()
    message(STATUS "VMA not found in third_party/ — fetching via FetchContent")
    FetchContent_Declare(VulkanMemoryAllocator
        GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
        GIT_TAG        v3.2.1
        GIT_SHALLOW    ON
    )
    FetchContent_GetProperties(VulkanMemoryAllocator)
    if(NOT vulkanmemoryallocator_POPULATED)
        FetchContent_Populate(VulkanMemoryAllocator)
    endif()
    set(VMA_INCLUDE_DIR "${vulkanmemoryallocator_SOURCE_DIR}/include")
endif()

# ── pybind11 ─────────────────────────────────────────────────────────────
# Silence "Using compatibility mode for Python, set PYBIND11_FINDPYTHON to NEW/OLD"
if(NOT DEFINED PYBIND11_FINDPYTHON)
    set(PYBIND11_FINDPYTHON NEW)
endif()
set(PYBIND11_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/pybind11")
if(EXISTS "${PYBIND11_DIR}/CMakeLists.txt")
    add_subdirectory("${PYBIND11_DIR}" pybind11)
    message(STATUS "pybind11 found in third_party/")
else()
    find_package(pybind11 QUIET)
    if(NOT pybind11_FOUND)
        message(STATUS "pybind11 not found — fetching via FetchContent")
        FetchContent_Declare(pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG        v2.13.6
            GIT_SHALLOW    ON
        )
        FetchContent_MakeAvailable(pybind11)
    endif()
endif()

# ── nlohmann/json (header-only, used by training/jsonl_reader) ───────────
set(NLOHMANN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/nlohmann")
if(EXISTS "${NLOHMANN_DIR}/json.hpp")
    message(STATUS "nlohmann/json found in third_party/")
else()
    find_package(nlohmann_json QUIET)
    if(NOT nlohmann_json_FOUND)
        message(STATUS "nlohmann/json not found — fetching via FetchContent")
        FetchContent_Declare(nlohmann_json
            GIT_REPOSITORY https://github.com/nlohmann/json.git
            GIT_TAG        v3.11.3
            GIT_SHALLOW    ON
        )
        set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(nlohmann_json)
        # Make headers available at third_party/nlohmann/ for #include consistency
        set(NLOHMANN_DIR "${nlohmann_json_SOURCE_DIR}/single_include/nlohmann")
    endif()
endif()

# ── BLAKE3 (C library for CubeMind VSA role generation) ──────────────────
set(BLAKE3_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/BLAKE3/c")
if(NOT EXISTS "${BLAKE3_DIR}/blake3.h")
    message(STATUS "BLAKE3 not found in third_party/ — fetching via FetchContent")
    FetchContent_Declare(BLAKE3
        GIT_REPOSITORY https://github.com/BLAKE3-team/BLAKE3.git
        GIT_TAG        1.8.2
        GIT_SHALLOW    ON
    )
    FetchContent_GetProperties(BLAKE3)
    if(NOT blake3_POPULATED)
        FetchContent_Populate(BLAKE3)
    endif()
    set(BLAKE3_DIR "${blake3_SOURCE_DIR}/c")
endif()
add_library(blake3 STATIC
    "${BLAKE3_DIR}/blake3.c"
    "${BLAKE3_DIR}/blake3_dispatch.c"
    "${BLAKE3_DIR}/blake3_portable.c"
)
target_include_directories(blake3 PUBLIC "${BLAKE3_DIR}")
if(MSVC)
    target_sources(blake3 PRIVATE
        "${BLAKE3_DIR}/blake3_sse2.c"
        "${BLAKE3_DIR}/blake3_sse41.c"
        "${BLAKE3_DIR}/blake3_avx2.c"
        "${BLAKE3_DIR}/blake3_avx512.c"
    )
    target_compile_options(blake3 PRIVATE /wd4244 /wd4100 /wd4127)
else()
    # GCC/Clang need per-file SIMD flags — without them, intrinsics fail
    set(_blake3_sse2 "${BLAKE3_DIR}/blake3_sse2.c")
    set(_blake3_sse41 "${BLAKE3_DIR}/blake3_sse41.c")
    set(_blake3_avx2 "${BLAKE3_DIR}/blake3_avx2.c")
    set(_blake3_avx512 "${BLAKE3_DIR}/blake3_avx512.c")
    target_sources(blake3 PRIVATE ${_blake3_sse2} ${_blake3_sse41} ${_blake3_avx2} ${_blake3_avx512})
    set_source_files_properties(${_blake3_sse2} PROPERTIES COMPILE_FLAGS "-msse2")
    set_source_files_properties(${_blake3_sse41} PROPERTIES COMPILE_FLAGS "-msse4.1")
    set_source_files_properties(${_blake3_avx2} PROPERTIES COMPILE_FLAGS "-mavx2")
    set_source_files_properties(${_blake3_avx512} PROPERTIES COMPILE_FLAGS "-mavx512f -mavx512vl")
endif()
message(STATUS "BLAKE3 configured at ${BLAKE3_DIR}")

# ══════════════════════════════════════════════════════════════════════════
# Core static library
# ══════════════════════════════════════════════════════════════════════════
#
# Architecture: extensible backend via ComputeBackend interface.
# Vulkan backend ships now. OpenGL and OpenCL backends can be added by
# implementing ComputeBackend and adding sources here.
#
# SPIR-V design note (ref: Francisco Letterio / DevSH):
# Pre-compiled fused SPIR-V shaders with rule-based OpGraph fusion.
# Don't try to hack around SPIR-V pointer/aliasing limitations.

add_library(grilly_core_lib STATIC
    # ── Vulkan backend ──
    cpp/src/device.cpp
    cpp/src/buffer_pool.cpp
    cpp/src/pipeline_cache.cpp
    cpp/src/command_batch.cpp
    cpp/src/vulkan/vk_backend.cpp
    # ── Backend-agnostic ──
    cpp/src/op_graph.cpp
    cpp/src/autograd.cpp
    # ── Ops ──
    cpp/src/ops/linear.cpp
    cpp/src/ops/activations.cpp
    cpp/src/ops/fused.cpp
    cpp/src/ops/batched_ops.cpp
    cpp/src/ops/layernorm.cpp
    cpp/src/ops/rmsnorm.cpp
    cpp/src/ops/attention.cpp
    cpp/src/ops/conv.cpp
    cpp/src/ops/kv_cache.cpp
    cpp/src/ops/swizzle.cpp
    cpp/src/ops/snn.cpp
    cpp/src/ops/attention_ops.cpp
    cpp/src/ops/pooling.cpp
    cpp/src/ops/batchnorm.cpp
    cpp/src/ops/loss.cpp
    cpp/src/ops/optimizer.cpp
    cpp/src/ops/embedding.cpp
    cpp/src/ops/learning.cpp
    cpp/src/ops/perceiver.cpp
    cpp/src/ops/perceiver_encoder.cpp
    cpp/src/ops/moqe_train.cpp
    cpp/src/ops/moe_forward.cpp
    cpp/src/ops/vsa_lm_forward.cpp
    cpp/src/ops/prefix_scan.cpp
    cpp/src/shader_fusion.cpp
    # ── Experimental ──
    cpp/src/experimental/paged_latent_pool.cpp
    cpp/src/experimental/fused_attention.cpp
    # ── CubeMind ──
    cpp/src/cubemind/vsa.cpp
    cpp/src/cubemind/block_ops.cpp
    cpp/src/cubemind/hmm_ops.cpp
    cpp/src/cubemind/tensor_ops.cpp
    cpp/src/cubemind/cube.cpp
    cpp/src/cubemind/cache.cpp
    cpp/src/cubemind/text_encoder.cpp
    cpp/src/cubemind/semantic_assigner.cpp
    cpp/src/cubemind/resonator.cpp
    # ── Training ──
    cpp/src/training/pipeline.cpp
    # ── Cognitive ──
    cpp/src/cognitive/world_model.cpp
    # ── Temporal ──
    cpp/src/temporal/vulkan_temporal.cpp
    # ── NN framework ──
    cpp/src/nn/tensor.cpp
    cpp/src/nn/parameter.cpp
    cpp/src/nn/module.cpp
    cpp/src/nn/surrogate.cpp
    cpp/src/nn/snn.cpp
    cpp/src/nn/containers.cpp
    cpp/src/nn/optimizer.cpp
    cpp/src/nn/dataloader.cpp
    cpp/src/io/grl_checkpoint.cpp
)

target_include_directories(grilly_core_lib PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/cpp/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party"
    "${VMA_INCLUDE_DIR}"
)

# Find shaderc for JIT shader fusion (ships with Vulkan SDK only).
#
# IMPORTANT: We intentionally restrict the search to $VULKAN_SDK and refuse
# to fall through to system paths. On Linux, distro-packaged
# /usr/lib64/libshaderc_combined.a is typically built by Google's upstream
# against libc++ (Clang). Linking it into a libstdc++ (GCC) build produces a
# .so that references std::__1::* symbols with no matching definition —
# manifesting at import time as:
#   ImportError: undefined symbol: _ZTT...basic_istringstream...
# The LunarG SDK ships a GCC/libstdc++-compatible build, so we require it
# explicitly. Shader JIT fusion is optional; if VULKAN_SDK is unset we just
# disable GRILLY_HAS_SHADERC rather than risk an ABI mismatch.
if(DEFINED ENV{VULKAN_SDK})
    find_library(SHADERC_LIB shaderc_combined
        HINTS "$ENV{VULKAN_SDK}/Lib" "$ENV{VULKAN_SDK}/lib"
        NO_DEFAULT_PATH)
    if(SHADERC_LIB)
        message(STATUS "Found shaderc: ${SHADERC_LIB}")
    else()
        message(WARNING "shaderc not found in VULKAN_SDK — JIT shader fusion disabled")
    endif()
else()
    message(STATUS "VULKAN_SDK not set — JIT shader fusion disabled "
                   "(install LunarG Vulkan SDK and export VULKAN_SDK to enable)")
endif()

target_link_libraries(grilly_core_lib PUBLIC
    Vulkan::Vulkan
    Eigen3::Eigen
    blake3
    pybind11::pybind11    # NN framework classes use py::array_t, py::dict
)
if(SHADERC_LIB)
    target_link_libraries(grilly_core_lib PUBLIC ${SHADERC_LIB})
    target_include_directories(grilly_core_lib PUBLIC "$ENV{VULKAN_SDK}/Include")
    target_compile_definitions(grilly_core_lib PUBLIC GRILLY_HAS_SHADERC=1)
endif()

# VMA implementation compiled in device.cpp
# Platform-specific flags
if(MSVC)
    target_compile_options(grilly_core_lib PRIVATE /W4 /permissive-)
    target_compile_options(grilly_core_lib PRIVATE /wd4127 /wd4244 /wd4702 /wd4100)
else()
    target_compile_options(grilly_core_lib PRIVATE -Wall -Wextra -Wpedantic)
endif()

# ── Python module (split bindings for parallel compilation) ──────────────
# NOTE: bindings.cpp is the legacy monolithic file — kept for reference but
# not compiled. bindings_core.cpp is the new entry point.
pybind11_add_module(grilly_core
    cpp/python/bindings_core.cpp
    cpp/python/bindings_linear.cpp
    cpp/python/bindings_activations.cpp
    cpp/python/bindings_conv.cpp
    cpp/python/bindings_attention.cpp
    cpp/python/bindings_normalization.cpp
    cpp/python/bindings_optim.cpp
    cpp/python/bindings_loss.cpp
    cpp/python/bindings_snn.cpp
    cpp/python/bindings_pooling.cpp
    cpp/python/bindings_misc.cpp
    cpp/python/bindings_siglip.cpp
    cpp/python/bindings_perceiver.cpp
    cpp/python/bindings_moqe_train.cpp
    cpp/python/bindings_moe.cpp
    cpp/python/bindings_fusion.cpp
    cpp/python/bindings_vsa_lm.cpp
    cpp/python/bindings_grl.cpp
    cpp/python/bindings_prefix_scan.cpp
    cpp/python/bindings_distillation.cpp
)
target_link_libraries(grilly_core PRIVATE grilly_core_lib)

# Install the extension module at site-packages root so `import grilly_core` works.
# The bridge (backend/_bridge.py) and tensor_conversion.py use top-level imports.
install(TARGETS grilly_core DESTINATION .)
