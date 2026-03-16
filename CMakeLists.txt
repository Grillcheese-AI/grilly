cmake_minimum_required(VERSION 3.20)

# CMake 4.x policy: honor <PACKAGE>_ROOT variables
if(POLICY CMP0144)
    cmake_policy(SET CMP0144 NEW)
endif()
# CMake 4.x policy: FindBoost — use legacy FindBoost module if available,
# so find_package(Boost) works with older non-CMake Boost installs
if(POLICY CMP0167)
    cmake_policy(SET CMP0167 OLD)
endif()

project(grilly_core LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(FetchContent)

# ── Find Vulkan SDK ──────────────────────────────────────────────────────
# Vulkan SDK must be installed — no FetchContent fallback (needs drivers).
find_package(Vulkan REQUIRED)
message(STATUS "Vulkan found: ${Vulkan_LIBRARIES}")
message(STATUS "Vulkan include: ${Vulkan_INCLUDE_DIRS}")

# ── Boost (header-only + atomic) ────────────────────────────────────────
# Uses: container (flat_map), lockfree, pool, atomic
find_package(Boost 1.74 QUIET COMPONENTS headers atomic container)
if(Boost_FOUND)
    message(STATUS "Boost found: ${Boost_VERSION} at ${Boost_INCLUDE_DIRS}")
    if(NOT TARGET Boost::atomic)
        message(STATUS "Boost::atomic not found as library — aliasing to headers")
        add_library(Boost::atomic ALIAS Boost::headers)
    endif()
    # container, lockfree, pool are header-only — create targets if missing
    foreach(_comp container lockfree pool)
        if(NOT TARGET Boost::${_comp})
            add_library(Boost::${_comp} ALIAS Boost::headers)
        endif()
    endforeach()
else()
    message(STATUS "Boost not found — fetching via FetchContent (header-only subset)")
    FetchContent_Declare(Boost
        GIT_REPOSITORY https://github.com/boostorg/boost.git
        GIT_TAG        boost-1.86.0
        GIT_SHALLOW    ON
        EXCLUDE_FROM_ALL
    )
    set(BOOST_INCLUDE_LIBRARIES headers atomic container lockfree pool)
    FetchContent_MakeAvailable(Boost)
endif()

# ── Eigen (header-only) ─────────────────────────────────────────────────
find_package(Eigen3 3.4 CONFIG QUIET)
if(Eigen3_FOUND)
    message(STATUS "Eigen3 found: ${Eigen3_VERSION}")
else()
    message(STATUS "Eigen3 not found — fetching via FetchContent")
    FetchContent_Declare(Eigen3
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG        3.4.0
        GIT_SHALLOW    ON
    )
    set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
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
    target_sources(blake3 PRIVATE
        "${BLAKE3_DIR}/blake3_sse2.c"
        "${BLAKE3_DIR}/blake3_sse41.c"
        "${BLAKE3_DIR}/blake3_avx2.c"
        "${BLAKE3_DIR}/blake3_avx512.c"
    )
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
)

target_include_directories(grilly_core_lib PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/cpp/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party"
    "${VMA_INCLUDE_DIR}"
)

target_link_libraries(grilly_core_lib PUBLIC
    Vulkan::Vulkan
    Boost::headers
    Boost::atomic
    Boost::container
    Boost::lockfree
    Boost::pool
    Eigen3::Eigen
    blake3
    pybind11::pybind11    # NN framework classes use py::array_t, py::dict
)

# VMA implementation compiled in device.cpp
# Platform-specific flags
if(MSVC)
    target_compile_options(grilly_core_lib PRIVATE /W4 /permissive-)
    target_compile_options(grilly_core_lib PRIVATE /wd4127 /wd4244 /wd4702 /wd4100)
else()
    target_compile_options(grilly_core_lib PRIVATE -Wall -Wextra -Wpedantic)
endif()

# ── Python module ────────────────────────────────────────────────────────
pybind11_add_module(grilly_core cpp/python/bindings.cpp)
target_link_libraries(grilly_core PRIVATE grilly_core_lib)

# Install the extension module into the grilly package
install(TARGETS grilly_core DESTINATION grilly)
